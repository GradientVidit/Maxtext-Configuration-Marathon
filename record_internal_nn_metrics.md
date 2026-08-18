## 1. Why does `record_internal_nn_metrics` exist?

During large-scale language model pretraining, loss curves and gradient norms provide only a macro-level summary of model health. When numerical instability, attention entropy collapse, or expert load imbalance occurs, diagnosing the root cause requires **micro-level inspection of internal neural network tensors**:

```text
                     MaxText Training Step
                              │
               record_internal_nn_metrics > 0?
                              │
             ┌────────────────┴────────────────┐
             ▼                                 ▼
           Level 0                           Level 1+
        (Production)                        (Telemetry)
     • Loss & TFLOPs only            • Query/Key/Value Activation Norms
     • Zero telemetry overhead       • Pre/Post Attention RMSNorm Scales
     • Maximum training speed        • SwiGLU Intermediate Magnitudes
                                     • Router Gating Entropy
                                     • Softmax Max/Min Logit Spreads
```

`record_internal_nn_metrics` enables telemetry instrumentation inside decoder layers, logging activation statistics without modifying forward/backward mathematics.

---

## 2. Options & Defaults

| Value | Telemetry Level | Performance Impact | Notes |
|---|---|---|---|
| `0` | Disabled. Only top-level loss and throughput metrics are logged. | $0\%$ overhead. | **Default**. Production training setting. |
| `1` | Basic layer statistics (layer norms, activation scales, logit min/max). | Minimal ($<1\%$ step time increase). | Recommended during initial run bring-up. |
| `2+` | Exhaustive tensor statistics (per-head attention entropy, gradient distributions). | Moderate ($2\text{--}5\%$ overhead). | Diagnostic / deep debugging mode. |

Default in `base.yml`: `0`

---

## 3. Metrics Captured by Layer Instrumentation

When enabled, MaxText logs scalar summary distributions to TensorBoard / W&B / GCS metrics:
1. **RMSNorm Scales:** Tracks if weights are drifting to large magnitudes ($\|\gamma\|_2$).
2. **Attention Logits:** Logs max, min, and mean pre-softmax logit values (identifies softmax saturation).
3. **Residual Stream Variance:** Monitors $\text{Var}(x_l)$ progression across depth to detect vanishing or exploding residual activations.
4. **Router Logits (MoE):** Tracks gating logits to detect expert starvation or representation collapse.

---

## 4. Interactions with other parameters

| Related Parameter | Interaction |
|---|---|
| [[gcs_metrics]] | When `gcs_metrics: true`, recorded internal statistics are serialized to GCS bucket metric directories. |
| [[enable_wandb]] | Routes internal metric histograms and scalar curves directly to Weights & Biases dashboards. |
| [[metrics_file]] | Writes recorded scalars to local JSON/CSV diagnostic files. |

---

## 5. Practical Scenarios

- **First 1,000 Steps of a New Architecture Run:** Set `record_internal_nn_metrics: 1` to verify activation norms, layer norm stability, and attention logit bounds before launching full production training.
- **Diagnosing Mid-Run Loss Spikes:** Temporarily resume a checkpoint with `record_internal_nn_metrics: 2` around the step of the loss spike to isolate which specific layer or head became unstable.
- **Production Pretraining:** Keep `record_internal_nn_metrics: 0` to eliminate auxiliary telemetry compute and host-device transfer overhead.

---

### One-line intuition

> **`record_internal_nn_metrics` instruments transformer layers to record internal activation norms, attention entropy, and logit distributions, providing deep diagnostics for training stability.**
