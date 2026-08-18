## 1. Why does `adam_weight_decay` exist?

In standard L2 regularization, weight decay is added directly to gradients before computing adaptive moments, causing weights with large historical gradients to decay disproportionately slowly.

AdamW decouples weight decay from gradient moment updates, applying shrinkage directly to weights on every step:

$$\theta_{t+1} = \theta_t (1 - \eta_t \cdot \lambda) - \eta_t \cdot \frac{m_t}{\sqrt{v_t} + \epsilon}$$

```text
Coupled vs Decoupled Decay:
Standard L2 (Coupled):    g_t = grad + λ * θ  ──> Distorts v_t and adaptive step sizes
AdamW (Decoupled):        θ = θ - η * (Update) - (η * λ) * θ  ──> True uniform shrinkage
```

`adam_weight_decay` ($\lambda$) defines the decoupled regularization coefficient applied to model weights.

---

## 2. Fundamentals & Mechanics

- Prevents weight norms from growing unbounded during pretraining.
- Default `0.1` matches Llama 2 / Llama 3 / Gemma pretraining configurations.

---

## 3. Options & Defaults

| Option | Value | Meaning |
|---|---|---|
| Default | `0.1` | Standard decoupled weight decay for modern Transformer pretraining. |
| Mild Regularization | `0.01` | Often used in fine-tuning to prevent aggressive parameter drift. |
| Disabled | `0.0` | Disables weight decay completely. |

---

## 4. Interactions & Dependencies

```text
adam_weight_decay ──> Applied to all parameters EXCEPT those in adamw_mask
```

- **`adamw_mask`:** Excludes 1D biases, LayerNorm/RMSNorm scale weights, and positional embeddings from decay.

---

## 5. Practical Scenarios & Failure Modes

- Setting `adam_weight_decay` too high (e.g. `1.0`) causes over-regularization and underfitting. Setting it to `0.0` during long pretraining allows weights to drift into high-norm regimes vulnerable to quantization degradation.

---

### One-line intuition

> **`adam_weight_decay` controls the decoupled weight decay regularization rate in AdamW, preventing parameter norms from drifting unbounded.**
