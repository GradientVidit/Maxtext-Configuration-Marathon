## 1. Why does `skip_step_on_spikes` exist?

In multi-billion parameter runs, transient numerical glitches (such as bad data batches or loss spikes) can corrupt optimizer momentum buffers and ruin weeks of training compute.

If MaxText can detect that a step's loss or gradient norm is an extreme statistical outlier before applying the weight update, it can safely discard that corrupted step:

```text
Spike Detection Pipeline:

Training Step N ──> Compute Loss & Grad Norm
                           │
             Is metric > Mean + K * Std ?
                           │
             ┌─────────────┴─────────────┐
             ▼                           ▼
           [No]                        [Yes]
       Apply Update                 SKIP STEP
   Update Rolling Stats        Drop Grad & State Update
                               Preserve Safe Model Weights
```

`skip_step_on_spikes` enables automatic skipping of training steps that exceed rolling statistical loss/gradient thresholds.

---

## 2. Fundamentals & Mechanics

When `skip_step_on_spikes: true`:
1. MaxText tracks a rolling history of metric values over a window of `skip_step_interval` steps.
2. It computes the rolling mean $\mu$ and standard deviation $\sigma$.
3. If the current step metric exceeds $\mu + \text{skip\_step\_scaling\_factor} \times \sigma$:
   - The parameter update is bypassed (weights and optimizer moments remain unchanged).
   - A warning log is dispatched.
   - Training advances to the next step.

---

## 3. Options & Defaults

| Option | Value | Meaning |
|---|---|---|
| Default | `false` | Step skipping is disabled; all computed steps apply updates. |
| Enabled | `true` | Detects and skips anomalous spike steps automatically. |

---

## 4. Interactions & Dependencies

```text
skip_step_on_spikes: true
         │
         ├──> skip_step_interval (Rolling history window size)
         └──> skip_step_scaling_factor (Standard deviation threshold)
```

---

## 5. Practical Scenarios & Failure Modes

- **Unattended Long-Term Runs:** Enabling `skip_step_on_spikes: true` acts as an insurance policy against web data anomalies during 100k+ step pretraining.
- **Early Training Warmup:** Spikes during the first few dozen steps can trigger false-positive skips if `skip_step_interval` is not yet well-populated.

---

### One-line intuition

> **`skip_step_on_spikes` automatically drops parameter updates on steps where loss or gradient norms exceed a statistical threshold, protecting runs from divergence.**
