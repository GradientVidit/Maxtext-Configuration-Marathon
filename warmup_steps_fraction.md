## 1. Why does `warmup_steps_fraction` exist?

At step 0, model weights are randomly initialized, and optimizer moment buffers (e.g. AdamW's first and second moments) are unpopulated zeros.

Applying full peak learning rate immediately causes massive, noisy parameter updates that destabilize attention heads and LayerNorm scales, often leading to immediate loss divergence:

```text
Without Warmup:
Step 0 (Random weights + Noisy Grads + Peak LR) ──> Loss EXPLOSION / NaNs

With Warmup (warmup_steps_fraction = 0.1):
Step 0..N_warmup: LR ramps 0.0 -> Peak LR
Optimizer states populate smoothly -> Gradients stabilize -> Clean Convergence
```

`warmup_steps_fraction` controls the proportion of the training schedule spent linearly ramping the learning rate from $0.0$ up to peak `learning_rate`.

---

## 2. Fundamentals & Mechanics

The number of warmup steps is computed as:

$$N_{\text{warmup}} = \text{learning\_rate\_schedule\_steps} \times \text{warmup\_steps\_fraction}$$

- During steps $0 \dots N_{\text{warmup}}$, the effective learning rate is:
  $$\eta_t = \text{learning\_rate} \times \left(\frac{t}{N_{\text{warmup}}}\right)$$
- Applies identically to both `'cosine'` and `'wsd'` schedules.

---

## 3. Options & Defaults

| Option | Value | Meaning |
|---|---|---|
| Default | `0.1` | 10% of the schedule is spent in linear warmup. |
| Large Pretraining | `0.01` to `0.03` | 1%–3% warmup is typical for 100k+ step runs (e.g. 2,000–3,000 steps). |
| Fine-Tuning | `0.05` to `0.10` | 5%–10% warmup to gently adapt pretrained representations. |

---

## 4. Interactions & Dependencies

```text
learning_rate_schedule_steps
             │
             ▼
   warmup_steps_fraction ───> Multiplied to get absolute Warmup Step Count
             │
             ▼
   learning_rate (Peak reached at end of warmup)
```

---

## 5. Practical Scenarios & Failure Modes

- **Warmup Proportion in Long Runs:** In a 500,000-step pretraining run, default `0.1` means 50,000 steps of warmup—which is unnecessarily long. For huge runs, explicitly reduce `warmup_steps_fraction` to `0.01` or `0.02` (5k–10k steps).
- **Zero Warmup (`0.0`):** Setting `0.0` risks immediate gradient explosions on large Transformer architectures.

---

### One-line intuition

> **`warmup_steps_fraction` sets the percentage of the schedule dedicated to linearly ramping the learning rate from zero to peak, stabilizing early optimizer statistics.**
