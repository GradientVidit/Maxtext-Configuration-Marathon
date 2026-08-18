## 1. Why does `learning_rate` exist?

Optimization of deep neural networks requires controlling the step size along the loss gradient surface:

$$\theta_{t+1} = \theta_t - \eta_t \cdot \hat{g}_t$$

If $\eta$ (the learning rate) is too large, parameter updates overshoot stable valleys, causing loss explosions and NaN gradients. If $\eta$ is too small, training crawls and fails to escape suboptimal local minima within the compute budget.

```text
Loss Surface Trajectory:
   High LR (1e-1):      Optimal LR (3e-4):       Low LR (1e-6):
      \     /                 \     /                 \     /
       \ O /                   \   /                   \   /
      ──\─/── (Overshoots)      \ O / (Converges)       \ / (Barely moves)
         V                        V                      V
```

`learning_rate` specifies the peak (maximum) learning rate achieved after the warmup phase.

---

## 2. Fundamentals & Mechanics

`learning_rate` serves as the scaling anchor for the entire schedule:

```text
LR Schedule Lifecycle:
     learning_rate (Peak) ──────────────────────────┐
               /                                     \
              / (Linear Warmup)                       \ (Cosine or WSD Decay)
             /                                         \
 0.0 ───────┘                                           └───> learning_rate * final_fraction
```

- During warmup (step $0 \dots N_{\text{warmup}}$), the effective LR scales linearly from $0$ up to `learning_rate`.
- During the decay phase, the LR transitions from `learning_rate` down to `learning_rate * learning_rate_final_fraction`.

---

## 3. Options & Defaults

| Option | Value | Meaning |
|---|---|---|
| Default | `3.e-5` (`0.00003`) | Conservative baseline peak learning rate. |
| Typical 7B-8B Pretraining | `3.e-4` (`0.0003`) | Standard peak LR for Llama/Gemma-scale dense models with AdamW. |
| Fine-tuning / LoRA | `1.e-5` to `5.e-5` | Smaller learning rate to avoid catastrophic forgetting. |

---

## 4. Interactions & Dependencies

```text
                          learning_rate (Peak)
                                   │
      ┌────────────────────────────┼────────────────────────────┐
      ▼                            ▼                            ▼
warmup_steps_fraction    learning_rate_final_fraction   opt_type / adam_weight_decay
(Ramps up to peak)       (Decays to peak * fraction)    (Coupled update dynamics)
```

- **Batch Size Scaling:** Doubling the global batch size often warrants scaling `learning_rate` by $\sqrt{2}$ or $2$ depending on the optimizer regime.
- **`opt_type: "muon"`:** Muon typically uses a significantly different optimal learning rate scale compared to AdamW (often higher, e.g. `0.02` to `0.05`).

---

## 5. Practical Scenarios & Failure Modes

- **Loss Divergence (NaNs):** Sudden loss spikes in early steps often indicate peak `learning_rate` is too aggressive for the model depth or batch size.
- **Slow Convergence:** If training loss decreases linearly without curvature, `learning_rate` is likely set too low.

---

### One-line intuition

> **`learning_rate` sets the maximum peak step size for parameter updates, acting as the primary anchor for the warmup and decay schedules.**
