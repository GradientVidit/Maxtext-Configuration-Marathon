## 1. Why does `distill_beta_schedule` exist?

When annealing the intermediate feature distillation weight $\beta$ from `distill_beta` to `distill_beta_end`, the shape of the decay curve determines how smoothly intermediate layer constraints are relaxed:

```text
Linear Decay ("linear"):
β(t) decreases at a steady rate per step.

Cosine Decay ("cosine"):
β(t) decays slowly early, accelerates mid-training, and smoothly approaches β_end.

Constant ("constant"):
β(t) remains fixed at distill_beta.
```

`distill_beta_schedule` sets the interpolation curve used to transition the intermediate feature loss weight $\beta$ over training.

---

## 2. Mechanics & Functional Curves

Given progress ratio $p \in [0, 1]$, $\beta_0 = \text{distill\_beta}$, and $\beta_1 = \text{distill\_beta\_end}$:

| Schedule | Formula | Description |
| :--- | :--- | :--- |
| `"constant"` (Default) | $\beta(t) = \beta_0$ | Constant loss weight; ignores `distill_beta_end`. |
| `"linear"` | $\beta(t) = \beta_0 + p \cdot (\beta_1 - \beta_0)$ | Uniform step-wise interpolation. |
| `"cosine"` | $\beta(t) = \beta_1 + \frac{1}{2}(\beta_0 - \beta_1)\left(1 + \cos(\pi p)\right)$ | Smooth S-curve transition preventing sharp optimization shifts. |

---

## 3. Options & Default

| Parameter | Type | Default | Options |
| :--- | :--- | :--- | :--- |
| `distill_beta_schedule` | `str` | `"constant"` | `"constant"`, `"linear"`, `"cosine"` |

---

## 4. Interactions with Related Parameters

| Related Parameter | Interaction |
| :--- | :--- |
| `distill_beta_end` | Must be specified for non-constant schedules to take effect. |
| `distill_beta` | Initial starting value $\beta_0$. |

---

## 5. Practical Guidance

| Scenario | Recommendation |
| :--- | :--- |
| **Static Feature Regularization** | `"constant"` |
| **Gradual Feature Release** | `"cosine"` (decaying to `0.0` for smooth representation handover) |

---

### One-line intuition

> `distill_beta_schedule` sets the schedule curve shape (`"constant"`, `"linear"`, or `"cosine"`) for transitioning the intermediate feature distillation loss weight $\beta$.
