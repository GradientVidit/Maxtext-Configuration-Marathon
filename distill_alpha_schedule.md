## 1. Why does `distill_alpha_schedule` exist?

When dynamically changing the distillation loss weight from `distill_alpha` to `distill_alpha_end`, the shape of the transition curve determines how abruptly the student shifts its learning objective:

```text
Linear Schedule ("linear"):
α(t) ───\ (Constant rate of change)
         \───> End

Cosine Schedule ("cosine"):
α(t) ───╮
         \  (Smooth start, accelerated mid-training drop, smooth plateau)
          ╰───> End

Constant Schedule ("constant"):
α(t) ────────────────────────── (Static throughout)
```

`distill_alpha_schedule` selects the functional form of the interpolation curve connecting `distill_alpha` and `distill_alpha_end`.

---

## 2. Mechanics & Mathematical Curves

Let progress ratio be $p = \min\left(1.0, \frac{\text{step}}{\text{total\_steps}}\right)$, $\alpha_0 = \text{distill\_alpha}$, and $\alpha_1 = \text{distill\_alpha\_end}$:

| Schedule | Formula | Description |
| :--- | :--- | :--- |
| `"constant"` (Default) | $\alpha(t) = \alpha_0$ | Ignores `distill_alpha_end`; stays fixed. |
| `"linear"` | $\alpha(t) = \alpha_0 + p \cdot (\alpha_1 - \alpha_0)$ | Steady, uniform rate of change across steps. |
| `"cosine"` | $\alpha(t) = \alpha_1 + \frac{1}{2}(\alpha_0 - \alpha_1)\left(1 + \cos(\pi p)\right)$ | Smooth, S-curve transition preventing sudden gradient shocks. |

---

## 3. Options & Default

| Parameter | Type | Default | Options |
| :--- | :--- | :--- | :--- |
| `distill_alpha_schedule` | `str` | `"constant"` | `"constant"`, `"linear"`, `"cosine"` |

---

## 4. Interactions with Related Parameters

| Related Parameter | Interaction |
| :--- | :--- |
| `distill_alpha_end` | Must be non-`None` for `"linear"` or `"cosine"` schedules to take effect. |
| `distill_alpha` | Initial starting value $\alpha_0$. |

---

## 5. Practical Guidance

| Scenario | Recommended Schedule |
| :--- | :--- |
| **Standard Static Run** | `"constant"` |
| **Gradual Annealing** | `"cosine"` (smoother convergence without optimization bumps) |
| **Direct Linear Decay** | `"linear"` |

---

### One-line intuition

> `distill_alpha_schedule` selects the mathematical trajectory (`"constant"`, `"linear"`, or `"cosine"`) used to interpolate the distillation loss weight $\alpha$ over training.
