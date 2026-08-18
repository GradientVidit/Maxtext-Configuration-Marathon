## 1. Why does `distill_temperature_schedule` exist?

When annealing the logit distillation temperature $T$ from `distill_temperature` to `distill_temperature_end`, the rate and smoothness of temperature cooling affect optimization stability:

```text
Linear Cooling ("linear"):
T(t) drops uniformly at each step.

Cosine Cooling ("cosine"):
T(t) begins cooling slowly, accelerates through mid-training, and levels off smoothly.

Constant ("constant"):
T(t) remains pinned at distill_temperature throughout training.
```

`distill_temperature_schedule` sets the interpolation shape for temperature cooling over training steps.

---

## 2. Mechanics & Functional Curves

Given progress ratio $p \in [0, 1]$, $T_0 = \text{distill\_temperature}$, and $T_1 = \text{distill\_temperature\_end}$:

| Schedule | Formula | Description |
| :--- | :--- | :--- |
| `"constant"` (Default) | $T(t) = T_0$ | Constant temperature; `distill_temperature_end` is ignored. |
| `"linear"` | $T(t) = T_0 + p \cdot (T_1 - T_0)$ | Uniform step-by-step cooling. |
| `"cosine"` | $T(t) = T_1 + \frac{1}{2}(T_0 - T_1)\left(1 + \cos(\pi p)\right)$ | Smooth S-curve cooling without abrupt gradient scale shifts. |

---

## 3. Options & Default

| Parameter | Type | Default | Options |
| :--- | :--- | :--- | :--- |
| `distill_temperature_schedule` | `str` | `"constant"` | `"constant"`, `"linear"`, `"cosine"` |

---

## 4. Interactions with Related Parameters

| Related Parameter | Interaction |
| :--- | :--- |
| `distill_temperature_end` | Must be set (not `None`) for non-constant schedules to operate. |
| `distill_temperature` | Starting temperature $T_0$. |

---

## 5. Practical Guidance

| Scenario | Recommendation |
| :--- | :--- |
| **Standard Baseline** | `"constant"` |
| **Annealed Distillation** | `"cosine"` (recommended to prevent sharp gradient shocks during cooling) |

---

### One-line intuition

> `distill_temperature_schedule` defines the cooling curve (`"constant"`, `"linear"`, or `"cosine"`) used to anneal logit distillation temperature over training.
