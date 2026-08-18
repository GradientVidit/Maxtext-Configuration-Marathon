## 1. Why does `distill_temperature_end` exist?

Logit distillation temperature $T$ controls the smoothness of the teacher's probability distribution.

In **Simulated Annealing Distillation**:
- **Early in training**: High temperature (e.g. $T = 3.0$ or $4.0$) flattens the distribution, exposing broad semantic rankings and soft inter-token relationships.
- **Late in training**: Cooling the temperature toward $T = 1.0$ sharpens probabilities, aligning the student's top-token certainty with realistic generation dynamics.

```text
Temperature Annealing:
Step 0:        distill_temperature (e.g. 3.0) ───┐
                                                 │  distill_temperature_schedule
                                                 ▼
Final Step:    distill_temperature_end (e.g. 1.0)
```

`distill_temperature_end` defines the target end temperature $T_{\text{end}}$ that `distill_temperature` anneals toward over the training duration.

Setting `distill_temperature_end: None` (the default) keeps temperature constant throughout training.

---

## 2. Mechanics & Scaling

When `distill_temperature_end` is specified:
- MaxText interpolates the temperature $T(t)$ at each training step between `distill_temperature` and `distill_temperature_end` according to `distill_temperature_schedule`.
- The loss scaling factor $T(t)^2$ is dynamically recalculated at each step to maintain balanced gradient magnitudes throughout the cooling process.

---

## 3. Options & Default

| Parameter | Type | Default | Options |
| :--- | :--- | :--- | :--- |
| `distill_temperature_end` | `float` / `None` | `None` | Positive float, or `None` (constant temperature) |

---

## 4. Interactions with Related Parameters

| Related Parameter | Interaction |
| :--- | :--- |
| `distill_temperature` | Starting temperature $T_0$ at step 0. |
| `distill_temperature_schedule` | Defines the curve (`"linear"`, `"cosine"`, or `"constant"`). |
| `distill_alpha` | Distillation loss computed with temperature $T(t)$ is weighted by `distill_alpha`. |

---

## 5. Practical Guidance

| Pattern | Configuration | Use Case |
| :--- | :--- | :--- |
| **Fixed Temperature** | `distill_temperature: 1.0`, `distill_temperature_end: None` | Standard baseline distillation. |
| **Cooling Schedule** | `distill_temperature: 3.0`, `distill_temperature_end: 1.0` | Soft early exploration cooling into sharp generation calibration. |

---

### One-line intuition

> `distill_temperature_end` sets the target end temperature for logit distillation, enabling temperature annealing from broad soft distributions to sharp predictive distributions.
