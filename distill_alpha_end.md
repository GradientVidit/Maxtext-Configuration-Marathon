## 1. Why does `distill_alpha_end` exist?

In knowledge distillation, keeping the balance weight $\alpha$ static throughout the entire training run is often sub-optimal:
- **Early in training**: The student benefits from heavy teacher guidance ($\alpha \approx 0.8$) to bootstrap representation learning quickly.
- **Late in training**: The student should transition toward hard ground-truth task labels ($\alpha \to 0.1$) to refine exact calibration and minimize test perplexity.

```text
Dynamic Alpha Scheduling:
Step 0:        distill_alpha (e.g. 0.8) ───┐
                                           │  distill_alpha_schedule (e.g. linear / cosine)
                                           ▼
Final Step:    distill_alpha_end (e.g. 0.1)
```

`distill_alpha_end` defines the target value that `distill_alpha` anneals toward by the end of training.

Setting `distill_alpha_end: None` (the default) keeps `distill_alpha` constant throughout training.

---

## 2. Mechanics & Progression

When `distill_alpha_end` is configured with a float value $v$:
- At step $0$, the alpha weight equals `distill_alpha`.
- Over `learning_rate_schedule_steps` (or total training steps), MaxText interpolates $\alpha(t)$ from `distill_alpha` to `distill_alpha_end` according to the curve defined by `distill_alpha_schedule`.
- At or beyond the final step, $\alpha(t)$ remains pinned at `distill_alpha_end`.

---

## 3. Options & Default

| Parameter | Type | Default | Options |
| :--- | :--- | :--- | :--- |
| `distill_alpha_end` | `float` / `None` | `None` | Float in `[0.0, 1.0]`, or `None` (constant alpha) |

---

## 4. Interactions with Related Parameters

| Related Parameter | Interaction |
| :--- | :--- |
| `distill_alpha` | Initial starting value of the schedule at step 0. |
| `distill_alpha_schedule` | Defines the interpolation trajectory (`"linear"`, `"cosine"`, or `"constant"`). |
| `steps` / `learning_rate_schedule_steps` | Determines the total step horizon over which the schedule transitions. |

---

## 5. Practical Guidance

| Pattern | Start / End Values | Purpose |
| :--- | :--- | :--- |
| **Fixed Alpha** | `distill_alpha: 0.5`, `distill_alpha_end: None` | Standard uniform distillation without scheduling. |
| **Teacher Annealing** | `distill_alpha: 0.9`, `distill_alpha_end: 0.1` | Heavy early teacher guidance decaying into hard task ground truth. |
| **Teacher Warmup** | `distill_alpha: 0.1`, `distill_alpha_end: 0.8` | Starts on ground truth data, gradually increasing teacher regularization. |

---

### One-line intuition

> `distill_alpha_end` sets the final target value for the distillation loss weight $\alpha$, enabling dynamic transition between teacher guidance and ground truth task training.
