## 1. Why does `distill_beta_end` exist?

Intermediate feature distillation forces the student's hidden states to match the teacher's representations.

However, strict intermediate layer alignment can become overly restrictive in later training stages:
- **Early in training**: A high feature weight $\beta \approx 0.5$ acts as structural scaffolding, quickly pulling the student's internal representations into the teacher's semantic manifold.
- **Late in training**: Decaying $\beta \to 0.0$ allows the student to specialize and independently fine-tune its own intermediate parameters to maximize final output accuracy.

```text
Dynamic Beta Scheduling:
Step 0:        distill_beta (e.g. 0.5) ───┐
                                          │  distill_beta_schedule (linear / cosine)
                                          ▼
Final Step:    distill_beta_end (e.g. 0.0)
```

`distill_beta_end` sets the final target value for the intermediate feature distillation loss weight $\beta$.

Setting `distill_beta_end: None` (the default) keeps `distill_beta` constant throughout training.

---

## 2. Mechanics & Progression

- When configured with a target value (e.g. `0.0`), MaxText dynamically interpolates $\beta(t)$ from `distill_beta` to `distill_beta_end` using `distill_beta_schedule`.
- When $\beta(t)$ approaches $0.0$, the gradient contribution from intermediate layer feature matching smoothly vanishes.

---

## 3. Options & Default

| Parameter | Type | Default | Options |
| :--- | :--- | :--- | :--- |
| `distill_beta_end` | `float` / `None` | `None` | Float $\ge 0.0$, or `None` (constant beta) |

---

## 4. Interactions with Related Parameters

| Related Parameter | Interaction |
| :--- | :--- |
| `distill_beta` | Initial starting value $\beta_0$. |
| `distill_beta_schedule` | Functional schedule shape (`"linear"`, `"cosine"`, or `"constant"`). |
| `distill_layer_indices` | The intermediate layers whose loss is weighted by $\beta(t)$. |

---

## 5. Practical Guidance

| Pattern | Setup | Benefit |
| :--- | :--- | :--- |
| **Feature Scaffolding Decay** | `distill_beta: 0.5`, `distill_beta_end: 0.0` | Uses teacher intermediate guidance early, releasing the student to fit task data freely later. |
| **Fixed Feature Weight** | `distill_beta: 0.1`, `distill_beta_end: None` | Continuous feature alignment throughout entire run. |

---

### One-line intuition

> `distill_beta_end` specifies the target end value for the intermediate feature distillation loss weight, allowing feature matching scaffolding to decay over training.
