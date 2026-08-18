## 1. Why does `distill_temperature` exist?

Standard softmax probability distributions over large vocabularies (e.g. 128k tokens) are extremely sharp: the top token receives nearly $99\%$ of the probability mass, while millions of plausible alternative tokens receive near-zero probabilities:

```text
Hard / Unscaled Softmax (T = 1.0):
Token Probabilities: [0.992, 0.005, 0.002, 0.0007, 0.0001, ...]
(Almost no information conveyed about non-top token relationships)

Softened Softmax (T = 3.0):
Token Probabilities: [0.35, 0.22, 0.18, 0.12, 0.08, ...]
(Exposes rich semantic similarities and relative ranking: "dark knowledge")
```

Geoffrey Hinton et al. termed these subtle relative probabilities among incorrect/alternative classes **"dark knowledge."**

`distill_temperature` ($T$) scales the logits before softmax:

$$p_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$

A higher temperature softens the distribution, revealing the fine-grained semantic structure and uncertainty learned by the teacher model.

---

## 2. Mechanics & Gradient Scaling

- **Logit Scaling**: Both student logits $z_s$ and teacher logits $z_t$ are divided by $T$ before computing the KL-divergence loss.
- **Gradient Magnitude**: Because dividing logits by $T$ reduces gradient magnitude by a factor of $1/T^2$, MaxText automatically scales the distillation loss by $T^2$ to keep gradient magnitudes balanced with standard task losses.

---

## 3. Options & Default

| Parameter | Type | Default | Options |
| :--- | :--- | :--- | :--- |
| `distill_temperature` | `float` | `1.0` | Positive float (typically `1.0` to `4.0`) |

---

## 4. Interactions with Related Parameters

| Related Parameter | Interaction |
| :--- | :--- |
| `distill_alpha` | Distillation loss computed with temperature $T$ is weighted by `distill_alpha`. |
| `distill_temperature_end` / `distill_temperature_schedule` | Allows annealing temperature over training (e.g., starting soft at $T=3.0$ and cooling to $T=1.0$). |

---

## 5. Practical Guidance & Tuning

| Temperature | Distribution Sharpness | Best For |
| :--- | :--- | :--- |
| `distill_temperature: 1.0` (Default) | Standard natural logits | Fine-tuning and cases where teacher is already well-calibrated. |
| `distill_temperature: 2.0` – `3.0` | Softened distribution; exposes semantic dark knowledge | Large-scale pretraining distillation across wide vocabularies. |
| `distill_temperature > 5.0` | Very flat, nearly uniform distribution | Generally avoids useful gradient focus; risks underfitting. |

---

### One-line intuition

> `distill_temperature` softens the teacher and student logit distributions in knowledge distillation, exposing relative token probabilities ("dark knowledge") for richer student learning.
