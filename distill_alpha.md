## 1. Why does `distill_alpha` exist?

In Knowledge Distillation, a smaller student model is trained simultaneously on two distinct learning objectives:
1. **Ground Truth Task Loss ($\mathcal{L}_{\text{task}}$)**: Standard cross-entropy loss against hard ground-truth target token labels.
2. **Teacher Distillation Loss ($\mathcal{L}_{\text{KD}}$)**: Kullback-Leibler (KL) divergence between the student's output probability distribution and the teacher model's softened output distribution.

```text
Student Logits ──┬──> Cross-Entropy vs Hard Ground Truth Labels ──> L_task
                 │                                                   │
                 └──> KL Divergence vs Softened Teacher Logits    ──> L_KD
                                                                     │
Combined Loss:  L = (1 - distill_alpha) * L_task + distill_alpha * L_KD
```

`distill_alpha` ($\alpha \in [0.0, 1.0]$) acts as the balancing weight between learning from ground-truth data and mimicking the teacher's soft probability distribution.

---

## 2. Mechanics & Loss Formulation

The total distillation loss computed in MaxText is:

$$\mathcal{L}_{\text{total}} = (1 - \alpha) \mathcal{L}_{\text{CE}}(y_{\text{student}}, y_{\text{true}}) + \alpha \cdot T^2 \cdot D_{\text{KL}}\left(\sigma\left(\frac{z_{\text{teacher}}}{T}\right) \;\Big\|\; \sigma\left(\frac{z_{\text{student}}}{T}\right)\right)$$

where:
- $\alpha = \text{distill\_alpha}$
- $T = \text{distill\_temperature}$
- The $T^2$ scaling factor ensures gradients from soft probabilities match the scale of hard-label cross-entropy gradients.

---

## 3. Options & Default

| Parameter | Type | Default | Options |
| :--- | :--- | :--- | :--- |
| `distill_alpha` | `float` | `0.5` | Range `[0.0, 1.0]` |

---

## 4. Interactions with Related Parameters

| Related Parameter | Interaction |
| :--- | :--- |
| `distill_temperature` | Controls distribution softness inside the KL-divergence term scaled by `distill_alpha`. |
| `distill_alpha_end` / `distill_alpha_schedule` | Allows `distill_alpha` to anneal dynamically over training steps (e.g. from 0.8 down to 0.1). |
| `distill_beta` | Adds an additional auxiliary intermediate feature distillation loss. |

---

## 5. Practical Scenarios & Trade-offs

| Value | Behavior | Ideal Use Case |
| :--- | :--- | :--- |
| `distill_alpha: 0.0` | 100% hard labels; distillation disabled | Standard autoregressive pretraining. |
| `distill_alpha: 0.5` (Default) | Equal balance between ground-truth labels and teacher guidance | General-purpose knowledge distillation. |
| `distill_alpha: 0.8` – `0.9` | Heavy reliance on teacher soft distribution | Distilling from very strong, high-capacity teacher models on noisy datasets. |
| `distill_alpha: 1.0` | Pure teacher mimicry (ignores ground truth labels entirely) | Synthetic sequence distillation. |

---

### One-line intuition

> `distill_alpha` balances the student loss between hard ground-truth cross-entropy and soft teacher logit KL-divergence in knowledge distillation.
