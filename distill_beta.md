## 1. Why does `distill_beta` exist?

Logit-based distillation only matches the final output predictions of the teacher and student models. However, deep neural networks can reach similar output logits through divergent intermediate feature trajectories.

To guide the student's internal representations directly:

```text
Teacher Model:  Layer 4 Out-Proj ───> Layer 8 Out-Proj ───> ... ───> Teacher Logits
                       │                     │                              │
                       │ Feature Loss        │ Feature Loss                 │ Logit KD Loss (distill_alpha)
                       ▼                     ▼                              ▼
Student Model:  Layer 2 Out-Proj ───> Layer 4 Out-Proj ───> ... ───> Student Logits
```

**Feature Distillation** (or intermediate representation matching) adds an auxiliary loss penalizing differences between corresponding hidden activations (specifically the output projections `out_proj`) of the teacher and student.

`distill_beta` ($\beta$) sets the loss weight multiplier for this auxiliary intermediate feature distillation loss.

Setting `distill_beta: 0.0` disables intermediate feature distillation.

---

## 2. Mechanics & Loss Calculation

When `distill_beta > 0.0`:
1. MaxText captures intermediate activation tensors from layers specified by `distill_layer_indices` during the forward pass.
2. For each matched layer pair, it computes the feature discrepancy metric (determined by `distill_feature_loss_type`, e.g. cosine distance or L2).
3. The total feature loss $\mathcal{L}_{\text{feat}}$ is averaged across all monitored layers and weighted by $\beta$:
   $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{logit\_KD}} + \beta \cdot \mathcal{L}_{\text{feat}}$$

---

## 3. Options & Default

| Parameter | Type | Default | Options |
| :--- | :--- | :--- | :--- |
| `distill_beta` | `float` | `0.0` | Float $\ge 0.0$ (e.g., `0.0`, `0.1`, `0.5`) |

---

## 4. Interactions with Related Parameters

| Related Parameter | Interaction |
| :--- | :--- |
| `distill_feature_loss_type` | Determines whether the feature loss is computed via `"cosine"` similarity or `"l2"` distance. |
| `distill_layer_indices` | Defines which intermediate layer activations are compared. |
| `distill_beta_end` / `distill_beta_schedule` | Allows dynamic annealing or warmup of the feature loss weight over time. |

---

## 5. Practical Guidance

| Setting | Behavior | Use Case |
| :--- | :--- | :--- |
| `distill_beta: 0.0` (Default) | Intermediate feature matching disabled; only output logit distillation is active. | Standard knowledge distillation. |
| `distill_beta: 0.1` – `0.5` | Forces student intermediate layers to align geometrically with the teacher. | Compressing deep teachers into shallow students (e.g. 32 layers $\to$ 16 layers). |
| `distill_beta > 1.0` | Feature loss dominates gradient updates; may constrain student flexibility. | Use caution; monitor task cross-entropy closely. |

---

### One-line intuition

> `distill_beta` sets the loss weight for auxiliary intermediate feature distillation, encouraging student internal activations to geometrically match the teacher model.
