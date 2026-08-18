## 1. Why does `distill_layer_indices` exist?

When distilling a deep teacher model into a smaller student, calculating intermediate feature losses across every layer is computationally wasteful and often architecturally incompatible (e.g. distilling a 64-layer teacher into a 16-layer student).

Instead, practitioners selectively align intermediate representations at key structural checkpoints:

```text
Teacher Layers: [0, 1, 2, 3, 4, 5, 6, 7, ..., 31]
                          │           │
                          ▼           ▼
Distillation Points:   Layer 7     Layer 15    Layer 23    Layer 31
                          ▲           ▲           ▲           ▲
                          │           │           │           │
Student Layers:        Layer 1     Layer 3     Layer 5     Layer 7
```

`distill_layer_indices` defines the specific list of layer indices where intermediate feature activations are extracted and aligned.

Setting `distill_layer_indices: None` (or empty) means intermediate feature distillation is skipped.

---

## 2. Mechanics & Layer Matching

When `distill_layer_indices` is specified as a list of integers (e.g. `[3, 7, 11]`):
- MaxText hooks into the `out_proj` activation outputs of those specified layer indices for both the teacher and student models.
- The intermediate activations are extracted during the forward pass and fed to the feature loss function configured by `distill_feature_loss_type`.

---

## 3. Options & Default

| Parameter | Type | Default | Options |
| :--- | :--- | :--- | :--- |
| `distill_layer_indices` | `list[int]` / `None` | `None` | List of 0-indexed integer layer positions (e.g. `[3, 7, 11]`, `[1, 2, 3]`), or `None` |

---

## 4. Interactions with Related Parameters

| Related Parameter | Interaction |
| :--- | :--- |
| `distill_beta` | Feature distillation loss is only active when `distill_beta > 0.0` and `distill_layer_indices` is not `None`. |
| `distill_feature_loss_type` | Specifies the loss function (`"cosine"` or `"l2"`) applied at the indexed layers. |
| `base_num_decoder_layers` | All indices in `distill_layer_indices` must be $< \text{base\_num\_decoder\_layers}$. |

---

## 5. Practical Scenarios

| Scenario | Value | Purpose |
| :--- | :--- | :--- |
| **Logit-Only Distillation** | `None` (Default) | Disables intermediate feature matching entirely. |
| **Uniform Depth Probing** | `[3, 7, 11, 15]` | Aligns intermediate representations at quartile depth intervals across a 16-layer model. |
| **Final-Stage Alignment** | `[15]` | Aligns only the penultimate representation layer before the final classifier head. |

---

### One-line intuition

> `distill_layer_indices` specifies the list of transformer layer positions where intermediate feature activations are extracted and aligned between student and teacher models.
