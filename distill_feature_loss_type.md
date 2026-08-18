## 1. Why does `distill_feature_loss_type` exist?

When comparing intermediate feature representations between a student and teacher model ($h_s$ and $h_t$), different distance metrics enforce different geometric constraints:

```text
Cosine Loss (distill_feature_loss_type: "cosine"):
Focuses on directional angle θ between vectors; invariant to absolute norm/magnitude:
Loss = 1 - (h_s · h_t) / (||h_s|| ||h_t||)

L2 / Mean Squared Error Loss (distill_feature_loss_type: "l2"):
Forces exact point-to-point numerical match in Euclidean space:
Loss = ||h_s - h_t||_2^2
```

In neural networks with LayerNorm or RMSNorm, the **direction** of activation vectors often encodes the core semantic representation, while vector norms can vary naturally across layers and architectures. Cosine similarity allows the student to mimic the teacher's geometric orientation without being overly constrained to duplicate exact scalar activation scales.

`distill_feature_loss_type` selects the mathematical distance metric used for intermediate feature distillation.

---

## 2. Mechanics & Mathematical Comparison

| Loss Type | Formula | Gradient Characteristics |
| :--- | :--- | :--- |
| `"cosine"` (Default) | $\mathcal{L}_{\text{feat}} = 1 - \frac{h_s \cdot h_t}{\|h_s\|_2 \|h_t\|_2 + \epsilon}$ | Scale-invariant; gradient updates only rotate feature directions toward the teacher. |
| `"l2"` | $\mathcal{L}_{\text{feat}} = \frac{1}{D} \|h_s - h_t\|_2^2$ | Sensitive to absolute activation scale; penalizes both angular error and norm differences. |

---

## 3. Options & Default

| Parameter | Type | Default | Options |
| :--- | :--- | :--- | :--- |
| `distill_feature_loss_type` | `str` | `"cosine"` | `"cosine"`, `"l2"` |

---

## 4. Interactions with Related Parameters

| Related Parameter | Interaction |
| :--- | :--- |
| `distill_beta` | Must be $> 0.0$ for `distill_feature_loss_type` to take effect. |
| `distill_layer_indices` | The chosen metric is computed across all layer pairs specified in `distill_layer_indices`. |

---

## 5. Practical Guidance & Best Practices

| Choice | Recommended When |
| :--- | :--- |
| `"cosine"` (Default) | Student and teacher have different layer widths, norm scales, or when using RMSNorm architectures (e.g. LLaMA, DeepSeek, Gemma). |
| `"l2"` | Student and teacher share identical hidden dimensions and identical initialization scales. |

---

### One-line intuition

> `distill_feature_loss_type` selects between cosine similarity (angular direction alignment) and L2 distance (Euclidean point matching) for intermediate layer feature distillation.
