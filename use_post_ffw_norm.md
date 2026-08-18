## 1. Why does `use_post_ffw_norm` exist?

In standard transformer blocks, the Feed-Forward Network (MLP / FFN) expands the hidden dimension (typically $4	imes$ to $8/3	imes d_{model}$) before projecting back down. In SwiGLU / GeGLU activations, the element-wise multiplication of two large projections can generate large activation magnitudes that get injected directly into the residual stream:

$$\text{Standard Pre-Norm FFN:}\quad x_{l+1} = x_l + \text{FFN}(\text{RMSNorm}(x_l))$$

**Post-FFW Normalization** places an additional normalization layer on the output of the feed-forward network prior to residual addition:

$$\text{With Post-FFW Norm:}\quad x_{l+1} = x_l + \text{RMSNorm}_{\text{post\_ffw}}(\text{FFN}(\text{RMSNorm}_{\text{pre\_ffw}}(x_l)))$$

```text
Standard Pre-Norm FFN:
  x ──┬──> RMSNorm_pre ──> FFN (Gate + Up + Down) ──────────> (+) ──> x_next
      └───────────────────────────────────────────────────────▲

With use_post_ffw_norm=True:
  x ──┬──> RMSNorm_pre ──> FFN (Gate + Up + Down) ──> RMSNorm ──> (+) ──> x_next
      └────────────────────────────────────────────────────────▲
```

---

## 2. Options & Defaults

| Value | Behavior | Notes |
|---|---|---|
| `false` | Standard Pre-Norm FFN without post-normalization. | **Default**. Standard for LLaMA 1/2/3, Mistral. |
| `true` | Inserts an RMSNorm layer after the FFN down-projection before residual addition. | Used in Gemma 4 and dual-norm architectures. |

Default in `base.yml`: `false`

---

## 3. Mechanics and Normalization Budget

Post-FFW normalization adds a single learnable scale vector $\gamma \in \mathbb{R}^{d_{model}}$ per layer ($d_{model}$ parameters). 

Because FFN blocks contribute the majority of raw parameter capacity and non-linear FLOPs in dense models, controlling the output scale of the FFN prevents intermediate activation drift from drowning out attention sub-block updates.

---

## 4. Interactions with other parameters

| Related Parameter | Interaction |
|---|---|
| [[use_post_attn_norm]] | Companion parameter for attention sub-blocks. Typically paired together. |
| [[mlp_activations]] | Non-linear activations like SwiGLU (`['silu', 'linear']`) benefit from post-FFW norm to clamp multiplicative output scales. |
| [[normalization_layer_epsilon]] | Sets the numerical stability constant $\epsilon$ for the post-FFW norm layer. |

---

## 5. Practical Scenarios

- **Standard LLM Pretraining:** Keep `use_post_ffw_norm: false`.
- **Gemma 4 Small / Compact Edge Models:** Set `use_post_ffw_norm: true` to reproduce published model weights.
- **Aggressive Quantization (INT4 / FP4):** Helps eliminate activation outliers in residual streams before quantization calibration.

---

### One-line intuition

> **`use_post_ffw_norm=true` places an RMSNorm layer on the FFN output before residual addition, preventing SwiGLU multiplicative scale explosion in deep models.**
