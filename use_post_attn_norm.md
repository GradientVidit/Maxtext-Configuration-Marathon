## 1. Why does `use_post_attn_norm` exist?

Standard modern autoregressive transformers utilize **Pre-LayerNorm (Pre-Norm)**: normalization is applied to the input of each sub-block, and the unnormalized sub-block output is added directly to the residual stream:

$$\text{Standard Pre-Norm:}\quad x_{l+1} = x_l + \text{Attention}(\text{RMSNorm}(x_l))$$

As models scale in depth ($L \ge 64$ layers), the residual stream accumulates variance linearly with depth: $\text{Var}(x_L) pprox L \cdot \text{Var}(x_0)$. This activation growth can cause late-layer attention outputs to become negligible relative to the massive residual stream norm, leading to training stagnation.

**Post-Attention Normalization (Dual-Norm)** introduces an additional normalization layer immediately after the attention projection *before* the residual addition:

$$\text{With Post-Attn Norm:}\quad x_{l+1} = x_l + \text{RMSNorm}_{\text{post}}(\text{Attention}(\text{RMSNorm}_{\text{pre}}(x_l)))$$

```text
Standard Pre-Norm:
  x ──┬──> RMSNorm_pre ──> Attention Block ──────────────> (+) ──> x_next
      └────────────────────────────────────────────────────▲

With use_post_attn_norm=True (Dual-Norm):
  x ──┬──> RMSNorm_pre ──> Attention Block ──> RMSNorm_post ──> (+) ──> x_next
      └────────────────────────────────────────────────────────▲
```

---

## 2. Options & Defaults

| Value | Behavior | Notes |
|---|---|---|
| `false` | Standard Pre-Norm only. | **Default**. Standard for LLaMA 1/2/3, Mistral, DeepSeek. |
| `true` | Adds a normalization layer (RMSNorm) to the attention output before residual addition. | Used in Gemma 4 small variants and dual-norm architectures. |

Default in `base.yml`: `false`

---

## 3. Benefits in Deep and Edge Architectures

1. **Residual Stream Variance Clamping:** Prevents attention output variance from exploding or diminishing relative to the residual trunk.
2. **Quantization Precision:** Post-normalized activations have strictly bounded dynamic range, significantly improving post-training 8-bit / 4-bit weight-activation quantization accuracy.

---

## 4. Interactions with other parameters

| Related Parameter | Interaction |
|---|---|
| [[use_post_ffw_norm]] | Sister parameter for the feed-forward / MLP sub-block. Often enabled together in dual-norm models. |
| [[normalization_layer_epsilon]] | Defines the $\epsilon$ numerical stability constant in the post-attention norm layer. |
| [[decoder_block]] | Specific decoder blocks (e.g. Gemma 4 small) automatically configure post-attention normalization. |

---

## 5. Practical Scenarios

- **Pretraining Standard Architectures (LLaMA / Mistral):** Leave `use_post_attn_norm: false`.
- **Reproducing Gemma 4 Edge Architectures:** Set `use_post_attn_norm: true` to match published specifications.
- **Deep Multi-Hundred Layer Networks:** Enable to stabilize gradient flow when depth $> 80$ layers.

---

### One-line intuition

> **`use_post_attn_norm=true` inserts an RMSNorm layer after the attention sub-block before residual addition, bounding activation variance growth in deep transformer architectures.**
