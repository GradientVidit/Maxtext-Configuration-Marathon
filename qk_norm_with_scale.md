## 1. Why does `qk_norm_with_scale` exist?

As transformer models scale in size and sequence length, the magnitude of Query ($Q$) and Key ($K$) vectors naturally increases. This causes the dot-product logits ($QK^T / \sqrt{d}$) to grow excessively large, causing softmax saturation and training instability.

**QK-Normalization** (Dehghani et al., 2023 / ViT-22B, Qwen 2, Gemma 2) normalizes $Q$ and $K$ vectors on the head dimension before computing the attention dot product:

$$\hat{Q} = rac{Q}{\|Q\|_2 + \epsilon}, \quad \hat{K} = rac{K}{\|K\|_2 + \epsilon}$$

However, pure unit-norm vectors constrain all attention logits to $[-1, 1]$, making it impossible for attention heads to achieve sharp, confident distributions when needed.

`qk_norm_with_scale` equips the normalization layers with learnable scale parameters $\gamma_q, \gamma_k$:

$$Q_{\text{norm}} = \gamma_q \odot \text{RMSNorm}(Q), \quad K_{\text{norm}} = \gamma_k \odot \text{RMSNorm}(K)$$

$$\text{Logits} = rac{(\gamma_q \odot \hat{Q})(\gamma_k \odot \hat{K})^T}{\sqrt{d}}$$

```text
Without Learnable Scale (False):
  Q, K ──> RMSNorm (Unit Variance) ──> Dot Product bounded in [-1, 1] (Too flat)

With Learnable Scale (True, Default):
  Q, K ──> RMSNorm ──> Scale by learnable γ_q, γ_k ──> Dot Product (Model learns sharp attention)
```

---

## 2. Options & Defaults

| Value | Behavior | Notes |
|---|---|---|
| `true` | Query and Key normalizations include learnable scale vectors $\gamma_q, \gamma_k$. | **Default**. Standard for Qwen 2 / 2.5, Gemma 2, Command-R+. |
| `false` | Normalization without learnable scale (pure unit sphere projections). | Restricts attention temperature to fixed geometric bounds. |

Default in `base.yml`: `true`

---

## 3. Why Learnable Scale is Critical for Attention Sharpness

In multi-head attention, different heads perform different tasks:
- **Broad retrieval heads:** Benefit from small $\gamma$, spreading attention across many context tokens.
- **Precise copying / induction heads:** Require large $\gamma$, concentrating softmax probability mass onto a single specific token.

`qk_norm_with_scale=true` lets each individual head learn its own effective inverse-temperature without risking runaway activation norms.

---

## 4. Interactions with other parameters

| Related Parameter | Interaction |
|---|---|
| [[v_norm_with_scale]] | Sister parameter for Value vector normalization. |
| [[attn_logits_soft_cap]] | Complementary stabilization: QK-norm normalizes vectors *before* dot product; soft-capping bounds scores *after* dot product. |
| [[use_qk_clip]] | Alternative approach: hard-clipping vector norms instead of continuous normalization. |

---

## 5. Practical Scenarios

- **Pretraining Qwen 2 / Gemma 2 Architectures:** Leave `qk_norm_with_scale: true` (default).
- **Preventing Training Instability at Scale:** For models $> 30\text{B}$ parameters, enabling QK-norm with scale eliminates attention logit explosion without needing manual learning rate decay.

---

### One-line intuition

> **`qk_norm_with_scale=true` adds learnable per-head scale vectors $\gamma_q, \gamma_k$ to normalized Query and Key projections, letting each head calibrate its own attention sharpness while guaranteeing numerical stability.**
