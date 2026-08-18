## 1. Why does `rope_theta_for_vit` exist?

When the Vision Transformer uses 2D Rotary Position Embeddings (`posemb_type_for_vit: "rope"`), the base frequency constant $\theta$ (theta) determines the wavelength spectrum across the attention head dimensions.

```text
2D RoPE Frequencies:
omega_k = 1.0 / (rope_theta_for_vit ^ (2k / D))
```

`rope_theta_for_vit` sets the base rotary frequency constant $\theta$ inside the ViT attention heads.

---

## 2. Options and Defaults

| Value | Behavior |
|---|---|
| `10000` (Default) | Standard base frequency for vision rotary embeddings |
| `1000000` | Scaled frequency for ultra-high-resolution vision grids |

---

## 3. Interactions

- **`posemb_type_for_vit`**: Must be `"rope"` for this parameter to take effect.

---

### One-line intuition
> **`rope_theta_for_vit` sets the base rotary frequency ($	heta$) for Vision Transformers utilizing 2D Rotary Position Embeddings.**
