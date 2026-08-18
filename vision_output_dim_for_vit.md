## 1. Why does `vision_output_dim_for_vit` exist?

Some vision encoder pipelines apply a final linear norm or projection immediately after the ViT transformer stack before handing features over to the projector or resampler.

`vision_output_dim_for_vit` defines the channel dimension emitted at the final exit boundary of the Vision Transformer encoder.

```text
ViT Final Layer ──► [ Output Head / LayerNorm ] ──► Dimension: vision_output_dim_for_vit (4096)
                                                            │
                                                            ▼
                                                    Multimodal Projector
```

---

## 2. Options and Defaults

| Value | Description |
|---|---|
| `4096` (Default) | Llama 4 Vision default output dimension |
| Matching `hidden_size_for_vit` | Standard when no intermediate bottleneck is used |

---

### One-line intuition
> **`vision_output_dim_for_vit` defines the final output channel dimensionality of the Vision Transformer before cross-modal projection.**
