## 1. Why does `hidden_size_for_vit` exist?

The Vision Transformer (ViT) processes image patch embeddings through a stack of self-attention and MLP layers. The width (channel embedding dimensionality) of these hidden states determines the representational capacity of the visual backbone. Because the vision encoder is an independent neural network from the main LLM decoder, its hidden dimension must be configured separately.

```text
Image Patch [14x14x3] ──► Patch Embedding Conv ──► Vector of dimension D (hidden_size_for_vit = 1408)
                                                          │
                                                          ▼
                                              [ ViT Transformer Block ]
                                              (Self-Attention on dim 1408)
```

`hidden_size_for_vit` defines the hidden feature dimension ($D_{\text{vit}}$) across all layers of the Vision Transformer.

---

## 2. Options and Defaults

| Value | Architecture / Scale | Parameter Scale |
|---|---|---|
| `1408` (Default) | Llama 4 Vision / EVA-CLIP-E / ViT-Giant | ~1 Billion params in ViT |
| `1024` | ViT-Large (CLIP-L) | ~300 Million params in ViT |
| `768` | ViT-Base | ~86 Million params in ViT |

MaxText config default: `1408` in `base.yml`.

---

## 3. Parameter Interactions

- **`num_attention_heads_for_vit`**: Head dimension is $D_{\text{head}} = \text{hidden\_size\_for\_vit} / \text{num\_attention\_heads\_for\_vit}$. Must divide evenly (e.g. $1408 / 16 = 88$).
- **`intermediate_size_for_vit`**: FFN hidden layer dimension (typically $4 \times \text{hidden\_size\_for\_vit}$).
- **`projector_input_dim_for_vit`**: Projector connects `hidden_size_for_vit` (or pixel-shuffled dimension) to the LLM.

---

### One-line intuition
> **`hidden_size_for_vit` sets the internal feature embedding width ($D_{\text{vit}}$) of the Vision Transformer encoder.**
