## 1. Why does `out_hidden_size_for_vit` exist?

In Qwen3-OmniMoE's vision encoder, the output projection maps internal ViT representations into a compact intermediate latent space before distributing them to the language model or DeepStack visual injection layers.

`out_hidden_size_for_vit` sets the output feature channel dimension of the Qwen3-OmniMoE vision encoder.

```text
ViT Encoder Stack ──► [ Spatial Merge & Output Projection ] ──► Dim: out_hidden_size_for_vit (512)
                                                                      │
                                                                      ▼
                                                          LLM / DeepStack Layers
```

---

## 2. Options and Defaults

| Value | Architecture Context |
|---|---|
| `512` (Default) | Qwen3-OmniMoE standard visual feature output dimension |
| Matching model preset | Configured automatically via model family configurations |

---

## 3. Interactions

- **`deepstack_visual_indexes_for_vit`**: Layers receiving DeepStack visual features expect inputs with channel depth `out_hidden_size_for_vit`.

---

### One-line intuition
> **`out_hidden_size_for_vit` specifies the output feature dimension of the Qwen3-OmniMoE vision encoder (default 512).**
