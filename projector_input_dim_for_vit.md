## 1. Why does `projector_input_dim_for_vit` exist?

Vision encoders and language model decoders are typically trained independently with completely different hidden dimensions (e.g. ViT output dimension $= 1408$ or $1152$, while LLM embedding dimension $= 4096$ or $8192$). 

Furthermore, if spatial downsampling (like 2x2 pixel shuffle) is applied to the vision features before projection, the channel dimension expands ($1408 	imes 4 = 5632$). 

```text
ViT Output [1408] ──► [ Pixel Shuffle 2x2 ] ──► Flattened Vector [projector_input_dim_for_vit = 4096]
                                                          │
                                                          ▼
                                                  [ Projector MLP ]
                                                          │
                                                          ▼
                                              [projector_output_dim_for_vit = 4096]
```

`projector_input_dim_for_vit` sets the exact input tensor channel dimension received by the multimodal projector.

---

## 2. Options and Defaults

| Value | Use Case |
|---|---|
| `4096` (Default) | Llama 4 Vision / standard large-scale projector input |
| Model-dependent | Equals `hidden_size_for_vit` or post-pixel-shuffle channel width |

---

## 3. Interactions

- **`projector_output_dim_for_vit`**: Output dimension of the projector.
- **`pixel_shuffle_ratio_for_vit`**: Governs channel expansion before the projector input.

---

### One-line intuition
> **`projector_input_dim_for_vit` sets the input feature dimension feeding into the multimodal projector MLP.**
