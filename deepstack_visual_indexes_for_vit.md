## 1. Why does `deepstack_visual_indexes_for_vit` exist?

In standard Vision-Language Models (VLMs), visual features are extracted **only from the final layer** of the Vision Transformer (ViT) and passed to the LLM input. However, earlier ViT layers capture fine-grained low-level visual geometry (sharp edges, small text, object bounding boxes), while deeper layers capture abstract semantic concepts. Relying solely on the final ViT layer discards crucial spatial and structural details.

**DeepStack** (introduced in advanced multimodal architectures like Qwen2.5-VL and Qwen3-VL) extracts multi-level visual features from **intermediate ViT layers** specified by index, projecting and fusing them into the language model to preserve fine-grained visual details.

```text
Standard VLM (Final Layer Only):
Input Patches ──► [ ViT Layer 0 ] ──► [ Layer 8 ] ──► ... ──► [ ViT Layer 33 (Final) ] ──► LLM
                                                                       ▲
                                            (Only final layer features used)

DeepStack Architecture (deepstack_visual_indexes_for_vit = [8, 16, 24]):
Input Patches ──► [ ViT Layer 0 ] ──► [ Layer 8 ]  ──────────────► Extract Level 1 ──┐
                                          │                                          │
                                      [ Layer 16 ] ──────────────► Extract Level 2 ──┼──► Multilevel Fusion -> LLM
                                          │                                          │
                                      [ Layer 24 ] ──────────────► Extract Level 3 ──┘
                                          │
                                      [ Layer 33 ] ──────────────► Final ViT Tokens ─┘
```

`deepstack_visual_indexes_for_vit` specifies the list of ViT layer indices from which intermediate multi-scale visual features are extracted.

---

## 2. Options and Defaults

| Value | Behavior | Model Architecture |
|---|---|---|
| `[]` (Default: empty list) | DeepStack disabled; only the final ViT layer output is passed | Standard single-layer ViT pipelines |
| `[8, 16, 24]` | Extracts intermediate features at ViT layers 8, 16, and 24 | Qwen2.5-VL / Qwen3-OmniMoE multi-scale vision |

---

## 3. Parameter Interactions

- **`num_hidden_layers_for_vit`**: ViT total depth. Specified indices must be strictly $< \text{num\_hidden\_layers\_for\_vit}$.
- **`out_hidden_size_for_vit`**: Channel dimension of the extracted and projected DeepStack visual features.
- **`spatial_merge_size_for_vit`**: Governs spatial downsampling on the extracted multi-level visual tokens.

---

### One-line intuition
> **`deepstack_visual_indexes_for_vit` specifies the list of intermediate ViT layer indices from which multi-level visual features are extracted to preserve fine-grained spatial details in deep multimodal models.**
