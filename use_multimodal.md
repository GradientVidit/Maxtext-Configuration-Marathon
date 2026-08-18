## 1. Why does `use_multimodal` exist?

Modern Large Language Models are no longer purely textual; they must process vision (images and videos) alongside text tokens. To support vision without rewriting or breaking text-only pipelines, MaxText encapsulates the entire visual processing pipeline—including Vision Transformer (ViT) encoders, spatial patchifiers, projectors, and multimodal token interleaving—behind a single master switch.

```text
use_multimodal: false (Pure Text Pipeline)
Text Input ───► Tokenizer ───► Text Embeddings ───► LLM Decoder ───► Text Output

use_multimodal: true (Vision-Language Pipeline)
Image/Video ──► ViT Encoder ──► Projector ──► Soft Visual Tokens ──┐
                                                                    ├──► LLM Decoder ──► Output
Text Input  ──► Tokenizer   ───────────────► Text Embeddings ───────┘
```

`use_multimodal` is the master configuration toggle enabling multimodal vision processing throughout the model graph, data loaders, and checkpoint loaders.

---

## 2. What does it actually control?

When set to `true`:
1. **Model Graph**: Instantiates the Vision Transformer (ViT) encoder and multimodal projector layers in Flax/NNX.
2. **Data Pipeline**: Activates image/video decoding, resizing, normalization, and patchification in Grain or TFDS data pipelines.
3. **Token Interleaving**: Replaces visual placeholder tokens (`image_placeholder`, `video_placeholder`) in the tokenized text sequence with the projected visual embedding vectors.

---

## 3. Options and Defaults

| Value | Behavior | Context |
|---|---|---|
| `false` (Default) | Standard pure-text LLM execution | Pre-training and fine-tuning text-only models (Llama 3, Gemma 2) |
| `true` | Instantiates ViT, projectors, and multimodal token merge | Vision-Language models (Llama 4 Vision, Gemma 3/4, Qwen2-VL, Qwen3-Omni) |

---

## 4. Parameter Interactions

- **`freeze_vision_encoder_params`**: Dictates whether the instantiated vision encoder updates during training.
- **`dtype_mm`**: Determines the arithmetic precision of the vision encoder.
- **`image_path` / `video_path`**: Local paths used for multimodal inference decoding.
- **`image_placeholder`**: The token string replaced by vision embeddings in the prompt stream.

---

## 5. Failure Modes

- Setting `use_multimodal: true` without multimodal dataset columns or input image paths will cause the data loader to raise `KeyError` or produce empty visual tensors.
- Attempting to load text-only checkpoints with `use_multimodal: true` will fail unless parameter initialization handles the missing ViT/projector weights.

---

### One-line intuition
> **`use_multimodal` is the master boolean flag that instantiates the Vision Transformer encoder, projector, and multimodal data pipeline for image and video processing.**
