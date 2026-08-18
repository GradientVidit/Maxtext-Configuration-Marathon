## 1. Why does `vision_output_length` exist?

In traditional VLMs, the number of visual tokens output by the vision encoder is directly tied to the image patch grid ($N = (H/P) 	imes (W/P)$). However, newer architectures (such as Gemma 4 or Perceiver/Q-Former style architectures) compress visual representations into a fixed number of learned "soft tokens" (e.g., exactly 256 or 512 tokens per image) regardless of raw image resolution.

`vision_output_length` specifies the exact number of soft visual tokens emitted by the vision encoder and projector into the language model.

```text
Raw Image ──► ViT Encoder (4096 raw patches)
                    │
                    ▼
       [ Visual Resampler / Gemma4 Tokenizer ]
                    │
                    ▼
     Compressed Soft Tokens (vision_output_length = 256) ──► LLM Decoder
```

---

## 2. Options and Defaults

| Value | Behavior | Model Architecture |
|---|---|---|
| `-1` (Default) | Unset / determined directly by ViT patch resolution and merge ratio | Standard Llama 4 Vision, Qwen2-VL, CLIP |
| Integer $> 0$ (e.g. `256`, `512`) | Compresses visual tokens to a fixed count | Gemma 4, Resampler architectures |

---

## 3. Interactions

- **`image_size_for_vit` & `patch_size_for_vit`**: Determines raw patch count before compression.
- **`max_target_length`**: Visual token footprint in the LLM sequence equals `vision_output_length`.

---

### One-line intuition
> **`vision_output_length` sets the fixed number of compressed soft visual tokens emitted per image (used in Gemma 4 architectures) regardless of raw patch count.**
