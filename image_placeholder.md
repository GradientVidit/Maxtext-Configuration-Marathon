## 1. Why does `image_placeholder` exist?

When a user writes a prompt containing both text and images (e.g., *"What is the difference between <|image|> and <|image|>? "*), the tokenizer must know precisely where visual representations belong in the sequence. 

Rather than hardcoding a special token ID, `image_placeholder` defines the exact text substring that marks an image insertion point.

```text
Prompt String:
"Here is a photo: <|image|>. Describe the color."
                      │
                      ▼
Tokenizer splits: ["Here is a photo: ", "<|image|>", ". Describe the color."]
                      │                      │                   │
                      ▼                      ▼                   ▼
Embeddings:     [Text Embeddings]      [ViT Soft Tokens]   [Text Embeddings]
```

`image_placeholder` is the string sentinel replaced by soft visual tokens in the token stream.

---

## 2. Options and Defaults

| Value | Model Convention |
|---|---|
| `"<|image|>"` (Default) | Standard MaxText / Qwen-VL / Llama convention |
| `"<image>"` | Classic LLaVA / Vicuna style |
| `"<|vision_start|><|vision_end|>"` | Explicit start/end bounding tokens |

---

## 3. Interactions

- **`use_multimodal`**: Must be enabled.
- **`image_path`**: The number of `<|image|>` placeholders in the prompt must match the number of loaded images.

---

### One-line intuition
> **`image_placeholder` defines the text marker string (default `"<|image|>"`) in the prompt that is substituted with projected visual embeddings.**
