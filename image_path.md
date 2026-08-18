## 1. Why does `image_path` exist?

During multimodal inference, evaluation, and prompt generation, users need a straightforward way to provide local image files to the model without orchestrating an external distributed dataset pipeline.

`image_path` specifies one or more local file paths to images to be encoded and interleaved into the inference prompt.

```text
CLI / Config:
image_path: "/tmp/dog.jpg,/tmp/cat.png"
prompt: "Describe these two images: <|image|><|image|>"

Execution:
1. Decode /tmp/dog.jpg -> ViT -> Soft Tokens 1 -> Replace first <|image|>
2. Decode /tmp/cat.png -> ViT -> Soft Tokens 2 -> Replace second <|image|>
3. Run Autoregressive Decode
```

---

## 2. Syntax and Formatting

- Single image: `image_path: "/data/images/chart.png"`
- Multiple images: Comma-separated string without spaces, e.g. `image_path: "/path/img1.jpg,/path/img2.jpg"`

---

## 3. Options and Defaults

| Value | Behavior |
|---|---|
| `""` (Default) | No local image loaded (text-only decode or dataset-driven evaluation) |
| `"/path/to/img.jpg"` | Loads, preprocesses, and tokenizes the specified image for decoding |

---

## 4. Interactions

- **`image_placeholder`**: The placeholder tag in `prompt` where image embeddings are inserted.
- **`use_multimodal`**: Must be `true` for `image_path` to be processed.

---

### One-line intuition
> **`image_path` accepts comma-separated local image file paths for direct standalone multimodal inference and decoding.**
