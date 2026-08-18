## 1. Why does `video_placeholder` exist?

Just as `<|image|>` marks static image insertion, video prompts require a distinct marker to indicate where spatiotemporal video tokens should be spliced into the prompt token stream.

`video_placeholder` defines the text marker string that indicates a video insertion point.

```text
Prompt: "Summarize the action in this clip: <|video|>"
                                             │
                                             ▼
                        Replaced with Spatiotemporal Video Tokens
```

---

## 2. Options and Defaults

| Value | Description |
|---|---|
| `"<|video|>"` (Default) | Standard MaxText video placeholder token |
| `"<video>"` | Alternative tokenizer convention |

---

## 3. Interactions

- **`video_path`**: Supplies the video file mapped to the placeholder.
- **`use_multimodal`**: Master toggle.

---

### One-line intuition
> **`video_placeholder` specifies the text sentinel string (default `"<|video|>"`) replaced by video frame token embeddings in the prompt sequence.**
