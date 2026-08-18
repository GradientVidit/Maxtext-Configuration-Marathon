## 1. Why does `audio_placeholder` exist?

In speech-enabled omni-modal models, audio input can occur before, after, or interleaved with text and images. The model needs an explicit boundary sentinel in the text prompt to locate the audio sequence.

`audio_placeholder` defines the text substring that marks an audio insertion point.

```text
Prompt: "Transcribe the following speech: <|audio|>"
                                             │
                                             ▼
                        Replaced with Projected Acoustic Tokens
```

---

## 2. Options and Defaults

| Value | Convention |
|---|---|
| `"<|audio|>"` (Default) | Standard MaxText / Qwen-Omni placeholder string |
| `"<audio>"` | Standard speech-LLM convention |

---

## 3. Interactions

- **`use_audio`**: Must be `true`.
- **`audio_path`**: Supplies the audio waveform mapped to this token.

---

### One-line intuition
> **`audio_placeholder` sets the text marker string (default `"<|audio|>"`) replaced by acoustic encoder tokens in the prompt stream.**
