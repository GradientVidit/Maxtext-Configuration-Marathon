## 1. Why does `encoder_layers_for_audio` exist?

The depth (number of stacked Transformer layers) of the audio encoder determines its capacity to transform low-level spectral energy into high-level phonetic and semantic acoustic representations.

```text
Mel Input ──► Conv Frontend ──► [ Layer 0 ] ──► [ Layer 1 ] ... ──► [ Layer (encoder_layers - 1) ] ──► Audio Projector
```

`encoder_layers_for_audio` sets the total number of transformer layers in the audio encoder stack.

---

## 2. Options and Defaults

| Value | Encoder Depth | Context |
|---|---|---|
| `2` (Default) | Lightweight 2-layer encoder | Qwen3-OmniMoE fast streaming acoustic front-end |
| `6` / `12` | Deep acoustic encoder | Whisper-style full speech transcription backbones |

---

## 3. Interactions

- **`freeze_audio_encoder_params`**: Dictates whether these layers are updated during training.

---

### One-line intuition
> **`encoder_layers_for_audio` specifies the depth (number of layers) in the audio transformer encoder stack.**
