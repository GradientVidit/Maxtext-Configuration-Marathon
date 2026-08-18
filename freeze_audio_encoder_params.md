## 1. Why does `freeze_audio_encoder_params` exist?

Similar to vision encoders, speech backbones (e.g., Whisper, Conformer, or USM) are pre-trained on massive speech corpora. When aligning an audio encoder with a language model, updating the delicate acoustic feature extractors early in training can destroy phonetic representations and cause gradient instability.

```text
Audio Input ──► [ Audio Encoder (FROZEN) ] ──► [ Audio Projector (TRAINING) ] ──► [ LLM (TRAINING) ]
                └──── Gradients = 0 ─────┘     └──────── Gradients Flow ───────────────────────────┘
```

`freeze_audio_encoder_params` prevents the audio encoder from receiving gradient updates during training.

---

## 2. Options and Defaults

| Value | Behavior | Recommended Stage |
|---|---|---|
| `true` (Default) | Audio encoder parameters are frozen | Multimodal alignment, projector pre-training |
| `false` | Audio encoder parameters are fully trainable | End-to-end speech-to-text fine-tuning, acoustic domain adaptation |

---

## 3. Interactions

- **`use_audio`**: Must be `true`.
- **`d_model_for_audio` / `encoder_layers_for_audio`**: Determines the frozen parameter footprint.

---

### One-line intuition
> **`freeze_audio_encoder_params` locks the acoustic encoder weights during multimodal training, preserving pretrained speech representations while saving optimizer memory.**
