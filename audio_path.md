## 1. Why does `audio_path` exist?

To test speech-to-text, audio question-answering, and acoustic reasoning during inference without spinning up a complex streaming audio server, MaxText provides `audio_path` to load local audio files directly.

`audio_path` specifies one or more local audio files (e.g. `.wav`, `.flac`, `.mp3`) for standalone evaluation.

```text
Audio File (.wav) ──► Audio Decoder ──► Mel Spectrogram ──► Audio Encoder ──► Replace <|audio|>
```

---

## 2. Options and Defaults

| Value | Behavior |
|---|---|
| `""` (Default) | Audio inference disabled |
| `"/path/speech1.wav,/path/speech2.wav"` | Comma-separated audio paths loaded into the acoustic pipeline |

---

## 3. Interactions

- **`use_audio`**: Master switch.
- **`audio_placeholder`**: Prompt placeholder replaced by audio embeddings.
- **`num_mel_bins_for_audio`**: Governs spectrogram extraction from the raw waveform.

---

### One-line intuition
> **`audio_path` specifies comma-separated local audio file paths for standalone acoustic evaluation and speech-language decoding.**
