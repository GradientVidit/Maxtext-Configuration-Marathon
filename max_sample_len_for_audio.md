## 1. Why does `max_sample_len_for_audio` exist?

Raw audio files loaded from disk have variable duration and sample lengths (e.g. 16,000 samples per second for 16kHz audio). For static-shape XLA compilation on TPUs, input audio waveform buffers must be padded or truncated to a fixed maximum length.

`max_sample_len_for_audio` sets the maximum raw audio sample length (in audio waveform samples) accepted by the data pipeline.

```text
Raw Audio Waveform (Samples): [ s_0 , s_1 , ... , s_L ]
                                     │
                                     ▼
                     Truncate or Pad to max_sample_len_for_audio (10000)
```

---

## 2. Options and Defaults

| Value | Raw Audio Duration (@ 16kHz) | Context |
|---|---|---|
| `10000` (Default) | ~0.625 seconds (short audio snippet / chunk) | Streaming audio chunking |
| `160000` | 10 seconds of 16kHz audio | Standard speech utterance processing |
| `480000` | 30 seconds of 16kHz audio | Whisper standard audio window |

---

### One-line intuition
> **`max_sample_len_for_audio` defines the static waveform buffer size in raw audio samples for audio loading and batch padding.**
