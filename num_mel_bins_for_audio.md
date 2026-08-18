## 1. Why does `num_mel_bins_for_audio` exist?

Raw audio waveforms (sampled at e.g. 16kHz or 24kHz) are converted into 2D time-frequency representations using a Short-Time Fourier Transform (STFT) followed by a Mel-scale filterbank. The number of mel frequency bins defines the vertical resolution of the acoustic spectrogram.

```text
Raw Waveform (.wav) ──► STFT ──► [ Mel Filterbank ] ──► Spectrogram [Time, num_mel_bins_for_audio (128)]
                                                               │
                                                               ▼
                                                  Convolutional Audio Frontend
```

`num_mel_bins_for_audio` sets the number of frequency bins in the input log-mel spectrogram.

---

## 2. Options and Defaults

| Value | Frequency Resolution | Standard Model |
|---|---|---|
| `128` (Default) | 128 frequency bins (high fidelity) | Qwen3-Omni, Whisper-v3 |
| `80` | 80 frequency bins (standard speech) | Whisper-v1/v2, standard ASR models |

---

## 3. Failure Modes

- Supplying spectrograms extracted with 80 bins when `num_mel_bins_for_audio: 128` will fail at the first 1D/2D convolution layer due to channel dimension mismatch.

---

### One-line intuition
> **`num_mel_bins_for_audio` sets the number of frequency bins (default 128) in the log-mel spectrogram feeding the audio encoder.**
