## 1. Why does `max_source_positions_for_audio` exist?

The audio encoder requires a static maximum sequence length (in spectrogram time frames or downsampled acoustic tokens) to pre-allocate positional embeddings and attention buffers in compiled XLA graphs.

`max_source_positions_for_audio` sets the maximum acoustic frame sequence length accepted by the audio encoder.

```text
Input Spectrogram Frames: [ 0 ... N-1 ]
                               │
                               ▼
        Assert N <= max_source_positions_for_audio (1500)
```

---

## 2. Options and Defaults

| Value | Maximum Audio Duration |
|---|---|
| `1500` (Default) | ~30 seconds of audio at standard 50Hz frame rate |
| `3000` | ~60 seconds of continuous audio |

---

## 3. Interactions

- **`max_sample_len_for_audio`**: Bounds raw input sample length before STFT extraction.

---

### One-line intuition
> **`max_source_positions_for_audio` defines the maximum acoustic frame sequence length (default 1500 frames $pprox 30\text{s}$) supported by the audio encoder.**
