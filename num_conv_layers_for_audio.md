## 1. Why does `num_conv_layers_for_audio` exist?

The front-end of an acoustic encoder utilizes 1D or 2D strided convolutions to downsample the raw spectrogram time frames (e.g. stride 2 per layer reduces time resolution by $2^N$). 

`num_conv_layers_for_audio` sets the number of sequential convolutional downsampling layers in the audio front-end.

```text
Spectrogram [T frames]
         │
         ├── Conv Layer 1 (Stride 2) ──► T / 2
         ├── Conv Layer 2 (Stride 2) ──► T / 4
         └── Conv Layer 3 (Stride 2) ──► T / 8  (num_conv_layers_for_audio = 3)
```

With 3 stride-2 layers, the temporal sequence is downsampled by $2^3 = 8	imes$, converting 1000 spectrogram frames into 125 compact audio tokens.

---

## 2. Options and Defaults

| Value | Temporal Downsampling Factor |
|---|---|
| `3` (Default) | 3 convolutional layers ($8\times$ temporal compression) |
| `2` | 2 convolutional layers ($4\times$ temporal compression) |
| `1` | 1 convolutional layer ($2\times$ temporal compression) |

---

### One-line intuition
> **`num_conv_layers_for_audio` sets the number of convolutional downsampling layers in the audio front-end, dictating temporal compression rate.**
