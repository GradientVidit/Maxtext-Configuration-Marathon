## 1. Why does `encoder_attention_heads_for_audio` exist?

Self-attention in the audio encoder captures long-range acoustic relationships, pitch contours, phoneme transitions, and background audio cues across time frames. Multi-head attention partitions the audio hidden space across parallel attention heads.

```text
Audio Hidden State [256] ──► Split into encoder_attention_heads_for_audio (4 heads)
                                     │
                                     ▼
                             Head Dim = 256 / 4 = 64
```

`encoder_attention_heads_for_audio` sets the number of attention heads in the audio encoder.

---

## 2. Options and Defaults

| Value | Head Count | Head Dimension |
|---|---|---|
| `4` (Default) | 4 heads | $256 / 4 = 64$ |
| `8` / `16` | 8 or 16 heads | For larger $d_{\text{model}}$ values ($512$ or $1024$) |

---

## 3. Interactions

- **`d_model_for_audio`**: Must be divisible by `encoder_attention_heads_for_audio`.

---

### One-line intuition
> **`encoder_attention_heads_for_audio` sets the number of attention heads in the acoustic transformer layers.**
