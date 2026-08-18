## 1. Why does `encoder_ffn_dim_for_audio` exist?

Each acoustic encoder transformer block contains a feed-forward network (FFN) that projects token states into a higher-dimensional space for non-linear feature transformation before projecting back to `d_model_for_audio`.

```text
Hidden State [256] ──► [ Linear 1: 256 -> encoder_ffn_dim_for_audio (512) ] ──► [ Activation ] ──► [ Linear 2: 512 -> 256 ]
```

`encoder_ffn_dim_for_audio` sets the intermediate expansion width in the audio encoder MLP.

---

## 2. Options and Defaults

| Value | Expansion Ratio |
|---|---|
| `512` (Default) | $2\times$ expansion ($2 \times 256$) |
| `1024` / `2048` | Standard $4\times$ expansion |

---

## 3. Interactions

- **`d_model_for_audio`**: Base dimension being expanded.
- **`activation_function_for_audio`**: Activation used in the FFN bottleneck.

---

### One-line intuition
> **`encoder_ffn_dim_for_audio` sets the intermediate feed-forward expansion dimension inside the audio encoder layers.**
