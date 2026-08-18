## 1. Why does `d_model_for_audio` exist?

The acoustic encoder transforms time-frequency spectrogram features into hidden neural representations. Acoustic signals are dense and continuous; setting an appropriate model embedding dimension ($d_{\text{model}}$) balances phonetic representation capacity with memory footprint.

```text
Log-Mel Filterbank ──► 1D Conv Downsampling ──► Vector of dim d_model_for_audio (256)
                                                        │
                                                        ▼
                                           [ Audio Transformer Block ]
```

`d_model_for_audio` defines the primary hidden channel dimensionality throughout the audio encoder stack.

---

## 2. Options and Defaults

| Value | Capacity | Context |
|---|---|---|
| `256` (Default) | Lightweight acoustic encoder | Qwen3-OmniMoE default |
| `512` / `768` | Medium/Large speech encoder | Whisper-Medium / Conformer-L |
| `1024` | Massive speech foundation model | Whisper-Large / USM |

---

## 3. Interactions

- **`encoder_attention_heads_for_audio`**: Head dimension $= d_{\text{model}} / H$. Must divide evenly (e.g. $256 / 4 = 64$).
- **`encoder_ffn_dim_for_audio`**: MLP expansion dimension inside each audio layer.
- **`scale_embedding_for_audio`**: Scales initial embeddings by $\sqrt{d_{\text{model}}}$.

---

### One-line intuition
> **`d_model_for_audio` sets the hidden feature dimension ($d_{\text{model}}$) of the acoustic transformer encoder.**
