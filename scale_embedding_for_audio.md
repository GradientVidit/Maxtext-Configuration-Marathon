## 1. Why does `scale_embedding_for_audio` exist?

In standard Transformer architectures (originating from *Attention Is All You Need* and adopted by Whisper), the initial input embeddings are multiplied by $\sqrt{d_{\text{model}}}$ before adding positional embeddings. This scales the learned embeddings so their variance matches the unit-variance sinusoidal positional signals.

```text
scale_embedding_for_audio: true
Input = (Conv_Embeddings * sqrt(d_model_for_audio)) + Positional_Embeddings

scale_embedding_for_audio: false
Input = Conv_Embeddings + Positional_Embeddings
```

`scale_embedding_for_audio` toggles $\sqrt{d_{\text{model}}}$ embedding scaling in the audio encoder.

---

## 2. Options and Defaults

| Value | Behavior | Convention |
|---|---|---|
| `true` (Default) | Multiplies embeddings by $\sqrt{d_{\text{model}}}$ | Whisper / Qwen-Omni convention |
| `false` | Passes embeddings directly without $\sqrt{d_{\text{model}}}$ scaling | Modern RoPE / LayerNorm-first conventions |

---

### One-line intuition
> **`scale_embedding_for_audio` toggles $\sqrt{d_{\text{model}}}$ scaling on acoustic embeddings before adding positional encodings.**
