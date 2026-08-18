## 1. Why does `attention_dropout_for_audio` exist?

Audio signals often contain background noise, reverberation, and channel distortion. During training on small or clean speech datasets, the acoustic attention heads can overfit to specific recording environments.

`attention_dropout_for_audio` applies dropout regularization to the attention probability matrix inside the audio encoder.

```text
Audio Attention Logits ──► Softmax ──► [ Dropout (rate = attention_dropout_for_audio) ] ──► Multiply by V
```

---

## 2. Options and Defaults

| Value | Behavior |
|---|---|
| `0.0` (Default) | Attention dropout disabled (standard in large-scale pre-training) |
| `0.1` | 10% dropout for regularizing speech fine-tuning |

---

### One-line intuition
> **`attention_dropout_for_audio` sets the dropout rate applied to the attention weight matrices within the audio encoder.**
