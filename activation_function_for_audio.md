## 1. Why does `activation_function_for_audio` exist?

Different audio encoder architectures (e.g. Whisper, Conformer, wav2vec 2.0) utilize different non-linear activation functions in their MLP blocks to balance gradient stability and expressiveness.

`activation_function_for_audio` specifies the non-linear activation function applied in the audio encoder's feed-forward layers.

```text
FFN Input ──► Linear ──► [ activation_function_for_audio: "gelu" / "relu" / "silu" ] ──► Linear
```

---

## 2. Options and Defaults

| Value | Behavior | Model Family |
|---|---|---|
| `"gelu"` (Default) | Gaussian Error Linear Unit | Qwen3-Omni, Whisper, BERT |
| `"silu"` / `"swish"` | Sigmoid Linear Unit | Llama-style architectures |
| `"relu"` | Rectified Linear Unit | Classic acoustic models |

---

### One-line intuition
> **`activation_function_for_audio` selects the non-linear activation function (default `"gelu"`) used inside the audio encoder FFN.**
