## 1. Why does `activation_dropout_for_audio` exist?

Inside the audio encoder's feed-forward layers, intermediate activations can co-adapt. Applying dropout immediately after the non-linear activation function prevents co-adaptation and regularizes acoustic training.

```text
Linear 1 ──► [ Activation: GELU ] ──► [ Dropout (rate = activation_dropout_for_audio) ] ──► Linear 2
```

`activation_dropout_for_audio` sets the dropout probability applied after activations in the audio MLP.

---

## 2. Options and Defaults

| Value | Behavior |
|---|---|
| `0.0` (Default) | No activation dropout |
| `0.1` | 10% dropout in audio MLP |

---

### One-line intuition
> **`activation_dropout_for_audio` sets the dropout probability applied after the non-linear activation inside the audio encoder FFN.**
