
## 1. Why does `dropout_rate` exist?

Dropout is a regularization technique: during training, randomly zero out a fraction `p` of activations at each step, forcing the model to learn redundant representations and preventing co-adaptation between units.

```text
Without dropout:
  neuron A + neuron B → output
  (A can always rely on B being present)

With dropout (rate=0.3):
  30% chance neuron A is zeroed
  30% chance neuron B is zeroed
  → each neuron must work independently
  → prevents co-adaptation, reduces overfitting
```

It exists in MaxText because different training scenarios have fundamentally different regularization needs.

---

## 2. Default

```yaml
dropout_rate: 0.0
```

**Off by default.** This is intentional: large-scale language model pretraining almost universally skips dropout. The dataset is so large that the model doesn't memorize it — overfitting is not the bottleneck.

---

## 3. Pretraining vs. fine-tuning

The core reason dropout defaults to 0:

```text
Pretraining (hundreds of billions of tokens):
    - Model never sees the same data twice
    - Regularization isn't needed — underfitting is the risk
    - Dropout slows convergence without benefit
    dropout_rate: 0.0  ← correct

Fine-tuning (hundreds of millions of tokens on small domain):
    - Model can overfit to the small dataset
    - Regularization helps
    dropout_rate: 0.05 – 0.1  ← often beneficial
```

---

## 4. Where dropout is applied in MaxText

When `dropout_rate > 0`, dropout is applied at:
- After the attention output
- After each MLP sub-layer
- After the input embedding

The same rate is used throughout — MaxText doesn't expose per-layer dropout rates.

---

## 5. Interaction with stochastic depth

Stochastic depth (randomly dropping entire layers) is a different form of regularization not directly controlled by `dropout_rate`. If you need stochastic depth, that requires separate configuration or code changes.

---

## 6. Practical guidance

| Scenario | `dropout_rate` |
|---|---|
| LLM pretraining at any scale | 0.0 |
| Fine-tuning on domain-specific data (<1B tokens) | 0.05–0.1 |
| Matching a specific model's published config | Check paper |
| Small model experiments where overfitting is visible | 0.1–0.3 |

If you set `dropout_rate > 0` for pretraining and wonder why your loss is higher/slower — this is probably why.

---

## 7. The train/eval distinction

Dropout is automatically disabled during evaluation (`model.eval()` equivalent). MaxText handles this through JAX's `is_training` flag passed to the forward pass. You don't need to manually disable it at eval time.

---

### One-line intuition

> **`dropout_rate` enables dropout regularization — defaults to 0.0 for pretraining (dataset size provides implicit regularization) and is only useful for fine-tuning on small datasets where overfitting is a real risk.**
