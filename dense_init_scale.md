
## 1. Why does `dense_init_scale` exist?

Weight initialization variance matters — initialize too large and activations explode in the first forward pass; too small and gradients vanish before learning can begin. MaxText uses scaled initialization by default (following μP-adjacent practices and the "1/sqrt(fan_in)" tradition), and `dense_init_scale` is a multiplicative knob on top of that base initialization.

```text
effective_init_std = base_init_std × dense_init_scale
```

This exists because:
1. Different architectures may need different initialization magnitudes even with the same shape
2. When replicating a specific model's published hyperparameters, the paper may specify an explicit init scale
3. Scaling experiments (testing different init regimes) benefit from a single scalar without editing initialization code

---

## 2. What it controls

`dense_init_scale` is applied to the initialization of **dense layer weight matrices** — primarily:
- Attention Q, K, V, and output projections
- MLP up/gate/down projections
- Output embedding (if not tied)

It does **not** control embedding lookup table initialization, layer norm scale/bias initialization, or bias terms.

---

## 3. Default

```yaml
dense_init_scale: 1.0
```

A scale of 1.0 leaves the underlying initialization scheme (typically truncated normal with std = `1/sqrt(fan_in)` or equivalent) unchanged. This is the neutral setting.

---

## 4. When to change it

| Scenario | Adjustment |
|---|---|
| Matching a specific paper's reported init scale | Set to match |
| Training instability early (loss spikes at step ~0) | Try reducing to 0.5–0.9 |
| μP (Maximal Update Parametrization) experiments | Set per the μP scale recipe for your width |
| Debugging: verifying init magnitude is correct | Set to known value, log weight norms at step 0 |

In practice, 1.0 works for the vast majority of runs. Tuning `dense_init_scale` is an advanced hyperparameter — most people don't touch it.

---

## 5. Interaction with model scale

At larger `base_emb_dim`, initialization variance naturally decreases (because `fan_in` is larger). If you're scaling up the model and want to preserve the activation magnitude at initialization, you might scale `dense_init_scale` inversely with `sqrt(emb_dim)` — but MaxText's default init already accounts for fan-in scaling, so this is usually unnecessary.

---

## 6. Connection to training stability

Poor initialization manifests as:
- Loss NaN or infinity at step 0–100
- Very large initial weight norms (check with `--log_weight_norms`)
- Gradient norms either enormous or zero from step 1

If you see any of these, `dense_init_scale` is one dial to adjust (alongside learning rate, clipping, and model size).

---

### One-line intuition

> **`dense_init_scale` is a multiplicative factor on top of the default weight initialization variance for dense layers — leave it at 1.0 unless you're matching a specific paper's init scheme or debugging early training instability.**
