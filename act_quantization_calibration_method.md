
## 1. Why does it exist?

Activation tensors in transformers have a fundamentally different statistical profile than weight tensors:

```text
Weights:        roughly Gaussian, stable across training steps
Activations:    data-dependent, can have heavy tails and outliers
```

LLM activations are notorious for **outlier channels** — specific embedding dimensions that take values 10–100x larger than typical. This is well-documented in the quantization literature (e.g., LLM.int8, SmoothQuant). With absmax calibration, a single outlier channel forces the scale to be large, which wastes quantization resolution on all the normal-valued channels.

`act_quantization_calibration_method` selects how Qwix handles this for activation tensors specifically.

---

## 2. The outlier problem, visually

```text
activation values per channel (simplified):
  channel 0:  [-2.1, 1.8, -0.9, 2.3]    (normal)
  channel 1:  [-0.1, 0.3, -0.2, 0.1]    (normal)
  channel 2:  [-87.4, 92.1, -103.5, 88.2] (outlier!)

absmax calibration:
  scale = 103.5 / 127 = 0.815
  channel 0's max of 2.3 → quantized to 3 (wasted 124 integer bins!)
  → poor resolution on 99% of channels to accommodate 1 outlier
```

Methods that clip outliers (percentile-based) or smooth them (SmoothQuant-style) preserve resolution for the bulk of the distribution.

---

## 3. Options

Default in base.yml:
```yaml
act_quantization_calibration_method: "absmax"
```

Valid values: any method supported by Qwix's `qconfig.py`. Common options include absmax, percentile-based clipping, and methods that handle outliers explicitly.

---

## 4. When to change from absmax

If you're seeing post-quantization accuracy degradation with int8 activations:
1. Profile which layers have outlier activations (tools like `jax.debug.inspect_array` or framework profilers)
2. Try a percentile-based method for activation calibration (e.g., clip the top 1% of values)
3. Compare accuracy on a held-out evaluation set

For many models, `"absmax"` is sufficient for weights but insufficient for activations — this is one of the first levers to pull when int8 quantization hurts accuracy.

---

## 5. Relationship to the three calibration method params

```text
weight_quantization_calibration_method → weights (stable, absmax fine)
act_quantization_calibration_method    → activations (outlier-prone, may need clipping)
bwd_quantization_calibration_method    → gradients (backward pass)
```

All three default to `"absmax"`. In practice, activation calibration is the most likely candidate for tuning.

---

## 6. Only meaningful with Qwix

Requires:
```yaml
use_qwix_quantization: true
```

Under AQT, this parameter is ignored.

---

### One-line intuition

> **`act_quantization_calibration_method` controls how Qwix determines activation quantization ranges — `"absmax"` is the default but outlier channels in LLM activations often demand percentile-based clipping to avoid wasting quantization resolution on rare extreme values.**
