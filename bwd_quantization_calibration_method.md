
## 1. Why does it exist?

During backpropagation through quantized layers, gradients flow backwards through the quantized operations. The quantized forward pass replaces the continuous matmul with a discrete one, and the backward pass must handle two things:

1. **Straight-through estimator (STE)**: since quantization is not differentiable, the gradient is typically passed through unchanged (STE)
2. **Gradient precision**: when the backward pass itself uses quantized operations (as in some QAT setups), those quantized backward-pass operations also need calibration

`bwd_quantization_calibration_method` selects how Qwix calibrates the quantization range for gradient tensors in the backward pass.

---

## 2. When does backward quantization actually happen?

In standard QAT with STE, gradients pass through the quantization op unchanged — backward-pass calibration doesn't matter because the backward path doesn't actually quantize anything.

In more aggressive quantized training setups (quantized gradients for memory reduction), the gradient tensors are themselves quantized, and calibration determines how well the gradient information is preserved.

```text
standard QAT (STE):
  forward: x_q = quantize(x)
  backward: dx = dL/dx  (full precision, passes through STE)
  → bwd_quantization_calibration_method doesn't matter

quantized backward pass:
  forward: x_q = quantize(x)
  backward: dx_q = quantize(dL/dx)  ← calibration matters here
  → bwd_quantization_calibration_method controls this calibration
```

---

## 3. Gradient distribution considerations

Gradients in neural networks are typically:
- Near-zero for most parameters (sparse signal)
- Well-behaved Gaussian-ish distribution
- Less prone to extreme outliers than activations (but this varies)

`"absmax"` tends to work fine for gradients because gradient distributions are typically better-behaved than activation distributions in the forward pass.

---

## 4. Options

Default in base.yml:
```yaml
bwd_quantization_calibration_method: "absmax"
```

Valid values: any method supported by Qwix's `qconfig.py`.

---

## 5. Only meaningful with Qwix

Requires:
```yaml
use_qwix_quantization: true
```

Under AQT, this is ignored.

---

## 6. The three calibration method params together

| Param | Tensor type | Typical outlier concern |
|---|---|---|
| `weight_quantization_calibration_method` | Weights | Low — stable Gaussian-ish |
| `act_quantization_calibration_method` | Activations | High — outlier channels common |
| `bwd_quantization_calibration_method` | Gradients | Medium — depends on training stage |

`"absmax"` is a fine default for all three. The activation method is the most likely to need tuning for production-quality quantization.

---

### One-line intuition

> **`bwd_quantization_calibration_method` calibrates Qwix's quantization range for gradient tensors in the backward pass — relevant only when gradients themselves are being quantized, and `"absmax"` is typically fine since gradient distributions are less outlier-prone than activations.**
