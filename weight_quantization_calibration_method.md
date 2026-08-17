
## 1. Why does it exist?

Quantization requires mapping continuous floating-point values to a discrete set of integer values. The quality of this mapping depends entirely on how you determine the **quantization range** — the [min, max] interval that gets mapped to the integer range.

Choose too wide a range → large quantization step size → many distinct float values collapse to the same integer → precision loss.
Choose too narrow a range → values outside the range get clipped → clipping error.

The calibration method determines how that range is chosen. There are multiple valid approaches with different accuracy/speed tradeoffs.

`weight_quantization_calibration_method` sets which one Qwix uses for weight tensors.

---

## 2. What calibration methods mean

| Method | How range is determined | Properties |
|---|---|---|
| `"absmax"` | `[-max(|x|), +max(|x|)]` | Simple, fast, sensitive to outliers |
| Min-max | `[min(x), max(x)]` | Full coverage, most sensitive to outliers |
| Percentile | `[p_low, p_high]` percentiles | Clips outliers, smoother distribution |
| Moving average | Running stats across batches | Stable over time, needs calibration data |

Qwix supports multiple methods — the full list is at [qwix/qconfig.py#L70-L80](https://github.com/google/qwix/blob/dc2a0770351c740e5ab3cce7c0efe9f7beacce9e/qwix/qconfig.py#L70-L80).

---

## 3. Why `absmax` is the default

For **weights** (unlike activations), the distribution is typically well-behaved and doesn't have extreme outliers. Absmax is:
- Computationally trivial (one reduction operation)
- Accurate enough for weights in most int8 regimes
- Symmetric, which matches the symmetry of weight distributions in trained models

For activations, outlier sensitivity matters more (some transformer activations have heavy-tailed distributions), which is why `act_quantization_calibration_method` may warrant different methods.

---

## 4. Options

Default in base.yml:
```yaml
weight_quantization_calibration_method: "absmax"
```

Valid values: any method string supported by Qwix's `qconfig.py`. Check the Qwix repo for the authoritative list.

---

## 5. The three calibration method params

These are three independent knobs covering different points in the computation graph:

```text
weight_quantization_calibration_method → how to calibrate weight tensors
act_quantization_calibration_method    → how to calibrate activation tensors
bwd_quantization_calibration_method    → how to calibrate gradient tensors (backward pass)
```

They can all be different. A common research configuration:
```yaml
weight_quantization_calibration_method: "absmax"       # weights are stable
act_quantization_calibration_method: "percentile_99"   # activations have outliers
bwd_quantization_calibration_method: "absmax"          # gradients are symmetric
```

---

## 6. Dependency on Qwix

These calibration method parameters only apply when:
```yaml
use_qwix_quantization: true
```

Under AQT (legacy backend), calibration is controlled by AQT's own configuration — these params are ignored.

---

### One-line intuition

> **`weight_quantization_calibration_method` tells Qwix how to determine the range-mapping for weight quantization — `"absmax"` (default) is fast and works well for typical weight distributions, but outlier-heavy activations often benefit from percentile-based calibration instead.**
