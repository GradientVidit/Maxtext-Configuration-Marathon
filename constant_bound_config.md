
## 1. Why does it exist?

Dynamic quantization calibrates scale factors at runtime by inspecting each tensor's actual values (e.g., finding the max absolute value). This works well for most cases but has a fundamental limitation: the scale varies batch-to-batch.

For **static quantization** (like `fp8_full`), the scale is fixed at calibration time and applied as a constant at inference. This is faster (no runtime scale computation) but requires you to specify what the scale bounds should be.

`constant_bound_config` provides those pre-computed static bounds to the AQT library.

---

## 2. What it configures

AQT's static scaling mode needs scale bounds for each quantized tensor in the model. The format is a comma-separated string of float values:

```yaml
constant_bound_config: "0.5, 0.5, 0.5, 0.5, 0.5, 0.5"
```

Each value is the bound for a specific quantized tensor (the exact mapping depends on which layers AQT quantizes and in what order — typically query, key, value, attention output, MLP wi, MLP wo projections).

---

## 3. Options

| Value | Behavior |
|---|---|
| `""` | Not used — dynamic scaling (default) |
| `"f1, f2, ..., fn"` | Comma-separated static bounds for AQT |

Default in base.yml:
```yaml
constant_bound_config: ""
```

---

## 4. When is this relevant?

Only when using a quantization scheme that requires static bounds — primarily `"fp8_full"`. The flow is:

```text
1. Train or calibrate model with dynamic fp8 quantization
2. Run calibration pass to determine appropriate scale bounds
3. Set constant_bound_config to those bounds
4. Deploy/export with fp8_full + constant_bound_config for static-scale inference
```

For `"int8"` dynamic quantization or training with `"fp8"`, this parameter is unused.

---

## 5. Why static bounds?

```text
dynamic scaling:
  at each forward step:
    compute scale = max(|x|)
    quantize using that scale
  → accurate but adds runtime overhead for scale computation
  → scale must be materialized and propagated

static scaling (constant_bound_config):
  scale is a compile-time constant
  → no runtime scale computation
  → XLA can fuse more aggressively
  → faster inference, but only if calibration bounds are representative
```

---

## 6. Miscalibrated bounds = accuracy degradation

If the bounds are too small: values larger than the bound get clipped → clipping error → model output degradation.
If the bounds are too large: quantization step size is large → more rounding error per value → precision loss.

The bounds need to be calibrated against a representative dataset.

---

### One-line intuition

> **`constant_bound_config` provides pre-computed static scale bounds to AQT for fixed-scale quantization schemes like `fp8_full` — replacing runtime scale computation with compile-time constants for faster inference at the cost of requiring accurate calibration.**
