
## 1. Why does it exist?

The `"intmp"` (integer mixed-precision) quantization scheme is not a single fixed configuration — it applies different quantization bitwidths and methods to different layers of the model. A 32-layer transformer might use int4 for some MLP layers, int8 for others, and keep attention projections in bfloat16, based on measured sensitivity.

This per-layer configuration can't fit in a single YAML field. It requires a structured config file. `quant_cfg_path` is the path to that file.

---

## 2. What the file contains

The config file at `quant_cfg_path` is a JSON or structured config that maps layer names/patterns to quantization settings:

```json
{
  "attention_qkv": {"bits": 8, "calibration": "absmax"},
  "attention_out": {"bits": 8, "calibration": "absmax"},
  "mlp_wi": {"bits": 4, "calibration": "percentile"},
  "mlp_wo": {"bits": 4, "calibration": "percentile"}
}
```

The exact schema is defined in `src/maxtext/configs/quantization/readme.md` and the MaxText quantization code.

---

## 3. Options

| Value | Behavior |
|---|---|
| `""` | No config file — not used (default) |
| Path string | Path to `intmp` quantization config file |

Default in base.yml:
```yaml
quant_cfg_path: ""
```

---

## 4. When is this relevant?

Only when `quantization: "intmp"`. The `intmp` scheme is designed for **inference deployment** — after pretraining, you run sensitivity analysis to determine which layers tolerate aggressive quantization (int4) vs. which need higher precision (int8 or fp16), then encode that in `quant_cfg_path`.

```text
workflow:
  1. Train with quantization: ""  (full precision)
  2. Run per-layer sensitivity analysis
  3. Build quant_cfg_path config assigning bits per layer
  4. Run PTQ or export with quantization: "intmp" + quant_cfg_path
  5. Deploy quantized model
```

For `"int8"` or `"fp8"` uniform quantization, `quant_cfg_path` is not used — all layers get the same treatment.

---

## 5. The MaxText quantization configs directory

MaxText ships example `intmp` config files in:
```text
src/maxtext/configs/quantization/
```

These provide starting points for common model families. The readme there describes the schema in detail.

---

## 6. Interaction with other params

| Param | Relationship |
|---|---|
| `quantization` | Must be `"intmp"` for this to matter |
| `use_qwix_quantization` | Determines which library reads `quant_cfg_path` |
| `weight_quantization_calibration_method` | Default calibration if not overridden per-layer in the config |

---

### One-line intuition

> **`quant_cfg_path` provides a per-layer quantization configuration file for the `"intmp"` scheme — because mixed-precision deployment requires different bitwidths on different layers, which can't be expressed in a single global setting.**
