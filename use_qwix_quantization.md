
## 1. Why does it exist?

MaxText originally built its quantization on **AQT (Accurate Quantized Training)** — a Google-internal library for doing quantized matrix multiplications in JAX. However, AQT is being phased out. The replacement is **Qwix** (also from Google), which is more actively maintained and has a cleaner API.

`use_qwix_quantization` is the migration switch: it selects Qwix as the quantization backend instead of AQT.

The inline comment in base.yml is explicit:
```yaml
use_qwix_quantization: false # [DEPRECATED: AQT will be removed in a future release.
                              # It is strongly recommended to set use_qwix_quantization to true]
```

---

## 2. What changes when you switch

```text
use_qwix_quantization=false (AQT backend, legacy):
  quantization logic → AQT library
  scale computation → AQT conventions
  checkpoint format → AQT-specific pytree structure

use_qwix_quantization=true (Qwix backend, current):
  quantization logic → Qwix library
  scale computation → Qwix conventions
  calibration method → weight_quantization_calibration_method / act_ / bwd_
  checkpoint format → Qwix pytree structure
```

The numerical results can differ slightly between backends due to different default calibration approaches and scale computation details.

---

## 3. Options

| Value | Behavior |
|---|---|
| `false` | AQT backend — legacy, will be removed (default in base.yml) |
| `true` | Qwix backend — recommended for all new work |

Default in base.yml:
```yaml
use_qwix_quantization: false
```

> **But the maintainers recommend `true` for all new work given the deprecation notice.**

---

## 4. Checkpoint compatibility

**AQT and Qwix produce checkpoints with different pytree structures.** A checkpoint saved with AQT (`use_qwix_quantization=false`) cannot be directly loaded with Qwix (`use_qwix_quantization=true`) and vice versa. Mixing them will produce shape/key mismatches at load time.

```text
if checkpoint was saved with AQT:
  → load with use_qwix_quantization=false

if checkpoint was saved with Qwix:
  → load with use_qwix_quantization=true
```

Be consistent across save and load.

---

## 5. Calibration method interaction

The `weight_quantization_calibration_method`, `act_quantization_calibration_method`, and `bwd_quantization_calibration_method` parameters only have effect when `use_qwix_quantization=true`. Under AQT, calibration is controlled by AQT's own configuration.

---

## 6. Practical recommendation

For any **new** pretraining or fine-tuning config:
```yaml
use_qwix_quantization: true
```

For **existing** configs with AQT checkpoints you're resuming:
```yaml
use_qwix_quantization: false
```
Until you're ready to migrate (which requires re-quantizing from a full-precision checkpoint).

---

### One-line intuition

> **`use_qwix_quantization=true` switches from the deprecated AQT quantization backend to Qwix — set it to `true` on all new work; existing AQT checkpoints can't be directly resumed under Qwix.**
