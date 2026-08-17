
## 1. Why does it exist?

A 7B–70B+ parameter model in bfloat16 is enormous — it saturates HBM and makes training memory-constrained. Quantization compresses weights and/or activations to lower bit-widths, enabling:
- Larger models on the same hardware
- Higher batch sizes at the same memory
- Faster memory-bandwidth-bound operations (transferring 4-bit values is 8x cheaper than 32-bit)

`quantization` is the master switch that selects which quantization scheme applies to the transformer layers.

```text
quantization = ""       → no quantization; everything in dtype (bfloat16)
quantization = "int8"   → 8-bit integer weights/activations via AQT
quantization = "fp8"    → 8-bit float GEMM on NVIDIA hardware
quantization = "intmp"  → mixed int precision, inference-oriented
```

---

## 2. How quantization actually works in MaxText

MaxText uses the **AQT library** (by default) or **Qwix** (if `use_qwix_quantization=true`) to replace matmuls with quantized equivalents.

The key idea: instead of computing `Y = X @ W` in bfloat16, compute:

```text
quantize(X) → X_q  (e.g., int8)
quantize(W) → W_q  (e.g., int8)
Y_q = X_q @ W_q    (integer matmul — faster, lower memory)
dequantize(Y_q) → Y (back to bfloat16 for subsequent ops)
```

Scale factors are computed per-tensor or per-channel (depending on the calibration method) to minimize the quantization error.

---

## 3. Options

| Value | Description | Hardware |
|---|---|---|
| `""` | No quantization — full bfloat16 | Any |
| `"int8"` | Dynamic-range 8-bit integer quantization via AQT | Any (TPU, GPU) |
| `"intmp"` | Mixed-precision integer (see `src/maxtext/configs/quantization/readme.md`) | Inference |
| `"fp8"` | 8-bit floating-point GEMMs | NVIDIA GPUs only |
| `"nanoo_fp8"` | FP8 GEMMs for AMD | AMD MI300/MI325 |
| `"fp8_full"` | FP8 with static (not dynamic) scaling | NVIDIA GPUs |

Default in base.yml:
```yaml
quantization: ""
```

---

## 4. int8 quantization mechanics

`"int8"` uses **dynamic-range quantization**:
- At each GEMM, compute scale factor from the max absolute value of the tensor
- Quantize: `x_q = round(x / scale)` — maps to [-128, 127]
- Compute integer matmul
- Dequantize: `y = y_q * scale_x * scale_w`

This is the most portable scheme — works on TPU and GPU. When using the **Qwix backend** (`use_qwix_quantization=true`), the calibration method is controlled by `weight_quantization_calibration_method` / `act_quantization_calibration_method`. Under the default **AQT backend**, calibration is handled internally by AQT.

---

## 5. fp8 quantization

FP8 (`"fp8"`, `"fp8_full"`) uses 8-bit floating-point formats (E4M3 or E5M2) rather than integers. Available on NVIDIA H100+ GPUs which have native FP8 tensor cores.

```text
FP8 E4M3: range ±448, 3-bit mantissa
FP8 E5M2: range ±57344, 2-bit mantissa
```

`"fp8"` uses dynamic scaling (scales computed on-the-fly); `"fp8_full"` uses static scales (calibrated once, applied fixed).

---

## 6. Interaction map

| Param | Depends on quantization being... |
|---|---|
| `use_qwix_quantization` | Any — switches backend (AQT vs Qwix) |
| `constant_bound_config` | Static scaling schemes (`fp8_full`) |
| `replicate_quant_scale` | Any — performance tuning for scale replication |
| `quant_cfg_path` | `"intmp"` — per-layer config |
| `checkpoint_is_quantized` | Any — tells loader checkpoint is pre-quantized |
| `save_quantized_params_path` | Any — save quantized weights to path |
| `weight_quantization_calibration_method` | Qwix backend |
| `act_quantization_calibration_method` | Qwix backend |
| `quantization_local_shard_count` | Any — sharding for range finding |

---

## 7. Training vs inference quantization

- **Quantization-Aware Training (QAT)**: use `quantization` during training. The model learns to compensate for quantization error.
- **Post-Training Quantization (PTQ)**: train full-precision, then export with `save_quantized_params_path` to quantize the checkpoint.
- **`"intmp"`**: primarily inference-oriented — mixed-precision for deployment, not training quality optimization.

---

### One-line intuition

> **`quantization` is the master scheme selector for model compression — `""` means full bfloat16, `"int8"` is the standard training-compatible path, and `"fp8"` variants require matching hardware (NVIDIA H100/AMD MI300).**
