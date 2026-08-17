
## 1. Why does it exist?

Training large models costs compute. Quantization reduces inference compute and memory. The typical workflow is: **train full-precision, then quantize for deployment**. Without `save_quantized_params_path`, doing this in MaxText would require a separate conversion pipeline outside the training loop.

`save_quantized_params_path` enables in-process, on-the-fly quantized weight export: MaxText quantizes the weights as part of the run and saves them to the specified path without stopping or modifying the training state.

```text
normal training run:
  checkpoints → gs://bucket/run/checkpoints/  (full precision)

with save_quantized_params_path:
  checkpoints → gs://bucket/run/checkpoints/  (full precision, untouched)
  quantized   → gs://bucket/run/int8_weights/ (quantized export)
```

---

## 2. What it saves

MaxText quantizes the model parameters using the active `quantization` scheme and writes them in AQT-compatible format to the specified path. This is:
- **Not** a training checkpoint (no optimizer state, no step count)
- **Only** the quantized parameters
- In a format loadable with `checkpoint_is_quantized=true` + `load_parameters_path`

---

## 3. Options

| Value | Behavior |
|---|---|
| `""` | No quantized export (default) |
| GCS/local path | Quantized parameters written to this path |

Default in base.yml:
```yaml
save_quantized_params_path: ""
```

---

## 4. When to use it

**Primary use case: Post-Training Quantization (PTQ) export.**

```text
workflow:
  1. Load a full-precision trained checkpoint (load_parameters_path)
  2. Set quantization: "int8" (or fp8, etc.)
  3. Set save_quantized_params_path: "gs://bucket/model_int8/"
  4. Run MaxText for a few steps (or 0 steps for pure export)
  5. The quantized weights are now at gs://bucket/model_int8/
  6. Deploy: load_parameters_path="gs://bucket/model_int8/" + checkpoint_is_quantized=true
```

This is cleaner than running a separate quantization pipeline because MaxText handles the JAX/XLA distributed quantization natively across all devices.

---

## 5. Quantization-Aware vs Post-Training

This parameter is primarily useful for **PTQ** (post-training quantization). During QAT, the training checkpoints are already saved in quantized form by the normal checkpointing pipeline — no separate export path is needed.

---

## 6. Interaction with `checkpoint_is_quantized`

These two parameters form the producer-consumer pair:

```text
save_quantized_params_path  → PRODUCE a quantized checkpoint
checkpoint_is_quantized     → CONSUME a quantized checkpoint
```

Always set `checkpoint_is_quantized=true` when loading the output of `save_quantized_params_path`.

---

### One-line intuition

> **`save_quantized_params_path` triggers an on-the-fly quantized weight export to the specified path during a run — the standard MaxText way to do post-training quantization without a separate conversion script.**
