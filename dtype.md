
## 1. Why does it exist?

Transformer inference and training are dominated by matrix multiplications. The numerical precision of those multiplications directly controls the tradeoff between:
- **Compute throughput** (lower precision = faster MXU operations)
- **Memory bandwidth** (lower precision = smaller tensors to move)
- **Numerical accuracy** (lower precision = more rounding error in each op)

`dtype` is the primary knob controlling what precision activations live in throughout the model's forward (and backward) pass.

```text
Token embeddings (dtype)
       ↓
 Attention QKV projections (dtype)
       ↓
 Attention scores + softmax (dtype, unless float32_qk_product overrides)
       ↓
 MLP (dtype, unless activations_in_float32 overrides pre-nonlinearity)
       ↓
 Output logits (cast to fp32 if cast_logits_to_fp32=true)
```

---

## 2. bfloat16 vs float32 — the key trade-off

| Property | bfloat16 | float32 |
|---|---|---|
| Exponent bits | 8 (same as fp32) | 8 |
| Mantissa bits | 7 | 23 |
| Dynamic range | ±3.4e38 | ±3.4e38 |
| Relative precision | ~1% error | ~0.00001% error |
| TPU MXU throughput | Full speed | 2x–4x slower |
| Memory footprint | 2 bytes/value | 4 bytes/value |

bfloat16 was designed specifically for neural network training: same dynamic range as float32 (no gradient overflow issues), but only 7-bit mantissa (some precision loss is acceptable for intermediate activations).

TPUs are tuned for bfloat16 — this is the native precision of the MXU. Running fp32 activations cuts throughput significantly.

---

## 3. Options

| Value | Behavior |
|---|---|
| `"bfloat16"` | Default — activations in bf16, fast MXU operations |
| `"float32"` | Full-precision activations — slower, higher memory, numerically safer |
| `"float16"` | IEEE fp16 — narrower dynamic range than bf16, overflow risk |

Default in base.yml:
```yaml
dtype: "bfloat16"
```

---

## 4. The three distinct dtype axes

Don't conflate these:

```text
dtype        → activations (this parameter)
weight_dtype → weight storage (see Part 3 — Core Model Architecture)
grad_dtype   → gradient accumulation
```

A typical production config runs:
```yaml
dtype: "bfloat16"       # fast compute
weight_dtype: "float32" # stable optimizer
grad_dtype: "float32"   # stable gradient accumulation
```

This is standard mixed-precision: bf16 compute, fp32 state.

---

## 5. Interactions with precision-override parameters

Several other parameters can locally override `dtype` at specific points:

| Param | What it overrides |
|---|---|
| `activations_in_float32` | Forces activations to fp32 before nonlinearities |
| `float32_qk_product` | Forces fp32 QK attention score computation |
| `float32_logits` | Forces fp32 inputs to attention softmax |
| `cast_logits_to_fp32` | Forces fp32 for output logits |
| `logits_dot_in_fp32` | fp32 for embedding dot-product logits |

These let you keep `dtype=bfloat16` globally but add targeted fp32 precision at specific numerically-sensitive operations.

---

## 6. When to change dtype

**Keep `bfloat16`** for nearly all pretraining — it's the reason TPU training is fast. Switching to `float32` halves or worse your throughput for marginal numerical benefit (the optimizer's fp32 master copy already handles the critical precision).

**Consider `float32`** if:
- Training instability (loss spikes, NaN) persists after other fixes and you want to rule out activation precision as the cause
- Debugging a numerical issue and need to isolate whether bf16 precision is the culprit

---

### One-line intuition

> **`dtype` sets the precision of all activations in the forward pass — `bfloat16` is the default because it matches the TPU MXU's native precision, cutting memory in half vs fp32 while preserving the dynamic range needed to avoid overflow.**
