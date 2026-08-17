
## 1. Why does it exist?

In mixed-precision training, you have at least two distinct numerical representations in play at the same time:
- The **weights** (what you store and update)
- The **gradients** (what you compute to update weights)

These can and often should be different precisions, because their numerical stability requirements differ. `grad_dtype` controls the gradient precision independently of everything else.

```text
forward pass:  activations in dtype (bfloat16)
               weights in weight_dtype (float32 master copy)
backward pass: gradients accumulated in grad_dtype (float32)
optimizer:     weight update in weight_dtype precision
```

---

## 2. Why float32 for gradients?

Gradients are the output of a chain of multiplications and additions across the entire backward pass. Small numerical errors compound — a gradient that should be 1e-6 can underflow to zero in bfloat16 (which has only ~7 bits of mantissa precision).

bf16 range: ±3.4e38, but only ~2 decimal digits of precision.
fp32 range: ±3.4e38, with ~7 decimal digits of precision.

For gradient *accumulation* specifically — summing many small values — fp32 matters because summing many small bfloat16 values loses precision via catastrophic cancellation.

```text
bfloat16 accumulation:
  sum = 0.0 + 1e-4 + 1e-4 + ... (many terms)
  → precision loss grows with number of terms

float32 accumulation:
  sum is accurate to ~7 significant figures
  → safe for typical gradient magnitudes
```

---

## 3. Options

| Value | Effect |
|---|---|
| `"float32"` | Gradients in fp32 — default, safe, standard |
| `"bfloat16"` | Gradients in bf16 — memory savings, precision risk |

Default in base.yml:
```yaml
grad_dtype: "float32"
```

---

## 4. When would you set grad_dtype to bfloat16?

Almost never in pretraining. Bfloat16 gradients are:
- Occasionally used in extremely memory-constrained inference-adjacent scenarios
- Theoretically possible for fine-tuning on very stable loss landscapes
- A research experiment, not a production choice

The risk is gradient underflow on small values and loss instability that's hard to diagnose (the model trains, but subtly worse).

---

## 5. The three-dtype picture

Don't conflate these three separate dtype parameters:

```text
dtype        → activations during forward pass (default: bfloat16)
weight_dtype → weight storage and master copy   (default: float32)
grad_dtype   → gradient accumulation            (default: float32)
```

A production pretraining run typically keeps `dtype=bfloat16` for compute efficiency while keeping `weight_dtype=float32` and `grad_dtype=float32` for numerical stability. This is the standard mixed-precision recipe.

---

## 6. Interaction with quantization

When quantization is active (`quantization != ""`), gradients still flow through the quantized forward pass in whatever precision the quantization scheme uses. `grad_dtype` controls where those gradients are accumulated/stored in the backward pass, not the precision of the quantized operations themselves.

---

### One-line intuition

> **`grad_dtype=float32` keeps gradients in full precision during accumulation — critical because bfloat16's 7-bit mantissa causes catastrophic cancellation when summing the many small gradient contributions from a deep network.**
