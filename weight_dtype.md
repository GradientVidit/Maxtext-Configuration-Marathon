
## 1. Why does `weight_dtype` exist?

A modern transformer has three distinct populations of numbers that can each have their own dtype:

```text
┌─────────────────────────────────────────┐
│  Parameter (weights)    ← weight_dtype  │
│  Activations            ← dtype         │
│  Gradients              ← grad_dtype    │
└─────────────────────────────────────────┘
```

These are separated because they have different precision requirements and different memory/compute tradeoffs:

- **Weights** are read every step but never modified in-place (the optimizer holds a separate master copy). Their dtype affects HBM bandwidth at every forward pass.
- **Activations** are computed fresh each step; precision matters for numerical stability through layers.
- **Gradients** are accumulated during backward; fp32 accumulation prevents gradient underflow in long runs.

Without this separation, you'd be forced to either make everything fp32 (slow) or everything bf16 (potentially unstable).

---

## 2. What it controls

`weight_dtype` controls the dtype that **model parameters are stored and loaded in** — the in-HBM representation of the weight matrices themselves.

```text
weight_dtype="float32"
    → embed, attention Q/K/V/O, MLP wi/wo matrices stored as fp32
    → each forward pass: fp32 weights → compute → bf16/fp32 activations

weight_dtype="bfloat16"
    → same matrices stored as bf16
    → 2× memory reduction per weight tensor
```

---

## 3. The mixed-precision pattern

The most common production setup is:

```yaml
weight_dtype: "bfloat16"   # weights in bf16 → 2× less HBM
dtype: "bfloat16"          # activations in bf16
grad_dtype: "float32"      # gradient accumulation in fp32
```

This is the standard AMP (Automatic Mixed Precision) approach for large-scale training. The optimizer state (master weights + momentum/variance) is kept in fp32 by the optimizer internally, separate from what `weight_dtype` controls.

---

## 4. Default

```yaml
weight_dtype: "float32"
```

The default is fp32 — safe, numerically stable, but memory-heavy. Real pretraining configs almost always override to `"bfloat16"`.

---

## 5. Options

| Value | Memory | Notes |
|---|---|---|
| `"float32"` | 4 bytes/param | Default. Numerically safest. Typical for debugging or small models. |
| `"bfloat16"` | 2 bytes/param | Standard for large-scale pretraining. Same exponent range as fp32, lower mantissa precision. |
| `"float16"` | 2 bytes/param | Narrower exponent range than bf16 → more overflow/underflow risk. Less common in MaxText. |

---

## 6. Interaction with `dtype`

`weight_dtype` ≠ `dtype`. They control different tensors:

```text
weight_dtype: "bfloat16"  →  weight tensors in bf16
dtype: "bfloat16"         →  activation tensors in bf16

If weight_dtype="float32" but dtype="bfloat16":
    weights are fp32, activations are bf16
    → XLA promotes or casts at the matmul boundary
    → This is legal but slightly wasteful vs. both being bf16
```

---

## 7. Interaction with quantization

If you're using `quantization: "int8"` or `quantization: "fp8"`, the quantization system operates on the weights at a lower level — `weight_dtype` sets the baseline dtype before quantization shards/scales are applied.

---

## 8. Memory math

For a 7B parameter model:
- `float32`: 7B × 4 bytes = **28 GB**
- `bfloat16`: 7B × 2 bytes = **14 GB**

That 14 GB difference directly determines whether a model fits on a given TPU/GPU configuration.

---

### One-line intuition

> **`weight_dtype` sets the in-HBM dtype of model weight matrices — the main lever for trading weight memory footprint against numerical precision, independently of activation or gradient dtypes.**
