
## 1. Why does it exist?

On TPUs, matrix multiplications are executed by the MXU (Matrix Multiply Unit). The MXU natively multiplies bfloat16 inputs and accumulates into float32 — even at the lowest precision setting. The `precision` hint controls something subtler: **how many passes the MXU makes** to improve the precision of the multiplication itself.

More passes = more accurate mantissa in the result = slower throughput.

`matmul_precision` is MaxText's way of setting that hint for all matrix multiplies in the model.

---

## 2. What the levels actually mean on TPU

A critical nuance: the MXU **always accumulates in float32**, even at `DEFAULT`. The difference between levels is how many multiplier passes it executes to approximate higher-precision multiplication:

```text
"default"  → 1-pass bfloat16 multiply, fp32 accumulate
             fastest; mantissa precision limited by bf16's 7 bits
"high"     → 3-pass algorithm (similar to bfloat16_3x)
             more accurate mantissa, ~3x slower multiply throughput
"highest"  → 9-pass algorithm (approximates float32×float32)
             most accurate, slowest; approaches true fp32 precision
```

The accumulator is fp32 in all three cases. What changes is the **number of MXU passes used to compute each multiply** before accumulation.

MaxText passes this directly to `jax.lax.Precision`:

| MaxText value | JAX equivalent | Hardware behavior |
|---|---|---|
| `"default"` | `Precision.DEFAULT` | 1-pass bf16 multiply, fp32 accumulate |
| `"high"` | `Precision.HIGH` | 3-pass — better mantissa precision |
| `"highest"` | `Precision.HIGHEST` | 9-pass — near fp32 multiply precision |

Note: On GPU (tensor cores), these settings have different hardware implementations — behavior varies by architecture. The description above is specifically for TPU MXU.

---

## 3. Options

Default in base.yml:
```yaml
matmul_precision: "default"
```

The `"default"` is the right choice for essentially all pretraining. One MXU pass per multiply is the design point for training LLMs.

---

## 4. The real relationship to dtype

`dtype` and `matmul_precision` operate at different levels:

```text
dtype            → what dtype the INPUTS to the GEMM are in (memory representation)
matmul_precision → how many passes the hardware uses for the multiply (computation precision)
```

You can have `dtype=bfloat16` (inputs in bf16) and `matmul_precision="highest"` (hardware uses 9 passes for the multiply), or leave it at `"default"` (1 pass). These are orthogonal dimensions of numerical precision.

---

## 5. When to change it

Almost never for training. The `"default"` mode is why TPU training is fast.

**Legitimate uses:**
- **Numerical debugging**: if you suspect GEMM precision is causing instability, `"highest"` isolates that variable — it rules out mantissa precision as the cause
- **Reference implementation**: when building a numerically exact reference to compare against, you want `"highest"` to minimize hardware-level rounding
- **Validating numerical equivalence**: matching a known-good CPU/GPU reference result

For production training, leave at `"default"`.

---

## 6. Performance impact

Switching from `"default"` to `"highest"` requires ~9x more MXU operations per GEMM. For a model where training costs millions of dollars, this overhead is never justified by marginal numerical accuracy in the multiply — the fp32 accumulation already handles catastrophic cancellation.

---

### One-line intuition

> **`matmul_precision` controls how many MXU passes XLA uses per matrix multiply — `"default"` is one-pass bfloat16 multiply with fp32 accumulation (the TPU's native operating point), and higher settings add passes to improve mantissa precision at a proportional throughput cost.**
