
## 1. Why does it exist?

MoE (Mixture of Experts) models with expert parallelism present a scheduling challenge: routing tokens to experts requires an all-to-all communication, and the forward pass through each expert takes compute time. The standard execution is sequential — communicate, then compute.

**Batch-split scheduling** overlaps these by splitting the token batch into micro-batches:

```text
standard MoE step:
  all-to-all (tokens → experts)  →  expert GEMM  →  all-to-all (results back)

batch-split schedule:
  micro-batch 1: all-to-all →  expert GEMM  →  all-to-all
  micro-batch 2:              [all-to-all happens here, overlapped with micro-batch 1 GEMM]
  ...
```

When `use_batch_split_schedule=true` and quantization is active, the quantization operations themselves also get split across micro-batches. `use_manual_quantization=true` tells MaxText to handle this quantization splitting manually rather than relying on JAX's automatic execution to figure it out.

---

## 2. The batch-split + quantization problem

Automatic quantization (via AQT or Qwix) expects to see the full tensor and compute scale factors from it. When you split the batch into micro-batches, each micro-batch is a partial tensor — the auto-computed scale may be wrong (based on partial data instead of the full batch).

`use_manual_quantization=true` handles scale computation correctly across the split — computing it from each micro-batch independently or using a pre-computed global scale.

---

## 3. Options

| Value | Behavior |
|---|---|
| `false` | Standard quantization handling (default) |
| `true` | Manual quantization handling for micro-batch splits |

Default in base.yml:
```yaml
use_manual_quantization: false
```

---

## 4. Dependency chain

This parameter is only meaningful when:
```yaml
use_batch_split_schedule: true   # (MoE batch splitting enabled)
quantization: "int8"             # (or any quantization scheme)
```

Without both of those, `use_manual_quantization` has no effect.

---

## 5. When to use it

If you're training a MoE model with expert parallelism and both:
- `use_batch_split_schedule: true`
- Any quantization enabled

then set `use_manual_quantization: true` to avoid incorrect scale computation across micro-batch boundaries.

For dense (non-MoE) training or any training without `use_batch_split_schedule`, this parameter is irrelevant — leave it `false`.

---

## 6. Backend consideration

Since `use_manual_quantization` works at a level that interacts with AQT/Qwix internals, verify compatibility with whichever backend you're using (`use_qwix_quantization=true` or `false`).

---

### One-line intuition

> **`use_manual_quantization=true` makes MaxText handle quantization scale computation correctly when the token batch is split into micro-batches for MoE scheduling overlap — only relevant when `use_batch_split_schedule=true` and quantization is active.**
