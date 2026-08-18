
## 1. Why a custom gather kernel for MoE

In MoE dispatch, after routing decisions are made, tokens need to be **gathered** from their original positions to the expert buffers. On standard TPU, JAX's default gather implementation uses general scatter/gather logic that isn't optimized for the MoE access pattern.

Google's Mosaic is an ML compiler for TPU that enables writing custom high-performance kernels that target the TPU's specific SIMD and memory hierarchy directly. A "Mosaic kernel" compiled for the MoE gather operation can exploit the structured access pattern of routing — tokens going to contiguous expert blocks — that the general gather doesn't leverage.

`use_gather_mosaic_kernel` switches the token gather operation from JAX's default to a custom Mosaic kernel.

---

## 2. What it does

```yaml
use_gather_mosaic_kernel: false  # (default) use JAX's standard gather
use_gather_mosaic_kernel: true   # use custom Mosaic kernel for token gather
```

---

## 3. This is a TPU-specific performance knob

Mosaic kernels are TPU-specific. This flag has no effect on GPU configurations. The benefit depends on:
- TPU generation and model
- Token gather sizes (batch × seq × emb_dim)
- Whether the gather is a bottleneck in your profile

---

## 4. Default

```yaml
use_gather_mosaic_kernel: false
```

Standard gather by default. Custom Mosaic kernel is an explicit opt-in.

---

## 5. Interaction with `use_ragged_sort`

Both are kernel-level optimizations for the MoE token permutation phase:

```text
use_ragged_sort         → optimizes the sort/permute step
use_gather_mosaic_kernel → optimizes the gather step
```

These operate on adjacent steps in the dispatch pipeline and can both be enabled together.

---

## 6. Options

| Value | Behavior |
|---|---|
| `false` (default) | Standard JAX gather |
| `true` | Custom Mosaic TPU kernel for gather — benchmark before enabling |

---

### One-line intuition

> **`use_gather_mosaic_kernel` replaces JAX's default token gather with a custom Mosaic TPU kernel optimized for MoE's structured access pattern — a TPU-specific throughput knob that requires benchmarking to validate benefit.**
