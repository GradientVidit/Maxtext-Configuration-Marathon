
## 1. What SparseCore is and why it matters

Google TPUs have a specialized hardware unit called **SparseCore** (present in TPU v5+) designed for accelerating operations common in recommendation systems — particularly sparse embedding lookups and gather/scatter over large sparse arrays.

MaxText's MoE ragged gather operation has structural similarities to sparse embedding lookup: you're gathering token activations from a large buffer based on routing indices. MaxText can use SparseCore-accelerated kernels for this gather instead of the standard TPU matrix engines.

`ragged_gather_fallback` is the escape hatch: when `true`, it forces the plain JAX reference implementation instead of the SparseCore kernel.

---

## 2. What it does

```yaml
ragged_gather_fallback: false  # (default) use SparseCore ragged-gather kernel
ragged_gather_fallback: true   # force JAX reference implementation (no SparseCore)
```

This is a **fallback**, not a feature flag. The name signals intent: you're falling back to a less optimized path for debugging or compatibility.

---

## 3. When to use it

**Debugging correctness:** if you suspect the SparseCore kernel has a bug, `ragged_gather_fallback=True` lets you verify that the JAX reference produces the correct result.

**Hardware without SparseCore:** on TPU versions without SparseCore, the SparseCore kernel path may error or degrade; the fallback ensures you get the JAX path.

**Profiling baseline:** measure JAX reference vs. SparseCore kernel performance.

**Production:** leave at `false`. SparseCore is faster if available.

---

## 4. Default

```yaml
ragged_gather_fallback: false
```

Use SparseCore kernel when available.

---

## 5. `ragged_gather_fallback` vs. `ragged_gather_reduce_fallback`

These control two different operations:

| Param | Operation |
|---|---|
| `ragged_gather_fallback` | Ragged gather (forward: collect tokens → expert buffers) |
| `ragged_gather_reduce_fallback` | Ragged gather-reduce (combines gather with reduction, used in combine step) |

They can be set independently — you might force fallback on just one to isolate which kernel is misbehaving.

---

## 6. Options

| Value | Behavior |
|---|---|
| `false` (default) | SparseCore ragged-gather kernel |
| `true` | JAX reference implementation |

---

### One-line intuition

> **`ragged_gather_fallback` forces the plain JAX reference implementation instead of the SparseCore-accelerated ragged gather kernel — a debugging fallback; leave `false` in production to use the faster hardware path.**
