## 1. Why does `force_q_layout` exist?

In high-performance attention algorithms, the memory stride and layout of multi-dimensional tensors directly dictate whether matrix units can stream data continuously or suffer from costly transposition memory copies:

```text
Default Layout:
Query Tensor Q: [..., Batch, Seq_Length, Num_Heads, Head_Dim]
Inner dot-product loop requires repeatedly slicing across Seq_Length.

Transposed / Forced Layout (force_q_layout: true):
Query Tensor Q: [..., Batch, Num_Heads, Head_Dim, Seq_Length]
Sequence dimension is contiguous at the minor physical axis, matching inner scan memory strides in JAX splash loops.
```

When running non-Pallas Splash Attention (`use_jax_splash: true`), XLA's automatic layout assignment can sometimes introduce redundant transposition copies (`kTranspose`) during inner loop unrolling.

`force_q_layout` explicitly enforces the Query tensor layout to `[..., NUM_HEADS, HEAD_DIM, SEQ_LENGTH]`, optimizing register and HBM memory layout for pure JAX Splash Attention loops.

---

## 2. Mechanics & Constraints

- **Applicability**: `force_q_layout` is specifically designed for and active when `use_jax_splash: true`.
- **Memory Optimization**: By placing `SEQ_LENGTH` at the minor dimension, block slicing along the sequence axis within `jax.lax.scan` directly matches the native vector register loads on accelerator memory.
- When `use_jax_splash: false`, hardware kernels (Pallas/Tokamax) manage their own internal VMEM/SRAM layout descriptors independently, making `force_q_layout` unnecessary.

---

## 3. Options & Default

| Parameter | Type | Default | Options |
| :--- | :--- | :--- | :--- |
| `force_q_layout` | `bool` | `false` | `true` (force `[..., H, D, S]` layout), `false` (default standard layout) |

---

## 4. Interactions with Related Parameters

| Related Parameter | Interaction |
| :--- | :--- |
| `use_jax_splash` | `force_q_layout` is only meaningful when `use_jax_splash: true`. |
| `compute_axis_order` / `ar_cache_axis_order` | Governs KV-cache layouts, whereas `force_q_layout` directly modifies intermediate Query projection memory strides. |

---

## 5. Practical Guidance

| Scenario | Recommendation |
| :--- | :--- |
| **Standard Training with Pallas Kernels** | Leave `force_q_layout: false`. |
| **Benchmarking or Profiling `use_jax_splash: true`** | Set `force_q_layout: true` to eliminate redundant XLA transposition ops in HLO graph traces. |

---

### One-line intuition

> `force_q_layout` forces the Query tensor into a head-major, sequence-minor layout (`[..., H, D, S]`) when using JAX Splash Attention to avoid XLA transposition overhead.
