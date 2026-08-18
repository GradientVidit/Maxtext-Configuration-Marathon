## 1. Why does it exist?

In standard Splash Attention backward execution, computing the Query gradient ($dQ$) and computing the Key/Value gradients ($dK, dV$) are executed as two separate sequential Pallas kernel launches:
1. Kernel 1: Streams KV to accumulate $dQ$.
2. Kernel 2: Streams Q to accumulate $dK$ and $dV$.

Running two distinct kernel launches requires reading the intermediate forward activations ($Q, K, V$, and softmax statistics) from HBM twice.

**Fused Backward Kernel** combines the $dQ$ and $dK/dV$ gradient computations into a single unified Pallas kernel launch, cutting intermediate HBM memory roundtrips in half.

```text
Unfused Backward (sa_use_fused_bwd_kernel: false):
  Launch 1: Read (Q, K, V, dO) from HBM ──→ Compute dQ
  Launch 2: Read (Q, K, V, dO) from HBM ──→ Compute dK, dV

Fused Backward (sa_use_fused_bwd_kernel: true):
  Single Launch: Read (Q, K, V, dO) once ──→ Compute dQ, dK, and dV concurrently in VMEM!
```

`sa_use_fused_bwd_kernel` enables a fused backward pass kernel for Splash Attention.

---

## 2. Options & Configuration

| Value | Meaning |
|---|---|
| `false` (default) | Two separate backward kernels for $dQ$ and $dK/dV$. |
| `true` | Fused unified backward kernel; reduces HBM memory bandwidth consumption. |

Default in `base.yml`:
```yaml
sa_use_fused_bwd_kernel: false
```

---

## 3. Practical Usage

- **Performance Benefit**: Reduces backward pass execution time for memory-bandwidth-bound attention layers on large sequence lengths.
- **VMEM Pressure**: Requires more concurrent on-chip Vector Memory to hold both $dQ$ and $dK/dV$ accumulators simultaneously.

---

### One-line intuition

> **`sa_use_fused_bwd_kernel` fuses the $dQ$ and $dK/dV$ gradient computations into a single unified Pallas kernel launch, halving HBM activation reads during the backward pass.**
