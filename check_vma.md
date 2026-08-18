## 1. Why does it exist?

During large-scale distributed training with Expert Parallelism (EP) or Fully Sharded Data Parallelism (FSDP), token routing and expert matrix multiplications execute inside `jax.experimental.shard_map` blocks (e.g. `sparse_matmul_route_and_compute` in `src/maxtext/layers/moe.py`).

By default, XLA performs conservative safety checks on Virtual Memory Addresses (VMA) across sharded devices to prevent buffer aliasing. Enabling **VMA checking (`check_vma: true`)** informs the JAX `shard_map` lowering pass that memory allocations satisfy strict contiguous address constraints across parallel devices, unlocking compiler memory layout optimizations and buffer reuse that improve kernel throughput.

```text
Standard MoE ShardMap (check_vma: false):
  `shard_map` generates conservative device memory barriers.

Optimized MoE ShardMap (check_vma: true):
  `shard_map(..., check_vma=True)` in `sparse_matmul_route_and_compute`
  ──→ Unlocks zero-copy buffer reuse & higher Model FLOPs Utilization (MFU).
```

`check_vma` enables strict VMA address checking in `shard_map` for improved execution performance in supported EP/FSDP configurations.

---

## 2. Strict Compatibility Constraints

MaxText strictly asserts that `check_vma: true` is only valid under a specific, well-defined combination of configuration settings:

```text
check_vma: true is ONLY supported when:
  ├── Parallelisms: EP or FSDP active along ICI (ici_expert_parallelism or ici_fsdp_parallelism)
  ├── shard_mode == "auto"
  ├── use_ragged_sort == false
  ├── use_ring_of_experts == false
  └── use_tokamax_gmm == false
```

If any of these constraints are violated, enabling `check_vma` will fail runtime assertion checks.

---

## 3. Options & Configuration

| Value | Meaning |
|---|---|
| `false` (default) | VMA checking disabled; standard memory allocation heuristics used. |
| `true` | Enables VMA validation inside `shard_map` for improved performance on compatible EP/FSDP workloads. |

Default in `base.yml`:
```yaml
check_vma: False
```

---

## 4. Practical Usage Guidelines

- **When to Enable**: Recommended for maximizing throughput on standard Megablox-based MoE or pure FSDP training runs where experimental ragged kernels are turned off.
- **When to Leave False**: Keep `false` if using Tokamax GMM (`use_tokamax_gmm: true`), Ring of Experts (`use_ring_of_experts: true`), or Pallas ragged sort kernels (`use_ragged_sort: true`).

---

### One-line intuition

> **`check_vma` enables virtual memory address validation inside MoE `shard_map` blocks, unlocking compiler buffer reuse optimizations for higher accelerator throughput.**
