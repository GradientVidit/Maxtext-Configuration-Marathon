
## 1. The ragged sorting problem in MoE

After routing, tokens need to be sorted/permuted so that tokens assigned to the same expert are contiguous before the expert GEMM:

```text
tokens: [t_0→expert_3, t_1→expert_1, t_2→expert_3, t_3→expert_0, ...]
         ↓ permute
grouped: [t_3, ..., t_1, ..., t_0, t_2, ...]
          (expert_0 first, then expert_1, then expert_3...)
```

This permutation is a **ragged sort** — tokens are sorted into variable-length groups (one per expert). The straightforward implementation uses a general sort, but for MoE the structure is known: you're partitioning into `num_experts` groups.

`use_ragged_sort` enables Pallas-based custom kernels specifically optimized for this ragged-sort structure.

---

## 2. What it does

```yaml
use_ragged_sort: false  # (default) use standard JAX sort for token permutation
use_ragged_sort: true   # use Pallas ragged-sort kernels in permute/unpermute
```

When `true`, the token permutation (and its inverse — unpermute) in the sparse MoE path uses hand-written Pallas kernels optimized for the ragged grouping pattern.

---

## 3. Where it runs

The ragged-sort kernels run inside:
- `permute` / `unpermute` — when `use_ring_of_experts=True`
- `local_permute` / local-unpermute — when `use_ring_of_experts=False` (but with EP > 1)

It's valid in both cases — EP > 1 is the only actual precondition.

---

## 4. This is a hardware-specific performance knob

Pallas kernels are compiled for specific accelerator targets. The benefit of `use_ragged_sort=True` depends heavily on:
- Accelerator type (TPU model, generation)
- Model shape (num_experts, seq_len, emb_dim)
- Memory access patterns at your specific batch size

This is not a "set once" default — it's something to benchmark.

---

## 5. Default

```yaml
use_ragged_sort: false
```

Standard JAX sort is the safe default. The Pallas ragged-sort path is an optimization that needs explicit validation on your hardware.

---

## 6. Options

| Value | Behavior |
|---|---|
| `false` (default) | Standard JAX sort for token permutation |
| `true` | Pallas ragged-sort kernels — potentially faster on TPU |

---

### One-line intuition

> **`use_ragged_sort` enables Pallas-optimized kernels for MoE token permutation/unpermutation — a hardware-specific performance knob that replaces general JAX sort with a kernel tuned for the ragged expert-grouping pattern; benchmark before enabling.**
