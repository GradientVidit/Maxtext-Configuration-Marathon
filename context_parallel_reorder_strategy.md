## 1. Why does it exist?

When `context_parallel_load_balance: true` is enabled, the input sequence chunks must be reordered or paired so that devices assigned early token positions receive an equal share of the lower-triangular causal attention workload compared to devices assigned later positions.

Different reordering patterns provide different trade-offs regarding memory layout continuity and kernel communication efficiency:

1. **`"dual_chunk_swap"`**: Divides each device's sequence slice into two sub-chunks and pairs chunk $i$ with chunk $2N - 1 - i$ from the opposite end of the sequence.
2. **`"striped"` (Zig-Zag / Striped Attention)**: Interleaves token chunks across devices in a round-robin cyclic stripe (Device 0 gets chunks $0, 2N-1, \dots$; Device 1 gets chunks $1, 2N-2, \dots$).
3. **`"auto"`**: Lets MaxText automatically pick the optimal chunking pattern based on hardware generation and the selected `context_parallel_strategy`.

```text
Sequence Partitioning (8 chunks on 4 devices):
  Chunks: [0, 1, 2, 3, 4, 5, 6, 7]

Dual Chunk Swap / Striped Allocation:
  Device 0 ──→ Chunks (0, 7)  ──→ 1 + 8 = 9 work units
  Device 1 ──→ Chunks (1, 6)  ──→ 2 + 7 = 9 work units
  Device 2 ──→ Chunks (2, 5)  ──→ 3 + 6 = 9 work units
  Device 3 ──→ Chunks (3, 4)  ──→ 4 + 5 = 9 work units
  ──→ Perfectly balanced causal workload across all 4 devices!
```

`context_parallel_reorder_strategy` selects how sequence tokens and chunks are reordered for causal load balancing in Context Parallelism.

---

## 2. Options & Configuration

| Option | Behavior |
|---|---|
| `"auto"` (default) | Automatically chooses the optimal reordering strategy based on context parallel configuration. |
| `"dual_chunk_swap"` | Explicit dual-chunk swapping (pairs symmetric chunks $i$ and $2N - 1 - i$). |
| `"striped"` | Cyclic round-robin striping of token blocks. |

Default in `base.yml`:
```yaml
context_parallel_reorder_strategy: "auto" # "auto", "dual_chunk_swap", or "striped"
```

---

## 3. Interactions with Related Parameters

- **`context_parallel_load_balance`**: Must be `true` for `context_parallel_reorder_strategy` to take effect.
- **`context_parallel_strategy`**: Works seamlessly with `"all_gather"`, `"ring"`, and `"usp"`.

---

### One-line intuition

> **`context_parallel_reorder_strategy` selects the chunk-pairing algorithm (`"auto"`, `"dual_chunk_swap"`, or `"striped"`) to balance causal attention workloads across context-parallel devices.**
