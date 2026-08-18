## 1. Why does it exist?

Unified Sequence Parallelism (USP) creates a 2D sequence-parallel mesh by combining:
1. **Ring Attention** along one dimension (sliding KV blocks in a ring).
2. **DeepSpeed-Ulysses** along a second dimension (performing an All-to-All collective across attention heads).

```text
Unified Sequence Parallelism (USP) 2D Grid:
                    Ulysses Axis (`ici_context_usp_ulysses_parallelism` = 4)
                   ┌──────────────┬──────────────┬──────────────┬──────────────┐
  Ring Axis        │   Chip 0     │   Chip 1     │   Chip 2     │   Chip 3     │
  (`context` = 2)  ├──────────────┼──────────────┼──────────────┼──────────────┤
                   │   Chip 4     │   Chip 5     │   Chip 6     │   Chip 7     │
                   └──────────────┴──────────────┴──────────────┴──────────────┘
```

`ici_context_usp_ulysses_parallelism` configures the size of the Ulysses All-to-All head-exchange dimension within a single TPU slice when `context_parallel_strategy='usp'`.

---

## 2. Options & Configuration

| Value | Meaning |
|---|---|
| `1` (default) | Pure Ring or All-Gather context parallelism (Ulysses axis disabled). |
| Integer $> 1$ (e.g. `2`, `4`, `8`) | Ulysses sequence-to-head All-to-All degree within the slice. |

Default in `base.yml`:
```yaml
ici_context_usp_ulysses_parallelism: 1 # Ulysses (head-exchange) dimension of the context parallelism; used by context_parallel_strategy='usp'.
```

---

## 3. Divisibility Constraints

When `ici_context_usp_ulysses_parallelism > 1`:
- `base_num_query_heads` and `base_num_kv_heads` must be divisible by `ici_context_usp_ulysses_parallelism`.

---

### One-line intuition

> **`ici_context_usp_ulysses_parallelism` sets the intra-slice all-to-all head-exchange degree for Unified Sequence Parallelism, enabling hybrid Ring-Ulysses sequence scaling.**
