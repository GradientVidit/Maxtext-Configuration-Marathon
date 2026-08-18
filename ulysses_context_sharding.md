## 1. Why does it exist?

Unified Sequence Parallelism (USP) combines **Ring Attention** (which passes Key/Value blocks in a circular ring along sequence chunks) and **DeepSpeed-Ulysses** (which performs an All-to-All collective to transpose sequence shards into head shards across devices).

Because Ulysses performs an All-to-All communication specifically across attention heads, it requires a dedicated physical mesh sub-axis that is distinct from standard ring-based context parallelism.

```text
Unified Sequence Parallelism (USP):
                Input Sequence [Batch, Length, Dim]
                                 │
     ┌───────────────────────────┴───────────────────────────┐
     ↓                                                       ↓
Ring Dimension (`context`)                     Ulysses Dimension (`context_usp_ulysses`)
(Shards sequence across ring)                  (All-to-All head transpose across mesh)
```

`ulysses_context_sharding` specifies the physical mesh axis used for the Ulysses All-to-All head-exchange dimension when `context_parallel_strategy='usp'`.

---

## 2. Options & Configuration

| Parameter | Supported Option | Default |
|---|---|---|
| `ulysses_context_sharding` | `"context_usp_ulysses"` | `"context_usp_ulysses"` |

Default in `base.yml`:
```yaml
ulysses_context_sharding: "context_usp_ulysses"
```

---

## 3. Interactions with USP

```text
context_parallel_strategy: "usp"
  ├── context_sharding: "context"
  ├── ulysses_context_sharding: "context_usp_ulysses"
  ├── ici_context_parallelism: 2              (Ring degree = 2)
  └── ici_context_usp_ulysses_parallelism: 4  (Ulysses degree = 4)
  ──→ Total Sequence Parallelism = 2 x 4 = 8
```

- **`context_parallel_strategy`**: When set to `'usp'`, MaxText splits sequence parallel operations across both `context` (ring) and `context_usp_ulysses` (Ulysses).
- **`ici_context_usp_ulysses_parallelism`**: Configures the size of this physical mesh axis.

---

### One-line intuition

> **`ulysses_context_sharding` designates the physical mesh axis (`"context_usp_ulysses"`) that handles the all-to-all head-exchange collective in Unified Sequence Parallelism (USP).**
