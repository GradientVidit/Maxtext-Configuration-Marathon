## 1. Why does it exist?

Context Parallelism (CP) allows training on long sequences by distributing sequence tokens across multiple accelerator devices. However, there are multiple fundamentally different algorithms for exchanging Key and Value states during attention:

1. **`"all_gather"`**: Gathers all KV chunks across the context axis into local memory before computing attention. Simple and fast for moderate sequence lengths, but requires enough local memory to hold the gathered KV activations.
2. **`"ring"` (Ring Attention)**: Overlaps compute and communication by circulating KV blocks in a circular ring between adjacent devices while computing attention on the previous block. Bounded memory footprint, but requires $N$ ring steps.
3. **`"ulysses"` (DeepSpeed-Ulysses)**: Transposes sequence-parallel tokens into head-parallel heads via an **All-to-All collective**, computes standard local FlashAttention across all sequence positions for local heads, and transposes back via another All-to-All. Very fast, but limited by head divisibility.
4. **`"usp"` (Unified Sequence Parallelism)**: Combines Ring Attention and Ulysses in a 2D mesh, allowing sequence parallelism to scale beyond the number of attention heads.

```text
                               Context Parallel Strategy
                                           │
         ┌───────────────────┬─────────────┴─────────────┬───────────────────┐
         ↓                   ↓                           ↓                   ↓
   "all_gather"            "ring"                    "ulysses"             "usp"
(Gather all KV,      (Circular bucket passing,   (All-to-All head     (2D Grid: Ring +
 fast for short CP)   constant memory bound)      transpose)           Ulysses combined)
```

`context_parallel_strategy` selects which sequence-parallel collective algorithm MaxText compiles for context-parallel layers.

---

## 2. Options & Strategy Comparison

| Strategy | Communication Collective | Memory Footprint | Head Count Constraint |
|---|---|---|---|
| `"all_gather"` (default) | Single `AllGather` | $O(S \cdot d)$ (Holds gathered KV) | None |
| `"ring"` | Ring `P2P` / circular shifts | $O(\frac{S}{N} \cdot d)$ (Constant tile size) | None |
| `"ulysses"` | Two `AllToAll` transposes | $O(\frac{S}{N} \cdot d)$ | Heads must divide CP degree |
| `"usp"` | 2D Ring + `AllToAll` | $O(\frac{S}{N} \cdot d)$ | Heads divide Ulysses degree |

Default in `base.yml`:
```yaml
context_parallel_strategy: "all_gather" # "all_gather", "ring", "ulysses", or "usp"
```

---

## 3. Interactions with Related Parameters

- **`ici_context_parallelism`**: Sizing of the primary context axis.
- **`ici_context_usp_ulysses_parallelism`**: Configures the Ulysses axis when `context_parallel_strategy="usp"`.
- **`context_parallel_load_balance`**: Rebalances causal masks across all strategies.

---

### One-line intuition

> **`context_parallel_strategy` selects the underlying sequence parallelism algorithm (`"all_gather"`, `"ring"`, `"ulysses"`, or `"usp"`), trading communication patterns against memory and head divisibility constraints.**
