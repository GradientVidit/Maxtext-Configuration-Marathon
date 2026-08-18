## 1. Why does it exist?

As transformer context lengths extend from 4k to 32k, 128k, or 1M+ tokens, the memory required to store intermediate attention activation maps scales quadratically $O(S^2)$ or linearly with flash attention $O(S \cdot d)$. Storing the activations of a 128k sequence easily overflows the HBM of a single accelerator chip.

Context Parallelism (CP) partitions the sequence dimension $S$ across devices within the high-speed Inter-Chip Interconnect (ICI).

```text
Sequence Length = 128k Tokens, ici_context_parallelism: 4
  Device 0: Tokens 0..32k     ──┐
  Device 1: Tokens 32k..64k   ──┼──[ Ring Attention / All-Gather over ICI ]
  Device 2: Tokens 64k..96k   ──┤
  Device 3: Tokens 96k..128k  ──┘
```

`ici_context_parallelism` sets the degree of context parallelism partitioning sequence lengths across chips within a single TPU slice.

---

## 2. Options & Configuration

| Value | Behavior |
|---|---|
| `1` (default) | Standard training (no sequence splitting on context axis). |
| Integer $> 1$ (e.g. `2`, `4`, `8`) | Divides input sequence length by `N` across chips in the slice. |

Default in `base.yml`:
```yaml
ici_context_parallelism: 1
```

---

## 3. Divisibility & Performance Rules

- **Divisibility**: The model's sequence length (`max_target_length`) must be divisible by `ici_context_parallelism`.
- **Strategy Support**: Supports Ring Attention (`"ring"`), All-Gather (`"all_gather"`), and Unified Sequence Parallelism (`"usp"`).
- **Communication Overlap**: On TPU ICI networks, KV token chunks circulate along optical rings while FlashAttention computation processes the previous chunk, completely hiding communication latency.

---

## 4. Interactions with Related Parameters

- **`context_parallel_strategy`**: Determines the underlying collective algorithm.
- **`context_parallel_load_balance`**: Rebalances causal mask computations across context chips.
- **`context_sharding`**: Selects the physical mesh axis (`"context"`).

---

### One-line intuition

> **`ici_context_parallelism` divides long sequence lengths across chips in a TPU slice over high-speed interconnects, allowing models to scale to million-token context windows.**
