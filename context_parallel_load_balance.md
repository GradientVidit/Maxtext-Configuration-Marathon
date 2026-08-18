## 1. Why does it exist?

In standard autoregressive language modeling, attention uses a **lower-triangular causal mask**: tokens can only attend to previous tokens.

When an input sequence of length $S$ is divided into $N$ contiguous chunks across $N$ context-parallel devices:
- **Device 0** (tokens $0 \dots \frac{S}{N}$): Attends only to itself $\implies$ computes only $1$ block of attention.
- **Device $N-1$** (tokens $\frac{(N-1)S}{N} \dots S$): Attends to all previous $N-1$ blocks plus itself $\implies$ computes $N$ blocks of attention!

```text
Without Load Balancing (Severe Imbalance):
  Device 0: [■                  ]  (1 Block  ──→ 90% Idle Time!)
  Device 1: [■■                 ]  (2 Blocks ──→ 80% Idle Time)
  Device 2: [■■■                ]  (3 Blocks ──→ 70% Idle Time)
  Device 3: [■■■■■■■■■■■■■■■■■■■]  (N Blocks ──→ Bottlenecks whole cluster)

With Load Balancing (context_parallel_load_balance: true):
  By pairing early chunks with late chunks (e.g. Chunk 0 + Chunk N-1):
  Device 0: [■■■■■■■■■■]  (Equal Workload!)
  Device 1: [■■■■■■■■■■]  (Equal Workload!)
  Device 2: [■■■■■■■■■■]  (Equal Workload!)
  Device 3: [■■■■■■■■■■]  (Equal Workload!)
```

`context_parallel_load_balance` eliminates this causal attention bottleneck by redistributing and pairing sequence chunks so that all context-parallel devices execute an equal amount of attention compute.

---

## 2. Fundamentals & Speedup

Without load balancing, total cluster execution time is bounded by the slowest device (Device $N-1$), which wastes $\approx 50\%$ of total accelerator FLOP capacity.

Enabling `context_parallel_load_balance: true` pairs complementary sequence slices, achieving near **2x speedup** in context-parallel attention computation.

---

## 3. Options & Configuration

| Value | Behavior |
|---|---|
| `true` (default) | Rebalances causal attention workload across context devices (Recommended). |
| `false` | Unbalanced contiguous allocation; later devices process disproportionately more tokens. |

Default in `base.yml`:
```yaml
context_parallel_load_balance: true
```

---

## 4. Interactions with Related Parameters

- **`context_parallel_reorder_strategy`**: Determines the concrete algorithm used to pair chunks (e.g. `"dual_chunk_swap"` or `"striped"`).
- **`context_parallel_strategy`**: Works with all CP strategies (`"all_gather"`, `"ring"`, `"ulysses"`, `"usp"`).

---

### One-line intuition

> **`context_parallel_load_balance` pairs early and late sequence chunks across devices, eliminating the 50% idle-time bubble caused by lower-triangular causal attention masks.**
