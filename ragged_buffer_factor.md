
## 1. The ragged buffer sizing problem

After routing, MoE dispatch produces a ragged structure: each expert holds a variable number of token activations. Before running expert MLPs, these activations need to be held in a buffer.

The question is: **how big should that buffer be?**

```text
worst case: one expert gets ALL tokens → buffer = batch_size × seq_len × emb_dim
best case (perfectly balanced): buffer = batch_size × seq_len × emb_dim / num_experts per expert
```

Allocating worst-case buffers wastes memory when routing is actually balanced. Allocating best-case buffers drops tokens if routing is imbalanced. `ragged_buffer_factor` sets where this tradeoff sits.

---

## 2. The formula

```text
balanced_size = (batch × seq_len × num_experts_per_tok) / num_experts  per expert

ragged_buffer_size = balanced_size × ragged_buffer_factor   (when > 0)
```

Special values:

| `ragged_buffer_factor` | Meaning |
|---|---|
| `-1.0` (default) | Worst-case sizing — buffer accommodates all tokens regardless of routing |
| `1.0` | Balanced sizing — buffer assumes perfectly even routing; tokens dropped if any expert overflows |
| `1.5` | 50% slack above balanced — tolerates mild imbalance |
| `2.0` | 2× balanced — tolerates significant imbalance before dropping |

---

## 3. Default: `-1.0`

```yaml
ragged_buffer_factor: -1.0
```

No token dropping from buffer overflow. The ragged buffer is large enough to hold all routed tokens no matter how imbalanced routing gets. This is the safe, lossless default — but uses more memory.

---

## 4. Relationship to `capacity_factor`

These are parallel controls for two different MoE implementation paths:

```text
capacity_factor      →  controls token dropping in the PADDED/DENSE path
ragged_buffer_factor →  controls token dropping in the SPARSE/RAGGED path
```

In production MaxText (with `sparse_matmul=true, megablox=true`), you're on the sparse/ragged path — `ragged_buffer_factor` is the relevant lever. Both default to `-1.0` (no dropping).

---

## 5. Memory implications

For a model with:
- `batch_size × seq_len = 16,384` tokens
- `num_experts = 64`
- `emb_dim = 7168`
- `num_experts_per_tok = 2`

Balanced tokens per expert: `16,384 × 2 / 64 = 512`

At `ragged_buffer_factor=-1.0` (worst case), max buffer = `16,384 × 2` per expert = 32,768 tokens.  
At `ragged_buffer_factor=1.0`, buffer = 512 tokens per expert.

The difference can be significant when `num_experts` is small and routing variance is high.

---

## 6. Interaction with related parameters

| Related param | Interaction |
|---|---|
| `capacity_factor` | Parallel concept for dense path; don't confuse them |
| `use_ring_of_experts` | Ring-of-experts changes how tokens are distributed across devices; affects what "balanced" means |
| `load_balance_loss_weight` | Stronger load balancing → routing more balanced in practice → safer to tighten `ragged_buffer_factor` |
| `num_experts_per_tok` | Higher k → higher tokens-per-expert on average → affects what `1.0` maps to in practice |

---

## 7. Practical scenarios

**Training quality matters:** leave at `-1.0`. Memory overhead is the cost, zero token dropping is the benefit.

**Memory-constrained large MoE:** try `1.25` or `1.5` — gives buffer slack while cutting peak memory usage.

**With strong load balancing (`load_balance_loss_weight=0.01`):** can safely go to `1.0` since routing is forced toward balance.

---

### One-line intuition

> **`ragged_buffer_factor` sets the size of the buffer holding routed token activations relative to the perfectly-balanced expectation — `-1.0` means "size for worst case, never drop"; positive values trade token dropping risk for lower memory overhead.**
