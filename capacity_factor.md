
## 1. The token-dropping problem

Ideal MoE routing: each expert sees exactly `(batch_size × seq_len × num_experts_per_tok) / num_experts` tokens — perfectly balanced.

Reality: routing is learned and can be highly imbalanced:

```text
expert_0: 200 tokens  ← heavy (2× expected)
expert_1:  10 tokens  ← very light
expert_2: 150 tokens
expert_3:  40 tokens  ← expected: ~100
```

If you let every token through regardless of load, the per-expert compute becomes unpredictable and memory allocation is hard to bound. The solution: define an **expert capacity** — a maximum number of tokens each expert will process — and drop tokens that arrive beyond that limit.

`capacity_factor` sets where that capacity limit sits, relative to the "perfectly balanced" baseline.

---

## 2. How capacity is computed

```text
balanced_tokens_per_expert = (batch × seq_len × num_experts_per_tok) / num_experts

expert_capacity = balanced_tokens_per_expert × capacity_factor
```

So:

| `capacity_factor` | Meaning |
|---|---|
| `-1.0` (default) | No limit — every token is processed regardless of expert load |
| `1.0` | Capacity = exactly balanced expectation; any token routed to an overloaded expert is **dropped** |
| `1.25` | 25% slack above balanced — tolerates mild imbalance before dropping |
| `2.0` | 2× balanced — tolerates significant imbalance |

---

## 3. The default: `-1.0`

```yaml
capacity_factor: -1.0
```

This is a sentinel: **no capacity limit, no token dropping**. Each expert processes all tokens routed to it regardless of how many arrive. Memory must be sized for worst-case load. This is the safe, lossless default.

---

## 4. What happens when a token is dropped

When `capacity_factor > 0` and an expert's capacity is exceeded:

```text
tokens routed to expert_0: [t_1, t_2, ..., t_200]
expert_0 capacity: 125 tokens

→ tokens t_126 through t_200 are DROPPED
→ those tokens get the identity (pass-through) value or zero for that layer
```

Token dropping degrades model quality but can be acceptable:
- If `load_balance_loss_weight > 0`, routing imbalance is penalized, making overflow rare
- Small amounts of dropping (< 1% of tokens) often have negligible quality impact
- At scale with tight memory budgets, this tradeoff is often worth it

---

## 5. Memory and compute implications

```text
capacity_factor=-1.0  →  allocate for worst-case load per expert
                      →  no dropping, but more memory reserved
                      →  safe for training quality

capacity_factor=1.0   →  allocate for perfectly balanced load
                      →  some tokens dropped if routing imbalanced
                      →  less memory, lower worst-case compute
```

For a large MoE model (256 experts) running on memory-constrained hardware, tightening `capacity_factor` from `-1.0` to `1.25` can meaningfully reduce peak memory usage.

---

## 6. Interaction with related parameters

| Related param | Interaction |
|---|---|
| `load_balance_loss_weight` | Higher aux loss weight → more balanced routing → overflow with tight `capacity_factor` is rarer |
| `ragged_buffer_factor` | Controls buffer sizing for the ragged (sparse) path, similar concept but for activation buffers |
| `num_experts_per_tok` | Higher k → higher average tokens per expert → more potential overflow |
| `use_random_routing` | Random routing → worst-case imbalance → needs higher `capacity_factor` |

---

## 7. Practical scenarios

**Training quality matters most:** leave at `-1.0`. Never drop tokens.

**Memory-constrained training:** try `capacity_factor=1.25`. Loses a small fraction of tokens on heavily-loaded experts but stays predictable.

**With strong load balancing aux loss (`load_balance_loss_weight=0.01`):** you can safely tighten to `1.0` since routing will be forced toward balance.

**Inference with static compilation:** must set a finite `capacity_factor` to get static buffer shapes for XLA compilation.

---

### One-line intuition

> **`capacity_factor` controls the maximum tokens each expert will process relative to the perfectly-balanced expectation — `-1.0` means "never drop tokens, size for worst case"; any positive value enables token dropping when an expert is overloaded beyond that multiplier.**
