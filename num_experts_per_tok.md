
## 1. Why does `num_experts_per_tok` exist?

Having `N` experts is only half the story. You also need to decide: **how many of those N experts does each token actually use?**

This is the top-k routing decision. It sits at the core of the MoE compute vs. quality tradeoff:

```text
num_experts=8, num_experts_per_tok=1   →  each token uses 1 expert (cheapest, most sparse)
num_experts=8, num_experts_per_tok=2   →  each token uses 2 experts (Mixtral default)
num_experts=8, num_experts_per_tok=8   →  each token uses all experts (= dense MLP, no sparsity)
```

`num_experts_per_tok` controls **k** in top-k routing.

---

## 2. The routing mechanism

```text
input token embedding (shape: [batch, seq, emb_dim])
         ↓
router linear layer → logits (shape: [batch, seq, num_experts])
         ↓
softmax or sigmoid over experts
         ↓
top-k selection → keep the k highest scores
         ↓
send token to k expert MLPs, weighted by their routing scores
         ↓
weighted sum of k expert outputs → final representation
```

`num_experts_per_tok` = k in that top-k step.

---

## 3. The compute cost formula

Active FLOP cost per MoE layer per token:

```text
FLOPs ∝ num_experts_per_tok × (3 × moe_expert_input_dim × base_moe_mlp_dim)
```

SwiGLU has 3 matrices (W_gate + W_up + W_down); doubling `num_experts_per_tok` doubles per-token compute in MoE layers but doesn't change parameter count.

---

## 4. Options

| Value | Effect |
|---|---|
| `1` (default) | Each token routed to exactly 1 expert — maximum sparsity |
| `2` | Classic Mixtral-style: 2-of-8 routing |
| `k` (general) | Each token activates k experts — must be ≤ `num_experts` |

If `num_experts_per_tok == num_experts`: every expert sees every token — equivalent to running all expert MLPs densely. MoE advantage disappears.

---

## 5. Quality vs. compute tradeoff

```text
higher k → more expert outputs blended per token
         → richer representation
         → more compute
         → routing more "conservative" (less selective)

lower k  → each token strongly specializes to fewer experts
         → cheaper
         → routing more decisive
         → risk: expert collapse if routing not balanced
```

Most practical MoE models use k=1 or k=2 relative to `num_experts=8–64`. Very large-expert models (e.g. DeepSeek-V3 with 256 experts) use higher k (8) to maintain activation coverage.

---

## 6. Interaction with related parameters

| Related param | Interaction |
|---|---|
| `num_experts` | k must be ≤ N; the "activation ratio" is k/N |
| `capacity_factor` | token dropping threshold is based on expected tokens-per-expert, which depends on k |
| `load_balance_loss_weight` | aux loss pushes toward even distribution across all experts, which matters more when k is small |
| `ragged_buffer_factor` | buffer size depends on expected tokens per expert; k affects the distribution |
| `norm_topk_prob` | when true, normalizes the k routing weights to sum to 1 (Qwen3-style) |
| `routed_scaling_factor` | scales the routing scores before weighting expert outputs |
| `n_routing_groups` / `topk_routing_group` | DeepSeek grouped routing — applies top-k within groups, changes effective k semantics |

---

## 7. Practical scenarios

**Reproducing Mixtral:** `num_experts=8`, `num_experts_per_tok=2` — 25% of experts active per token.

**DeepSeek-V3:** `num_experts=256`, `num_experts_per_tok=8` — 3.1% active, very sparse.

**Debugging routing (not quality):** `num_experts_per_tok=num_experts` — turns MoE into effectively dense; useful to isolate whether routing is causing issues.

**Memory-constrained with many experts:** keep k small — more experts means more parameters, but k controls how much compute you spend per token.

---

### One-line intuition

> **`num_experts_per_tok` is the k in top-k routing — it controls how many experts each token actually uses, directly setting the per-token compute cost while leaving total parameter count unchanged.**
