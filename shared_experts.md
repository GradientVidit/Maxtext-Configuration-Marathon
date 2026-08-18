
## 1. The DeepSeek shared expert idea

Standard top-k MoE: each token routes to k specialized experts out of N total. The remaining N-k experts see no contribution from that token.

DeepSeek introduced a modification: in addition to the k routed experts, some experts process **every single token** regardless of routing. These are "shared experts."

```text
Standard MoE:
  token → router → expert_3, expert_7  (only 2 of 256 experts)
  
DeepSeek shared experts:
  token → router → expert_3, expert_7  (routed to 2 of 256)
         └→ always → shared_expert_0   (always runs)
```

The shared experts provide a "common pathway" that every token passes through, complementing the specialized routing.

---

## 2. What it controls

```yaml
shared_experts: 0   # (default) no shared experts
shared_experts: 1   # one shared expert that processes every token
shared_experts: 2   # two shared experts
```

The shared experts are separate from `num_experts` — they're additional, always-active MLP units.

---

## 3. Why this design choice

- **Universal features:** some processing (e.g. syntactic normalization, common transformations) is useful for every token regardless of content — shared experts can learn these
- **Compute stability:** regardless of routing variance, every token has the shared expert's contribution — reduces dependency on routing quality
- **Quality at high sparsity:** when `num_experts_per_tok/num_experts` is very small (e.g. 8/256 = 3.1% in DeepSeek-V3), shared experts compensate for the sparse activation

---

## 4. Parameter and compute cost

Each shared expert adds:
```text
parameters: 3 × moe_expert_input_dim × base_moe_mlp_dim  (W_gate + W_up + W_down, same as a routed expert)
compute:    always runs, so its FLOPs are NOT sparse — adds to base compute per token
```

With `shared_experts=1` and `num_experts_per_tok=8`:
```text
effective compute per token = (8 + 1) × expert_mlp_cost
```

---

## 5. Default

```yaml
shared_experts: 0
```

No shared experts. Fully sparse routing. This is correct for standard Mixtral-style MoE.

---

## 6. Interaction with related parameters

| Related param | Interaction |
|---|---|
| `num_experts` | Shared experts are separate from the N routed experts |
| `num_experts_per_tok` | Total per-token expert activations = k routed + shared_experts |
| `load_balance_loss_weight` | Applies to the routed experts only; shared experts always active |
| `first_num_dense_layers` | Shared experts only apply in MoE layers |

---

### One-line intuition

> **`shared_experts` adds always-active expert MLPs that process every token regardless of routing — the DeepSeek design choice for compensating high routing sparsity with a stable "common pathway" through which all tokens pass.**
