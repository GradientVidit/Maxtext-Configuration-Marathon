
## 1. Why does `num_experts` exist?

A standard transformer MLP maps every token through the **same** weight matrix every layer:

```text
token → [shared W_up, W_gate, W_down] → output
```

A Mixture-of-Experts layer replaces that single MLP with `N` independent expert MLPs:

```text
token → router → expert_0  ┐
                  expert_1  │ only top-k of these
                  ...       │ actually execute
                  expert_N  ┘
```

The key insight: **total parameter count scales with N, but per-token FLOP cost scales only with top-k** (the number of experts each token actually uses). This is how MoE achieves "more parameters → more knowledge capacity" without proportionally more compute per token.

`num_experts` sets how many expert MLPs exist in each MoE layer.

---

## 2. The dense-to-MoE spectrum

```text
num_experts=1      →  plain dense MLP (MoE disabled)
num_experts=8      →  small MoE (e.g. Mixtral-style)
num_experts=64     →  medium MoE
num_experts=256    →  large MoE (e.g. DeepSeek-V3 style: 256+1 shared)
```

At `num_experts=1`, every other MoE parameter (`num_experts_per_tok`, `megablox`, routing weights, etc.) is structurally a no-op — MaxText is just running a dense model.

---

## 3. Where does it slot into the architecture?

Each MoE decoder layer contains:

```text
input
  ↓
[attention block]
  ↓
router (linear projection → softmax → top-k)
  ↓
  ├── expert_0: W_up, W_gate, W_down (shape: [moe_expert_input_dim, base_moe_mlp_dim])
  ├── expert_1: same shape
  ...
  └── expert_{N-1}: same shape
  ↓
weighted sum of top-k expert outputs
  ↓
output
```

`num_experts` determines how many of those expert weight matrices exist. They are independent — no weight sharing between them.

---

## 4. Options

| Value | Effect |
|---|---|
| `1` (default) | Dense model — MoE disabled entirely |
| `> 1` | Enables MoE with that many experts per MoE layer |

Integer. There's no hard upper bound in config, but in practice it must be compatible with:
- The expert parallelism mesh axis (if using EP)
- `shard_exp_on_fsdp` constraints (must be multiple of FSDP degree if enabled)
- Memory: expert weight matrices scale linearly with `num_experts`

---

## 5. Interaction with related parameters

| Related param | Interaction |
|---|---|
| `num_experts_per_tok` | top-k: how many of the `num_experts` each token actually routes to |
| `capacity_factor` | gates how many tokens each expert can absorb (token dropping) |
| `load_balance_loss_weight` | aux loss to push routing toward using all `num_experts` evenly |
| `base_moe_mlp_dim` | hidden dim of each expert's MLP |
| `moe_expert_input_dim` | input dim to expert blocks |
| `shard_exp_on_fsdp` | requires `num_experts` divisible by FSDP degree |
| `n_routing_groups` | DeepSeek grouped routing — divides the `num_experts` into groups |
| `shared_experts` | additional experts that see every token (on top of `num_experts` routed ones) |
| `first_num_dense_layers` | first N layers skip MoE, the rest use `num_experts` |

---

## 6. Parameter count implications

For a model with `num_experts=N`, each MoE layer adds:

```text
N × (3 × moe_expert_input_dim × base_moe_mlp_dim)  ← for SwiGLU experts (W_gate + W_up + W_down)
```

SwiGLU has 3 matrices per expert, not 2. This is why MoE models can have enormous parameter counts (e.g. 671B for DeepSeek-V3) while active parameters per token stay much smaller.

---

## 7. Practical scenarios

**You're running a dense baseline:** leave at `1`. All MoE machinery is compiled out.

**Adding MoE to experiment:** set `num_experts=8`, `num_experts_per_tok=2` — classic Mixtral ratio (1/4 experts active per token).

**Reproducing DeepSeek-V3 style:** `num_experts=256`, `num_experts_per_tok=8`, `shared_experts=1`, `n_routing_groups=4`, `topk_routing_group=4`, `first_num_dense_layers=3`.

**Memory constrained:** watch that each additional expert adds its full set of weight matrices. With `num_experts=64` and large `base_moe_mlp_dim`, expert weights can dominate memory.

---

### One-line intuition

> **`num_experts` sets how many independent MLP experts exist in each MoE layer — `1` means dense model, anything higher enables MoE where total parameters scale with N but per-token compute scales only with `num_experts_per_tok`.**
