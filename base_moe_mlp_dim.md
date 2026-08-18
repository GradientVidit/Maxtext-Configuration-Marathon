
## 1. The expert MLP hidden dimension

Each expert MLP in an MoE layer is a standard (typically SwiGLU-style) feed-forward network:

```text
input (moe_expert_input_dim)
    ↓
W_gate, W_up  → [moe_expert_input_dim, base_moe_mlp_dim]
    ↓  SiLU gate + multiply
W_down        → [base_moe_mlp_dim, moe_expert_input_dim]
    ↓
output (moe_expert_input_dim)
```

`base_moe_mlp_dim` is the **intermediate (hidden) dimension** inside each expert — the "expansion" dimension.

---

## 2. The relationship to `base_mlp_dim`

In a fully-MoE model (where every FFN layer is MoE), the expert hidden dim should match the dense FFN hidden dim:

```text
fully-MoE model:  base_moe_mlp_dim == base_mlp_dim

hybrid model (some dense, some MoE layers):
    base_mlp_dim      → dimension for dense FFN layers
    base_moe_mlp_dim  → dimension for MoE expert MLPs (can differ)
```

MaxText documents this explicitly: for a fully-MoE model, `base_mlp_dim` must equal `base_moe_mlp_dim`.

---

## 3. The default: `-1` (auto-infer)

```yaml
base_moe_mlp_dim: -1
```

MaxText infers this — typically defaults to `base_mlp_dim`. Override only when the MoE expert hidden dimension needs to differ from the dense FFN dimension.

---

## 4. Parameter count impact

With `num_experts=N`, each MoE layer's expert weights are:

```text
N × (2 × moe_expert_input_dim × base_moe_mlp_dim)   ← W_up and W_gate
N × (1 × base_moe_mlp_dim × moe_expert_input_dim)   ← W_down
```

For DeepSeek-V3 style models, `base_moe_mlp_dim` is often set to a smaller value than `base_mlp_dim` to keep total parameter count manageable despite having 256 experts.

---

## 5. Options

| Value | Meaning |
|---|---|
| `-1` (default) | Auto-infer — typically set to `base_mlp_dim` |
| `> 0` | Explicit MoE expert hidden dimension |

---

## 6. Interaction with related parameters

| Related param | Interaction |
|---|---|
| `base_mlp_dim` | Must equal `base_moe_mlp_dim` for fully-MoE models |
| `moe_expert_input_dim` | The input/output dim of each expert (other axis of the weight matrix) |
| `num_experts` | All N experts share the same `base_moe_mlp_dim` |
| `wi_tile_fwd_mlp_dim` | GMM tiling along this dim for the up-projection |
| `wo_tile_fwd_mlp_dim` | GMM tiling along this dim for the down-projection |

---

### One-line intuition

> **`base_moe_mlp_dim` sets the hidden (expansion) dimension inside each expert's MLP — leave at `-1` to inherit from `base_mlp_dim`, but set explicitly in hybrid models or when experts should have a smaller hidden dim than dense FFN layers to manage parameter count.**
