
## 1. Why this exists

MoE expert blocks receive tokens after routing. The token representations enter expert MLPs with a specific input dimension — which MaxText calls `moe_expert_input_dim`.

In many configurations, this is simply the model's embedding dimension (`base_emb_dim`). But in architectures with separate MoE pathways, attention output projections, or decoupled MoE feed-forward inputs, the dimension tokens enter expert blocks with can differ from the base embedding dimension.

`moe_expert_input_dim` makes this explicit and separately configurable.

---

## 2. What it controls

The `W_up` weight matrix of each expert MLP has shape:

```text
[moe_expert_input_dim, base_moe_mlp_dim]
```

And the `W_down` weight has shape:

```text
[base_moe_mlp_dim, moe_expert_input_dim]
```

Setting `moe_expert_input_dim` determines the first dimension of all expert weight matrices.

---

## 3. The default: `-1` (auto-infer)

```yaml
moe_expert_input_dim: -1
```

MaxText infers this from context — typically set to `base_emb_dim`. You only need to override when the architecture explicitly decouples the expert input dimension from the embedding dimension.

---

## 4. Options

| Value | Meaning |
|---|---|
| `-1` (default) | Auto-infer (typically = `base_emb_dim`) |
| `> 0` | Explicit expert input dimension |

---

## 5. When to set it explicitly

- Architecture has a non-standard MoE input projection that changes the feature dimension before expert blocks
- You're implementing a custom model where expert input and embedding dims intentionally differ
- Debugging: set explicitly to confirm what you think is being used is actually being used

In standard Mixtral/Switch/DeepSeek configurations, leave at `-1`.

---

## 6. Interaction with related parameters

| Related param | Interaction |
|---|---|
| `base_emb_dim` | Default inferred value — typically `moe_expert_input_dim = base_emb_dim` |
| `base_moe_mlp_dim` | The expert hidden dimension — other axis of the weight matrix |
| `num_experts` | All `num_experts` expert weight matrices share this input dim |

---

### One-line intuition

> **`moe_expert_input_dim` sets the feature dimension of tokens entering expert blocks — leave at `-1` to auto-infer from `base_emb_dim`; only set explicitly for non-standard architectures where expert input dimension differs from embedding dimension.**
