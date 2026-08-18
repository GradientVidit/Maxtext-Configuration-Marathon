
## 1. Why DeepSeek keeps initial layers dense

MoE layers are more complex than dense layers — they involve routing, dispatch, expert compute, and combine operations. The routing mechanism is also learned, which means early in training (and early in the model's depth), the routing is essentially random. Dense layers at the beginning of the model:

1. **Process embeddings more stably** — dense FFNs transform raw embeddings uniformly, which is a good inductive bias for early-layer processing
2. **Avoid routing noise early in depth** — the first few layers do fundamental token processing (understanding syntax, basic semantics); routing to specialists isn't helpful yet
3. **Match the published architecture** — DeepSeek models (V2, V3, etc.) define a specific number of initial dense layers as part of their architecture design

`first_num_dense_layers` specifies how many of the initial decoder layers use standard dense MLP instead of MoE.

---

## 2. What it does

```yaml
first_num_dense_layers: 0  # (default) all layers are MoE (if num_experts > 1)
first_num_dense_layers: 3  # first 3 layers are dense, rest are MoE
```

For a model with 32 decoder layers and `first_num_dense_layers=3`:
```text
Layer 0: dense MLP
Layer 1: dense MLP
Layer 2: dense MLP
Layer 3: MoE (num_experts experts)
...
Layer 31: MoE
```

---

## 3. Parameter count note

Dense layers use `base_mlp_dim` for their hidden dimension. MoE layers use `base_moe_mlp_dim` per expert. Dense layers don't multiply parameters by `num_experts`, so `first_num_dense_layers` dense layers have significantly fewer parameters than MoE layers.

---

## 4. Default

```yaml
first_num_dense_layers: 0
```

All layers are MoE when `num_experts > 1`. This is correct for fully-MoE architectures.

---

## 5. Interaction with related parameters

| Related param | Interaction |
|---|---|
| `num_experts` | `first_num_dense_layers` only matters when `num_experts > 1` |
| `base_mlp_dim` | Used by the dense layers |
| `base_moe_mlp_dim` | Used by the MoE layers |
| `shared_experts` | Shared experts are on MoE layers only |

---

## 6. Practical use

**Reproducing DeepSeek-V3:** `first_num_dense_layers=3` — DeepSeek-V3 has 3 initial dense layers (layers 1–3), followed by 58 MoE layers (4–61). DeepSeek-V2 also uses 1 dense layer. Match the model's config exactly.

**Hybrid architecture experiments:** use this to explore how many dense vs. MoE layers affects quality and compute.

**Dense baseline with gradual MoE:** start with `first_num_dense_layers=N`, gradually reduce N to transition more layers to MoE.

---

### One-line intuition

> **`first_num_dense_layers` keeps the first N decoder layers as plain dense MLP instead of MoE — matching DeepSeek's architecture design where early layers process raw embeddings without the complexity of learned routing.**
