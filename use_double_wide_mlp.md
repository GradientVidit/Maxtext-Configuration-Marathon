## 1. Why does `use_double_wide_mlp` exist?

When trailing decoder layers share their Key and Value projections (via `num_kv_shared_layers > 0`), those layers eliminate their $W_k$ and $W_v$ weight tensors, saving parameter capacity.

To prevent the total expressive parameter budget of the model from dropping, **Double-Wide MLP** reallocates the saved parameter budget by doubling the intermediate dimension of the Feed-Forward Network (MLP) **specifically on the KV-shared layers**:

```text
Standard Layer (Unshared KV):
  ├── Attention Block: Own Q, K, V Projections (d_model → d_k, d_v)
  └── MLP Block:       Standard Width = base_mlp_dim (e.g. 8192)

KV-Shared Layer (use_double_wide_mlp=True):
  ├── Attention Block: NO K, V Projections (Reuses previous layer's K, V)
  └── MLP Block:       DOUBLE Width = 2 * base_mlp_dim (e.g. 16384)
```

$$\text{Standard MLP Intermediate Width} = d_{mlp}$$

$$\text{KV-Shared Layer Intermediate Width} = 2 	imes d_{mlp}$$

---

## 2. Options & Defaults

| Value | Behavior | Notes |
|---|---|---|
| `false` | All layers (both unshared and KV-shared) use the same standard `base_mlp_dim`. | **Default**. |
| `true` | Layers identified as KV-shared automatically instantiate an MLP with $2 	imes \text{base\_mlp\_dim}$. | Used in Gemma 4 small to achieve parameter neutrality. |

Default in `base.yml`: `false`

---

## 3. Parameter Neutrality Math

In SwiGLU architectures, the MLP consists of three matrices (Gate, Up, Down), totaling $3 	imes d_{model} 	imes d_{mlp}$ parameters.

By skipping KV projections (saving $pprox 2 	imes d_{model} 	imes d_{kv}$) and expanding the MLP by $+d_{mlp}$, the total parameter count per layer remains roughly constant while shifting the compute profile toward dense feed-forward processing.

---

## 4. Interactions with other parameters

| Related Parameter | Interaction |
|---|---|
| [[num_kv_shared_layers]] | Prerequisite. `use_double_wide_mlp` only affects layers where KV projections are shared. Has no effect if `num_kv_shared_layers: 0`. |
| [[base_mlp_dim]] | The baseline dimension that gets multiplied by $2	imes$ on shared layers. |
| [[scan_layers]] | Requires `scan_layers: false` because different layers have different MLP weight matrix shapes ($d_{mlp}$ vs $2 d_{mlp}$). |

---

## 5. Practical Scenarios

- **Training Gemma 4 Small (E2B / E4B):** Set `use_double_wide_mlp: true` alongside `num_kv_shared_layers` to match published Google DeepMind architecture specifications.
- **Standard Pretraining (LLaMA / Gemma 2):** Leave `false`.

---

### One-line intuition

> **`use_double_wide_mlp=true` doubles the MLP intermediate dimension ($2 	imes \text{base\_mlp\_dim}$) on KV-shared layers, trading KV projection weights for deeper feed-forward capacity.**
