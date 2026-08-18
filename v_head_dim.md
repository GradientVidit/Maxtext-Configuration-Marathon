## 1. Why does `v_head_dim` exist?

In standard Multi-Head Attention (MHA), Query, Key, and Value projections all share a single uniform head dimension:

$$\text{Standard MHA:}\quad d_q = d_k = d_v = \text{head\_dim}$$

In Multi-Head Latent Attention (MLA), because Queries, Keys, and Values are decoupled and reconstructed from low-rank latent bottlenecks ($c_q$ and $c_{kv}$), the **Value head dimension ($d_v$) is completely independent of the Query/Key dimensions**:

$$\text{Query/Key Head Dimension} = d_{\text{nope}} + d_{\text{rope}} = 128 + 64 = 192$$

$$\text{Value Head Dimension} = d_v = \text{v\_head\_dim} = 128$$

```text
MLA Projection Dimensions:
  Q Head: [ 128 (NoPE) + 64 (RoPE) ] = 192 dims  <── Addressing & Alignment
  K Head: [ 128 (NoPE) + 64 (RoPE) ] = 192 dims  <── Addressing & Alignment
  V Head: [ 128 dims ]                           <── Content Retrieval & Aggregation
```

`v_head_dim` specifies the feature dimensionality $d_v$ of each attention head's Value representation.

---

## 2. Options & Defaults

| Value | Behavior | Notes |
|---|---|---|
| `128` | Value head dimension of $d_v = 128$. | **Default**. Exact DeepSeek-V2 / DeepSeek-V3 specification. |
| Any integer $> 0$ | Custom Value head dimension. | Total output projection input width is $N_{\text{heads}} 	imes \text{v\_head\_dim}$. |

Default in `base.yml`: `128`

---

## 3. Impact on Output Projection ($W_o$)

The output projection matrix $W_o$ combines the weighted Value vectors across all $N_h$ attention heads back into the residual stream $d_{model}$:

$$\text{Attention Out} = \text{Concat}(head_0, head_1, \dots, head_{N_h-1}) \cdot W_o$$

$$W_o \in \mathbb{R}^{(N_h \cdot d_v) 	imes d_{model}}$$

For DeepSeek-V2 ($N_h = 128$, $d_v = 128$, $d_{model} = 5120$):
$$(128 	imes 128) 	imes 5120 = 16{,}384 	imes 5120 = 83{,}886{,}080 \text{ parameters per layer}$$

---

## 4. Interactions with other parameters

| Related Parameter | Interaction |
|---|---|
| [[kv_lora_rank]] | Value vectors are reconstructed from the $c_{kv}$ latent via up-projection $W_{UV} \in \mathbb{R}^{d_c 	imes (N_h \cdot d_v)}$. |
| [[qk_nope_head_dim]], [[qk_rope_head_dim]] | Query/Key dimensions, decoupled from $d_v$. |
| [[attention_type]] | Active when `attention_type: 'mla'`. |

---

## 5. Practical Scenarios

- **DeepSeek-V2 / V3 Pretraining:** Keep `v_head_dim: 128`.
- **Ablation Studies:** Adjusting $d_v$ independently of $d_{qk}$ allows studying the impact of value retrieval capacity without altering attention score resolution.

---

### One-line intuition

> **`v_head_dim` defines the feature width ($d_v=128$) of Value vectors in MLA, decoupled from Query/Key dimensions to optimize content retrieval independently of addressing.**
