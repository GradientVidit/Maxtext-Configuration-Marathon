## 1. Why does `gdn_value_head_dim` exist?

In linear attention models like Gated DeltaNet, value vectors represent the actual content being stored and retrieved from the recurrent state matrix $S_t$.

The recurrent state update in Gated DeltaNet is formulated as:

$$S_t = S_{t-1}(I - \beta_t k_t k_t^T) + \beta_t v_t k_t^T$$

where:
- $k_t \in \mathbb{R}^{d_k}$ (key vector)
- $v_t \in \mathbb{R}^{d_v}$ (value vector)
- $S_t \in \mathbb{R}^{d_k \times d_v}$ (recurrent memory matrix per head)

```text
Query Projection ──> q_t [d_k] ──┐
Key Projection   ──> k_t [d_k] ──┼──> Associative Read: y_t = S_{t-1} q_t  [d_v]
Value Projection ──> v_t [d_v] ──┘    Associative Write: S_t = S_{t-1}(I - β k k^T) + β v k^T
```

`gdn_value_head_dim` sets $d_v$, the channel capacity of the retrieved token representations, decoupling it from the query/key addressing dimension $d_k$.

---

## 2. Mechanics & Scaling

1. **State Matrix Geometry**: The size of each recurrent head's memory tensor is $d_k \times d_v$. If `gdn_key_head_dim = 128` and `gdn_value_head_dim = 128`, each head maintains a $128 \times 128 = 16,384$ parameter state.
2. **Output Projection**: Output retrieved from each head has dimension $d_v$. Concatenating across $H_v$ value heads yields an intermediate representation of shape `[Batch, Seq_Len, gdn_num_value_heads * gdn_value_head_dim]` before the final linear out-projection.
3. **Decoupled Asymmetry**: While standard self-attention typically forces $d_k = d_v$, linear recurrence can scale $d_v$ higher than $d_k$ to preserve rich feature expressions without inflating the key addressing rank.

---

## 3. Options & Default

| Parameter | Type | Default | Options |
| :--- | :--- | :--- | :--- |
| `gdn_value_head_dim` | `int` | `128` | Positive integers (e.g. `64`, `128`, `256`) |

---

## 4. Interactions with Related Parameters

| Related Parameter | Interaction |
| :--- | :--- |
| `gdn_key_head_dim` | Together define the recurrent memory tensor shape $[d_k, d_v]$ per head. |
| `gdn_num_value_heads` | Total value projection dimension entering the GDN block is `gdn_num_value_heads * gdn_value_head_dim`. |
| `gdn_chunk_size` | Chunked parallel scan performs GEMM operations of size `[gdn_chunk_size, gdn_value_head_dim]`. |

---

## 5. Practical Scenarios & Failure Modes

| Scenario | Recommendation / Symptom | Solution |
| :--- | :--- | :--- |
| **Standard Qwen3-Next Training** | `gdn_value_head_dim: 128` | Retains full numerical alignment with TPU v4/v5/v6 MXU tiling dimensions ($128 \times 128$). |
| **Mismatched Head Projections** | Output projection layer dim does not match `base_emb_dim` | Ensure the final linear projection maps `gdn_num_value_heads * gdn_value_head_dim` back to `base_emb_dim`. |
| **Over-allocating $d_v$** | Setting $d_v=256$ doubles recurrent state memory per head | Monitor device HBM during training chunk scan and inference generation. |

---

### One-line intuition

> `gdn_value_head_dim` sets the feature dimension of value vectors in Gated DeltaNet, determining the column width of the recurrent memory matrix and output channel size per head.
