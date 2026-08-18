## 1. Why does `gdn_num_value_heads` exist?

In Gated DeltaNet (GDN), the number of value heads determines the total number of parallel recurrent memory state matrices maintained by the layer.

```text
Layer Memory Storage:
Total Memory Matrices = gdn_num_value_heads (H_v)
Each Matrix S^{(i)} has shape [d_k, d_v]

Total Recurrent State Size per Layer = H_v × d_k × d_v  scalars
```

In standard softmax attention, increasing value heads increases the KV cache storage linearly with sequence length $S$. In GDN, the state size is **constant with respect to sequence length $O(1)$**, so scaling $H_v$ directly expands the model's fixed-size recurrent capacity without bloating long-context memory.

`gdn_num_value_heads` controls $H_v$, establishing the multi-head parallelism of the retrieved representation channels.

---

## 2. Mechanics & Output Projection

During the forward pass:

```text
Sequence Input X: [Batch, Seq_Len, Hidden_Dim]
          │
          ├───> Value Projections V: [Batch, Seq_Len, H_v, d_v]
          │
          ▼
   Chunked Parallel Recurrent Scan (H_v parallel streams)
          │
          ▼
Retrieved Outputs Y: [Batch, Seq_Len, H_v, d_v]
          │
          ▼  Reshape / Flatten
   [Batch, Seq_Len, H_v * d_v]
          │
          ▼  Out-Projection (Linear)
   [Batch, Seq_Len, Hidden_Dim]
```

- Each value head produces a $d_v$-dimensional output vector per token.
- All $H_v$ head outputs are concatenated into a single wide feature vector of size $H_v \times d_v$ and passed to the output projection matrix.

---

## 3. Options & Default

| Parameter | Type | Default | Options |
| :--- | :--- | :--- | :--- |
| `gdn_num_value_heads` | `int` | `32` | Positive integers, typically a power of 2 (e.g., `16`, `32`, `64`) |

---

## 4. Interactions with Related Parameters

| Related Parameter | Interaction |
| :--- | :--- |
| `gdn_num_key_heads` | Must divide `gdn_num_value_heads`. The ratio $H_v / H_k$ is the grouping factor. |
| `gdn_value_head_dim` | Total hidden dimension before the final out-projection is `gdn_num_value_heads * gdn_value_head_dim`. |
| `base_emb_dim` | The output projection maps `gdn_num_value_heads * gdn_value_head_dim` back to `base_emb_dim`. |
| `mesh_axes` / `logical_axis_rules` | In MaxText, GDN projections and head dimensions are sharded across the `tensor` or `fsdp` mesh axes. |

---

## 5. Practical Guidance & Failure Modes

| Scenario | Recommendation |
| :--- | :--- |
| **Qwen3-Next Standard** | `gdn_num_value_heads: 32` with `gdn_num_key_heads: 16` (2:1 ratio). |
| **Scaling Model Width** | Scale `gdn_num_value_heads` proportionally with `base_emb_dim` to maintain balanced width across attention and MLP layers. |
| **Sharding Constraint** | `gdn_num_value_heads` must be divisible by the tensor parallelism degree ($TP$) to avoid uneven head splitting across TPU cores. |

---

### One-line intuition

> `gdn_num_value_heads` sets the number of value heads in Gated DeltaNet, defining the number of parallel recurrent memory state matrices and the width of the output representation.
