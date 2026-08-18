## 1. Why does `engram_num_heads` exist?

When Engram retrieves N-gram embeddings from its hash tables, these raw hash embeddings must be projected and integrated into the transformer's multi-head hidden representation.

Rather than performing a single flat linear projection, Engram utilizes a **multi-head attention/projection structure**:

```text
Engram Retrieved Embeddings
             │
             ▼
   Split across engram_num_heads (H_engram)
   ┌─────────────┼─────────────┐
   ▼             ▼             ▼
Head 0        Head 1        Head (H_engram - 1)
(dim = d_e)   (dim = d_e)   (dim = d_e)
   │             │             │
   └─────────────┬─────────────┘
                 ▼
   Concatenate & Project back to Model Hidden Dim
```

This multi-head projection allows the model to route different aspects of the retrieved n-gram features (syntactic roles, entity types, semantic categories) into separate representation subspaces.

`engram_num_heads` sets the number of parallel projection heads allocated to the Engram integration block.

---

## 2. Mechanics & Output Geometry

- **Total Intermediate Dimension**: The total width of the unprojected Engram feature space is:
  $$\text{Total Engram Dim} = \text{engram\_num\_heads} \times \text{engram\_head\_dim}$$
  With default `engram_num_heads = 8` and `engram_head_dim = 1280`, total intermediate width is $8 \times 1280 = 10,240$.
- **Gating & Fusion**: Each head's output undergoes head-wise gating and temporal convolution before being linearly projected back to `base_emb_dim` and added to the residual stream.

---

## 3. Options & Default

| Parameter | Type | Default | Options |
| :--- | :--- | :--- | :--- |
| `engram_num_heads` | `int` | `8` | Positive integer (e.g., `4`, `8`, `16`) |

---

## 4. Interactions with Related Parameters

| Related Parameter | Interaction |
| :--- | :--- |
| `engram_head_dim` | Dictates individual head width; total Engram channel capacity is `engram_num_heads * engram_head_dim`. |
| `engram_layers` | Only instantiated on layers designated by `engram_layers`. |
| `base_emb_dim` | Output projection projects from `engram_num_heads * engram_head_dim` back to `base_emb_dim`. |

---

## 5. Practical Guidance

| Scenario | Recommendation | Notes |
| :--- | :--- | :--- |
| **DeepSeek Engram Standard** | `engram_num_heads: 8` | Standard preset matching published DeepSeek Engram architecture. |
| **Scaled Down / Small Model** | `engram_num_heads: 4` | Scales down Engram parameter overhead proportionally with smaller transformer widths. |

---

### One-line intuition

> `engram_num_heads` specifies the number of projection heads in the Engram module, partitioning retrieved n-gram memory features into distinct subspace channels.
