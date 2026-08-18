## 1. Why does `indexer_head_dim` exist?

The DSA Indexer is a specialized, lightweight multi-head scoring module that predicts relevance scores between queries and cached tokens.

Like standard attention heads, the indexer queries ($Q_{idx}$) and keys ($K_{idx}$) project hidden states into a dedicated subspace:

$$Q_{idx} \in \mathbb{R}^{B 	imes N_{idx\_heads} 	imes d_{idx}}, \quad K_{idx} \in \mathbb{R}^{B 	imes S 	imes N_{idx\_heads} 	imes d_{idx}}$$

$$\text{Indexer Logits} = rac{Q_{idx} K_{idx}^T}{\sqrt{d_{idx}}}$$

`indexer_head_dim` sets the dimensionality $d_{idx}$ of each indexer head.

---

## 2. Options & Defaults

| Value | Behavior | Notes |
|---|---|---|
| `128` | Head dimension of $d_{idx}=128$. | **Default**. Exact DeepSeek-V3.2 specification. |
| Any integer $> 0$ | Custom indexer head dimension. | Governs indexer query/key projection width. |

Default in `base.yml`: `128`

---

## 3. Indexer Dimension vs. Main Attention Dimension

```text
Layer Module              Head Count (N_heads)    Head Dimension (d_head) Total Projection Width
─────────────────────────────────────────────────────────────────────────────────────────────
Main MLA Attention (Q)    128                     192 (128 NoPE + 64 RoPE) 24,576
DSA Indexer               64                      128                      8,192
```

The Indexer uses a narrower head dimension and lower total projection width than the main attention layers, allowing fast candidate scoring before dispatching heavy tensor contractions.

---

## 4. Interactions with other parameters

| Related Parameter | Interaction |
|---|---|
| [[use_indexer]] | Parent switch: `indexer_head_dim` is active only when `use_indexer: true`. |
| [[indexer_n_heads]] | Determines total indexer query/key parameters: $d_{model} 	imes (N_{idx\_heads} 	imes d_{idx})$. |
| [[indexer_topk]] | Number of tokens selected from the resulting score matrix. |

---

## 5. Practical Scenarios

- **DeepSeek-V3.2 Configurations:** Keep `indexer_head_dim: 128`.
- **Ablating Indexer Capacity:** Reducing to $d_{idx}=64$ cuts indexer parameter overhead in half for smaller distilled models.

---

### One-line intuition

> **`indexer_head_dim` sets the representation width ($d_{idx}=128$) of each DSA indexer head, defining the feature capacity used to evaluate token relevance scores.**
