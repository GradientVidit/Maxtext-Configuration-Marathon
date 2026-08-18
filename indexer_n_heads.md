## 1. Why does `indexer_n_heads` exist?

In DeepSeek Sparse Attention (DSA), a single query token may need to retrieve past tokens based on multiple diverse semantic relationships simultaneously (e.g. tracking variable definitions in code, identifying syntax anchors, and following conversational dialogue history).

`indexer_n_heads` provides the Indexer with multiple parallel attention heads ($N_{idx\_heads}$), allowing it to score token affinity across multiple orthogonal semantic subspaces before aggregating scores:

```text
Query X ──> Indexer Query Projection (indexer_n_heads = 64):
  ├── Head 0:  Scores semantic entity matching
  ├── Head 1:  Scores syntactic / code structure matching
  ├── ...
  └── Head 63: Scores long-range conversational dependencies

Scores from all 64 heads are averaged/reduced ──> Global Per-Token Score ──> Top-k Selection
```

---

## 2. Options & Defaults

| Value | Behavior | Notes |
|---|---|---|
| `64` | Instantiates 64 parallel indexer scoring heads. | **Default**. Exact DeepSeek-V3.2 specification. |
| Any integer $> 0$ | Number of indexer heads. | Must be compatible with tensor parallelism sharding rules. |

Default in `base.yml`: `64`

---

## 3. Parameter Footprint

$$\text{Indexer Projection Parameters per Layer} = 2 	imes d_{model} 	imes (N_{idx\_heads} 	imes d_{idx})$$

For DeepSeek-V3.2 ($d_{model}=5120$, $N_{idx}=64$, $d_{idx}=128$):
$$2 	imes 5120 	imes (64 	imes 128) = 2 	imes 5120 	imes 8192 = 83{,}886{,}080 \text{ parameters per layer}$$

---

## 4. Interactions with other parameters

| Related Parameter | Interaction |
|---|---|
| [[use_indexer]] | Parent switch: `indexer_n_heads` is active only when `use_indexer: true`. |
| [[indexer_head_dim]] | Per-head feature width ($d_{idx}=128$). |
| [[mla_qk_head_chunk_size]] | Limits HBM footprint by sequentially evaluating the QK matrix in the Indexer across unsharded local heads. |

---

## 5. Practical Scenarios

- **DeepSeek-V3.2 Pretraining:** Keep `indexer_n_heads: 64`.
- **Sharding on Tensor Parallelism:** Ensure `indexer_n_heads` is a multiple of your model's tensor parallel (TP) degree.

---

### One-line intuition

> **`indexer_n_heads` sets the number of parallel scoring heads ($N_{idx}=64$) in the DSA indexer, providing multi-aspect semantic coverage for sparse token retrieval.**
