## 1. Why does `indexer_topk` exist?

After the DSA Indexer scores the query against all $N$ tokens in the sequence context, it must extract a sparse candidate set for the main attention mechanism to process.

`indexer_topk` sets the exact number of top-scoring tokens ($k$) retained:

$$\text{Candidate Set } \mathcal{S}_q = \text{TopK}_{j \in [1, N]}(\text{IndexerScore}(q, k_j), \text{topk}=k)$$

$$\text{Main MLA Attention is evaluated strictly over } j \in \mathcal{S}_q$$

```text
Full Context: N = 128,000 tokens
       │
       ▼
[ DSA Indexer Module ] ──> Evaluates 128,000 scores
       │
       ▼
[ Top-K Selection (indexer_topk = 2048) ] ──> Selects top 2048 token indices
       │
       ▼
[ Main MLA Attention ] ──> Evaluates ONLY 2,048 tokens (98.4% computation skipped!)
```

---

## 2. Options & Defaults

| Value | Behavior | Notes |
|---|---|---|
| `2048` | Selects $k=2048$ tokens per query. | **Default**. Exact DeepSeek-V3.2 specification. |
| Any integer $> 0$ (e.g. `1024`, `4096`) | Custom candidate subset size. | Higher = higher retrieval recall; Lower = higher compute speedup. |

Default in `base.yml`: `2048`

---

## 3. Scaling Properties

| Context Window ($N$) | Tokens Evaluated | Compute Reduction vs Full Attention |
|---|---|---|
| $8{,}192$ (8K) | $2{,}048$ | $4.0	imes$ |
| $32{,}768$ (32K) | $2{,}048$ | $16.0	imes$ |
| $131{,}072$ (128K) | $2{,}048$ | $\mathbf{64.0	imes}$ |
| $1{,}048{,}576$ (1M) | $2{,}048$ | $\mathbf{512.0	imes}$ |

As sequence length approaches 1M tokens, main attention FLOPs remain fixed at a constant $k=2048$ budget.

---

## 4. Interactions with other parameters

| Related Parameter | Interaction |
|---|---|
| [[use_indexer]] | Parent switch: `indexer_topk` is active only when `use_indexer: true`. |
| [[indexer_use_approx_top_k]] | When enabled, uses fast approximate selection rather than exact full sorting. |
| [[indexer_mask_exact_topk]] | Enforces hard binary masking to precisely the top-$k$ tokens. |

---

## 5. Practical Scenarios

- **DeepSeek-V3.2 Pretraining & Serving:** Set `indexer_topk: 2048`.
- **Extreme Speed Inference on Shorter Prompts ($N \le 16\text{K}$):** Set `indexer_topk: 1024` to halve attention latency.

---

### One-line intuition

> **`indexer_topk` defines the exact number of top-ranking tokens ($k=2048$) selected by the DSA indexer, fixing main attention computation to a constant budget regardless of context length.**
