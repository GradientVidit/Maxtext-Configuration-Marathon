## 1. Why does `indexer_use_approx_top_k` exist?

Evaluating exact $\text{TopK}$ across $N = 128{,}000+$ tokens requires sorting or running a min-heap algorithm over all $N$ scores, which has $O(N \log k)$ complexity and introduces synchronization barriers across accelerator cores.

**Approximate Top-K** replaces the exact sort with a fast, bucketed thresholding algorithm:
1. It partitions score ranges into histogram bins or estimates a dynamic score cutoff threshold.
2. Tokens exceeding the threshold are collected into the candidate buffer.
3. This achieves **$O(N)$ linear time complexity** with zero global sorting overhead.

```text
Exact Top-K (indexer_use_approx_top_k=False):
  N = 128K Scores ──> Full Array Sort / Min-Heap (O(N log k)) ──> Exactly 2048 Highest Tokens

Approximate Top-K (indexer_use_approx_top_k=True):
  N = 128K Scores ──> Fast Histogram / Cutoff Sampling (O(N)) ──> ~95% True Top-K Recall (Much Faster)
```

---

## 2. Options & Defaults

| Value | Selection Algorithm | Execution Speed | Recall Accuracy |
|---|---|---|---|
| `false` | Exact Top-$K$ sort. | Standard. | $100\%$ exact top-$k$. **Default**. |
| `true` | Fast approximate bucketed Top-$K$. | Up to $3	imes$ faster top-$k$ dispatch on TPUs/GPUs. | Tuned by `indexer_approx_top_k_recall` (e.g. $95\%$). |

Default in `base.yml`: `false`

---

## 3. Speed vs. Recall Trade-Off

Because the Indexer selects $k=2048$ tokens out of a massive sequence, the difference between the 2048th token and the 2050th token is statistically negligible. A $5\%$ drop in recall (retrieving 1945 of the true top-2048 tokens plus 103 high-ranking nearby candidates) produces virtually zero downstream perplexity degradation while eliminating sorting bottlenecks.

---

## 4. Interactions with other parameters

| Related Parameter | Interaction |
|---|---|
| [[indexer_approx_top_k_recall]] | Target recall factor when approximate top-$k$ is enabled (default `0.95`). |
| [[indexer_topk]] | Target candidate count $k$. |
| [[use_indexer]] | Prerequisite flag. |

---

## 5. Practical Scenarios

- **Pretraining Phase:** Keep `indexer_use_approx_top_k: false` for strict numerical stability.
- **Ultra-Long Context Inference ($N \ge 128\text{K}$):** Set `indexer_use_approx_top_k: true` to prevent the Top-$K$ kernel from becoming a latency bottleneck during autoregressive decoding.

---

### One-line intuition

> **`indexer_use_approx_top_k=true` replaces exact $O(N \log k)$ sorting with a fast $O(N)$ bucketed selection algorithm, boosting top-$k$ dispatch speed at massive context lengths.**
