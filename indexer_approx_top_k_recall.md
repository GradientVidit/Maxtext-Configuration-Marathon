## 1. Why does `indexer_approx_top_k_recall` exist?

When using approximate top-$k$ token selection (`indexer_use_approx_top_k: true`), the algorithm must calibrate its selection threshold to balance **computational speed** against **retrieval fidelity**.

Recall is defined as the fraction of true top-$k$ tokens captured by the approximate algorithm:

$$\text{Recall} = rac{|\mathcal{S}_{\text{approx}} \cap \mathcal{S}_{\text{exact}}|}{k}$$

`indexer_approx_top_k_recall` sets the target recall value $R \in (0.0, 1.0]$.

```text
Higher Recall (e.g. 0.99):
  - Calibrates tighter candidate thresholds.
  - Captures 99% of exact top-k tokens.
  - Slightly higher computation cost (closer to exact sort).

Lower Recall (e.g. 0.90):
  - Uses wider, faster bucketing.
  - Captures 90% of exact top-k tokens.
  - Maximum kernel execution speed.
```

---

## 2. Options & Defaults

| Value | Behavior | Notes |
|---|---|---|
| `0.95` | Targets $95\%$ recall of the true top-$k$ token set. | **Default**. Optimal empirical trade-off. |
| Any float in $(0.0, 1.0]$ | Custom recall target. | Setting `1.0` forces exact top-$k$ behavior. |

Default in `base.yml`: `0.95`

---

## 3. Calibration Mechanics

The approximate top-$k$ kernel uses `indexer_approx_top_k_recall` to dynamically adjust the score quantile boundary:

$$\text{Threshold } 	au = \text{Quantile}\left(\text{Scores}, 1.0 - rac{k \cdot (1 + \text{buffer})}{N}
ight)$$

At $R=0.95$, the kernel guarantees that at least $0.95 	imes 2048 = 1945$ of the absolute highest-scoring tokens are included in the attention candidate buffer.

---

## 4. Interactions with other parameters

| Related Parameter | Interaction |
|---|---|
| [[indexer_use_approx_top_k]] | Prerequisite: `indexer_approx_top_k_recall` is active only when `indexer_use_approx_top_k: true`. |
| [[indexer_topk]] | Target candidate count $k$. |
| [[use_indexer]] | Root parent switch. |

---

## 5. Practical Scenarios

- **Production Serving:** Leave `indexer_approx_top_k_recall: 0.95` (default).
- **Quality-Critical Long-Context Reasoning:** Bump to `0.98` if needle-in-a-haystack accuracy shows sensitivity to marginal candidate drops.

---

### One-line intuition

> **`indexer_approx_top_k_recall` defines the target accuracy fraction ($0.95$) for approximate top-$k$ selection, tuning the trade-off between kernel dispatch latency and retrieval precision.**
