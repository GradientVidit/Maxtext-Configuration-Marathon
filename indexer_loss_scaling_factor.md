## 1. Why does `indexer_loss_scaling_factor` exist?

The DSA Indexer requires an explicit auxiliary loss signal to learn which tokens are important. It is trained by minimizing the **Kullback-Leibler (KL) divergence** between the Indexer's predicted token probability distribution and the full attention score distribution from the main model:

$$\mathcal{L}_{\text{indexer}} = D_{\text{KL}}\left(\text{Softmax}(\text{DenseAttnScores}) \parallel \text{Softmax}(\text{IndexerScores})
ight)$$

The total multi-task loss is:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{cross-entropy}} + \lambda_{\text{idx}} \cdot \mathcal{L}_{\text{indexer}}$$

$$\text{Where:}\quad \lambda_{\text{idx}} = \text{indexer\_loss\_scaling\_factor}$$

```text
Forward Pass:
  Main Attention Logits ──> Full Attention Distribution (Teacher)
                                         │
                                         ▼   KL Divergence Loss
  Indexer Logits        ──> Indexer Distribution (Student)
                                         │
                                         ▼
                            Multiply by indexer_loss_scaling_factor (λ_idx)
                                         │
                                         ▼
                               Add to Total Loss
```

When `indexer_loss_scaling_factor = 0.0`, the indexer auxiliary loss is disabled, and no indexer gradients are computed.

---

## 2. Options & Defaults

| Value | Behavior | Notes |
|---|---|---|
| `0.0` | Indexer auxiliary loss disabled. | **Default**. Used during pure inference or when indexer weights are frozen. |
| Any float $> 0.0$ (e.g. `0.1`, `0.05`) | Multiplies the Indexer KL-divergence auxiliary loss before adding to total loss. | Enables `indexer_sparse_training` modes. |

Default in `base.yml`: `0.0`

---

## 3. Gating Role in Indexer Training

`indexer_loss_scaling_factor` acts as the **master training gate** for all indexer optimization logic:
- If `indexer_loss_scaling_factor == 0.0`, MaxText skips calculating indexer KL divergence and backpropagating indexer loss entirely, even if `use_indexer: true`.
- To train the indexer, `indexer_loss_scaling_factor` must be set to a positive value (typically `0.1` or `0.01`).

---

## 4. Interactions with other parameters

| Related Parameter | Interaction |
|---|---|
| [[indexer_sparse_training]] | Active only when `indexer_loss_scaling_factor > 0.0`. |
| [[use_indexer]] | Parent feature switch. |
| [[indexer_topk]] | Governs the token subset evaluated in sparse training mode. |

---

## 5. Practical Scenarios

- **Inference Only / Zero-Shot Evaluation:** Leave `indexer_loss_scaling_factor: 0.0`.
- **Training DeepSeek-V3.2 Indexer:** Set `indexer_loss_scaling_factor: 0.1` during Phase 1 dense warm-up, and reduce to `0.01` during Phase 2 sparse co-training.

---

### One-line intuition

> **`indexer_loss_scaling_factor` sets the weight $\lambda_{idx}$ for the Indexer's KL-divergence auxiliary loss, serving as the master switch that enables indexer training.**
