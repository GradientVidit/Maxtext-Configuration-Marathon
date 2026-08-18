## 1. Why does `indexer_mask_exact_topk` exist?

After the DSA Indexer identifies the set of top-scoring token positions, there are two distinct ways to construct the attention mask for the main MLA attention computation:

```text
1. Hard Exact Masking (indexer_mask_exact_topk = True, Default):
   Tokens in Top-K Set ──> Mask Value = 0.0 (Fully attend)
   Tokens outside Top-K ──> Mask Value = -1e9 / -inf (Zero attention weight)

2. Soft / Threshold Masking (indexer_mask_exact_topk = False):
   Applies a smooth sigmoid / threshold multiplier based on indexer score confidence,
   allowing marginal tokens near the boundary to retain partial gradient contributions.
```

`indexer_mask_exact_topk` toggles between hard binary top-$k$ masking and soft thresholding.

---

## 2. Options & Defaults

| Value | Masking Behavior | Notes |
|---|---|---|
| `true` | Strictly binary mask: attends to exactly the top-$k$ tokens; all other positions are completely masked out. | **Default**. Standard for DeepSeek-V3.2. |
| `false` | Soft thresholded mask where token contributions are weighted by indexer confidence. | Experimental for differentiable routing. |

Default in `base.yml`: `true`

---

## 3. Why Hard Masking is Preferred in Production

1. **Hardware Efficiency:** Hard binary masking allows the attention kernel to execute sparse gather-gemm operations, completely skipping unselected tokens in memory reads.
2. **Attention Weight Normalization:** Soft masking distorts the softmax denominator by mixing fractional scores, whereas hard masking preserves standard softmax probability distributions over the active set.

---

## 4. Interactions with other parameters

| Related Parameter | Interaction |
|---|---|
| [[indexer_topk]] | Defines the active set size $k$ when hard masking is enabled. |
| [[use_indexer]] | Parent switch. |
| [[indexer_sparse_training]] | During sparse training, hard masking ensures gradients flow strictly through the active $k$ token pathways. |

---

## 5. Practical Scenarios

- **Standard Pretraining & Inference:** Leave `indexer_mask_exact_topk: true` (default).
- **Differentiable Router Research:** Set `false` when exploring end-to-end continuous relaxation of the token selection mechanism.

---

### One-line intuition

> **`indexer_mask_exact_topk=true` enforces strict binary masking on the top-$k$ token set, allowing unselected tokens to be completely skipped during main attention computation.**
