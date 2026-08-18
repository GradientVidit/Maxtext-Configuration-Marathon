## 1. Why does `packing` exist?

Text documents vary wildly in length. In a naive batching pipeline, variable-length documents are padded with `<pad>` tokens to the maximum sequence length `max_target_length`.

```text
Without Packing (Padding Waste):
Seq 1: [ Doc A (200 tokens) | <pad> <pad> <pad> ... (1848 padding tokens) ]  --> 90% compute wasted!
Seq 2: [ Doc B (500 tokens) | <pad> <pad> <pad> ... (1548 padding tokens) ]  --> 75% compute wasted!

With Packing (Zero Waste):
Seq 1: [ Doc A (200) | Doc B (500) | Doc C (1300) | Doc D (48) ]            --> 100% compute utilized!
```

Padding tokens consume full quadratic $O(S^2)$ attention compute and linear feed-forward compute without contributing to gradient updates. 

`packing: true` concatenates multiple independent documents into a single contiguous sequence of length `max_target_length`, using segment IDs or block-diagonal attention masks to prevent cross-document attention leakage. This increases effective training throughput by **2x to 5x**.

---

## 2. Mechanics & Segment Masks

When `packing: true`:
1. The dataset pipeline packs documents together until the token budget `max_target_length` is reached.
2. It constructs two key auxiliary tensors:
   - `decoder_segment_ids`: Integer tensor indicating which document each token belongs to (e.g. `[1, 1, ..., 2, 2, ..., 3, 3]`).
   - `decoder_positions`: Integer position IDs resetting to $0$ at the start of each packed document.
3. The attention layer uses these IDs to mask out attention across different segment IDs.

```text
Tokens:      [ A1  A2  A3  B1  B2  C1  C2  C3  C4 ]
Positions:   [  0   1   2   0   1   0   1   2   3 ]
Segment IDs: [  1   1   1   2   2   3   3   3   3 ]
```

---

## 3. Options & Default

| Parameter | Type | Default | Options |
| :--- | :--- | :--- | :--- |
| `packing` | `bool` | `true` | `true` (pack examples), `false` (pad examples with `<pad>`) |

---

## 4. Interactions with Related Parameters

- **`grain_packing_type`**: Strategy used by Grain (`first_fit`, `best_fit`, `concat_then_split`).
- **`max_segments_per_seq`**: Hardware constraint for TransformerEngine on GPUs.
- **`grain_use_elastic_iterator`**: Incompatible with packing; requires `packing: false`.
- **`attention`**: Attention kernels (FlashAttention, CudnnFlashTE) must support segment ID masks or cumulative sequence lengths.

---

## 5. Practical Scenarios & Failure Modes

| Scenario | Symptom / Failure | Fix |
| :--- | :--- | :--- |
| **Short-document pretraining (e.g. Wikipedia/CommonCrawl)** | 70%+ compute wasted on pad tokens if packing disabled | Ensure `packing: true`. |
| **Attention contamination across documents** | Model attends to unrelated documents in same sequence | Verify segment masking is supported by chosen attention kernel. |
| **Elastic training with Grain** | Crash on startup | Elastic iterator requires `packing: false`. |

---

### One-line intuition

> `packing` packs multiple short documents into fixed-length sequences without padding, eliminating wasted FLOPs and multiplying effective training throughput.
