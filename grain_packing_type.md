## 1. Why does `grain_packing_type` exist?

When packing multiple variable-length text records into fixed-size sequences of length $S$, the bin-packing strategy directly determines how much padding remains in each packed sequence.

```text
Bin Packing Strategies:
Sequence Length = 1000

First Fit:
Doc 1 (600) -> Placed in Seq 1
Doc 2 (500) -> Cannot fit Seq 1, opens Seq 2 (Seq 1 left with 400 pad tokens)

Best Fit:
Doc 1 (600) -> Placed in Seq 1
Doc 2 (500) -> Placed in Seq 2
Doc 3 (400) -> Searches open bins -> Fits EXACTLY into Seq 1! (0 pad tokens)
```

`grain_packing_type` chooses the packing algorithm used by Grain's packing transformations (`grain.experimental.BestFitPackIterDataset` vs. `FirstFitPackIterDataset`).

---

## 2. Mechanics & Experimental Benchmarks

Grain supports three distinct packing algorithms in `src/maxtext/input_pipeline/_grain_data_processing.py`:

```text
┌─────────────────────┬────────────────────────────────────────────────────────────────────────┐
│ Strategy            │ Underlying Implementation & Properties                                 │
├─────────────────────┼────────────────────────────────────────────────────────────────────────┤
│ 'first_fit'         │ Inserts each example into the first open bin that has room.            │
│                     │ Fast, but leaves significant fragmentation / padding waste (~15-20%).   │
├─────────────────────┼────────────────────────────────────────────────────────────────────────┤
│ 'best_fit'          │ Uses grain.experimental.BestFitPackIterDataset across num_packing_bins.│
│                     │ Searches for the tightest fit. Reduces padding by up to 27.6%          │
│                     │ compared to first_fit with negligible CPU overhead.                     │
├─────────────────────┼────────────────────────────────────────────────────────────────────────┤
│ 'concat_then_split' │ Concatenates all documents into an unbroken token stream and slices    │
│                     │ strictly at S. 100% token density, but splits document boundaries.     │
└─────────────────────┴────────────────────────────────────────────────────────────────────────┘
```

```text
BestFitPackIterDataset Dataflow:
Input Stream ──> [ Bin Pool: 1024 Bins ] ──> Tightest Fit Match ──> Packed Tensor [B, S]
                                                                        (98%+ Token Density)
```

---

## 3. Options & Default

| Parameter | Type | Default | Options |
| :--- | :--- | :--- | :--- |
| `grain_packing_type` | `str` | `'first_fit'` | `'first_fit'`, `'best_fit'`, `'concat_then_split'` |

---

## 4. Interactions with Related Parameters

- **`packing`**: Must be `true` for `grain_packing_type` to take effect.
- **`max_segments_per_seq`**: Caps the maximum number of document segments allowed in any single sequence bin (crucial for GPU TransformerEngine).
- **`dataset_type`**: Must be `'grain'`.

---

## 5. Practical Scenarios & Failure Modes

| Scenario | Symptom / Failure | Fix |
| :--- | :--- | :--- |
| **High padding observed despite `packing: true`** | `first_fit` leaves leftover gaps in sequence bins | Set `grain_packing_type: 'best_fit'` to minimize padding and boost effective MFU. |
| **Pretraining where splitting documents across batches is acceptable** | Want 100% packed token density | Set `grain_packing_type: 'concat_then_split'`. |

---

### One-line intuition

> `grain_packing_type` selects the bin-packing algorithm (`first_fit`, `best_fit`, or `concat_then_split`) used by Grain, where `best_fit` reduces padding waste by up to 27.6% over baseline `first_fit`.
