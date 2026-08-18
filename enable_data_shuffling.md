## 1. Why does `enable_data_shuffling` exist?

Training a model on an unshuffled or statically ordered dataset leads to severe correlation bias, where consecutive batches from the same document or domain distort gradient moments and degrade generalization.

Conversely, during data pipeline debugging and determinism testing, developers must force the data loader to yield examples in strictly sequential, deterministic order:

```text
Dataset Shuffling Pipeline:
Raw Data Stream: [Doc A][Doc B][Doc C][Doc D]
                        │
             enable_data_shuffling ?
                        │
          ┌─────────────┴─────────────┐
          ▼                           ▼
        [true]                     [false]
   Pseudo-random             Strict sequential
   Permutation Stream         Deterministic Stream
   [Doc C][Doc A][Doc D]      [Doc A][Doc B][Doc C]
```

`enable_data_shuffling` controls whether the training input data pipeline permutes examples.

---

## 2. Fundamentals & Mechanics

- **`true` (Default):** Enables random record shuffling in tf.data or Grain data loaders using `data_shuffle_seed`.
- **`false`:** Reads dataset shards in natural, unpermuted disk order.

---

## 3. Options & Defaults

| Option | Value | Meaning |
|---|---|---|
| Default | `true` | Shuffles dataset for standard pretraining and fine-tuning. |
| Sequential Mode | `false` | Deterministic sequential reading for debugging data loaders. |

---

## 4. Interactions & Dependencies

```text
enable_data_shuffling: true ──> Governed by data_shuffle_seed
```

---

## 5. Practical Scenarios & Failure Modes

- **Accidental Sequential Training:** Setting `false` during multi-epoch pretraining causes the model to overfit to localized document clusters.

---

### One-line intuition

> **`enable_data_shuffling` toggles input dataset permutation, ensuring diverse mini-batch distributions during training.**
