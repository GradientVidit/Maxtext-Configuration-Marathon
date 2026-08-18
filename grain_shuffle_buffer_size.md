## 1. Why does `grain_shuffle_buffer_size` exist?

Sequential storage formats like **Parquet** or **TFRecord** store data in contiguous row groups. If read sequentially without shuffling, the model will see all samples from topic A, then all samples from topic B, damaging gradient generalization.

```text
Sequential Records: [ A1, A2, A3, B1, B2, B3 ]
                           │
                           ▼
  [ Shuffle Buffer: Capacity grain_shuffle_buffer_size ]
                           │
                           ▼
Randomized Output:  [ A2, B1, A3, B3, A1, B2 ]
```

`grain_shuffle_buffer_size` sets the size of the in-memory window used to randomly sample and interleave records.

---

## 2. Mechanics

Grain loads `grain_shuffle_buffer_size` elements into an internal buffer. For every element consumed by the batch loader, a new element from the storage stream replaces it at random.

---

## 3. Options & Default

| Parameter | Type | Default | Options |
| :--- | :--- | :--- | :--- |
| `grain_shuffle_buffer_size` | `int` | `100` | Positive integer (e.g. `100` to `10_000`) |

---

## 4. Interactions with Related Parameters

- **`grain_file_type: 'parquet'`**: Crucial for Parquet/TFRecord; ArrayRecord uses global index permutation instead.
- **`data_shuffle_seed`**: PRNG seed for deterministic shuffle order.

---

## 5. Practical Scenarios & Failure Modes

| Scenario | Symptom / Failure | Fix |
| :--- | :--- | :--- |
| **Training on clustered Parquet files** | Loss oscillations due to correlated batches | Increase `grain_shuffle_buffer_size` to 1,000 or 5,000. |

---

### One-line intuition

> `grain_shuffle_buffer_size` defines the in-memory reservoir size used to randomize records when streaming sequential formats like Parquet.
