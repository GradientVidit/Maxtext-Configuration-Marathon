## 1. Why does `grain_file_type` exist?

Grain supports two primary serialization formats:
1. **ArrayRecord (`'arrayrecord'`)**: Google's high-performance record storage format supporting random access, indexed chunking, and multi-threaded parallel streaming from cloud buckets.
2. **Parquet (`'parquet'`)**: Apache columnar storage format standard in modern data lakes.

```text
                               grain_file_type
                                      │
               ┌──────────────────────┴──────────────────────┐
               ▼                                             ▼
       'arrayrecord'                                     'parquet'
(Indexed fast random access)                    (Columnar Arrow storage)
```

`grain_file_type` instructs Grain which reader and deserializer to use.

---

## 2. Mechanics & Performance

- **ArrayRecord**: Designed for random and sharded reads without parsing unnecessary columns. Preferred for maximum TPU training MFU.
- **Parquet**: Reads row groups; requires a shuffle buffer (`grain_shuffle_buffer_size`) to achieve pseudorandom distribution.

---

## 3. Options & Default

| Parameter | Type | Default | Options |
| :--- | :--- | :--- | :--- |
| `grain_file_type` | `str` | `'arrayrecord'` | `'arrayrecord'`, `'parquet'` |

---

## 4. Interactions with Related Parameters

- **`grain_shuffle_buffer_size`**: Critical when using `'parquet'` due to sequential row-group reads.
- **`grain_train_files`**: ArrayRecord supports weighted multi-source strings.

---

## 5. Practical Scenarios & Failure Modes

| Scenario | Symptom / Failure | Fix |
| :--- | :--- | :--- |
| **Ingesting Parquet data with default config** | ArrayRecord decoder error on Parquet magic bytes | Set `grain_file_type: 'parquet'`. |
| **Max throughput pretraining** | Parquet decompression can bottleneck high-speed TPU v5e pods | Convert dataset to ArrayRecord for optimal streaming. |

---

### One-line intuition

> `grain_file_type` selects the underlying storage container format (ArrayRecord or Parquet) for Grain data loaders.
