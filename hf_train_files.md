## 1. Why does `hf_train_files` exist?

When loading raw data files (JSONL, Parquet, CSV) via the HuggingFace pipeline without a formal dataset script, users need to pinpoint specific glob patterns for training files.

```text
hf_train_files: "data/train-*.parquet"
                       │
                       ▼
  Loads matching files for training split only
```

`hf_train_files` mirrors the `data_files` mapping in `datasets.load_dataset()`.

---

## 2. Mechanics

MaxText passes `hf_train_files` as the `data_files` argument for the training split in `datasets.load_dataset()`.

---

## 3. Options & Default

| Parameter | Type | Default | Options |
| :--- | :--- | :--- | :--- |
| `hf_train_files` | `str` | `''` | File path, URL, or glob pattern (e.g. `'train/*.parquet'`) |

---

## 4. Interactions with Related Parameters

- **`hf_eval_files`**: The evaluation counterpart.
- **`dataset_type: hf`**: Required.

---

## 5. Practical Scenarios & Failure Modes

| Scenario | Symptom / Failure | Fix |
| :--- | :--- | :--- |
| **Loading standalone Parquet files from local disk** | Need to bypass standard Hub repo metadata | Set `hf_path: "parquet"` and `hf_train_files: "/data/*.parquet"`. |

---

### One-line intuition

> `hf_train_files` specifies the file path pattern or glob for training files when loading via Hugging Face.
