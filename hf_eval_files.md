## 1. Why does `hf_eval_files` exist?

When evaluation data is stored in standalone files separate from training data files, `hf_eval_files` specifies the evaluation file pattern.

```text
hf_eval_files: "data/eval-*.parquet"
```

`hf_eval_files` supplies explicit evaluation file patterns to Hugging Face `load_dataset()`.

---

## 2. Mechanics

Passed as `data_files` for the evaluation dataset instance.

---

## 3. Options & Default

| Parameter | Type | Default | Options |
| :--- | :--- | :--- | :--- |
| `hf_eval_files` | `str` | `''` | File path or glob pattern for evaluation data |

---

## 4. Interactions with Related Parameters

- **`hf_train_files`**: Training file counterpart.

---

## 5. Practical Scenarios & Failure Modes

| Scenario | Symptom / Failure | Fix |
| :--- | :--- | :--- |
| **Evaluation data located in separate GCS or local directory** | Explicit glob binding prevents training contamination | Set `hf_eval_files: "eval/*.jsonl"`. |

---

### One-line intuition

> `hf_eval_files` sets the file paths or glob pattern for evaluation data in Hugging Face pipelines.
