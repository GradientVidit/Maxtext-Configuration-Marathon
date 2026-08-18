## 1. Why does `grain_eval_files` exist?

Validation and out-of-sample perplexity monitoring in Grain require independent evaluation data paths.

```text
grain_train_files: "gs://bucket/train/*.array_record"
grain_eval_files:  "gs://bucket/eval/*.array_record"
```

`grain_eval_files` specifies the file patterns for Grain evaluation splits.

---

## 2. Mechanics

Instantiates a dedicated evaluation Grain data loader using eval worker configs (`grain_worker_count_eval`).

---

## 3. Options & Default

| Parameter | Type | Default | Options |
| :--- | :--- | :--- | :--- |
| `grain_eval_files` | `str` | `''` | GCS path pattern string |

---

## 4. Interactions with Related Parameters

- **`grain_train_files`**: Training file counterpart.
- **`grain_worker_count_eval`**: Number of worker processes for eval.

---

## 5. Practical Scenarios & Failure Modes

| Scenario | Symptom / Failure | Fix |
| :--- | :--- | :--- |
| **Missing eval files** | Periodic eval step fails | Provide valid eval file pattern or disable eval. |

---

### One-line intuition

> `grain_eval_files` designates the file patterns for evaluation data within the Grain pipeline.
