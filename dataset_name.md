## 1. Why does `dataset_name` exist?

TensorFlow Datasets maintains an extensive registry of standardized datasets with explicit version tags (e.g. `c4/en:3.1.0`, `wikipedia/20220301.en`).

```text
dataset_name: "c4/en:3.1.0"
                     │
                     ▼
         Pulls C4 English Version 3.1.0
```

`dataset_name` identifies which TFDS catalog dataset or version to instantiate.

---

## 2. Mechanics

In TFDS loading routines:
```python
builder = tfds.builder(dataset_name, data_dir=dataset_path)
```
MaxText verifies that the schema contains text features matching `train_data_columns`.

---

## 3. Options & Default

| Parameter | Type | Default | Options |
| :--- | :--- | :--- | :--- |
| `dataset_name` | `str` | `'c4/en:3.1.0'` | Any registered TFDS dataset name/version string |

---

## 4. Interactions with Related Parameters

- **`eval_dataset_name`**: The evaluation counterpart.
- **`dataset_type: tfds`**: Active only for TFDS pipelines.

---

## 5. Practical Scenarios & Failure Modes

| Scenario | Symptom / Failure | Fix |
| :--- | :--- | :--- |
| **Invalid TFDS name or missing dataset version** | `tfds.core.registered.DatasetNotFoundError` | Verify exact name with `tfds.list_builders()`. |

---

### One-line intuition

> `dataset_name` specifies the TFDS dataset identifier and version string used for training under the TFDS pipeline.
