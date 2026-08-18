## 1. Why does `train_split` exist?

Datasets in TFDS or HuggingFace partition data into distinct subsets (e.g. `train`, `train[:90%]`, `validation`, `test`).

```text
Dataset Shards
      │
      ├── [ train_split: 'train' ]          ──> Training Loop
      │
      └── [ eval_split: 'validation' ]      ──> Evaluation Loop
```

`train_split` specifies the slice or split name designated for training.

---

## 2. Mechanics

In TFDS:
```python
dataset = builder.as_dataset(split=train_split)
```
This supports TFDS slicing API syntax (e.g. `'train[:10000]'` or `'train'`).

---

## 3. Options & Default

| Parameter | Type | Default | Options |
| :--- | :--- | :--- | :--- |
| `train_split` | `str` | `'train'` | Split name string or TFDS slice definition |

---

## 4. Interactions with Related Parameters

- **`eval_split`**: The corresponding split for validation.
- **`dataset_type: tfds` / `dataset_type: hf`**: Used to select dataset partitions.

---

## 5. Practical Scenarios & Failure Modes

| Scenario | Symptom / Failure | Fix |
| :--- | :--- | :--- |
| **Dataset uses `'training'` instead of `'train'`** | `ValueError: Split 'train' not found in dataset` | Set `train_split: 'training'`. |

---

### One-line intuition

> `train_split` designates the specific split or slice of the TFDS dataset to stream into the training loop.
