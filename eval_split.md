## 1. Why does `eval_split` exist?

To reliably monitor generalization without data leakage, evaluation must run strictly on an out-of-sample validation or test split.

```text
train_split: 'train'              ──> Updates weights
eval_split:  'validation'         ──> Computes out-of-sample validation loss
```

`eval_split` selects the split partition evaluated during periodic validation steps.

---

## 2. Mechanics

Passed directly to the dataset builder:
```python
eval_ds = builder.as_dataset(split=eval_split)
```

---

## 3. Options & Default

| Parameter | Type | Default | Options |
| :--- | :--- | :--- | :--- |
| `eval_split` | `str` | `'validation'` | `'validation'`, `'test'`, `'eval'`, or slice string |

---

## 4. Interactions with Related Parameters

- **`train_split`**: Training split.
- **`eval_dataset_name` / `eval_per_device_batch_size`**: The evaluation execution environment.

---

## 5. Practical Scenarios & Failure Modes

| Scenario | Symptom / Failure | Fix |
| :--- | :--- | :--- |
| **Dataset only contains `'test'` split** | Split `'validation'` not found error | Set `eval_split: 'test'`. |

---

### One-line intuition

> `eval_split` specifies the validation split name used to evaluate generalization performance.
