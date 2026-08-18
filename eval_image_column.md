## 1. Why does `eval_image_column` exist?

Multimodal evaluation suites (such as VQA, MathVista, or DocVQA) may store visual inputs under distinct column names compared to pretraining web scrapes.

```text
Eval Multimodal Record:
{
  "query": "What is depicted?",
  "image_file": "<bytes>"
}
          │
          ▼
 [eval_image_column]: 'image_file'
```

`eval_image_column` specifies the visual data field for evaluation splits.

---

## 2. Mechanics

When evaluating Vision-Language Models, the evaluation pipeline extracts the image tensor identified by `eval_image_column`, normalizes the image, and computes validation metrics alongside text queries.

---

## 3. Options & Default

| Parameter | Type | Default | Options |
| :--- | :--- | :--- | :--- |
| `eval_image_column` | `str` | `'image'` | Column name string (e.g. `'image'`, `'visual'`, `'image_path'`) |

---

## 4. Interactions with Related Parameters

- **`train_image_column`**: Training counterpart.
- **`eval_data_columns`**: Accompanies the image data for text processing.

---

## 5. Practical Scenarios & Failure Modes

| Scenario | Symptom / Failure | Fix |
| :--- | :--- | :--- |
| **Eval benchmark has visual column named `'visual'`** | `KeyError: 'image'` during eval step | Set `eval_image_column: 'visual'`. |

---

### One-line intuition

> `eval_image_column` defines the image column name for evaluation datasets in multimodal training runs.
