## 1. Why does `train_data_columns` exist?

Modern datasets in Parquet, JSONL, HuggingFace, or TFDS contain rich tabular structures with multiple diverse columns (e.g. `text`, `prompt`, `response`, `chosen`, `rejected`, `metadata`, `author`).

```text
Dataset Row:
{
  "text": "Regular pretraining text...",
  "chosen": "High-quality aligned answer...",
  "rejected": "Low-quality answer...",
  "id": 1024
}
               │
               ▼
      [train_data_columns]
               │
  ┌────────────┴────────────┐
  ▼                         ▼
['text']            ['chosen', 'rejected']
(Standard Pretrain/SFT)   (DPO Training)
```

`train_data_columns` tells the input pipeline which specific keys/fields to extract and feed into the tokenization and loss computation graphs.

---

## 2. Mechanics & Objective Routing

In standard pretraining and SFT, `train_data_columns` is typically `['text']`. 

In Direct Preference Optimization (`use_dpo: true`), the pipeline extracts two simultaneous text streams (`chosen` and `rejected`) to compute implicit reward margins:

```text
train_data_columns: ['chosen', 'rejected']
                      │
           ┌──────────┴──────────┐
           ▼                     ▼
    Tokenize Chosen      Tokenize Rejected
           │                     │
           └──────────┬──────────┘
                      ▼
             DPO Pairwise Loss
```

---

## 3. Options & Default

| Parameter | Type | Default | Options |
| :--- | :--- | :--- | :--- |
| `train_data_columns` | `list[str]` | `['text']` | `['text']`, `['chosen', 'rejected']`, `['input_ids']`, or custom feature keys |

---

## 4. Interactions with Related Parameters

- **`use_dpo`**: When enabled, `train_data_columns` must contain the chosen and rejected column identifiers.
- **`eval_data_columns`**: Mirror configuration for evaluation splits.
- **`tokenize_train_data`**: Determines whether the extracted column contents are treated as raw strings or integer token lists.

---

## 5. Practical Scenarios & Failure Modes

| Scenario | Symptom / Failure | Fix |
| :--- | :--- | :--- |
| **Parquet dataset uses column name `"content"` instead of `"text"`** | `KeyError: 'text'` during dataset batch iteration | Set `train_data_columns: ['content']`. |
| **DPO run with default columns** | `KeyError` or mismatch in DPO loss computation | Set `train_data_columns: ['chosen', 'rejected']`. |

---

### One-line intuition

> `train_data_columns` specifies the dataset field names to ingest for training, routing plain text for pretraining or pairwise chosen/rejected streams for preference optimization.
