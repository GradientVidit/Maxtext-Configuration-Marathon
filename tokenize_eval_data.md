## 1. Why does `tokenize_eval_data` exist?

Evaluation datasets often have different formatting or provenance than training datasets. For example, a training dataset might be pre-tokenized into ArrayRecord binary shards for maximum throughput, while an eval dataset is loaded directly as raw JSONL text or from HuggingFace Hub.

```text
Training Stream: Pre-tokenized Binaries ──> tokenize_train_data: false
Eval Stream:     Raw Text Prompts      ──> tokenize_eval_data: true
```

`tokenize_eval_data` provides an independent on/off switch for runtime tokenization of evaluation splits.

---

## 2. Mechanics

```text
Eval Loop:
  if tokenize_eval_data:
      raw_text -> tokenizer.encode() -> batch -> eval_step()
  else:
      int_tokens -> batch -> eval_step()
```

This prevents coupling eval dataset requirements to the offline preparation state of the training dataset.

---

## 3. Options & Default

| Parameter | Type | Default | Options |
| :--- | :--- | :--- | :--- |
| `tokenize_eval_data` | `bool` | `true` | `true`, `false` |

---

## 4. Interactions with Related Parameters

- **`tokenize_train_data`**: The training equivalent.
- **`eval_data_columns`**: Specifies the column name(s) in the evaluation dataset to be read.
- **`eval_dataset_name` / `hf_eval_files` / `grain_eval_files`**: Data source for evaluation batches.

---

## 5. Practical Scenarios & Failure Modes

| Scenario | Symptom / Failure | Fix |
| :--- | :--- | :--- |
| **Pre-tokenized eval dataset passed with `tokenize_eval_data: true`** | Eval step fails with string expectation error | Set `tokenize_eval_data: false`. |
| **Raw eval strings with `tokenize_eval_data: false`** | Eval crashes due to string tensors fed to JAX compute | Set `tokenize_eval_data: true`. |

---

### One-line intuition

> `tokenize_eval_data` independently controls whether evaluation data is tokenized dynamically from raw text or ingested as pre-tokenized integer arrays.
