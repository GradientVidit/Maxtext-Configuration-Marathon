## 1. Why does `eval_data_columns` exist?

Evaluation datasets often originate from standard benchmarks (e.g., MMLU, GSM8k, MT-Bench) that use different column naming schemas than the primary training corpus.

```text
Training Schema:    { "text": "..." }           ──> train_data_columns: ['text']
Eval Schema (MMLU): { "question_prompt": "..." } ──> eval_data_columns: ['question_prompt']
```

`eval_data_columns` independently configures which fields to extract during validation passes.

---

## 2. Mechanics

The evaluation pipeline extracts the keys listed in `eval_data_columns`, passes them through the eval tokenizer (`tokenize_eval_data`), and feeds the token tensors to the validation step.

```text
Eval Dataset ──> Extract [eval_data_columns] ──> Eval Tokenizer ──> Validation Forward Pass
```

---

## 3. Options & Default

| Parameter | Type | Default | Options |
| :--- | :--- | :--- | :--- |
| `eval_data_columns` | `list[str]` | `['text']` | List of column strings (e.g. `['text']`, `['prompt', 'response']`) |

---

## 4. Interactions with Related Parameters

- **`train_data_columns`**: Training counterpart.
- **`tokenize_eval_data`**: Dictates whether evaluation columns require runtime tokenization.
- **`eval_per_device_batch_size`**: Controls the batch volume fed to the evaluation loop.

---

## 5. Practical Scenarios & Failure Modes

| Scenario | Symptom / Failure | Fix |
| :--- | :--- | :--- |
| **Evaluation benchmark uses `"query"` field** | `KeyError: 'text'` during periodic evaluation step | Set `eval_data_columns: ['query']`. |
| **DPO Evaluation** | DPO eval requires evaluating both chosen and rejected completions | Set `eval_data_columns: ['chosen', 'rejected']`. |

---

### One-line intuition

> `eval_data_columns` specifies which dataset columns to read during evaluation, decoupling validation schemas from training data formatting.
