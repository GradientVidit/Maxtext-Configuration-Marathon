## 1. Why does `tokenize_train_data` exist?

Training datasets exist in two distinct formats:
1. **Raw text**: JSONL, Parquet, or TFDS containing raw Unicode strings (e.g. `{"text": "Once upon a time..."}`).
2. **Pre-tokenized data**: Binary files (e.g. ArrayRecord, TFRecord, NumPy memmap) containing fixed integer arrays of token IDs.

```text
                  Training Dataset Pipeline
                              │
              Is data raw text or pre-tokenized?
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
  tokenize_train_data: true       tokenize_train_data: false
              │                               │
       Raw Text Strings               Pre-tokenized IDs
              │                               │
       Run Tokenizer                   Skip Tokenizer
              │                               │
              └───────────────┬───────────────┘
                              ▼
                     Tensor Batches [B, S]
```

Tokenizing raw text on-the-fly consumes host CPU and memory bandwidth. For large-scale pretraining (hundreds of billions of tokens), datasets are usually pre-tokenized offline to save CPU overhead. `tokenize_train_data` toggles runtime tokenization on or off for the training split.

---

## 2. Mechanics

```text
Pipeline flow when tokenize_train_data: true:
  Dataset -> Extract 'text' -> Tokenize (SentencePiece/HF) -> Pack/Pad -> Device

Pipeline flow when tokenize_train_data: false:
  Dataset -> Extract 'tokens' (int array) -> Pack/Pad -> Device
```

When set to `false`, MaxText bypasses all subword encoding operations and expects integer sequences directly from the dataset record.

---

## 3. Options & Default

| Parameter | Type | Default | Options |
| :--- | :--- | :--- | :--- |
| `tokenize_train_data` | `bool` | `true` | `true` (tokenize raw text), `false` (data is already tokenized) |

---

## 4. Interactions with Related Parameters

- **`tokenizer_path` / `tokenizer_type`**: Required when `tokenize_train_data: true`; ignored when `false`.
- **`train_data_columns`**: When `tokenize_train_data: true`, points to string columns (e.g. `['text']`); when `false`, points to integer token array columns (e.g. `['input_ids']` or `['tokens']`).
- **`tokenize_eval_data`**: Independent toggle for evaluation datasets.

---

## 5. Practical Scenarios & Failure Modes

| Scenario | Symptom / Failure | Fix |
| :--- | :--- | :--- |
| **Dataset is pre-tokenized ints, but `tokenize_train_data: true`** | Tokenizer throws `TypeError: Expected string, got int/list` | Set `tokenize_train_data: false`. |
| **Dataset is raw strings, but `tokenize_train_data: false`** | XLA or Tensor shape error: string dtype passed to JAX device array | Set `tokenize_train_data: true`. |
| **Pre-tokenized data with different vocab** | Model trains but loss does not converge (token ID mismatch) | Ensure offline pre-tokenization used the identical tokenizer model. |

---

### One-line intuition

> `tokenize_train_data` tells MaxText whether to tokenize raw text on-the-fly or load pre-tokenized integer arrays directly, eliminating CPU bottlenecking during large pretraining runs.
