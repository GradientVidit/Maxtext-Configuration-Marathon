## 1. Why does `add_eos` exist?

Without an explicit End-Of-Sequence (`<eos>`) token, an autoregressive language model has no way of learning when a document or response has concluded, leading to endless generation during inference until context limits are reached.

```text
Raw Tokens:                [ "The", "end", "." ]
                                   │
                    ┌──────────────┴──────────────┐
                    ▼                             ▼
              add_eos: true                 add_eos: false
                    │                             │
          [ "The", "end", ".", <eos> ]      [ "The", "end", "." ]
```

`add_eos` ensures that every document or sample ends with an `<eos>` token, teaching the model the terminating distribution.

---

## 2. Mechanics

During tokenization preprocessing:

```text
Text -> Tokens -> [Optional BOS] + Tokens + [EOS if add_eos else None] -> Truncate/Pack
```

When packing sequences (`packing: true`), `<eos>` indicates to segment boundaries and attention masking where document boundaries occur.

---

## 3. Options & Default

| Parameter | Type | Default | Options |
| :--- | :--- | :--- | :--- |
| `add_eos` | `bool` | `true` | `true` (append EOS token), `false` (do not append) |

---

## 4. Interactions with Related Parameters

- **`add_bos`**: Prepends the BOS token.
- **`packing`**: In packed sequences, `<eos>` acts as the natural delimiter separating distinct examples.
- **`use_truncation`**: If sequences exceed `max_target_length`, truncation happens *after* or *before* EOS appending depending on the trim policy in `TokenizeAndTrim`.

---

## 5. Practical Scenarios & Failure Modes

| Scenario | Symptom / Failure | Fix |
| :--- | :--- | :--- |
| **Model repeats or rambles endlessly during inference** | Training data lacked `<eos>` delimiters; model never learned stopping probability | Ensure `add_eos: true` during training and fine-tuning. |
| **Pre-tokenized dataset already contains EOS** | Double `<eos>` tokens (`<eos> <eos>`) inserted at document ends | Set `add_eos: false`. |

---

### One-line intuition

> `add_eos` appends the End-Of-Sequence token to training examples, teaching the model when to terminate generation.
