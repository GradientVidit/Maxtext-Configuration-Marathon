## 1. Why does `add_bos` exist?

Autoregressive causal language models require a canonical starting point to condition generation and establish attention boundaries. Many architectures (e.g. Llama, Gemma, Mistral) expect every independent sequence to begin with a special Beginning-Of-Sequence (`<bos>`) token (typically token ID 1 or 2).

```text
Raw Tokens:                [ "The", "cat", "sat" ]
                                   │
                    ┌──────────────┴──────────────┐
                    ▼                             ▼
              add_bos: true                 add_bos: false
                    │                             │
          [ <bos>, "The", "cat", "sat" ]    [ "The", "cat", "sat" ]
```

`add_bos` specifies whether the input pipeline automatically prepends the `<bos>` token to tokenized sequences.

---

## 2. Mechanics

In Grain and TFDS tokenization transformations (`_grain_tokenizer.py` and `_tfds_data_processing.py`):

```text
Sequence Processing:
  tokens = tokenizer.encode(text)
  if add_bos:
      tokens = [bos_id] + tokens
  if add_eos:
      tokens = tokens + [eos_id]
```

When sequence packing is enabled (`packing: true`), `<bos>` tokens mark the start of each individual segment within a packed sequence, signaling document boundaries to causal attention masks.

---

## 3. Options & Default

| Parameter | Type | Default | Options |
| :--- | :--- | :--- | :--- |
| `add_bos` | `bool` | `true` | `true` (prepend BOS token), `false` (do not prepend) |

---

## 4. Interactions with Related Parameters

- **`add_eos`**: Appends the corresponding End-Of-Sequence token.
- **`packing`**: When packing multiple documents into a single sequence of length `max_target_length`, each segment gets its own BOS token if `add_bos: true`.
- **`tokenizer_type`**: The BOS token ID is retrieved directly from the configured tokenizer backend.

---

## 5. Practical Scenarios & Failure Modes

| Scenario | Symptom / Failure | Fix |
| :--- | :--- | :--- |
| **Fine-tuning a model trained without BOS (e.g. GPT-2 / GPT-J)** | Distribution shift; unnecessary BOS tokens pollute context | Set `add_bos: false`. |
| **Tokenizer already automatically inserts BOS (HF tokenizers default)** | Double BOS tokens prepended (`<bos> <bos> Once upon...`) | Set `add_bos: false` or configure tokenizer `add_special_tokens=False`. |

---

### One-line intuition

> `add_bos` determines whether the Beginning-Of-Sequence (`<bos>`) token is prepended to input token streams to mark document boundaries.
