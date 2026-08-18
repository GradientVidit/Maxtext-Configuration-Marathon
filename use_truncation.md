## 1. Why does `use_truncation` exist?

When a raw document is longer than the model's context window (`max_target_length`), the input pipeline faces a fundamental architectural choice:
1. **Truncation (`TokenizeAndTrim`)**: Discard all tokens beyond `max_target_length`. Simple, but throws away trailing text.
2. **Chunking (`TokenizeAndChunk`)**: Split the long document into multiple consecutive chunks of `max_target_length`, treating each chunk as a separate training sample.

```text
Document: [ Token 1 ............................ Token 8192 ] (max_target_length = 2048)
                                    │
            ┌───────────────────────┴───────────────────────┐
            ▼                                               ▼
  use_truncation: true                            use_truncation: false
(TokenizeAndTrim)                               (TokenizeAndChunk)
            │                                               │
  [ Token 1 .. 2048 ]                             [ Chunk 1: 1 .. 2048 ]
  (Tokens 2049 .. 8192 discarded)                 [ Chunk 2: 2049 .. 4096 ]
                                                  [ Chunk 3: 4097 .. 6144 ]
                                                  [ Chunk 4: 6145 .. 8192 ]
```

`use_truncation` toggles between trimming long documents versus chunking them into multiple training examples.

---

## 2. Mechanics in Grain Pretraining

In `src/maxtext/input_pipeline/_grain_tokenizer.py`:
- When `use_truncation: true`, MaxText applies `TokenizeAndTrim`:
  ```python
  tokens = tokens[:max_target_length]
  ```
- When `use_truncation: false`, MaxText applies `TokenizeAndChunk`:
  ```python
  chunks = [tokens[i : i + max_target_length] for i in range(0, len(tokens), max_target_length)]
  ```

> [!NOTE]
> `use_truncation: false` (chunking) is currently implemented only in Grain's pretraining preprocessing pipeline.

---

## 3. Options & Default

| Parameter | Type | Default | Options |
| :--- | :--- | :--- | :--- |
| `use_truncation` | `bool` | `true` | `true` (trim to max length), `false` (chunk into multiple sequences) |

---

## 4. Interactions with Related Parameters

- **`max_target_length`**: Defines the chunk / truncation cutoff length.
- **`dataset_type`**: Full chunking support (`use_truncation: false`) requires `dataset_type: grain`.
- **`packing`**: If `packing: true`, chunked segments can be packed alongside other shorter sequences to fill batches efficiently.

---

## 5. Practical Scenarios & Failure Modes

| Scenario | Symptom / Failure | Fix |
| :--- | :--- | :--- |
| **Large-scale book/corpus pretraining** | High data waste when long 20k-token documents are trimmed at 4k tokens | Set `use_truncation: false` with `dataset_type: grain` to ingest full corpus. |
| **Instruction fine-tuning (SFT)** | Chunking cuts question-answer pairs arbitrarily in half, breaking dialogue semantics | Keep `use_truncation: true` for SFT datasets. |

---

### One-line intuition

> `use_truncation` controls whether long documents exceeding the context length are clipped and discarded (`true`) or split into consecutive training chunks (`false`).
