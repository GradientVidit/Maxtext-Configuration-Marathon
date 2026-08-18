## 1. Why does `max_prefill_predict_length` exist?

Autoregressive inference in `decode.py` operates in two distinct phases:

1. **Prefill Phase (Prompt Ingestion)**: Processes the prompt tokens in parallel to populate the Key-Value (KV) cache.
2. **Decode Phase (Token Generation)**: Generates one token at a time autoregressively using cached KV representations.

```text
Inference Execution:
Prompt Tokens: [ p_0, p_1, ..., p_K ] (Length K ≤ max_prefill_predict_length)
       │
       ▼ (Prefill: Parallel MatMul)
Populate KV Cache up to index K
       │
       ▼ (Decode: Sequential Step-by-Step)
Step K+1 ──> Generate token K+1 ──> Append to KV cache
Step K+2 ──> Generate token K+2 ──> Append to KV cache
... up to max_target_length
```

Because XLA requires static shapes, the prefill compilation must know the maximum prompt length to compile the prefill graph without dynamic shape recompilations.

`max_prefill_predict_length` defines the static maximum prompt token capacity for the prefill stage.

---

## 2. What it actually controls

```yaml
max_prefill_predict_length: 64
```

- Sets the buffer size allocated for prompt processing during decoding.
- Prompts shorter than `max_prefill_predict_length` are padded; prompts longer than this threshold must be truncated or rejected.

---

## 3. Options and Defaults

| Value | Meaning | Typical Usage |
|---|---|---|
| `64` (default) | Compact prompt capacity | Short interactive testing, benchmark smoke tests |
| `512` – `2048` | Medium prompt capacity | Standard question-answering, dialogue prompts |
| `8192` – `32768` | Long prompt capacity | Long-document summarization, RAG prefill |

---

## 4. Interactions and Constraints

- **Constraint**: `max_prefill_predict_length <= max_target_length`.
- **Generation Budget**: The maximum number of new tokens generated during autoregression is $    ext{max\_target\_length} -     ext{max\_prefill\_predict\_length}$.

---

## 5. Practical Scenarios

- **Interactive Sampling (`decode.py`)**: If testing with a prompt longer than 64 tokens (e.g. 200 tokens), increase `max_prefill_predict_length: 256` or `512`. Otherwise, prompt tokens beyond index 64 are truncated.

---

### One-line intuition

> **`max_prefill_predict_length` sets the maximum prompt length processed in parallel during the prefill phase of autoregressive decoding.**
