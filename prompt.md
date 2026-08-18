## 1. Why does `prompt` exist?

When testing, validating, or running interactive generation scripts (such as `decode.py`), MaxText needs an initial input string to initiate autoregressive token prediction:

```text
prompt: "I love to"
       │
       ▼ (Tokenizer)
Token IDs: [ 40, 1849, 284 ]
       │
       ▼ (Prefill & Autoregressive Decoding)
Generated Output: "I love to explore high-performance distributed TPU systems..."
```

`prompt` specifies the input string passed to the tokenizer to start decoding.

---

## 2. What it actually controls

```yaml
prompt: "I love to"
```

- Provides the raw text string for sampling in `decode.py`.
- Used in integration tests to verify model convergence and generative output quality across checkpoints.

---

## 3. Options and Examples

```yaml
prompt: "I love to"                                      # Default smoke-test prompt
prompt: "Translate English to German:\nHello, world!"    # Task-specific prompt
prompt: "Write a Python script that calculates primes:" # Code-generation prompt
```

---

## 4. Interactions

- **`tokenizer_path` & `tokenizer_type`**: The string is tokenized using the configured tokenizer.
- **`max_prefill_predict_length`**: The number of tokens produced by tokenizing `prompt` must not exceed `max_prefill_predict_length`.

---

## 5. Practical Scenarios

- **Testing checkpoint quality**: Pass custom prompts via CLI:
```bash
python maxtext/decode.py maxtext/configs/base.yml prompt="The future of artificial intelligence is"
```

---

### One-line intuition

> **`prompt` sets the initial input text string for autoregressive language generation in `decode.py`.**
