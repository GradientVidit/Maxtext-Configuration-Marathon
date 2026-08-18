## 1. Why does `decode_sampling_top_k` exist?

Softmax distributions over large vocabularies ($V \ge 128{,}000$) often assign non-zero probabilities to thousands of irrelevant or ungrammatical tail tokens. Pure random sampling from the full vocabulary risks picking nonsensical tokens from the long tail.

**Top-$K$ Sampling** truncates the vocabulary distribution to only the $k$ highest-scoring tokens, discarding the remaining $V - k$ tokens before sampling:

```text
Vocabulary (V = 128,000)
             │
             ▼ (Rank Logits)
Top-K Candidates: [ tok_1, tok_2, ..., tok_K ] (k = 40)
             │
             ▼ (Mask out all remaining 127,960 tokens)
Sample strictly from Top-40 Tokens
```

`decode_sampling_top_k` sets the maximum candidate count $k$.

---

## 2. What it actually controls

```yaml
decode_sampling_top_k: 0
```

- When `0` (default): Top-$K$ filtering is disabled.
- When `> 0` (e.g. `40`, `50`): MaxText extracts the top-$k$ logits using `jax.lax.top_k`, sets all other logits to $-\infty$, and renormalizes the distribution.

---

## 3. Options and Defaults

| Value | Meaning | Typical Usage |
|---|---|---|
| `0` (default) | Top-$K$ filtering disabled | Greedy decoding or pure nucleus sampling |
| `1` | Single highest token selected | Mathematically equivalent to greedy argmax |
| `40` – `50` | Top 40–50 tokens retained | Standard sweet spot for balanced diversity |
| `100` – `200` | Broad candidate set | High-entropy creative text generation |

---

## 4. Interactions and Composite Stacking

- **`decode_sampling_strategy`**: Active when set to `"topk"` or `"composite"`.
- **In `"composite"` mode**: MaxText first applies Top-$K$ truncation, then applies Top-$P$ nucleus filtering on the remaining $k$ tokens, and finally samples with `temperature`.

```text
Composite Pipeline:
Logits ──> [ Top-K Filter (k=40) ] ──> [ Top-P Filter (p=0.95) ] ──> [ Temp Sample (T=0.7) ]
```

---

## 5. Practical Scenarios

- **Eliminating Tail Hallucinations**: Set `decode_sampling_top_k: 40` with `decode_sampling_strategy: "topk"`.

---

### One-line intuition

> **`decode_sampling_top_k` restricts the decoding candidate pool to the $k$ highest-probability tokens, filtering out improbable tail tokens.**
