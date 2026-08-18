
## 1. The problem: two separate weight matrices vs. one

A standard language model has two distinct large weight matrices:

```text
Input side:   E ∈ ℝ^{vocab × emb_dim}    (token embedding lookup)
Output side:  U ∈ ℝ^{emb_dim × vocab}    (unembedding / logit projection)
```

These two matrices are both `vocab × emb_dim`. For a 50k vocab and 4096 embedding dimension:

```text
E: 50,000 × 4,096 × 4 bytes = 800 MB
U: 50,000 × 4,096 × 4 bytes = 800 MB
Total: 1.6 GB just for embedding/unembedding
```

**Weight tying** (a.k.a. input-output embedding tying) reuses the same matrix for both. This:
- Halves memory for these two layers (~800 MB saved)
- Enforces that similar words have similar embedding and output representations
- Reduces the parameter count by `vocab × emb_dim`

`logits_via_embedding: true` enables this weight tying.

---

## 2. Default

```yaml
logits_via_embedding: false
```

Off by default — separate embedding and unembedding matrices. For large vocab + large model, tying saves significant memory. Whether to enable it depends on the architecture being replicated (many published models use tying, some don't).

---

## 3. How it works mechanically

```text
logits_via_embedding: false (default):
    token_ids → E[token_ids] → [batch, seq, emb_dim]  (embedding lookup)
    hidden states → U × hidden → [batch, seq, vocab]  (separate matrix)

logits_via_embedding: true:
    token_ids → E[token_ids] → [batch, seq, emb_dim]  (embedding lookup)
    hidden states → E^T × hidden → [batch, seq, vocab] (same matrix, transposed)
```

With tying, the same matrix `E` is used in both directions — once as a lookup (input) and once as a projection (output, transposed).

---

## 4. The normalization twist

When using tied embeddings, the scale of the embedding vectors matters for both the input and the output logits. If the embeddings are large, the logits will also be large (since it's a dot product with the same matrix). This can cause problems.

`normalize_embedding_logits` addresses this:

```yaml
logits_via_embedding: true
normalize_embedding_logits: true  # normalize before softmax
```

The normalization prevents the logit scale from being dominated by embedding magnitude rather than actual token predictions.

---

## 5. Model families that use tying

| Model family | Tied embeddings? |
|---|---|
| GPT-2 | Yes |
| Original PaLM | Yes |
| LLaMA 2 | No (separate) |
| LLaMA 3 | No |
| Gemma | Yes |
| T5 | Yes |

Check the model's paper or config before assuming either default is correct.

---

## 6. Memory impact calculation

For LLaMA 3 8B: `vocab=128k`, `emb_dim=4096`, bf16:
```text
Saved by tying = 128,000 × 4,096 × 2 bytes ≈ 1 GB
```
For models with large vocabularies, this is meaningful. For small vocab models (~32k), the savings are smaller but still nonzero.

---

## 7. Interaction with quantization

If `weight_dtype: "bfloat16"` and `logits_via_embedding: true`, the tied matrix is in bf16 for both lookup and output projection. Quantizing this matrix affects both pathways simultaneously — double the impact of quantization errors.

---

### One-line intuition

> **`logits_via_embedding` reuses the input token embedding matrix as the output logit projection — halving memory for these two large matrices, but requiring `normalize_embedding_logits: true` to prevent logit scale issues.**
