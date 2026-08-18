
## 1. Why does `normalize_embedding_logits` exist?

When `logits_via_embedding: true`, the output logits are computed as:

```text
logits = hidden_state @ E^T
```

where `E` is the embedding matrix (shared with input embeddings). The magnitude of these logits depends on:
1. The norm of `hidden_state` (grows throughout training)
2. The norm of embedding vectors in `E` (also evolves during training)

Both can grow large, making logits large, making the softmax output sharply peaked (effectively "overconfident"). This is a numerical instability risk — especially in bf16, where large values have poor precision.

`normalize_embedding_logits` inserts a normalization step before the final softmax to counteract this.

---

## 2. Default

```yaml
normalize_embedding_logits: true
```

On by default — but **only has any effect when `logits_via_embedding: true`**. With `logits_via_embedding: false` (separate unembedding matrix), this flag is a no-op.

---

## 3. How it works

With `normalize_embedding_logits: true`:

```text
raw_logits = hidden_state @ E^T
scaled_logits = raw_logits / sqrt(emb_dim)
softmax(scaled_logits) → probabilities
```

The scaling by `1/sqrt(emb_dim)` is the same scaling used in attention (dot-product scaling). It prevents the dot products from growing proportional to embedding dimension, which would cause the softmax to become increasingly peaked as the model trains.

---

## 4. Interaction with `logits_dot_in_fp32`

Both flags protect the same logit computation at different precision levels:

```text
normalize_embedding_logits: true
    → scale the logits (prevent magnitude issues)

logits_dot_in_fp32: true
    → compute the dot product in fp32 (prevent precision issues in bf16)
```

For tied embedding models, using both is the safest option:

```yaml
logits_via_embedding: true
normalize_embedding_logits: true
logits_dot_in_fp32: true
```

---

## 5. When to disable

The only reason to set `normalize_embedding_logits: false` is when:
- Replicating a specific model's architecture that uses tied embeddings *without* normalization (rare)
- You're adding your own custom scaling externally

For any new model using tied embeddings, leave this `true`.

---

## 6. The dependency chain

```text
logits_via_embedding: false  →  normalize_embedding_logits is a no-op
logits_via_embedding: true   →  normalize_embedding_logits: true  (recommended)
                              →  normalize_embedding_logits: false (risky — large logit scale)
```

Setting `normalize_embedding_logits: true` when `logits_via_embedding: false` does nothing — there's no embedding dot product to normalize.

---

### One-line intuition

> **`normalize_embedding_logits` scales down the tied-embedding logits by `1/sqrt(emb_dim)` to prevent overconfident softmax from unnormalized embedding magnitudes — only meaningful when `logits_via_embedding: true`.**
