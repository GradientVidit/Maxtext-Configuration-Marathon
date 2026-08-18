
## 1. The problem: bf16 precision in large dot products

The final logit computation is a large dot product:

```text
logits = hidden_state @ unembedding_matrix^T
       [batch, seq, emb_dim] × [emb_dim, vocab]
       → [batch, seq, vocab]
```

In bf16, the mantissa has only 7 bits (~2.3 decimal digits of precision). For a dot product summing `emb_dim=4096` products, accumulated rounding errors in bf16 can cause meaningful numerical differences in the logit values — which cascade through the softmax into probability and loss errors.

`logits_dot_in_fp32: true` casts the inputs to fp32 before this matmul, so the accumulation happens with full 23-bit mantissa precision.

---

## 2. Default

```yaml
logits_dot_in_fp32: false
```

Off by default. The difference is typically small enough that it doesn't affect the loss curve meaningfully for most training runs.

---

## 3. What exactly gets cast

```text
logits_dot_in_fp32: false (default):
    hidden (bf16) @ unembedding (bf16) → logits (bf16)

logits_dot_in_fp32: true:
    hidden.astype(fp32) @ unembedding.astype(fp32) → logits (fp32)
    then used downstream in loss computation
```

This is a **local** cast — only the inputs to this specific matmul. The rest of the model continues in bf16.

---

## 4. Relationship to `cast_logits_to_fp32`

These are complementary but target different things:

```text
logits_dot_in_fp32: true
    → Cast inputs BEFORE the logit matmul (higher precision accumulation)
    → Prevents rounding errors during the dot product

cast_logits_to_fp32: true (default: true)
    → Cast result AFTER the logit matmul to fp32
    → Ensures downstream operations (softmax, cross-entropy) use fp32
```

For maximum stability:
```yaml
logits_dot_in_fp32: true
cast_logits_to_fp32: true
```

For production default (safe, slightly less precise):
```yaml
logits_dot_in_fp32: false
cast_logits_to_fp32: true
```

---

## 5. Cost

Casting to fp32 before the logit matmul increases memory for these tensors and can slow down the matmul if the hardware doesn't natively accelerate fp32 matmuls as fast as bf16. On TPUs, bf16 matmuls are significantly faster than fp32 — so `logits_dot_in_fp32: true` can noticeably increase step time for large vocab models.

---

## 6. When to enable

- Reproducing a published model's exact perplexity numbers
- Debugging numerical divergence between expected and actual loss
- High-precision evaluation where small logit differences matter
- Models with very large vocabulary (>100k tokens) where logit scale issues are amplified

---

### One-line intuition

> **`logits_dot_in_fp32` casts the hidden state and unembedding weights to fp32 before the final logit matmul, preventing bf16 accumulation errors in a computation where precision directly affects loss — at the cost of slower matmul on TPUs.**
