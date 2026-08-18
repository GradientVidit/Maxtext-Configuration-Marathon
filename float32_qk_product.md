
## 1. Why does `float32_qk_product` exist?

Dot-product attention computes:

```text
attention_scores = Q @ K^T / sqrt(head_dim)
                   ─────────────────────────
                   [seq, head_dim] × [head_dim, seq] → [seq, seq]
```

In bf16, the matmul accumulates `head_dim` products. For `head_dim=128`, that's 128 multiply-accumulate operations. The accumulated rounding error in bf16 can shift the attention scores by a small but non-trivial amount.

This matters particularly because:
- The softmax is exponentially sensitive to score differences
- Tokens with close scores in float32 may flip their relative ordering in bf16
- With long sequences (seq_len=8192+), the softmax denominator becomes large and precise per-element scores matter more

`float32_qk_product: true` casts Q and K to fp32 before the `QK^T` matmul, making the attention score computation higher precision.

---

## 2. Default

```yaml
float32_qk_product: false
```

Off by default. The precision difference is small enough for most training runs to ignore. Flash attention implementations handle this internally by maintaining fp32 accumulators, so when using flash attention, this flag may be redundant.

---

## 3. What exactly gets cast

```text
float32_qk_product: false:
    Q (bf16) @ K^T (bf16) → scores (bf16)

float32_qk_product: true:
    Q.astype(fp32) @ K^T.astype(fp32) → scores (fp32) → rescale → softmax
```

Only Q and K are cast — V (value) is unaffected since it's not part of the score computation.

---

## 4. Interaction with `float32_logits`

These are two separate precision checkpoints in the attention computation:

```text
Q @ K^T          ← float32_qk_product controls this matmul's precision
    ↓
/ sqrt(head_dim)
    ↓
[+ mask, + bias]
    ↓
softmax input    ← float32_logits controls precision here (before softmax)
    ↓
softmax(scores) @ V  ← value aggregation (unaffected by either flag)
```

`float32_qk_product` catches precision at the matmul.  
`float32_logits` catches precision at the softmax input.  
They're complementary; for maximum attention precision, set both `true`.

---

## 5. Flash attention overlap

When `attention: "flash"` or `attention: "autoselected"` (which may pick flash), the flash attention kernel maintains internal fp32 accumulators regardless of input dtype. In that case, `float32_qk_product` is effectively a no-op — flash attention already handles this.

For `attention: "dot_product"` (the reference implementation), `float32_qk_product` does meaningful work.

---

## 6. When to enable

- Debugging attention: verifying that patterns seen in bf16 aren't artifacts of precision
- Long-context training (>16k tokens) where score precision matters more
- Matching a paper that specifically describes fp32 attention scores
- Using dot_product attention (not flash) in high-precision mode

---

### One-line intuition

> **`float32_qk_product` casts Q and K to fp32 before the attention score matmul, preventing bf16 accumulation errors from distorting relative attention weights — most relevant when using dot-product (non-flash) attention.**
