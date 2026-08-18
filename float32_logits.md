
## 1. Why does `float32_logits` exist?

After computing `Q @ K^T`, the raw attention logits (pre-softmax scores) get passed directly into `softmax`. The softmax function is:

```text
softmax(z_i) = exp(z_i) / Σ_j exp(z_j)
```

This is exponentially sensitive to the values of `z_i`. In bf16:
- Representable values have ~3 decimal digits of precision
- Two logits that differ by 0.001 in fp32 may appear identical in bf16
- The exponentiation amplifies this difference: `exp(5.001)` vs `exp(5.000)` is a 0.1% difference in the softmax output

`float32_logits: true` casts the attention logits (the `Q @ K^T / sqrt(d)` result) to fp32 **before** the softmax, giving the exponential computation full precision.

---

## 2. Default

```yaml
float32_logits: false
```

Off by default. For most training runs with standard head_dim and sequence lengths, the precision difference is negligible. Flash attention implementations typically handle this internally.

---

## 3. Placement in the attention computation

```text
Q (bf16) @ K^T (bf16)      ← float32_qk_product handles this step
    ↓
raw_scores (bf16)
    / sqrt(head_dim)
    [+ causal mask]
    [+ position bias]
    ↓
logits (bf16 or fp32)      ← float32_logits casts HERE (before softmax)
    ↓
softmax(logits)
    ↓
context = softmax_out @ V
```

---

## 4. Difference from `float32_qk_product`

| Flag | What it casts | Where |
|---|---|---|
| `float32_qk_product` | Q, K inputs | Before `QK^T` matmul |
| `float32_logits` | Scaled scores | Before softmax |

Both address attention precision but at different points. `float32_logits` is the more critical one for softmax stability — even if the `QK^T` matmul ran in bf16, casting the result to fp32 before softmax still helps significantly.

---

## 5. Interaction with flash attention

Flash attention computes the softmax internally with fp32 accumulators regardless of input/output dtype. When `attention: "flash"` is selected, `float32_logits` is largely redundant — the kernel already handles this.

For `attention: "dot_product"` (the JAX reference implementation), `float32_logits` matters.

---

## 6. Combined with `cast_logits_to_fp32` for output logits

Don't confuse these:
- `float32_logits` → precision of **attention logits** (inside attention, before softmax)
- `cast_logits_to_fp32` → precision of **output logits** (at model output, before cross-entropy)

These are completely different "logits" in different parts of the model.

---

### One-line intuition

> **`float32_logits` casts the pre-softmax attention scores to fp32, preventing bf16 rounding from distorting the exponential sensitivity of softmax — most relevant when not using flash attention.**
