
## 1. Why does `cast_logits_to_fp32` exist?

The final stage of a language model forward pass is:

```text
hidden_state (bf16)
    ↓
logit projection → raw logits (bf16)
    ↓
softmax + cross-entropy loss
```

The softmax and cross-entropy loss are **numerically sensitive operations**:
- Softmax: `exp(logit_i) / sum(exp(logit_j))` — exponentials amplify small errors
- Cross-entropy: `−log(softmax_output)` — logarithm amplifies errors near 0

When these operations run in bf16 (7-bit mantissa), small errors in logit values get amplified. Casting logits to fp32 before these operations gives the loss computation full precision where it matters most.

---

## 2. Default

```yaml
cast_logits_to_fp32: true
```

**On by default** — this is one of MaxText's safety defaults. The comment in base.yml says: "the higher precision is generally beneficial, but it can vary slightly."

---

## 3. What it controls

```text
cast_logits_to_fp32: true:
    logits (bf16) → logits.astype(fp32) → softmax/cross-entropy in fp32

cast_logits_to_fp32: false:
    logits (bf16) → softmax/cross-entropy in bf16 → higher rounding risk
```

The cast happens **after** the logit matmul, **before** the loss computation.

---

## 4. Cost and benefit

**Benefit:** More numerically stable loss computation. Important especially for:
- Large vocabulary (≥32k tokens) where the softmax denominator sums many terms
- Low-probability targets (rare tokens) where `log(p)` is very negative
- Long training runs where accumulated numerical errors matter

**Cost:**
- The fp32 logit tensor uses 2× the memory vs. bf16 logits
- For large vocab (128k tokens), this is `batch × seq × 128k × 4 bytes` of extra memory per step

---

## 5. Relationship to `logits_dot_in_fp32`

```text
logits_dot_in_fp32  →  affects the matmul accumulation (input side)
cast_logits_to_fp32 →  affects softmax/loss computation (output side)
```

Having both `true` covers both sides of the logit choke point. Having only `cast_logits_to_fp32: true` (the default) covers the most impactful side (the loss function).

---

## 6. When to disable

The only reason to set `cast_logits_to_fp32: false`:
- Memory-critical training where you cannot afford the extra logit memory
- Experiments specifically testing the impact of lower-precision loss computation
- Matching a model implementation that intentionally uses bf16 throughout

In practice, the memory cost is often acceptable and the stability benefit is real — leave it `true`.

---

### One-line intuition

> **`cast_logits_to_fp32` ensures the softmax and cross-entropy loss are computed in fp32 precision — the default `true` is correct for virtually all training runs since the loss function is where numerical errors hurt most.**
