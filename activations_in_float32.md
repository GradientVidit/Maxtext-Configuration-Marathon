
## 1. Why does it exist?

Nonlinearities like SiLU (Swish) and GELU are mathematically smooth but numerically picky at specific input ranges. In bfloat16, the 7-bit mantissa means values that differ only in the 8th+ significant bit are indistinguishable — they round to the same representable value.

For a nonlinearity applied to nearly-zero inputs, this rounding causes the gradient through the nonlinearity to be wrong in ways that accumulate subtly across layers and training steps.

```text
before SiLU:   x = 0.00312 (bfloat16) → rounds to 0.003125
               x = 0.00314             → also rounds to 0.003125
               → same gradient, different actual inputs = precision loss

activations_in_float32 = true:
               x = 0.00312 (float32) → preserved as 0.00312
               x = 0.00314           → preserved as 0.00314
               → correct, distinct gradients
```

`activations_in_float32` casts the activation tensor to fp32 immediately before the nonlinearity, then casts back to `dtype` after. It's a targeted precision injection, not a full switch to fp32.

---

## 2. What it actually does in code

When `activations_in_float32=true`:
1. Activation tensor (in `dtype = bfloat16`) is cast to `float32`
2. SiLU/GELU/etc. is applied in float32
3. Result is cast back to bfloat16

```text
MLP hidden state (bf16)
         ↓
    cast to fp32    ← activations_in_float32
         ↓
  apply nonlinearity (fp32)
         ↓
    cast back to bf16
         ↓
  continue in bf16
```

The overhead is two dtype casts per MLP layer — minimal compute cost.

---

## 3. Options

| Value | Behavior |
|---|---|
| `false` | Nonlinearities applied in `dtype` (bfloat16 by default) — default |
| `true` | Activations cast to fp32 before each nonlinearity, then cast back |

Default in base.yml:
```yaml
activations_in_float32: false
```

---

## 4. When to use it

This is a precision debugging and stability tool, not a standard training knob.

**Use it when:**
- Training instability (loss spikes) persists and you suspect bfloat16 precision in the MLP nonlinearity
- Training a model with very deep MLPs or unusual activation functions where rounding near zero matters more
- Ablating whether bf16 nonlinearities are hurting model quality

**Don't use it by default** — the overhead is small but nonzero, and modern transformer architectures (LLaMA, Qwen, etc.) train stably with bf16 SiLU without this flag.

---

## 5. Interaction with `dtype`

This only matters when `dtype != "float32"`. If `dtype="float32"`, everything is already in fp32 — this flag does nothing meaningful.

```text
dtype=bfloat16, activations_in_float32=false → nonlinearities in bf16
dtype=bfloat16, activations_in_float32=true  → nonlinearities in fp32
dtype=float32,  activations_in_float32=true  → same (no-op, already fp32)
```

---

## 6. Related point-precision overrides

| Param | Where it applies |
|---|---|
| `activations_in_float32` | Before nonlinearities (MLP) |
| `float32_qk_product` | QK dot-product in attention |
| `float32_logits` | Inputs to attention softmax |
| `cast_logits_to_fp32` | Final output logits |

These are all surgical fp32 injections into an otherwise-bfloat16 forward pass. Each targets a different numerically-sensitive operation.

---

### One-line intuition

> **`activations_in_float32=true` casts activations to fp32 immediately before each nonlinearity and back after — a surgical precision boost for the specific point where bfloat16's limited mantissa is most likely to hurt gradient correctness.**
