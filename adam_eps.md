## 1. Why does `adam_eps` exist?

In AdamW, parameter updates divide by the square root of the second moment:

$$\Delta \theta = -\frac{\eta}{\sqrt{v_t} + \epsilon} \cdot m_t$$

If a parameter has zero or near-zero gradients (e.g. inactive tokens in embeddings), $v_t \to 0$. Without $\epsilon$, this causes catastrophic division-by-zero, creating NaNs and crashing training:

```text
Denominator Calculation:
  sqrt(v_t) + adam_eps
      │           │
   (May be 0)  (1e-8 safety floor)
      └─────┬─────┘
            ▼
    Guaranteed > 0
```

`adam_eps` adds a small numerical constant outside the square root in the Adam denominator.

---

## 2. Fundamentals & Mechanics

- Added outside the root: $\frac{1}{\sqrt{v_t} + \epsilon}$.
- Prevents division-by-zero and bounds the maximum possible coordinate update magnitude to $\frac{\eta}{\epsilon} \cdot m_t$.

---

## 3. Options & Defaults

| Option | Value | Meaning |
|---|---|---|
| Default | `1.e-8` (`1e-8`) | Standard numerical stabilizer for single and mixed precision. |
| High Precision Stability | `1.e-6` or `1.e-5` | Used in bfloat16 mixed-precision training to prevent underflow instability. |

---

## 4. Interactions & Dependencies

```text
adam_eps (outside root) vs adam_eps_root (inside root)
```

---

## 5. Practical Scenarios & Failure Modes

- If training with `bfloat16` activations and encountering sporadic NaN updates in zero-gradient layers, increasing `adam_eps` to `1e-6` or `1e-5` improves numerical stability.

---

### One-line intuition

> **`adam_eps` adds a numerical safety constant outside the square root in the Adam denominator to prevent division-by-zero on sparse or zero gradients.**
