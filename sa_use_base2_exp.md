## 1. Why does it exist?

TPU Vector Processing Units (VPUs) have dedicated, single-cycle hardware instructions for computing base-2 exponentiations ($2^x$), whereas computing natural base-$e$ exponentiations ($e^x = \exp(x)$) requires multiplying by $\log_2(e)$ inside polynomial approximation routines.

By transforming the attention softmax computation into base-2:

$$\exp(x) = 2^{x \cdot \log_2(e)}$$

the kernel pre-scales logits by $\log_2(e)$ during the QK dot-product scaling step and executes base-2 hardware exponent instructions directly, accelerating softmax evaluation without loss of numerical accuracy.

```text
Natural Exp: Logits ──→ Natural Exp Function (Multiple VPU Instruction Cycles)
Base-2 Exp:  Logits * log2(e) ──→ Dedicated Hardware 2^x Unit (Single Cycle!)
```

`sa_use_base2_exp` enables base-2 exponent computation in Splash Attention.

---

## 2. Options & Configuration

| Value | Meaning |
|---|---|
| `true` (default) | Uses fast base-2 hardware exponential instructions (Tokamax default). |
| `false` | Uses standard natural exponential function. |

Default in `base.yml`:
```yaml
sa_use_base2_exp: true # defaults to true in Tokamax
```

---

### One-line intuition

> **`sa_use_base2_exp` switches the softmax exponential to base-2 ($2^x$), directly utilizing dedicated TPU single-cycle hardware exponent units for faster attention compute.**
