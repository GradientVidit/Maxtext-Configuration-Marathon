## 1. Why does it exist?

In long-context attention, streaming massive Key ($K$) tensors into on-chip Vector Memory (VMEM) is a primary memory bandwidth bottleneck.

Quantizing Key tensors to 8-bit (`jnp.float8_e4m3fn`) halves the byte volume transferred from HBM to VMEM across all attention layers.

`experimental_sa_quant_k_fp8` is an experimental flag that quantizes the Key ($K$) tensor in Splash Attention to `jnp.float8_e4m3fn` without scaling factors.

---

## 2. Options & Configuration

| Value | Meaning |
|---|---|
| `false` (default) | Key tensor uses standard `bfloat16` precision. |
| `true` | Quantizes Key tensor to `float8_e4m3fn` in Splash Attention. |

Default in `base.yml`:
```yaml
experimental_sa_quant_k_fp8: False
```

---

### One-line intuition

> **`experimental_sa_quant_k_fp8` quantizes Key tensors to `float8_e4m3fn` in Splash Attention, halving Key streaming bandwidth across long sequences.**
