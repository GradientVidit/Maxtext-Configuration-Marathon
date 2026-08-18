## 1. Why does it exist?

During attention computation, reading Query tensors from memory accounts for a substantial fraction of memory bandwidth.

**FP8 Quantization** reduces tensor precision from 16-bit (BF16, 2 bytes) to 8-bit (`jnp.float8_e4m3fn`, 1 byte). Quantizing the Query tensor in Splash Attention cuts the required memory bandwidth for Query reads by 50% and leverages hardware FP8 Matrix Multiply Units for higher FLOP throughput.

```text
BF16 Query Tensor: 2 Bytes per element ──→ Standard Memory Bandwidth
FP8 Query Tensor:  1 Byte per element  ──→ 2x Bandwidth Reduction!
```

`experimental_sa_quant_q_fp8` is an experimental flag that quantizes the Query ($Q$) tensor in Splash Attention to `jnp.float8_e4m3fn` without scaling factors.

---

## 2. Options & Configuration

| Value | Meaning |
|---|---|
| `false` (default) | Query tensor is processed in standard model activation precision (`bfloat16`). |
| `true` | Quantizes Query tensor directly to `float8_e4m3fn` in Splash Attention. |

Default in `base.yml`:
```yaml
experimental_sa_quant_q_fp8: False
```

---

## 3. Companion Parameter

- **`experimental_sa_quant_k_fp8`**: Quantizes Key tensors to FP8.

---

### One-line intuition

> **`experimental_sa_quant_q_fp8` quantizes the Query tensor to `float8_e4m3fn` in Splash Attention, halving Query memory bandwidth during attention computation.**
