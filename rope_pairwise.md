## 1. Why does `rope_pairwise` exist?

In JAX/Flax, 2D rotary transformations can be expressed either:
1. By reshaping the head dimension into pairs and multiplying by a $2 \times 2$ rotation matrix:
   $$\text{Shape: } [\text{batch}, \text{seq}, \text{heads}, d/2, 2]$$
2. Or by flattening the tensor back into the standard head dimension:
   $$\text{Shape: } [\text{batch}, \text{seq}, \text{heads}, d]$$

```text
Pairwise (Rank-5) Tensor:
[Batch, Seq, Heads, d/2, 2] ──(Preserves [Real, Imag] Coordinate Pairs)
           │
           ▼ Flatten
Flat (Rank-4) Tensor:
[Batch, Seq, Heads, d]
```

`rope_pairwise` controls whether the rotary embedding computation maintains the rank-5 pairwise representation throughout intermediate attention stages or flattens it.

---

## 2. What it actually controls

```yaml
rope_pairwise: false
```

- When `false` (default): MaxText flattens the rotary-transformed tensors back to standard 4D shape `[batch, sequence, heads, head_dim]`.
- When `true`: Returns the rank-5 pair tensor `[batch, sequence, heads, half_dim, 2]`, where the last axis of size 2 holds `[real, imag]` (or `[x_0, x_1]`) coordinates for custom fused kernels.

---

## 3. Options and Defaults

| Value | Tensor Output Shape | Memory Layout | Typical Use |
|---|---|---|---|
| `false` (default) | `[B, T, N, D]` | Standard contiguous 4D | General training, FlashAttention, standard XLA |
| `true` | `[B, T, N, D/2, 2]` | 5D pairwise representation | Custom fused attention kernels, complex-number math |

---

## 4. Interactions with Attention Kernels

- **Fused Attention Kernels**: Most FlashAttention and Pallas/Mosaic TPU attention implementations expect 4D inputs `[B, T, N, D]`. Setting `rope_pairwise: true` requires downstream kernels that support the 5D pair layout.

---

## 5. Practical Scenarios

- **Standard MaxText Training & Decoding**: Leave `rope_pairwise: false`.

---

### One-line intuition

> **`rope_pairwise` keeps rotary embedding tensors in a 5D `[B, T, N, D/2, 2]` coordinate-pair format rather than flattening them to standard 4D `[B, T, N, D]`.**
