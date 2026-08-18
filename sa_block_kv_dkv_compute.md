## 1. Why does it exist?

Within each outer Key/Value memory tile of size `sa_block_kv_dkv`, matrix multiplication operations ($dK = dS^T \times Q$ and $dV = S^T \times dO$) execute on the TPU Matrix Multiply Units (MXUs) in sub-tiles.

`sa_block_kv_dkv_compute` configures the inner compute sub-tile size along the KV dimension for the $dK/dV$ backward pass.

---

## 2. Options & Configuration

| Value | Meaning |
|---|---|
| `512` (default) | Compute sub-tile matches outer memory transfer size. |
| Positive integer (e.g. `128`, `256`, `512`) | Fine-grained compute sub-tile size. |

Default in `base.yml`:
```yaml
sa_block_kv_dkv_compute: 512
```

---

### One-line intuition

> **`sa_block_kv_dkv_compute` controls the inner matrix multiplication tile size (default 512) for computing Key/Value gradients on TPU systolic arrays.**
