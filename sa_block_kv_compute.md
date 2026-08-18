## 1. Why does it exist?

While `sa_block_kv` defines the size of the KV tile transferred from HBM into on-chip Vector Memory (VMEM) via Direct Memory Access (DMA), the TPU's Matrix Multiply Units (MXUs) may execute matrix multiplications most efficiently at a smaller sub-tile granularity.

`sa_block_kv_compute` sets the inner compute tile size used by the MXU during the matrix multiplication steps ($Q \times K^T$ and $\text{Softmax} \times V$) within the larger outer memory tile.

```text
Outer Memory Tile (sa_block_kv = 512):
  ┌────────────────────────┬────────────────────────┐
  │ Compute Sub-Tile (512) │ Compute Sub-Tile (512) │
  └────────────────────────┴────────────────────────┘
```

`sa_block_kv_compute` specifies the inner compute block size along the KV dimension during the Splash Attention forward pass.

---

## 2. Options & Configuration

| Value | Meaning |
|---|---|
| `512` (default) | Compute granularity matches the outer memory transfer tile size. |
| Positive integer (e.g. `128`, `256`, `512`) | Sub-block size for fine-grained MXU execution. |

Default in `base.yml`:
```yaml
sa_block_kv_compute: 512
```

---

### One-line intuition

> **`sa_block_kv_compute` sets the inner matrix multiplication tile size (default 512) executed on TPU systolic arrays within each loaded KV memory chunk.**
