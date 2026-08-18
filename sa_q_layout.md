## 1. Why does it exist?

TPU Matrix Multiply Units (MXUs) and Vector Memory DMA controllers have strict hardware alignment and memory striding preferences (e.g., contiguous 128-byte vector registers along the inner head dimension).

In Flash/Splash attention kernels, the input Query tensor can be structured in memory either with sequence as the minor dimension or head dimension as the minor dimension:
- `HEAD_DIM_MINOR`: shape `[B, S, H, D]` where head dimension $D$ is the fastest-moving contiguous axis.
- Non-minor layouts: require explicit `jnp.swapaxes` transpose operations inside the kernel wrapper before launching the Pallas TPU Mosaic grid.

```text
HEAD_DIM_MINOR layout [B, S, H, D]:
  Memory: ──→ [ ... D elements contiguous ... ]
  ──→ Zero-copy Direct Memory Access (DMA) straight into TPU VMEM!

Non-minor layout (e.g. [B, H, D, S]):
  Kernel Wrapper ──→ Calls `jnp.swapaxes` ──→ Memory transposition overhead
```

`sa_q_layout` configures the physical memory layout descriptor for Query tensors in the Splash Attention kernel.

---

## 2. Options & Configuration

| Value | Tensor Striding | Kernel Action |
|---|---|---|
| `"HEAD_DIM_MINOR"` (default) | `[Batch, Seq, Heads, HeadDim]` | Direct DMA streaming into Pallas kernel; no transpose required. |
| Other layout strings | Custom tensor ordering | Kernel executes `swapaxes` to align dimensions. |

Default in `base.yml`:
```yaml
sa_q_layout: "HEAD_DIM_MINOR"
```

---

## 3. Companion Layouts

- **`sa_k_layout`**: Layout descriptor for Key tensors.
- **`sa_v_layout`**: Layout descriptor for Value tensors.

---

### One-line intuition

> **`sa_q_layout` declares the memory layout for Query tensors (default `"HEAD_DIM_MINOR"`), allowing Direct Memory Access into TPU vector registers without runtime transposition.**
