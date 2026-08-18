## 1. Why does it exist?

Key tensors $[B, S, H, D]$ are streamed into TPU Vector Memory (VMEM) and multiplied against Query matrices ($Q \times K^T$).

When Key tensors are laid out with the head dimension as the minor (contiguous) stride (`"HEAD_DIM_MINOR"`), the DMA subsystem can stream blocks directly into TPU systolic arrays without runtime memory transpositions or `swapaxes` overhead.

```text
HEAD_DIM_MINOR layout [B, S, H, D]:
  Memory: ──→ [ ... D elements contiguous ... ]
  ──→ Streamed directly into systolic arrays for QK^T matrix multiplication.

Non-minor layout:
  Requires `jnp.swapaxes` before passing to the Pallas Mosaic attention kernel.
```

`sa_k_layout` specifies the physical memory layout for Key tensors in Splash Attention.

---

## 2. Options & Configuration

| Value | Meaning |
|---|---|
| `"HEAD_DIM_MINOR"` (default) | Formats Key tensors with head dimension $D$ as the minor stride for zero-copy DMA streaming. |

Default in `base.yml`:
```yaml
sa_k_layout: "HEAD_DIM_MINOR"
```

---

### One-line intuition

> **`sa_k_layout` configures the physical memory layout (default `"HEAD_DIM_MINOR"`) for Key tensors, enabling zero-copy DMA streaming into TPU systolic arrays.**
