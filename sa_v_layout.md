## 1. Why does it exist?

Value tensors $[B, S, H, D]$ are multiplied by the post-softmax attention score matrix ($S \times V$) to produce final attention outputs. Aligning Value tensors with the head dimension as the minor axis (`"HEAD_DIM_MINOR"`) avoids expensive transpose operations in memory before kernel execution.

```text
HEAD_DIM_MINOR layout [B, S, H, D]:
  Memory: ──→ [ ... D elements contiguous ... ]
  ──→ Directly multiplied with post-softmax attention scores without layout conversion.

Non-minor layout:
  Requires `jnp.swapaxes` in the kernel wrapper, adding memory bandwidth overhead.
```

`sa_v_layout` specifies the physical memory layout for Value tensors in Splash Attention.

---

## 2. Options & Configuration

| Value | Meaning |
|---|---|
| `"HEAD_DIM_MINOR"` (default) | Formats Value tensors with the head dimension as the minor stride. |

Default in `base.yml`:
```yaml
sa_v_layout: "HEAD_DIM_MINOR"
```

---

### One-line intuition

> **`sa_v_layout` configures the physical memory layout (default `"HEAD_DIM_MINOR"`) for Value tensors in Splash Attention to eliminate runtime axis swapping.**
