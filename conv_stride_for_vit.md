## 1. Why does `conv_stride_for_vit` exist?

In Vision Transformers, the patch extraction layer is implemented via a 2D convolution (`nn.Conv2d` / `flax.linen.Conv`) with kernel size equal to `patch_size_for_vit`. The convolution stride determines whether patches are strictly adjacent (non-overlapping) or overlapping.

```text
conv_stride == patch_size (Non-overlapping):
[ Patch 0: pixels 0..13 ] [ Patch 1: pixels 14..27 ] ...

conv_stride < patch_size (Overlapping Patches):
[ Patch 0: pixels 0..13 ]
       [ Patch 1: pixels 7..20 ] (Higher visual fidelity, more tokens)
```

`conv_stride_for_vit` sets the step size (stride) of the patch projection convolution.

---

## 2. Options and Defaults

| Value | Behavior |
|---|---|
| `14` (Default) | Stride matches `patch_size_for_vit: 14` (non-overlapping adjacent patches) |
| Smaller integer (e.g. `7`) | Overlapping patches for dense visual feature extraction |

---

## 3. Interactions

- **`patch_size_for_vit`**: Base kernel size for the patch convolution.

---

### One-line intuition
> **`conv_stride_for_vit` sets the stride of the 2D patch-embedding convolution, controlling patch overlap and token density.**
