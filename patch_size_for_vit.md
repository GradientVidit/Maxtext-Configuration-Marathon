## 1. Why does `patch_size_for_vit` exist?

Transformers cannot process individual continuous pixels directly. A Vision Transformer divides the 2D image into non-overlapping spatial patches of size $P 	imes P$ (e.g. $14 	imes 14$ pixels), flattening each patch into a 1D vector before projection.

The patch size governs the spatial token granularity: smaller patches capture fine-grained details but generate exponentially more tokens; larger patches produce fewer tokens but lose subtle visual information.

```text
Image (336 x 336) with patch_size_for_vit = 14:
Grid: (336 / 14) x (336 / 14) = 24 x 24 = 576 patches.

Image (336 x 336) with patch_size_for_vit = 28:
Grid: (336 / 28) x (336 / 28) = 12 x 12 = 144 patches (4x fewer tokens).
```

`patch_size_for_vit` sets the height and width in pixels of each visual patch.

---

## 2. Options and Defaults

| Value | Resolution Granularity | Typical Model |
|---|---|---|
| `14` (Default) | High granularity ($14 	imes 14$ pixels per token) | Llama 4 Vision, SigLIP, CLIP-L/14 |
| `16` | Standard granularity ($16 	imes 16$) | Original ViT, DINO |
| `28` | Low compute / coarse granularity | Ultra-fast mobile vision encoders |

---

## 3. Interactions

- **`conv_stride_for_vit`**: Typically equals `patch_size_for_vit` for non-overlapping patches.
- **`tile_size_for_vit` / `image_size_for_vit`**: Must be divisible by `patch_size_for_vit`.

---

### One-line intuition
> **`patch_size_for_vit` sets the square pixel dimension ($P 	imes P$) of raw image patches converted into individual visual tokens.**
