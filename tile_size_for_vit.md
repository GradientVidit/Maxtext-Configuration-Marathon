## 1. Why does `tile_size_for_vit` exist?

High-resolution images (e.g. $4\text{K}$ or $1080\text{p}$) cannot be fed into standard ViTs as single monolithic images without either massive downsampling (losing fine text/details) or quadratic attention explosion. Modern VLMs partition high-resolution images into a grid of smaller square **tiles** (e.g. $336 	imes 336$ or $448 	imes 448$), encoding each tile independently before fusing them with an overview thumbnail.

```text
High-Res Input (672 x 672)
┌──────────────┬──────────────┐
│ Tile 1 (336) │ Tile 2 (336) │
├──────────────┼──────────────┤
│ Tile 3 (336) │ Tile 4 (336) │  ──► ViT processes each 336x336 tile
└──────────────┴──────────────┘
```

`tile_size_for_vit` defines the square pixel resolution of individual image tiles in tiled vision processing pipelines.

---

## 2. Options and Defaults

| Value | Model Family |
|---|---|
| `336` (Default) | Llama 4 Vision / LLaVA-NeXT (CLIP-Large 336px tile) |
| `448` | InternVL / SPHINX |
| `896` | Gemma 3 single-canvas mode |

---

## 3. Interactions

- **`patch_size_for_vit`**: Number of patches per tile $= (\text{tile\_size} / \text{patch\_size})^2 = (336/14)^2 = 576$ patches.
- **`image_size_for_vit`**: Sets global canvas dimension.

---

### One-line intuition
> **`tile_size_for_vit` defines the square pixel dimension of sub-image tiles in high-resolution partitioned vision pipelines.**
