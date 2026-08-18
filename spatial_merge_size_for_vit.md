## 1. Why does `spatial_merge_size_for_vit` exist?

In Qwen-VL and Qwen3-Omni architectures, raw 2D image patches produce a dense token grid. To make high-resolution image processing computationally tractable in the language model, spatial merging fuses an $M \times M$ window of adjacent spatial patch tokens into a single merged token before feeding the LLM.

```text
Raw Patch Grid (spatial_merge_size_for_vit = 2):
┌──────┬──────┐ ┌──────┬──────┐
│ P0,0 │ P0,1 │ │ P0,2 │ P0,3 │
├──────┼──────┤ ├──────┼──────┤
│ P1,0 │ P1,1 │ │ P1,2 │ P1,3 │
└──────┴──────┘ └──────┴──────┘
       │               │
       ▼               ▼
 [ Merged Token 0 ] [ Merged Token 1 ]  (4x sequence length reduction)
```

`spatial_merge_size_for_vit` defines the 2D window stride $M$ (e.g. $2$ for $2 \times 2$ merging) used to compress visual tokens.

---

## 2. Options and Defaults

| Value | Window Size | Token Reduction Factor |
|---|---|---|
| `2` (Default) | $2 \times 2$ window | $4\times$ fewer visual tokens ($M^2 = 4$) |
| `1` | $1 \times 1$ (no merge) | Full patch token density |
| `4` | $4 \times 4$ window | $16\times$ token compression |

---

## 3. Interactions

- **`temporal_patch_size_for_vit`**: Merges along the temporal axis while `spatial_merge_size_for_vit` merges along spatial axes.
- **`out_hidden_size_for_vit`**: The channel dimension after linear projection of the merged tokens.

---

### One-line intuition
> **`spatial_merge_size_for_vit` sets the $M 	imes M$ spatial patch merging window (default 2 for $2	imes 2=4	imes$ compression) in Qwen-style vision encoders.**
