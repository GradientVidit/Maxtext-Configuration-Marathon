## 1. Why does `pixel_shuffle_ratio_for_vit` exist?

High-resolution vision encoders produce thousands of visual patch tokens (e.g. $64 	imes 64 = 4096$ tokens). Passing 4096 visual tokens into a 32-layer LLM creates massive computational overhead. 

Pixel shuffle (spatial merging / space-to-depth) reduces the spatial sequence length by a factor of 4 while quadrupling the channel dimension, merging $2 	imes 2$ neighboring spatial patches into a single high-information token.

```text
Spatial 2x2 Patches (4 tokens, dim D):
┌──────┬──────┐
│ (0,0)│ (0,1)│
├──────┼──────┤
│ (1,0)│ (1,1)│
└──────┴──────┘
       │
       ▼ Pixel Shuffle Downsampling (ratio = 0.5)
Single Spatial Token (1 token, dim 4*D)
```

`pixel_shuffle_ratio_for_vit` controls the spatial scale factor used during pixel shuffle downsampling.

---

## 2. Options and Defaults

| Value | Spatial Token Reduction | Channel Expansion |
|---|---|---|
| `0.5` (Default) | $2 	imes 2$ spatial merge ($4	imes$ fewer tokens) | $4	imes$ channel dimension |
| `1.0` | No pixel shuffle downsampling | Channel dimension unchanged |

---

## 3. Interactions

- **`projector_input_dim_for_vit`**: Channel dimension after pixel shuffle must match projector input width.

---

### One-line intuition
> **`pixel_shuffle_ratio_for_vit` sets the spatial downsampling ratio (default 0.5 for $2	imes 2$ space-to-depth compression) to reduce visual token count before feeding the LLM.**
