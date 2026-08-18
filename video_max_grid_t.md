## 1. Why does `video_max_grid_t` exist?

Video processing requires sampling frames across the temporal dimension $T$. Without a hard bound, high-FPS or long videos would produce an astronomical number of spatiotemporal patch tokens, easily exceeding model context limits (`max_target_length`) and causing out-of-memory errors.

`video_max_grid_t` sets the maximum temporal grid size (maximum number of temporal frame slices or 3D temporal patches) permitted during video patchification.

```text
Video Timeline: [ Frame 1 ... Frame 300 ]
                       │
                       ▼
Temporal Subsampling / 3D Patching (video_max_grid_t = 16)
                       │
                       ▼
Bounded Temporal Grid: [ T_0 , T_1 , ... , T_15 ] (Max 16 temporal steps)
```

---

## 2. Options and Defaults

| Value | Behavior |
|---|---|
| `null` (Default) | Unset; determined dynamically by dataset pipeline or model preset |
| Integer (e.g. `8`, `16`, `32`) | Hard ceiling on the temporal grid dimension $T$ |

---

## 3. Interactions

- **`video_max_grid_h` & `video_max_grid_w`**: Bounds the spatial height and width grid dimensions alongside temporal depth.
- **`temporal_patch_size_for_vit`**: Determines how many consecutive frames are collapsed into one temporal patch.

---

### One-line intuition
> **`video_max_grid_t` caps the maximum number of temporal frame patches extracted from a video to bound token counts and memory.**
