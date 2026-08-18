## 1. Why does `video_max_grid_w` exist?

Along with height and temporal bounds, wide-aspect-ratio videos (e.g., 21:9 or 16:9 widescreen formats) can produce disproportionately large horizontal patch counts.

`video_max_grid_w` sets the maximum width dimension of the patch grid during video patchification.

```text
Video Frame Width W ──► Patch Grid Width = W / Patch_Size
                             │
                             ▼
               Clamp to video_max_grid_w
```

---

## 2. Options and Defaults

| Value | Meaning |
|---|---|
| `null` (Default) | Unbounded / model-determined default |
| Integer (e.g. `16`, `32`) | Maximum horizontal patch grid slots |

---

## 3. Interactions

- **`video_max_grid_h`**: Height counterpart.
- **`video_max_grid_t`**: Temporal counterpart.

---

### One-line intuition
> **`video_max_grid_w` caps the horizontal patch grid resolution for video frames to bound token generation in wide aspect ratios.**
