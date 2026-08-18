## 1. Why does `video_max_grid_h` exist?

In dynamic-resolution video encoders (such as NaViT or Qwen2-VL), video frames are partitioned into a 2D spatial grid $(H_{\text{grid}} 	imes W_{\text{grid}})$. Setting an explicit ceiling on the height grid dimension ensures that tall or high-resolution video frames do not generate excessive visual tokens.

`video_max_grid_h` defines the maximum height dimension of the patch grid during video patchification.

```text
Video Frame Height H ──► Patch Grid Height = H / Patch_Size
                              │
                              ▼
                Clamp to video_max_grid_h
```

---

## 2. Options and Defaults

| Value | Meaning |
|---|---|
| `null` (Default) | Unbounded / determined by model defaults |
| Integer (e.g. `16`, `32`) | Maximum height patch grid slots |

---

## 3. Interactions

- **`video_max_grid_w`**: Spatial width counterpart.
- **`video_max_grid_t`**: Temporal depth counterpart.

---

### One-line intuition
> **`video_max_grid_h` caps the vertical patch grid resolution for video frames to prevent spatial token explosion.**
