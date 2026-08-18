## 1. Why does `temporal_patch_size_for_vit` exist?

When encoding video frames, adjacent video frames exhibit extreme temporal redundancy. Standard 2D patchification on every frame independently generates unmanageable sequence lengths. 

3D spatiotemporal patchification extends the 2D spatial patch into the temporal domain, grouping $T_{\text{patch}}$ consecutive frames into a single 3D tubelet patch.

```text
Video Frames (T = 4 frames):
[ Frame 0 ] [ Frame 1 ]   [ Frame 2 ] [ Frame 3 ]
└──────────┬──────────┘   └──────────┬──────────┘
           │                         │
           ▼                         ▼
   [ Temporal Patch 0 ]      [ Temporal Patch 1 ]
   (temporal_patch_size = 2 -> 2x temporal compression)
```

`temporal_patch_size_for_vit` sets the temporal stride and kernel depth along the frame axis.

---

## 2. Options and Defaults

| Value | Behavior |
|---|---|
| `2` (Default) | Groups 2 consecutive frames into 1 temporal patch ($2\times$ temporal compression) |
| `1` | Each frame processed as an independent temporal step (no temporal merging) |
| `4` | Aggressive 4-frame temporal grouping for long videos |

---

## 3. Interactions

- **`spatial_merge_size_for_vit`**: Operates on spatial dimensions ($H, W$) while temporal patch size operates along $T$.
- **`use_mrope`**: 3D M-RoPE assigns temporal positional IDs corresponding to the temporal patch index.

---

### One-line intuition
> **`temporal_patch_size_for_vit` defines how many consecutive video frames are grouped into a single 3D spatiotemporal tubelet patch.**
