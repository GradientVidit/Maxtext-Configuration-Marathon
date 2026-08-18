## 1. Why does `position_id_per_seconds` exist?

When videos and audio streams are ingested by an omni-modal model, frames can be sampled at variable frame rates (e.g. 1 FPS, 5 FPS, 24 FPS, 60 FPS). If temporal position IDs were simply raw frame indices ($0, 1, 2, \dots$), a 1-second video sampled at 24 FPS would appear to the model as 24 times "longer" in time than a 1-second video sampled at 1 FPS.

To anchor temporal Rotary Position Embeddings to physical wall-clock playback time, MaxText scales temporal coordinates by `position_id_per_seconds`.

```text
Video Frame at Timestamp t = 2.4 seconds:
Temporal Position ID = int(t * position_id_per_seconds)
With position_id_per_seconds = 25:
Position ID = int(2.4 * 25) = 60

Audio Chunk at Timestamp t = 2.4 seconds:
Position ID = int(2.4 * 25) = 60 (Perfect Temporal Alignment!)
```

`position_id_per_seconds` defines the temporal coordinate quantization frequency (in position IDs per second).

---

## 2. Options and Defaults

| Value | Time Granularity | Standard Setting |
|---|---|---|
| `25` (Default) | 25 position IDs per second (40ms resolution) | Qwen2-VL / Qwen3-Omni default |
| `50` | 50 position IDs per second (20ms resolution) | High-precision audio alignment |
| `1` | 1 position ID per second | Coarse-grained video understanding |

---

## 3. Interactions

- **`use_mrope`**: Operates on the temporal section $d_T$ of M-RoPE.
- **`use_audio_in_video`**: Ensures synchronized audio and video tokens share identical temporal position IDs.

---

### One-line intuition
> **`position_id_per_seconds` anchors M-RoPE temporal position IDs to physical playback time (default 25 IDs/second), ensuring consistent time representations across variable frame rates.**
