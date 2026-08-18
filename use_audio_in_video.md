## 1. Why does `use_audio_in_video` exist?

Video container files (e.g., MP4) contain both visual frames (video stream) and a synchronized soundtrack (audio stream). In many video reasoning tasks (e.g. understanding dialog, musical cues, or vehicle sirens), analyzing visual frames alone misses critical context.

`use_audio_in_video` controls whether the multimodal data loader automatically demuxes and extracts the audio track from video inputs to feed into the acoustic encoder.

```text
Input Video (.mp4)
       │
       ├── Visual Track ──► ViT Encoder ────► Visual Tokens ──┐
       │                                                      ├──► Multimodal LLM
       └── Audio Track  ──► Audio Encoder ──► Audio Tokens  ──┘
           (Activated when use_audio_in_video: true)
```

---

## 2. Options and Defaults

| Value | Behavior |
|---|---|
| `false` (Default) | Only visual frames are processed; audio track is ignored |
| `true` | Extracts both visual frames and audio stream for joint audio-visual reasoning |

---

## 3. Parameter Interactions

- **`use_audio`**: Must be `true` to process the extracted audio track.
- **`video_path`**: Supplies the container file.

---

### One-line intuition
> **`use_audio_in_video` enables automatic extraction and encoding of the audio soundtrack from video container files for joint audio-visual understanding.**
