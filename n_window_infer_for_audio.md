## 1. Why does `n_window_infer_for_audio` exist?

During inference, batch sizes are typically smaller and TPU memory pressure is lower than during distributed training. Expanding the attention window during inference allows the audio encoder to capture broader acoustic context across long utterances without incurring training-time memory spikes.

`n_window_infer_for_audio` specifies the attention window size used specifically during audio inference.

```text
Training:  n_window_for_audio = 50   (Tight window for high training batch throughput)
Inference: n_window_infer_for_audio = 800 (Expanded window for maximum acoustic context)
```

---

## 2. Options and Defaults

| Value | Description |
|---|---|
| `800` (Default) | Extended 800-frame attention window during inference |
| Matching `n_window_for_audio` | Keeps training and inference windows identical |

---

### One-line intuition
> **`n_window_infer_for_audio` defines the expanded temporal attention window size (default 800 frames) used during audio inference.**
