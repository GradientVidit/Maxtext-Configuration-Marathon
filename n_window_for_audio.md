## 1. Why does `n_window_for_audio` exist?

In long audio recordings, calculating full quadratic self-attention across thousands of audio frames is memory-intensive and biologically unnatural for speech perception. Windowed (chunked) attention restricts attention to a localized temporal window of $W$ frames.

`n_window_for_audio` sets the temporal attention window size used during audio encoder training.

```text
Audio Frames: [ 0 ........................................ 1500 ]
              └── Window 0 (50) ──┘
                    └── Window 1 (50) ──┘
```

---

## 2. Options and Defaults

| Value | Context |
|---|---|
| `50` (Default) | 50-frame local temporal attention window during training |
| Integer $> 0$ | Custom training window length |

---

## 3. Interactions

- **`n_window_infer_for_audio`**: Inference counterpart, which can be configured with a larger window for global context.

---

### One-line intuition
> **`n_window_for_audio` defines the local temporal attention window size (default 50 frames) used during audio encoder training.**
