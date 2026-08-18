## 1. Why does `conv_chunksize_for_audio` exist?

The audio convolutional front-end processes continuous streaming audio inputs. To support streaming speech processing without waiting for the full audio file to buffer, 1D/2D convolutions operate over streaming chunks of size $C_{\text{chunk}}$.

`conv_chunksize_for_audio` sets the chunk size for convolutional front-end processing in the audio encoder.

```text
Streaming Audio Stream ──► Slice into Chunks of conv_chunksize_for_audio (500) ──► Conv Frontend
```

---

## 2. Options and Defaults

| Value | Behavior |
|---|---|
| `500` (Default) | 500-frame convolutional chunk processing |
| Integer $> 0$ | Custom streaming chunk size |

---

### One-line intuition
> **`conv_chunksize_for_audio` defines the frame chunk size for the audio encoder's convolutional front-end.**
