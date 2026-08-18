## 1. Why does `output_dim_for_audio` exist?

After the audio encoder finishes processing the acoustic sequence, its final representations must be projected to match the LLM decoder embedding dimension or the audio projector input dimension.

`output_dim_for_audio` specifies the final feature channel dimension emitted by the audio encoder.

```text
Audio Transformer Final Layer ──► Output Linear Projection ──► Dim: output_dim_for_audio (512)
                                                                       │
                                                                       ▼
                                                             Multimodal Projector / LLM
```

---

## 2. Options and Defaults

| Value | Description |
|---|---|
| `512` (Default) | Standard output dimension for Qwen3-OmniMoE audio encoder |
| Matching `base_emb_dim` | Direct projection to LLM embedding dimension |

---

### One-line intuition
> **`output_dim_for_audio` specifies the final output channel dimensionality (default 512) emitted by the audio encoder.**
