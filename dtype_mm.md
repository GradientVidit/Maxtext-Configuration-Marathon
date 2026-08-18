## 1. Why does `dtype_mm` exist?

The Vision Transformer (ViT) encoder and the main autoregressive language model (LLM) have vastly different compute profiles and numerical tolerances. While LLM decoders frequently run in `bfloat16` or quantized formats (`fp8`), vision encoders processing high-resolution continuous pixel signals can suffer from numerical underflow or loss spikes during patch projection and LayerNorm if low precision is used improperly.

Conversely, running the ViT in `float32` provides maximum numerical precision for pixel embeddings while letting the main decoder run in high-throughput `bfloat16`.

```text
Input Image ──► [ ViT Encoder (dtype_mm = "float32") ]
                       │
                 High precision pixel features
                       │
                       ▼
                [ Projector ] ──► Cast to LLM dtype (bfloat16) ──► [ LLM Decoder (bfloat16) ]
```

`dtype_mm` sets the numerical floating-point data type used specifically inside the multimodal vision encoder.

---

## 2. Supported Options

| Value | Numerical Format | Precision vs Speed |
|---|---|---|
| `"float32"` (Default) | Full 32-bit single precision | Maximum stability; default in MaxText `base.yml` |
| `"bfloat16"` | 16-bit Brain Floating Point | 2x speedup on TPU MXUs, halves ViT activation memory |
| `"float16"` | 16-bit IEEE Float | Standard on GPUs; prone to underflow in deep ViT without loss scaling |

---

## 3. Parameter Interactions

- **`dtype`**: Sets the global LLM precision (usually `bfloat16`).
- **`weight_dtype`**: Controls weight storage format; `dtype_mm` governs the arithmetic execution dtype inside the ViT.

---

### One-line intuition
> **`dtype_mm` sets the arithmetic floating-point precision specifically for the Vision Transformer encoder, allowing full fp32 precision for pixel operations independent of the LLM dtype.**
