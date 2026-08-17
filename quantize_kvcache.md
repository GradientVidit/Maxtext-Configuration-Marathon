
## 1. Why does it exist?

During inference and serving, the **KV cache** is the dominant memory consumer for long sequences. A single serving request with a 128K token context on a large model can require tens of GB of KV cache — at bfloat16 (2 bytes per value). This sets a hard upper bound on how many concurrent requests can be served.

Quantizing the KV cache to int8 (1 byte per value) halves its memory footprint, enabling:
- 2x longer contexts at the same memory budget
- 2x more concurrent requests
- Or a smaller GPU/TPU requirement for the same workload

`quantize_kvcache` is the switch to enable this.

---

## 2. KV cache vs weight quantization — they are independent

This is a common source of confusion:

```text
quantization (weights/activations)     ← controls GEMM precision during forward pass
quantize_kvcache (KV cache storage)    ← controls how cached K/V tensors are stored
```

You can independently mix these:
```yaml
quantization: ""             # full bfloat16 weights
quantize_kvcache: true       # but store KV cache in int8
```

This is a common inference configuration: keep weights in bf16 for compute accuracy, but compress the KV cache for memory.

---

## 3. Options

| Value | Behavior |
|---|---|
| `false` | KV cache stored in `dtype` (bfloat16 by default) — default |
| `true` | KV cache stored in `kv_quant_dtype` (int8 by default), quantized per `kv_quant_axis` |

Default in base.yml:
```yaml
quantize_kvcache: false
```

---

## 4. Accuracy impact

KV cache quantization introduces approximation: the cached keys and values are stored at lower precision and dequantized when used in attention computations. This can slightly degrade output quality, especially at:
- Very long sequences (more accumulated rounding)
- Heads with extreme key/value distributions
- Tasks requiring precise long-range recall

The impact is typically small for int8, and empirically acceptable for most production serving use cases.

---

## 5. Companion parameters

| Param | Role when `quantize_kvcache=true` |
|---|---|
| `kv_quant_axis` | Which axis of the KV tensor to quantize over |
| `kv_quant_dtype` | The target storage dtype (default: `"int8"`) |

These only matter when `quantize_kvcache=true`. When KV cache quantization is off, `kv_quant_axis` must be `""`.

---

## 6. Training vs inference

`quantize_kvcache` is primarily an **inference/serving** optimization. During training, the KV cache is not reused across steps (each forward pass recomputes attention from scratch), so quantizing it only affects training if using cached decoding/inference mode during evaluation.

---

### One-line intuition

> **`quantize_kvcache=true` compresses the KV cache from bfloat16 to int8 (halving its memory) — an inference-time memory optimization that's completely independent of weight quantization and enables 2x longer contexts or 2x more concurrent requests.**
