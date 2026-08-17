
## 1. Why does it exist?

When `quantize_kvcache=true`, the KV cache must be stored in some low-precision format. `kv_quant_dtype` specifies that format.

The quantized KV cache dtype is separate from the model's computation dtype (`dtype`), weight dtype (`weight_dtype`), and the quantization scheme for weights (`quantization`). The KV cache can be in int8 regardless of what the rest of the model uses.

---

## 2. Why int8 specifically?

The design constraints for KV cache storage:
- Must be **compact** (memory is the bottleneck for long contexts)
- Must be **fast to dequantize** (attention reads the KV cache every decoding step)
- Must have **sufficient range** for typical key/value distributions

int8 (8-bit signed integer, range [-128, 127]) meets all three:
- Compact: 1 byte per value vs. 2 bytes for bf16
- Fast: hardware-level int8-to-float conversion
- Range: with per-position or per-head scale factors, captures typical KV distributions accurately

---

## 3. Options

| Value | Behavior |
|---|---|
| `"int8"` | Store KV cache as 8-bit signed integers (default) |
| Other dtype strings | Other quantized formats if supported |

Default in base.yml:
```yaml
kv_quant_dtype: "int8"
```

---

## 4. The full KV cache quantization picture

Three parameters work together:

```text
quantize_kvcache: true    → enable KV cache quantization
kv_quant_axis: "heads_and_dkv"  → what axis to compute scales over
kv_quant_dtype: "int8"    → what dtype to store quantized values in
```

The runtime process:
```text
K, V tensors (bfloat16)
       ↓
compute scale (per kv_quant_axis)
       ↓
quantize: K_q = round(K / scale)  [stored as int8]
       ↓
cache K_q  [1 byte per value]

at attention time:
  K = dequantize(K_q, scale)  [back to bfloat16 for matmul]
```

---

## 5. Dependency

Only meaningful when:
```yaml
quantize_kvcache: true
```

If KV cache quantization is off, `kv_quant_dtype` is ignored — the cache is stored in whatever `dtype` is.

---

## 6. Memory savings

```text
bfloat16 KV cache per token:
  2 bytes × (num_heads × head_dim) × 2 (K and V) per layer

int8 KV cache per token:
  1 byte × (num_heads × head_dim) × 2 (K and V) per layer
```

Plus a tiny overhead for scale factors (one float32 per position, or per head-position, depending on `kv_quant_axis`). Net result: ~50% KV cache memory reduction.

---

### One-line intuition

> **`kv_quant_dtype="int8"` specifies that the quantized KV cache is stored as 8-bit signed integers — the standard format that achieves 50% KV memory reduction with fast hardware dequantization at attention time.**
