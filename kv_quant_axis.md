
## 1. Why does it exist?

When quantizing the KV cache (see `quantize_kvcache`), you must decide **along which dimension** the quantization scale factor is computed. This choice directly controls the accuracy vs. compute trade-off.

The KV cache tensor shape is:
```text
[batch, heads, sequence, kv_dim]
```

Quantizing per-`kv_dim` means one scale per head-position pair. Quantizing over both `heads` and `kv_dim` means one scale per sequence position. These two axes give different granularity of quantization.

---

## 2. The two axes explained

### `"dkv"` — quantize over the KV feature dimension

```text
for each (batch, head, sequence_position):
  scale = max(|K[b, h, seq, :]|)  ← one scale per head-position
  K_q = round(K / scale)
```

Each head-position gets its own scale. Better accuracy because the scale adapts to the distribution within each head at each position, but more scales to store and more compute for dequantization.

### `"heads_and_dkv"` — quantize over both heads and KV feature dimension

```text
for each (batch, sequence_position):
  scale = max(|K[b, :, seq, :]|)  ← one scale per sequence position
  K_q = round(K / scale)
```

One scale covers all heads at a given sequence position. Fewer scales to store, simpler computation, but the single scale must accommodate all heads' value ranges simultaneously — less accurate if heads have very different distributions.

---

## 3. Options

| Value | Behavior | Trade-off |
|---|---|---|
| `""` | Only valid when `quantize_kvcache=false` | N/A |
| `"dkv"` | Quantize over the KV-dim axis (per head-position scale) | Better accuracy, more overhead |
| `"heads_and_dkv"` | Quantize over heads AND KV-dim (per-sequence-position scale) | Faster compute, slightly less accurate |

Default in base.yml:
```yaml
kv_quant_axis: "heads_and_dkv"
```

MaxText's comment explicitly says: "default to `heads_and_dkv` for faster computation; `dkv` is expected with better accuracy but degraded computation."

---

## 4. When to use each

**`"heads_and_dkv"` (default):**
- Production serving where throughput matters
- Most attention heads have similar value distributions
- Acceptable for most benchmarks

**`"dkv"`:**
- Quality-sensitive applications where serving accuracy is paramount
- Models with high inter-head variance in KV distributions
- When you observe accuracy degradation with `heads_and_dkv`

---

## 5. Dependency

Only meaningful when:
```yaml
quantize_kvcache: true
```

When `quantize_kvcache=false`, `kv_quant_axis` must be `""`.

---

### One-line intuition

> **`kv_quant_axis` controls quantization granularity in the KV cache — `"heads_and_dkv"` (default) uses one scale per sequence position (faster), while `"dkv"` uses one per head per position (more accurate but slower).**
