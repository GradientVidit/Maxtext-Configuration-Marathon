
## 1. Why does it exist?

Determining the quantization range (calibration) requires computing statistics over a tensor — typically a reduction operation like max, percentile, or mean. For large models distributed across many devices and slices, doing this reduction over the entire global tensor is expensive.

`quantization_local_shard_count` controls how many **shards** the range-finding operation is split across, trading calibration accuracy against computation cost.

---

## 2. What "local shard count" means

In a multi-slice TPU setup, model weights are partitioned across slices. The range-finding reduction can be done:

```text
Global (shard_count = 1 or all slices together):
  max(|W|) computed across all slices
  → most accurate, but requires cross-slice communication
  → slower for very large tensors on many slices

Local per-slice (-1 default = number of slices):
  each slice computes max(|W_local_shard|)
  → approximate, slightly less accurate
  → no cross-slice communication needed for range finding
  → faster
```

Setting `quantization_local_shard_count` to the number of slices means each slice calibrates independently on its local shard.

---

## 3. Options

| Value | Behavior |
|---|---|
| `-1` | Default to number of slices — each slice calibrates independently |
| Positive integer | Override shard count explicitly |

Default in base.yml:
```yaml
quantization_local_shard_count: -1
```

---

## 4. Accuracy vs performance trade-off

```text
Fewer shards (→ 1 = global):
  More communication, more accurate range → better quantization accuracy

More shards (→ num_slices = default):
  Less communication, per-shard range → slightly less accurate but scalable
```

For typical int8 quantization, the per-shard range approximation is negligible in practice — the calibration error introduced is smaller than the quantization step size itself.

For very sensitive quantization regimes (aggressive bitwidths, fp8 static scaling) or models with extreme weight variation across slices, global calibration may matter.

---

## 5. When to change it

**Leave at `-1`** (per-slice default) for essentially all use cases. The performance benefit of local calibration is meaningful at scale (many slices), and the accuracy cost is typically negligible.

**Override to a smaller number** if you observe per-slice calibration causing noticeable accuracy degradation — e.g., if one slice has an outlier shard that skews the local scale.

---

## 6. Dependency

Only relevant when quantization is active:
```yaml
quantization: "int8"  # or any non-empty quantization scheme
```

With `quantization: ""`, there's no range-finding operation, so this parameter has no effect.

---

### One-line intuition

> **`quantization_local_shard_count` shards the quantization range-finding operation across slices to avoid cross-slice communication — the default `-1` (= number of slices) is almost always correct; only tune it if per-shard calibration demonstrably hurts accuracy.**
