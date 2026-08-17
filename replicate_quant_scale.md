
## 1. Why does it exist?

In quantization, each tensor that gets quantized has an associated **scale factor** — a scalar that maps the full-precision range to the quantized integer/fp8 range. Under 2D sharding (where both a data-parallel dimension and a model-parallel dimension are active), XLA sometimes generates an inefficient computation pattern for these scale factors: instead of computing and using the scale locally, it tries to fuse it across devices in a way that prevents effective XLA kernel fusion.

`replicate_quant_scale` solves this by broadcasting/replicating the scale factor across the relevant device dimension, trading a small amount of memory for better XLA kernel fusion.

---

## 2. The underlying XLA issue

Without replication, the scale tensor has a different sharding annotation than the data tensor. XLA's fusion logic requires tensors in a fused kernel to have compatible sharding — if they don't, XLA inserts explicit AllGather or reshape operations that break fusion.

```text
Without replicate_quant_scale:
  scale (unsharded) + data (sharded) → XLA can't fuse → separate ops + overhead

With replicate_quant_scale:
  scale (replicated = same shape as sharded context) + data (sharded) → fuseable → efficient
```

---

## 3. Options

| Value | Behavior |
|---|---|
| `false` | Scale not explicitly replicated — XLA decides (default) |
| `true` | Scale replicated across devices before use — forces fuseable layout |

Default in base.yml:
```yaml
replicate_quant_scale: false
```

---

## 4. When to use it

This is a **performance tuning knob**, not a correctness knob. The model produces identical outputs either way.

**Use it when:**
- Using quantization (`quantization != ""`) under 2D sharding (e.g., tensor parallelism + FSDP simultaneously)
- Profiling shows XLA is inserting unexpected AllGathers or memory operations around scale computation
- You see worse-than-expected throughput despite quantization being active

**Leave it false** when using 1D sharding or no sharding — replication adds a small memory overhead for no benefit.

---

## 5. What "2D sharding" means here

2D sharding = sharding along two mesh axes simultaneously, e.g.:
```text
mesh axes: (data, model)
weight: sharded on (data,) for FSDP
scale:   not sharded (scalar or 1D)
```

When the weight is gathered across the `data` axis and the scale is scalar, XLA may not know how to efficiently co-locate the scale computation with the gathered weight. `replicate_quant_scale=true` makes the scale explicitly available on all devices in the `data` partition.

---

## 6. Memory cost

The scale tensor is tiny compared to model weights — typically one float32 per quantized layer per axis. Replicating it across the `data` dimension adds negligible memory.

---

### One-line intuition

> **`replicate_quant_scale=true` prevents an XLA fusion break when using quantization under 2D sharding by broadcasting scale factors across devices — a pure performance knob with no effect on correctness.**
