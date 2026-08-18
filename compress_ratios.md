## 1. Why does `compress_ratios` exist?

In deep neural networks, early layers, middle layers, and late layers serve distinct representational roles:
- **Early layers:** Require high-resolution, uncompressed attention to parse local syntax and token order.
- **Late / Deep layers:** Process high-level abstract semantics and can tolerate aggressive compression and heavy-hitter token downsampling without losing task accuracy.

`compress_ratios` provides **per-layer compression configuration** for Compressed Attention:

```text
Example: compress_ratios = [0, 4, 16, 128] for a 4-layer model

Layer 0 (Ratio = 0):   Dense Uncompressed Attention (100% tokens attended)
Layer 1 (Ratio = 4):   4x Compression (attends to every 4th token / 25% budget)
Layer 2 (Ratio = 16):  16x Compression (attends to top clusters / 6.25% budget)
Layer 3 (Ratio = 128): 128x Heavy Compression (attends strictly to macro landmarks / <1% budget)
```

---

## 2. Options & Defaults

| Value | Behavior | Notes |
|---|---|---|
| `[]` | Disabled / empty list. Standard uniform attention across all layers. | **Default**. |
| List of integers (e.g. `[0, 0, 4, 4, 16, 16]`) | Specifies the compression stride / downsampling ratio for each successive layer. | `0` indicates no compression (dense attention) for that layer. |

Default in `base.yml`: `[]`

---

## 3. List Length Alignment

The length of `compress_ratios` must match the total number of decoder layers (`base_num_decoder_layers`). If configured, each integer entry $r_l$ in the list dictates the compression factor applied at layer $l$:
- $r_l = 0$: Layer $l$ runs full dense attention.
- $r_l > 0$: Layer $l$ compresses Key/Value sequence length by a factor of $r_l$.

---

## 4. Interactions with other parameters

| Related Parameter | Interaction |
|---|---|
| [[attention_type]] | Active when `attention_type: 'compressed'`. |
| [[compressed_rope_max_timescale]] | Sets the RoPE timescale for compressed attention layers. |
| [[o_lora_rank]], [[o_groups]] | Companion parameters configuring the output projection in compressed layers. |

---

## 5. Practical Scenarios

- **Progressive Context Compression:** Use `compress_ratios: [0, 0, 2, 2, 4, 4, 8, 8, 16, 16]` to maintain dense precision in shallow layers while scaling sub-quadratic throughput in deeper layers.
- **Uniform Dense Training:** Keep `compress_ratios: []` (default).

---

### One-line intuition

> **`compress_ratios` defines a per-layer list of compression factors (e.g. `[0, 4, 128]`), allowing shallow layers to retain dense attention while deeper layers use aggressive sequence downsampling.**
