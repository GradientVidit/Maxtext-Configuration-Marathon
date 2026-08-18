## 1. Why does `compressed_rope_max_timescale` exist?

Rotary Position Embeddings (RoPE) assign sinusoidal frequencies to attention dimensions using a geometric base timescale $	heta$:

$$\omega_k = rac{1}{	heta^{2k / d}} \quad \text{where } 	heta = \text{rope\_max\_timescale}$$

When attention is compressed by downsampling or striding tokens (via `compress_ratios`), the effective sequence distance between adjacent compressed tokens increases by the compression factor $r$. 

If standard RoPE frequencies were used on compressed tokens, positional rotations would rotate too quickly, causing high-frequency positional aliasing.

`compressed_rope_max_timescale` sets a dedicated, expanded RoPE base timescale ($	heta_{\text{compressed}}$) **specifically for Compressed Attention layers**:

$$	heta_{\text{compressed}} = \text{compressed\_rope\_max\_timescale} \quad (\text{Default: } 160{,}000)$$

```text
Standard Attention Layers:
  RoPE Base Timescale = 10,000 (Standard rotation for uncompressed token positions [0, 1, 2, 3...])

Compressed Attention Layers (compressed_rope_max_timescale = 160_000):
  RoPE Base Timescale = 160,000 (Slower rotation, matching compressed token distances [0, 16, 32, 48...])
```

---

## 2. Options & Defaults

| Value | Behavior | Notes |
|---|---|---|
| `160000` | Slower RoPE base frequency scale of $	heta = 160{,}000$. | **Default**. Calibrated for up to 160K compressed sequence lengths. |
| `0` or negative | Disabled. Compressed layers inherit the model's global default `rope_max_timescale`. | |
| Any integer $> 0$ | Custom RoPE max timescale for compressed layers. | |

Default in `base.yml`: `160_000`

---

## 3. Positional Extrapolation at Long Sequence Lengths

Larger base timescales ($	heta \ge 160{,}000$) slow down the rotation of high-dimensional rotary features:
- Preserves relative distance sensitivity across hundreds of thousands of tokens.
- Eliminates positional out-of-distribution artifacts when extrapolating compressed layers to long contexts.

---

## 4. Interactions with other parameters

| Related Parameter | Interaction |
|---|---|
| [[compress_ratios]] | Active on layers where $r_l > 0$. |
| [[attention_type]] | Active when `attention_type: 'compressed'`. |
| [[rope_max_timescale]] | Global default RoPE timescale for uncompressed layers. |

---

## 5. Practical Scenarios

- **Training Compressed Long-Context Models:** Leave `compressed_rope_max_timescale: 160000` (default) to ensure downsampled attention layers maintain stable positional encoding across 128K+ sequences.

---

### One-line intuition

> **`compressed_rope_max_timescale` sets a slower RoPE rotation timescale ($	heta=160{,}000$) for Compressed Attention layers, preventing positional aliasing across downsampled token distances.**
