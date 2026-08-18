
## 1. Why does `global_head_dim` exist?

Gemma4 introduced a hybrid local/global attention architecture — alternating between **sliding-window (local)** attention layers and **full (global)** attention layers. These two layer types can, and in Gemma4 do, use **different head dimensions**:

```text
Gemma4 layer stack:
  Layer 0: local sliding-window attention  → head_dim (standard)
  Layer 1: local sliding-window attention  → head_dim (standard)
  ...
  Layer K: global full attention           → global_head_dim (larger)
  ...
```

The global attention layers need to capture long-range dependencies across the full context — giving them a larger per-head subspace (`global_head_dim`) allows them to encode more nuanced position/content information without increasing all layers' memory.

This parameter exists because MaxText's general attention infrastructure uses `head_dim` for all layers, and `global_head_dim` is the escape hatch for architectures where global layers need a different value.

---

## 2. Default

```yaml
global_head_dim: 0
```

`0` means "this parameter is unused." If `decoder_block` is not Gemma4 (or a similar hybrid architecture), this has no effect whatsoever.

---

## 3. When it's active

Only when:
1. `decoder_block: "gemma4"` (or an architecture that mixes local/global attention layers)
2. The model's `global_num_kv_heads > 0` (i.e., global layers exist)

In that case, global attention layers use `global_head_dim` for their Q/K/V projections, while local layers use `head_dim`.

---

## 4. Interaction with `global_num_kv_heads`

These two parameters always appear together for Gemma4:

```yaml
global_head_dim: 256       # head dim for global layers
global_num_kv_heads: 4     # KV heads for global layers
head_dim: 128              # head dim for local layers
base_num_kv_heads: 4       # KV heads for local layers
```

They configure the global-layer attention independently of the local-layer attention.

---

## 5. KV cache implications

Global attention layers' KV cache uses `global_head_dim`, which can be significantly larger than `head_dim`. This means the KV cache entries for global layers are larger:

```text
KV cache per token per global layer = 2 × global_num_kv_heads × global_head_dim × dtype_bytes
```

When setting this, account for the memory difference vs. local layers.

---

## 6. Safe to ignore for most models

Unless you're specifically configuring Gemma4 or another hybrid local/global architecture, `global_head_dim` is a no-op. Leave it at 0.

---

### One-line intuition

> **`global_head_dim` is the per-head subspace dimension used exclusively by Gemma4's full-context "global" attention layers — has no effect on any other architecture.**
