
## 1. Why does `head_dim` exist separately?

In multi-head attention, each head computes attention over a low-dimensional subspace of the residual stream. The relationship between the three key dimensions is:

```text
emb_dim = num_query_heads × head_dim  (canonical constraint)

Q projection: [emb_dim] → [num_heads, head_dim]
K projection: [emb_dim] → [num_kv_heads, head_dim]
V projection: [emb_dim] → [num_kv_heads, head_dim]
```

`head_dim` controls how large each head's subspace is. Without an explicit `head_dim` parameter, you'd have to derive it from `emb_dim / num_heads` — which breaks as soon as you want to use different head_dim vs. what that ratio gives.

`head_dim` is made explicit because:
- Some architectures use non-standard head_dim (e.g., MLA uses a compressed KV dimension that differs from query head_dim)
- GQA configs may want larger head_dim with fewer heads vs. smaller head_dim with more heads
- RoPE positional encoding scales with head_dim, so it matters for position encoding bandwidth

---

## 2. Default

```yaml
head_dim: 128
```

128 is the standard for most modern LLMs. Combined with `base_num_query_heads: 16` and `base_emb_dim: 2048`: `16 × 128 = 2048`. ✓

---

## 3. The canonical constraint

```text
base_emb_dim = base_num_query_heads × head_dim
```

This should hold in normal configs. If it doesn't, MaxText still works — the attention output projection handles the mismatch via `attention_output_dim` — but deviating from this is unusual.

---

## 4. Impact on compute

Each attention head computes:

```text
QK^T: [seq_len, head_dim] × [head_dim, seq_len] → [seq_len, seq_len]
```

Compute per head scales linearly with `head_dim` for the projection matmuls and is independent for the attention itself (the `seq × seq` part dominates at long context). Larger `head_dim` → more expressive per-head representations, but diminishing returns beyond ~128–256.

---

## 5. Flash attention alignment

Flash attention implementations often have alignment requirements on `head_dim`. Common required values are powers of 2 (64, 128, 256) or multiples of 64. Non-aligned `head_dim` values can cause fallback to slower implementations.

MaxText's default of 128 is the sweet spot: supports all major flash attention backends.

---

## 6. `head_dim` vs. `global_head_dim`

`head_dim` is the standard per-head dimension used in all normal attention layers.

`global_head_dim` is Gemma4-specific — used only for that model's "global" attention layers that operate at a different granularity than the local sliding-window layers. See [[global_head_dim]].

---

## 7. RoPE interaction

RoPE (Rotary Position Embedding) encodes position by rotating pairs of values within `head_dim`. The maximum representable frequency in RoPE scales with `head_dim` — larger `head_dim` → more frequency "slots" → better long-range position encoding. At 128, standard RoPE works well up to ~100k tokens with appropriate `rope_max_timescale` tuning.

---

### One-line intuition

> **`head_dim` is the size of each attention head's subspace — 128 is the modern standard, it must satisfy `emb_dim = num_q_heads × head_dim`, and it determines flash attention alignment and RoPE encoding bandwidth.**
