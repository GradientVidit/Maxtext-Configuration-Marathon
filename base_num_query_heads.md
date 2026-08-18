
## 1. Why does the query head count exist?

Multi-head attention computes `num_heads` parallel attention distributions over the sequence. Each head learns to attend to different relationships:

```text
Input: [batch, seq_len, emb_dim]
          │
          ├─ Q_head_0: [seq_len, head_dim] ──→ attn_head_0
          ├─ Q_head_1: [seq_len, head_dim] ──→ attn_head_1
          │  ...
          └─ Q_head_N: [seq_len, head_dim] ──→ attn_head_N
                                                    │
                                               concatenate
                                                    │
                                               output proj → [seq_len, emb_dim]
```

`base_num_query_heads` is the number of those parallel attention heads that compute **queries** (Q). This is always ≥ the number of KV heads.

---

## 2. Default

```yaml
base_num_query_heads: 16
```

With `head_dim: 128`, this gives 16 × 128 = 2048 total query projection dimension, matching `base_emb_dim: 2048`.

---

## 3. Relationship to `head_dim` and `base_emb_dim`

The standard constraint:

```text
base_emb_dim = base_num_query_heads × head_dim
```

Example: `4096 = 32 × 128`. When this holds, the Q projection is a square reshape of the residual stream — the most natural configuration.

---

## 4. Relationship to `base_num_kv_heads`

This is the key design decision for attention efficiency:

```text
base_num_query_heads  vs  base_num_kv_heads
─────────────────────────────────────────────
32  ==  32   → Multi-Head Attention (MHA)    — standard
32  ==  8    → Grouped-Query Attention (GQA) — LLaMA 2 70B, Mistral
32  ==  1    → Multi-Query Attention (MQA)   — original PaLM
```

**MHA**: every Q head has its own K and V. Full expressiveness, highest KV cache memory.

**GQA**: groups of Q heads share one K, V pair. Reduces KV cache by `num_q_heads / num_kv_heads`. No significant quality loss empirically (LLaMA 2 style).

**MQA**: all Q heads share one K, V. Maximum KV cache reduction, but can hurt quality on some tasks.

---

## 5. KV cache memory impact

```text
KV cache per token = 2 × base_num_kv_heads × head_dim × num_layers × dtype_bytes

MHA (32 KV heads): 2 × 32 × 128 × 32 × 2 = 524,288 bytes/token
GQA (8 KV heads):  2 × 8  × 128 × 32 × 2 = 131,072 bytes/token  (4× smaller)
```

GQA's cache reduction directly enables larger batch sizes or longer sequences at inference.

---

## 6. Common values by model size

| Model | base_num_query_heads | base_num_kv_heads | Style |
|---|---|---|---|
| Default (~117M) | 16 | 16 | MHA |
| LLaMA 2 7B | 32 | 32 | MHA |
| LLaMA 2 70B | 64 | 8 | GQA |
| Mistral 7B | 32 | 8 | GQA |
| Gemma 2 9B | 16 | 8 | GQA |

---

## 7. Sharding

`base_num_query_heads` must be divisible by the tensor parallelism degree. With 8-way tensor parallelism, you need at least 8 query heads (and ideally a multiple of 8).

---

### One-line intuition

> **`base_num_query_heads` sets how many parallel attention distributions the model computes — and its ratio to `base_num_kv_heads` determines whether you're using MHA, GQA, or MQA, which directly controls KV cache size at inference.**
