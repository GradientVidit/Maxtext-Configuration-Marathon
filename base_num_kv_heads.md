
## 1. Why does `base_num_kv_heads` exist separately from query heads?

Standard multi-head attention has a 1:1 ratio between query heads and key/value heads. This is elegant, but expensive at inference — the KV cache grows linearly with `num_heads`.

The insight behind GQA: **queries need many heads to attend to diverse patterns, but keys and values don't need to be unique per query head**. Multiple query heads can share a single K,V pair with minimal quality loss.

`base_num_kv_heads` is the lever to control this sharing.

---

## 2. The three attention modes

```text
base_num_kv_heads == base_num_query_heads
    → Multi-Head Attention (MHA)
    → Every Q head has its own K, V
    → Max expressiveness, max KV cache

base_num_kv_heads < base_num_query_heads (but > 1)
    → Grouped-Query Attention (GQA)
    → Groups of Q heads share one K, V
    → Intermediate KV cache, ~same quality as MHA

base_num_kv_heads == 1
    → Multi-Query Attention (MQA)
    → All Q heads share one K, V
    → Minimum KV cache, some quality degradation
```

---

## 3. Mechanics of GQA

With `base_num_query_heads=32` and `base_num_kv_heads=8`:

```text
Q heads:  Q_0  Q_1  Q_2  Q_3  Q_4  Q_5  Q_6  Q_7  ... Q_31
                │                   │
                └── all share KV_0  └── all share KV_1
              (group of 4)        (group of 4)
```

Each group of 4 Q heads shares one K head and one V head. The KV projection is 4× smaller than a full MHA projection.

---

## 4. KV cache arithmetic

```text
KV cache bytes per token =
    2 × base_num_kv_heads × head_dim × num_layers × dtype_bytes

Llama 2 7B  (MHA, 32 KV heads):  2×32×128×32×2 = 524 KB/token
Llama 2 70B (GQA, 8 KV heads):   2×8×128×80×2  = 327 KB/token
```

At a 100k-token context: 52 GB vs 33 GB just for KV cache. GQA is not optional at long contexts.

---

## 5. Default

```yaml
base_num_kv_heads: 16
```

Equal to `base_num_query_heads: 16` → plain MHA in the default config.

---

## 6. Divisibility constraints

`base_num_query_heads` must be divisible by `base_num_kv_heads`:

```text
32 Q heads / 8 KV heads = 4  ✓ (group size 4)
32 Q heads / 6 KV heads = 5.3 ✗  (invalid)
```

MaxText will error if this constraint is violated.

---

## 7. Interaction with `head_dim`

`base_num_kv_heads × head_dim` gives the dimension of K and V projections. This is smaller than the full `base_emb_dim` in GQA configs — the K and V weight matrices are narrower than Q.

---

## 8. When to use what

| Scenario | Choice |
|---|---|
| Training quality research | MHA (num_kv = num_q) |
| Production inference, medium context | GQA (num_kv = num_q / 4 or / 8) |
| Extreme context length (>64k tokens) | GQA or MQA (num_kv = 1) |
| Matching a published model exactly | Check the paper's head config |

---

### One-line intuition

> **`base_num_kv_heads` controls how many unique key/value projections exist — setting it below `base_num_query_heads` enables GQA/MQA, which shrinks the KV cache proportionally and is essential for long-context inference efficiency.**
