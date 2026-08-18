
## 1. The embedding-dimension communication bottleneck

In expert parallelism (EP), the all-gather for token dispatch collects tokens from all EP devices. This gather has two axes it could shard across:

```text
Token axis:       split batch of tokens across EP devices
Embedding axis:   split the embedding dimension of each token across EP devices
```

`num_moe_emb_chunks` enables a pipelining strategy along the **embedding dimension**: split the embedding into N chunks and overlap the all-gather of chunk k+1 with the GMM compute on chunk k.

---

## 2. The mechanic

With `num_moe_emb_chunks=4` and embedding dim 7168:

```text
emb chunk 0 (dim 0:1792):    [all-gather] → [GMM partial] 
emb chunk 1 (dim 1792:3584):             [all-gather] → [GMM partial]
emb chunk 2 (dim 3584:5376):                          [all-gather] → [GMM partial]
emb chunk 3 (dim 5376:7168):                                       [all-gather] → [GMM partial]
                                                                                        ↓
                                                                               accumulate across chunks
```

The GMM for each embedding chunk is independent, so outputs can be accumulated as chunks complete.

---

## 3. This vs. `num_moe_token_chunks`

These are orthogonal chunking strategies:

| Param | Chunks along | Overlap type |
|---|---|---|
| `num_moe_token_chunks` | Token batch dimension | Chunk k+1 dispatch ↔ chunk k GMM |
| `num_moe_emb_chunks` | Embedding dimension | Chunk k+1 gather ↔ chunk k GMM partial result |

Both can be used together in principle, though the interaction can be complex.

---

## 4. Default

```yaml
num_moe_emb_chunks: 0
```

`0` = disabled. No embedding-dimension chunking.

---

## 5. Options

| Value | Meaning |
|---|---|
| `0` (default) | Disabled |
| `N > 0` | Split embedding into N chunks for overlap |

---

## 6. When to use it

**Embedding-dimension is large and embedding all-gather is the bottleneck:** enables hiding that communication.

**Token batch is small but embedding is large:** `num_moe_emb_chunks` may be more beneficial than `num_moe_token_chunks` in this shape.

**In practice:** this is a benchmarking knob. Profile first, then decide if embedding-dim chunking helps at your specific model/hardware shape.

---

### One-line intuition

> **`num_moe_emb_chunks` splits the token embedding dimension into N chunks to overlap the MoE token all-gather with GMM computation — a latency-hiding technique complementary to `num_moe_token_chunks`, but along the embedding axis instead of the token axis.**
