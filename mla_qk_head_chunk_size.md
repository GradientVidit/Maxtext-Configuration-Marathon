
## 1. Why does `mla_qk_head_chunk_size` exist?

MLA (Multi-head Latent Attention) is DeepSeek's attention variant that uses a low-rank decomposition of K and V to compress the KV cache. During attention computation, MLA needs to evaluate the QK product across all attention heads.

The problem: evaluating all heads simultaneously requires materializing the full QK matrix in HBM (High Bandwidth Memory):

```text
Full QK matrix size = batch × num_heads × seq_len × seq_len
```

At large `num_heads` and long `seq_len`, this HBM footprint becomes prohibitive. If it exceeds available HBM, XLA either fails or spills to slower memory.

`mla_qk_head_chunk_size` limits this by evaluating the QK matrix **sequentially across chunks of heads**:

```text
mla_qk_head_chunk_size=64:

Iteration 0: evaluate heads 0–63   → QK chunk → attn output chunk
Iteration 1: evaluate heads 64–127  → QK chunk → attn output chunk
...
accumulate outputs → final attention result
```

This trades compute (redundant operations per chunk) for peak HBM reduction.

---

## 2. Default

```yaml
mla_qk_head_chunk_size: 0
```

`0` = disabled. All heads are evaluated simultaneously in one pass. This is the fastest option when HBM is sufficient.

---

## 3. When MLA is active

This parameter **only matters when `attention_type: "mla"` is set**. For standard MHA, GQA, or MQA attention, the standard QK computation doesn't have the same HBM profile, and this parameter has no effect.

---

## 4. The speed vs. memory tradeoff

```text
mla_qk_head_chunk_size=0   (no chunking):
    Peak HBM: full QK matrix materialized at once
    Speed: fastest (single kernel call)
    
mla_qk_head_chunk_size=N:
    Peak HBM: QK for N heads only at any time
    Speed: slower by roughly num_heads/N × overhead
    
Benefit: enables MLA on hardware where full QK would OOM
```

---

## 5. Choosing a chunk size

The chunk size should be:
- A divisor of `base_num_query_heads` (otherwise the last chunk is irregular)
- As large as HBM allows (larger = fewer iterations = faster)
- A power of 2 or multiple of 32 for hardware alignment

Example: with 128 query heads:
```text
chunk=128  → equivalent to disabled (all at once)
chunk=64   → 2 iterations, 50% peak HBM reduction
chunk=32   → 4 iterations, 75% peak HBM reduction
chunk=16   → 8 iterations, 87.5% peak HBM reduction
```

---

## 6. Interaction with sharding

In tensor-parallel setups, `base_num_query_heads` is divided across devices. The effective heads per device is `total_heads / tensor_parallel_degree`. `mla_qk_head_chunk_size` operates on this local (per-device) head count.

---

### One-line intuition

> **`mla_qk_head_chunk_size` reduces peak HBM usage in MLA attention by evaluating the QK matrix in sequential head-chunks instead of all at once — set `0` (disabled) unless you're OOMing specifically from MLA's QK materialization.**
