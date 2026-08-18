## 1. Why does `chunk_attn_window_size` exist?

While sliding window attention computes a rolling $W$-token window for every individual token, **Chunked Attention** partitions the entire token sequence into discrete, non-overlapping blocks ("chunks") of fixed size $C$.

Within each chunk, tokens attend causally to each other (left-to-right), while attention across chunks is also strictly causal (a token in chunk $k$ attends to its current chunk plus all preceding chunks):

```text
Sequence of 12 Tokens partitioned into Chunks of size C = 4:

Chunk 0: [t0, t1, t2, t3]       <-- Tokens attend within Chunk 0
Chunk 1: [t4, t5, t6, t7]       <-- Tokens attend within Chunk 1 + all of Chunk 0
Chunk 2: [t8, t9, t10, t11]     <-- Tokens attend within Chunk 2 + Chunks 0 & 1

Attention Mask Grid (C=4):
[ X X X X . . . . . . . . ]  Chunk 0
[ X X X X X X X X . . . . ]  Chunk 1
[ X X X X X X X X X X X X ]  Chunk 2
```

Chunking aligns attention boundaries with hardware block sizes (such as TPU VPU tiles and GPU Tensor Core tile multiples), making memory access patterns highly structured.

---

## 2. Options & Defaults

| Value | Behavior | Notes |
|---|---|---|
| `0` | Chunked attention disabled. | **Default**. |
| Any integer $> 0$ (e.g. `512`, `1024`) | Sets chunk block size $C$ for `attention_type: 'chunk'`. | Sequence length must typically be divisible by $C$. |

Default in `base.yml`: `0`

---

## 3. Chunked Attention vs. Sliding Window

```text
Feature                  Sliding Window (sliding_window_size)    Chunked (chunk_attn_window_size)
────────────────────────────────────────────────────────────────────────────────────────────────
Window Boundaries        Continuous rolling per token            Discrete static chunk boundaries
Hardware Tiling          Irregular band diagonal                 Aligned rectangular block GEMMs
Past Context Retention   Oldest tokens evicted beyond W          Retains past chunks hierarchically
Primary Advantage        Smooth local continuity                 Hardware-friendly block scheduling
```

---

## 4. Interactions with other parameters

| Related Parameter | Interaction |
|---|---|
| [[attention_type]] | Active when `attention_type: 'chunk'`. |
| [[sliding_window_size]] | Mutually exclusive attention localization strategies. |
| [[causal_block_size]] | Similar block concept, but `causal_block_size` is specifically for Block Diffusion where within-block attention is strictly bidirectional. |

---

## 5. Practical Scenarios

- **Long-Document Processing:** Process long sequences in structured $1024$-token chunks where intra-chunk information is densely processed while inter-chunk context is causally aggregated.
- **Hardware-Tiled Training:** When maximizing TPU matrix multiplication unit (MXU) efficiency by aligning attention blocks directly with native hardware tile dimensions.

---

### One-line intuition

> **`chunk_attn_window_size` divides sequences into fixed non-overlapping blocks of size $C$, enabling structured block-causal attention that matches hardware tile boundaries.**
