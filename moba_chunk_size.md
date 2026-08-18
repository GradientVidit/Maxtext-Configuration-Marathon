## 1. Why does `moba_chunk_size` exist?

In Mixture of Block Attention (MoBA), the input sequence is divided into non-overlapping blocks ("chunks"). `moba_chunk_size` defines the number of tokens ($B$) contained in each block.

The chunk size determines the **granularity of the sparse routing decision**:

```text
Sequence of N = 4096 Tokens:

moba_chunk_size B = 256 (Fine-Grained):
  Sequence = 16 Blocks of 256 tokens.
  Router selects Top-4 blocks: Query attends to exactly 1024 tokens across 4 distinct regions.
  Higher router selectivity, slightly higher routing overhead.

moba_chunk_size B = 1024 (Coarse-Grained, Default):
  Sequence = 4 Blocks of 1024 tokens.
  Router selects Top-2 blocks: Query attends to 2048 tokens in large contiguous memory blocks.
  Maximizes accelerator GEMM throughput on TPU/GPU hardware.
```

---

## 2. Options & Defaults

| Value | Behavior | Notes |
|---|---|---|
| `1024` | Block size $B=1024$ tokens. | **Default**. Optimal alignment with TPU / GPU block GEMM tile dimensions. |
| Any integer $> 0$ (e.g. `256`, `512`, `2048`) | Custom block size. | Must evenly divide the training sequence length. |

Default in `base.yml`: `1024`

---

## 3. Hardware Alignment vs. Routing Specificity Tradeoff

$$\text{Effective Attended Context} = \text{moba\_topk} 	imes \text{moba\_chunk\_size}$$

1. **Smaller Chunks ($B=256$):** Allows queries to pinpoint small, isolated needles of relevant information across a long document, but produces smaller matmuls that may under-utilize TPU matrix units (MXUs).
2. **Larger Chunks ($B=1024$):** Maximizes systolic array tensor utilization by executing large contiguous GEMMs ($1024 	imes 1024$), but pulls in surrounding irrelevant tokens alongside relevant ones.

---

## 4. Interactions with other parameters

| Related Parameter | Interaction |
|---|---|
| [[moba]] | Parent switch: `moba_chunk_size` is active only when `moba: true`. |
| [[moba_topk]] | Together, `topk * chunk_size` defines the total number of key-value tokens evaluated per query. |

---

## 5. Practical Scenarios

- **TPU / GPU Cluster Pretraining:** Keep `moba_chunk_size: 1024` to match native accelerator matrix tiling rules.
- **Short-to-Medium Sequences ($N \le 16\text{K}$):** Use `moba_chunk_size: 256` or `512` so the sequence contains enough total blocks for dynamic top-$k$ routing to be meaningful.

---

### One-line intuition

> **`moba_chunk_size` sets the token block width ($B=1024$) in Mixture of Block Attention, balancing fine-grained routing selectivity against hardware GEMM efficiency.**
