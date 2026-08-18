## 1. Why does `moba_topk` exist?

After the MoBA router computes relevance affinity scores between a query and all $M = N / B$ sequence blocks, it must decide how many blocks to attend to.

`moba_topk` sets the exact number of top-ranking blocks ($k$) selected per query:

```text
Query q ──> Block Relevance Router
                 │
                 ├── Block 0 Score: 0.12
                 ├── Block 1 Score: 0.89  <── Selected (Top 1)
                 ├── Block 2 Score: 0.05
                 ├── Block 3 Score: 0.74  <── Selected (Top 2)
                 ├── Block 4 Score: 0.41
                 └── Block 5 Score: 0.65  <── Selected (Top 3)

With moba_topk = 3: Query evaluates fine attention ONLY over Blocks 1, 3, and 5.
```

`moba_topk` is the primary **sparsity and quality dial** in MoBA.

---

## 2. Options & Defaults

| Value | Behavior | Notes |
|---|---|---|
| `8` | Each query attends to its top-8 most relevant blocks. | **Default**. |
| Any integer $> 0$ | Number of blocks $k$ attended per query. | When $k = M$ (all blocks), MoBA degenerates to full dense attention. |

Default in `base.yml`: `8`

---

## 3. Total Context Budget Math

$$\text{Active Context per Query} = k 	imes \text{moba\_chunk\_size}$$

$$\text{Default Setup:}\quad 8 	imes 1024 = 8{,}192 \text{ tokens}$$

| Sequence Length ($N$) | Full Attention FLOPs | MoBA ($k=8, B=1024$) FLOPs | FLOP Reduction |
|---|---|---|---|
| $16{,}384$ (16K) | $16\text{K} 	imes 16\text{K} pprox 268\text{M}$ | $16\text{K} 	imes 8\text{K} pprox 134\text{M}$ | $2.0	imes$ |
| $65{,}536$ (64K) | $64\text{K} 	imes 64\text{K} pprox 4.29\text{B}$ | $64\text{K} 	imes 8\text{K} pprox 536\text{M}$ | $8.0	imes$ |
| $262{,}144$ (256K) | $256\text{K} 	imes 256\text{K} pprox 68.7\text{B}$ | $256\text{K} 	imes 8\text{K} pprox 2.14\text{B}$ | $\mathbf{32.0	imes}$ |

As context length grows, compute scales **linearly with sequence length** ($O(N)$) rather than quadratically.

---

## 4. Interactions with other parameters

| Related Parameter | Interaction |
|---|---|
| [[moba]] | Prerequisite flag. |
| [[moba_chunk_size]] | Multiplier determining total active tokens per query ($k \cdot B$). |

---

## 5. Practical Scenarios

- **Pretraining Long Context Models:** Set `moba_topk: 8` with `moba_chunk_size: 1024` for standard 128K context scaling.
- **Needle-In-A-Haystack Optimization:** If the model struggles with multi-document retrieval across 256K context, increase `moba_topk: 12` or `16` to expand the retrieval candidate pool.

---

### One-line intuition

> **`moba_topk` defines how many blocks ($k=8$) each query attends to in Mixture of Block Attention, governing the trade-off between retrieval recall and computational speedup.**
