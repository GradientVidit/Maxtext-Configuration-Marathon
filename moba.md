## 1. Why does `moba` exist?

As context windows scale to 128K, 1M, or beyond, full dense attention becomes computationally intractable ($O(N^2)$ FLOPs). Static sparse attention patterns (e.g. fixed sliding windows or dilated strides) reduce compute but fundamentally restrict long-range associative recall.

**Mixture of Block Attention (MoBA)** (introduced by **Moonshot AI**, 2025) is a dynamic sparse attention mechanism inspired by Mixture of Experts (MoE). 

Instead of attending to all tokens or relying on a static window, MoBA partitions the sequence into non-overlapping blocks of size `moba_chunk_size` ($B$). Each query dynamically routes to and attends only to its **top-$k$ most relevant blocks**:

```text
Sequence of N Tokens partitioned into M Blocks of size B:

Full Attention: Every query attends to all N tokens ──> O(N²) FLOPs

MoBA (moba=True, moba_topk=k):
  1. Partition sequence into M = N / B blocks.
  2. Compute coarse block-level relevance scores for each query.
  3. Select top-k blocks (moba_topk) per query.
  4. Compute fine-grained attention strictly over selected k * B tokens ──> O(N · k · B) FLOPs
```

---

## 2. Options & Defaults

| Value | Behavior | Notes |
|---|---|---|
| `false` | MoBA disabled. Standard dense attention is evaluated. | **Default**. Standard for fixed short/medium sequence models. |
| `true` | Enables Mixture of Block Attention dynamic sparse routing. | Pairs with `moba_chunk_size` and `moba_topk`. |

Default in `base.yml`: `false`

---

## 3. Computational Complexity and Speedup

At sequence length $N = 131{,}072$ (128K tokens) with `moba_chunk_size: 1024` and `moba_topk: 8`:
- **Full Attention Tokens per Query:** $131{,}072$ tokens.
- **MoBA Attended Tokens per Query:** $k 	imes B = 8 	imes 1024 = 8{,}192$ tokens.
- **Theoretical Attention FLOP Reduction:** $rac{131{,}072}{8{,}192} = \mathbf{16	imes \text{ Speedup}}$.

---

## 4. Interactions with other parameters

| Related Parameter | Interaction |
|---|---|
| [[moba_chunk_size]] | Sets the block width $B$ (default `1024`). |
| [[moba_topk]] | Sets the number of active blocks $k$ routed per query (default `8`). |
| [[attention_type]] | Operates over the underlying attention projection layers. |
| [[attention]] | Requires a kernel backend capable of executing sparse block gathers. |

---

## 5. Practical Scenarios

- **Ultra-Long-Context Pretraining (64K–1M tokens):** Set `moba: true` with `moba_chunk_size: 1024` and `moba_topk: 8` to train models on book-length documents with sub-quadratic compute.
- **Fine-tuning Long-Context Extensions:** Continual pretraining of standard models with MoBA sparse layers to extend context window with minimal performance regression.

---

### One-line intuition

> **`moba=true` enables Moonshot AI's Mixture of Block Attention, dynamically routing each query to its top-$k$ most relevant context blocks to achieve $O(N \cdot k \cdot B)$ complexity on 100K+ sequences.**
