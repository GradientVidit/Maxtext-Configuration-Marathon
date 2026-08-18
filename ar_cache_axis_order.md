## 1. Why it exists: memory-bandwidth optimization during autoregressive token decode

During autoregressive (AR) token generation, the model executes one single token step at a time for all concurrent requests:

```text
Autoregressive Step Operation:
┌────────────────────────────────────────────────────────────────────────┐
│ 1. Current Step Query: Shape [Batch, 1, Num_Heads, Head_Dim]          │
│ 2. Load KV Cache from HBM: Shape permuted by `ar_cache_axis_order`     │
│ 3. Append new Key/Value token at current sequence position             │
│ 4. Compute Attention Scores: Q @ K^T across all past positions [0...T] │
│ 5. Compute Attention Output: Softmax(Scores) @ V                       │
└────────────────────────────────────────────────────────────────────────┘
```

Because decode steps are **strictly memory-bandwidth bound** (the entire multi-gigabyte KV cache must be streamed from HBM into vector registers on every single token generation), the physical arrangement of axes in memory directly dictates whether TPU vector units can load data using coalesced burst reads or strided memory fetches.

`ar_cache_axis_order` sets the physical permutation of the four KV cache axes during the **autoregressive decode stage**.

---

## 2. Mechanics: memory layout and step slice updates

The four logical axes are defined as:
```text
  0: CACHE_BATCH    (B - active generation slots)
  1: CACHE_SEQUENCE (L - maximum context length)
  2: CACHE_HEADS    (H - number of KV heads)
  3: CACHE_KV       (D - head dimension)
```

Setting `ar_cache_axis_order` arranges the storage buffer in memory:

```text
Logical Layout:
  [Batch, Sequence, Heads, D_kv]  ── (0, 1, 2, 3)
                │
                ▼ Permutation: "1,2,0,3" (Default)
Physical Memory Layout:
  [Sequence, Heads, Batch, D_kv]  ── (1, 2, 0, 3)
```

### Why `"1,2,0,3"` optimizes decode steps:
- **Slice update along Sequence (1)**: In step $t$, updating the cache requires writing to position index $t$. In a sequence-major layout, position $t$ is a contiguous sub-tensor slice across heads and batch items.
- **Fast Contiguous Vector Loads on `D_KV` (3)**: The lowest level matrix/vector multiply loads contiguous slices of size `D_KV`, perfectly filling 128-byte TPU vector registers.
- **Batch vectorization**: Grouping `Batch` adjacent to `D_KV` maximizes parallel memory lane utilization across concurrent request streams.

---

## 3. Options & Configuration

Default in `base.yml`:

```yaml
ar_cache_axis_order: "1,2,0,3"
```

| Value | Physical Dimension Order | Properties | Primary Use Case |
|---|---|---|---|
| `"1,2,0,3"` (default) | `(SEQUENCE, HEADS, BATCH, D_KV)` | Sequence-sliced; optimal for XLA TPU memory loads and in-place dynamic slice updates. | Production serving on Google Cloud TPUs (v4, v5e, v5p, v6e). |
| `"0,1,2,3"` | `(BATCH, SEQUENCE, HEADS, D_KV)` | Standard batch-first ordering. | Interoperability with frameworks requiring batch-contiguous memory layouts. |
| `"0,2,1,3"` | `(BATCH, HEADS, SEQUENCE, D_KV)` | Head-contiguous ordering. | Used when bridging with custom CUDA kernels or Triton flash-decoding layouts. |

---

## 4. Parameter Interactions

```text
┌───────────────────────────────────────────────────────────┐
│                    ar_cache_axis_order                    │
└─────────────┬───────────────────────────────┬─────────────┘
              │                               │
              ▼                               ▼
┌───────────────────────────┐   ┌───────────────────────────┐
│prefill_cache_axis_order   │   │compute_axis_order         │
│If identical to AR layout, │   │Aligns attention step      │
│prefill-to-decode handoff  │   │computations with cache    │
│is zero-copy / no transpose│   │memory strides.            │
└───────────────────────────┘   └───────────────────────────┘
```

- **`prefill_cache_axis_order`**: Matching `ar_cache_axis_order: "1,2,0,3"` with `prefill_cache_axis_order: "1,2,0,3"` ensures that when the prefill phase finishes, the cache can be fed directly into the decode loop without triggering an expensive full-tensor copy or transpose.
- **`compute_axis_order`**: Governs the layout of intermediate activations generated during the attention step.

---

## 5. Practical Scenarios & Failure Modes

### Benchmarking Inter-Token Latency (ITL)
When profiling token generation speed on a TPU v5e pod with batch size 32:
- With `"1,2,0,3"`, the decode kernel achieves near-theoretical peak HBM read bandwidth.
- Switching to a non-contiguous stride (like `"0,1,3,2"`) forces strided gathers, causing Inter-Token Latency (ITL) to degrade significantly.

### What breaks if misconfigured:
- **XLA Compilation Failures**: Specifying invalid or unsupported permutations will cause XLA graph lowering to fail with layout assertion errors in attention custom calls.

---

### One-line intuition

> **`ar_cache_axis_order` specifies the physical memory layout of the KV cache during autoregressive generation, optimizing memory bandwidth utilization and step slice updates during iterative decoding.**
