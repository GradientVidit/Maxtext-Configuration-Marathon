## 1. Why it exists: hardware-aligned memory strides for prompt KV cache

In multi-head attention, the Key-Value (KV) cache is logically a 4-dimensional tensor with four distinct axes:

```text
Logical Axes:
  0: CACHE_BATCH    (B - batch size / concurrent requests)
  1: CACHE_SEQUENCE (L - sequence length / context positions)
  2: CACHE_HEADS    (H - number of KV attention heads)
  3: CACHE_KV       (D - head dimension / d_kv)
```

However, hardware accelerators (like TPU Matrix Units and GPU Tensor Cores) do not access multidimensional tensors abstractly—they execute vector loads and matrix multiplies across physical contiguous memory buffers. The **stride order** of these dimensions determines:
1. **Memory coalescing and DMA burst throughput**: Loading contiguous blocks of HBM vs. scattered strided reads.
2. **XLA fusion efficiency**: Enabling XLA to fuse attention projections directly into cache update operations without inserting costly transpose / copy kernels.
3. **Serving framework interoperability**: Matching the layout expectations of external serving engines (e.g. JetStream, vLLM, TPU-inference).

`prefill_cache_axis_order` defines the physical permutation of the four cache axes for the KV cache produced during the **prompt prefill stage**.

---

## 2. Mechanics: axis permutation and memory layout

The logical layout is indexed as `0, 1, 2, 3 = (BATCH, SEQUENCE, HEADS, D_KV)`.

When `prefill_cache_axis_order` is set, MaxText transposes and stores the prefill cache according to the specified permutation string:

```text
Logical Shape:
  [Batch, Sequence, Heads, D_kv]  ── (0, 1, 2, 3)
                │
                ▼ Permutation: "1,2,0,3" (Default)
Physical Memory Layout:
  [Sequence, Heads, Batch, D_kv]  ── (1, 2, 0, 3)
```

### Why `"1,2,0,3"` is the default:
- **Major dimension = Sequence (1)**: Allows indexing and slicing along the time/position dimension efficiently.
- **Inner contiguous dimension = D_KV (3)**: Preserves contiguous vector registers for head projections ($d_{\text{kv}} = 64, 128$), aligning with the 128-byte hardware vector register width on Cloud TPUs.
- **Middle dimensions = (Heads, Batch)**: Optimizes multi-head broadcasting and sharding across tensor parallel axes.

---

## 3. Options & Configuration

Default in `base.yml`:

```yaml
prefill_cache_axis_order: "1,2,0,3"
```

| Permutation String | Physical Dimension Ordering | Characteristics | Use Case |
|---|---|---|---|
| `"1,2,0,3"` (default) | `(SEQUENCE, HEADS, BATCH, D_KV)` | Optimal for XLA attention fusions on Cloud TPUs; minimizes memory padding in prefill. | Standard MaxText prefill on TPU v4/v5e/v5p/v6e. |
| `"0,1,2,3"` | `(BATCH, SEQUENCE, HEADS, D_KV)` | Canonical standard logical layout. | Direct compatibility with standard PyTorch / HuggingFace tensor formats. |
| `"0,2,1,3"` | `(BATCH, HEADS, SEQUENCE, D_KV)` | Traditional Multi-Head Attention (MHA) layout. | Frameworks like vLLM / TensorRT-LLM that expect contiguous sequence per head. |

---

## 4. Parameter Interactions

```text
┌───────────────────────────────────────────────────────────┐
│                 prefill_cache_axis_order                  │
└─────────────┬───────────────────────────────┬─────────────┘
              │                               │
              ▼                               ▼
┌───────────────────────────┐   ┌───────────────────────────┐
│ar_cache_axis_order        │   │compute_axis_order         │
│Decode phase cache layout. │   │Physical axis layout for   │
│Often kept identical to    │   │attention compute tensors  │
│avoid transpose on handoff.│   │(Q, K, V activations).     │
└───────────────────────────┘   └───────────────────────────┘
              │
              ▼
┌───────────────────────────┐
│stack_prefill_result_cache │
│Stacks prefill cache across│
│layers preserving this axis│
│order.                     │
└───────────────────────────┘
```

- **`ar_cache_axis_order`**: While prefill and decode caches can technically have different layouts, keeping them identical (`"1,2,0,3"`) eliminates transpositions when handing off the prefill cache to the decode generation loop.
- **`compute_axis_order`**: If `compute_axis_order` differs from `prefill_cache_axis_order`, XLA must emit layout conversion copies prior to attention dot-product operations.
- **`reshape_q`**: Complements axis layout choices to align query matrices with key/value memory strides.

---

## 5. Practical Scenarios & Failure Modes

### Performance Impact of Layout Mismatches
Changing `prefill_cache_axis_order` does not alter model math or output tokens, but it drastically impacts HBM throughput:
- **Optimal (`"1,2,0,3"`)**: Attention dot products execute with peak memory bandwidth.
- **Suboptimal layout**: XLA inserts explicit `transpose` instructions in the HLO graph, increasing memory traffic and reducing prefill TFLOP/s by 15–30%.

### What breaks if misconfigured:
- **Invalid Permutation String**: Providing strings that do not contain all 4 indices (e.g. `"1,2,3"` or `"0,1,2,4"`) triggers a configuration parsing validation error on startup.

---

### One-line intuition

> **`prefill_cache_axis_order` controls the physical memory stride permutation of the KV cache during prompt ingestion, optimizing TPU memory coalescing and eliminating runtime transposes in attention kernels.**
