## 1. Why it exists: memory layout constraints for attention compute kernels

In transformer attention computation, intermediate activation tensors (Query, Key, Value, Attention Weights) pass through a sequence of matrix multiplications, scaling, masking, and softmax operations:

```text
Attention Computation Pipeline:
[X] ──> Q, K, V Projections ──> [Compute Layout Permutation] ──> Q @ K^T ──> Softmax ──> Scores @ V
                                              │
                     ┌────────────────────────┴────────────────────────┐
                     ▼                                                 ▼
             Layout: "0,1,2,3"                                 Layout: "0,2,1,3"
       [Batch, Length, Head, D_kv]                       [Batch, Head, Length, D_kv]
       - Token-major ordering                            - Head-major ordering
       - Efficient sequence sharding                     - Efficient batched GEMM per head
```

Hardware matrix units (like TPU MXUs) require specific matrix operands to be contiguous in memory to achieve high Model Flops Utilization (MFU). While mathematically any axis ordering yields the same result, in practice only a tiny subset of tensor permutations have hand-tuned, hardware-accelerated XLA and Mosaic kernels.

`compute_axis_order` configures the physical axis permutation for general compute and activation tensors throughout attention calculations.

---

## 2. Mechanics: supported layout permutations

Logical compute axes are defined as:
```text
  0: BATCH   (Batch dimension)
  1: LENGTH  (Sequence / Token dimension)
  2: HEAD    (Number of attention heads)
  3: D_KV    (Head dimension)
```

MaxText strictly supports **two specific compute layouts**:

```text
Permutation 1: "0,1,2,3" (Default)
  Shape: [Batch, Length, Head, D_kv]
  - Token-major layout.
  - Keeps tokens contiguous within sequence blocks; well-suited for sequence parallelism and long-context chunking.

Permutation 2: "0,2,1,3"
  Shape: [Batch, Head, Length, D_kv]
  - Head-major layout.
  - Keeps sequence length contiguous per head; matches classic Multi-Head Attention (MHA) batched GEMM structures.
```

---

## 3. Options & Configuration

Default in `base.yml`:

```yaml
compute_axis_order: "0,1,2,3"
```

| Value | Tensor Physical Shape | Characteristics | Compatibility / Usage |
|---|---|---|---|
| `"0,1,2,3"` (default) | `[Batch, Length, Head, D_kv]` | Token-major; optimized for MaxText default attention fusions, Splash Attention, and TPU sharding. | Standard MaxText pretraining, fine-tuning, and default inference serving. |
| `"0,2,1,3"` | `[Batch, Head, Length, D_kv]` | Head-major; standard in PyTorch / HuggingFace MHA implementations. | Useful when integrating with external serving libraries (like vLLM, TensorRT-LLM) or benchmarking against standard head-major kernels. |

> [!WARNING]
> MaxText does **not** support arbitrary 4D permutations (e.g. `"2,0,1,3"` or `"3,2,1,0"`). Attempting to pass any value other than `"0,1,2,3"` or `"0,2,1,3"` will fail validation.

---

## 4. Parameter Interactions

```text
┌───────────────────────────────────────────────────────────┐
│                    compute_axis_order                     │
└─────────────┬───────────────────────────────┬─────────────┘
              │                               │
              ▼                               ▼
┌───────────────────────────┐   ┌───────────────────────────┐
│reshape_q                  │   │prefill_cache_axis_order / │
│Toggled in conjunction with│   │ar_cache_axis_order        │
│compute layout to align    │   │Aligns compute tensor      │
│query tensor shapes.       │   │outputs with cache memory. │
└───────────────────────────┘   └───────────────────────────┘
```

- **`reshape_q`**: Works alongside `compute_axis_order` to control whether the query tensor is reshaped or flattened before entering the attention dot-product kernel.
- **`prefill_cache_axis_order` & `ar_cache_axis_order`**: Determines whether XLA needs to perform an axis transposition when saving projected Key/Value activations into the KV cache.
- **`attention` (e.g., `flash`, `dot_product`)**: Specific attention backends are optimized for particular compute layouts.

---

## 5. Practical Scenarios & Failure Modes

### Integrating with External Serving Engines
When exporting or co-running MaxText models in environments designed around Hugging Face or vLLM memory conventions:
```yaml
compute_axis_order: "0,2,1,3"
```
This forces intermediate attention projections into head-major order `[Batch, Head, Length, D_kv]`, aligning memory layouts with downstream CUDA / Triton kernels.

### What breaks if misconfigured:
- **Unsupported layout error**: Passing an unsupported permutation (e.g., `"1,2,0,3"`) raises:
  ```text
  ValueError: Unsupported compute_axis_order: 1,2,0,3. Supported values are '0,1,2,3' and '0,2,1,3'.
  ```

---

### One-line intuition

> **`compute_axis_order` defines whether attention activations use token-major (`"0,1,2,3"`) or head-major (`"0,2,1,3"`) memory layout, aligning tensor shapes with optimized TPU matrix multiplication kernels.**
