## 1. Why it exists: explicit tensor reshaping for optimized attention contracts

In transformer attention implementations, the Query ($Q$) tensor is projected from the input hidden state and multiplied against the Key ($K$) tensor:

```text
Query Projection & Attention Contracting:
[Input: B, L, E] ──> Q_proj ──> [Q: B, L, H, D_k] ──> Reshape / Transpose ──> Dot-Product with K
                                                             │
                              ┌──────────────────────────────┴──────────────────────────────┐
                              ▼                                                             ▼
                     reshape_q: false                                              reshape_q: true
            Keeps standard 4D tensor:                                    Flattens/Reshapes dimensions:
            Shape: [Batch, Length, Heads, D_k]                           Shape: [Batch * Heads, Length, D_k]
            (Direct multi-head attention)                                (Batched 2D/3D GEMM optimization)
```

Depending on the underlying attention implementation (e.g. standard dot-product attention, Splash Attention, custom Mosaic TPU kernels, or foreign serving backends like vLLM/TPU-inference), the kernel may expect Query tensors in a specific flattened or reshaped format (such as collapsing batch and heads into a single leading batch dimension) to maximize hardware tiling efficiency on TPU Matrix Units (MXUs).

`reshape_q` is a low-level layout control flag that explicitly toggles reshaping the Query tensor prior to the attention score computation.

---

## 2. Mechanics: tensor transformation before attention dot-product

Inside the attention computation layer:

```text
 1. Linear Projection:
    q = DenseGeneral(inputs)     # Output shape: [Batch, Length, Num_Heads, Head_Dim]
                  │
                  ▼
 2. Check: `reshape_q` flag
                  │
        ┌─────────┴─────────┐
        ▼                   ▼
 `reshape_q: false`    `reshape_q: true`
 ┌─────────────────┐   ┌────────────────────────────────────────────────────────┐
 │ Keep original   │   │ Reshape tensor dimensions:                             │
 │ 4D rank         │   │ q = q.reshape((batch * num_heads, length, head_dim))   │
 └────────┬────────┘   └────────────────────────┬───────────────────────────────┘
          │                                     │
          └─────────────────┬───────────────────┘
                            │
                            ▼
 3. Attention Score Calculation: Softmax(q @ k.T) @ v
```

When enabled:
- The Query tensor is reshaped to combine outer dimensions, allowing XLA to lower the operation as a single large contiguous batched matrix multiplication (`jax.lax.batch_matmul`) rather than looping or broadcasting over multiple distinct rank dimensions.
- The resulting attention output tensor is reshaped back to canonical model dimensions `[Batch, Length, Hidden_Dim]` after attention aggregation.

---

## 3. Options & Configuration

Default in `base.yml`:

```yaml
reshape_q: false
```

| Value | Query Tensor Handling | Hardware & Kernel Effect | Recommended For |
|---|---|---|---|
| `false` (default) | Preserves standard 4D rank `[Batch, Length, Heads, D_k]`. | Relies on XLA compiler pattern-matching to fuse multi-dimensional attention. | Standard training and default MaxText attention pipelines. |
| `true` | Explicitly reshapes Query dimensions prior to contracting. | Forces 2D/3D batched GEMM layout; matches specific optimized kernel interfaces. | Specialized TPU inference kernels, custom serving integrations, or matching vLLM / TPU-inference layout contracts. |

---

## 4. Parameter Interactions

```text
┌───────────────────────────────────────────────────────────┐
│                         reshape_q                         │
└─────────────┬───────────────────────────────┬─────────────┘
              │
              ▼
┌───────────────────────────────────────────────────────────┐
│ Interacts directly with:                                  │
│ - compute_axis_order (determines initial axis positions)  │
│ - attention (e.g. dot_product, flash, splash)             │
│ - prefill_cache_axis_order / ar_cache_axis_order          │
└───────────────────────────────────────────────────────────┘
```

- **`compute_axis_order`**: Dictates the starting shape of the Query tensor before `reshape_q` is applied.
- **`attention`**: When using custom attention backends (like `dot_product` vs `flash`), kernel requirements dictate whether `reshape_q` should be toggled.

---

## 5. Practical Scenarios & Failure Modes

### Fine-Tuning Attention Kernel Throughput
In specialized low-latency decode serving environments, compiling attention as a reshaped batch GEMM can reduce XLA instruction overhead and eliminate intermediate tensor copies.

### What breaks if misconfigured:
- **Attention Kernel Shape Assertion**: Toggling `reshape_q: true` in combination with an attention kernel that expects a strict 4D layout will trigger a tensor rank mismatch error during graph compilation.

---

### One-line intuition

> **`reshape_q` toggles explicit dimension reshaping on the Query tensor before attention contracting, formatting tensor ranks to match specialized hardware GEMM layouts and custom inference kernels.**
