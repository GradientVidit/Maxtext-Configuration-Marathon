## 1. Why does `gdn_chunk_size` exist?

Linear recurrent models (RNNs, State Space Models, DeltaNet) have theoretical $O(S)$ sequential complexity. However, pure token-by-token sequential recurrence runs notoriously slowly on modern accelerator hardware (TPUs/GPUs) because matrix units sit idle waiting for step-to-step serial data dependencies.

To achieve high hardware MFU (Model FLOPs Utilization), Gated DeltaNet uses **Chunked Parallel Scan**:

```text
Naive Sequential Recurrence:
Step 1 ──> Step 2 ──> Step 3 ──> ... ──> Step S
[Serial dependencies prevent hardware parallelism; low accelerator utilization]

Chunked Parallel Scan (Chunk Size = C):
┌─── Chunk 0 (C tokens) ───┐   ┌─── Chunk 1 (C tokens) ───┐
│ Dense Matmuls within C   │   │ Dense Matmuls within C   │
└────────────┬─────────────┘   └────────────┬─────────────┘
             ▼                              ▼
      Inter-chunk Associative Parallel Prefix Scan (Across Chunks)
```

By breaking a sequence of length $S$ into chunks of length $C$, intra-chunk interactions are computed using highly parallel matrix multiplications (GEMMs), while inter-chunk state propagation is resolved via an associative parallel scan.

`gdn_chunk_size` specifies the block length $C$ for this chunked scan algorithm.

---

## 2. Mechanics & Mathematical Decomposition

The chunked parallel scan splits computation into three distinct stages:

```text
Sequence of Length S  ──>  Divided into N = S / C Chunks of size C
                                    │
                                    ▼
1. Intra-chunk Local Computation (Matrix Multiply)
   - Compute intra-chunk attention-like matrix using tensor cores:
     M_local = (Q_chunk K_chunk^T) ⊙ Mask_causal
   - Accumulate local value updates and state delta within chunk C
                                    │
                                    ▼
2. Inter-chunk State Transition (Parallel Scan)
   - Compute aggregate transition matrices A_chunk for each chunk
   - Run associative parallel prefix scan across chunk boundaries:
     S_{chunk_i} = S_{chunk_{i-1}} A_{chunk_i} + B_{chunk_i}
                                    │
                                    ▼
3. Intra-chunk State Materialization
   - Combine incoming chunk initial state S_{chunk_i} with intra-chunk tokens
```

- Larger chunk sizes increase GEMM arithmetic intensity on TPU MXUs.
- Smaller chunk sizes reduce peak intermediate memory buffers allocated for intra-chunk tensor tiles.

---

## 3. Options & Default

| Parameter | Type | Default | Options |
| :--- | :--- | :--- | :--- |
| `gdn_chunk_size` | `int` | `64` | Powers of 2, typically `32`, `64`, `128` |

---

## 4. Interactions with Related Parameters

| Related Parameter | Interaction |
| :--- | :--- |
| `max_target_length` | Total sequence length $S$ should ideally be divisible by `gdn_chunk_size` to avoid ragged remainder tiles. |
| `gdn_conv_kernel_dim` | The 1D causal conv kernel width ($K$) must be smaller than `gdn_chunk_size` ($K \le C$). |
| `gdn_key_head_dim` / `gdn_value_head_dim` | Intra-chunk GEMM matrices have dimensions $[C, d_k]$ and $[C, d_v]$. |

---

## 5. Practical Guidance & Tuning

| Chunk Size | Hardware Behavior & Trade-off | Recommendation |
| :--- | :--- | :--- |
| `gdn_chunk_size: 64` | **Standard Default**: Optimal balance of MXU compute tiling and intermediate memory overhead on TPU v4/v5/v6. | Recommended for training and long-context prefill. |
| `gdn_chunk_size: 128` | Higher arithmetic intensity on very long sequences; slightly increases activation HBM footprint. | Test on TPU v5p/v6e if training throughput is compute-bound. |
| `gdn_chunk_size: 16` or `32` | Lower memory consumption, but causes lower MXU utilization due to under-tiling. | Only useful under extreme HBM memory pressure. |

---

### One-line intuition

> `gdn_chunk_size` defines the sequence tile size for Gated DeltaNet's chunked parallel scan, balancing high-throughput tensor-core GEMMs within chunks against parallel prefix scan across chunks.
