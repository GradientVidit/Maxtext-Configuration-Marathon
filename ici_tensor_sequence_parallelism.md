## 1. Why does it exist?

Standard Tensor Parallelism replicates activation tensors along the sequence dimension across all TP devices during operations like Layer Normalization and Dropout. For long context sequences, this replicated activation memory consumes a large portion of device HBM.

Tensor-Sequence Parallelism (SP) partitions activations along the sequence length dimension in regions outside the column/row matrix multiplications:
- Before Column Matmul: **All-Gather** sequence chunks.
- After Row Matmul: **Reduce-Scatter** output activations back into sequence chunks.

```text
Standard TP:               Tensor-Sequence Parallelism (SP):
  LayerNorm: [B, S, H]       LayerNorm: [B, S/N, H]  <-- N-fold activation memory reduction!
      ↓ (All-Reduce)             ↓ (All-Gather ──→ Matmul ──→ Reduce-Scatter)
  Matmul:    [B, S, H]       Matmul:    [B, S/N, H]
```

`ici_tensor_sequence_parallelism` sets the degree of sequence-parallel tensor partitioning within a single TPU slice.

---

## 2. Options & Configuration

| Value | Meaning |
|---|---|
| `1` (default) | Standard tensor parallelism or FSDP (no sequence splitting on tensor axis). |
| Integer $> 1$ | Splits sequence activations across `N` chips over the ICI interconnect. |

Default in `base.yml`:
```yaml
ici_tensor_sequence_parallelism: 1
```

---

## 3. Interactions with Related Parameters

- **`te_comm_gemm_overlap`**: Transformer Engine's collective GEMM overlap algorithm is only valid when tensor-sequence parallelism is active.
- **`ici_tensor_parallelism`**: Operates in coordination with tensor-sequence parallelism to control intra-layer sharding.

---

### One-line intuition

> **`ici_tensor_sequence_parallelism` shards activations along the sequence dimension during LayerNorm and Dropout over the fast chip interconnect, cutting activation memory consumption across tensor-parallel devices.**
