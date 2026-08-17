
## 1. Why does it exist?

At large scale with tensor-sequence parallelism, the MLP forward pass involves:
1. **AllGather** — gather activations across tensor-parallel devices before the weight matrix multiply
2. **GEMM** — the actual matrix multiply
3. **ReduceScatter** — scatter-reduce the results back

Normally these happen sequentially:
```text
AllGather → GEMM → ReduceScatter
```

During AllGather and ReduceScatter, the accelerators wait for network communication to finish before computing. At scale, these collectives can take a significant fraction of step time.

TransformerEngine's collective-GEMM-overlap algorithm restructures this so that **communication for the next chunk overlaps with the GEMM compute of the current chunk**:

```text
AllGather(chunk 1) → GEMM(chunk 1) → ReduceScatter(chunk 1)
                  AllGather(chunk 2) overlaps with GEMM(chunk 1)
```

`te_comm_gemm_overlap` enables this overlap — and specifies which projections it covers.

---

## 2. What it controls

| Value | What overlaps |
|---|---|
| `"disabled"` | No overlap — sequential communication and compute (default) |
| `"mlp"` | Overlap only the MLP up/down projection collectives |
| `"full"` | Overlap MLP projections AND attention QKV/output projections |

Default in base.yml:
```yaml
te_comm_gemm_overlap: "disabled"
```

---

## 3. Why tensor-sequence parallelism is required

Tensor-sequence parallelism (TP+SP) is a specific parallelism strategy where:
- The model weights are sharded along a tensor-parallel dimension
- The sequence dimension is also sharded (for attention)
- This requires AllGather before and ReduceScatter after each matrix multiply

Without TP+SP, there's no collective communication around GEMMs, so there's nothing to overlap. Enabling `te_comm_gemm_overlap` without TP+SP has no meaningful effect.

---

## 4. TransformerEngine dependency

`te_comm_gemm_overlap` relies on **NVIDIA TransformerEngine** — a CUDA library for optimized transformer operations. This means:
- Only works on **NVIDIA GPUs**
- Requires TE to be installed in the environment
- The `"cudnn_flash_te"` attention backend is related but separate

---

## 5. `"mlp"` vs `"full"`

```text
"mlp":
  Overlaps up/down projection collectives in MLP layers
  Lower risk — MLP projections are simpler to pipeline
  Smaller benefit: covers MLP compute/comms only

"full":
  Also overlaps QKV and output projection collectives in attention
  Higher benefit: covers more of the step time
  Higher risk: more complex pipelining, possible correctness edge cases
```

Start with `"mlp"` when first enabling this; validate throughput/accuracy before moving to `"full"`.

---

## 6. When to use it

**Use:** Large-scale GPU training with TP+SP (e.g., 8B+ models on H100s with tensor parallelism) where collective communication is a measurable fraction of step time.

**Don't use:** TPU training (no TE), single-device training, or any configuration without tensor-sequence parallelism — there's no benefit and it adds complexity.

---

### One-line intuition

> **`te_comm_gemm_overlap` uses TransformerEngine to pipeline GEMM computation with AllGather/ReduceScatter communication in tensor-parallel MLP (and optionally attention) layers — a NVIDIA-GPU-only throughput optimization that only matters with tensor-sequence parallelism active.**
