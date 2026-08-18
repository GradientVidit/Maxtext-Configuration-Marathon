## 1. Why does it exist?

Tensor Parallelism (Megatron-LM style TP) splits individual linear operations and attention heads across devices:
- Column-Parallel: Splitting QKV projections and MLP up-projections along output feature dimensions.
- Row-Parallel: Splitting attention output projections and MLP down-projections along input feature dimensions, followed by an All-Reduce.

Because Tensor Parallelism requires multiple All-Reduce operations on every single layer, it is strictly bandwidth- and latency-bound. The ultra-fast optical/copper links of TPU Inter-Chip Interconnects (ICI) provide the sub-microsecond latency necessary to execute TP efficiently.

```text
Intra-Layer Tensor Parallelism (ici_tensor_parallelism: 4):
  Linear Projection:
    Device 0: Heads 0..7   ──┐
    Device 1: Heads 8..15  ──┼──[ High-Speed ICI All-Reduce ]──→ Combined Activations
    Device 2: Heads 16..23 ──┤
    Device 3: Heads 24..31 ──┘
```

`ici_tensor_parallelism` specifies the degree of intra-layer Tensor Parallelism within a single TPU slice.

---

## 2. Options & Configuration

| Value | Meaning | Recommended Use Case |
|---|---|---|
| `1` (default) | Pure FSDP (No tensor parallelism); maximizes compute-communication overlap for standard training. | High-throughput pretraining and fine-tuning. |
| Power of 2 (e.g. `2`, `4`, `8`) | Splits attention heads and MLP hidden dimensions across `N` chips. | Low-latency inference serving or fitting large heads into small HBM. |

Default in `base.yml`:
```yaml
ici_tensor_parallelism: 1
```

---

## 3. Divisibility Constraints

When `ici_tensor_parallelism > 1`:
- `base_num_query_heads` must be divisible by `ici_tensor_parallelism`.
- `base_num_kv_heads` must be divisible by `ici_tensor_parallelism` (or replicated).
- `base_mlp_dim` must be divisible by `ici_tensor_parallelism`.

---

## 4. Practical Trade-offs

- **Training vs. Inference**: In large-batch training, FSDP is generally preferred over TP because FSDP collectives can be hidden behind compute. In interactive inference serving (e.g. JetStream/vLLM), setting `ici_tensor_parallelism: 4` or `8` dramatically cuts Time-to-First-Token (TTFT) and per-token latency.

---

### One-line intuition

> **`ici_tensor_parallelism` splits individual attention heads and MLP projections across devices over the high-speed chip interconnect, slashing per-token latency during inference and fitting huge layers.**
