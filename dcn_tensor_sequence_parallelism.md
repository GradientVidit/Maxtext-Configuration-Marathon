## 1. Why does it exist?

Tensor-Sequence Parallelism (e.g. Megatron-LM Sequence Parallelism / DeepSpeed SP) splits activations along the sequence length dimension during dropout and layer normalization, and along hidden features during matrix multiplications.

It replaces standard All-Reduce operations with **All-Gather** and **Reduce-Scatter** collectives on every forward and backward layer boundary.

Like standard Tensor Parallelism, running Sequence Parallelism across separate TPU slices over the Data Center Network (DCN) incurs prohibitive latency. MaxText flags `dcn_tensor_sequence_parallelism` as **never recommended** for $> 1$.

```text
DCN Tensor-Sequence Parallelism:
  Every layer requires cross-slice All-Gather and Reduce-Scatter over DCN.
  Result: Network saturation and drastic MFU collapse.
```

---

## 2. Options & Configuration

| Value | Meaning | Status |
|---|---|---|
| `1` (default) | Confines tensor-sequence collectives strictly within local slice ICI interconnects. | **Standard / Recommended** |
| $> 1$ | Forces tensor-sequence collectives across slices over DCN. | **Never Recommended** |

Default in `base.yml`:
```yaml
dcn_tensor_sequence_parallelism: 1 # never recommended
```

---

### One-line intuition

> **`dcn_tensor_sequence_parallelism` configures sequence-parallel tensor collective communication across datacenter networks — should remain `1` to avoid cross-slice all-gather bottlenecks.**
