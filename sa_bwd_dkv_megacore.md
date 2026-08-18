## 1. Why does it exist?

Certain TPU generations (such as TPU v4 and TPU v5p) feature **Megacore** architectures, where two physical TensorCore processing engines share a single coherent Vector Memory (VMEM) / HBM subsystem.

When computing Key/Value gradients ($dK, dV$) across multiple KV attention heads (such as in Grouped-Query Attention / GQA or Multi-Query Attention / MQA), scheduling work across the two sub-cores requires careful grid partitioning.

`sa_bwd_dkv_megacore` enables megacore-parallel KV-head groups in the static $dK/dV$ Pallas kernel grid, assigning separate KV head groups to run concurrently on paired Megacore engines.

```text
Without Megacore Scheduling:
  Single core processes KV heads sequentially while paired core sits idle.

With sa_bwd_dkv_megacore: true:
  Megacore 0: Computes dKV for KV-Head Group A
  Megacore 1: Computes dKV for KV-Head Group B (Parallel execution on paired core!)
```

---

## 2. Options & Configuration

| Value | Meaning |
|---|---|
| `false` (default) | Standard grid scheduling. |
| `true` | Enables megacore-parallel KV head scheduling in the static $dKV$ grid. |

Default in `base.yml`:
```yaml
sa_bwd_dkv_megacore: false  # megacore-parallel kv-head groups in the static dkv grid
```

---

### One-line intuition

> **`sa_bwd_dkv_megacore` partitions Key/Value gradient computation across paired TPU Megacores, accelerating backward attention passes on Megacore-enabled hardware (like TPU v4 and v5p).**
