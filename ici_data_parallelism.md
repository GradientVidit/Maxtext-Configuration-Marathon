## 1. Why does it exist?

Within a single TPU pod slice (where devices are connected via ultra-fast Inter-Chip Interconnect / ICI), model execution can partition devices among parameter sharding (FSDP), tensor parallelism, expert parallelism, and pure data parallelism.

Pure Data Parallelism replicates the entire model across devices and splits only the batch dimension. In contrast, FSDP shards model weights, gradients, and optimizer states, drastically reducing per-device HBM usage.

```text
Pure ICI Data Parallelism:
  Device 0: Full Model Weights (100% HBM) ──→ Batch Shard 0
  Device 1: Full Model Weights (100% HBM) ──→ Batch Shard 1

ICI FSDP (ZeRO-3):
  Device 0: 50% Weights (50% HBM) ──→ Gathers weights on the fly
  Device 1: 50% Weights (50% HBM) ──→ Gathers weights on the fly
```

`ici_data_parallelism` specifies the degree of pure, unsharded data parallelism within a single slice.

---

## 2. Fundamentals & Options

| Value | Behavior |
|---|---|
| `1` (default) | Pure data parallelism is disabled within the slice; all intra-slice data partitioning is handled via FSDP (`ici_fsdp_parallelism`), maximizing memory efficiency. |
| Integer $> 1$ | Allocates `N` fully replicated model copies within the slice. |

Default in `base.yml`:
```yaml
ici_data_parallelism: 1
```

---

## 3. ICI Data Parallelism vs. ICI FSDP

- **Memory Footprint**: `ici_data_parallelism` requires each device to hold all weights and optimizer states, whereas `ici_fsdp_parallelism` divides parameters evenly across devices.
- **When to use `ici_data_parallelism > 1`**: Only for small models that easily fit in device memory and where you want to eliminate FSDP All-Gather communication overhead entirely.

---

## 4. Interactions with Related Parameters

- **`ici_fsdp_parallelism`**: Recommended auto-sharding axis for ICI (`-1`).
- **`shard_optimizer_over_data`**: When `true`, shards optimizer states across `ici_data_parallelism` replicas (ZeRO-1).

---

### One-line intuition

> **`ici_data_parallelism` configures pure, unsharded model replication across devices within a TPU slice — defaults to `1` in favor of memory-efficient FSDP.**
