## 1. Why does it exist?

In distributed training using pure Data Parallelism (`data` axis), model parameters and gradients are replicated across devices while batches are partitioned. However, storing full optimizer states (such as Adam's first moment $m$ and second moment $v$, which take $2 \times 4 = 8$ bytes per parameter in FP32) on every single replica wastes valuable High Bandwidth Memory (HBM).

**ZeRO-1 (Zero Redundancy Optimizer Stage 1)** partitions only the optimizer states across data-parallel replicas, while keeping model weights replicated. Each device updates only its $1/N$-th partition of the optimizer state and broadcasts the updated parameter slice to sibling devices.

```text
Standard Data Parallelism:
  Device 0: Model Weights (Replicated) + Optimizer States (100% Full Copy)
  Device 1: Model Weights (Replicated) + Optimizer States (100% Full Copy)

ZeRO-1 Optimizer Sharding (shard_optimizer_over_data: true):
  Device 0: Model Weights (Replicated) + Optimizer States (50% Shard)
  Device 1: Model Weights (Replicated) + Optimizer States (50% Shard)
  ──→ Significant HBM savings with zero forward-pass All-Gather communication!
```

`shard_optimizer_over_data` enables ZeRO-1 style optimizer state sharding over the physical `data` mesh axis.

---

## 2. Fundamentals & Comparison with FSDP (ZeRO-3)

| Feature | Standard DP | ZeRO-1 (`shard_optimizer_over_data`) | Full FSDP (ZeRO-3) |
|---|---|---|---|
| Model Weights | Replicated | Replicated | Sharded (All-Gathered per layer) |
| Gradients | Replicated/Reduced | Reduced across data | Sharded (Reduce-Scattered) |
| Optimizer States | Replicated | **Sharded over `data` axis** | Sharded over `fsdp` axis |
| Forward Comm | None | **None** | All-Gather per layer |
| Backward Comm | All-Reduce | Reduce-Scatter + All-Gather | Reduce-Scatter |

---

## 3. Options & Configuration

| Value | Meaning |
|---|---|
| `false` (default) | Optimizer state follows standard sharding rules (e.g. sharded on `fsdp` or replicated on `data`). |
| `true` | Shards optimizer state across all devices in the `data` mesh axis. |

Default in `base.yml`:
```yaml
shard_optimizer_over_data: false
```

---

## 4. Practical Scenarios

- **When to Use**: Ideal when you want to avoid FSDP's forward-pass All-Gather latency overhead, but your model's Adam optimizer states do not fit into single-device HBM alongside model weights.
- **Pairs Well With**: Configurations using `ici_data_parallelism > 1` or `dcn_data_parallelism > 1`.

---

### One-line intuition

> **`shard_optimizer_over_data` implements ZeRO-1 optimizer partitioning over the data-parallel axis, freeing massive HBM by sharding Adam states without incurring forward-pass weight communication.**
