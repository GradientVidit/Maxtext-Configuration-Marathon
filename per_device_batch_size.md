## 1. Why does `per_device_batch_size` exist?

In distributed JAX training across TPU and GPU clusters, batch sizing is specified **per accelerator device** (or per physical core) rather than globally. 

```text
                               Global Batch Size
                                      │
              ┌───────────────────────┼───────────────────────┐
              ▼                       ▼                       ▼
           Device 0                Device 1                Device N-1
  per_device_batch_size   per_device_batch_size   per_device_batch_size
```

Global batch size is derived mathematically as:
$$ ext{Global Batch Size} =  ext{per\_device\_batch\_size}  imes  ext{data\_parallel\_devices}$$

`per_device_batch_size` is declared as a **float** (e.g. `12.0`, `4.5`, `0.5`) rather than an integer to support advanced data-loading optimizations such as `expansion_factor_real_data` and fractional batch step calculations in batch-size ramp-up.

---

## 2. Mechanics & Memory Footprint

`per_device_batch_size` directly dictates peak High Bandwidth Memory (HBM) consumed by forward activations:

```text
Activation Memory ~ O(per_device_batch_size * max_target_length * hidden_dim * num_layers)
```

```text
per_device_batch_size: 12.0 ──> 12 sequences per TPU core
                                       │
                      Exceeds HBM? ────┼──── Fits HBM?
                                       │            │
                                       ▼            ▼
                                      OOM       Max MXU Utilization (~60% MFU)
```

Tuning `per_device_batch_size` balances compute throughput (keeping Matrix Multiply Units saturated) against HBM capacity.

---

## 3. Options & Default

| Parameter | Type | Default | Valid Range |
| :--- | :--- | :--- | :--- |
| `per_device_batch_size` | `float` | `12.0` | Any positive float (typically `1.0`, `2.0`, `4.0`, `8.0`, `12.0`, `16.0`) |

---

## 4. Interactions with Related Parameters

- **`expansion_factor_real_data`**: When `> 1`, each reading host loads `per_device_batch_size * expansion_factor_real_data`.
- **`enable_rampup_batch_size`**: Dynamic scaling starts at `per_device_batch_size_start` and increments toward `per_device_batch_size`.
- **`eval_per_device_batch_size`**: If eval batch size is `0.0`, it defaults to `per_device_batch_size`.
- **`ici_data_parallelism` / `ici_fsdp_parallelism`**: Scales the global batch size proportionally.

---

## 5. Practical Scenarios & Failure Modes

| Scenario | Symptom / Failure | Fix |
| :--- | :--- | :--- |
| **Out-of-Memory (OOM) on startup during forward pass** | TPU HBM allocation failure | Reduce `per_device_batch_size` (e.g. from 12.0 to 4.0 or 2.0). |
| **Low Model Flops Utilization (MFU < 30%)** | MXUs starved for compute; memory bandwidth bound | Increase `per_device_batch_size` to increase arithmetic intensity. |
| **Changing cluster size from 128 to 256 TPUs** | Global batch size doubles unexpectedly, affecting learning rate dynamics | Adjust `per_device_batch_size` or data parallel degree to keep global batch constant. |

---

### One-line intuition

> `per_device_batch_size` sets the training batch volume per accelerator core, acting as the primary lever balancing HBM memory consumption against TPU/GPU compute utilization.
