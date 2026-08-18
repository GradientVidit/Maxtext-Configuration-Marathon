## 1. Why does `grain_worker_count` exist?

Data loading tasks (reading storage, decompressing ArrayRecord/Parquet files, string parsing, subword tokenization, and bin packing) are CPU-bound. If a single host process handles data preparation alongside JAX accelerator dispatch, the TPU/GPU will suffer data starvation.

```text
Single Worker Process:
[ Read Storage -> Decompress -> Tokenize -> Pack ] ──> TPU waits idle (Data Starvation)

Multi-Worker Grain:
Worker 0 ──[ Tokenize & Pack ]──┐
Worker 1 ──[ Tokenize & Pack ]──┼──> Shared Memory IPC Queue ──> Accelerator Saturated
Worker N ──[ Tokenize & Pack ]──┘
```

`grain_worker_count` sets the number of dedicated multiprocessing worker processes spawned per host for Grain data loading.

---

## 2. Mechanics & Auto-Tuning via `pick_performance_config`

```text
grain_worker_count: -1 ──> Auto-tuning via grain.experimental.pick_performance_config()
grain_worker_count: > 0 ──> Spawns explicit static number of MultiprocessPrefetchIterDataset workers
```

When set to `-1`, Grain executes `grain.experimental.pick_performance_config(ds, ram_budget_mb=grain_ram_budget_mb)`:
1. Samples dataset elements to estimate average memory footprint per batch.
2. Analyzes available CPU cores and the user-specified `grain_ram_budget_mb`.
3. Automatically derives the optimal worker count and prefetch buffer sizes.

> [!NOTE]
> For strict cross-cluster determinism, manually specifying a fixed worker count (e.g. `4` or `8`) ensures identical multi-process interleaving regardless of differing host memory sizes.

---

## 3. Options & Default

| Parameter | Type | Default | Options |
| :--- | :--- | :--- | :--- |
| `grain_worker_count` | `int` | `1` | `1`, positive integer (e.g. `4`, `8`, `16`), or `-1` (auto-tune) |

---

## 4. Interactions with Related Parameters

- **`grain_ram_budget_mb`**: Memory ceiling used by `pick_performance_config` when `grain_worker_count: -1`.
- **`grain_per_worker_buffer_size`**: Buffer capacity allocated per worker in shared memory (`/dev/shm`).
- **`grain_worker_count_eval`**: Independent worker count for evaluation data loading.

---

## 5. Practical Scenarios & Failure Modes

| Scenario | Symptom / Failure | Fix |
| :--- | :--- | :--- |
| **TPU step time fluctuates due to CPU tokenization bottleneck** | Single worker cannot tokenize text fast enough to saturate TPUs | Set `grain_worker_count: 8` or `-1`. |
| **Host OOM / `/dev/shm` exhaustion** | Excessive workers allocate too many shared memory queues | Reduce `grain_worker_count` (e.g. from 32 to 8) and keep `grain_per_worker_buffer_size: 1`. |

---

### One-line intuition

> `grain_worker_count` controls how many parallel CPU worker processes are spawned per host to tokenize and prepare data, supporting automated tuning via `grain.experimental.pick_performance_config` when set to `-1`.
