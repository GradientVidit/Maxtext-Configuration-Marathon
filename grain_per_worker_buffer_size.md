## 1. Why does `grain_per_worker_buffer_size` exist?

Multiprocessing data loaders communicate with the main process through inter-process communication (IPC) queues and shared memory buffers.

```text
Worker Process ──> [ Shared Memory Queue (Capacity: buffer_size) ] ──> Main Trainer
```

`grain_per_worker_buffer_size` controls the queue capacity per worker process, buffering ready batches so transient network spikes don't stall training.

---

## 2. Mechanics

Higher buffer sizes smooth out I/O jitter but consume more host RAM. MaxText defaults this to `1` to prevent excessive memory retention when batch tensors are large.

---

## 3. Options & Default

| Parameter | Type | Default | Options |
| :--- | :--- | :--- | :--- |
| `grain_per_worker_buffer_size` | `int` | `1` | Positive integer (typically `1` to `4`) |

---

## 4. Interactions with Related Parameters

- **`grain_worker_count`**: Total buffered batches in host memory = $ ext{worker\_count}  imes  ext{per\_worker\_buffer\_size}$.

---

## 5. Practical Scenarios & Failure Modes

| Scenario | Symptom / Failure | Fix |
| :--- | :--- | :--- |
| **Host shared memory exhaustion (`SIGBUS` / `/dev/shm` full)** | Too many large batches buffered simultaneously | Keep `grain_per_worker_buffer_size: 1` or expand `/dev/shm`. |

---

### One-line intuition

> `grain_per_worker_buffer_size` sets the depth of the pre-allocated batch queue for each data-loading worker process.
