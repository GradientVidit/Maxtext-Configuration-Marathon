## 1. Why does `grain_num_threads_eval` exist?

Evaluation datasets stored on Google Cloud Storage (GCS) or remote storage require concurrent I/O threads to stream, parse, and interleave records without stalling the evaluation coordinator.

```text
Eval Worker Process
        │
        ├── Thread 1 (Read Eval Chunk from GCS)
        ├── Thread 2 (Read Eval Chunk from GCS)
        └── Thread N (Read Eval Chunk from GCS)  [grain_num_threads_eval = 16]
```

`grain_num_threads_eval` sets the thread pool size per evaluation worker process used to read evaluation files concurrently.

---

## 2. Mechanics & Storage Interleaving

- **ArrayRecord (`'arrayrecord'`)**: Passed directly into Grain's `ReadOptions.num_threads` for parallel chunk reads.
- **Parquet (`'parquet'`)**: Dictates how many Parquet evaluation files are read and interleaved in parallel.

---

## 3. Options & Default

| Parameter | Type | Default | Options |
| :--- | :--- | :--- | :--- |
| `grain_num_threads_eval` | `int` | `16` | Positive integer (e.g. `8`, `16`, `32`) |

---

## 4. Interactions with Related Parameters

- **`grain_num_threads`**: Training thread count.
- **`grain_prefetch_buffer_size_eval`**: Threads push fetched raw records into the eval prefetch buffer.

---

## 5. Practical Scenarios & Failure Modes

| Scenario | Symptom / Failure | Fix |
| :--- | :--- | :--- |
| **High GCS round-trip latency during eval** | Single-threaded read causes validation step pauses | Keep default `16` to pipeline cloud reads. |

---

### One-line intuition

> `grain_num_threads_eval` defines the thread concurrency per worker process used to stream evaluation dataset files from storage.
