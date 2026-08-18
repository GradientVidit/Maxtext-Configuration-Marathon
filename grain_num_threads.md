## 1. Why does `grain_num_threads` exist?

Within each Grain worker process, reading raw data records from cloud storage or local disk requires concurrent network I/O operations.

```text
Worker Process
      │
      ├── Thread 1 (GCS Read)
      ├── Thread 2 (GCS Read)   ===> High concurrency over cloud storage
      └── Thread N (GCS Read)
```

For **ArrayRecord**, `grain_num_threads` sets `ReadOptions.num_threads`. For **Parquet**, it controls how many files are read and interleaved in parallel.

---

## 2. Mechanics & Default

MaxText defaults `grain_num_threads` to `16`, matching Grain's internal package recommendation for cloud storage throughput.

---

## 3. Options & Default

| Parameter | Type | Default | Options |
| :--- | :--- | :--- | :--- |
| `grain_num_threads` | `int` | `16` | Positive integer (e.g. `8`, `16`, `32`) |

---

## 4. Interactions with Related Parameters

- **`grain_prefetch_buffer_size`**: Thread pool prefetches records into this buffer.
- **`grain_num_threads_eval`**: Eval counterpart.

---

## 5. Practical Scenarios & Failure Modes

| Scenario | Symptom / Failure | Fix |
| :--- | :--- | :--- |
| **Mixing many data sources creates thread explosion** | 20 sources x 16 threads = 320 threads exhausting CPU context switching | Lower `grain_num_threads` to 4 or 8. |

---

### One-line intuition

> `grain_num_threads` defines the thread concurrency per worker used to read and stream data files from storage.
