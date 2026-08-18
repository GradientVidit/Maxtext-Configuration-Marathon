## 1. Why does `grain_prefetch_buffer_size` exist?

Cloud storage requests to Google Cloud Storage (GCS) have variable network latency. If records are fetched one by one on demand, latency spikes directly translate into accelerator stalls.

```text
GCS ──> [ Prefetch Buffer: 500 records ] ──> Transformer Pipeline
```

`grain_prefetch_buffer_size` maintains a sliding buffer of raw prefetched records per worker and per dataset, ensuring continuous data supply.

---

## 2. Mechanics

In ArrayRecord `ReadOptions`, `prefetch_buffer_size` controls how many unparsed records are held in memory ahead of the transformation pipeline.

---

## 3. Options & Default

| Parameter | Type | Default | Options |
| :--- | :--- | :--- | :--- |
| `grain_prefetch_buffer_size` | `int` | `500` | Positive integer (e.g. `100` to `1000`) |

---

## 4. Interactions with Related Parameters

- **`grain_num_threads`**: Threads actively fill the prefetch buffer.
- **`grain_prefetch_buffer_size_eval`**: Eval counterpart.

---

## 5. Practical Scenarios & Failure Modes

| Scenario | Symptom / Failure | Fix |
| :--- | :--- | :--- |
| **Memory pressure with massive text documents** | 500 long documents consume several gigabytes of host RAM | Lower `grain_prefetch_buffer_size` to 50 or 100. |

---

### One-line intuition

> `grain_prefetch_buffer_size` sets the number of raw records prefetched in memory per worker to absorb cloud storage network latency.
