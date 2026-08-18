## 1. Why does `grain_prefetch_buffer_size_eval` exist?

Evaluation routines iterate through validation sets in bursts at scheduled checkpoint intervals. Network latency spikes when opening cloud storage files can delay validation execution.

```text
GCS Remote Storage ──> [ Eval Prefetch Buffer (500 records) ] ──> Evaluation Forward Loop
```

`grain_prefetch_buffer_size_eval` establishes a sliding reservoir of prefetched raw records per worker and per dataset during evaluation streaming.

---

## 2. Mechanics & Default

MaxText defaults `grain_prefetch_buffer_size_eval` to `500`. 

```text
Eval Iterator Request ──> Consumes from prefetch buffer (0ms wait) ──> Background thread refills buffer
```

For most small to medium evaluation benchmarks, 500 records allows caching the entire validation sample upfront.

---

## 3. Options & Default

| Parameter | Type | Default | Options |
| :--- | :--- | :--- | :--- |
| `grain_prefetch_buffer_size_eval` | `int` | `500` | Positive integer (e.g. `100` to `1000`) |

---

## 4. Interactions with Related Parameters

- **`grain_prefetch_buffer_size`**: Training counterpart.
- **`grain_num_threads_eval`**: Threads that populate the prefetch buffer.

---

## 5. Practical Scenarios & Failure Modes

| Scenario | Symptom / Failure | Fix |
| :--- | :--- | :--- |
| **Evaluating large multimodal vision-text datasets** | 500 large multimodal records cause host memory bloat | Lower `grain_prefetch_buffer_size_eval` to 50 or 100. |

---

### One-line intuition

> `grain_prefetch_buffer_size_eval` sets the prefetch buffer size for raw evaluation records to eliminate network I/O stalls during validation passes.
