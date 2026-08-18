## 1. Why does `grain_worker_count_eval` exist?

Evaluation runs periodically and processes a smaller volume of data than training. Running the same number of data workers for eval as training may waste system resources or cause unnecessary thread thrashing.

```text
Training: 8 Workers (Max throughput streaming)
Eval:     1 Worker  (Sufficient for small validation benchmarks)
```

`grain_worker_count_eval` configures the worker process count specifically for the evaluation data loader.

---

## 2. Mechanics & Default

Defaults to `1`, which is typically sufficient for validation sets while keeping host RAM free.

---

## 3. Options & Default

| Parameter | Type | Default | Options |
| :--- | :--- | :--- | :--- |
| `grain_worker_count_eval` | `int` | `1` | Positive integer or `-1` |

---

## 4. Interactions with Related Parameters

- **`grain_worker_count`**: Training worker count.
- **`grain_per_worker_buffer_size_eval`**: Eval worker queue size.

---

## 5. Practical Scenarios & Failure Modes

| Scenario | Symptom / Failure | Fix |
| :--- | :--- | :--- |
| **Massive validation set causes eval to take hours** | Single eval worker bottlenecks data loading | Increase `grain_worker_count_eval` to 4 or 8. |

---

### One-line intuition

> `grain_worker_count_eval` sets the number of worker processes dedicated to evaluation data loading in Grain.
