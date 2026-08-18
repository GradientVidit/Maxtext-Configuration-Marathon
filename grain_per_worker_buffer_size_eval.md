## 1. Why does `grain_per_worker_buffer_size_eval` exist?

During validation and evaluation loops, Grain worker processes prepare batch tensors and place them into shared IPC memory queues for the evaluation coordinator.

```text
Eval Worker Process ──> [ Eval IPC Batch Queue (Capacity: buffer_size_eval) ] ──> Eval Step Forward Pass
```

Because evaluation typically runs over smaller validation benchmarks and executes less frequently than training steps, maintaining large pre-allocated batch queues in host RAM wastes memory. `grain_per_worker_buffer_size_eval` sets the batch queue depth per evaluation worker independently from training.

---

## 2. Mechanics & Default

MaxText defaults this to `1` per worker process:

```text
Worker 0 ──> [ Ready Eval Batch (Depth 1) ] ──> Evaluator
```

This prevents holding multiple unconsumed evaluation batches in host RAM simultaneously.

---

## 3. Options & Default

| Parameter | Type | Default | Options |
| :--- | :--- | :--- | :--- |
| `grain_per_worker_buffer_size_eval` | `int` | `1` | Positive integer (typically `1` or `2`) |

---

## 4. Interactions with Related Parameters

- **`grain_worker_count_eval`**: Total buffered evaluation batches = $\text{grain\_worker\_count\_eval} \times \text{grain\_per_worker\_buffer\_size\_eval}$.
- **`grain_per_worker_buffer_size`**: Training counterpart.

---

## 5. Practical Scenarios & Failure Modes

| Scenario | Symptom / Failure | Fix |
| :--- | :--- | :--- |
| **Standard evaluation passes** | Default of 1 minimizes memory footprint while keeping validation steady | Keep default `1`. |
| **Eval latency jitter on slow network mounts** | Coordinator waits for next batch | Increase to `2` to buffer one extra batch ahead. |

---

### One-line intuition

> `grain_per_worker_buffer_size_eval` sets the output queue depth for evaluation data workers, keeping evaluation memory overhead low.
