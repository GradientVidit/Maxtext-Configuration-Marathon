## 1. Why does `max_inflight_computations` exist?

JAX executes asynchronously: Python enqueues computation graphs on the accelerator queue and immediately returns control to the host loop before the device finishes execution.

Without a bound on this queue, the host Python process can race ahead thousands of steps into the future, allocating device buffers, input tensors, and callback handlers until host RAM or device command queues exhaust:

```text
Unbounded Queue (max_inflight_computations = ∞):
Host (Python): [Step 1][Step 2][Step 3]...[Step 500] ──> Out of Host RAM / Queue Overflow
Device (TPU):  [Step 1 In-Progress..................]

Bounded Queue (max_inflight_computations = 2):
Host (Python): [Step 1][Step 2] ──> BLOCKS until Step 1 completes
Device (TPU):  [Step 1][Step 2] ──> Continuous pipelining with zero device bubble
```

`max_inflight_computations` caps how many uncompleted step graphs can be queued simultaneously on the accelerator backend.

---

## 2. Fundamentals & Mechanics

MaxText uses a circular deque or semaphore of futures (train state references) to throttle Python dispatch.

- When the number of dispatched-but-unfinished steps reaches `max_inflight_computations`, the host thread calls `.block_until_ready()` on the oldest in-flight step.
- Setting `max_inflight_computations: 2` (double-buffering) ensures that while the accelerator executes Step $N$, the host prepares and dispatches Step $N+1$. When Step $N$ finishes, Step $N+1$ begins instantly with zero idle latency.

---

## 3. Options & Defaults

| Option | Value | Meaning |
|---|---|---|
| Default | `2` | Double-buffered execution. Optimal for almost all TPU/GPU training. |
| Synchronous | `1` | Strictly synchronous dispatch. Python waits for each step to finish. Useful for stepping through memory profilers. |
| Deep Pipelining | `3` or `4` | Accommodates high host-dispatch jitter on complex input data pipelines at the cost of slightly higher host memory. |

---

## 4. Interactions & Dependencies

```text
                  max_inflight_computations
                              │
            ┌─────────────────┴─────────────────┐
            ▼                                   ▼
    profile_cleanly                   colocated_python_data_input
(Inserts explicit sync barriers)     (Host data generation cadence)
```

- **`profile_cleanly: true`:** Profile boundaries insert an explicit `block_until_ready()` on the train state, temporarily overriding pipelining during traced steps to isolate per-step execution traces.

---

## 5. Practical Scenarios & Failure Modes

- **Host Memory Leaks:** If host RAM climbs steadily over thousands of steps, check if `max_inflight_computations` was set excessively high.
- **Step Jitter / Bubbles:** If hardware trace timelines show small gaps between training steps, increasing from `1` to `2` eliminates host-dispatch latency from the critical path.

---

### One-line intuition

> **`max_inflight_computations` bounds the asynchronous JAX device execution queue, enabling double-buffered dispatch without blowing up host memory.**
