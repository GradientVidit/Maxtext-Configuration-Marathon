## 1. Why does `profile_cleanly` exist?

Because JAX executes asynchronously, training steps pipeline and overlap on the device: Step $N+1$'s data transfer and forward pass begin before Step $N$'s backward pass finishes.

In a profiler trace, this asynchronous overlap smears step boundaries across the timeline, making it difficult to measure the exact wall-clock duration and FLOPs of a single isolated step:

```text
Pipelined Trace (profile_cleanly = false):
Device: [Step 1 Fwd][Step 1 Bwd / Step 2 Fwd overlapping][Step 2 Bwd / Step 3 Fwd]
        (Boundaries are blurry and overlapped)

Clean Trace (profile_cleanly = true):
Device: [Step 1 Fwd & Bwd] ──[ Barrier ]──> [Step 2 Fwd & Bwd] ──[ Barrier ]──>
        (Strict step boundaries for exact per-step measurement)
```

`profile_cleanly` inserts explicit synchronization barriers (`block_until_ready()`) at step boundaries during profiled steps.

---

## 2. Fundamentals & Mechanics

- **`true` (Default):** Calls `block_until_ready()` on the training state PyTree at the start and end of each profiled step.
- Isolates step timelines cleanly in TensorBoard / XPlane traces for precise MFU accounting.

---

## 3. Options & Defaults

| Option | Value | Meaning |
|---|---|---|
| Default | `true` | Forces clean step boundary alignment during profiling. |
| Asynchronous | `false` | Allows uninhibited asynchronous pipelining during profiling. |

---

## 4. Interactions & Dependencies

- Interacts with `max_inflight_computations` during the profiled window.

---

## 5. Practical Scenarios & Failure Modes

- When measuring raw device pipelining overhead, temporarily set `profile_cleanly: false` to observe natural inter-step concurrency.

---

### One-line intuition

> **`profile_cleanly` inserts synchronization barriers at step boundaries during profiling to ensure clean, unblurred trace timelines.**
