## 1. Why does `profiler_steps` exist?

Hardware profiling traces record microsecond-by-microsecond events across every chip, generating gigabytes of data per minute.

Tracing too many steps exhausts host memory and disk space, while tracing fewer than 2 steps fails to capture the inter-step pipeline overlap and double-buffering dynamics:

```text
Trace Duration Window:
Step N (Skipped) ──>[Step N+1][Step N+2][Step N+3][Step N+4][Step N+5] ──> Trace Ends
                     └──────────────────────┬──────────────────────┘
                                            ▼
                                    profiler_steps = 5
```

`profiler_steps` specifies how many consecutive steps are recorded in a profiling session.

---

## 2. Fundamentals & Mechanics

- Default `5` provides an ideal sample size: captures step boundaries, pipeline overlaps, and steady-state MXU utilization without blowing up trace file size.

---

## 3. Options & Defaults

| Option | Value | Meaning |
|---|---|---|
| Default | `5` | Traces 5 consecutive training steps. |
| Minimal | `2` | Minimal window to capture step-to-step overlap. |
| Deep Analysis | `10` | Extended trace for observing multi-microbatch pipeline repeats. |

---

## 4. Interactions & Dependencies

- Bounded by `skip_first_n_steps_for_profiler` (start offset) and `profiler`.

---

## 5. Practical Scenarios & Failure Modes

- Setting `profiler_steps: 500` will quickly fill host `/tmp` and cause out-of-disk crashes during profile serialization.

---

### One-line intuition

> **`profiler_steps` sets the number of consecutive training steps captured during a profiling session.**
