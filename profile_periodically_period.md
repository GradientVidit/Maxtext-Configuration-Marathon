## 1. Why does `profile_periodically_period` exist?

In long-running pretraining runs (spanning weeks or months), performance is not static: memory fragmentation, host garbage collection, dynamic sequence packing distributions, or thermal throttling can degrade MFU over time.

A one-time profile at step 1 cannot detect mid-run performance degradations:

```text
Periodic Profiling Cadence:
Step 0..1000 ──>[ Profile Traced ]──> Step 1001..2000 ──>[ Profile Traced ]──> ...
                 └────────┬──────┘                        └────────┬──────┘
                          ▼                                        ▼
             profile_periodically_period = 1000
```

`profile_periodically_period` configures recurring profiling sessions at regular step intervals throughout training.

---

## 2. Fundamentals & Mechanics

- **`-1` (Default):** Periodic profiling is disabled; MaxText profiles once at startup per `skip_first_n_steps_for_profiler`.
- **`N > 0`:** Automatically triggers a `profiler_steps` trace every $N$ steps.

---

## 3. Options & Defaults

| Option | Value | Meaning |
|---|---|---|
| Default | `-1` | Disabled (profiles only once at start). |
| Periodic Interval | `1000`, `5000` | Re-runs profiler every $N$ steps throughout training. |

---

## 4. Interactions & Dependencies

- Reuses `profiler_steps` for each periodic trace capture.

---

## 5. Practical Scenarios & Failure Modes

- **Monitoring Production Runs:** Setting `profile_periodically_period: 10_000` captures regular snapshots to verify consistent hardware throughput across multi-week training jobs.

---

### One-line intuition

> **`profile_periodically_period` sets a recurring step interval for capturing hardware performance traces repeatedly throughout long training runs.**
