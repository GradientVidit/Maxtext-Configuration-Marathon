## 1. Why does `skip_first_n_steps_for_profiler` exist?

The first few steps of a MaxText run are distorted by non-steady-state transients:
1. XLA / PjRT JIT compilation and graph lowering.
2. Device memory allocation and buffer initialization.
3. Cold data loader cache filling.

If a profiler triggers immediately at step 0, the profile trace will capture seconds of host compilation overhead rather than true steady-state step execution:

```text
Training Timeline:
[Step 0: JIT Compilation (200s)] ──>[Step 1: Steady State (45ms)] ──>[Step 2: Steady State (45ms)]
 └──────────────┬───────────────┘    └─────────────────────┬─────────────────────┘
                │                                          │
    skip_first_n_steps_for_profiler: 1           Profile Window Begins Here
           (Skipped)                              (Captures true steady state)
```

`skip_first_n_steps_for_profiler` defines the number of initial steps to bypass before triggering trace capture.

---

## 2. Fundamentals & Mechanics

- Ensures hardware performance metrics reflect actual compiled execution.
- Default `1` bypasses the initial compilation and graph build step.

---

## 3. Options & Defaults

| Option | Value | Meaning |
|---|---|---|
| Default | `1` | Bypasses step 0 compilation overhead. |
| Warmup Skip | `5` to `10` | Bypasses early data loader buffering and optimizer warmup steps. |

---

## 4. Interactions & Dependencies

```text
skip_first_n_steps_for_profiler ──> profiler_steps (Traced window)
```

---

## 5. Practical Scenarios & Failure Modes

- Setting `0` produces enormous trace files dominated by XLA compiler traces rather than MXU execution schedules.

---

### One-line intuition

> **`skip_first_n_steps_for_profiler` skips early initialization and compilation steps so the profiler records only steady-state execution.**
