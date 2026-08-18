## 1. Why does `learning_rate_schedule_steps` exist?

In MaxText, the total training steps (`steps`) and the duration over which the learning rate schedule is computed (`learning_rate_schedule_steps`) can be configured independently.

This separation allows developers to:
1. **End training early:** Plan a 100k-step decay schedule but stop at 50k steps.
2. **Post-decay evaluation:** Finish decaying the LR at step 80k, and run the remaining 20k steps at constant zero learning rate to measure clean, non-decaying loss metrics.

```text
Schedule Decoupling Dynamics:

Case 1: learning_rate_schedule_steps == steps (100k)
LR ^     /\
   |    /  \
   |___/____\________> Step 100k (Decay finishes exactly as run ends)

Case 2: learning_rate_schedule_steps (80k) < steps (100k)
LR ^     /\
   |    /  \
   |___/____\────────> Step 80k (Reaches floor, then runs at LR=0 until 100k)
```

`learning_rate_schedule_steps` defines the base step count across which warmup, stable, and decay phases are calculated.

---

## 2. Fundamentals & Mechanics

- **Default `-1`:** Dynamically matches `learning_rate_schedule_steps = steps`.
- **Mismatch Behaviors:**
  - **`learning_rate_schedule_steps < steps`:** The schedule completes its decay by `learning_rate_schedule_steps`. For all subsequent steps up to `steps`, the learning rate is clamped to `0.0`.
  - **`learning_rate_schedule_steps > steps`:** Training terminates at `steps` before the learning rate has finished its full decay trajectory.

---

## 3. Options & Defaults

| Option | Value | Meaning |
|---|---|---|
| Default | `-1` | Auto-inherits the value of `steps` (schedule exactly matches run length). |
| Explicit Integer | `N > 0` | Schedules warmup and decay across $N$ steps regardless of `steps`. |

---

## 4. Interactions & Dependencies

```text
learning_rate_schedule_steps
             │
             ├─ Multiplied by warmup_steps_fraction
             ├─ Multiplied by wsd_decay_steps_fraction
             └─ Compared against steps (Termination)
```

- **`steps: -1`:** If `steps: -1` and `learning_rate_schedule_steps: 100_000`, `steps` adopts `100_000`.

---

## 5. Practical Scenarios & Failure Modes

- **Zero-LR Cooldown Phase:** Setting `learning_rate_schedule_steps: 90_000` with `steps: 100_000` gives 10,000 steps of zero-LR evaluation to verify model stability.
- **Accidental Truncation:** Setting `learning_rate_schedule_steps: 200_000` with `steps: 100_000` means training stops when the learning rate is still halfway through its decay, leaving significant accuracy on the table.

---

### One-line intuition

> **`learning_rate_schedule_steps` defines the step horizon used to calculate schedule phases, defaulting to match total training `steps` when set to `-1`.**
