## 1. Why does `steps` exist?

Training a large language model requires a definitive termination horizon for execution loops, checkpoint planning, and token budgeting.

Without a global step counter, training pipelines cannot coordinate asynchronous checkpoint milestones, time-based dataset streams, or learning rate schedules.

```text
       Start Step (0 or resumed)
                   │
                   ▼
  ┌───────────────────────────────────┐
  │         Training Loop             │ ◄─── Checkpoint periodically
  │  Step i -> Fwd -> Bwd -> Update   │ ◄─── Evaluate / Log metrics
  └───────────────────────────────────┘
                   │
           i == steps ?
          ┌────────┴────────┐
          ▼                 ▼
        [No]              [Yes]
     Continue        Save Final Checkpoint
    Next Step        Terminate Cleanly
```

`steps` defines the absolute target step count where MaxText halts execution.

---

## 2. Fundamentals & Mechanics

In MaxText, the main training loop iterates until `step >= config.steps`.

- **Auto-resumption:** If resuming from a checkpoint at step `50_000` with `steps: 150_001`, MaxText executes `100_001` additional steps.
- **Inheritance mode:** Setting `steps: -1` configures MaxText to dynamically set `steps = learning_rate_schedule_steps`.
- **Completion hook:** Reaching `steps` triggers final metric flushes and `save_checkpoint_on_completion` (if enabled).

```text
Case A (Explicit steps):
  steps: 100_000
  learning_rate_schedule_steps: 100_000
  -> Schedule matches full training run.

Case B (Inherit from LR schedule):
  steps: -1
  learning_rate_schedule_steps: 80_000
  -> steps dynamically resolves to 80_000.
```

---

## 3. Options & Defaults

| Option | Value | Meaning |
|---|---|---|
| Default | `150_001` | Pre-configured baseline step count. |
| Inheritance | `-1` | Copies value from `learning_rate_schedule_steps`. |
| Custom Integer | `N > 0` | Explicit total training steps to execute. |

---

## 4. Interactions & Dependencies

```text
                      ┌───────────────────┐
                      │       steps       │
                      └─────────┬─────────┘
            ┌───────────────────┼───────────────────┐
            ▼                   ▼                   ▼
┌───────────────────────┐ ┌───────────┐ ┌───────────────────────┐
│ learning_rate_        │ │checkpoint_│ │ save_checkpoint_     │
│ schedule_steps        │ │period     │ │ on_completion         │
│ (Schedule decay math) │ │(Cadence)  │ │ (Final state save)    │
└───────────────────────┘ └───────────┘ └───────────────────────┘
```

- **`learning_rate_schedule_steps`:** If `steps > learning_rate_schedule_steps`, the LR decays to `learning_rate * learning_rate_final_fraction` at `learning_rate_schedule_steps`, and subsequent steps run at constant LR `0.0`.
- **`checkpoint_period`:** Checkpoints are triggered at `step % checkpoint_period == 0`. Ensure `steps` is aligned with `checkpoint_period` if you want a periodic checkpoint right at the end.
- **`save_checkpoint_on_completion`:** When `step == steps`, MaxText saves a final checkpoint regardless of `checkpoint_period`.

---

## 5. Practical Scenarios & Failure Modes

- **Overriding on CLI for quick testing:** `python3 -m maxtext.trainers.pre_train.train ... steps=10` runs a 10-step sanity check.
- **Resuming past finished steps:** Resuming a checkpoint at step `100_000` with `steps: 100_000` causes MaxText to immediately save completion artifacts and exit without training.
- **Setting `steps: -1` without `learning_rate_schedule_steps`:** If `learning_rate_schedule_steps` is also `-1`, config validation raises an error.

---

### One-line intuition

> **`steps` defines the absolute global step count where the training loop terminates, acting as the master clock for the entire run.**
