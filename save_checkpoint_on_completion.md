## What problem does it solve?

Suppose:

```yaml
checkpoint_period: 10_000
```

and your training ends at step `53,000`.

Periodic checkpointing gives you something like:

```text
0 ── 10k ── 20k ── 30k ── 40k ── 50k ── 53k
                                      ↑       ↑
                                  checkpoint  job ends
```

Without `save_checkpoint_on_completion`:

```text
latest checkpoint = 50k
```

With:

```yaml
save_checkpoint_on_completion: true
```

MaxText additionally saves the final state at completion:

```text
latest checkpoint = 53k
```

So the purpose is simply:

> **Don't lose the final training progress just because the final step doesn't coincide with `checkpoint_period`.**

---

## `true` vs `false`

```yaml
save_checkpoint_on_completion: true
```

→ save a final checkpoint when the run completes normally.

```yaml
save_checkpoint_on_completion: false
```

→ don't create this additional completion checkpoint.

It does **not** affect periodic checkpointing.

---

## Relationship with the other checkpoint parameters

Think of the three as:

```text
enable_checkpointing
        │
        └── Should checkpointing happen at all?
                    │
                    ↓
          ┌─────────┴─────────┐
          ↓                   ↓
 checkpoint_period     save_checkpoint_on_completion
          ↓                   ↓
    periodic saves       final save
```

And:

```yaml
save_checkpoint_on_start: true
```

is the corresponding **initial** save.

So:

```text
start                    periodic             completion
  │                          │                     │
  ↓                          ↓                     ↓
save_checkpoint_on_start   checkpoint_period   save_checkpoint_on_completion
```

---

## One important distinction

This is about **normal completion**.

If your TPU job is suddenly killed/preempted, this flag is not triggered because the process does not reach its natural end-of-training loop. 

- **Normal completion:** If the job finishes naturally at `steps` or `max_target_seconds`, `save_checkpoint_on_completion` ensures the final state is captured.
- **Crash/Preemption:** If the job is terminated unexpectedly, the final state is only protected by your `checkpoint_period` interval (periodic saves), or by `enable_autocheckpoint: true` if enabled — which defaults to `false`.

### One-line intuition

> **`save_checkpoint_on_completion: true` (the default) ensures MaxText saves the final training state when the job finishes normally — preventing loss of progress between the last periodic checkpoint and the final step.**