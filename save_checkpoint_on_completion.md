
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

If your TPU job is suddenly killed/preempted, this flag isn't what protects you. That's what **periodic checkpointing** and, in newer MaxText, features such as `enable_autocheckpoint` are for. The current config separately defines autocheckpointing as saving at the preemption step. ([GitHub](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/maxtext/configs/base.yml?utm_source=chatgpt.com "maxtext/src/maxtext/configs/base.yml at main · AI-Hypercomputer/maxtext · GitHub"))

### One-line intuition

> **`save_checkpoint_on_completion=true` ensures MaxText saves the final training state when the job finishes, even if the final step isn't a regular `checkpoint_period` boundary.**