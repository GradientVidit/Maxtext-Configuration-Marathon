
## 1. The preemption problem

Preemptible hardware (spot VMs, preemptible TPUs) can be reclaimed at any time. The OS/infrastructure sends a **SIGTERM signal** — a "you're about to be killed" warning — typically giving the process 30–90 seconds before actual termination (SIGKILL).

With standard fixed-period checkpointing:

```text
last checkpoint: step 40_000
current step:    step 49_999
SIGTERM received
→ process killed
→ 9,999 steps of work lost
```

The longer `checkpoint_period`, the worse the worst-case loss.

---

## 2. What `enable_autocheckpoint` does

```yaml
enable_autocheckpoint: true
```

MaxText registers a SIGTERM handler. When the signal arrives:

```text
SIGTERM received
         ↓
autocheckpoint triggers
         ↓
save checkpoint NOW (at current step)
         ↓
process exits cleanly
```

The checkpoint is saved at the exact step where preemption happened, not at the previous `checkpoint_period` multiple.

From MaxText's config:
```yaml
# enables autocheckpoint, which saves a checkpoint at the preemption step.
enable_autocheckpoint: false
```

---

## 3. The SIGTERM window

Infrastructure providers vary in how much notice they give:
- Google Cloud (preemptible TPUs): ~30 seconds
- AWS Spot: ~2 minutes
- Azure Spot: ~30 seconds

A 30-second window must be large enough to:
1. Finish the current step (or abort it cleanly)
2. Capture model state from accelerators
3. Serialize and write to GCS

For large models, step 2+3 may take longer than the window. This is why async checkpointing is important — if the checkpoint I/O pipeline is already warm, the actual "capture and initiate save" portion is brief; the slow write can continue in background for as long as the window allows.

---

## 4. Relationship to `enable_continuous_checkpointing`

These are complementary strategies:

```text
enable_continuous_checkpointing
    → continuously minimizes the gap between steps and last saved checkpoint
    → reduces work lost BEFORE the preemption signal

enable_autocheckpoint
    → catches the exact moment of preemption
    → eliminates the gap BETWEEN last continuous checkpoint and the preemption step
```

For maximum resilience on preemptible hardware, enable both:

```yaml
enable_continuous_checkpointing: true
enable_autocheckpoint: true
```

---

## 5. Relationship to `async_checkpointing`

When autocheckpoint fires, it benefits from `async_checkpointing: true`:
- State capture from accelerators is quick (synchronous part)
- GCS write continues asynchronously in the brief time before SIGKILL

Without async, the write must complete within the SIGTERM window, which may not be achievable for large models.

---

## 6. Default

```yaml
enable_autocheckpoint: false
```

Off by default — adds signal handler complexity and the behavior is only valuable on preemptible hardware.

---

## 7. Options

| Value | Behavior |
|---|---|
| `false` | No signal handler; preemption loses work since last periodic checkpoint |
| `true` | SIGTERM handler installed; checkpoint saved at preemption step |

---

### One-line intuition

> **`enable_autocheckpoint` installs a SIGTERM signal handler that triggers an immediate checkpoint save when the infrastructure signals preemption — preventing loss of all work since the last periodic checkpoint.**
