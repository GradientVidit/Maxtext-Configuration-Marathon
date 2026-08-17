
## 1. The problem with fixed-interval checkpointing

With `checkpoint_period: 10_000`, the save schedule is rigid:

```text
step 0       → possibly save (if save_checkpoint_on_start)
step 10_000  → save
step 20_000  → save
step 20_001  → preempted → lose 1 step
step 19_001  → preempted → lose 9_001 steps
```

The worst-case loss is almost `checkpoint_period` worth of work. On large clusters this can be hours.

---

## 2. What continuous checkpointing does

```yaml
enable_continuous_checkpointing: true
```

Instead of waiting for a fixed step count, Orbax initiates the **next checkpoint save as soon as the previous one finishes**:

```text
Training:  ─────────────────────────────────────→
Ckpt I/O:      [ckpt 1][ckpt 2][ckpt 3][ckpt 4]
```

The system always has the most recent checkpoint possible in flight. If preemption hits:

```text
worst-case loss ≈ time to write one checkpoint
```

rather than `checkpoint_period × step_time`.

---

## 3. How it interacts with `checkpoint_period`

When `enable_continuous_checkpointing: true`, the fixed `checkpoint_period` step trigger is **effectively superseded**. The system no longer waits for step multiples — it continuously cycles through saves as fast as async I/O allows.

This means `checkpoint_period` has minimal effect when continuous checkpointing is active.

---

## 4. Performance overhead

Writing a checkpoint continuously sounds expensive. In practice:

- Checkpoints are written **asynchronously** — the accelerator (TPU/GPU) keeps training while the CPU/network handles the previous checkpoint's I/O.
- The key cost is the **state capture** (snapshotting the current model/optimizer state from the accelerator's memory), which is brief.
- If saving a checkpoint takes longer than the time between captures, the system waits for the previous save to finish before capturing the next — it doesn't pile up an infinite queue.

Benchmarks on Llama-3.1-70B showed that continuous checkpointing meaningfully reduced mean time between checkpoints with only modest per-step overhead.

---

## 5. When to use it

| Scenario | Use continuous? |
|---|---|
| Highly preemptible environment (spot VMs, preemptible TPUs) | Yes |
| Long stable runs on reserved hardware | No — fixed period is fine |
| Runs where minimizing lost work is critical | Yes |
| I/O bandwidth is already saturated | Be careful — profile first |

---

## 6. Relationship to `enable_autocheckpoint`

These are complementary:

```text
enable_continuous_checkpointing
    → minimizes the gap between last completed checkpoint and current step
      under normal operation

enable_autocheckpoint
    → triggers an emergency checkpoint on the SIGTERM preemption signal
      (catches the moment of actual preemption)
```

For maximum resilience, both can be enabled together.

---

## 7. Default

```yaml
enable_continuous_checkpointing: false
```

Off by default — the standard fixed-period behavior is simpler and sufficient for non-preemptible hardware.

---

### One-line intuition

> **`enable_continuous_checkpointing` removes the fixed step-interval cadence and instead saves checkpoints as fast as async I/O allows — minimizing the amount of training progress lost to preemption.**
