## 1. Why does it exist?

In large-scale machine learning, progress is tracked and benchmarked in **discrete optimization steps**. When using Multi-Tier Checkpointing (`enable_multi_tier_checkpointing: true`), engineers often prefer that permanent cloud storage checkpoints align strictly with deterministic training milestones (e.g., every 5,000 steps or matching learning rate decay schedules) rather than arbitrary wall-clock timestamps.

```text
Step Count:  0 ─── 500 ─── 1000 ─── 1500 ─── 2000
Local Tier:  |||||||||||||||||||||||||||||||||||||| (e.g. every 20 steps)
GCS Backup:                ▲                  ▲     (every 1000 steps)
```

`multi_tier_checkpointing_backup_interval_steps` sets the exact training step cadence at which local checkpoints are backed up to persistent cloud storage (GCS).

---

## 2. Fundamentals & Mechanics

- **Step-synchronous triggers**: At step $S$, if $S \pmod{\text{backup\_interval\_steps}} == 0$, the checkpointer initiates an asynchronous background upload of the local checkpoint snapshot to the remote `base_output_directory`.
- **Determinism**: Unlike time-based intervals, step intervals ensure that identical checkpoints are generated at identical steps across repeated runs, simplifying evaluation and ablation studies.

---

## 3. Options & Configuration

| Value | Meaning |
|---|---|
| `null` / `None` (default) | Step-based backup is disabled. (Must set `multi_tier_checkpointing_backup_interval_minutes` if MTC is enabled). |
| Positive integer (e.g. `1000`, `5000`) | Backs up to GCS every `N` training steps. |

Default in `base.yml`:
```yaml
multi_tier_checkpointing_backup_interval_steps: null
```

---

## 4. Interactions & Validation Constraints

```text
Constraint Rule:
  enable_multi_tier_checkpointing == true
  ==> (multi_tier_checkpointing_backup_interval_steps is not None) 
      XOR 
      (multi_tier_checkpointing_backup_interval_minutes is not None)
```

- **`local_checkpoint_period`**: Must be smaller than `multi_tier_checkpointing_backup_interval_steps`. For example, `local_checkpoint_period: 25` and `multi_tier_checkpointing_backup_interval_steps: 1000`.
- **`checkpoint_period`**: In standard single-tier checkpointing, `checkpoint_period` directly governs GCS writes. In multi-tier checkpointing, this backup interval takes precedence for GCS sync frequency.

---

## 5. Practical Guidance

- **Production Pretraining**: Using step-based intervals (e.g., `2500` or `5000` steps) is standard practice in production pretraining runs because it aligns checkpoint archives with validation evaluation routines and dataset epoch boundaries.
- **Troubleshooting**: Setting this to an excessively small number (e.g. `50` steps) defeats the purpose of multi-tier checkpointing by saturating GCS network bandwidth with constant cloud uploads.

---

### One-line intuition

> **`multi_tier_checkpointing_backup_interval_steps` sets the deterministic step frequency for transferring fast local ramdisk checkpoints to persistent cloud storage.**
