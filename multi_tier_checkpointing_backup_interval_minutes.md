## 1. Why does it exist?

When using Multi-Tier Checkpointing (`enable_multi_tier_checkpointing: true`), training state is saved frequently to fast local storage (such as a host ramdisk). However, local storage is ephemeral — if all hosts or VMs in a cluster are terminated simultaneously, all local checkpoints are lost.

To guarantee durability, local checkpoints must periodically be backed up to persistent cloud storage (GCS).

```text
Local Storage (Ramdisk) ───[ every 20 steps ]───→ Rapid Micro-Save (Ephemeral)
                                                        │
                      [ every N minutes ]───────────────┘
                              ↓
              GCS Bucket (Persistent Backup)
```

`multi_tier_checkpointing_backup_interval_minutes` specifies the **wall-clock interval (in minutes)** at which the latest local checkpoint is asynchronously synced to persistent GCS storage.

---

## 2. Fundamentals & Mechanics

- **Time-based scheduling**: Rather than counting training steps (which may fluctuate depending on batch size, dataloader stalls, or dynamic sequence lengths), this parameter triggers GCS offloading on elapsed wall-clock minutes.
- **Asynchronous Sync**: The backup process copies data from local disk/ramdisk to GCS in the background without halting accelerator matrix multiplication units.

```text
Time (min):  0 ─── 10 ─── 20 ─── 30 ─── 40 ─── 50 ─── 60
Local Save:  |--|--|--|--|--|--|--|--|--|--|--|--|--|--| (frequent)
GCS Sync:    ▲                           ▲             (e.g. interval = 30 min)
             └─ Step 300 uploaded        └─ Step 900 uploaded
```

---

## 3. Options & Configuration

| Value | Meaning |
|---|---|
| `null` / `None` (default) | Time-based backup is disabled. (Must set `multi_tier_checkpointing_backup_interval_steps` instead if MTC is enabled). |
| Positive float/int (e.g. `30` or `60`) | Backs up local checkpoints to GCS every `N` minutes of active training. |

Default in `base.yml`:
```yaml
multi_tier_checkpointing_backup_interval_minutes: null
```

---

## 4. Mutual Exclusivity Rule

MaxText enforces that **exactly one** backup interval method is active when `enable_multi_tier_checkpointing: true`:

```text
Valid combinations:
  1) backup_interval_minutes: 30  AND  backup_interval_steps: null   [VALID]
  2) backup_interval_minutes: null AND backup_interval_steps: 1000   [VALID]

Invalid combinations:
  3) backup_interval_minutes: 30  AND  backup_interval_steps: 1000   [INVALID - CONFLICT]
  4) backup_interval_minutes: null AND backup_interval_steps: null   [INVALID - NO BACKUP]
```

---

## 5. Practical Tuning Advice

- **When to choose minutes over steps**: Wall-clock time is preferable when debugging or during variable-throughput phases (e.g., pipeline warmups or variable sequence lengths), ensuring predictable cloud backup frequency.
- **Recommended Values**: Typically `30` to `60` minutes. Shorter intervals (e.g. `< 10` minutes) increase GCS network egress and metadata operations, while excessively long intervals (`> 180` minutes) increase worst-case recovery loss if an unrecoverable full-cluster crash occurs.

---

### One-line intuition

> **`multi_tier_checkpointing_backup_interval_minutes` sets the wall-clock cadence in minutes for syncing ephemeral local ramdisk checkpoints to durable GCS storage.**
