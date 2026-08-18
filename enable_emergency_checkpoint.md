## 1. Why does it exist?

During large-scale distributed training on preemptible/spot cloud instances, failures or preemptions can occur at any moment. Saving full checkpoints directly to remote cloud buckets (GCS) at high frequency is too slow and resource-intensive. However, saving infrequently risks losing hours of training progress.

```text
Without Emergency Checkpointing:
  Checkpoint to GCS every 5000 steps.
  Preemption at step 4999 ──→ Lose 4999 steps of work.

With Emergency Checkpointing:
  Regular Checkpoint: Saved to GCS every 5000 steps (Durable)
  Emergency Checkpoint: Saved to Local Disk/Ramdisk every 50 steps (Rapid)
  
  Preemption at step 4999 ──→ Restores from step 4950 via local host storage!
```

`enable_emergency_checkpoint` is an Orbax experimental feature that provides a streamlined, lightweight two-tier checkpointing mechanism: saving fast local copies on host storage at short step intervals alongside standard persistent saves to cloud storage.

---

## 2. Fundamentals & Restore Mechanics

Unlike Multi-Tier Checkpointing (which manages continuous periodic background GCS syncing intervals), Emergency Checkpointing operates as a simpler safety net:
1. **Periodic Local Saves**: Saves lightweight state snapshots to `local_checkpoint_directory` every `local_checkpoint_period` steps.
2. **Local Broadcast on Restore**: When a job restarts following a failure, Orbax first inspects the local storage of all available hosts. If a valid local checkpoint exists on any surviving slice/host, it broadcasts that state to the remaining workers rather than re-downloading from GCS.

```text
                       Training Loop
                             │
              Every `local_checkpoint_period` (e.g. 50 steps)
                             ↓
              Save to `local_checkpoint_directory` (Host RAM/NVMe)
                             │
              On Preemption & Restart:
              Check local directory across hosts
                 ├── Found?  ──→ Fast peer-to-peer broadcast & resume
                 └── Absent? ──→ Fallback to standard GCS checkpoint
```

---

## 3. Options & Configuration

| Value | Meaning |
|---|---|
| `false` (default) | Emergency checkpointing is disabled; training relies solely on standard GCS checkpointing. |
| `true` | Enables emergency checkpointing. Requires `local_checkpoint_directory` and `local_checkpoint_period > 0`. |

Default in `base.yml`:
```yaml
enable_emergency_checkpoint: false
```

---

## 4. Interactions & Requirements

```text
enable_emergency_checkpoint: true
  ├── local_checkpoint_directory: "/tmp/emergency_ckpt"  (Required, non-empty)
  └── local_checkpoint_period: 50                        (Required, positive int)
```

- **`enable_multi_tier_checkpointing`**: Multi-tier checkpointing is an evolution of emergency checkpointing. Do not enable both simultaneously.
- **`enable_checkpointing`**: Master checkpointing flag must remain `true`.

---

## 5. Practical Use Cases

- **Spot/Preemptible Clusters**: Highly recommended for large cluster training on cost-effective preemptible TPU/GPU nodes where single-host maintenance events occur frequently.
- **Ramdisk Mount**: For peak efficiency, mount `local_checkpoint_directory` onto a RAM disk (`tmpfs`) or local attached NVMe SSD rather than a slow spinning disk.

---

### One-line intuition

> **`enable_emergency_checkpoint` saves frequent local host checkpoints so restarted jobs can recover nearly all completed steps by broadcasting local state instead of pulling full checkpoints from GCS.**
