## 1. Why does it exist?

Standard checkpointing writes model weights and optimizer states directly to remote cloud storage (e.g. GCS). At multi-terabyte scale, writing to GCS introduces network latency and I/O bottlenecks. If a training job saves checkpoints frequently to minimize preemption loss, GCS overhead hurts training throughput. If checkpoints are saved infrequently, a preemption can lose hours of expensive computation.

```text
Standard GCS Checkpointing:
  Train (steps 0..1000) ──→ Save to GCS (slow) ──→ Preemption at step 999 loses ~999 steps!

Multi-Tier Checkpointing:
  Local Tier (Ramdisk/SSD): Saved every 20 steps (near-zero latency)
  Backup Tier (GCS):        Backed up every 60 minutes
  
  Preemption at step 999 ──→ Restores from Local Tier at step 980 (lost only 19 steps!)
```

`enable_multi_tier_checkpointing` activates Orbax's multi-tiered checkpointing engine. It decouples fast, local checkpoint saves from slower, durable cloud backups.

---

## 2. Architecture & Mechanics

Multi-Tier Checkpointing (MTC) operates with a two-tiered hierarchy:

```text
               ┌───────────────────────────────┐
               │         TPU Workers           │
               └──────────────┬────────────────┘
                              │
               Step Period (e.g. every 20 steps)
                              ↓
      ┌─────────────────────────────────────────────────┐
      │  Tier 1: High-Speed Local Storage (Ramdisk/SSD) │  <-- Near-instant save
      └───────────────────────┬─────────────────────────┘
                              │
               Backup Interval (e.g. every 60 min)
                              ↓
      ┌─────────────────────────────────────────────────┐
      │  Tier 2: Persistent Cloud Storage (GCS Bucket)  │  <-- Durable backup
      └─────────────────────────────────────────────────┘
```

### Rapid Restore via Local Broadcast
When a single host or slice is preempted and restarted:
1. Orbax inspects `local_checkpoint_directory` across remaining alive workers.
2. If healthy slices retain the latest local checkpoint, workers **broadcast** the checkpoint across the fast interconnect (or DCN) rather than pulling terabytes from GCS.
3. If all local tiers are wiped (e.g. complete cluster recreation), Orbax falls back to the durable GCS tier.

---

## 3. Options & Configuration

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `enable_multi_tier_checkpointing` | `bool` | `false` | Master switch for multi-tier checkpointing. |

Default in `base.yml`:
```yaml
enable_multi_tier_checkpointing: false
```

When set to `true`, the following companion parameters **must** also be configured:
- `local_checkpoint_directory`: Path to ramdisk or local host NVMe (e.g. `"/local/ramdisk"`).
- `local_checkpoint_period`: Integer > 0 (e.g. `20`).
- Either `multi_tier_checkpointing_backup_interval_minutes` or `multi_tier_checkpointing_backup_interval_steps`.

---

## 4. Interactions with Related Parameters

```text
enable_multi_tier_checkpointing: true
  ├── local_checkpoint_directory: "/ramdisk"  (Must be non-empty)
  ├── local_checkpoint_period: 20             (Must be > 0)
  ├── multi_tier_checkpointing_backup_interval_minutes: 30 (Or steps variant)
  └── mtc_data_parallelism: 0                 (Auto-derived to num_slices)
```

- **`enable_checkpointing`**: Must be `true`.
- **`enable_emergency_checkpoint`**: Mutually exclusive in practice — MTC is a superset of emergency checkpointing with scheduled GCS offloading.
- **`base_output_directory`**: Used as the destination for the persistent GCS backup tier.

---

## 5. Practical Scenarios & Failure Modes

- **Ramdisk Configuration**: On Google Kubernetes Engine (GKE), ramdisks are typically mounted via CSI drivers (`emptyDir` with `medium: Memory`).
- **Disk Full Errors**: If `local_checkpoint_period` is very frequent and local checkpoints are not pruned, ramdisks can run out of memory. Orbax handles rolling retention internally.
- **Missing Required Settings**: Setting `enable_multi_tier_checkpointing: true` without specifying `local_checkpoint_directory` or a backup interval raises an immediate assertion error during config validation.

---

### One-line intuition

> **`enable_multi_tier_checkpointing` enables a two-tier Orbax checkpointer that writes rapid micro-checkpoints to local host ramdisk while periodically syncing durable snapshots to GCS, drastically cutting preemption rollback losses.**
