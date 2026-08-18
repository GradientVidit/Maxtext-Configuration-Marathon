## 1. Why does `elastic_backup_kind` exist?

When a distributed slice failure occurs during elastic training, the surviving slices must restore a recent, consistent copy of the model parameters and optimizer states before resuming computation on the resized mesh.

If every slice failure forced the job to read a standard Orbax checkpoint from Google Cloud Storage (GCS), the recovery process would suffer high network latency and metadata contention across storage buckets:

```text
GCS Checkpoint Recovery:
Slice Loss ──> Query GCS Manifest ──> Download gigabytes/terabytes from cloud ──> High recovery latency

Fast In-Memory Snapshot Recovery (elastic_backup_kind: "snapshot"):
Slice Loss ──> Read replicated in-memory state snapshot from surviving hosts ──> Immediate, sub-second recovery
```

`elastic_backup_kind` selects the underlying state backup mechanism used by the elastic training coordinator to preserve and restore model state across slice dropout events.

---

## 2. Mechanics & Backup Strategies

- **`"snapshot"` (Default)**: Maintains lightweight, asynchronous in-memory state snapshots across the cluster or fast local persistent storage. When a slice is lost, surviving slices reload the snapshot directly from high-speed inter-node memory buffers without GCS round-trips.
- **State Re-sharding**: Once the snapshot is loaded, Pathways automatically re-distributes and re-shards arrays across the new surviving TPU device mesh.

---

## 3. Options & Default

| Parameter | Type | Default | Options |
| :--- | :--- | :--- | :--- |
| `elastic_backup_kind` | `str` | `"snapshot"` | `"snapshot"`, `"checkpoint"` |

---

## 4. Interactions with Related Parameters

| Related Parameter | Interaction |
| :--- | :--- |
| `elastic_enabled` | `elastic_backup_kind` is active only when `elastic_enabled: true`. |
| `enable_checkpointing` | Standard Orbax periodic checkpointing to GCS operates independently for long-term disaster recovery. |

---

## 5. Practical Guidance

| Setting | Mechanism | Recommendation |
| :--- | :--- | :--- |
| `"snapshot"` (Default) | In-memory / fast local replication | **Recommended**: Delivers fastest recovery time after preemption. |
| `"checkpoint"` | Relies on disk/GCS checkpoints | Slower recovery, but minimizes local host RAM overhead. |

---

### One-line intuition

> `elastic_backup_kind` defines the state preservation mechanism (e.g. `"snapshot"`) used to quickly restore model state when an elastic training job reconfigures around lost slices.
