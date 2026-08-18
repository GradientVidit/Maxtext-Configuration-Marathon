## 1. Why does it exist?

Both **Multi-Tier Checkpointing** (`enable_multi_tier_checkpointing`) and **Emergency Checkpointing** (`enable_emergency_checkpoint`) rely on writing high-frequency intermediate state snapshots to local storage on each host VM rather than sending them across the network to a remote cloud bucket (GCS).

To execute these fast writes, Orbax needs a dedicated local directory path on each host's filesystem.

```text
Host VM Filesystem:
  /mnt/ramdisk/  <── `local_checkpoint_directory`
       └── {run_name}/
            └── checkpoints/
                 └── step_1020/ (High-speed local write, bypassing network)
```

`local_checkpoint_directory` specifies the local filesystem path on each host where frequent intermediate checkpoints are stored.

---

## 2. Fundamentals & Storage Types

The physical storage underlying `local_checkpoint_directory` determines write latency:

```text
1. Ramdisk (tmpfs / memory-backed):
   TPU HBM ──(PCIe/Host Bus)──→ Host RAM (/tmpfs/...)
   Latency: Sub-second. Zero disk I/O bottlenecks.

2. Local NVMe SSD:
   TPU HBM ──(PCIe)──→ Local NVMe Scratch Disk (/mnt/disks/local-ssd/...)
   Latency: Seconds. High capacity, survives process crashes (but not VM deletion).

3. Standard Root Boot Disk:
   TPU HBM ──(Network Block / Standard Disk)──→ /tmp/...
   Latency: Higher. Risk of filling root filesystem.
```

---

## 3. Options & Configuration

| Value | Meaning |
|---|---|
| `""` (default) | Empty string. Required to remain empty when emergency and multi-tier checkpointing are disabled. |
| `"/path/to/local/dir"` | Absolute path on the host VM (e.g., `"/mnt/ramdisk"` or `"/tmp/local_ckpt"`). |

Default in `base.yml`:
```yaml
local_checkpoint_directory: ""
```

---

## 4. Validation Rules & Constraints

MaxText strictly asserts:
- If `enable_emergency_checkpoint: true` OR `enable_multi_tier_checkpointing: true`:
  - `local_checkpoint_directory` **must not be empty** (`""`).
- If neither feature is enabled:
  - `local_checkpoint_directory` should remain empty.

```text
Features Enabled?
  YES (emergency or multi-tier) ──→ local_checkpoint_directory MUST be non-empty path
  NO                            ──→ local_checkpoint_directory must be ""
```

---

## 5. Practical Setup Recommendations

- **Ramdisk Mount on GKE**: In Kubernetes/GKE TPU pods, configure a `tmpfs` volume mount:
  ```yaml
  volumeMounts:
  - mountPath: /local_ramdisk
    name: ramdisk-volume
  volumes:
  - name: ramdisk-volume
    emptyDir:
      medium: Memory
      sizeLimit: 64Gi
  ```
  Then set `local_checkpoint_directory: "/local_ramdisk"`.
- **Capacity Management**: Ensure the mounted volume has enough capacity to hold at least 1–2 full uncompressed local checkpoints for all shards residing on that host.

---

### One-line intuition

> **`local_checkpoint_directory` points Orbax to the host's local RAM disk or NVMe scratch storage for ultra-fast, network-free intermediate checkpoint writes.**
