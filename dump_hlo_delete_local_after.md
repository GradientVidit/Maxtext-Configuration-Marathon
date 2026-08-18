## 1. Why does `dump_hlo_delete_local_after` exist?

HLO files for 70B+ parameter models can easily consume 5–10 GB of disk per host.

On cloud VM root disks or ephemeral container filesystems, retaining these large temporary dump files after they have been safely copied to GCS risks filling up the disk and crashing training:

```text
Local Scratch Lifecycle:
XLA writes HLO to /tmp/ ──> Uploads to GCS ──>[ dump_hlo_delete_local_after: true ]──> Local files PURGED
```

`dump_hlo_delete_local_after` toggles automatic deletion of local HLO files once their remote upload finishes.

---

## 2. Fundamentals & Mechanics

- **`true` (Default):** Deletes contents of `dump_hlo_local_dir` immediately after GCS upload.
- **`false`:** Retains files on the local host disk for immediate local SSH inspection.

---

## 3. Options & Defaults

| Option | Value | Meaning |
|---|---|---|
| Default | `true` | Cleans up local scratch directory after GCS upload. |
| Retain Local | `false` | Keeps local files for direct VM shell analysis. |

---

## 4. Interactions & Dependencies

- Operates on files in `dump_hlo_local_dir`.

---

## 5. Practical Scenarios & Failure Modes

- Set `false` during local workstation development to avoid waiting for GCS downloads.

---

### One-line intuition

> **`dump_hlo_delete_local_after` automatically deletes local scratch HLO files after uploading them to GCS to avoid disk exhaustion.**
