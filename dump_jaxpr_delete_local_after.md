## 1. Why does `dump_jaxpr_delete_local_after` exist?

Jaxpr text dumps for huge models can reach gigabytes in size.

Deleting local files after successful GCS upload prevents ephemeral host disks from filling up:

```text
Staged /tmp/jaxpr_dump/ ──> Uploaded to GCS ──>[ dump_jaxpr_delete_local_after: true ]──> Local file deleted
```

`dump_jaxpr_delete_local_after` toggles automatic purging of local jaxpr files after remote upload.

---

## 2. Fundamentals & Mechanics

- Default `true` ensures clean scratch disks.

---

## 3. Options & Defaults

| Option | Value | Meaning |
|---|---|---|
| Default | `true` | Deletes local jaxpr dump files after upload. |
| Keep Local | `false` | Retains files locally on host disk. |

---

## 4. Interactions & Dependencies

- Operates on `dump_jaxpr_local_dir`.

---

## 5. Practical Scenarios & Failure Modes

- Set `false` for instant inspection when SSH'd directly into the worker VM.

---

### One-line intuition

> **`dump_jaxpr_delete_local_after` purges local jaxpr files after GCS upload to conserve host disk space.**
