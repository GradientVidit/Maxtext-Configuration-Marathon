## 1. Why does `dump_hlo_local_dir` exist?

XLA's native C++ compiler dumps HLO text and protobuf files directly to the local POSIX filesystem before any Python GCS upload logic can execute:

```text
XLA Compiler Core ──>[ Writes raw HLO files ]──> dump_hlo_local_dir ("/tmp/xla_dump/")
                                                          │
                                                    (Uploaded to GCS)
```

`dump_hlo_local_dir` defines the local filesystem directory where XLA initially writes raw HLO files.

---

## 2. Fundamentals & Mechanics

- Passed to `--xla_dump_to` in XLA compiler flags.
- Default `"/tmp/xla_dump/"` uses the local temporary ramdisk/scratch space.

---

## 3. Options & Defaults

| Option | Value | Meaning |
|---|---|---|
| Default | `"/tmp/xla_dump/"` | Standard local scratch directory. |
| Custom Path | `"/local/scratch/hlo"` | Custom high-capacity local NVMe mount. |

---

## 4. Interactions & Dependencies

```text
dump_hlo_local_dir ──> dump_hlo_delete_local_after (Cleanup)
```

---

## 5. Practical Scenarios & Failure Modes

- Ensure the target local directory has sufficient free space (at least 2–5 GB) to hold uncompressed HLO text files.

---

### One-line intuition

> **`dump_hlo_local_dir` sets the local scratch path where the XLA compiler initially writes raw HLO dump files.**
