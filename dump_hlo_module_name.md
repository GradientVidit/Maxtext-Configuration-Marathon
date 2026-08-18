## 1. Why does `dump_hlo_module_name` exist?

Even if multiple modules were written locally, engineers typically only want to upload the main training step graph to GCS to minimize network transfer time and storage cost:

```text
Local Scratch (/tmp/xla_dump/):
  jit_train_step.before_optimizations.txt ──> Uploaded to GCS (Matches dump_hlo_module_name)
  jit_train_step.after_optimizations.txt  ──> Uploaded to GCS (Matches dump_hlo_module_name)
  helper_eval.txt                         ──> Skipped from GCS upload
```

`dump_hlo_module_name` specifies the module name substring filter applied during GCS upload.

---

## 2. Fundamentals & Mechanics

- Default `"jit_train_step"` uploads only primary training graphs.
- Set to `""` to upload all locally dumped files.

---

## 3. Options & Defaults

| Option | Value | Meaning |
|---|---|---|
| Default | `"jit_train_step"` | Uploads only modules containing `"jit_train_step"`. |
| Upload All | `""` | Uploads all dumped files in local directory. |

---

## 4. Interactions & Dependencies

- Governs the upload filter from `dump_hlo_local_dir` to `dump_hlo_gcs_dir`.

---

## 5. Practical Scenarios & Failure Modes

- Ensure `dump_hlo_module_name` matches the substring dumped by `dump_hlo_local_module_name`.

---

### One-line intuition

> **`dump_hlo_module_name` filters which locally dumped HLO module files are uploaded to Google Cloud Storage.**
