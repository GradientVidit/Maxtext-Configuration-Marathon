## 1. Why does `managed_mldiagnostics_run_group` exist?

Large LLM training campaigns often consist of multiple interconnected runs (e.g. pretraining baseline, learning rate ablations, fine-tuning stages, and recovery restarts).

Without an explicit grouping key, individual runs appear disconnected in cloud monitoring dashboards:

```text
Diagnostics Dashboard:
Run Group: "llama3-70b-scaling-study"
  ├── Run 1: lr-1e-4
  ├── Run 2: lr-3e-4
  └── Run 3: lr-5e-4
```

`managed_mldiagnostics_run_group` assigns an experiment group tag to aggregate related runs in Cloud Diagnostics dashboards.

---

## 2. Fundamentals & Mechanics

- String tag passed as metadata to Google Cloud ML Diagnostics.
- **Default `""` (Empty):** Run is cataloged standalone under its `run_name`.

---

## 3. Options & Defaults

| Option | Value | Meaning |
|---|---|---|
| Default | `""` | No run group assigned. |
| Group Tag | String (e.g. `"gemma2-pretrain-v1"`) | Groups related runs in cloud dashboards. |

---

## 4. Interactions & Dependencies

- Tag attached when `managed_mldiagnostics: true`.

---

## 5. Practical Scenarios & Failure Modes

- Use consistent naming conventions (e.g. `"project-experiment-date"`) to simplify multi-run performance comparisons.

---

### One-line intuition

> **`managed_mldiagnostics_run_group` provides a shared grouping identifier to organize and compare related runs in Cloud Diagnostics.**
