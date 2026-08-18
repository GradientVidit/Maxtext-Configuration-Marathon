## 1. Why does it exist?

Every training run produces dozens of artifacts: model checkpoints, TensorBoard/W&B metrics, compiled HLO dumps, JAXPR graphs, and serialized run configurations. Without a centralized root path, outputs would either scatter across local host filesystems (which are ephemeral on cloud VMs/TPUs) or require manual path configuration for every single subsystem.

```text
Without base_output_directory:
  Checkpoints ──→ /tmp/checkpoints/ (lost on preemption)
  Metrics     ──→ gs://my-bucket/metrics/
  Configs     ──→ local filesystem
  HLO dumps   ──→ /var/log/xla/

With base_output_directory:
  gs://my-bucket/experiments/
       └── {run_name}/
            ├── checkpoints/
            ├── metrics/
            ├── configs/
            └── xla_dump/
```

`base_output_directory` defines the single GCS (or shared filesystem) root URI under which all run-specific subdirectories are automatically structured.

---

## 2. Fundamentals & Directory Structure

When `base_output_directory` and `run_name` are set, MaxText constructs deterministic subpaths:

```text
{base_output_directory}/{run_name}/
  ├── checkpoints/
  │    └── items/
  │         └── {step}/
  ├── metrics/
  │    └── events.out.tfevents.*
  ├── config.yaml
  └── xla_dump/
```

- **Persistence**: Because TPUs and compute nodes are ephemeral (especially on Spot/Preemptible VMs), storing outputs to GCS (`gs://...`) ensures that progress is never lost when a slice is restarted.
- **Auto-resume discovery**: Checkpoint restoration logic scans `{base_output_directory}/{run_name}/checkpoints/` to locate the latest completed step.

---

## 3. Options & Configuration

| Value | Meaning |
|---|---|
| `""` (default) | Empty string. MaxText will raise an error if checkpointing, GCS metrics, or config dumping is enabled without setting this. |
| `"gs://<bucket-name>/<prefix>"` | Standard GCS path. Recommended for all multi-host and cloud training runs. |
| `"/local/path"` | Local or NFS/Lustre filesystem path. Useful for single-host local debugging. |

Default in `base.yml`:
```yaml
base_output_directory: ""
```

---

## 4. Interactions with Related Parameters

```text
base_output_directory ──┬──→ {base_output_directory}/{run_name}/checkpoints (if enable_checkpointing: true)
                        ├──→ {base_output_directory}/{run_name}/metrics     (if gcs_metrics: true)
                        ├──→ {base_output_directory}/{run_name}/config.yaml (if save_config_to_gcs: true)
                        └──→ {base_output_directory}/{run_name}/xla_dump    (if dump_hlo_gcs_dir is default)
```

- **`run_name`**: Combined directly with `base_output_directory`. If `run_name: "llama3-70b-run1"`, all outputs live under `{base_output_directory}/llama3-70b-run1/`.
- **`load_parameters_path` / `load_full_state_path`**: These can point to explicit checkpoint paths inside any `base_output_directory` (e.g. from an earlier pretraining phase).
- **`gcs_metrics`**: When `true`, scalar metrics and loss trajectories are written to `{base_output_directory}/{run_name}/metrics/`.

---

## 5. Practical Scenarios & Pitfalls

- **Leaving it empty**: Launching a job with default `base_output_directory: ""` while `enable_checkpointing: true` causes training to crash at initialization or during the first checkpoint save attempt.
- **Bucket Permissions**: All TPU worker VMs in the pod slice must have write permissions (`roles/storage.objectAdmin` or equivalent IAM) to the target GCS bucket.
- **Trailing Slashes**: MaxText normalizes paths with `os.path.join` / `gfile`, but providing clean URIs without redundant slashes (e.g., `"gs://my-bucket/maxtext_runs"`) prevents directory duplication.

---

### One-line intuition

> **`base_output_directory` is the root GCS or filesystem storage path under which MaxText deterministically namespaces all checkpoints, metrics, and run artifacts by `run_name`.**
