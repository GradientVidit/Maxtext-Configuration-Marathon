## 1. Why it exists: configuration provenance and post-override verification

In MaxText, runtime parameters are composed dynamically through a multi-tier hierarchy of defaults, base YAML configs, model-specific overrides (via `model_name`), command-line flags, and environment variables:

```text
Configuration Resolution Cascade:
┌────────────────────────────────────────────────────────┐
│ 1. src/maxtext/configs/base.yml (Global Defaults)      │
├────────────────────────────────────────────────────────┤
│ 2. Model Specific Preset (e.g. models/llama3-70b.yml)  │
├────────────────────────────────────────────────────────┤
│ 3. Derived Parameters & Auto-Sharding Calculations     │
├────────────────────────────────────────────────────────┤
│ 4. CLI Arguments (e.g. `per_device_batch_size=4`)      │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
              Final Resolved Configuration
```

Without logging the final resolved configuration state:
- Developers debugging unexpected loss curves or OOMs cannot verify whether a specific CLI flag was overridden by a model preset or if a typo silently fell back to an unwanted default.
- Post-mortem analysis of failed or diverged distributed training runs is impossible without an immutable record of the exact hyperparameter state at step 0.

`log_config` instructs MaxText to print the complete, fully-resolved configuration dictionary to `stdout` at job initialization.

---

## 2. Mechanics: pyconfig resolution and stdout serialization

At the entrypoint of `train.py` or `inference_microbenchmark.py`, after `pyconfig.initialize()` processes all arguments:

```text
 Run `pyconfig.initialize(sys.argv)`
               │
               ▼
 Check: `config.log_config` (Default: `true`)
               │
               ▼
 ┌───────────────────────────────────────────────────────────┐
 │ Format Complete Config Dict into Pretty YAML / Key-Values │
 │ Print to `stdout` (Host 0):                               │
 │                                                           │
 │ ========================================================  │
 │                       MaxText Config                      │
 │ ========================================================  │
 │ run_name: "llama3-70b-v5p-run-1"                          │
 │ base_output_directory: "gs://my-bucket/maxtext"           │
 │ learning_rate: 0.00015                                    │
 │ mesh_axes: ['data', 'fsdp', 'tensor']                     │
 │ ici_tensor_parallelism: 8                                 │
 │ ... (all 100+ fully-resolved parameters)                  │
 │ ========================================================  │
 └─────────────────────────────┬─────────────────────────────┘
                               │
                               ▼
 Proceed with Mesh Creation and Model Initialization
```

Only **Host 0** (process 0 in distributed multi-host setups) prints this block to prevent log spam across hundreds of cluster worker nodes.

---

## 3. Options & Configuration

Default in `base.yml`:

```yaml
log_config: true
```

| Value | Behavior | Recommended Use Case |
|---|---|---|
| `true` (default) | Prints the full resolved config at startup on Host 0. | **Production runs, research experiments, CI logs** (ensures reproducible run provenance). |
| `false` | Silences config printing at startup. | Minimal log outputs, automated unit tests that assert strict stdout formatting. |

---

## 4. Parameter Interactions

```text
┌───────────────────────────────────────────────────────────┐
│                        log_config                         │
└─────────────┬───────────────────────────────┬─────────────┘
              │
              ▼
┌───────────────────────────────────────────────────────────┐
│ Captures and documents the resolved state of:             │
│ - model_name & override_model_config                      │
│ - logical_axis_rules & override_logical_axis_rules        │
│ - Hardware mesh dimensions (dcn_*, ici_*)                 │
│ - save_config_to_gcs (saves config file to GCS bucket)    │
└───────────────────────────────────────────────────────────┘
```

- **`save_config_to_gcs`**: While `log_config` prints the resolved parameters to console logs, `save_config_to_gcs: true` writes the YAML file to the GCS checkpoint directory. Together, they provide both console visibility and persistent storage provenance.

---

## 5. Practical Scenarios & Failure Modes

### Verifying CLI Overrides
When testing a new learning rate or sharding strategy via CLI:
```bash
python3 src/maxtext/train.py src/maxtext/configs/base.yml \
  model_name=llama2-7b \
  learning_rate=3e-4 \
  ici_fsdp_parallelism=4
```
Checking the `log_config` banner in the job log verifies that `learning_rate: 0.0003` and `ici_fsdp_parallelism: 4` took effect and were not overwritten by model preset defaults.

---

### One-line intuition

> **`log_config` prints the fully resolved configuration dictionary at startup, providing an immutable record of all active hyperparameters, derived dimensions, and CLI overrides.**
