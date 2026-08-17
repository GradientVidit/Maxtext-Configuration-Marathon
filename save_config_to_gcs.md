
## 1. Why does it exist?

MaxText runs involve many config sources: command-line overrides, model-specific config files, base.yml, environment variables. After all of these are merged and resolved, the actual config that drove a run can be meaningfully different from what you think you ran.

`save_config_to_gcs` captures the **resolved, post-merge config** and writes it to GCS, creating a reproducibility artifact that answers: "What exactly did this run see as its configuration?"

```text
Config resolution:
  base.yml
    + model_config.yml
    + CLI flags
    + env vars
    = resolved config
              ↓
    save_config_to_gcs=true
              ↓
    gs://{base_output_directory}/{run_name}/config.json (or similar)
```

---

## 2. What it writes

The fully-resolved config object — every parameter and its effective value for that run — is serialized and written to GCS at:

```text
{base_output_directory}/{run_name}/
```

This is stored alongside checkpoints and metrics, making the run directory a self-contained artifact.

---

## 3. Options

| Value | Behavior |
|---|---|
| `false` | Config not saved to GCS (default) |
| `true` | Resolved config written to GCS at run start |

Default in base.yml:
```yaml
save_config_to_gcs: false
```

---

## 4. Why the resolved config matters

Consider this scenario: you ran an experiment six months ago and want to reproduce it. You have the checkpoint. You think you know the config. But:

- Did you override `dtype` from the CLI that day?
- Was `quantization` set to `""` or `"int8"`?
- What was the exact `learning_rate` schedule that the model_config set?

Without the resolved config artifact, you're guessing. With `save_config_to_gcs=true`, you have ground truth.

---

## 5. When to use it

**Turn it on for any run you might want to reproduce or audit later.** Cost is negligible (a few KB of JSON). The cost of not having it when you need it — reconstructing configs from logs, git history, and memory — can be significant.

```text
always-on settings for production pretraining:
  save_config_to_gcs: true
  gcs_metrics: true
  enable_checkpointing: true
```

**Leave it off for:** quick local tests, debugging runs, benchmarks where reproducibility isn't needed.

---

## 6. Relationship to other run artifacts

```text
{base_output_directory}/{run_name}/
  checkpoints/   ← model state
  metrics/       ← scalars (if gcs_metrics=true)
  *.json         ← resolved config (if save_config_to_gcs=true)
```

Together, these three make a run fully self-documenting.

---

### One-line intuition

> **`save_config_to_gcs=true` writes the fully-resolved config (after all overrides are merged) to GCS at run start — the cheapest way to ensure future you can reproduce exactly what past you actually ran.**
