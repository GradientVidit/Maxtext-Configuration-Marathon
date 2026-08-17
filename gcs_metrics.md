
## 1. Why does it exist?

Training runs on TPU pods dump metrics (loss, step time, TFLOPs, learning rate) to the host process's stdout/stderr by default. That output lives only as long as the job's log stream. When you want to **persist metrics beyond the job's lifetime** and query them later without relying on W&B or an external service, you need a durable, queryable storage layer.

`gcs_metrics` writes those same scalars to GCS under a path that's automatically organized by run.

```text
base_output_directory/
  run_name/
    metrics/          ← gcs_metrics writes here
      000001.json
      000002.json
      ...
```

---

## 2. What gets written

MaxText writes scalar metrics: loss, step time (sec/step), TFLOPs, tokens/sec, learning rate, gradient norm, and any other scalars that the training loop reports. One record per training step or aggregated interval, depending on the implementation.

The GCS path is:
```text
{base_output_directory}/{run_name}/metrics/
```

This means your metrics are co-located with your checkpoints, automatically namespaced by run.

---

## 3. Options

| Value | Behavior |
|---|---|
| `false` | No GCS metrics written (default) |
| `true` | Scalars written to GCS after each step |

Default in base.yml:
```yaml
gcs_metrics: false
```

---

## 4. When to use it

**Production training runs** where you want metrics persisted independently of:
- The job log stream (which disappears when the job ends)
- External services like W&B (which require API keys, network to external endpoints, and incur cost)

GCS metrics are especially useful for:
- Postmortem analysis of a crashed run
- Comparing loss curves across runs by reading the GCS paths programmatically
- Feeding metrics into custom dashboards or BigQuery pipelines

---

## 5. Cost and performance considerations

Each metric write is a small GCS PUT. At typical step speeds (a few seconds per step), this is negligible overhead. For very short steps (sub-second), accumulating and batching writes before pushing to GCS would be preferable, but MaxText doesn't expose a batching knob for this.

Storage cost is minimal — scalar metrics in JSON format per step is tiny compared to checkpoints.

---

## 6. Interaction with other logging params

```text
gcs_metrics=true   → writes to GCS (durable, queryable, no external dependency)
enable_wandb=True  → writes to W&B (richer dashboards, external service)
metrics_file       → writes to local file (for testing/CI only)
save_config_to_gcs → saves resolved config to GCS (separate from metrics)
```

All are additive — enabling one doesn't disable the others.

---

## 7. Requirements

- `base_output_directory` must be a GCS path (`gs://...`). Local directory + `gcs_metrics=true` won't make sense — the write will target a local path structured like GCS.
- The service account running the job needs write access to the GCS bucket.

---

### One-line intuition

> **`gcs_metrics=true` durably persists training scalars (loss, TFLOPs, etc.) to GCS under `{base_output_directory}/{run_name}/metrics/`, so you can analyze them long after the job's log stream is gone.**
