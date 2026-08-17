
## 1. Why does it exist?

MaxText's primary metrics destination is GCS (via `gcs_metrics`) or W&B (via `enable_wandb`). Both require network and external services. For **local testing and CI** — where you want to assert on scalar values like loss or TFLOPs without spinning up GCS or W&B — writing to a local file is simpler and faster.

`metrics_file` is that local escape hatch.

```text
training step
    │
    ├──→ GCS metrics    (if gcs_metrics=true)
    ├──→ W&B            (if enable_wandb=True)
    └──→ local file     (if metrics_file != "")
```

---

## 2. What it writes

When set, MaxText writes scalar metrics — loss, step time, TFLOPs, learning rate, etc. — to the specified path in a structured text/JSON format, one record per training step.

The file is written **locally on the host that runs the Python process** (i.e., the coordinator host in multi-host training). This is not sharded or distributed.

---

## 3. Options

| Value | Behavior |
|---|---|
| `""` | No local metrics file written (default) |
| Any file path string | Write metrics to that file |

Default in base.yml:
```yaml
metrics_file: ""
```

Example:
```yaml
metrics_file: "/tmp/run_metrics.json"
```

---

## 4. When to use it

**Primary use case: unit tests and integration tests.**

MaxText's own test suite uses `metrics_file` to verify that a training run is numerically correct — e.g., check that loss is below a threshold after N steps, or that TFLOPs are within an expected range. The test reads the file after the run and asserts on values.

```text
test harness:
  run MaxText with metrics_file=/tmp/test_metrics.json
  → read /tmp/test_metrics.json
  → assert loss < 5.0 after 3 steps
```

For interactive development/debugging it's also useful to see metrics immediately in the filesystem without needing GCS access.

---

## 5. What it does NOT do

- It does not replace GCS metrics — both can be active simultaneously.
- It does not work as a distributed logging sink — only the coordinator writes to it.
- It does not handle log rotation or file size limits. Long runs will append indefinitely.

---

## 6. Interaction with other logging params

```text
metrics_file  → local file (for testing)
gcs_metrics   → GCS path under base_output_directory/run_name/metrics/
enable_wandb  → Weights & Biases external service
```

All three are independent. Setting one does not enable or disable the others.

---

### One-line intuition

> **`metrics_file` writes scalar training metrics to a local path — its real purpose is testing and CI, where you want to assert on loss/throughput values without touching GCS or W&B.**
