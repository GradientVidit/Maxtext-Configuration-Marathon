
## 1. Why does it exist?

GCS metrics and local file logging give you raw scalar data — they don't give you **interactive visualizations, run comparison, system metrics overlay, or collaboration**. Weights & Biases (W&B) is the de-facto standard platform for those needs in ML research.

`enable_wandb` wires MaxText's metrics output to the W&B SDK so every logged scalar flows into your W&B dashboard in real time.

```text
training step
    │
    ├──→ GCS / local (if enabled)
    └──→ W&B run dashboard  (if enable_wandb=True)
               ↓
        project: wandb_project_name
        run:     wandb_run_name (or auto-named)
```

---

## 2. What gets sent to W&B

The same scalars MaxText logs elsewhere: loss, step time, TFLOPs, learning rate, gradient norm, memory usage, and any custom scalars the training loop emits. W&B receives these via `wandb.log()` calls, which the MaxText logging layer wraps when `enable_wandb=True`.

---

## 3. Options

| Value | Behavior |
|---|---|
| `False` | W&B disabled (default) |
| `True` | Enables W&B logging; requires W&B API key to be set in environment |

Default in base.yml:
```yaml
enable_wandb: False
```

---

## 4. Prerequisites

W&B logging requires:
1. The `wandb` Python package installed in the environment
2. A W&B API key available — typically via `WANDB_API_KEY` environment variable or `wandb login` having been run previously
3. Network access from the training hosts to `api.wandb.ai`

On TPU pods, step 3 is the most likely failure point — worker VMs may not have outbound internet access depending on the VPC configuration.

---

## 5. Companion parameters

| Param | Role |
|---|---|
| `wandb_project_name` | W&B project to log runs under — defaults to `"maxtext"` |
| `wandb_run_name` | Name of the W&B run — defaults to `""` (W&B auto-assigns) |

These only have effect when `enable_wandb=True`. When W&B is off, they're ignored.

---

## 6. W&B vs GCS metrics

| | W&B | GCS metrics |
|---|---|---|
| Interactive dashboard | ✓ | ✗ |
| Run comparison | ✓ | manual |
| Real-time updates | ✓ | ✗ |
| External service required | ✓ | ✗ (just GCS) |
| Queryable programmatically | via W&B API | GCS reads |
| Cost | W&B plan pricing | GCS storage |

For production training on GCP, many teams use both: W&B for interactive monitoring during the run, GCS metrics for durable archival.

---

## 7. What breaks if wrong

If `enable_wandb=True` but W&B is not authenticated or network is blocked, the training run will fail or hang at the W&B initialization step. This is a common gotcha on TPU pod workers without external internet.

---

### One-line intuition

> **`enable_wandb=True` routes MaxText's training metrics into Weights & Biases for real-time dashboards and run comparison — but requires W&B auth and outbound network access from all training hosts.**
