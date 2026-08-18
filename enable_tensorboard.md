## 1. Why does `enable_tensorboard` exist?

TensorBoard is the standard visualization toolkit for tracking machine learning metrics over time: loss curves, learning rate schedules, gradient norms, throughput metrics, and validation scores.

```text
Training Loop (Every step / log_period)
                  │
                  ▼ (enable_tensorboard: true)
          [ SummaryWriter ]
                  │
        ┌─────────┴─────────┐
        ▼                   ▼
Local Event Files     GCS Bucket / Vertex AI TensorBoard
(/tmp/tensorboard)    (gs://my-bucket/tensorboard)
```

`enable_tensorboard` is the master toggle for initializing the TensorBoard `SummaryWriter` and logging event files in MaxText.

---

## 2. What it actually controls

```yaml
enable_tensorboard: true
```

- When `true` (default): MaxText initializes a `SummaryWriter` pointed at `base_output_directory/tensorboard` (or Vertex AI) and writes scalar loss, step time, learning rate, and evaluation metrics at each `log_period`.
- When `false`: Disables TensorBoard event writing entirely (useful for pure benchmarking to eliminate file I/O).

---

## 3. Options and Defaults

| Value | Behavior | Overhead | Recommended Use |
|---|---|---|---|
| `true` (default) | Logs all metrics to TensorBoard event files | Minimal (Async writer) | Standard training, fine-tuning, evaluation |
| `false` | Completely disables TensorBoard writing | 0% | Micro-benchmarking, maximum I/O isolation |

---

## 4. Interactions and Output Destinations

- **`base_output_directory`**: Default destination path (`<base_output_directory>/<run_name>/tensorboard/`).
- **`use_vertex_tensorboard`**: When `true`, redirects logging to Google Cloud's managed Vertex AI TensorBoard.
- **`log_period`**: Controls the step frequency of TensorBoard scalar flushes.

---

## 5. Practical Scenarios

- **Standard Training**: Always keep `true`.
- **Viewing Logs**:
```bash
tensorboard --logdir=gs://my-bucket/maxtext/my_run/tensorboard
```

---

### One-line intuition

> **`enable_tensorboard` is the master switch that writes loss curves, learning rates, and throughput scalars to TensorBoard event files.**
