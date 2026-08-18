## 1. Why does `eval_interval` exist?

Monitoring validation loss during training is critical for detecting overfitting, learning rate anomalies, or data contamination. However, running an evaluation loop pauses accelerator training:

```text
Training Loop Timeline:
[ Train 1,000 Steps ] ──> [ Pause: Run Eval Loop (eval_steps) ] ──> [ Resume Training 1,000 Steps ]
                                 ▲
                                 │
                          eval_interval = 1000
```

If evaluation runs too frequently, TPU compute is wasted on non-training inference. If run too infrequently, overfitting or loss divergence is detected too late.

`eval_interval` sets the step frequency at which MaxText pauses training to run an evaluation pass.

---

## 2. What it actually controls

```yaml
eval_interval: -1
```

- When `-1` (default): Evaluation during training is **disabled**.
- When `> 0` (e.g. `500`, `1000`): MaxText pauses training every `eval_interval` steps, iterates over validation batches for `eval_steps`, logs validation metrics to TensorBoard/GCS, and resumes training.

---

## 3. Options and Defaults

| Value | Meaning | Training Impact | Typical Use Case |
|---|---|---|---|
| `-1` (default) | Evaluation loop disabled | 0% overhead | Fast smoke tests, pure throughput benchmarking |
| `250` – `500` | Frequent evaluation | ~5–10% overhead | Fine-tuning runs, small dataset convergence monitoring |
| `1000` – `5000` | Periodic evaluation | < 2% overhead | Large-scale LLM pretraining runs |

---

## 4. Interactions and Requirements

- **`eval_steps`**: Must be configured with `eval_interval > 0` to define how many validation batches are evaluated.
- **Eval Dataset Configs**: Requires validation dataset parameters (`eval_dataset_name`, `grain_eval_files`, or `hf_eval_split`) to be defined.

---

## 5. Practical Scenarios

- **Production LLM Training**:
```yaml
eval_interval: 1000
eval_steps: 50
eval_start_step: 1000
```

---

### One-line intuition

> **`eval_interval` sets the training step cadence for running validation passes, or `-1` to disable evaluation entirely.**
