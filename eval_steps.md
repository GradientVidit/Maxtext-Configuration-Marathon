## 1. Why does `eval_steps` exist?

Validation datasets can be massive (e.g., millions of tokens). Iterating through an entire validation dataset during periodic checkpoints would stall training for hours. 

Conversely, if an evaluation loop runs without a bounded step limit, the validation iterator will exhaust all data and crash with `OutOfRangeError` or iterator exhaustion exceptions:

```text
Without eval_steps:
Eval Loop ──> Iterates indefinitely ──> Validation Data Exhausted ──> CRASH / Exception!

With eval_steps = 50:
Eval Loop ──> Evaluates exactly 50 batches ──> Computes Mean Validation Loss ──> Resume Training
```

`eval_steps` sets the exact, bounded number of batches evaluated during each periodic evaluation pass.

---

## 2. What it actually controls

```yaml
eval_steps: -1
```

- When `-1` (default): Unset.
- When `> 0` (e.g. `50`, `100`): MaxText runs exactly `eval_steps` iterations over the evaluation dataset during each validation cycle and averages the validation loss.

---

## 3. Options and Defaults

| Value | Behavior | Metric Reliability | Eval Pause Duration |
|---|---|---|---|
| `-1` (default) | Unset | — | — |
| `10` – `20` | Rapid sample evaluation | Moderate variance | Very fast (~seconds) |
| `50` – `100` | Robust validation estimate | Low variance | Standard (~1–2 minutes) |
| `500`+ | Comprehensive validation | Extremely low variance | Extended pause |

---

## 4. Interactions and Best Practices

- **Must pair with `eval_interval`**: If `eval_interval > 0`, always explicitly set `eval_steps: N` (e.g., `50`).
- **Data Exhaustion Prevention**: Ensure that `eval_steps * eval_per_device_batch_size * num_devices` does not exceed the total validation dataset size per epoch unless the validation iterator is configured to repeat.

---

## 5. Practical Scenarios

- **Recommended Setup**:
```yaml
eval_interval: 500
eval_steps: 64
```

---

### One-line intuition

> **`eval_steps` bounds the number of validation batches processed per evaluation pass, preventing data exhaustion errors and controlling evaluation pause duration.**
