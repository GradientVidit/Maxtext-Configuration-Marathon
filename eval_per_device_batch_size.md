## 1. Why does `eval_per_device_batch_size` exist?

Evaluation steps execute only the forward pass (inference) and do not retain intermediate activations for backpropagation. Consequently, evaluation requires significantly less High Bandwidth Memory (HBM) than training.

```text
Training Step: Forward Pass + Store Activations + Backward Pass (Gradients) + Optimizer State
               ==> High HBM usage (e.g. per_device_batch_size: 4.0)

Evaluation Step: Forward Pass Only (No Activations/Gradients Saved)
               ==> Low HBM usage (can support larger eval_per_device_batch_size: 8.0 or 16.0)
```

`eval_per_device_batch_size` allows configuring a larger batch size for evaluation passes to accelerate validation throughput and maximize accelerator utilization.

---

## 2. Mechanics & Fallback Logic

```text
eval_per_device_batch_size configured?
               │
       ┌───────┴───────┐
       ▼               ▼
     > 0.0           == 0.0 (Default)
       │               │
  Use specified   Fallback to training
 eval batch size   per_device_batch_size
```

When set to `0.0`, MaxText automatically mirrors `per_device_batch_size`.

---

## 3. Options & Default

| Parameter | Type | Default | Options |
| :--- | :--- | :--- | :--- |
| `eval_per_device_batch_size` | `float` | `0.0` | `0.0` (inherit from `per_device_batch_size`) or positive float |

---

## 4. Interactions with Related Parameters

- **`per_device_batch_size`**: Default fallback source when `eval_per_device_batch_size: 0.0`.
- **`eval_data_columns` / `eval_dataset_name`**: The evaluation data pipeline that consumes this batch size.
- **`generate_padding_batch_eval`**: Generates dummy padding batches to ensure eval dataset size matches device batch multiples.

---

## 5. Practical Scenarios & Failure Modes

| Scenario | Symptom / Failure | Fix |
| :--- | :--- | :--- |
| **Validation takes too long during training checkpoints** | Small eval batch underutilizes TPU cores | Set `eval_per_device_batch_size: 16.0` or `32.0`. |
| **Eval batch OOM on long context validation** | Eval sequence length longer than train length | Lower `eval_per_device_batch_size` below training batch size. |

---

### One-line intuition

> `eval_per_device_batch_size` customizes evaluation batch dimensions independently from training, taking advantage of lower memory overhead during forward-only validation.
