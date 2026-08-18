## 1. Why does `enable_rampup_batch_size` exist?

At the very beginning of training a large language model from scratch, the model weights are randomly initialized. Gradients computed across a massive global batch (e.g. millions of tokens) can be noisy and unstable, leading to early loss spikes or divergence.

```text
Training Start:
Without Ramp-up: Step 0 begins immediately at FULL batch size (e.g. 4M tokens)
                 ──> Risk of early gradient instability / loss spike

With Ramp-up:    Step 0 starts at small batch (e.g. 500k tokens)
                 Step 100 ──> 1M tokens
                 Step 200 ──> 2M tokens
                 Step 500 ──> Full 4M tokens (Stable, conditioned weights)
```

Batch-size ramp-up (popularized by OpenAI GPT-3 and NVIDIA Megatron-LM) stabilizes early optimization by gradually increasing the batch size from a smaller initial value up to the full target batch size across the first several hundred thousand samples.

`enable_rampup_batch_size` is the master switch enabling this dynamic batch scheduling.

---

## 2. Mechanics & Stage Calculations

When enabled, MaxText divides the ramp-up phase into discrete linear stages:

```text
per_device_batch_size_start (e.g. 4.0)
         │
         ▼  (+ per_device_batch_size_increment e.g. 2.0)
   Batch = 6.0
         │
         ▼  (+ 2.0)
   Batch = 8.0
         │
         ▼
   per_device_batch_size (e.g. 12.0) [Target reached]
```

The step count per stage is computed using `global_rampup_samples`. Once the cumulative samples processed exceed `global_rampup_samples`, training locks into the final `per_device_batch_size`.

---

## 3. Options & Default

| Parameter | Type | Default | Options |
| :--- | :--- | :--- | :--- |
| `enable_rampup_batch_size` | `bool` | `false` | `true` (enable gradual batch expansion), `false` (fixed batch size from step 0) |

---

## 4. Interactions with Related Parameters

```text
enable_rampup_batch_size: true
            │
  ┌─────────┼─────────────────────────────┐
  ▼         ▼                             ▼
per_device_batch_size_start  per_device_batch_size_increment  global_rampup_samples
(Starting point)             (Step size per stage)            (Ramp-up duration)
```

- **`per_device_batch_size`**: The terminal batch size target.
- **`learning_rate_schedule_steps`**: Often combined with learning rate warmup for optimal stability.

---

## 5. Practical Scenarios & Failure Modes

| Scenario | Symptom / Failure | Fix |
| :--- | :--- | :--- |
| **Pretraining a 70B+ model from scratch** | Loss diverges in the first 50 steps at full batch size | Set `enable_rampup_batch_size: true` with 5M–10M sample ramp-up. |
| **Fine-tuning / SFT runs** | Fine-tuning starts from pretrained weights and does not need batch ramp-up | Keep `enable_rampup_batch_size: false`. |

---

### One-line intuition

> `enable_rampup_batch_size` activates progressive batch size scaling during initial training steps to prevent early gradient explosions and stabilize optimization.
