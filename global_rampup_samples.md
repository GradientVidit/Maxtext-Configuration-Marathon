## 1. Why does `global_rampup_samples` exist?

Batch ramp-up needs a pacing clock to determine how many total training samples to process before the ramp-up concludes and full batch size is sustained.

```text
Sample Count: 0 ─────────────────────────────> global_rampup_samples ───> Full Run
Batch Size:   [ per_device_batch_size_start ] ───> [ Step Increments ] ───> [ Final Batch Size ]
```

`global_rampup_samples` specifies the total cumulative training samples (across all distributed devices) to process throughout the entire ramp-up phase.

---

## 2. Mechanics

MaxText tracks cumulative global samples:
$$ ext{Samples per Stage} = rac{ ext{global\_rampup\_samples}}{ ext{Number of Stages}}$$

At each stage boundary, MaxText increases the per-device batch size until `global_rampup_samples` is exhausted.

---

## 3. Options & Default

| Parameter | Type | Default | Options |
| :--- | :--- | :--- | :--- |
| `global_rampup_samples` | `int` | `500` | Any positive integer (e.g. `500` for testing, `10_000_000` for LLM pretraining) |

---

## 4. Interactions with Related Parameters

- **`enable_rampup_batch_size`**: Must be enabled.
- **`learning_rate_schedule_steps`**: Often configured so that batch ramp-up matches the duration of learning rate warmup.

---

## 5. Practical Scenarios & Failure Modes

| Scenario | Symptom / Failure | Fix |
| :--- | :--- | :--- |
| **Default `500` used in production run** | Ramp-up finishes in under 5 seconds, neutralizing its stabilizing benefits | Set `global_rampup_samples` to a meaningful number of samples (e.g. 5M–20M). |

---

### One-line intuition

> `global_rampup_samples` sets the sample threshold that governs how quickly the model transitions from initial to full training batch size.
