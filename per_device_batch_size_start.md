## 1. Why does `per_device_batch_size_start` exist?

When batch-size ramp-up is enabled, the training loop must know where to begin its progression.

```text
Step 0: per_device_batch_size_start (e.g. 2.0)
                │
                ▼ (Gradual Increments)
Step N: per_device_batch_size (e.g. 16.0)
```

`per_device_batch_size_start` sets the initial per-device batch size used on step 0 of training.

---

## 2. Mechanics & Divisibility Constraint

MaxText requires that the difference between target and starting batch size divides cleanly by the increment:

$$rac{ ext{per\_device\_batch\_size} -  ext{per\_device\_batch\_size\_start}}{ ext{per\_device\_batch\_size\_increment}} \in \mathbb{Z}^+$$

```text
Example:
  per_device_batch_size_start = 4.0
  per_device_batch_size = 12.0
  per_device_batch_size_increment = 2.0
  Stages = (12 - 4) / 2 = 4 discrete stages (4 -> 6 -> 8 -> 10 -> 12)
```

---

## 3. Options & Default

| Parameter | Type | Default | Options |
| :--- | :--- | :--- | :--- |
| `per_device_batch_size_start` | `float` | `4.0` | Positive float less than `per_device_batch_size` |

---

## 4. Interactions with Related Parameters

- **`enable_rampup_batch_size`**: Master toggle.
- **`per_device_batch_size`**: Upper target limit.
- **`per_device_batch_size_increment`**: Step increment between stages.

---

## 5. Practical Scenarios & Failure Modes

| Scenario | Symptom / Failure | Fix |
| :--- | :--- | :--- |
| **`per_device_batch_size_start >= per_device_batch_size`** | Assertion error or ramp-up skipped immediately | Ensure start batch size is strictly smaller than target batch size. |
| **Uneven divisibility** | Irregular final ramp-up stage | Ensure `(target - start) % increment == 0`. |

---

### One-line intuition

> `per_device_batch_size_start` defines the initial per-device batch size at step zero from which the ramp-up schedule begins.
