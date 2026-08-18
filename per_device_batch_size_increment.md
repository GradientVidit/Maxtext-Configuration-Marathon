## 1. Why does `per_device_batch_size_increment` exist?

Batch-size ramp-up should not jump abruptly from a small batch to a huge batch in a single step; it must expand smoothly through intermediate stages.

```text
Start: 4.0 ──[+2.0]──> 6.0 ──[+2.0]──> 8.0 ──[+2.0]──> 10.0 ──[+2.0]──> Target: 12.0
```

`per_device_batch_size_increment` defines the step size by which the per-device batch size increases at each transition boundary during the ramp-up period.

---

## 2. Mechanics

The total number of ramp-up stages $K$ is:
$$K = rac{ ext{per\_device\_batch\_size} -  ext{per\_device\_batch\_size\_start}}{ ext{per\_device\_batch\_size\_increment}}$$

Each stage processes an equal fraction of `global_rampup_samples`, steadily increasing the accelerator batch workload until full capacity is reached.

---

## 3. Options & Default

| Parameter | Type | Default | Options |
| :--- | :--- | :--- | :--- |
| `per_device_batch_size_increment` | `float` | `2.0` | Positive float that evenly divides `(per_device_batch_size - per_device_batch_size_start)` |

---

## 4. Interactions with Related Parameters

- **`enable_rampup_batch_size`**: Must be `true`.
- **`per_device_batch_size_start` & `per_device_batch_size`**: Define the interval bounds.

---

## 5. Practical Scenarios & Failure Modes

| Scenario | Symptom / Failure | Fix |
| :--- | :--- | :--- |
| **Increment does not divide interval evenly** | Truncation or abrupt jump on final stage | Set increment to an exact divisor (e.g., for start=2 and target=10, increment=2 or 4). |

---

### One-line intuition

> `per_device_batch_size_increment` dictates the step size added to the batch size at each stage transition during batch ramp-up.
