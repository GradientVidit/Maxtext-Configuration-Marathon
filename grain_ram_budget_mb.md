## 1. Why does `grain_ram_budget_mb` exist?

When worker auto-tuning is enabled (`grain_worker_count: -1`), Grain must determine how many worker processes and buffer allocations can fit in memory without causing the host Linux OS to trigger the Out-Of-Memory (OOM) Killer.

```text
grain_ram_budget_mb: 1024 (1 GB RAM budget)
                      │
                      ▼
 Grain Auto-tuner calculates max safe worker count & buffer sizes
```

`grain_ram_budget_mb` provides the explicit RAM budget constraint used by `grain.experimental.pick_performance_config`.

---

## 2. Mechanics

Grain's performance tuner inspects record sizes and estimates memory consumption:
$$ ext{Max Workers} = \lfloor rac{ ext{grain\_ram\_budget\_mb}}{ ext{Estimated Memory per Worker}} 
floor$$

---

## 3. Options & Default

| Parameter | Type | Default | Options |
| :--- | :--- | :--- | :--- |
| `grain_ram_budget_mb` | `int` | `1024` | RAM budget in megabytes (e.g. `1024` = 1GB, `8192` = 8GB) |

---

## 4. Interactions with Related Parameters

- **`grain_worker_count: -1`**: Only used when auto-tuning is active.

---

## 5. Practical Scenarios & Failure Modes

| Scenario | Symptom / Failure | Fix |
| :--- | :--- | :--- |
| **TPU host has 32GB free RAM but auto-tuner picks only 1 worker** | Default `1024` MB budget is too conservative | Increase `grain_ram_budget_mb: 8192` (8GB). |

---

### One-line intuition

> `grain_ram_budget_mb` defines the memory ceiling used by Grain's auto-tuning algorithm to size data worker pools safely.
