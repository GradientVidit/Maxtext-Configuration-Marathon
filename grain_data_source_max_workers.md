## 1. Why does `grain_data_source_max_workers` exist?

When blending multi-source datasets (e.g. 10 different GCS paths in a mixture), Grain uses a `concurrent.futures.ThreadPoolExecutor` to query metadata, inspect shard counts, and initialize data streams across all sources simultaneously.

```text
Dataset Mixture (10 Sources)
             │
             ├── ThreadPoolExecutor (Capacity: grain_data_source_max_workers)
             │
             ├── Source 1 Init
             ├── Source 2 Init
             └── Source N Init
```

`grain_data_source_max_workers` caps the thread pool size to prevent excessive thread contention during dataset initialization.

---

## 2. Mechanics & Default

Defaults to `16`.

---

## 3. Options & Default

| Parameter | Type | Default | Options |
| :--- | :--- | :--- | :--- |
| `grain_data_source_max_workers` | `int` | `16` | Positive integer |

---

## 4. Interactions with Related Parameters

- **`grain_train_files` / `grain_train_mixture_config_path`**: Only used when mixing multiple data sources.

---

## 5. Practical Scenarios & Failure Modes

| Scenario | Symptom / Failure | Fix |
| :--- | :--- | :--- |
| **Large mixture with 100+ dataset sources** | Startup takes a long time due to serial source checks | Increase `grain_data_source_max_workers` to 32 or 64. |

---

### One-line intuition

> `grain_data_source_max_workers` sets the thread pool capacity for concurrently initializing and reading multiple Grain dataset sources in a mixture.
