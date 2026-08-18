## 1. Why does `grain_train_mixture_config_path` exist?

Production pretraining runs blend dozens of heterogeneous data domains (e.g., Wikipedia, arXiv, Math, Code, CommonCrawl, Books), each with custom sampling proportions and dataset paths. Specifying 50 weighted sources inside a single CLI string is impractical.

```text
grain_train_mixture_config_path: "gs://bucket/configs/mixture_v3.json"
                                    │
                                    ▼
       Loads JSON containing 20+ dataset sources & sampling weights
```

`grain_train_mixture_config_path` points to an external JSON configuration file defining the exact dataset mixture.

---

## 2. Mechanics & JSON Schema

```json
{
  "sources": [
    {"pattern": "gs://bucket/web/*.array_record", "weight": 0.5},
    {"pattern": "gs://bucket/code/*.array_record", "weight": 0.3},
    {"pattern": "gs://bucket/math/*.array_record", "weight": 0.2}
  ]
}
```

Grain parses this JSON and constructs a multi-source mixture dataset with deterministic pseudorandom interleaving.

---

## 3. Options & Default

| Parameter | Type | Default | Options |
| :--- | :--- | :--- | :--- |
| `grain_train_mixture_config_path` | `str` | `''` | Path or GCS URI to mixture JSON file |

---

## 4. Interactions with Related Parameters

- **`grain_train_files`**: Overridden or used as fallback if config path is omitted.
- **`grain_data_source_max_workers`**: Thread pool size for reading multiple mixture sources.

---

## 5. Practical Scenarios & Failure Modes

| Scenario | Symptom / Failure | Fix |
| :--- | :--- | :--- |
| **Invalid JSON syntax in mixture file** | `json.JSONDecodeError` on startup | Validate JSON schema before launching TPU cluster. |

---

### One-line intuition

> `grain_train_mixture_config_path` specifies a JSON file defining complex multi-source dataset mixtures and their sampling weights for Grain.
