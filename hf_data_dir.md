## 1. Why does `hf_data_dir` exist?

Some HuggingFace dataset repositories organize files into specific subdirectories (e.g. `data/`, `raw/`, `en/`).

```text
hf_path: "my-org/my-dataset"
hf_data_dir: "subfolder_v2/"
```

`hf_data_dir` mirrors the `data_dir` parameter in `datasets.load_dataset()`, instructing HuggingFace to scan a specific subdirectory inside the repository.

---

## 2. Mechanics

Passed into `datasets.load_dataset(..., data_dir=hf_data_dir)`.

---

## 3. Options & Default

| Parameter | Type | Default | Options |
| :--- | :--- | :--- | :--- |
| `hf_data_dir` | `str` | `''` | Subdirectory path within dataset repo |

---

## 4. Interactions with Related Parameters

- **`hf_path`**: Root repository path.

---

## 5. Practical Scenarios & Failure Modes

| Scenario | Symptom / Failure | Fix |
| :--- | :--- | :--- |
| **Target Parquet files live in `v1_cleaned/` subdir** | Root repo contains raw and processed data; loader reads raw by default | Set `hf_data_dir: "v1_cleaned"`. |

---

### One-line intuition

> `hf_data_dir` points Hugging Face's dataset loader to a specific subfolder within the repository.
