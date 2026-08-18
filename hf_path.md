## 1. Why does `hf_path` exist?

The HuggingFace Hub hosts tens of thousands of open-source datasets (e.g. `tiiuae/falcon-refinedweb`, `allenai/c4`, `Open-Orca/OpenOrca`). When `dataset_type: "hf"`, MaxText uses HuggingFace's `datasets.load_dataset()` to pull and stream records directly.

```text
hf_path: "HuggingFaceH4/ultrachat_200k"
                     │
                     ▼
       HuggingFace Hub / Local Path
                     │
                     ▼
       datasets.load_dataset(path=hf_path)
```

`hf_path` mirrors the `path` argument of `datasets.load_dataset()`, defining the repository ID or local script/directory on disk.

---

## 2. Mechanics

In `src/maxtext/input_pipeline/_hf_data_processing.py`:
```python
dataset = datasets.load_dataset(
    path=config.hf_path,
    name=config.hf_name or None,
    data_dir=config.hf_data_dir or None,
    data_files=config.hf_train_files or None,
    token=config.hf_access_token or None,
)
```

---

## 3. Options & Default

| Parameter | Type | Default | Options |
| :--- | :--- | :--- | :--- |
| `hf_path` | `str` | `''` | HF repo ID (e.g. `'roneneldan/TinyStories'`) or local directory path |

---

## 4. Interactions with Related Parameters

- **`dataset_type`**: Must be set to `"hf"`.
- **`hf_name`**: Sub-dataset configuration within `hf_path`.
- **`hf_access_token`**: Needed for gated repositories (e.g., Llama/Gemma datasets).

---

## 5. Practical Scenarios & Failure Modes

| Scenario | Symptom / Failure | Fix |
| :--- | :--- | :--- |
| **Gated dataset requires authorization** | `GatedRepoError` or 401 Unauthorized | Pass `hf_access_token` with valid read token. |
| **Local Arrow/Parquet directory** | HF attempts Hub lookup if path not absolute | Provide absolute local path to directory in `hf_path`. |

---

### One-line intuition

> `hf_path` designates the Hugging Face repository ID or local dataset directory passed to `datasets.load_dataset()`.
