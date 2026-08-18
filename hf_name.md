## 1. Why does `hf_name` exist?

Many Hugging Face repositories contain multiple dataset configurations or subsets (e.g., in `wikitext`, subsets are `'wikitext-103-v1'`, `'wikitext-2-raw-v1'`; in MMLU, subsets represent subjects).

```text
HF Repo (hf_path): "wikitext"
                          │
          ┌───────────────┴───────────────┐
          ▼                               ▼
hf_name: "wikitext-2-raw-v1"     hf_name: "wikitext-103-v1"
```

`hf_name` specifies the exact dataset configuration name within the repository.

---

## 2. Mechanics

Directly passed to HuggingFace `load_dataset(path=hf_path, name=hf_name)`.

---

## 3. Options & Default

| Parameter | Type | Default | Options |
| :--- | :--- | :--- | :--- |
| `hf_name` | `str` | `''` | Dataset subset string or empty string if repo has no configurations |

---

## 4. Interactions with Related Parameters

- **`hf_path`**: Parent repository containing the configuration.
- **`dataset_type: hf`**: Required.

---

## 5. Practical Scenarios & Failure Modes

| Scenario | Symptom / Failure | Fix |
| :--- | :--- | :--- |
| **Multi-config dataset loaded without `hf_name`** | `ValueError: Config name is missing. Please pick one among: ['...']` | Set `hf_name` to the desired subset. |

---

### One-line intuition

> `hf_name` specifies the subset or configuration name within a multi-config Hugging Face dataset repository.
