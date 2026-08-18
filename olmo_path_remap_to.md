## 1. Why does `olmo_path_remap_to` exist?

Acts as the target replacement string paired with `olmo_path_remap_from`.

```text
Original Path: [olmo_path_remap_from] + remainder
                       │
                       ▼
Resolved Path: [olmo_path_remap_to]   + remainder
```

`olmo_path_remap_to` provides the destination prefix substituted into OLMo dataset paths.

---

## 2. Mechanics

Applied directly during memory-mapping of `.npy` arrays in `_olmo_data_processing.py`.

---

## 3. Options & Default

| Parameter | Type | Default | Options |
| :--- | :--- | :--- | :--- |
| `olmo_path_remap_to` | `str` | `''` | Replacement directory prefix string |

---

## 4. Interactions with Related Parameters

- **`olmo_path_remap_from`**: Matching prefix.

---

## 5. Practical Scenarios & Failure Modes

| Scenario | Symptom / Failure | Fix |
| :--- | :--- | :--- |
| **Mounting bucket under `/mnt/gcs/`** | Fast local access via GCSFuse | Set `olmo_path_remap_to: "/mnt/gcs/"`. |

---

### One-line intuition

> `olmo_path_remap_to` defines the replacement prefix string applied to rewrite OLMo dataset file locations.
