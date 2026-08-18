## 1. Why does `olmo_path_remap_from` exist?

An OLMo index JSON is typically generated on one machine (e.g. referencing `gs://my-bucket/dolma/`), but during training on TPU VMs, the data files might be accessed via a local FUSE mount (e.g. `/mnt/gcsfuse/dolma/`) or a different directory tree.

```text
Path inside Index JSON: "gs://my-bucket/dolma/shard_001.npy"
                              │
               [olmo_path_remap_from]: "gs://my-bucket/"
               [olmo_path_remap_to]:   "/mnt/disks/dolma/"
                              │
                              ▼
Actual Path Opened:     "/mnt/disks/dolma/shard_001.npy"
```

`olmo_path_remap_from` specifies the prefix to search for and replace in the index paths.

---

## 2. Mechanics

During file resolution, MaxText performs string prefix substitution:
```python
if path.startswith(config.olmo_path_remap_from):
    path = config.olmo_path_remap_to + path[len(config.olmo_path_remap_from):]
```

---

## 3. Options & Default

| Parameter | Type | Default | Options |
| :--- | :--- | :--- | :--- |
| `olmo_path_remap_from` | `str` | `''` | Substring / prefix to find in index paths |

---

## 4. Interactions with Related Parameters

- **`olmo_path_remap_to`**: The replacement string.
- **`dataset_type: olmo_grain`**: Active for OLMo pipelines.

---

## 5. Practical Scenarios & Failure Modes

| Scenario | Symptom / Failure | Fix |
| :--- | :--- | :--- |
| **Index built on GCS URIs but training uses local SSD mount** | `FileNotFoundError` trying to open `gs://` path locally | Set `olmo_path_remap_from: "gs://bucket/"` and `olmo_path_remap_to: "/mnt/ssd/"`. |

---

### One-line intuition

> `olmo_path_remap_from` defines the path prefix to search and replace in OLMo index files for environment path translation.
