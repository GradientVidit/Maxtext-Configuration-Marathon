## 1. Why does `dataset_path` exist?

When using TensorFlow Datasets (`dataset_type: "tfds"`), dataset records (like C4) are stored as serialized shards in a local directory or Google Cloud Storage bucket.

```text
dataset_path: "gs://my-maxtext-dataset/c4/"
                     │
                     ▼
          Loads TFDS directory shards
                     │
                     ▼
          Feeds TFDS data pipeline
```

`dataset_path` provides the root filesystem path or GCS URI where TFDS datasets have been prepared or downloaded.

---

## 2. Mechanics

In `_tfds_data_processing.py`, `dataset_path` is passed to `tfds.builder_from_directory(dataset_path)` or `tfds.load(data_dir=dataset_path)`:

```text
[dataset_path] ──> tfds.builder_from_directory ──> tf.data.Dataset ──> Preprocessing
```

---

## 3. Options & Default

| Parameter | Type | Default | Options |
| :--- | :--- | :--- | :--- |
| `dataset_path` | `str` | `""` | Local directory path or GCS URI (`gs://bucket/path/`) |

---

## 4. Interactions with Related Parameters

- **`dataset_type`**: Must be `"tfds"`.
- **`dataset_name`**: Specifies the TFDS dataset name inside `dataset_path`.

---

## 5. Practical Scenarios & Failure Modes

| Scenario | Symptom / Failure | Fix |
| :--- | :--- | :--- |
| **Empty `dataset_path` with `dataset_type: "tfds"`** | TFDS attempts to download to default local user directory, filling root disk | Set `dataset_path: "gs://your-bucket/c4"` or pre-download dataset. |

---

### One-line intuition

> `dataset_path` specifies the GCS bucket or local directory path containing TFDS dataset shards for `dataset_type: tfds`.
