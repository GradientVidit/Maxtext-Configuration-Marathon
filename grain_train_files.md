## 1. Why does `grain_train_files` exist?

**Grain** (`google-grain`) is Google's next-generation data loading library designed specifically for large-scale distributed ML. In Grain, datasets are sharded across many physical files on cloud storage (typically ArrayRecord format).

```text
Single Source:
grain_train_files: "gs://bucket/data/shard_*.array_record"

Multiple Weighted Sources (Mixture):
grain_train_files: "gs://bucket/web.array_record*,0.6;gs://bucket/code.array_record*,0.4"
```

`grain_train_files` specifies the path patterns and optional mixture weights for training data.

---

## 2. Mechanics & Inlined Mixture Syntax

Grain supports inline data mixtures using semicolon (`;`) and comma (`,`) delimiters:
`"pattern1,weight1;pattern2,weight2;..."`

```text
"gs://bucket/c4_*.array_record,0.7;gs://bucket/github_*.array_record,0.3"
                                   │
         ┌─────────────────────────┴─────────────────────────┐
         ▼                                                   ▼
  70% Sampling Probability                            30% Sampling Probability
   (Web Corpus Stream)                                  (Code Corpus Stream)
```

> [!NOTE]
> Multi-source weighted mixing via `grain_train_files` is supported only for `grain_file_type: 'arrayrecord'`.

---

## 3. Options & Default

| Parameter | Type | Default | Options |
| :--- | :--- | :--- | :--- |
| `grain_train_files` | `str` | `''` | Single glob pattern or semicolon-separated weighted patterns |

---

## 4. Interactions with Related Parameters

- **`dataset_type: grain`**: Master switch.
- **`grain_file_type`**: `'arrayrecord'` or `'parquet'`.
- **`grain_train_mixture_config_path`**: External JSON file alternative for complex mixtures.

---

## 5. Practical Scenarios & Failure Modes

| Scenario | Symptom / Failure | Fix |
| :--- | :--- | :--- |
| **No files match pattern** | `FileNotFoundError` or empty Grain iterator | Check GCS path and wildcards (`*`). |
| **Multi-file mixture with Parquet** | Unsupported format exception | Use ArrayRecord for weighted mixtures or specify single Parquet pattern. |

---

### One-line intuition

> `grain_train_files` defines the GCS path patterns and mixture weights for training data loaded through the high-performance Grain pipeline.
