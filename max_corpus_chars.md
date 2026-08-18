## 1. Why does `max_corpus_chars` exist?

In legacy TensorFlow Datasets (TFDS) and dynamic text-loading pipelines, loading an unbounded text corpus into memory or processing infinite streaming buffers without safety caps can trigger catastrophic host RAM exhaustions.

```text
Dataset Stream ──> [ Character Counter ] ──> Limit reached (max_corpus_chars)?
                                                        │
                                          ┌─────────────┴─────────────┐
                                          ▼                           ▼
                                      Cap text                    Continue
                                (Prevent Host OOM)
```

`max_corpus_chars` acts as a dataset-size safety valve and debugging limiter, bounding the total number of characters ingested from raw text sources.

---

## 2. Mechanics

In TFDS input pipelines (`_tfds_data_processing.py`), `max_corpus_chars` caps the character stream before batch packing and caching.

```text
Raw Corpus Stream ──> Enforce len(stream) <= max_corpus_chars ──> Tokenize & Pack
```

---

## 3. Options & Default

| Parameter | Type | Default | Valid Values |
| :--- | :--- | :--- | :--- |
| `max_corpus_chars` | `int` | `10_000_000` | Positive integer (e.g. `10_000_000` for ~10MB text debug sample, or higher for full runs) |

---

## 4. Interactions with Related Parameters

- **`dataset_type: tfds`**: Primarily consumed by TFDS preprocessing loops.
- **`num_epoch`**: Controls how many passes are made over the capped corpus.

---

## 5. Practical Scenarios & Failure Modes

| Scenario | Symptom / Failure | Fix |
| :--- | :--- | :--- |
| **TFDS training finishes prematurely after a few hundred steps** | `max_corpus_chars: 10_000_000` capped the dataset at 10M characters | Increase `max_corpus_chars` to `1_000_000_000` or switch to `dataset_type: grain`. |
| **Rapid local unit testing of data pipelines** | Downloading terabytes of C4 is slow | Set `max_corpus_chars: 100_000` for instant local pipeline validation. |

---

### One-line intuition

> `max_corpus_chars` sets a ceiling on total characters ingested by TFDS pipelines to prevent host memory exhaustion and facilitate quick pipeline testing.
