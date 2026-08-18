## 1. Why does `olmo_apply_ngram_filter` exist?

Web crawl corpora (such as Dolma or CommonCrawl) frequently contain degenerate, corrupted, or repetitive text sequences (e.g. repeated spam, endless copyright notices, boilerplate navigation headers). 

Training on repetitive sequences can corrupt language model representations and degrade downstream reasoning.

```text
Incoming Sequence: [ "buy now buy now buy now buy now buy now ..." ]
                                    │
                                    ▼
                       [olmo_apply_ngram_filter]
                                    │
        ┌───────────────────────────┴───────────────────────────┐
        ▼                                                       ▼
olmo_apply_ngram_filter: true               olmo_apply_ngram_filter: false
        │                                                       │
Repetitive n-gram detected                   Sequence passed to model
        ▼                                                       ▼
Masked / Skipped (Protected weights)         Risk of model degradation
```

`olmo_apply_ngram_filter: true` applies OLMo-core's high-performance n-gram repetition filter to detect and mask out repetitive instances.

---

## 2. Mechanics

Integrates OLMo's native C++/Cython n-gram filter:
- Computes repetition statistics over sliding token windows.
- Masks out samples where repetitive n-grams exceed threshold criteria.

---

## 3. Options & Default

| Parameter | Type | Default | Options |
| :--- | :--- | :--- | :--- |
| `olmo_apply_ngram_filter` | `bool` | `true` | `true` (filter repetitive n-grams), `false` (pass all instances) |

---

## 4. Interactions with Related Parameters

- **`dataset_type: olmo_grain`**: Used in OLMo data pipelines.

---

## 5. Practical Scenarios & Failure Modes

| Scenario | Symptom / Failure | Fix |
| :--- | :--- | :--- |
| **Pretraining on noisy web text** | Repetitive text creates severe loss spikes | Keep `olmo_apply_ngram_filter: true`. |
| **Domain-specific synthetic repeating data** | Legitimate repetitive sequences filtered out | Set `olmo_apply_ngram_filter: false`. |

---

### One-line intuition

> `olmo_apply_ngram_filter` filters out degenerate repetitive token patterns from OLMo training streams to safeguard model representation quality.
