## 1. Why does `eval_dataset_name` exist?

Evaluation may require measuring loss or perplexity on a different TFDS dataset than the training corpus (or a specific validation variant).

```text
Train: dataset_name = 'c4/en:3.1.0'
Eval:  eval_dataset_name = 'c4/en:3.1.0' (or 'lambada', 'wikitext', etc.)
```

`eval_dataset_name` independently sets the TFDS dataset name for evaluation passes.

---

## 2. Mechanics

MaxText instantiates an independent TFDS builder for evaluation using `eval_dataset_name` and `eval_split`.

---

## 3. Options & Default

| Parameter | Type | Default | Options |
| :--- | :--- | :--- | :--- |
| `eval_dataset_name` | `str` | `'c4/en:3.1.0'` | Valid TFDS dataset string |

---

## 4. Interactions with Related Parameters

- **`eval_split`**: Which split of `eval_dataset_name` to read.
- **`dataset_type: tfds`**: Active when TFDS is chosen.

---

## 5. Practical Scenarios & Failure Modes

| Scenario | Symptom / Failure | Fix |
| :--- | :--- | :--- |
| **Evaluating cross-domain generalization on Wikipedia** | Set `eval_dataset_name: 'wikipedia/20220301.en'` while keeping training on C4 | Configure `eval_dataset_name` to validation source. |

---

### One-line intuition

> `eval_dataset_name` designates the TFDS dataset used for validation and perplexity monitoring.
