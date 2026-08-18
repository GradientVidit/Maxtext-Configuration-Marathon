## 1. Why does `hf_eval_split` exist?

HuggingFace datasets define validation partitions under various split names (e.g. `validation`, `eval`, `test`, `test[:1000]`).

```text
hf_eval_split: "validation"
                     │
                     ▼
  Pulls validation split from Hugging Face dataset
```

`hf_eval_split` identifies the evaluation split name in `datasets.load_dataset()`.

---

## 2. Mechanics

Specifies `split=hf_eval_split` during evaluation dataset creation.

---

## 3. Options & Default

| Parameter | Type | Default | Options |
| :--- | :--- | :--- | :--- |
| `hf_eval_split` | `str` | `''` | Split name string (e.g. `'validation'`, `'test'`) |

---

## 4. Interactions with Related Parameters

- **`hf_eval_files`**: Alternative way to supply eval data files.

---

## 5. Practical Scenarios & Failure Modes

| Scenario | Symptom / Failure | Fix |
| :--- | :--- | :--- |
| **Validation split named `'test'` on Hub** | Split `'validation'` not found | Set `hf_eval_split: "test"`. |

---

### One-line intuition

> `hf_eval_split` defines the split name used to load evaluation data from a Hugging Face dataset.
