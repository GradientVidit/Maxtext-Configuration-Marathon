## 1. Why does `num_epoch` exist?

Machine learning models iterate across their training dataset multiple times when training sets are smaller than the desired training token budget (e.g. fine-tuning on domain data or small instruction sets).

```text
Dataset: [ Shard 1 | Shard 2 | Shard 3 ]
                      │
             ┌────────┴────────┐
             ▼                 ▼
       num_epoch: 1      num_epoch: 3
             │                 │
         1 Pass            3 Full Passes
     (Pretraining)        (Fine-Tuning)
```

`num_epoch` specifies the number of complete passes the dataset iterator performs before raising `StopIteration`.

---

## 2. Mechanics

In Grain, TFDS, and HuggingFace pipelines:
```python
dataset = dataset.repeat(num_epoch)
```
For pretraining runs over trillion-token corpora, `num_epoch` is usually set to `1` (or steps are bounded by `steps`). For SFT / DPO runs, `num_epoch` is set to `3` to `5` epochs with learning rate decay schedules.

---

## 3. Options & Default

| Parameter | Type | Default | Options |
| :--- | :--- | :--- | :--- |
| `num_epoch` | `int` | `1` | Positive integer (e.g. `1`, `3`, `5`) or `None`/`-1` for infinite repeat |

---

## 4. Interactions with Related Parameters

- **`steps`**: Total training optimizer steps. If `steps` is reached before `num_epoch` completes, training terminates at `steps`.
- **`dataset_type`**: Handled natively across Grain, TFDS, and HF dataset pipelines.

---

## 5. Practical Scenarios & Failure Modes

| Scenario | Symptom / Failure | Fix |
| :--- | :--- | :--- |
| **SFT dataset exhausted before target step count** | Training crashes with `OutOfRangeError` or premature termination | Increase `num_epoch` or configure infinite dataset repetition. |
| **Overfitting on small fine-tuning datasets** | Training loss drops to zero while validation loss spikes | Reduce `num_epoch` (e.g. from 10 to 3). |

---

### One-line intuition

> `num_epoch` sets the total number of complete passes the input pipeline makes across the training dataset before stopping.
