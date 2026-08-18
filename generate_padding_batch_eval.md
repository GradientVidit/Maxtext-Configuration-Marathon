## 1. Why does `generate_padding_batch_eval` exist?

Validation benchmarks (such as HumanEval with 164 problems, or GSM8K with 1,319 problems) have fixed sample counts that rarely divide evenly by the global evaluation batch size.

```text
Benchmark: 1,319 samples. Eval Global Batch: 128 (10 full batches + 39 leftover)
Batch 0..9: 128 samples (Fast compiled execution)
Batch 10:   39 samples  ──> Shape change triggers JIT recompile or fails SPMD sharding
```

`generate_padding_batch_eval: true` pads the leftover evaluation examples into a full-sized batch with masked evaluation metrics, allowing the validation run to complete seamlessly without recompilation.

---

## 2. Mechanics

The evaluation loop generates dummy padding examples to complete the final batch. Metrics aggregation checks the valid sample mask and ignores padded instances from accuracy / perplexity scores.

---

## 3. Options & Default

| Parameter | Type | Default | Options |
| :--- | :--- | :--- | :--- |
| `generate_padding_batch_eval` | `bool` | `false` | `true` (pad eval tail), `false` (drop remainder) |

---

## 4. Interactions with Related Parameters

- **`eval_per_device_batch_size`**: Determines the eval batch dimensions.
- **`generate_padding_batch_train`**: Training counterpart.

---

## 5. Practical Scenarios & Failure Modes

| Scenario | Symptom / Failure | Fix |
| :--- | :--- | :--- |
| **Evaluation crashes at the end of validation loop** | Shape mismatch error on final validation step | Set `generate_padding_batch_eval: true`. |
| **Evaluation accuracy skewed by dummy samples** | Padded samples included in score calculation | Ensure metric masking is active when padding batches. |

---

### One-line intuition

> `generate_padding_batch_eval` fills the final evaluation batch with dummy instances so that evaluation runs to completion on fixed-size benchmark datasets without JIT recompilation.
