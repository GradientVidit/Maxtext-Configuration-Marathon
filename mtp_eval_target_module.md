
## 1. Why does `mtp_eval_target_module` exist?

Multi-Token Prediction adds multiple auxiliary heads that each predict a different future token position. During training, all heads contribute to the loss. During evaluation, you can ask:

**"How good is this model at predicting 2, 3, or N tokens ahead?"**

One specific metric is the **token acceptance rate** — conceptually related to speculative decoding: if the model predicts `t_{i+2}` while computing `t_i`, how often is that prediction correct? This is similar to the "acceptance rate" in speculative decoding where a draft model's predictions are accepted by a larger oracle.

`mtp_eval_target_module` selects **which MTP head to use for this evaluation** — the specific depth of future prediction to measure.

---

## 2. Default

```yaml
mtp_eval_target_module: 0
```

`0` = evaluation of MTP heads is disabled. No acceptance rate or similar metrics are computed. This is the correct default when `mtp_num_layers: 0` (since there are no heads to evaluate) and is a reasonable default even when MTP is enabled (if you don't care about per-head metrics).

---

## 3. The indexing convention

`mtp_eval_target_module` is **1-indexed**:

```text
mtp_eval_target_module: 0   → evaluation disabled
mtp_eval_target_module: 1   → evaluate the first MTP head (predicts t_{i+2})
mtp_eval_target_module: 2   → evaluate the second MTP head (predicts t_{i+3})
```

The "1" refers to the first auxiliary head, not the main head. The main head (predicting `t_{i+1}`) is always evaluated as part of the standard metrics.

---

## 4. Constraint: must be ≤ `mtp_num_layers`

```text
mtp_num_layers: 1, mtp_eval_target_module: 1  → valid (evaluating head 1 of 1)
mtp_num_layers: 1, mtp_eval_target_module: 2  → invalid (head 2 doesn't exist)
mtp_num_layers: 0, mtp_eval_target_module: 1  → invalid (no MTP heads at all)
```

---

## 5. What metrics it enables

When set to a valid head index, MaxText can compute:
- **Token acceptance rate**: the fraction of `t_{i+k}` predictions from MTP head k that match the ground truth
- This is a proxy for how useful the MTP heads would be as speculative decoding draft heads

---

## 6. DeepSeek-V3 context

DeepSeek-V3 uses MTP heads that can be repurposed for speculative decoding at inference — the head that predicted the correct next-next token during training can be used to generate draft tokens during inference. `mtp_eval_target_module` is how you monitor whether this is working.

---

## 7. Practical guidance

| Scenario | Setting |
|---|---|
| No MTP | `0` (default) |
| MTP enabled, only care about training loss | `0` |
| MTP enabled, want to monitor acceptance rate | `1` (or whichever head you care about) |
| MTP enabled, evaluating for speculative decoding use | `1` |

---

### One-line intuition

> **`mtp_eval_target_module` selects which MTP head (1-indexed) to evaluate during validation — set to `0` to disable MTP-specific evaluation, or to `1` to track the first auxiliary head's acceptance rate as a speculative decoding proxy.**
