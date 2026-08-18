## 1. Why does it exist?

While `logical_axis_rules` defines the global mapping of logical tensor dimensions to physical mesh axes for the training forward/backward pass, evaluation steps often have different sharding requirements (e.g. evaluating with different head partitioning or sequence parallelism).

Instead of replacing the full mesh via an external file, `logical_axis_rules_for_eval` allows providing an inline list of axis rules applied specifically during evaluation.

```text
Training:   uses `logical_axis_rules`
Evaluation: uses `logical_axis_rules_for_eval` (if non-empty)
```

`logical_axis_rules_for_eval` provides an evaluation-specific override list for logical axis sharding rules.

---

## 2. Options & Configuration

| Value | Meaning |
|---|---|
| `[]` (default) | Empty list. Evaluation inherits the exact `logical_axis_rules` used in training. |
| List of `[logical_name, [mesh_axes...]]` pairs | Replaces logical axis rules during evaluation execution. |

Default in `base.yml`:
```yaml
logical_axis_rules_for_eval: []
```

---

## 3. Interactions & Usage

- If `logical_axis_rules_for_eval` is empty (`[]`), MaxText evaluates using the training rules.
- If populated, the evaluation graph is compiled with these sharding rules, allowing distinct tensor partitioning for validation loss and perplexity benchmarks.

---

### One-line intuition

> **`logical_axis_rules_for_eval` lets you provide an inline list of logical-to-physical sharding rules specifically for evaluation steps while leaving training rules untouched.**
