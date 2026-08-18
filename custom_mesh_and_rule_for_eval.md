## 1. Why does it exist?

During pretraining, a model might be sharded with a configuration optimized for massive global batch throughput (such as high FSDP, Pipeline Parallelism, or Expert Parallelism). However, during periodic evaluation or validation decoding, the workload shifts:
- Batch size is much smaller (often 1 or 2 per device).
- Sequence length might be much longer (requiring Context Parallelism instead of pure FSDP).
- Expert Parallelism might create idle chips if evaluation batches do not uniformly saturate all experts.

Rather than running evaluation with sub-optimal training shardings or launching a separate cluster, MaxText allows specifying an alternate mesh and rule set specifically for the evaluation step.

```text
Training Phase:
  Uses `custom_mesh_and_rule` or default mesh (e.g. Pure FSDP + Expert Parallelism)

Evaluation Phase:
  Switches to `custom_mesh_and_rule_for_eval` (e.g. "ep-as-cp" for long sequence eval)
```

`custom_mesh_and_rule_for_eval` selects a custom named YAML configuration from `configs/mesh_and_rule/` applied exclusively during evaluation.

---

## 2. Options & Configuration

| Value | Meaning |
|---|---|
| `""` (default) | Evaluation uses the exact same mesh and logical axis rules as training. |
| String (e.g. `"ep-as-cp"`) | Loads `src/maxtext/configs/mesh_and_rule/<name>.yml` specifically for the evaluation loop. |

Default in `base.yml`:
```yaml
custom_mesh_and_rule_for_eval: ""
```

---

## 3. Interactions with Related Parameters

- **`logical_axis_rules_for_eval`**: Companion parameter for inline rule overrides.
- **`custom_mesh_and_rule`**: Sets the training-time mesh rule preset.

---

### One-line intuition

> **`custom_mesh_and_rule_for_eval` dynamically overrides the device mesh and sharding rules during evaluation steps (e.g. switching from Expert Parallelism to Context Parallelism).**
