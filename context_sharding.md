## 1. Why does it exist?

Context Parallelism (CP) splits long sequence dimensions across multiple devices. Normally, this partitioning occurs along the dedicated physical mesh axis named `'context'`.

However, in advanced configurations — such as running Mixture of Experts (MoE) models where you want to repurpose Expert Parallelism (EP) device axes for Context Parallelism during evaluation (`custom_mesh_and_rule="ep-as-cp"`) — the physical axis acting as the sequence partitioner changes from `'context'` to `'expert'`.

```text
Standard Training:
  Context Parallelism ──→ Sharded along physical axis: "context"

EP-as-CP Configuration (custom_mesh_and_rule="ep-as-cp"):
  Context Parallelism ──→ Sharded along physical axis: "expert"
```

`context_sharding` declares which physical mesh axis serves as the primary context-parallelism dimension for input sequence processing and load balancing.

---

## 2. Fundamentals & Options

| Value | When to Use |
|---|---|
| `"context"` (default) | Standard training and inference setups with a dedicated context-parallel axis. |
| `"expert"` | Only used in conjunction with `custom_mesh_and_rule: "ep-as-cp"` to reuse expert-parallel device axes for context parallelism. |

Default in `base.yml`:
```yaml
context_sharding: "context"
```

---

## 3. Interactions with Related Parameters

```text
custom_mesh_and_rule: "ep-as-cp" ──→ context_sharding: "expert"
                                        │
                                        ↓
                      Input sequences split across expert mesh axis
```

- **`custom_mesh_and_rule`**: When set to `"ep-as-cp"`, `context_sharding` is typically set to `"expert"`.
- **`ici_context_parallelism` vs `ici_expert_parallelism`**: When `context_sharding: "expert"`, sequence chunking scales with `ici_expert_parallelism`.
- **`context_parallel_load_balance`**: Operates along the physical axis specified by `context_sharding`.

---

## 4. Practical Guidelines

- Keep default `context_sharding: "context"` for all standard dense models and standard MoE training runs.
- Switch to `"expert"` only when testing long-context evaluation on MoE clusters without altering physical hardware allocations.

---

### One-line intuition

> **`context_sharding` selects which physical mesh axis (`"context"` or `"expert"`) carries the sequence-splitting dimension for context parallelism and load balancing.**
