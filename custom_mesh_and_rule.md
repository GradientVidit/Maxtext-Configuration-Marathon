## 1. Why does it exist?

MaxText models are completely decoupled from physical hardware layout through two core data structures:
1. `mesh_axes`: The list of physical mesh axes (e.g. `data`, `fsdp`, `tensor`, `expert`, `context`).
2. `logical_axis_rules`: The mapping of logical model tensor dimensions (e.g. `activation_batch`, `heads`, `mlp`, `embed`) to physical mesh axes.

In default training, these definitions are read from `base.yml`. However, specialized execution patterns — such as repurposing expert-parallel axes for context parallelism during inference (`ep-as-cp`), or applying custom 2D/3D parallelism strategies — require redefining both the mesh axes and the entire rule table simultaneously.

Instead of overriding dozens of lines in CLI flags or copying giant YAML dictionaries, MaxText provides modular rule files located in `src/maxtext/configs/mesh_and_rule/`.

```text
Default Training:
  Uses inline `mesh_axes` and `logical_axis_rules` from base.yml

Custom Mesh and Rule (custom_mesh_and_rule: "ep-as-cp"):
  Pulls `src/maxtext/configs/mesh_and_rule/ep-as-cp.yml`
  ──→ Overrides mesh_axes
  ──→ Overrides logical_axis_rules wholesale
```

`custom_mesh_and_rule` selects an external named YAML configuration from `configs/mesh_and_rule/` to override the entire physical mesh and logical axis mapping table.

---

## 2. Fundamentals & Common Presets

MaxText ships with several specialized mesh and rule templates:
- **`"ep-as-cp"`**: Repurposes Expert Parallelism (EP) hardware axes to serve as Context Parallelism (CP) axes during evaluation/inference, allowing long context processing without reallocating physical TPU clusters.
- **Custom Research Meshes**: Allows teams to maintain version-controlled, architecture-specific sharding schemes (e.g., hybrid tensor-sequence or pipeline-parallel rules) in dedicated YAML files.

---

## 3. Options & Configuration

| Value | Meaning |
|---|---|
| `""` (default) | Uses default `mesh_axes` and `logical_axis_rules` defined in `base.yml`. |
| String (e.g. `"ep-as-cp"`) | Loads `src/maxtext/configs/mesh_and_rule/<name>.yml` to override mesh definitions. |

Default in `base.yml`:
```yaml
custom_mesh_and_rule: ""
```

---

## 4. Interactions with Related Parameters

```text
custom_mesh_and_rule: "ep-as-cp"
  ├── Replaces: mesh_axes
  ├── Replaces: logical_axis_rules
  └── Affects: context_sharding (can now be set to "expert")
```

- **`override_logical_axis_rules`**: If `custom_mesh_and_rule` is used, its loaded rules take precedence.
- **`custom_mesh_and_rule_for_eval`**: Used when you want the training phase to use one rule set, but evaluation/inference to switch to another (e.g. training on EP, evaluating with EP-as-CP).

---

## 5. Practical Use Cases

- **Serving MoE Models with Long Context**: MoE models trained with Expert Parallelism can reuse those same physical chips to scale context window length during evaluation or decode by setting `custom_mesh_and_rule: "ep-as-cp"`.

---

### One-line intuition

> **`custom_mesh_and_rule` loads a modular YAML preset from `configs/mesh_and_rule/` (like `ep-as-cp`) to completely replace the default physical mesh axes and logical axis sharding rules in one shot.**
