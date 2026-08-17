
## 1. Why does it exist?

W&B organizes runs into **projects**. Without a project assignment, runs end up in a default bucket that becomes unnavigable once you accumulate dozens of experiments. `wandb_project_name` puts runs where you want them in the W&B hierarchy.

```text
W&B workspace
  └── project: "maxtext"       ← wandb_project_name
        ├── run: "qwen3_bf16"
        ├── run: "qwen3_int8"
        └── run: "llama3_8b"
```

---

## 2. What it controls

When `enable_wandb=True`, MaxText calls `wandb.init(project=wandb_project_name, ...)`. The project determines:
- Where in the W&B UI the run appears
- Which runs are grouped together for comparison
- Access control (W&B project-level permissions)

If the project doesn't exist yet in your W&B workspace, W&B creates it automatically on the first run.

---

## 3. Options

| Value | Behavior |
|---|---|
| `"maxtext"` | Default — all runs land in a project named "maxtext" |
| Any string | The W&B project name to use |

Default in base.yml:
```yaml
wandb_project_name: "maxtext"
```

---

## 4. When to change it

The default `"maxtext"` is reasonable for personal use but becomes cluttered when you have multiple distinct research threads. Use different project names to separate:

```text
wandb_project_name: "maxtext-pretraining"    # large-scale pretrain runs
wandb_project_name: "maxtext-finetune"        # fine-tuning experiments
wandb_project_name: "maxtext-quant-ablation"  # quantization ablations
```

---

## 5. Dependency

This parameter only has effect when:
```yaml
enable_wandb: True
```

If `enable_wandb=False`, `wandb_project_name` is completely ignored.

---

## 6. Interaction with `wandb_run_name`

```text
wandb_project_name → WHERE the run is stored in W&B
wandb_run_name     → WHAT the run is called within that project
```

These are independent: same project, different names for different experiments.

---

### One-line intuition

> **`wandb_project_name` is the W&B project bucket your run lands in — change it from the default `"maxtext"` to keep distinct research threads cleanly separated in the W&B UI.**
