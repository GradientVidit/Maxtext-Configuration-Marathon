
## 1. Why does it exist?

Within a W&B project, runs need a human-readable identity. Without an explicit name, W&B auto-assigns random names like `"lucky-wind-42"` or `"glowing-moon-17"`. These are fine for throwaway experiments but meaningless for production runs where you need to find a specific run months later.

`wandb_run_name` lets you give the W&B run a meaningful, searchable name.

---

## 2. Relationship to `run_name`

These are different things:

```text
run_name         → MaxText's internal experiment identity
                   Determines checkpoint namespace, GCS paths
                   e.g. "qwen3_bf16_pretrain_v1"

wandb_run_name   → The label this run gets inside W&B
                   Only affects how it appears in the W&B UI
                   e.g. "qwen3_bf16_pretrain_v1" (or anything else)
```

They *can* be the same string — in fact, making them match is a good practice for traceability. But they're stored separately and serve separate systems.

---

## 3. Options

| Value | Behavior |
|---|---|
| `""` | W&B auto-assigns a random name (default) |
| Any string | Used as the W&B run display name |

Default in base.yml:
```yaml
wandb_run_name: ""
```

---

## 4. When to set it

**Always set it for any run you care about.** Random names make auditing and comparison painful. A naming convention like:

```yaml
wandb_run_name: "qwen3_8b_bf16_lr2e-4_bs512"
```

immediately conveys model, precision, learning rate, and batch size in the W&B table view.

**Leave it empty when:** quickly prototyping and the run is ephemeral — you won't go back to look at it.

---

## 5. Dependency

Only meaningful when:
```yaml
enable_wandb: True
```

W&B run names don't exist as a concept when W&B is disabled.

---

## 6. Note: W&B run name vs W&B run ID

W&B distinguishes between:
- **Run name** (`wandb_run_name`): human-readable, can be duplicated across projects
- **Run ID**: auto-generated unique identifier, not controlled by this parameter

Both are visible in the W&B UI. The run name is what shows in the table; the run ID is what URLs use.

---

### One-line intuition

> **`wandb_run_name` sets the human-readable label for this run inside W&B — make it match `run_name` or encode the key hyperparameters so you can find it six months from now.**
