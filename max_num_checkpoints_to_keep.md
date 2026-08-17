
## 1. What it controls

MaxText saves a checkpoint every `checkpoint_period` steps. Without a retention limit, these pile up indefinitely. `max_num_checkpoints_to_keep` caps how many checkpoint directories are retained on disk.

```text
checkpoint/1000   ← oldest
checkpoint/2000
checkpoint/3000
checkpoint/4000
checkpoint/5000   ← newest
```

With `max_num_checkpoints_to_keep: 3`, after saving step 5000, steps 1000 and 2000 are pruned:

```text
checkpoint/3000
checkpoint/4000
checkpoint/5000
```

Default:

```yaml
max_num_checkpoints_to_keep: None  # keep all checkpoints
```

---

## 2. Why the default is `None`

`None` means **keep every checkpoint forever**. This is safe but expensive in storage. The reasoning for a permissive default is that deleting checkpoints is irreversible — MaxText would rather waste storage than silently discard a checkpoint you needed.

---

## 3. Storage math

Each checkpoint of a model is roughly proportional to the model parameter count × bytes-per-param × 2 (params + optimizer state for full checkpoints).

For a 7B parameter model in bfloat16:
```text
7B × 2 bytes = 14 GB for params alone
optimizer state ≈ 2× params for Adam = ~28 GB additional
→ ~42 GB per full checkpoint
```

With `checkpoint_period: 1_000` and no pruning, you could accumulate hundreds of GB quickly.

Setting:
```yaml
max_num_checkpoints_to_keep: 3
```
caps your checkpoint storage footprint at `3 × checkpoint_size`.

---

## 4. What "prune" means in practice

When a checkpoint exceeds the keep limit, MaxText (via Orbax) **deletes** the oldest checkpoint. But the exact behavior depends on `checkpoint_todelete_subdir` and `checkpoint_todelete_full_path`:

- If those are set → old checkpoints are **moved** to the trash directory instead of hard-deleted.
- If they are `None` (default) and on GCS (`gs://`) → hard-deleted immediately.
- If they are `None` and on local disk → hard-deleted immediately.

---

## 5. The recovery window concept

Think of `max_num_checkpoints_to_keep` as defining a **recovery window**:

```text
keep = 3, period = 10_000 steps
→ recovery window = 30_000 steps of history
```

If your job fails at step 57,000, and the last three checkpoints are at 50,000, 40,000, 30,000, you can resume from 50,000 at worst.

---

## 6. When `None` is appropriate

- Short runs where total storage cost is bounded anyway.
- Runs where you want to be able to roll back to any earlier step (debugging, ablation analysis).
- Any time you explicitly want to preserve full training history.

---

## 7. Practical recommendations

| Scenario | Setting |
|---|---|
| Long multi-week run, storage is constrained | `2` or `3` |
| Research with rollback needs | `5` or `10` |
| Short experiment, keep everything | `None` |
| You have `checkpoint_todelete_subdir` as a safety net | Can use `2` safely |

---

## 8. Options

```yaml
max_num_checkpoints_to_keep: None    # default — keep all
max_num_checkpoints_to_keep: 1       # only the latest — risky
max_num_checkpoints_to_keep: 3       # common for production runs
max_num_checkpoints_to_keep: 5       # more headroom
```

Setting to `1` is risky: if the system crashes mid-checkpoint-write, you have no fallback.

---

### One-line intuition

> **`max_num_checkpoints_to_keep` is a storage retention policy — it tells Orbax how many checkpoint snapshots to keep alive, deleting older ones once the cap is exceeded.**
