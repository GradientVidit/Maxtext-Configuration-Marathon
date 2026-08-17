
## 1. The core problem: destructive pruning

When `max_num_checkpoints_to_keep` is set, Orbax deletes old checkpoints. A hard delete on GCS is **immediate and irreversible**.

If you delete the wrong checkpoint — or if there's a bug in the pruning logic — it's gone. There's no recycle bin.

`checkpoint_todelete_subdir` is a safety net:

> Instead of hard-deleting, move old checkpoints to a subdirectory first.

---

## 2. What it does

```yaml
checkpoint_todelete_subdir: ".todelete"
```

When a checkpoint is due for pruning, instead of:

```text
delete gs://bucket/run/checkpoints/1000
```

MaxText moves it to:

```text
gs://bucket/run/checkpoints/.todelete/1000
```

The checkpoint still exists — just in a staging area. You can manually inspect or recover it before it's truly gone.

---

## 3. The critical caveat: ignored for `gs://`

From MaxText's own config comment:

```yaml
# Subdirectory to move checkpoints to before deletion.
# For example: ".todelete" (Ignored if directory is prefixed with gs://)
checkpoint_todelete_subdir: None
```

**If your `base_output_directory` is a `gs://` path, this parameter is ignored.** The underlying GCS delete is still used directly.

This is because GCS object "moves" are actually copy-then-delete operations, and Orbax's GCS handler may not implement the move-to-subdir pattern. It only applies to **local filesystem** checkpoints.

---

## 4. Difference from `checkpoint_todelete_full_path`

Both serve the same soft-delete purpose, but differ in scope:

```text
checkpoint_todelete_subdir
    → a subdirectory RELATIVE to the checkpoint directory
    → e.g., ".todelete" becomes checkpoints/.todelete/

checkpoint_todelete_full_path
    → an ABSOLUTE path anywhere on the filesystem
    → e.g., /mnt/scratch/ckpt_trash/
```

Use `checkpoint_todelete_subdir` when you want trash to live alongside your checkpoints. Use `checkpoint_todelete_full_path` when you want it on a separate disk/mount.

You can only use one at a time — they're alternatives.

---

## 5. Default

```yaml
checkpoint_todelete_subdir: None
```

`None` means no soft-delete staging — checkpoints are hard-deleted when pruned.

---

## 6. Practical use

This is primarily useful when:
- Working with **local storage** (not GCS)
- You want a manual review step before space is freed
- Your pruning logic or retention count might be misconfigured

For GCS-based training, skip this — use `checkpoint_todelete_full_path` if a GCS-compatible trash path is needed (and confirm your MaxText version supports it).

---

### One-line intuition

> **`checkpoint_todelete_subdir` moves pruned checkpoints to a subdirectory rather than hard-deleting them — a local-filesystem-only soft-delete safety net.**
