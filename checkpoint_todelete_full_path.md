
## 1. Purpose

The twin of `checkpoint_todelete_subdir`. Both implement the same soft-delete idea — when a checkpoint is pruned, don't hard-delete it; move it somewhere first.

The difference is **path specification style**:

```text
checkpoint_todelete_subdir
    → relative path, becomes a subdirectory of the checkpoint dir

checkpoint_todelete_full_path
    → absolute path, can be anywhere on the accessible filesystem
```

---

## 2. When to use full path vs subdir

```yaml
# Subdir: trash lives next to your checkpoints
checkpoint_todelete_subdir: ".todelete"
# result: /path/to/checkpoints/.todelete/

# Full path: trash lives on a separate volume
checkpoint_todelete_full_path: "/mnt/scratch/checkpoint_trash"
```

Use `checkpoint_todelete_full_path` when:
- The trash needs to live on a different disk/mount (e.g., cheap cold storage)
- You're sharing a single trash directory across multiple runs
- You want explicit control over the trash location

---

## 3. Mutual exclusivity

These two parameters are alternatives. Set at most one. If both are set, behavior is undefined (check your MaxText version's handling).

---

## 4. GCS caveat (same as `checkpoint_todelete_subdir`)

The soft-delete via move is only meaningful on local filesystems. For GCS-based checkpoints (`gs://...`), this flag may also be ignored depending on how MaxText's Orbax integration implements the deletion.

If you need a GCS "trash" pattern, a possible approach is using GCS Object Lifecycle rules to delay actual deletion — that's outside MaxText's config and handled at the GCS bucket level.

---

## 5. Default

```yaml
checkpoint_todelete_full_path: None
```

`None` = disabled, hard-delete when pruning.

---

## 6. Operational pattern

```text
1. Set max_num_checkpoints_to_keep: 3
2. Set checkpoint_todelete_full_path: /mnt/scratch/trash
3. Old checkpoints accumulate in /mnt/scratch/trash
4. Periodically, manually review and rm -rf /mnt/scratch/trash/*
```

This gives you a manual review window before space is permanently freed.

---

### One-line intuition

> **`checkpoint_todelete_full_path` is the absolute-path version of `checkpoint_todelete_subdir` — it specifies where to move pruned checkpoints instead of deleting them immediately.**
