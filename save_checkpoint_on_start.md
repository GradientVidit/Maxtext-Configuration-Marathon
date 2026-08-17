
## 1. Why does it exist?

Normally, checkpointing happens periodically:

```text
start
  │
  ↓
training ───────────→ checkpoint ─────→ checkpoint ─────→ ...
                         ↑
                  checkpoint_period
```

With:

```yaml
save_checkpoint_on_start: true
```

you additionally get:

```text
start
  │
  ↓
checkpoint  ← immediately
  │
  ↓
training ───────────→ checkpoint ─────→ ...
```

This gives you a checkpoint representing the **initial state of the run**.

---

## 2. What is the initial checkpoint useful for?

Mainly **reproducibility / recovery / debugging**.

For example, you're about to run a long experiment:

```text
initial model
    ↓
checkpoint 0  ← saved immediately
    ↓
100k training steps
```

If something goes wrong later, you have a known checkpoint representing the state **before training changed the model**.

It's particularly useful when you want a guaranteed starting checkpoint even if `checkpoint_period` is large.

---

## 3. `true` vs `false`

|Value|Behavior|
|---|---|
|`false`|Don't create a checkpoint specifically at startup|
|`true`|Save a checkpoint immediately at startup|

Default:

```yaml
save_checkpoint_on_start: false
```

---

## 4. Relationship with `enable_checkpointing`

This is important:

```text
enable_checkpointing
        ↓
    master switch
```

If checkpointing is disabled:

```yaml
enable_checkpointing: false
```

then `save_checkpoint_on_start` isn't useful.

The intended combination is:

```yaml
enable_checkpointing: true
save_checkpoint_on_start: true
```

which means:

> **Checkpoint normally, and also create one immediately at the beginning.**

---

## 5. Don't confuse it with `save_checkpoint_on_completion`

They are mirror concepts:

```text
save_checkpoint_on_start
        ↓
      START
        ↓
   save checkpoint


save_checkpoint_on_completion
        ↓
    COMPLETION
        ↓
   save checkpoint
```

Neither controls periodic checkpointing; that's handled by `checkpoint_period`.

---

### One-line intuition

> **`save_checkpoint_on_start=true` creates an initial checkpoint immediately when training begins, giving you a saved snapshot of the pre-training state.**