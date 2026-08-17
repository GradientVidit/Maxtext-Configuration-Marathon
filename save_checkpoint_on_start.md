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

## 2a. When used with `load_parameters_path` (fine-tuning)

The step-0 checkpoint captures:
- The **pretrained weights** (as loaded via `load_parameters_path`)
- A **freshly initialized optimizer** (no Adam moments yet — those start accumulating from step 0 onward)
- Step counter = 0

This is valuable because it creates a MaxText-native Orbax checkpoint of the starting weights. Future runs can use this checkpoint directly as `load_parameters_path`, bypassing the original HF→Orbax conversion pipeline.

```yaml
# first run: convert + cache via step-0 checkpoint
load_parameters_path: gs://bucket/hf_converted/orbax_ckpt
save_checkpoint_on_start: true
run_name: finetune_v1

# future run: load the already-MaxText-format step-0 checkpoint
load_parameters_path: gs://bucket/finetune_v1/checkpoints/0
```

---

## 3. `true` vs `false`

|Value|Behavior|
|---|---|
|`false`|Don't create a checkpoint specifically at startup|
|`true`|Save a checkpoint immediately at startup|

Default:

```yaml
save_checkpoint_on_start: true
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

> **`save_checkpoint_on_start: true` (the default) saves a checkpoint before the first training step — capturing either random init or loaded pretrained weights — giving you an unconditional baseline to roll back to.**