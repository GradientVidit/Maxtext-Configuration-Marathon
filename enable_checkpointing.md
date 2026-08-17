
## 1. Why does it exist?

During training, MaxText periodically saves the training state:

```text
Training
   │
   ├── step 1
   ├── step 2
   ├── ...
   ├── step 10000
   │      ↓
   │   checkpoint
   └── ...
```

If the job is interrupted, that checkpoint can be used to resume instead of losing all progress.

So:

```yaml
enable_checkpointing: true
```

means:

> **Save checkpoints during the run.**

---

## 2. What exactly gets saved?

The normal training checkpoints are **full training-state checkpoints**, containing what is needed to resume training. MaxText describes these as "Stacked Training Checkpoints that contain the full state needed to resume." ([MaxText](https://maxtext.readthedocs.io/en/latest/reference/core_concepts/checkpoints.html?utm_source=chatgpt.com "Checkpoints — MaxText documentation"))

That's why this parameter is connected to:

```text
enable_checkpointing
        ↓
periodically save full state
        ↓
run_name / checkpoint directory
        ↓
future resume
```

This is different from `load_parameters_path`, which is about **reading** parameters from an existing checkpoint.

---

## 3. `true` vs `false`

### `true`

```yaml
enable_checkpointing: true
```

Checkpoint saving is enabled.

How frequently?

```yaml
checkpoint_period: 10_000
```

So, by default, MaxText saves periodically according to `checkpoint_period`. ([GitHub](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/maxtext/configs/base.yml?utm_source=chatgpt.com "maxtext/src/maxtext/configs/base.yml at main · AI-Hypercomputer/maxtext · GitHub"))

### `false`

```yaml
enable_checkpointing: false
```

Don't save the normal training checkpoints.

This is useful for short experiments, compilation/performance tests, or runs where you don't need recovery.

For example, MaxText's Kimi pretraining correctness/performance example explicitly uses:

```text
enable_checkpointing=false
```

for a 5-step test run. ([GitHub](https://github.com/AI-Hypercomputer/maxtext/blob/main/tests/end_to_end/tpu/kimi/Run_Kimi.md?utm_source=chatgpt.com "maxtext/tests/end_to_end/tpu/kimi/Run_Kimi.md at main · AI-Hypercomputer/maxtext · GitHub"))

---

## 4. Important: it doesn't control loading

Don't confuse:

```text
enable_checkpointing
    → SHOULD I SAVE checkpoints?

load_parameters_path
    → WHERE should I LOAD parameters from?

load_full_state_path
    → WHERE should I LOAD full training state from?
```

`enable_checkpointing=false` does **not** mean "don't load a checkpoint." Loading and saving are separate concerns.

---

## 5. Interaction with `async_checkpointing`

These two work together:

```yaml
enable_checkpointing: true
async_checkpointing: true
```

means:

```text
training
   │
   ├──────────────→ continue training
   │
   └→ checkpoint saving happens asynchronously
```

If:

```yaml
async_checkpointing: false
```

MaxText uses synchronous checkpointing instead. The current config explicitly states this behavior. ([GitHub](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/maxtext/configs/base.yml?utm_source=chatgpt.com "maxtext/src/maxtext/configs/base.yml at main · AI-Hypercomputer/maxtext · GitHub"))

So:

> `enable_checkpointing` = **whether to checkpoint**  
> `async_checkpointing` = **how checkpoint saving happens**

---

## One-line intuition

> **`enable_checkpointing` is the master on/off switch for MaxText's periodic checkpoint saving; when enabled, the training state is saved so the run can recover from interruptions.** ([MaxText](https://maxtext.readthedocs.io/en/latest/reference/core_concepts/checkpoints.html?utm_source=chatgpt.com "Checkpoints — MaxText documentation"))

The next parameters (`save_checkpoint_on_completion`, `async_checkpointing`, `checkpoint_period`) are the interesting ones because they define **when and how** those checkpoints are saved.