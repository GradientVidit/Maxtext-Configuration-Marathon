
## 1. Why does it exist?

A training run has more state than just model weights:

```text
Full training state
├── Model parameters
├── Optimizer state
├── Training step
└── Other training state
```

`load_full_state_path` restores that **full state**.

So if training stopped at:

```text
step 50,000
```

you can restore the checkpoint and continue approximately as:

```text
50,000 → 50,001 → 50,002 → ...
```

rather than starting a new optimization process from the same weights.

---

## 2. `load_full_state_path` vs `load_parameters_path`

This is the key distinction:

|Parameter|Loads|Typical use|
|---|---|---|
|`load_parameters_path`|**Model weights only**|Fine-tuning / initializing from pretrained model|
|`load_full_state_path`|**Entire training state**|Resume training|

Conceptually:

```text
load_parameters_path
        ↓
   weights
        ↓
new optimizer
new training step
```

versus:

```text
load_full_state_path
        ↓
weights
+ optimizer state
+ step
+ other state
        ↓
continue training
```

---

## 3. Why is optimizer state important?

Consider Adam.

The optimizer maintains state such as its momentum/second-moment estimates:

```text
weights
optimizer
 ├── m
 └── v
```

If you only restore weights:

```text
checkpoint
   ↓
weights ✓
m       ✗
v       ✗
```

you're **not truly continuing the previous optimization trajectory**.

With full-state restoration:

```text
checkpoint
   ↓
weights ✓
m       ✓
v       ✓
step    ✓
```

you can genuinely resume the training run.

---

## 4. What does the path point to?

It points to the **full MaxText/Orbax checkpoint**.

For example, conceptually:

```yaml
load_full_state_path: gs://my-bucket/run/checkpoints/50000
```

The exact directory layout depends on the checkpoint format/version.

This is different from pointing specifically at the `items` directory used when loading only parameters.

---

## 5. Important interaction with `run_name`

MaxText has automatic checkpoint resumption based on `run_name`.

So there are two ways to resume:

### Automatic

```yaml
run_name: my_training_run
```

MaxText finds the checkpoint associated with that run.

### Explicit

```yaml
load_full_state_path: gs://bucket/checkpoint/50000
```

You explicitly tell MaxText which full checkpoint to restore.

So:

```text
run_name
    → "resume my existing run"

load_full_state_path
    → "resume from this exact full checkpoint"
```

---

## 6. Can you use both `load_parameters_path` and `load_full_state_path`?

No. MaxText documents them as **mutually exclusive**.

Choose based on your intent:

```text
"I want the weights"
        ↓
load_parameters_path


"I want to continue the training state"
        ↓
load_full_state_path
```

---

## One-line intuition

> **`load_full_state_path` restores an entire MaxText training checkpoint—weights plus optimizer/training state—so you can continue training rather than merely initialize from existing weights.**

The three checkpoint concepts you've now seen:

```text
load_parameters_path
    → initialize from weights

load_full_state_path
    → resume from complete training state

run_name
    → identify a run and enable automatic checkpoint resumption
```