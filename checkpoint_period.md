
## 1. What it controls

`checkpoint_period` is the **step interval** between successive checkpoint saves.

```text
training step 0
training step 1
...
training step 10_000  → save checkpoint
...
training step 20_000  → save checkpoint
...
```

Default:

```yaml
checkpoint_period: 10_000
```

---

## 2. Why this number matters

Checkpointing costs real time and I/O:

```text
step N → capture state → serialize → write to GCS
```

Even with `async_checkpointing: true`, the state capture still briefly touches the training loop, and the background I/O consumes network bandwidth. Setting the period too low wastes these resources. Setting it too high means losing more work after a preemption.

The trade-off is:

```text
smaller period → less work lost per failure, more I/O cost
larger period  → less I/O cost, more work lost per failure
```

---

## 3. How to think about the right value

The right `checkpoint_period` is a function of:

| Factor | Effect |
|---|---|
| Preemption probability | Higher probability → smaller period is worth it |
| Checkpoint save time | If saving takes 10 min, you don't want to checkpoint every 1000 steps |
| Step duration | Saving every N wall-clock minutes matters more than every N steps |
| Storage cost | More frequent saves = more storage used (until `max_num_checkpoints_to_keep` prunes them) |

A rough heuristic for large-scale TPU training: target checkpoints every **10–30 minutes of wall-clock time**. The right step count to achieve that depends on your step duration, which varies enormously (a step on a v4-8 with a 7B model is very different from a step on a v5p-512 with the same model). So size your `checkpoint_period` in wall-clock terms first, then convert to steps based on measured step time.

---

## 4. The interaction with preemption

On preemptible TPUs or spot VMs, if the job dies at step 19,999 and `checkpoint_period: 10_000`, you lose nearly 10,000 steps of work. This is expensive.

Consider:

```yaml
checkpoint_period: 1_000  # for high-preemption environments
```

Or use `enable_autocheckpoint: true` alongside a regular period so that a checkpoint is also triggered on the preemption signal itself.

---

## 5. Interaction with `max_num_checkpoints_to_keep`

Smaller `checkpoint_period` generates more checkpoint directories. Without pruning:

```text
step  1000 → checkpoint/1000
step  2000 → checkpoint/2000
step  3000 → checkpoint/3000
...
```

This blows up storage. `max_num_checkpoints_to_keep` bounds how many are retained. Pair them:

```yaml
checkpoint_period: 1_000
max_num_checkpoints_to_keep: 5
```

---

## 6. Interaction with `enable_continuous_checkpointing`

When `enable_continuous_checkpointing: true`, checkpoints are generated continuously (as fast as async I/O allows) rather than at fixed `checkpoint_period` intervals. In that mode, `checkpoint_period` is effectively superseded.

---

## 7. Options

It's an integer — any positive integer is valid:

```yaml
checkpoint_period: 10_000   # default — reasonable for non-preemptible jobs
checkpoint_period: 1_000    # aggressive — good for preemptible spot VMs
checkpoint_period: 5_000    # moderate
checkpoint_period: 100_000  # rare — very large-scale runs with stable hardware
```

---

### One-line intuition

> **`checkpoint_period` sets how many training steps pass between checkpoint saves — balancing the cost of I/O against the amount of work you're willing to lose on a failure.**
