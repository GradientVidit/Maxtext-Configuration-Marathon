
## 1. Why does it exist?

Checkpointing involves expensive I/O:

```text
TPU
 ↓
capture model state
 ↓
serialize
 ↓
write to storage (e.g. GCS)
```

If this happens synchronously, the training loop waits:

```text
Training ──────┐
               ↓
          save checkpoint
               ↓
Training ──────┘
```

So checkpoint I/O creates a **pause in accelerator utilization**.

With asynchronous checkpointing:

```text
Training ───────────────────────────────→
             │
             └──→ checkpoint saving
                    ↓
                 CPU/background
                    ↓
                   GCS
```

The training loop can proceed while the previous checkpoint is being written. MaxText describes this as hiding checkpoint I/O latency and improving accelerator utilization. ([MaxText](https://maxtext.readthedocs.io/en/latest/reference/architecture/architecture_overview.html?utm_source=chatgpt.com "Architecture overview — MaxText documentation"))

---

## 2. What does `true` actually mean?

```yaml
async_checkpointing: true
```

The training step is blocked only for the minimum time needed to **capture the state**, after which the actual serialization/storage work happens asynchronously. ([MaxText](https://maxtext.readthedocs.io/en/latest/guides/checkpointing_solutions/emergency_checkpointing.html?utm_source=chatgpt.com "Emergency checkpointing — MaxText documentation"))

So it's **not**:

> "Don't wait at all."

There still has to be a point where MaxText captures a consistent snapshot of the model state.

The expensive storage operation is what gets overlapped.

---

## 3. `false`

```yaml
async_checkpointing: false
```

uses synchronous checkpointing:

```text
training
   ↓
checkpoint
   ↓
wait until save completes
   ↓
training continues
```

MaxText explicitly recommends trying synchronous checkpointing if you encounter problems with the asynchronous checkpointer. ([GitHub](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/maxtext/configs/base.yml?utm_source=chatgpt.com "maxtext/src/maxtext/configs/base.yml at main · AI-Hypercomputer/maxtext · GitHub"))

---

## 4. Why is this particularly important on TPU?

At scale, checkpoints can be huge.

For example:

```text
large model
   ↓
many GB/TB of state
   ↓
distributed checkpoint write
   ↓
GCS
```

If every checkpoint causes all TPU hosts to sit idle while storage finishes, you waste expensive accelerator time.

Async checkpointing lets:

```text
TPU → keep training
CPU → handle checkpoint serialization/I/O
```

which is why MaxText calls it a **critical performance optimization**. ([MaxText](https://maxtext.readthedocs.io/en/latest/reference/architecture/architecture_overview.html?utm_source=chatgpt.com "Architecture overview — MaxText documentation"))

---

## 5. Interaction with `enable_checkpointing`

These are different:

```yaml
enable_checkpointing: true
async_checkpointing: true
```

means:

> **Save checkpoints, and save them asynchronously.**

Whereas:

```yaml
enable_checkpointing: false
async_checkpointing: true
```

means:

> **Don't checkpoint at all.**

`async_checkpointing` only determines **how** enabled checkpointing is performed.

---

## 6. Options

|Value|Behavior|
|---|---|
|`true`|Asynchronous checkpoint saving; preferred for performance|
|`false`|Synchronous checkpoint saving; useful for debugging/checkpointer problems|

Current MaxText defaults to `true`. ([GitHub](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/maxtext/configs/base.yml?utm_source=chatgpt.com "maxtext/src/maxtext/configs/base.yml at main · AI-Hypercomputer/maxtext · GitHub"))

### One-line intuition

> **`async_checkpointing` lets MaxText overlap checkpoint I/O with training, so TPU computation doesn't sit idle waiting for the checkpoint to finish writing.**