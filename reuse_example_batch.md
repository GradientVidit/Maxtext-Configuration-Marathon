
## 1. Why does it exist?

Profiling a training run has a problem: **data loading time contaminates step timing**.

```text
step N timing = data loading + preprocessing + TPU compute
```

When you want to benchmark pure TPU throughput (step time, MFU, TFLOPs), you don't want the data pipeline jitter in the measurement. `reuse_example_batch` removes the data loading component entirely by feeding the same batch over and over.

```text
reuse_example_batch = 0 (normal):
  step 1 → load batch 1 → forward+backward
  step 2 → load batch 2 → forward+backward
  step 3 → load batch 3 → forward+backward

reuse_example_batch = 1 (benchmark mode):
  step 1 → load batch 1 → forward+backward
  step 2 → [same batch 1] → forward+backward
  step 3 → [same batch 1] → forward+backward
```

---

## 2. What it actually does

On the first step, MaxText loads one real batch from the dataset. From step 2 onwards, the same in-memory batch is reused. The data loader is never called again.

This means:
- No dataset I/O after step 1
- No preprocessing overhead
- No input pipeline jitter
- XLA may also fuse/optimize more aggressively since the input is a compile-time constant shape with consistent data

---

## 3. Options

| Value | Behavior |
|---|---|
| `0` | Normal training — new batch each step (default) |
| `1` (or any truthy integer) | Benchmark mode — reuse the first batch every step |

Default in base.yml:
```yaml
reuse_example_batch: 0
```

---

## 4. When to use it

**Use:** When measuring raw TPU throughput — step time, MFU (Model FLOP Utilization), tokens/sec, TFLOPs — and you want to isolate accelerator performance from data pipeline performance.

**Don't use:** For any real training. The model will overfit to one batch instantly (gradient updates repeatedly on the same data), loss will plummet, and the training is meaningless.

```text
benchmark setup:
  reuse_example_batch: 1
  steps: 20  (just enough for XLA compilation + warm-up + stable timing)
  → measure: step time, MFU
```

---

## 5. Interaction with data loading

When `reuse_example_batch=1`, the tf.data / dataset pipeline effectively runs for one step only. This means dataset config (shuffle, repeat, prefetch) doesn't matter in benchmark mode — all of it is bypassed after step 1.

---

## 6. What breaks if wrong

Setting this to `1` in a real training run means the model trains exclusively on one batch forever. Loss will drop to near-zero quickly (extreme overfitting), the run is scientifically worthless, and you won't notice it from the loss curve alone until you realize training loss is suspiciously perfect while eval is garbage.

---

### One-line intuition

> **`reuse_example_batch=1` eliminates data loading from the timing loop by feeding the same batch every step — useful only for benchmarking raw TPU throughput, never for real training.**
