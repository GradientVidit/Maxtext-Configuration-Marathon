
## 1. Why does this exist?

`pipeline_fsdp_ag_once` eliminates per-microbatch all-gathers by gathering everything upfront. `pipeline_fsdp_ag_per_repeat` takes a finer-grained approach: **prefetch** the weight gathers for the next iteration ahead of time, so the communication is overlapped with the current iteration's compute — rather than blocking on it.

The difference is timing:

```text
ag_once:        [gather ALL weights] → [mb0] → [mb1] → [mb2] → ...
ag_per_repeat:  [gather mb1 weights] overlapped with [mb0 compute]
                [gather mb2 weights] overlapped with [mb1 compute]
                ...
```

`ag_per_repeat` keeps memory lower than `ag_once` (only prefetching one microbatch's worth of weights at a time) while still hiding communication latency behind compute.

---

## 2. What "per repeat" means

In the circular pipeline, the decoder layers are organized as a stacked weight tensor with shape `[num_repeats, num_stages, layers_per_stage, ...]`. Each stage works on one repeat's worth of weights per pass through all microbatches. A "repeat" advances only after all `num_pipeline_microbatches` have passed through — from `pipeline.py`:

```python
repeat_ids = microbatches_processed // num_pipeline_microbatches
```

So "per repeat" prefetching means: while the current batch of microbatches is being computed using repeat K's weights, the hardware simultaneously all-gathers the weights for repeat K+1:

```text
Repeat K (all microbatches):   [compute with weights_K] + [prefetch weights_K+1]
Repeat K+1 (all microbatches): [compute with weights_K+1] + [prefetch weights_K+2]
```

This requires the prefetch to complete before repeat K+1 begins.

---

## 3. Memory comparison

```text
ag_once:        Full un-sharded weights in HBM for entire repeat duration
ag_per_repeat:  One microbatch's worth of prefetched weights in HBM extra
Default (none): Only sharded weights in HBM; full weights transiently during microbatch
```

`ag_per_repeat` is a middle ground: less persistent memory than `ag_once`, but slightly more than the bare default.

---

## 4. Options

| Value | Behavior |
|---|---|
| `false` | Default — no weight prefetching |
| `true` | Prefetch weight gathers ahead of per-repeat computation |

Default: `false`.

---

## 5. When to enable

Enable when:
- FSDP all-gather latency is measurably adding to per-step time
- `pipeline_fsdp_ag_once=false` (don't use both simultaneously)
- Per-repeat compute is long enough to hide the prefetch communication
- Memory pressure prevents using `ag_once`

---

## 6. Interaction with `pipeline_fsdp_ag_once`

These are mutually exclusive approaches to FSDP weight gathering in PP:

| Setting | Communication | Memory |
|---|---|---|
| Neither (default) | Per iteration, dynamic gather from stacked weights | Minimal |
| `ag_once` | Full gather once before scan; weights held pre-gathered | Highest |
| `ag_per_repeat` | Prefetched per repeat, overlapped with compute | Medium |

Don't set both to `true`.

---

### One-line intuition

> **`pipeline_fsdp_ag_per_repeat` prefetches FSDP weight all-gathers ahead of each pipeline repeat so communication overlaps with compute — a middle path between per-microbatch blocking gathers and the full upfront gather of `ag_once`.**
