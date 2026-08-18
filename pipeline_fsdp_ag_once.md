
## 1. Why does this exist?

In pipeline parallelism with FSDP, each pipeline stage holds a **sharded** (not full) copy of its layers' weights. Before processing a microbatch, the stage must **all-gather** the full weights from other FSDP peers. By default, this happens **once per microbatch**:

```text
Microbatch 0: all-gather weights → compute → reduce-scatter gradients
Microbatch 1: all-gather weights → compute → reduce-scatter gradients
Microbatch 2: all-gather weights → compute → reduce-scatter gradients
Microbatch 3: all-gather weights → compute → reduce-scatter gradients
```

With 4 microbatches per repeat, you pay 4× the all-gather cost. `pipeline_fsdp_ag_once` collapses this to a single all-gather before the first microbatch:

```text
[all-gather all weights once]
Microbatch 0: compute
Microbatch 1: compute
Microbatch 2: compute
Microbatch 3: compute
[gradients reduce-scattered at optimizer step, outside the pipeline scan]
```

---

## 2. The memory cost

Gathering all weights once means the **full (un-sharded) weights must fit in HBM** for the duration of the repeat, plus the sharded gradients accumulate across microbatches. This is typically bf16 weights × (1/FSDP_degree) normally, vs. bf16 weights × full size with `ag_once=true`.

```text
Normal FSDP: HBM holds W/N at rest, W temporarily during each microbatch
ag_once:     HBM holds W permanently throughout the repeat
```

The tradeoff is: less communication, more persistent memory pressure.

---

## 3. The ZeRO analogy (and its limits)

MaxText's own comments call this pattern "similar in spirit to ZeRO-1". The analogy is loose: ZeRO-1 technically shards **optimizer state**, while ZeRO-2/3 shard parameters. What the comment means: like ZeRO-1's optimizer-state sharding philosophy — gather once, compute over many microbatches, scatter once — the per-microbatch gather overhead is eliminated. The actual mechanism is: `pipeline.py` calls `all_gather_over_fsdp(layers_params, ...)` once before the `scan_body` loop, so all microbatch iterations share the pre-gathered weights.

---

## 4. The alternative the comments suggest

MaxText's own comments note:

> An alternative to setting this to true may be to replace any FSDP with DP and use optimizer offloading if necessary.

Meaning: if you have enough device memory to eliminate FSDP sharding entirely, you avoid the all-gather overhead altogether. `ag_once` is a middle ground for when you need FSDP but the per-microbatch gather cost is visibly hurting throughput.

---

## 5. Options

| Value | Behavior |
|---|---|
| `false` | Default — all-gather weights once per microbatch |
| `true` | All-gather all weights once before the first microbatch in each repeat |

Default: `false`.

---

## 6. Interaction with `pipeline_fsdp_ag_per_repeat`

These are two different prefetching strategies and should not both be set to `true` simultaneously:

- `pipeline_fsdp_ag_once=true`: gather all weights before any microbatch computation
- `pipeline_fsdp_ag_per_repeat=true`: prefetch weight gathers ahead of computation per repeat, with more fine-grained pipelining of the prefetch itself

They represent different points on the memory/communication tradeoff curve.

---

### One-line intuition

> **`pipeline_fsdp_ag_once` eliminates per-microbatch FSDP all-gathers by gathering all weights once before a pipeline repeat — trading persistent HBM residency of the full weights for dramatically fewer collective operations.**
