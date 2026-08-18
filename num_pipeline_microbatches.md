
## 1. The pipeline bubble problem

A pipeline with `num_stages` stages has an unavoidable **startup bubble** and **drain bubble**:

```text
Time →

Stage 0: [work][work][work][work][ idle ]
Stage 1: [idle][work][work][work][work  ]
Stage 2: [idle][idle][work][work][work  ]
Stage 3: [idle][idle][idle][work][work  ]
         ^startup^                ^drain^
```

The bubble is `(num_stages - 1)` pipeline iterations of wasted time on each end. Microbatching is how you fill that dead space with useful work.

---

## 2. How microbatches fill the bubble

Split the global batch into microbatches. Each stage processes one microbatch at a time and immediately passes activations forward — while stage N is still processing microbatch K, stage N+1 can start on microbatch K+1:

```text
                Microbatch
Time → MB0  MB1  MB2  MB3

Stage 0: [0 ][1 ][2 ][3 ]
Stage 1:    [0 ][1 ][2 ][3 ]
Stage 2:       [0 ][1 ][2 ][3 ]
Stage 3:          [0 ][1 ][2 ][3 ]
```

Now the pipeline is almost always doing useful work. The bubble fraction:

```text
bubble = (num_stages - 1) / (num_pipeline_repeats × num_pipeline_microbatches + num_stages - 1)
```

More microbatches → smaller bubble.

---

## 3. The microbatch size trade-off

```text
microbatch_size = global_batch_size / num_pipeline_microbatches
```

More microbatches = smaller microbatch size = less arithmetic intensity per stage per pass. Very small microbatches can hurt roofline efficiency because the per-element work diminishes while per-call overhead stays constant (kernel launch, all-reduce setup, etc.).

The sweet spot balances bubble reduction against per-step compute efficiency.

---

## 4. Default and constraints

Default: `-1` → set equal to `num_stages`.

```text
num_stages = 4 → num_pipeline_microbatches = 4 (default)
```

This is the minimum needed for the pipeline to be fully occupied (one microbatch per stage). Must be a multiple of `num_stages`.

When `pipeline_delay_activation_forwarding=true`, the minimum is raised to `2 × num_stages` (and is set to that by default with the delay enabled).

---

## 5. Options

| Value | Meaning |
|---|---|
| `-1` | Auto-set to `num_stages` (minimum for full pipeline occupancy) |
| `N` | Use N microbatches; must be a positive multiple of `num_stages` |

---

## 6. Interaction with memory

Each stage must hold activations for all in-flight microbatches simultaneously (up to `num_stages` in the GPipe schedule). With circular scheduling, memory pressure is bounded differently. In MaxText's implementation, peak activation memory scales with the number of microbatches in flight at once.

---

## 7. Interaction with FSDP weight gathering

With `pipeline_fsdp_ag_once=false` (default): FSDP weights are all-gathered once per microbatch. More microbatches = more all-gather operations total.

With `pipeline_fsdp_ag_once=true`: weights all-gathered once per repeat, regardless of microbatch count — so increasing microbatches only costs activation memory, not communication.

---

### One-line intuition

> **`num_pipeline_microbatches` splits the global batch so stages can work in parallel, shrinking the idle bubble — but each additional microbatch also shrinks per-stage compute size, so there's a crossover point where more microbatches starts hurting rather than helping.**
