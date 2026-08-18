
## 1. Why does this exist?

Pipeline parallelism (PP) partitions the model's decoder stack across devices by **assigning consecutive layer groups to each pipeline stage**. The most basic choice is how many layers land on each stage.

This parameter makes that choice explicit. Without it, PP would always put exactly 1 layer per stage — which maximizes the number of stages but also maximizes pipeline bubble overhead relative to useful compute per stage.

---

## 2. The core identity

```text
num_decoder_layers = num_stages × num_layers_per_pipeline_stage × num_pipeline_repeats
```

`num_layers_per_pipeline_stage` is the middle factor. Everything else flows from it once `num_stages` (set by `ici_pipeline_parallelism`) and `num_decoder_layers` are fixed.

---

## 3. How it maps onto hardware

```text
Device mesh: [stage=4]

num_decoder_layers = 32
num_layers_per_pipeline_stage = 4
num_pipeline_repeats = 32 / (4 × 4) = 2

Stage 0:  layers  0–3   (repeat 0), layers 16–19 (repeat 1)
Stage 1:  layers  4–7   (repeat 0), layers 20–23 (repeat 1)
Stage 2:  layers  8–11  (repeat 0), layers 24–27 (repeat 1)
Stage 3:  layers 12–15  (repeat 0), layers 28–31 (repeat 1)
```

Each stage runs its layers sequentially; the pipeline orchestrates passing activations between stages.

---

## 4. The compute-vs-bubble trade-off

```text
bubble fraction = (num_stages - 1) / (num_pipeline_repeats × num_pipeline_microbatches + num_stages - 1)
```

More layers per stage → fewer repeats → **larger bubble fraction** (bad for efficiency).
Fewer layers per stage → more repeats → **smaller bubble fraction** (better efficiency) but less compute per stage to overlap communication with.

The optimal point depends on the ratio of communication latency to per-stage compute time. More layers per stage only helps if the extra compute successfully hides the inter-stage communication.

```text
1 layer/stage:   max stages, smallest bubble (most repeats) — but least compute per stage,
                 making it harder for XLA to hide inter-stage communication behind compute
4 layers/stage:  fewer stages, more compute per activation pass (easier to hide comms),
                 but larger bubble fraction (fewer repeats to amortize startup/drain)
```

---

## 5. Options

Integer, default `1`.

| Value | Effect |
|---|---|
| `1` | One layer per stage — maximum stages, minimum per-stage compute |
| `N` | N layers sequentially processed per stage per pass |

Must satisfy: `num_decoder_layers % (num_stages × N) == 0`.

---

## 6. Interaction with scan settings

When `scan_layers_per_stage=true`, the N layers per stage are scanned (stacked) rather than unrolled. This is the inner scan loop. Default recommendation is `scan_layers_per_stage=false` (don't scan the inner loop), keeping the scan on the outer pipeline-iteration loop (`scan_pipeline_iterations=true`).

If N is large, scanning layers per stage becomes more attractive to save compile time, at the cost of potentially extra rematerialization.

---

## 7. Interaction with `inhomogeneous_layer_cycle_interval`

If the model has inhomogeneous layers:
```text
inhomogeneous_layer_cycle_interval = 4
num_layers_per_pipeline_stage should be a multiple of 4
```
Otherwise each stage contains a partial cycle and scan assumptions may break.

---

### One-line intuition

> **`num_layers_per_pipeline_stage` sets how many decoder layers each pipeline stage processes sequentially per pass — a direct trade-off between per-stage compute density (higher = more overlap potential) and pipeline bubble size (higher = larger bubble).**
