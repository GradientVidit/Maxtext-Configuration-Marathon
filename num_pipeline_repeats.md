
## 1. Why does this exist?

When the number of decoder layers exceeds `num_stages × num_layers_per_pipeline_stage`, the pipeline must **cycle through the stages multiple times** to process all layers. Each full cycle through all stages is one "repeat."

```text
num_decoder_layers = 64
num_stages = 4
num_layers_per_pipeline_stage = 4

Layers needed per repeat = 4 × 4 = 16
num_pipeline_repeats = 64 / 16 = 4

Repeat 0: stages 0–3 handle layers  0–15
Repeat 1: stages 0–3 handle layers 16–31
Repeat 2: stages 0–3 handle layers 32–47
Repeat 3: stages 0–3 handle layers 48–63
```

`num_pipeline_repeats` is that `4`. It's usually derived automatically — you set it explicitly only when you want to fix it and let `num_layers_per_pipeline_stage` be the derived variable.

---

## 2. Auto-derivation

Default is `-1`, meaning MaxText computes:

```text
num_pipeline_repeats = num_decoder_layers / (num_stages × num_layers_per_pipeline_stage)
```

This must come out to a positive integer; if it doesn't, the config is invalid.

---

## 3. Why repeats matter for efficiency

The pipeline bubble fraction is:

```text
bubble = (num_stages - 1) / (num_pipeline_repeats × num_pipeline_microbatches + num_stages - 1)
```

More repeats → bubble fraction shrinks → more of the pipeline time is useful compute. Repeats are the primary lever for amortizing the fixed pipeline startup/drain cost across more work.

```text
num_stages=4, num_pipeline_microbatches=4:

num_pipeline_repeats=1: bubble = 3 / (1×4 + 3) = 3/7 ≈ 43%
num_pipeline_repeats=4: bubble = 3 / (4×4 + 3) = 3/19 ≈ 16%
num_pipeline_repeats=8: bubble = 3 / (8×4 + 3) = 3/35 ≈ 9%
```

This is why models with more layers relative to the pipeline degree are more pipeline-efficient.

---

## 4. Options

| Value | Meaning |
|---|---|
| `-1` | Auto-derive from `num_decoder_layers / (num_stages × num_layers_per_pipeline_stage)` |
| `N > 0` | Explicitly set to N repeats; `num_layers_per_pipeline_stage` must be consistent |

---

## 5. Interaction with microbatches

`num_pipeline_repeats` and `num_pipeline_microbatches` are both bubble-reducers — they multiply together in the denominator of the bubble fraction. The key difference:

- **More repeats** = more decoder-layer weights stored per stage (circular PP keeps all repeat weights as a stacked tensor `[num_repeats, num_stages, ...]` — more repeats = larger stack)
- **More microbatches** = smaller per-microbatch size = potentially lower arithmetic intensity

They're complementary knobs. For a fixed model and hardware, you usually maximize repeats (constrained by total layer count), then tune microbatches for the residual bubble.

---

## 6. Interaction with `pipeline_fsdp_ag_once`

When `pipeline_fsdp_ag_once=true`, FSDP weights are gathered once per repeat (not per microbatch). More repeats = more all-gather operations total, but each is amortized over all microbatches within that repeat.

---

### One-line intuition

> **`num_pipeline_repeats` is the number of times all pipeline stages cycle through together — more repeats means more decoder layers per stage and a smaller pipeline bubble, since the fixed startup/drain cost is amortized over more work.**
