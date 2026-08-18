
## 1. The problem it solves

Standard SPMD pipeline parallelism requires:

```text
num_decoder_layers % (num_stages × num_layers_per_pipeline_stage) == 0
```

When this doesn't divide evenly — for example a 35-layer model with a 4-stage pipeline — you can't pipeline all 35 layers cleanly. You'd need to either pad with dummy layers or restructure the architecture.

`pipeline_parallel_layers` provides a third option: **pipeline only a subset of layers** and let the `stage` mesh axis act as plain data parallelism for the rest.

---

## 2. How it works

```text
num_decoder_layers = 35
ici_pipeline_parallelism = 4
pipeline_parallel_layers = 32   ← pipeline only first 32 layers

Layers 0–31:   pipelined across 4 stages (8 per stage)
Layers 32–34:  "stage" axis acts as data parallelism (replicated)
```

The non-pipelined tail layers still use the `stage` mesh axis, but for data-parallel replication rather than pipeline-sequential execution.

---

## 3. Options

| Value | Meaning |
|---|---|
| `-1` | Default — pipeline all `num_decoder_layers` layers |
| `N` | Pipeline only the first N layers; remainder uses stage axis as DP |

Must satisfy: `N % (num_stages × num_layers_per_pipeline_stage) == 0`.

---

## 4. When to use this

- Model with an awkward layer count that doesn't cleanly divide by your PP degree
- Architectures where early layers benefit from pipelining but late layers are cheap enough to replicate
- Debugging: start by pipelining just a few layers to validate PP setup before scaling to all layers

---

## 5. What doesn't change

Embedding layers, final norm, and logits are not decoder layers — they're never pipelined regardless of this setting. This parameter only concerns the decoder stack.

---

## 6. Interaction with the core identity

The `num_decoder_layers = num_stages × num_layers_per_pipeline_stage × num_pipeline_repeats` identity applies **only to the pipelined portion**:

```text
pipeline_parallel_layers = num_stages × num_layers_per_pipeline_stage × num_pipeline_repeats
```

The non-pipelined layers (decoder layers N through total) sit outside this identity.

---

### One-line intuition

> **`pipeline_parallel_layers` lets you pipeline a divisible prefix of the decoder stack and fall back to data-parallel replication for any remaining layers that don't fit cleanly into the pipeline — the escape hatch when layer count isn't divisible by PP degree.**
