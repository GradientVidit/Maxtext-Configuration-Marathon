## 1. Why does it exist?

Pipeline Parallelism (PP) splits the model's sequential transformer layers across multiple stages. In intra-slice pipeline parallelism, stages reside on different groups of chips within the same physical TPU slice connected via ICI.

```text
TPU Pod Slice (e.g. 64 Chips):
  Stage 0 (Chips 0..15):   Layers 0..7   ──[ ICI Activation Pass ]──┐
  Stage 1 (Chips 16..31):  Layers 8..15  ◄──────────────────────────┘
  Stage 2 (Chips 32..47):  Layers 16..23 ──[ ICI Activation Pass ]──┐
  Stage 3 (Chips 48..63):  Layers 24..31 ◄──────────────────────────┘
```

`ici_pipeline_parallelism` configures the degree of pipeline stages within a single TPU slice.

---

## 2. Fundamentals & Options

| Value | Meaning |
|---|---|
| `1` (default) | Intra-slice pipeline parallelism is disabled; layers are distributed via FSDP/TP instead. |
| Integer $> 1$ | Allocates `N` pipeline stages within the slice. |

Default in `base.yml`:
```yaml
ici_pipeline_parallelism: 1
```

---

## 3. Interactions with Related Parameters

- **`num_layers_per_pipeline_stage`**: Number of decoder layers assigned to each pipeline stage.
- **`num_pipeline_microbatches`**: Must be a multiple of the total number of pipeline stages (`num_stages = ici_pipeline_parallelism * dcn_pipeline_parallelism`).
- **`pipeline_parallel_layers`**: Allows restricting pipelining to a specific subset of layers.

---

### One-line intuition

> **`ici_pipeline_parallelism` divides the layers of a model into sequential pipeline stages hosted on separate chip groups within the same TPU slice.**
