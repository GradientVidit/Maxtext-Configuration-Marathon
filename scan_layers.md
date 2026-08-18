
## 1. The two ways to iterate over decoder layers

When MaxText processes N decoder layers during the forward pass, there are two structural options:

**Unrolled (unstacked)**: each layer's weights are separate named arrays, each layer op is a separate HLO operation:

```text
layer_0_params, layer_1_params, ..., layer_N_params  ← separate arrays
layer_0_forward(layer_0_params, x)
layer_1_forward(layer_1_params, ...)
...
```

**Scanned (stacked)**: all layers' weights are stacked into a single array along a "layer" axis, and `jax.lax.scan` iterates over that axis:

```text
stacked_params[0, :], stacked_params[1, :], ..., stacked_params[N-1, :]
jax.lax.scan(layer_forward, stacked_params, x)
```

`scan_layers=true` selects the scanned structure.

---

## 2. Why scanning matters

The scanned structure has several consequences:

| Property | Unrolled | Scanned |
|---|---|---|
| HLO graph size | O(N) separate ops | O(1) scan op |
| Compile time | Grows with N | Near-constant |
| Checkpoint format | N separate weight arrays | 1 stacked array with layer axis |
| Remat policy | Per-layer manual | Uniform across all layers via scan |
| XLA optimization freedom | Per-layer independent | Uniform treatment |

For models with many layers (e.g. 80+ layer Llama2-65B), unrolled compilation is impractically slow. Scanning is the only viable option at scale.

---

## 3. The checkpoint implication

The most important consequence for day-to-day operation: **the checkpoint format is different** between scanned and unrolled runs.

- `scan_layers=true` → checkpoint stores `params/decoder/layers/` as a single stacked tensor with shape `[num_layers, ...]`
- `scan_layers=false` → checkpoint stores `params/decoder/layers_0/`, `params/decoder/layers_1/`, ... as separate tensors

**MaxText handles this automatically when resuming**: the `scan_layers` value is auto-detected from the checkpoint's own metadata, not read from the config. This prevents accidentally loading a stacked checkpoint with `scan_layers=false` or vice versa.

---

## 4. Default

```yaml
scan_layers: true  # default
```

Recommended for virtually all training runs. The only reason to use `scan_layers=false`:
- Very small models (< 8 layers) where unrolled HLO is manageable
- Converting/inspecting checkpoints with tools that don't understand the stacked format
- Specific XLA optimization scenarios where per-layer fusion differs

---

## 5. Interaction with pipeline parallelism

When pipeline parallelism is active, the recommended configuration is:

```yaml
scan_layers: false               # unroll layers per stage
scan_pipeline_iterations: true   # scan the PP microbatch loop (outer loop)
```

In MaxText's pipelined execution model (`pipeline.py`), the fundamental scan loop is over microbatches (`scan_pipeline_iterations`). Layer indexing within a stage is integrated into the circular pipeline state management (`stages_output = stages_output[0] if scan_layers`). Scanning layers inside each stage on top of scanning pipeline microbatches creates nested scan structures that complicate trace levels and rematerialization policy attachment.

Therefore, for pipeline parallelism, set `scan_layers: false` and rely on `scan_pipeline_iterations: true`.

---

## 6. Options

| Value | Behavior |
|---|---|
| `true` | Default — stack layers, use `jax.lax.scan` over the layer dimension |
| `false` | Unroll layers — separate named weight arrays, separate HLO ops |

---

## 7. `param_scan_axis`

When `scan_layers=true`, `param_scan_axis` specifies which axis of the stacked weight tensor is the layer axis (default `1`). This affects how parameters are indexed during the scan and how checkpoints are laid out.

---

### One-line intuition

> **`scan_layers` determines whether decoder layers are implemented as a single `jax.lax.scan` over a stacked parameter array (true, default) or as N separate unrolled operations (false) — scan is necessary for fast compilation at scale, but should be disabled in favor of scanning PP iterations when using pipeline parallelism.**
