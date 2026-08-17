
## 1. The scan/unroll distinction in MaxText

MaxText's training uses `jax.lax.scan` to iterate over Transformer layers. This is a performance optimization:

```text
Without scan:
  layer_0(x) → layer_1(x) → layer_2(x) → ...
  (each layer is a separate XLA op — many small ops)

With scan:
  scan(layer_fn, x, stacked_params)
  (one op that loops — compiled as a single, optimized kernel)
```

For training, scan is significantly faster and more memory-efficient. Parameters are **stacked** across the layer dimension:

```text
Without scan:  layer_0.weight, layer_1.weight, layer_2.weight  (separate arrays)
With scan:     weights[layer_dim=0], weights[layer_dim=1], ...  (one stacked array)
```

---

## 2. The problem with scanned checkpoints for inference

When you save a training checkpoint, parameters are in **scanned (stacked)** format. Inference engines (JetStream, vLLM, HuggingFace Transformers) expect **unscanned (per-layer)** format.

So before deploying a MaxText-trained model, you typically run `generate_param_only_checkpoint.py` to:

1. Load the training checkpoint (scanned format)
2. Strip optimizer state (params only)
3. Convert from scanned → unscanned layout
4. Save as a params-only checkpoint ready for inference

---

## 3. What `force_unroll` controls

```yaml
force_unroll: false
```

This flag is specifically for the `generate_param_only_checkpoint` script. It controls whether to **unroll the scan loop** during conversion.

```yaml
force_unroll: true
```

Forces explicit unrolling: the stacked parameter arrays are separated into explicit per-layer arrays:

```text
weights[0] → layer_0.weight
weights[1] → layer_1.weight
...
```

```yaml
force_unroll: false
```

The scan structure is left as-is in the output checkpoint (or converted through a different code path that doesn't require explicit unrolling).

---

## 4. When you need it

Set `force_unroll: true` when:
- You're generating a checkpoint for inference with a framework that requires per-layer (unscanned) parameter layout
- You're converting a checkpoint to HuggingFace format (which is always unscanned)
- JetStream or another serving engine requires unscanned format

The MaxText docs for checkpoint conversion workflows (e.g., Kimi K2, Llama) explicitly pass `force_unroll=true` when preparing inference checkpoints.

---

## 5. `force_unroll` only applies to `generate_param_only_checkpoint`

During normal training, this flag has **no effect**. Training always uses the scanned layout internally — `force_unroll` is purely about how the output of the param-only checkpoint generator is structured.

```text
train.py:
  force_unroll → ignored

generate_param_only_checkpoint.py:
  force_unroll: false → keep scan structure
  force_unroll: true  → unroll into per-layer arrays
```

---

## 6. Options

| Value | Effect |
|---|---|
| `false` | Default — don't unroll; scanned layout preserved in output |
| `true` | Unroll scan loop; output has explicit per-layer parameter structure |

---

### One-line intuition

> **`force_unroll` controls whether `generate_param_only_checkpoint` converts the training checkpoint's stacked (scanned) layer parameters into separate per-layer arrays — required when the target inference framework expects an unscanned layout.**
