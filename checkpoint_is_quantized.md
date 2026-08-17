
## 1. Why does it exist?

There are two distinct kinds of parameter checkpoints in MaxText:

1. **Standard training checkpoints** (including QAT runs): weights remain in float32/bfloat16 even during quantized training. Orbax saves them normally.
2. **AQT exported quantized checkpoints**: produced by `save_quantized_params_path`. These store weights as int8/fp8 tensors with accompanying scale factors in AQT's own pytree structure.

When loading type 2, MaxText needs to know it's reading an AQT-exported format — not because the bits are ambiguous, but because the **pytree structure is different**. Without the flag, the loader tries to match AQT's scale-tensor keys against a standard parameter tree, causing shape/key mismatches.

`checkpoint_is_quantized` tells the loader: "use the AQT-aware loading path for this checkpoint."

---

## 2. The problem it solves

```text
Without checkpoint_is_quantized=true:

  AQT checkpoint (int8 tensors + scale tensors in AQT pytree)
        ↓
  loader expects standard parameter pytree (float32/bf16)
        ↓
  key/shape mismatch: scale tensor keys don't exist in standard tree
        ↓
  load fails or model initializes wrong

With checkpoint_is_quantized=true:

  AQT checkpoint (int8 tensors + scale tensors)
        ↓
  loader uses AQT-aware path: reads scale tensors + int8 weights
        ↓
  correctly reconstructs quantized parameter pytree
  → model trains/infers from correct state
```

> **Important**: A standard QAT training checkpoint (where you trained with `quantization: "int8"` but did NOT set `save_quantized_params_path`) does NOT need this flag — its weights remain in float32/bfloat16 in standard Orbax format.

---

## 3. Options

| Value | Behavior |
|---|---|
| `false` | Checkpoint loaded as standard full-precision checkpoint (default) |
| `true` | Checkpoint loaded as AQT quantized checkpoint |

Default in base.yml:
```yaml
checkpoint_is_quantized: false
```

---

## 4. When to set it

Set this to `true` when:
- Resuming a run that was saved with `quantization != ""`
- Loading a quantized checkpoint produced by `save_quantized_params_path`
- Doing inference from a quantized checkpoint

The flag must match the checkpoint format. If the checkpoint was saved quantized and you load it without this flag, the behavior is undefined (likely garbage outputs).

---

## 5. Relationship to `save_quantized_params_path`

These two are complementary:

```text
save_quantized_params_path:
  during a run → save quantized weights to a path
  → produces a quantized checkpoint

checkpoint_is_quantized:
  on a subsequent run → "the path I'm loading from has a quantized checkpoint"
  → load it correctly
```

A typical PTQ workflow:
```yaml
# Run 1: export quantized checkpoint
quantization: "int8"
save_quantized_params_path: "gs://bucket/model_int8/"

# Run 2: load and deploy quantized model
load_parameters_path: "gs://bucket/model_int8/"
checkpoint_is_quantized: true
quantization: "int8"
```

---

## 6. AQT vs Qwix distinction

`checkpoint_is_quantized=true` specifically signals an **AQT-format** quantized checkpoint. If you trained with `use_qwix_quantization=true` (Qwix backend), the checkpoint format may differ — verify with the specific backend's documentation whether this flag applies.

---

### One-line intuition

> **`checkpoint_is_quantized=true` tells the checkpoint loader the file was saved in AQT quantized format (int8 weights + scale tensors) rather than full-precision — without it, quantized checkpoints are loaded as garbage.**
