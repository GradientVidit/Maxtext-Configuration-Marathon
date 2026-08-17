
## 1. The problem: foreign checkpoint formats

MaxText's internal checkpoint format is Orbax. But you may want to load a checkpoint from a completely different source:

- A checkpoint saved as **safetensors** (common HuggingFace format)
- A proprietary format from another training framework
- A checkpoint with a different key structure than MaxText expects

Simply pointing `load_parameters_path` at a safetensors file won't work — MaxText would try to deserialize it as an Orbax checkpoint and fail (or silently produce garbage).

---

## 2. What `checkpoint_conversion_fn` does

```yaml
checkpoint_conversion_fn: none
```

When set to a function reference (not `none`), MaxText will:

1. Load the raw checkpoint data from the specified path
2. Pass it through the conversion function
3. Use the resulting dict as if it were a native MaxText checkpoint

```text
raw checkpoint data (e.g., safetensors)
           ↓
  checkpoint_conversion_fn
           ↓
  MaxText-compatible parameter dict
           ↓
  model initialization
```

The function acts as a translation layer between the external format and MaxText's internal structure.

---

## 3. The `source_checkpoint_layout` connection

These two parameters work together:

```yaml
source_checkpoint_layout: "safetensors"
checkpoint_conversion_fn: <some_conversion_function>
```

`source_checkpoint_layout` tells MaxText **how to read** the raw checkpoint bytes. `checkpoint_conversion_fn` then tells it **how to map** the loaded dict to MaxText's expected structure.

From MaxText's config:
```yaml
# optional checkpoint context to use for loading.
# options: "orbax", "safetensors"
source_checkpoint_layout: "orbax"

# function for processing loaded checkpoint dict into a format
# maxtext can understand. (for other formats, i.e. safetensors)
checkpoint_conversion_fn: none
```

If `source_checkpoint_layout: "orbax"`, the checkpoint is read natively and no conversion function is needed. If `source_checkpoint_layout: "safetensors"`, the raw tensors are loaded and then `checkpoint_conversion_fn` translates key names / tensor layouts.

---

## 4. What the conversion function actually does

A typical conversion function for a HuggingFace → MaxText workflow:
- Renames parameter keys (HF uses `model.layers.0.self_attn.q_proj.weight`; MaxText uses its own naming)
- Transposes or reshapes tensors where layouts differ (e.g., column-major vs row-major weight matrices)
- Handles tied embeddings, bias terms that MaxText may not have, etc.

MaxText's checkpoint conversion utilities (`src/maxtext/checkpoint_conversion/`) contain model-specific implementations.

---

## 5. When to use it

- Loading a safetensors model checkpoint directly into MaxText without running the full conversion pipeline
- Custom checkpoint formats from non-MaxText training runs
- Testing or debugging with externally-sourced weights

For most production use cases (pretrained → fine-tune via MaxText), the recommended path is to run MaxText's standalone conversion scripts first and produce a proper Orbax checkpoint, then use `load_parameters_path`. That way `checkpoint_conversion_fn` stays `none`.

---

## 6. Options

```yaml
checkpoint_conversion_fn: none             # default — no conversion, native Orbax
checkpoint_conversion_fn: maxtext.checkpoint_conversion.utils.safetensors_to_maxtext
```

The value is a **dotted Python module path** to a callable in MaxText's codebase (not arbitrary user code). The function is called with the loaded checkpoint dict and returns a MaxText-compatible parameter pytree.

A conversion function has a signature roughly like:

```python
def my_conversion_fn(checkpoint_dict: dict) -> dict:
    # remap keys, reshape tensors
    return maxtext_compatible_dict
```

---

### One-line intuition

> **`checkpoint_conversion_fn` is a plugin hook that translates a foreign checkpoint format (e.g., safetensors) into MaxText's internal parameter dict — needed only when loading from non-Orbax sources.**
