
## 1. What it declares

When MaxText loads a checkpoint via `load_parameters_path`, it needs to know **what format that checkpoint is in** so it can choose the right reader.

```yaml
source_checkpoint_layout: "orbax"   # default
```

This is a declaration, not a conversion — it tells MaxText which deserialization path to use before the checkpoint data reaches `checkpoint_conversion_fn`.

---

## 2. Available options

From MaxText's base.yml:
```yaml
# optional checkpoint context to use for loading.
# options: "orbax", "safetensors"
source_checkpoint_layout: "orbax"
```

| Value | Meaning |
|---|---|
| `"orbax"` | Checkpoint was written by Orbax (the normal MaxText training output format). Read using TensorStore/OCDBT/Zarr. |
| `"safetensors"` | Checkpoint is in HuggingFace safetensors format. Read using the safetensors loader. |

---

## 3. The safetensors format

Safetensors is a format developed by HuggingFace for storing tensors:
- Zero-copy loading (memory-mapped)
- No pickle → no arbitrary code execution on load
- Simple header + flat binary layout
- Widely adopted in the HuggingFace ecosystem

When you want to initialize MaxText with weights from a HuggingFace model that's distributed as `.safetensors` files:

```yaml
load_parameters_path: "/path/to/model.safetensors"
source_checkpoint_layout: "safetensors"
checkpoint_conversion_fn: <hf_to_maxtext_converter>
```

MaxText loads the file as safetensors, then converts the resulting dict.

---

## 4. The pipeline when `source_checkpoint_layout: "safetensors"`

```text
load_parameters_path (path to .safetensors file)
        ↓
source_checkpoint_layout = "safetensors"
        → use safetensors loader to read raw tensors
        ↓
raw dict {key: tensor, ...}  (HF naming convention)
        ↓
checkpoint_conversion_fn
        → remap keys, reshape tensors
        ↓
MaxText-compatible parameter dict
        ↓
model initialization
```

---

## 5. The pipeline when `source_checkpoint_layout: "orbax"` (default)

```text
load_parameters_path (path to Orbax checkpoint)
        ↓
source_checkpoint_layout = "orbax"
        → use Orbax/TensorStore reader
        ↓
MaxText-compatible parameter dict
        (no conversion needed — already in native format)
        ↓
model initialization
```

`checkpoint_conversion_fn` can remain `none` in this case.

---

## 6. When to change from default

Only change `source_checkpoint_layout` if you're loading a non-Orbax checkpoint. For any checkpoint created by MaxText's own training (`train.py`) or by the Orbax-based conversion scripts, the default `"orbax"` is correct.

---

## 7. Default

```yaml
source_checkpoint_layout: "orbax"
```

---

### One-line intuition

> **`source_checkpoint_layout` declares the on-disk format of the checkpoint being loaded — `"orbax"` for native MaxText checkpoints, `"safetensors"` when loading directly from HuggingFace-format weights.**
