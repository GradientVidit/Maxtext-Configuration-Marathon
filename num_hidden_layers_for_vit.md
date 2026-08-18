## 1. Why does `num_hidden_layers_for_vit` exist?

The depth (number of stacked Transformer encoder layers) of the Vision Transformer determines its capacity to extract hierarchical visual representations, progressing from low-level edges to high-level semantic object concepts.

```text
Input Patches ──► [ Layer 0 ] ──► [ Layer 1 ] ──► ... ──► [ Layer (num_hidden_layers_for_vit - 1) ] ──► Output
```

`num_hidden_layers_for_vit` sets the total layer count in the Vision Transformer backbone.

---

## 2. Options and Defaults

| Value | ViT Variant | Parameter Scale |
|---|---|---|
| `34` (Default) | Llama 4 Vision ViT | Giant backbone (~1B params) |
| `24` | ViT-Large | ~300M params |
| `12` | ViT-Base | ~86M params |

---

## 3. Interactions

- **`remat_policy_for_vit`**: Rematerialization overhead scales with layer depth.
- **`deepstack_visual_indexes_for_vit`**: References specific layer indices within this depth.

---

### One-line intuition
> **`num_hidden_layers_for_vit` specifies the total number of Transformer layers in the Vision Transformer encoder stack.**
