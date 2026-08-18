## 1. Why does `remat_policy_for_vit` exist?

High-resolution vision processing generates massive intermediate activation tensors during the Vision Transformer forward pass (e.g. processing 1024x1024 images generates thousands of patch tokens per layer). When fine-tuning the vision encoder (`freeze_vision_encoder_params: false`), storing all ViT layer activations for the backward pass will rapidly exhaust TPU High Bandwidth Memory (HBM).

Rematerialization (gradient checkpointing) recomputes forward activations during the backward pass instead of holding them in memory.

```text
Full Storage (remat_policy_for_vit = "none"):
Forward:  [ViT Layer 0] ──► [ViT Layer 1] ──► ... ──► [ViT Layer 33]
Memory:   [== Save 0 ==]    [== Save 1 ==]            [== Save 33 ==]  (Severe HBM Spike!)

Rematerialization (remat_policy_for_vit = "minimal" / "full"):
Forward:  [ViT Layer 0] ──► [ViT Layer 1] ──► ... ──► [ViT Layer 33]
Memory:   [ Save input ]                              [ Save input ]   (Low HBM!)
Backward: Recompute forward activations on-the-fly from layer inputs.
```

`remat_policy_for_vit` configures the gradient checkpointing strategy specifically for the Vision Transformer encoder.

---

## 2. Policy Options

| Value | Behavior | Memory Savings vs Compute Overhead |
|---|---|---|
| `"minimal"` (Default) | Checkpoints key boundary activations; balances recompute and memory | Standard default in MaxText |
| `"full"` | Recomputes entire ViT attention and MLP blocks | Maximum HBM savings; allows huge image resolutions |
| `"none"` | Saves all intermediate activations | Zero recompute overhead; requires ample HBM |
| `"save_dot_except_mlp"` / `"qkv_proj_offloaded"` | Fine-grained remat policies | Advanced hardware tuning |

---

## 3. Interactions

- **`freeze_vision_encoder_params`**: If `true`, the ViT has no backward pass, making `remat_policy_for_vit` essentially a no-op.
- **`remat_policy`**: Controls the separate rematerialization policy for the main LLM decoder stack.

---

### One-line intuition
> **`remat_policy_for_vit` controls gradient checkpointing and activation recomputation specifically inside the Vision Transformer encoder to prevent HBM exhaustion during image training.**
