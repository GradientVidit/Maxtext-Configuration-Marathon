## 1. Why does `freeze_vision_encoder_params` exist?

In modern Vision-Language Model (VLM) training recipes, high-quality pretrained vision backbones (e.g., SigLIP, CLIP, or EVA-CLIP) already possess rich visual representations. Training the entire vision encoder end-to-end from step 0 often leads to catastrophic forgetting, destabilizes cross-modal alignment, and drastically increases gradient memory and backward-pass compute.

```text
Training with freeze_vision_encoder_params: true (Standard Recipe):
[ Raw Image ] ──► [ ViT Encoder (FROZEN: Gradients = 0) ] ──► [ Projector (TRAINABLE) ] ──► [ LLM (TRAINABLE) ]
                  ▲ No gradient computation or optimizer state!

Training with freeze_vision_encoder_params: false (Full Finetuning):
[ Raw Image ] ──► [ ViT Encoder (TRAINABLE) ] ──────────────► [ Projector (TRAINABLE) ] ──► [ LLM (TRAINABLE) ]
                  ▲ High HBM memory, full backward pass
```

`freeze_vision_encoder_params` controls whether the Vision Transformer weights receive gradient updates and optimizer state allocations during training.

---

## 2. Memory and Computational Impact

Setting `freeze_vision_encoder_params: true`:
1. **Zero Optimizer Memory**: No Adam moments ($m, v$) are allocated for the ViT's hundreds of millions of parameters.
2. **Skipped VJP Compute**: The backward pass stops at the input of the projector; no vector-Jacobian products (VJPs) are evaluated across the ViT layers.
3. **Huge Throughput Boost**: Saves significant TPU HBM and FLOPs per step.

---

## 3. Options and Defaults

| Value | Behavior | When to Use |
|---|---|---|
| `true` (Default) | ViT weights are completely frozen (eval mode) | Stage-1 projector pre-training, standard visual instruction tuning |
| `false` | ViT weights are actively trained with gradients | Stage-2 full end-to-end multimodal fine-tuning, domain adaptation |

---

## 4. Interactions

- **`use_multimodal`**: Must be `true`.
- **`trainable_parameters_mask`**: Works alongside parameter filtering rules to mask out ViT variables from Optax updates.
- **`remat_policy_for_vit`**: If frozen, rematerialization for ViT is largely irrelevant because activations do not need to be saved for backward passes.

---

### One-line intuition
> **`freeze_vision_encoder_params` stops gradient updates and optimizer allocations for the Vision Transformer, implementing the standard frozen-backbone multimodal training recipe.**
