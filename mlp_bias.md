
## 1. Why MLP bias exists

Standard transformer MLP layers (and expert MLPs) typically omit bias terms in their weight matrices — a choice that reduces parameters slightly and simplifies training at scale. Many modern architectures (LLaMA, Mistral, Qwen, DeepSeek) follow this pattern.

However, some architectures include a learnable bias added to each linear projection:

```text
Without bias:  output = W × input
With bias:     output = W × input + b
```

`mlp_bias` adds this learnable bias term to the MoE expert MLP's linear projections.

---

## 2. What it controls

```yaml
mlp_bias: false   # (default) no bias in expert MLP matmuls
mlp_bias: true    # add learnable bias to expert MLP projections
```

When `true`, each expert's up-projection (`W_up`), gate projection (`W_gate`), and down-projection (`W_down`) get a separate bias vector.

---

## 3. Why it was added to MaxText

MaxText's comment is explicit: `mlp_bias` was "originally added to support the GPT-OSS model architecture." GPT-2 and related GPT-OSS models use bias terms in their MLP layers. This flag enables loading and fine-tuning these models correctly.

---

## 4. Parameter count impact

For each expert with `moe_expert_input_dim=D` and `base_moe_mlp_dim=F`:

```text
Without bias: 2 × D × F + F × D = 3 × D × F parameters per expert
With bias:    3 × D × F + F + 2 × F = 3DF + 3F parameters per expert
```

The bias adds `3F` parameters per expert — negligible compared to the `3DF` weight parameters.

---

## 5. Default

```yaml
mlp_bias: false
```

No bias. Correct for all modern MoE architectures (Mixtral, DeepSeek, Qwen).

---

## 6. When to enable

- Fine-tuning or reproducing GPT-OSS (GPT-2 style) MoE models that include bias
- Loading a checkpoint from an architecture that was trained with bias
- Matching a specific published model config that uses bias

---

### One-line intuition

> **`mlp_bias` adds a learnable bias vector to the expert MLP projections — off by default and irrelevant for modern MoE architectures; exists to support GPT-OSS and similar legacy architectures that include MLP bias terms.**
