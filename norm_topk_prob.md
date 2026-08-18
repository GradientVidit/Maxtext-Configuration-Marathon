
## 1. The routing probability normalization question

Standard top-k MoE routing produces k routing weights for each token. These weights come from a softmax over all `num_experts` logits, then the top-k are selected:

```text
logits:       [0.3,  0.8,  0.6,  0.1,  0.5, ...]  (num_experts values)
softmax:      [0.08, 0.21, 0.16, 0.05, 0.13, ...]  (sums to 1.0)
top-2 select: [---, 0.21, 0.16, ---, ---, ...]

sum of selected weights: 0.21 + 0.16 = 0.37
```

The selected weights sum to 0.37 — **less than 1.0**. The unselected experts' probability mass is discarded.

Two philosophies about what to do:

1. **Use raw selected weights** — the scaling reflects routing confidence
2. **Renormalize to sum to 1** — ensures consistent output scale regardless of routing confidence

`norm_topk_prob` enables philosophy #2.

---

## 2. What it does

```yaml
norm_topk_prob: false  # (default) use raw top-k routing weights
norm_topk_prob: true   # renormalize top-k weights to sum to 1.0
```

When `true`:

```text
selected weights before norm: [0.21, 0.16]  (sum=0.37)
after normalization:          [0.57, 0.43]  (sum=1.00)
```

Each token's expert outputs are then weighted by these normalized values.

---

## 3. Why some architectures use this

Qwen3's router design relies on normalized probabilities for stable training — when routing weights always sum to 1.0, the output magnitude is more predictable and doesn't vary with how "confident" the router is. Without normalization, a token routed to two low-confidence experts has a dampened output versus one with two high-confidence assignments.

The normalization makes the MoE layer behave like a fixed-scale weighted average regardless of routing distribution sharpness. Qwen3 is the most prominent model to set this flag in MaxText configs, but the concept appears in other MoE designs as well.

---

## 4. Effect on gradient flow

Normalizing changes how gradients flow back through routing:

```text
without norm: ∂L/∂logits depends directly on absolute probability values
with norm:    ∂L/∂logits depends on relative ratios between selected experts
```

This can affect how aggressively the router learns to differentiate between experts. In practice, Qwen3 trains fine with it; the effect on other architectures is worth monitoring.

---

## 5. Options

| Value | Behavior |
|---|---|
| `false` (default) | Raw top-k weights — standard Mixtral/Switch Transformer/DeepSeek behavior |
| `true` | Normalized top-k weights to sum to 1 — used in Qwen3 and similar designs |

---

## 6. Interaction with related parameters

| Related param | Interaction |
|---|---|
| `num_experts_per_tok` | k=1 makes normalization a no-op (single weight is already 1.0 after norm); only matters for k≥2 |
| `routed_scaling_factor` | Applied after normalization if norm is enabled; scales the combined output |
| `load_balance_loss_weight` | Aux loss operates on pre-normalization probabilities; normalization doesn't remove the need for load balancing |

---

## 7. Practical guidance

**Reproducing Qwen3:** set `norm_topk_prob=true`.

**Mixtral, DeepSeek, Switch Transformer, or generic MoE:** leave `false`.

**When `num_experts_per_tok=1`:** normalization has no effect — a single weight normalized to 1.0 is identical to using it raw.

**Note:** Qwen3 uses softmax scoring with this renormalization. DeepSeek-V3 uses sigmoid scoring without renormalization — these are two different solutions to the same "output scale" problem.

---

### One-line intuition

> **`norm_topk_prob` renormalizes the selected top-k routing weights to sum to 1.0, ensuring consistent output scale regardless of router confidence — a Qwen3-specific design choice; leave `false` for other architectures.**
