
## 1. What routing scores are and why scaling them matters

After the router linear projection, each token has a score for each expert. In most MoE architectures the flow is:

```text
token embedding → linear layer → expert logits → softmax → routing probs → top-k select → combine
```

DeepSeek-V3 uses a different flow:

```text
token embedding → linear layer → expert logits → sigmoid (per-expert, independent) → top-k select
→ normalize selected scores (to sum to 1) → multiply by routed_scaling_factor → combine
```

`routed_scaling_factor` is the final multiplier applied to the **normalized** routing weights before they're used to combine expert outputs.

---

## 2. What it controls

```yaml
routed_scaling_factor: 1.0  # (default) no scaling
routed_scaling_factor: 2.5  # DeepSeek-V3 value
```

At `1.0`: identity — no scaling effect. The combined expert outputs are weighted by the normalized routing weights directly.

At `2.5` (DeepSeek-V3): the normalized weights are amplified by 2.5× before weighting expert outputs. This scales up the contribution of expert activations relative to the residual stream.

---

## 3. Why DeepSeek-V3 uses this

DeepSeek-V3's flow is: sigmoid scores → top-k selection → normalize to sum 1 → multiply by `routed_scaling_factor` → weight expert outputs.

The normalization after sigmoid produces weights summing to 1.0, similar to `norm_topk_prob=True`. The `routed_scaling_factor` then amplifies those normalized weights before the final combine. The specific value of 2.5 was determined empirically — it controls the "gain" of the MoE layer relative to the residual connection.

---

## 4. Default

```yaml
routed_scaling_factor: 1.0
```

No scaling. Correct for Mixtral-style MoE (softmax probabilities, no extra scaling) and any architecture that doesn't apply a post-normalization multiplier.

---

## 5. Interaction with `norm_topk_prob` and `routed_score_func`

| `routed_score_func` | `norm_topk_prob` | `routed_scaling_factor` | Effect |
|---|---|---|---|
| `softmax` | `false` | `1.0` | Raw softmax weights (Mixtral style) |
| `softmax` | `true` | `1.0` | Normalized weights to sum=1 (Qwen3 style) |
| `sigmoid` | — | `2.5` | Sigmoid → normalize → scale by 2.5 (DeepSeek-V3 style) |

Note: `norm_topk_prob` and DeepSeek's internal normalization are related but independent code paths — check the specific architecture's implementation.

---

### One-line intuition

> **`routed_scaling_factor` is a final multiplier on normalized routing weights before combining expert outputs — at `1.0` it's a no-op; DeepSeek-V3 uses `2.5` as a learned empirical gain on their sigmoid→normalize→scale routing pipeline.**
