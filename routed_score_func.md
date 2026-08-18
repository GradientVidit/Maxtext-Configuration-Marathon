
## 1. What the router scoring function is

The MoE router maps token embeddings to per-expert scores. The final routing decision depends on how those scores are computed from the router's output logits:

```text
token embedding → linear projection → router logits → scoring function → routing weights → top-k select
```

The "scoring function" is the final transformation from raw logits to routing weights. Options include:
- **softmax** — exponentiated logits normalized over all experts (default for most MoE)
- **sigmoid** — independent probability per expert (used in some architectures)
- **softmax with temperature** — softmax with a tunable sharpness parameter

`routed_score_func` selects which scoring function is used.

---

## 2. What it controls

```yaml
routed_score_func: ""      # (default) use the architecture's built-in default
routed_score_func: "softmax"  # explicit softmax
routed_score_func: "sigmoid"  # sigmoid per-expert
```

The empty string `""` falls back to whatever the decoder block's default scoring function is for routing. Override with an explicit value to select a specific function.

---

## 3. softmax vs. sigmoid

**softmax (standard):**
```text
score_i = exp(logit_i) / Σ_j exp(logit_j)
Σ scores = 1.0
→ naturally normalized over all experts
→ selecting top-k discards (1 - scores of selected)
```

**sigmoid (DeepSeek-V3 style):**
```text
score_i = 1 / (1 + exp(-logit_i))
Scores are independent — no normalization constraint
→ each expert gets an independent probability
→ after top-k selection, DeepSeek-V3 normalizes the selected scores internally
  (not via the norm_topk_prob flag, but in the architecture's own forward pass)
→ then multiplies by routed_scaling_factor (2.5) for the final combine weights
```

DeepSeek-V3 uses sigmoid scoring with its own normalization + scaling step. This is why `routed_score_func="sigmoid"` and `routed_scaling_factor=2.5` are set together.

---

## 4. Default

```yaml
routed_score_func: ""
```

Architecture default. For most architectures, this is softmax. Override explicitly when implementing DeepSeek-V3 style sigmoid routing.

---

## 5. Interaction with related parameters

| Related param | Interaction |
|---|---|
| `norm_topk_prob` | Normalization of top-k weights — interacts with scoring function output scale |
| `routed_scaling_factor` | Scaling of final routing weights — especially relevant with sigmoid (scores not normalized) |
| `load_balance_loss_weight` | Aux loss operates on routing probabilities; scoring function affects their distribution |

---

### One-line intuition

> **`routed_score_func` selects how router logits are converted to routing weights — leave empty for architecture default (softmax), or set to `"sigmoid"` for DeepSeek-V3 style independent per-expert scoring.**
