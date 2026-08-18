## 1. The load balancing problem — the DeepSeek-V3 angle

Traditional MoE load balancing uses an auxiliary loss (`load_balance_loss_weight`) to push routing toward uniform expert utilization. This approach has a well-documented flaw: the aux loss creates a tension with the main loss — the model is simultaneously trying to route tokens optimally for quality **and** route them uniformly for balance. These goals conflict, and the aux loss degrades model quality.

DeepSeek-V3 introduced an "auxiliary-loss-free" alternative: a **learnable per-expert bias** added to the routing logits **for the top-k selection step only**. The routing decision becomes:

```text
routing_decision_score_i = token_score_i + bias_i   ← used for which experts are selected
gating_weight_i          = token_score_i             ← raw score, no bias, used for output weighting
```

Critical distinction: the bias affects **which experts are chosen**, but the **actual combination weights** applied to expert outputs still use the raw (unbiased) scores. This preserves the integrity of the learned representations.

The `bias_i` is updated dynamically during training to nudge routing toward underused experts — without touching the main gradient signal.

`routed_bias` enables this mechanism.

---

## 2. How the bias update works

```text
After each training step:
    for each expert i:
        if load(expert_i) > target_load:
            bias_i -= routed_bias_update_rate   # reduce appeal → reduce routing to this expert
        else:
            bias_i += routed_bias_update_rate   # increase appeal → route more to this expert
```

The bias is updated outside of standard gradient descent — it's a feedback controller on top of the learned router.

---

## 3. What `routed_bias` controls

```yaml
routed_bias: false   # (default) no routing bias
routed_bias: true    # enable DeepSeek-V3 style auxiliary-loss-free load balancing bias
```

---

## 4. vs. `load_balance_loss_weight`

| Mechanism | Gradient through? | Quality impact | DeepSeek approach |
|---|---|---|---|
| `load_balance_loss_weight > 0` | Yes — aux loss touches main training | Minor quality degradation | V2 and earlier |
| `routed_bias=True` | No — bias updated outside main gradient | Minimal quality impact | V3 ("aux-loss-free") |

> **Important nuance:** DeepSeek-V3's training still used a very small complementary sequence-level balance loss (α=0.0001) to prevent extreme within-sequence imbalance. "Auxiliary-loss-free" refers to removing the heavy batch-level aux loss, not removing all balance supervision entirely.

They're alternatives at the main balance level. Don't combine a large `load_balance_loss_weight` with `routed_bias=True`.

---

## 5. Default

```yaml
routed_bias: false
routed_bias_update_rate: 0.0
```

Both default to "off." Enable the pair together.

---

## 6. Interaction with `routed_bias_update_rate`

`routed_bias_update_rate` is the step size of the feedback update. Too large → oscillation. Too small → slow balancing.

DeepSeek-V3 paper used γ=0.001 for the first 14.3T training tokens, then set it to 0 for the final 500B tokens.

---

### One-line intuition

> **`routed_bias` adds a per-expert bias to routing scores used only for expert selection — not for output weighting — letting a feedback controller balance expert load without touching the main training gradient.**
