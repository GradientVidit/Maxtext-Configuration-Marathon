
## 1. Why does `float32_gate_logits` exist?

In MoE, the **router** (gate network) decides which experts each token goes to. The router computes logits over all experts:

```text
gate_logits = token_hidden_state @ router_weight_matrix
              [emb_dim] × [emb_dim, num_experts] → [num_experts]
```

These gate logits then determine:
- Which `num_experts_per_tok` experts are selected (top-k)
- The routing weights assigned to those experts (softmax or sigmoid of logits)

The routing decision is discrete — small changes in logit values can flip which expert a token goes to. In bf16, rounding can cause tokens near the selection boundary to route to a **different expert** than they would in fp32:

```text
fp32 gate_logits: [2.301, 2.300, 1.500, ...]  → expert 0 selected (top-1)
bf16 gate_logits: [2.3,   2.3,   1.5,   ...]  → tie → may select expert 1 instead
```

`float32_gate_logits: true` prevents this by computing the router in fp32.

---

## 2. Default

```yaml
float32_gate_logits: false
```

Off by default. Unlike `float32_weight_sum` (which defaults `true` because it impacts final quality), gate logit precision issues are more subtle — they affect routing distribution, not the arithmetic of aggregation. MaxText defaults this off as an optimization, since the router computation is fast and the precision difference is usually small in practice.

---

## 3. What gets cast

```text
float32_gate_logits: false:
    hidden (bf16) @ router_weights (bf16) → gate_logits (bf16) → top-k → routing

float32_gate_logits: true:
    hidden.astype(fp32) @ router_weights.astype(fp32) → gate_logits (fp32) → top-k → routing
```

The cast is local to the router computation. The selected expert outputs and their aggregation are separate (controlled by `float32_weight_sum`).

---

## 4. When it matters most

```text
More impactful when:
  - num_experts is very large (many experts close to the selection boundary)
  - num_experts_per_tok is small (e.g., top-1 routing)
  - emb_dim is large (more accumulated rounding errors in the gate dot product)
  - Using auxiliary-free load balancing (where exact routing distribution matters more)

Less impactful when:
  - num_experts is small
  - Large num_experts_per_tok (less sensitive to boundary effects)
  - Dense models (num_experts=1) — completely irrelevant
```

---

## 5. Load balancing implications

The load balance loss (`load_balance_loss_weight`) penalizes uneven routing. If gate logits in bf16 introduce systematic routing biases (some experts consistently getting more/less due to rounding), the load balance signal becomes misleading. `float32_gate_logits: true` ensures the routing distribution reflects actual model preferences, not numerical artifacts.

---

## 6. Interaction with `float32_weight_sum`

```text
float32_gate_logits  → precision of "which experts and with what weights" (router)
float32_weight_sum   → precision of "combining those expert outputs" (aggregator)
```

For maximum MoE numerical stability, enable both:
```yaml
float32_gate_logits: true
float32_weight_sum: true
```

---

### One-line intuition

> **`float32_gate_logits` computes the MoE router scores in fp32 to prevent bf16 rounding from causing incorrect expert selection at the routing boundary — defaults off since the impact is subtle, but worth enabling for large expert pools or sensitive routing.**
