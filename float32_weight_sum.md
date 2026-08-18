
## 1. Why does `float32_weight_sum` exist?

In Mixture of Experts (MoE), a router assigns each token to a subset of experts with associated routing weights (scores). The final output is a weighted sum of expert outputs:

```text
output = Σ_i (router_weight_i × expert_i_output)
```

In bf16, this weighted sum has precision issues:
- `router_weight_i` values are typically in `[0, 1]` — small, precise values
- Expert outputs can be large (MLP intermediate values)
- Multiplying and summing many such products in bf16 → accumulated rounding error

The "weight sum" here is specifically the final aggregation of expert outputs, not the weight matrices themselves.

`float32_weight_sum: true` performs this aggregation in fp32 to prevent precision loss in the expert combination step.

---

## 2. Default

```yaml
float32_weight_sum: true
```

**On by default.** This is one of the places where MaxText's defaults reflect hard-won experience: MoE models saw measurable quality degradation with bf16 expert aggregation, so fp32 is the safer default.

---

## 3. What it controls precisely

In the MoE forward pass:

```text
expert_outputs: list of [batch, seq_per_expert, emb_dim] tensors in bf16
router_weights: [num_selected_experts] in bf16

float32_weight_sum: false:
    combined = Σ_i (router_weights[i].astype(bf16) × expert_outputs[i])
    → accumulates in bf16

float32_weight_sum: true:
    combined = Σ_i (router_weights[i].astype(fp32) × expert_outputs[i].astype(fp32))
    → accumulates in fp32, then cast back to bf16
```

---

## 4. When MoE is active

This flag only has an effect when:
- `num_experts > 1` (MoE is actually being used)
- The MoE path is active (the model is using sparse routing)

For dense models (`num_experts=1`), this flag is a no-op.

---

## 5. Cost

Casting the expert aggregation to fp32 doubles the memory for these intermediate tensors temporarily. In practice, this is a small fraction of total model memory (the expert MLPs themselves are much larger). The compute cost is also modest since this is a simple weighted sum, not a matmul.

---

## 6. When to disable

The only reason to set `float32_weight_sum: false`:
- Reducing memory to absolute minimum (the savings are small)
- Specifically testing the precision impact on MoE quality
- Matching a reference implementation that uses bf16 throughout

In general, leave it `true`.

---

## 7. Interaction with `float32_gate_logits`

These are two different precision knobs in the MoE pipeline:

```text
float32_gate_logits  → precision of the ROUTER computation (who gets selected)
float32_weight_sum   → precision of the AGGREGATION (how selected experts combine)
```

Both default to or recommend higher precision because MoE routing is numerically sensitive at both stages.

---

### One-line intuition

> **`float32_weight_sum` aggregates MoE expert outputs in fp32 to prevent bf16 rounding errors from corrupting the final weighted combination of expert outputs — on by default because MoE aggregation is numerically sensitive.**
