
## 1. The load imbalance problem

MoE routing is learned — there's nothing in the base loss function that forces tokens to spread evenly across experts. Without any explicit incentive, the router tends to collapse: a few experts get most of the load, the rest are barely used.

```text
ideal (8 experts, uniform):     [12.5%, 12.5%, 12.5%, 12.5%, 12.5%, 12.5%, 12.5%, 12.5%]
collapsed (without aux loss):   [80%,    5%,    5%,    3%,    2%,    2%,    2%,    1%]
```

Expert collapse wastes capacity — you pay for 8 experts but effectively use 1.

The fix: add an auxiliary loss term that penalizes imbalanced routing. `load_balance_loss_weight` is the coefficient on that penalty.

---

## 2. What the aux loss does

The standard load-balancing auxiliary loss (from the Switch Transformer / Mixtral lineage):

```text
L_aux = num_experts × Σ_i (f_i × P_i)
```

where:
- `f_i` = fraction of tokens actually routed to expert i
- `P_i` = average routing probability for expert i over the batch
- Minimizing this pushes both `f_i` and `P_i` toward `1/num_experts`

The total training loss becomes:

```text
L_total = L_main + load_balance_loss_weight × L_aux
```

`load_balance_loss_weight` scales how strongly this penalty pulls routing toward balance.

---

## 3. The default: `0.0`

```yaml
load_balance_loss_weight: 0.0
```

Disabled. This makes sense for:
- Dense baseline runs (`num_experts=1` — aux loss is meaningless)
- Models using `routed_bias` (DeepSeek-V3 style) — the learnable bias handles load balancing without an aux loss
- Debugging routing in isolation

---

## 4. Typical values

| Value | Behavior |
|---|---|
| `0.0` (default) | Disabled — no explicit load balancing |
| `0.001` | Standard — Mixtral 8x7B uses this value (`router_aux_loss_coef=0.001`) |
| `0.01` | Higher pressure — can cause routing entropy to over-regularize |
| `0.1` | Too strong in most cases — forces near-uniform routing regardless of token content |

Values above `0.01` are rarely beneficial and can actually hurt quality by forcing the router to make near-uniform decisions regardless of token content. Switch Transformer (2021) explored a wide range; most production models converge on 0.001–0.01.

---

## 5. Interaction with related parameters

| Related param | Interaction |
|---|---|
| `num_experts` | More experts → more important to balance (collapse is worse with 256 experts than 8) |
| `capacity_factor` | Tighter capacity enables safer use of higher `load_balance_loss_weight` |
| `routed_bias` | DeepSeek-V3 alternative — uses a learnable bias instead of aux loss; don't combine large `load_balance_loss_weight` with `routed_bias=true` |
| `use_random_routing` | Random routing guarantees balance by construction — aux loss is irrelevant |

---

## 6. Two paradigms: aux loss vs. bias correction

```text
aux loss (load_balance_loss_weight > 0)
  → gradient signal pushes router weights toward balanced output
  → affects training dynamics; slightly hurts main loss quality
  → standard in Mixtral, Switch Transformer

bias correction (routed_bias=true)
  → learnable per-expert bias nudges routing toward underused experts
  → doesn't touch the main gradient flow
  → DeepSeek-V3's "auxiliary-loss-free" balancing
```

Pick one. They're alternatives, not complements.

---

## 7. Practical scenarios

**Standard MoE training (Mixtral-style):** `load_balance_loss_weight=0.001` — matches Mixtral 8x7B's actual config.

**DeepSeek-V3 style:** `load_balance_loss_weight=0.0`, `routed_bias=true`, `routed_bias_update_rate=0.001`.

**Debugging expert collapse:** increase temporarily to `0.01` to force exploration, then reduce.

**Dense model (`num_experts=1`):** leave at `0.0` — no effect.

---

### One-line intuition

> **`load_balance_loss_weight` scales an auxiliary loss that penalizes routing imbalance — without it, MoE routers tend to collapse to a few dominant experts, wasting the capacity of the rest.**
