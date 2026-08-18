
## 1. Why random routing exists as an option

MoE systems have two separable concerns:

```text
1. Routing quality     — does the router learn to assign tokens to the right experts?
2. Routing mechanics   — does the dispatch/gather infrastructure work correctly?
```

When you first build or debug an MoE training pipeline, concern #2 is often what breaks. But if the router is also learning, failures are ambiguous — is the model not converging because the router is bad, or because the dispatch is broken?

`use_random_routing` isolates concern #2 by removing the router from the equation entirely. Tokens are assigned to experts uniformly at random, bypassing all learned routing logic.

---

## 2. What it does

```yaml
use_random_routing: false  # (default) use learned top-k router
use_random_routing: true   # assign tokens to experts uniformly at random
```

When `true`, the router's learned weights are ignored. Each token gets assigned to a randomly selected expert (or k random experts if `num_experts_per_tok > 1`). No softmax, no gradient through routing decisions.

---

## 3. What you can (and can't) test with it

**With `use_random_routing=true` you CAN verify:**
- Expert dispatch/gather infrastructure works (tokens actually reach experts and come back)
- Memory layouts and buffer sizing are correct
- MoE compute path produces valid gradients
- Load balancing under worst-case random assignment

**You CANNOT test:**
- Router quality or convergence
- Load balancing effectiveness (random routing will be naturally balanced in expectation)
- Whether `load_balance_loss_weight` actually helps

---

## 4. Interaction with load balancing

Random routing produces approximately balanced load (in expectation, `1/num_experts` tokens per expert). This means:

- `capacity_factor=1.0` will work fine with random routing
- `load_balance_loss_weight > 0` has no gradient signal through routing (since there's no router to train) — aux loss is computed but doesn't affect anything meaningful
- `ragged_buffer_factor=1.0` is safe since load is balanced

---

## 5. Default

```yaml
use_random_routing: false
```

Confirmed in `base.yml`. This is always the right default for real training — learned routing is the whole point of MoE.

---

## 6. When to use it

**Commissioning a new MoE model:** start with `use_random_routing=true` to verify the dispatch infrastructure works before adding routing complexity.

**Debugging dispatch/gather kernels:** isolates kernel correctness from routing issues.

**Benchmarking expert compute:** random routing gives deterministic, balanced load useful for benchmarking expert MLP throughput.

**Never in production training:** random routing doesn't learn expert specialization. The whole value of MoE is learned specialization — random routing throws it away.

---

### One-line intuition

> **`use_random_routing` replaces the learned router with random expert assignment — useful for debugging the MoE dispatch infrastructure in isolation, never for actual training.**
