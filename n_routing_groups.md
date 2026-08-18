
## 1. The problem with pure top-k routing at scale

Standard top-k MoE routing: each token can be sent to **any** of the N experts. With N=256 experts (DeepSeek-V3 scale) and k=8, each token accesses 8 of 256 experts — but these 8 could be spread across all 256, requiring potential communication with every device in an expert-parallel setup.

DeepSeek introduced **grouped routing**: divide the N experts into G groups of N/G experts each. A token first selects a subset of groups to route into, then selects its k experts from within those groups. This limits the "fan-out" of routing — a token only communicates with a subset of devices, not all of them.

`n_routing_groups` sets G — the number of groups.

---

## 2. The routing mechanism with groups

```text
Without groups:
  token → router → top-k from ALL 256 experts
  
With n_routing_groups=4 (64 experts/group) and topk_routing_group=4:
  token → router → score all 4 groups
                → select top 4 groups (all of them here, but the mask keeps scores valid within-group only)
                → pick top-8 experts from within those 4 groups
                → total: 8 experts, but drawn from the 4-group constrained space
```

This is DeepSeek-V3's design: 256 experts in 4 groups of 64, all 4 groups selected, 8 experts total.

---

## 3. Expert parallelism alignment

The primary benefit depends on the specific grouping config:

- **When topk_routing_group < n_routing_groups:** a token only communicates with `topk_routing_group / n_routing_groups` fraction of EP devices — reducing all-to-all fan-out. Example: 8 groups, 3 selected = 3/8 devices per token.

- **DeepSeek-V3 (4 groups, 4 selected):** all groups are in scope, so there's no fan-out reduction from grouping. The benefit here is different: group-level scoring normalizes the expert selection to prevent any single group from dominating, which improves load balance across expert regions.

The grouping mechanism is flexible — the benefit shifts from fan-out reduction (partial topk) to load regularization (full topk) depending on the ratio.

---

## 4. Default

```yaml
n_routing_groups: -1   # disabled
topk_routing_group: -1  # disabled
```

`-1` = grouped routing disabled. Standard top-k routing over all experts.

---

## 5. The companion: `topk_routing_group`

| Param | Controls |
|---|---|
| `n_routing_groups` | How many groups N experts are divided into |
| `topk_routing_group` | How many of those groups each token routes into |

These are always set together. A token routes into `topk_routing_group` groups and picks `num_experts_per_tok` total experts from within those groups.

---

## 6. Constraints and practical notes

- `num_experts` must be divisible by `n_routing_groups` (clean group partition)
- `num_experts_per_tok` ÷ `topk_routing_group` doesn't need to be exact integer division — the actual implementation selects the top-k experts across all unmasked groups after scoring, so experts aren't rigidly split equally across selected groups. In DeepSeek-V3 (8 experts, 3 groups selected), the 8 top-k experts are drawn from the pool of experts in the 3 groups, not 2-3 from each.

---

## 7. Practical use

**Reproducing DeepSeek-V3:** `n_routing_groups=4`, `topk_routing_group=4`, `num_experts=256`, `num_experts_per_tok=8`. Note: with 4 groups and topk=4, all groups are selected — the grouping primarily constrains expert selection to be within-group relative to other scoring mechanisms.

**Generic MoE:** leave at `-1`. Grouped routing only matters when EP is large and all-to-all fan-out is the bottleneck.

---

### One-line intuition

> **`n_routing_groups` divides experts into G groups and limits routing to `topk_routing_group` of them, reducing the all-to-all fan-out in expert parallelism — DeepSeek-V3's mechanism for making 256-expert routing practical at large EP scale.**
