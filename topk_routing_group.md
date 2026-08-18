
## 1. What this controls

`topk_routing_group` specifies how many expert groups each token routes into, in DeepSeek's grouped routing design.

Works together with `n_routing_groups`: given N experts divided into G groups, a token first selects `topk_routing_group` groups, then selects `num_experts_per_tok` experts from within those groups.

---

## 2. The selection hierarchy

```text
num_experts=256, n_routing_groups=4 → 4 groups of 64 experts each

topk_routing_group=4 → each token selects all 4 groups
num_experts_per_tok=8 → then picks 8 experts total from within those 4 groups
                    → selection is constrained within-group, not global
```

---

## 3. Expert parallelism implication

With 8 groups and 3 selected per token:
```text
EP communication fan-out per token = 3/8 of EP devices
vs. standard routing fan-out = potentially all EP devices
```

This is the core benefit: each token only needs results from a bounded number of EP workers, reducing all-to-all volume.

---

## 4. Default

```yaml
topk_routing_group: -1
```

Disabled. Set together with `n_routing_groups`.

---

## 5. Constraint

- `num_experts` must be divisible by `n_routing_groups` — for clean group partitioning
- The top-k expert selection happens across the union of experts in the selected groups, not by rigidly allocating k/topk_routing_group per group. So `num_experts_per_tok` and `topk_routing_group` don't need an exact divisibility relationship; you simply select your k best experts from within the unmasked groups.

---

## 6. DeepSeek-V3 config

```yaml
num_experts: 256
n_routing_groups: 4
topk_routing_group: 4
num_experts_per_tok: 8
```

Note: with 4 groups and topk=4, all groups are in scope. The grouping mechanism in DeepSeek-V3 constrains selection to be within-group relative to the group-level scores, rather than a pure global top-k.

---

### One-line intuition

> **`topk_routing_group` is the "top-k groups" in DeepSeek's two-stage routing — each token first picks this many expert groups, then picks its `num_experts_per_tok` experts from within them, bounding the all-to-all fan-out in expert parallelism.**
