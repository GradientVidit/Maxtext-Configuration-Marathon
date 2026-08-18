
## 1. What hash-based routing is

All the routing mechanisms discussed so far (top-k learned routing, bias-corrected routing, grouped routing) rely on the router **learning** to assign tokens to experts through gradient descent. This learning is what enables expert specialization — different experts become good at different types of tokens.

An alternative that bypasses learning entirely: **hash-based routing**. A token is assigned to an expert based on a deterministic hash of some token attribute (e.g. token ID, position, or a combination). No router network, no gradient, no learned weights.

`first_num_hash_layers` specifies how many of the initial MoE layers use hash routing instead of learned routing.

---

## 2. Why hash routing exists

DeepSeek V4 (and related research) explores hash routing for initial layers as a way to:
- **Bootstrap routing stability:** hash routing is deterministic and perfectly balanced by design — useful early in training before the learned router has converged
- **Reduce early-training routing noise:** in early steps, the router is essentially random anyway; hash routing makes the randomness deterministic and balanced
- **Compute savings:** no router forward/backward pass for hash-routed layers

---

## 3. The structure with hash routing

```text
Layer 0: MoE with hash routing (no learned router, no router gradients)
...
Layer K: MoE with hash routing
Layer K+1: MoE with learned routing  ← standard
...
```

`first_num_hash_layers=K` → first K MoE layers use hash routing.

Note: this interacts with `first_num_dense_layers`. Dense layers come first, then hash-routed MoE layers, then learned-routed MoE layers.

---

## 4. Default

```yaml
first_num_hash_layers: 0
```

All MoE layers use learned routing. Hash routing is a DeepSeek V4 experimental feature.

---

## 5. Options

| Value | Effect |
|---|---|
| `0` (default) | All MoE layers use learned routing |
| `K > 0` | First K MoE layers use hash routing; remainder use learned routing |

---

## 6. When to use it

**Reproducing DeepSeek V4:** set to match their architecture specification.

**Experimenting with routing stability:** hash routing in early layers can be a training stability tool when learned routing collapses.

**In production MoE:** leave at `0`. Learned routing is what provides expert specialization.

---

### One-line intuition

> **`first_num_hash_layers` routes tokens to experts via a deterministic hash function (no learning) in the first N MoE layers — a DeepSeek V4 technique for routing stability; in standard MoE training, leave at `0`.**
