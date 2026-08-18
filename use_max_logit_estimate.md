## 1. Why does it exist?

Online Flash Attention maintains running maximum values $m_i$ for each Query row to prevent numerical overflow in the $\exp(S_{ij} - m_i)$ calculation. In the first tile iteration, $m_i$ is initialized to $-\infty$, and intermediate scaling factors must be computed dynamically as new tiles arrive.

If an approximate upper bound for attention logits is known beforehand (e.g. from logit soft-capping or activation bounds), supplying a static max-logit estimate simplifies the initial online softmax scaling steps in the kernel.

```text
Standard Online Softmax:
  Tile 0: m = -inf ──→ Dynamic rescale ──→ Tile 1: update m ──→ Dynamic rescale

With use_max_logit_estimate > 0:
  Kernel uses pre-seeded logit estimate ──→ Reduces initial scaling adjustments
```

`use_max_logit_estimate` provides an optional upper-bound logit estimate for Splash Attention numerical scaling.

---

## 2. Options & Configuration

| Value | Meaning |
|---|---|
| `-1` (default) | No estimate provided; standard dynamic online softmax max-tracking is used. |
| Positive float $> 0$ | Explicit max-logit estimate passed to the kernel. |

Default in `base.yml`:
```yaml
use_max_logit_estimate: -1 # -1 means no estimate, any > 0 value will be used as max logit estimate
```

---

### One-line intuition

> **`use_max_logit_estimate` provides a pre-seeded maximum logit value to Splash Attention to optimize initial online softmax scaling.**
