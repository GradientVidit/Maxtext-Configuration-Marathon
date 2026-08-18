## 1. Why does it exist?

In standard attention softmax computations, normalizing logits requires dividing by the sum of exponentials:

$$P_{ij} = \frac{\exp(S_{ij})}{\sum_k \exp(S_{ik})} = \exp(S_{ij}) \times \frac{1}{\sum_k \exp(S_{ik})}$$

On TPU Vector Processing Units (VPUs), floating-point division is significantly more expensive than floating-point multiplication.

Fusing the reciprocal computation ($1 / \sum \exp$) calculates the inverse scalar once and replaces per-element vector divisions with high-speed vector multiplications.

```text
Without Reciprocal Fusion:
  Every attention logit element divides by denominator (Slow vector division).

With sa_fuse_reciprocal: true:
  Compute scalar reciprocal = 1.0 / denominator once
  ──→ Multiply all elements by scalar reciprocal (Fast vector multiply).
```

`sa_fuse_reciprocal` enables reciprocal fusion in the Splash Attention softmax calculation.

---

## 2. Options & Configuration

| Value | Meaning |
|---|---|
| `true` (default) | Fuses the reciprocal calculation (Tokamax default); maximizes VPU performance. |
| `false` | Unfused standard division. |

Default in `base.yml`:
```yaml
sa_fuse_reciprocal: true # defaults to true in Tokamax
```

---

### One-line intuition

> **`sa_fuse_reciprocal` fuses the softmax normalization division into a single reciprocal scalar multiplication, avoiding expensive per-element floating-point divisions on TPU vector units.**
