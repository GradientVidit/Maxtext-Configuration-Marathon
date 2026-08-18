## 1. Why does it exist?

Standard Splash Attention uses a static grid scheduler to assign sequence blocks to TPU execution cores.

**Tokamax** (Google's optimized kernel library for JAX) includes an advanced dynamic splash-attention scheduler that optimizes block visitation orders, improves causal mask chunk skipping, and enhances memory access locality across hardware cores.

```text
Static Grid Scheduler (use_splash_scheduler: false):
  Fixed block-by-block execution regardless of causal sparsity.

Tokamax Splash Scheduler (use_splash_scheduler: true):
  Optimized block scheduling, skipping fully masked causal blocks and balancing core loads.
```

`use_splash_scheduler` enables the Tokamax splash attention scheduler for improved kernel scheduling efficiency.

---

## 2. Options & Configuration

| Value | Meaning |
|---|---|
| `false` (default) | Standard static Splash Attention scheduling. |
| `true` | Activates the Tokamax splash attention scheduler. |

Default in `base.yml`:
```yaml
use_splash_scheduler: false # to use tokamax splash attention scheduler.
```

---

### One-line intuition

> **`use_splash_scheduler` activates Tokamax's advanced splash attention block scheduler, optimizing block traversal orders and causal mask evaluation across TPU cores.**
