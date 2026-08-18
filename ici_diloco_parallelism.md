## 1. Why does it exist?

**DiLoCo** (Distributed Low-Communication) optimization trains multiple replica models independently on local batches for $H$ inner optimization steps before exchanging pseudo-gradients via an outer optimizer update.

While DiLoCo is primarily intended for cross-datacenter (DCN) scaling, `ici_diloco_parallelism` allows running multiple independent DiLoCo replica islands within a single large TPU slice (e.g., testing DiLoCo convergence algorithms on a single pod without requiring multi-cluster infrastructure).

---

## 2. Options & Configuration

| Value | Meaning |
|---|---|
| `1` (default) | DiLoCo intra-slice training is disabled (standard synchronous training). |
| Integer $> 1$ | Allocates `N` independent DiLoCo optimizer islands within the slice. |

Default in `base.yml`:
```yaml
ici_diloco_parallelism: 1
```

---

### One-line intuition

> **`ici_diloco_parallelism` configures independent DiLoCo training islands within a single TPU slice for local experimentation and algorithm verification.**
