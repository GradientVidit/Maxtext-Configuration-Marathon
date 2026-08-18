
## 1. What GMM v2 is

When `use_tokamax_gmm=True`, MaxText uses Tokamax's GMM kernel for MoE expert matmuls. Tokamax has released multiple kernel versions. `use_gmm_v2` selects between v1 and v2.

---

## 2. What it controls

```yaml
use_gmm_v2: false  # (default) Tokamax GMM v1
use_gmm_v2: true   # Tokamax GMM v2
```

**Precondition:** `use_tokamax_gmm: true`. Without it, `use_gmm_v2` has no effect.

---

## 3. Typical differences between GMM versions

GMM v2 in Tokamax typically offers:
- Updated tiling strategies for newer TPU generations
- Different memory access patterns that may be faster or slower depending on model shape
- Potential bug fixes or numerical improvements from the v1 kernel

The exact performance difference depends on hardware and model shape — benchmark both.

---

## 4. Default

```yaml
use_gmm_v2: false
```

V1 is the stable default. V2 is an explicit upgrade path.

---

## 5. Decision tree

```text
use_tokamax_gmm=false →  megablox/JAX backend, use_gmm_v2 irrelevant
use_tokamax_gmm=true  →  use Tokamax
    use_gmm_v2=false  →  Tokamax v1
    use_gmm_v2=true   →  Tokamax v2 (benchmark vs v1 on your hardware)
```

---

### One-line intuition

> **`use_gmm_v2` selects Tokamax's v2 GMM kernel over v1 — only meaningful when `use_tokamax_gmm=True`; benchmark both versions on your hardware and model shape before committing.**
