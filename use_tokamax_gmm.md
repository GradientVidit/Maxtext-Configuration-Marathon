
## 1. What Tokamax is

MaxText's default GMM (Grouped Matrix Multiply) implementation for MoE uses MegaBlocks or JAX's built-in ragged-dot primitives. Tokamax is an alternative high-performance grouped matmul library with its own kernel implementations.

The key differentiator: Tokamax's GMM backend supports **tunable tiling for both forward and backward passes** (all 18 `wi_tile_*` / `wo_tile_*` configs), while the default megablox/JAX backend only supports the 6 forward-pass tiling configs.

---

## 2. What it controls

```yaml
use_tokamax_gmm: false  # (default) use megablox/JAX ragged-dot GMM
use_tokamax_gmm: true   # use Tokamax library GMM kernel
```

When `true`, all MoE expert matmuls route through Tokamax's GMM kernel instead of the default backend.

---

## 3. When Tokamax matters

**Forward pass only:** megablox/JAX is often competitive. The 6 forward tiling params are sufficient.

**Backward pass tuning:** only Tokamax supports the `dlhs`/`drhs` backward-pass tile configs. If backward pass GMM is the bottleneck (which matters for training), Tokamax is required to tune it.

**Hardware-specific performance:** Tokamax kernels may be tuned for specific TPU generations in ways that outperform the generic JAX ragged-dot.

---

## 4. Default

```yaml
use_tokamax_gmm: false
```

The default megablox/JAX path is the stable, well-tested option. Tokamax is an explicit opt-in for performance optimization.

---

## 5. Interaction with `use_gmm_v2`

```yaml
use_gmm_v2: false  # use Tokamax GMM v1 (when use_tokamax_gmm=True)
use_gmm_v2: true   # use Tokamax GMM v2 kernel
```

`use_gmm_v2` is only meaningful when `use_tokamax_gmm=True`. V2 is a newer Tokamax kernel implementation, potentially with different performance characteristics.

---

## 6. Options

| `use_tokamax_gmm` | `use_gmm_v2` | Effective kernel |
|---|---|---|
| `false` | Any | megablox/JAX ragged-dot (default) |
| `true` | `false` | Tokamax GMM v1 |
| `true` | `true` | Tokamax GMM v2 |

---

### One-line intuition

> **`use_tokamax_gmm` switches MoE expert matmuls from the default megablox/JAX ragged-dot to Tokamax's GMM kernel — the main reason to switch is backward-pass tiling control (dlhs/drhs configs) which only Tokamax supports.**
