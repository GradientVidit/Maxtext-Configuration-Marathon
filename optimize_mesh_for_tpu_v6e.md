## 1. Why does it exist?

**TPU v6e (Trillium)** introduces a revised physical chip packaging and interconnect topology compared to previous generations (TPU v4/v5e/v5p). Its intra-node and inter-node optical routing links have specific bandwidth ratios and torus characteristics.

Mapping standard legacy device meshes onto TPU v6e without taking its physical packaging into account can result in suboptimal collective communication paths for all-gather and reduce-scatter operations.

`optimize_mesh_for_tpu_v6e` applies hardware-specific mesh coordinate transformations tuned explicitly for the TPU v6e (Trillium) interconnect topology.

```text
Without optimize_mesh_for_tpu_v6e:
  Generic mesh mapping ──→ Collective communication may traverse sub-optimal v6e inter-tray links.

With optimize_mesh_for_tpu_v6e: true:
  Reorders mesh coordinates to align with Trillium's physical trays and ICI links ──→ Maximum bandwidth utilization.
```

---

## 2. Options & Configuration

| Value | Meaning |
|---|---|
| `false` (default) | Standard generic mesh generation. |
| `true` | Applies Trillium-specific (TPU v6e) mesh optimizations. |

Default in `base.yml`:
```yaml
optimize_mesh_for_tpu_v6e: false
```

---

## 3. Practical Usage Guidelines

- **On TPU v6e (Trillium)**: Recommended to set `optimize_mesh_for_tpu_v6e: true` when running on v6e pod slices (especially large multi-host slices) to ensure collective operations achieve peak interconnect throughput.
- **On TPU v4 / v5e / v5p / GPUs**: Leave `false`.

---

### One-line intuition

> **`optimize_mesh_for_tpu_v6e` reorders device mesh axes to align with TPU v6e (Trillium) physical tray topologies for maximum interconnect efficiency.**
