## 1. Why does it exist?

By default, JAX constructs logical device meshes by assuming a standard multidimensional grid mapped onto available physical devices. However, on complex hardware topologies (such as 3D TPU torus rings or 2D twisted meshes), mapping communication-heavy dimensions (like Expert Parallelism All-to-All or Ring Attention) arbitrarily onto the hardware can cause collective traffic to cross slow diagonal links or create network hop congestion.

**`custom_mesh`** allows selecting predefined physical mesh layouts optimized for specific interconnect communication patterns.

```text
Default Flat Mesh Mapping:
  Physical Chips mapped in linear order ──→ Multiple hops for ring communication.

Custom Hybrid Ring Mesh (`custom_mesh: 'hybrid_ring_64x4'`):
  Reorders physical device coordinates into a 64x4 hybrid ring ──→ Minimum optical hops!
```

`custom_mesh` selects a predefined physical mesh topology (e.g. `'hybrid_ring_64x4'`, `'hybrid_ring_32x8'`) optimized for specific accelerator interconnect geometries.

---

## 2. Options & Configuration

| Value | Meaning |
|---|---|
| `""` (default) | Default logical mesh construction via standard JAX `create_device_mesh`. |
| `'hybrid_ring_64x4'` | Predefined hybrid 64x4 ring topology for 256-chip slices. |
| `'hybrid_ring_32x8'` | Predefined hybrid 32x8 ring topology for 256-chip slices. |

Default in `base.yml`:
```yaml
custom_mesh: "" # Available options: ['hybrid_ring_64x4', 'hybrid_ring_32x8']
```

---

## 3. Practical Usage Guidelines

- **When to Use**: Used primarily when profiling shows network link contention during heavy MoE Expert Parallelism or Ring Attention on 256-chip TPU pod slices.

---

### One-line intuition

> **`custom_mesh` applies a predefined hardware-optimized physical device ordering (like `hybrid_ring_64x4`) to minimize communication hops in high-throughput ring collectives.**
