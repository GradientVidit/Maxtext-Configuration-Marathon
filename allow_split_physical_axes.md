## 1. Why does it exist?

When JAX constructs a multi-dimensional device mesh using `jax.experimental.mesh_utils.create_device_mesh`, it attempts to map each logical mesh axis (e.g. `fsdp: 8`, `tensor: 4`) directly onto a contiguous physical dimension of the TPU accelerator hardware torus.

On certain hardware topologies, the physical torus dimensions may not match the requested logical axis sizes (e.g. a physical torus dimension of size $16$ needs to be split across two logical axes of sizes $4$ and $4$).

By default, `create_device_mesh` forbids splitting a single physical torus dimension across multiple logical axes to prevent non-contiguous communication patterns. Enabling `allow_split_physical_axes` lifts this restriction.

```text
Physical Torus Dimension = 16 chips
Requested Logical Axes = (fsdp: 4, tensor: 4)

With allow_split_physical_axes: false:
  Error / Restriction: Cannot map (4, 4) cleanly to physical dim 16 without splitting.

With allow_split_physical_axes: true:
  JAX splits the physical 16-chip dimension into two logical axes of 4 and 4.
```

`allow_split_physical_axes` allows JAX's mesh creation utilities to split physical device torus dimensions when constructing the logical mesh.

---

## 2. Options & Configuration

| Value | Meaning |
|---|---|
| `false` (default) | Strict contiguous mapping; physical torus axes are not split. |
| `true` | Allows JAX to factorize/split physical device axes to accommodate non-standard logical mesh shapes. |

Default in `base.yml`:
```yaml
allow_split_physical_axes: false
```

---

## 3. Interactions with Mesh Parameters

- Passes directly to `jax.experimental.mesh_utils.create_device_mesh(..., allow_split_physical_axes=allow_split_physical_axes)`.
- Useful when experimenting with custom parallelism degrees (e.g. odd combinations of FSDP, TP, and EP) that do not align 1:1 with physical hardware dimensions.

---

### One-line intuition

> **`allow_split_physical_axes` enables JAX to divide physical hardware torus axes across multiple logical mesh dimensions when exact 1:1 dimension matching is impossible.**
