## 1. Why does `subslice_shape` exist?

In Google Cloud TPU environments using Pathways single-controller architecture, hardware pods are deployed as massive 2D or 3D torus topologies (for example, a full 256-chip Trillium / TPU v6e pod arranged as a $16 \times 16$ mesh).

Developers often want to partition a large dedicated pod into smaller sub-slices for prototyping, multi-tenant jobs, or smaller model runs without physically reconfiguring TPU pod slices.

```text
Full Pod Physical Mesh (16x16 = 256 chips):
┌──────────────────────────────┬──────────────────────────────┐
│ Subslice (8,8) = 64 chips    │ Subslice (8,8) = 64 chips    │
│ [Active Job]                 │ [Available]                  │
├──────────────────────────────┼──────────────────────────────┤
│ Subslice (8,8) = 64 chips    │ Subslice (8,8) = 64 chips    │
│ [Available]                  │ [Available]                  │
└──────────────────────────────┴──────────────────────────────┘
```

`subslice_shape` defines the logical 2D/3D grid dimensions of a sub-slice within a larger Pathways-managed TPU cluster.

---

## 2. Syntax and Formatting

- Format: String of comma-separated dimension integers `"x,y"` or `"x,y,z"`.
- Example: `"8,8"` allocates an $8 \times 8 = 64$ chip subgrid from a $16 \times 16$ physical pod.
- Example: `"4,4,4"` allocates a 64-chip 3D subslice from a 3D TPU v4/v5p mesh.

---

## 3. Options and Defaults

| Value | Behavior |
|---|---|
| `""` (Default: empty string) | Uses the full physical TPU slice allocated to the job |
| `"8,8"` | Partitions an $8 \times 8$ chip subgrid under Pathways |
| `"4,8,2"` | Partitions a 3D coordinate subslice |

---

## 4. Interactions

- **`enable_single_controller`**: Pathways single-controller mode must be active.
- **`mesh_axes` / `ici_data_parallelism`**: Sharding mesh must fit within the specified subslice dimensions ($X \times Y \times Z$).

---

### One-line intuition
> **`subslice_shape` partitions a subgrid (e.g. `"8,8"` for 64 chips) from a larger Pathways TPU pod for flexible multi-tenant allocation.**
