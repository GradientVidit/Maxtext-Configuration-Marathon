## 1. Why does it exist?

MaxText's device mesh defines 12 named physical axes (e.g. `['diloco', 'data', 'stage', 'fsdp', 'fsdp_transpose', 'context', 'context_usp_ulysses', 'context_autoregressive', 'tensor', 'tensor_sequence', 'expert', 'autoregressive']`).

In any given training run, only 1, 2, or 3 of these axes are active (size $> 1$), while the remaining 9+ axes have size `1`.

In JAX's internal type system, including size-1 mesh axes inside tensor type signatures (e.g. `Sharding(mesh, ('fsdp', 'data', 'tensor', ...))`) bloats JAXPR graph representations, complicates compiler pattern matching, and adds unnecessary metadata overhead.

```text
Without removal (remove_size_one_mesh_axis_from_type: false):
  Tensor Sharding Type: NamedSharding(mesh=(data:1, fsdp:8, tensor:1, expert:1, ...))

With removal (remove_size_one_mesh_axis_from_type: true):
  Tensor Sharding Type: NamedSharding(mesh=(fsdp:8))  <-- Clean, stripped type signature!
```

`remove_size_one_mesh_axis_from_type` configures `jax.config.jax_remove_size_one_mesh_axis_from_type` to strip inactive size-1 mesh dimensions from tensor type representations.

---

## 2. Options & Configuration

| Value | Meaning |
|---|---|
| `true` (default) | Strips size-1 axes from JAX types for cleaner JAXPRs and faster compilation. |
| `false` | Retains full 12-axis type signatures in JAX types. |

Default in `base.yml`:
```yaml
remove_size_one_mesh_axis_from_type: true # Whether to remove size one mesh axis from type through jax.config.
```

---

### One-line intuition

> **`remove_size_one_mesh_axis_from_type` strips unused size-1 axes from JAX tensor sharding types, streamlining internal graph representation and compiler analysis.**
