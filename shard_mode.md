## 1. Why does it exist?

In distributed JAX, there are two primary paradigms for expressing how arrays and computations are partitioned across accelerator devices:

1. **Automatic Sharding (`"auto"`)**: You specify sharding constraints at program boundaries (inputs, outputs, and intermediate logical axis rules via `logical_axis_rules`), and the XLA compiler backend (GSPMD or Shardy) automatically propagates shardings through all intermediate operations using cost-based graph analysis.
2. **Explicit Sharding (`"explicit"`)**: You take full manual control over every collective and tensor partition using `jax.experimental.shard_map` (`shard_map`), explicitly mapping sub-arrays to per-device kernels.

```text
shard_mode: "auto" (SPMD / Sharding Propagation)
  Logical Axis Rules ──→ JAX PartitionSpec ──→ XLA Compiler Infers All Collectives & Intermediates

shard_mode: "explicit" (shard_map / Manual SPMD)
  Developer defines exact per-device functions and manual collective communications
```

`shard_mode` chooses between high-level automatic compiler sharding propagation and low-level explicit `shard_map` execution.

---

## 2. Fundamentals & Mechanics

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                           shard_mode: "auto"                            │
│  - Default mode across all modern MaxText model architectures.          │
│  - Declarative: Define logical axis rules once.                         │
│  - XLA optimizer places AllGather, ReduceScatter, and AllToAll ops.    │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                         shard_mode: "explicit"                          │
│  - Bypasses automatic compiler sharding propagation.                    │
│  - Relies on explicit `shard_map` wrapping of layer sub-graphs.         │
│  - Used for custom kernel research or low-level collective debugging.   │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Options & Configuration

| Value | Behavior | Recommended Use Case |
|---|---|---|
| `"auto"` (default) | Automatic sharding propagation via XLA GSPMD / Shardy. | Production training and serving for all standard models. |
| `"explicit"` | Explicit SPMD sharding using `shard_map`. | Low-level kernel development and custom communication experiments. |

Default in `base.yml`:
```yaml
shard_mode: "auto"
```

---

## 4. Interactions with Related Parameters

- **`shardy`**: When `shard_mode: "auto"`, `shardy: true` uses the newer MLIR-based Shardy propagation engine, while `shardy: false` uses legacy GSPMD.
- **`logical_axis_rules`**: Directly consumed by `"auto"` mode to assign `PartitionSpec` to every tensor.
- **`check_vma`**: Requires `shard_mode: "auto"`.

---

## 5. Practical Recommendations

- **Standard Training**: Always use `"auto"`. It allows MaxText to seamlessly scale identical model code from a single TPU chip to a 65,000-chip pod slice.
- **When to Avoid `"explicit"`**: Explicit mode requires custom manual handling of weight gathering and sharding layouts for every layer, and is primarily reserved for specialized research kernels.

---

### One-line intuition

> **`shard_mode` determines whether tensor sharding across the device mesh is resolved automatically by XLA compiler propagation (`"auto"`) or defined manually via explicit `shard_map` blocks (`"explicit"`).**
