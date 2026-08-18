## 1. Why it exists: inspecting tensor sharding and logical-to-physical mesh bindings

In large-scale JAX models, distributed execution relies on **SPMD (Single Program, Multiple Data) sharding**. In MaxText, model architectures are decoupled from physical hardware meshes by using **logical axis names** (e.g. `activation_embed_and_logits_batch`, `activation_length`, `mlp`, `heads`, `embed`) rather than hardcoded physical hardware mesh axes (`data`, `fsdp`, `tensor`):

```text
Logical Model Definition (Architecture Layer)
           │
           │ `logical_axis_rules` Mapping
           ▼
Physical Device Mesh (Hardware Topology: e.g. [fsdp: 4, tensor: 8])
           │
           ▼
JAX NamedSharding & PartitionSpec Generation
```

During model execution and layer instantiation, MaxText wraps tensor partitioning inside `sharding.maybe_shard_with_logical(...)`:

```python
xent = sharding.maybe_shard_with_logical(
    xent,
    ("activation_embed_and_logits_batch", "activation_length"),
    model.mesh,
    config.shard_mode,
    debug_sharding=config.debug_sharding,
)
```

A common bug during model bring-up or custom architecture modification is **unintended tensor replication**:
- If an axis rule contains a typo or misses a dimension, a multi-gigabyte weight or activation tensor is silently replicated across all devices instead of sharded.
- This causes immediate Out-Of-Memory (OOM) crashes during compilation or introduces unneeded multi-gigabyte all-gather communications that devastate step time.

`debug_sharding` is the diagnostic toggle that forces `maybe_shard_with_logical` and parameter initialization routines to log the exact `NamedSharding`, `PartitionSpec`, global shape, and per-device local shard shapes directly to `stdout`.

---

## 2. Mechanics: PyTree parameter and activation sharding inspection

When `debug_sharding: true`, MaxText activates diagnostic hooks across parameter trees and layer activations:

```text
 Model Initialization & Layer Execution
                   │
                   ▼
 Check: `config.debug_sharding: true`
                   │
                   ▼
 ┌───────────────────────────────────────────────────────────────────────┐
 │ Traverse Parameter PyTree & Layer Activations:                        │
 │ For each tensor (e.g. 'decoder/layers_0/mlp/wi_0', 'xent'):          │
 │ 1. Query logical axes (e.g. ('fsdp', 'tensor'))                       │
 │ 2. Resolve `NamedSharding` against `model.mesh`                       │
 │ 3. Inspect global shape and calculate local per-chip shard shape      │
 │ 4. Print diagnostic report to `stdout` (Host 0)                       │
 └───────────────────────────────────┬───────────────────────────────────┘
                                     │
                                     ▼
                      Diagnostic Table Printed:
 ┌───────────────────────────────────────────────────────────────────────┐
 │ Tensor: 'decoder/layers_0/mlp/wi_0'                                   │
 │ Global Shape: (4096, 14336)                                           │
 │ Sharding: NamedSharding(mesh, PartitionSpec(('fsdp',), ('tensor',)))  │
 │ Per-Device Local Shard Shape: (1024, 1792)                            │
 │ Memory per Device: 3.67 MB                                            │
 └───────────────────────────────────────────────────────────────────────┘
```

The diagnostic output highlights:
- Which tensor dimensions are partitioned along `fsdp`, `tensor`, `data`, `expert`, or `pipeline` mesh axes.
- Which tensor dimensions are unreplicated or replicated across the mesh (represented as `PartitionSpec(None, ...)`).

---

## 3. Options & Configuration

Default in `base.yml`:

```yaml
debug_sharding: false
```

| Value | Behavior | stdout Output Volume | Recommended Use Case |
|---|---|---|---|
| `false` (default) | Parameter and activation sharding details are not printed. | Minimal. | Standard production runs, pretraining, and daily inference. |
| `true` | Prints full `NamedSharding` and `PartitionSpec` details for every tensor and layer activation. | High (hundreds of lines detailing every tensor). | **Model bring-up, debugging OOMs, verifying new parallelism strategies (FSDP, TP, EP, Ulysses).** |

---

## 4. Parameter Interactions

```text
┌───────────────────────────────────────────────────────────┐
│                      debug_sharding                       │
└─────────────┬───────────────────────────────┬─────────────┘
              │ (when true)
              ▼
┌───────────────────────────────────────────────────────────┐
│ Directly inspects the output of:                          │
│ - logical_axis_rules & override_logical_axis_rules        │
│ - Hardware parallelism dims (ici_tensor_parallelism, etc.)│
│ - shardy (Shardy vs GSPMD partitioning specifications)    │
│ - moe_dispatch_no_expert_sharding                         │
└───────────────────────────────────────────────────────────┘
```

- **`logical_axis_rules`**: The rules that define how tensor logical dimensions map to mesh axes. `debug_sharding` confirms whether your rules were applied correctly.
- **`shardy`**: Whether Shardy (JAX 0.7+) or legacy GSPMD is generating the partition layouts.
- **`shard_mode`**: Governs whether sharding constraints are applied via explicit layout annotations or constraint primitives.

---

## 5. Practical Scenarios & Failure Modes

### Debugging Unexpected Out-Of-Memory (OOM) Errors
If a 70B model crashes with an OOM on a 64-chip TPU slice:
```bash
python3 src/maxtext/train.py src/maxtext/configs/base.yml \
  model_name=llama3-70b \
  debug_sharding=true
```
Inspecting the output reveals:
```text
Tensor: decoder/layers_0/attn/q_proj -> Sharding: PartitionSpec(None, None)  <-- BUG: Replicated!
```
The query projection was missing from `logical_axis_rules`, causing full replication. Fixing the rule immediately shards the tensor and resolves the OOM.

---

### One-line intuition

> **`debug_sharding` logs the exact JAX `NamedSharding` and per-device shard shapes resolved by `maybe_shard_with_logical`, allowing developers to catch unintended weight replication and sharding bugs before execution begins.**
