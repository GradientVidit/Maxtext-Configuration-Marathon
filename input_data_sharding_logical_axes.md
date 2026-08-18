## 1. Why does it exist?

When an input batch array with shape `[batch_size, seq_len]` is prepared by MaxText's dataset loader, JAX needs to bind the batch dimension and sequence dimension to logical axis names so that the input sharding can be resolved against `logical_axis_rules`.

Without explicit logical axis names for input tensors, the input pipeline wouldn't know whether dimension 0 is a standard batch dimension or an embedding batch dimension, or whether dimension 1 is a regular sequence length or a normalized sequence length.

```text
Raw Dataset Tensor: shape [B, S]
         │
  [ input_data_sharding_logical_axes ]
         │
  Dimension 0 ──→ 'activation_embed_and_logits_batch'
  Dimension 1 ──→ 'activation_norm_length'
         │
  Mapped via `logical_axis_rules` to physical mesh axes
```

`input_data_sharding_logical_axes` specifies the logical axis names assigned to the dimensions of the input batch array.

---

## 2. Fundamentals & Defaults

Default in `base.yml`:
```yaml
input_data_sharding_logical_axes: ['activation_embed_and_logits_batch', 'activation_norm_length']
```

- **Index 0 (`'activation_embed_and_logits_batch'`):** Represents the batch dimension of the inputs. In `logical_axis_rules`, this logical name is mapped to `data`, `fsdp`, and `context` axes so the batch is split evenly across all data-parallel devices.
- **Index 1 (`'activation_norm_length'`):** Represents the sequence length dimension. In `logical_axis_rules`, this is mapped to `context` and `tensor_sequence` axes so sequences can be split when context parallelism is enabled.

---

## 3. Options & Configuration

| Parameter | Type | Default |
|---|---|---|
| `input_data_sharding_logical_axes` | `list of str` | `['activation_embed_and_logits_batch', 'activation_norm_length']` |

---

## 4. Interactions with Related Parameters

```text
input_data_sharding_logical_axes: ['activation_embed_and_logits_batch', 'activation_norm_length']
                       │
             Resolved against `logical_axis_rules`:
                       │
  'activation_embed_and_logits_batch' ──→ ('data', 'fsdp', 'context')
  'activation_norm_length'            ──→ ('context', 'tensor_sequence')
```

- **`data_sharding`**: Pairs with these logical axis names to build the input JAX `NamedSharding` / `PartitionSpec`.
- **`context_parallel_strategy`**: When context parallelism is enabled, the second logical axis (`activation_norm_length`) is partitioned across the context mesh dimension.

---

### One-line intuition

> **`input_data_sharding_logical_axes` assigns semantic logical names to the input tensor's batch and sequence dimensions so the data loader can correctly shard input batches over data- and context-parallel axes.**
