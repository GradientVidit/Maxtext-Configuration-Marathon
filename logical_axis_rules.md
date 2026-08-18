## 1. Why does it exist?

In traditional distributed frameworks, model code is tightly coupled to specific parallelism strategies: if you want Megatron tensor parallelism, you write custom column-parallel and row-parallel linear layers with manual all-reduces. If you want FSDP, you wrap layers in specialized PyTorch hooks. If you want to switch from pure FSDP to hybrid Tensor+Expert+Context parallelism, you must rewrite the model layers.

MaxText solves this with a clean two-level indirection:
1. **Model code only names semantic dimensions**: Layers define their weights and activations with *logical names* (e.g. `'embed'`, `'heads'`, `'mlp'`, `'activation_batch'`, `'activation_length'`, `'expert'`).
2. **`logical_axis_rules` maps logical names to physical mesh axes**: A centralized table maps each logical dimension to zero, one, or more physical mesh axes.

```text
Model Layer Definition (Semantic / Hardware-Agnostic):
  Linear(inputs, name_axes=('activation_batch', 'embed')) ──→ weights('embed', 'mlp')
                                  │
                       [ logical_axis_rules ]
                                  │
    ┌─────────────────────────────┴─────────────────────────────┐
    ↓                                                           ↓
Strategy A (Pure FSDP):                     Strategy B (Hybrid TP + EP + CP):
  'activation_batch' ──→ ('fsdp',)             'activation_batch' ──→ ('data',)
  'embed'            ──→ ('fsdp_transpose',)   'heads'            ──→ ('tensor',)
  'mlp'              ──→ ()                    'expert'           ──→ ('expert',)
```

`logical_axis_rules` is the global routing table that determines the concrete physical sharding of every weight and activation tensor in MaxText without altering a single line of model architecture code.

---

## 2. Fundamentals & Rule Structure

Each entry in `logical_axis_rules` is a pair: `[logical_axis_name, [physical_mesh_axes...]]`:

```text
['activation_batch', ['data', 'fsdp', 'context']]
       ▲                               ▲
       │                               │
Logical Dimension Name          Ordered list of Physical Mesh Axes
(from Flax Linen model code)    participating in sharding this dimension
```

### Key Functional Categories in the Table:

1. **Vocabulary Embedding**:
   - `['vocab', ['data', 'fsdp_transpose', 'tensor']]`
   - `['embed', ['fsdp', 'tensor_sequence']]`
2. **Attention Projections & Heads**:
   - `['heads', ['tensor', 'tensor_sequence', 'autoregressive']]`
   - `['kv_heads', ['tensor', 'tensor_sequence', 'autoregressive']]`
   - `['kv_head_dim', []]` (Unsharded / fully replicated)
3. **Dense & MoE MLPs**:
   - `['mlp', ['fsdp_transpose', 'tensor', 'tensor_sequence']]`
   - `['expert', ['expert']]`
4. **Activations (Forward Pass Flow)**:
   - `['activation_batch', ['data', 'fsdp', 'context']]`
   - `['activation_length', ['context', 'tensor_sequence']]`
5. **Inference & KV Cache**:
   - `['cache_batch', ['data', 'fsdp', 'context']]`
   - `['cache_heads', ['tensor', 'tensor_sequence']]`

---

## 3. Fallback & Rule Resolution Mechanics

MaxText's axis rule parser evaluates rules sequentially:
- If a logical rule lists physical axes `['tensor', 'tensor_sequence']`, MaxText checks whether `tensor` or `tensor_sequence` mesh axes have a size $> 1$ in the active hardware mesh.
- If multiple rules exist for the same logical axis, the first rule whose physical axes are actively present in the mesh is chosen.
- If a logical axis is mapped to `[]` (empty list), that dimension is left unsharded (replicated across all devices).

```text
Mesh Configured: ici_fsdp_parallelism=8, ici_tensor_parallelism=1

Evaluate Rule: ['mlp', ['tensor', 'fsdp_transpose']]
  - 'tensor' size is 1 (inactive)
  - 'fsdp_transpose' size is 1 (inactive)
  ──→ Result: 'mlp' dimension remains replicated on device.

Evaluate Rule: ['activation_batch', ['data', 'fsdp']]
  - 'data' size is 1
  - 'fsdp' size is 8 (active!)
  ──→ Result: 'activation_batch' is sharded across the 8 FSDP devices.
```

---

## 4. Interactions with Other Parameters

- **`override_logical_axis_rules`**: When `false` (default), CLI-passed rules merge with this table. When `true`, CLI rules replace the entire table.
- **`custom_mesh_and_rule`**: Overrides this entire table with a named preset from `configs/mesh_and_rule/`.
- **`logical_axis_rules_for_eval`**: Specifies an alternate table applied specifically during evaluation.

---

## 5. Practical Tuning Examples

- **Enabling Megatron Tensor Parallelism**:
  Ensure `heads` and `mlp` rules contain `tensor`, then set `ici_tensor_parallelism: 8`.
- **Enabling Context Parallelism**:
  Ensure `activation_length` and `activation_norm_length` contain `context`, then set `ici_context_parallelism: 4`.

---

### One-line intuition

> **`logical_axis_rules` is MaxText's central routing table that translates semantic model dimensions (like heads, batch, mlp, and embed) into concrete physical mesh partitions across your TPU/GPU pod.**
