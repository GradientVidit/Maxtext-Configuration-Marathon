## 1. Why does it exist?

To distribute a neural network across thousands of accelerator chips, JAX constructs an N-dimensional **Logical Device Mesh** (`jax.sharding.Mesh`). Every physical chip in the cluster is assigned a coordinate along named dimensions.

Without named physical axes, sharding code would have to deal directly with raw integer device IDs or flat physical hardware rings. Named mesh axes provide the abstraction layer: you declare what orthogonal dimensions exist in your cluster (e.g. data parallelism, tensor parallelism, pipeline stages, expert groups, sequence shards), and then bind tensors to those named dimensions.

```text
Physical Accelerator Cluster (e.g. 2048 chips)
                      │
           Mapped into Logical Mesh
                      │
           mesh_axes: ['data', 'fsdp', 'tensor', 'expert', ...]
                      │
         Coordinate per device: (data=0, fsdp=3, tensor=1, expert=0)
```

`mesh_axes` defines the full ordered list of named physical axis dimensions that compose MaxText's multi-dimensional device mesh.

---

## 2. Fundamentals & The Default Axis Spectrum

MaxText defines an expressive list of 12 physical mesh axes by default in `base.yml`:

```yaml
mesh_axes: [
  'diloco',
  'data',
  'stage',
  'fsdp',
  'fsdp_transpose',
  'context',
  'context_usp_ulysses',
  'context_autoregressive',
  'tensor',
  'tensor_sequence',
  'expert',
  'autoregressive'
]
```

### Purpose of Each Physical Axis:
1. **`diloco`**: Distributed Low-Communication training axis (for asynchronous cross-cluster optimization).
2. **`data`**: Pure data parallelism (replicated weights, partitioned batch).
3. **`stage`**: Pipeline parallelism stage axis.
4. **`fsdp`**: Fully Sharded Data Parallelism (ZeRO-3 style parameter/gradient/optimizer sharding).
5. **`fsdp_transpose`**: Orthogonal 2D-FSDP transpose axis for 2D parameter sharding.
6. **`context`**: Context parallelism (splitting the sequence dimension of activations across chips).
7. **`context_usp_ulysses`**: All-to-all Ulysses head-exchange dimension in Unified Sequence Parallelism (USP).
8. **`context_autoregressive`**: Autoregressive causal sequence parallelism dimension.
9. **`tensor`**: Megatron-style intra-layer Tensor Parallelism (splitting attention heads / MLP hidden dims).
10. **`tensor_sequence`**: Sequence-parallel variant of tensor parallelism (splitting along sequence in TP blocks).
11. **`expert`**: Expert Parallelism (routing different MoE experts to different devices).
12. **`autoregressive`**: Dedicated autoregressive decoding axis.

---

## 3. How Axes are Sized (ICI vs. DCN)

The size of each named axis is determined by the product of its intra-slice (`ici_*_parallelism`) and inter-slice (`dcn_*_parallelism`) values:

$$\text{Axis Size} = \text{ici\_axis\_size} \times \text{dcn\_axis\_size}$$

```text
             mesh_axes: ['data', 'fsdp', 'tensor', 'expert', ...]
                             │
     ┌───────────────────────┴───────────────────────┐
     ↓                                               ↓
ici_*_parallelism                               dcn_*_parallelism
(Within fast ICI chip interconnect)             (Across datacenter network)
```

---

## 4. Interactions with `logical_axis_rules`

`mesh_axes` supplies the vocabulary of physical dimensions. `logical_axis_rules` maps logical model dimensions onto this vocabulary:

```text
Logical Tensor Name: 'activation_batch' ──→ Mapped to: ('data', 'fsdp')
Logical Tensor Name: 'heads'            ──→ Mapped to: ('tensor',)
Logical Tensor Name: 'expert'           ──→ Mapped to: ('expert',)
```

If a physical axis is omitted from `mesh_axes`, rules referencing that axis cannot resolve.

---

## 5. Practical Guidelines

- **Modification**: You rarely need to modify `mesh_axes` directly unless you are inventing a brand-new parallelism paradigm or creating a custom mesh preset in `configs/mesh_and_rule/`.
- **Order of Axes**: The ordering in `mesh_axes` determines how physical device coordinates are unpacked from multidimensional mesh arrays.

---

### One-line intuition

> **`mesh_axes` declares the complete catalog of named orthogonal dimensions (e.g. data, fsdp, tensor, expert, context) that form MaxText's multi-dimensional device mesh.**
