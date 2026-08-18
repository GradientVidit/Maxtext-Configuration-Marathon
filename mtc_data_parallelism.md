## 1. Why does it exist?

In distributed multi-slice architectures, the physical TPU cluster is partitioned into multiple identical replicas running data parallelism across slices (DCN) and within slices (ICI). When Multi-Tier Checkpointing (MTC) performs a **fast local restore**, workers need to know how many identical parallel pipelines exist so that:
1. Slices holding valid local checkpoints can broadcast identical model weights to sibling replicas.
2. The checkpointer avoids redundant parallel reads and coordinate collective transfers cleanly across the mesh.

```text
Multi-Slice Cluster:
  Slice 0 (Replica 0) ───[Holds Local Checkpoint]───┐
                                                    │ Broadcast across DCN
  Slice 1 (Replica 1) ◄─────────────────────────────┤
  Slice 2 (Replica 2) ◄─────────────────────────────┤
  Slice 3 (Replica 3) ◄─────────────────────────────┘
  (Total Identical Data Parallel Pipelines = mtc_data_parallelism)
```

`mtc_data_parallelism` defines the total degree of data parallelism participating in Multi-Tier Checkpointing.

---

## 2. Fundamentals & Mathematics

In MaxText's device mesh, total data parallelism is the product of intra-slice (ICI) data parallelism and inter-slice (DCN) data parallelism:

$$\text{MTC Data Parallelism} = \text{ICI Data Parallelism} \times \text{DCN Data Parallelism}$$

When left at its default value (`0`), MaxText automatically computes this degree from the configured cluster geometry:
- In multi-slice training, it defaults to the total number of TPU slices (`num_slices` / DCN data degree).

---

## 3. Options & Configuration

| Value | Meaning |
|---|---|
| `0` (default) | Auto-detect: MaxText calculates `mtc_data_parallelism` based on the active mesh and slice count. |
| Integer $> 0$ | Explicitly sets the number of identical data-parallel pipelines. |

Default in `base.yml`:
```yaml
mtc_data_parallelism: 0
```

---

## 4. Interactions with Mesh Parameters

```text
dcn_data_parallelism ──┐
                       ├──→ Total Data Parallel Replicas ──→ mtc_data_parallelism (if 0, auto-set)
ici_data_parallelism ──┘
```

- **`enable_multi_tier_checkpointing`**: Only utilized when MTC is enabled.
- **`num_slices`**: In standard setups where each slice represents one DCN data-parallel replica, `mtc_data_parallelism` matches `num_slices`.

---

## 5. Practical Usage

- **Keep Default (`0`)**: In almost all production setups on GKE and TPU multislice, leaving `mtc_data_parallelism: 0` allows MaxText to automatically extract the correct parallelism degree from JAX device topology.
- **When to Override**: Only override if running specialized custom mesh geometries where data-parallel replicas span fractional slices or non-standard hybrid topologies.

---

### One-line intuition

> **`mtc_data_parallelism` specifies the total number of identical data-parallel model replicas in the cluster to coordinate peer-to-peer local checkpoint broadcast upon recovery.**
