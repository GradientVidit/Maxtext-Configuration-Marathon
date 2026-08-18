## 1. Why does it exist?

A TPU "Slice" is an autonomous, physically contiguous pod of TPU chips connected via high-speed optical torus interconnects (e.g., a `v5p-1024` slice contains 1,024 chips). Multi-slice training pools multiple independent slices together over a Data Center Network (DCN).

MaxText needs to know the total number of slices to configure DCN mesh axes (`dcn_*_parallelism`), setup distributed rendezvous, and calculate multi-tier checkpointing topologies.

```text
Cluster Layout:
  Slice 0 (v5p-1024) ──┐
  Slice 1 (v5p-1024) ──┼──[ Data Center Network (DCN) ]
  Slice 2 (v5p-1024) ──┤
  Slice 3 (v5p-1024) ──┘
  ──→ Total Slices = num_slices: 4
```

`num_slices` specifies the total count of TPU pod slices participating in the training job.

---

## 2. Fundamentals & Auto-Detection

- **Live Cloud Runs**: In standard live multi-host training on Google Cloud / GKE, `num_slices: -1` causes MaxText to automatically query the TPU runtime metadata and infer the active slice count.
- **AOT Compilation**: During Ahead-of-Time compilation, setting `compile_topology_num_slices` automatically sets `num_slices`.
- **Non-TPU Platforms**: On GPU and CPU backends, `num_slices` is automatically fixed to `1`.

---

## 3. Options & Configuration

| Value | Meaning |
|---|---|
| `-1` (default) | Automatically inferred from cluster hardware runtime or AOT compilation config. |
| Integer $\ge 1$ | Explicitly overrides the number of TPU slices. |

Default in `base.yml`:
```yaml
num_slices: -1
```

---

## 4. When to Manually Override

- **Disaggregated Reinforcement Learning (RL)**: In disaggregated RL workloads (where separate slices run actor environments while other slices run learner gradient updates), manually setting `num_slices` partitions the cluster between roles.
- **AOT Mocking**: When compiling graphs offline for a multislice cluster.

---

### One-line intuition

> **`num_slices` defines the total number of TPU pod slices in the cluster, automatically discovered at runtime or set explicitly for disaggregated RL and AOT compilation.**
