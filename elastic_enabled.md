## 1. Why does `elastic_enabled` exist?

Large-scale pretraining across thousands of TPU chips typically spans multiple multi-host pods or slices.

In standard static distributed execution (McJAX / static mesh), if a single slice experiences a hardware glitch, preemption, or network partition:
- The entire distributed JAX runtime crashes.
- All healthy slices sit idle.
- The job orchestrator must tear down the entire cluster, request a full set of replacement nodes, re-compile the computational graph, and restore the last global checkpoint from GCS.

```text
Static Multi-Slice Failure:
Slice 0 (Healthy) ───┐
Slice 1 (Healthy) ───┼──> Slice 2 Fails ──> ENTIRE JOB CRASHES ──> Cluster Teardown ──> Cold Restart
Slice 2 (FAILS)   ───┘

Elastic Training (elastic_enabled: true with Pathways):
Slice 0 (Healthy) ───┐
Slice 1 (Healthy) ───┼──> Slice 2 Fails ──> Coordinator detects loss ──> Resizes mesh to 2 slices ──> Training Continues!
Slice 2 (FAILS)   ───┘
```

**Elastic Training** enables fault-tolerant dynamic execution under Google Pathways (Single Controller mode), allowing a training job to dynamically continue with surviving slices if some slices fail or are preempted.

`elastic_enabled` is the master boolean switch that activates elastic training fault tolerance in MaxText.

---

## 2. Mechanics & Strict Prerequisites

> [!IMPORTANT]
> Elastic training is **Pathways-specific** (Single Controller mode) and does **not** work on McJAX (multi-controller JAX).

When `elastic_enabled: true`:
1. The centralized Pathways coordinator monitors heartbeat signals across all active TPU slices.
2. In the background, periodic state snapshots are preserved via `elastic_backup_kind`.
3. If a slice drops out, the coordinator waits up to `elastic_timeout_seconds` before isolating the dead slice, re-sharding the mesh across the remaining slices, restoring the latest in-memory state snapshot, and resuming the training loop.

---

## 3. Options & Default

| Parameter | Type | Default | Options |
| :--- | :--- | :--- | :--- |
| `elastic_enabled` | `bool` | `false` | `true` (enable Pathways elastic fault tolerance), `false` (standard static execution) |

---

## 4. Interactions with Related Parameters

| Related Parameter | Interaction |
| :--- | :--- |
| `grain_use_elastic_iterator` | Must be set to `true` when elastic training is enabled so data pipelines can dynamically re-coordinate worker streams. |
| `packing` | Must be `false` when `grain_use_elastic_iterator: true`. |
| `elastic_min_slice_count` | Enforces a lower bound on how small the surviving cluster can shrink before aborting. |
| `elastic_timeout_seconds` | Wait duration before treating an unresponsive slice as permanently lost. |
| `elastic_max_retries` | Maximum number of allowable slice drop/recovery events. |

---

## 5. Practical Guidance & Failure Modes

| Scenario | Symptom / Failure | Solution |
| :--- | :--- | :--- |
| `elastic_enabled: true` on McJAX | Raises initialization error; McJAX does not support dynamic slice reconfiguration. | Use Pathways single-controller execution or set `elastic_enabled: false`. |
| Elastic Training with `packing: true` | Iterator checkpoint restoration error during slice shrinkage. | Set `packing: false`. |
| Spot Preemption on Large Pods | Slices drop dynamically, job automatically continues on remaining capacity without GCS restart delay. | Keep `elastic_enabled: true` for large spot TPU runs. |

---

### One-line intuition

> `elastic_enabled` is the master toggle for Pathways-based elastic training, allowing MaxText jobs to dynamically survive TPU slice failures without crashing the run.
