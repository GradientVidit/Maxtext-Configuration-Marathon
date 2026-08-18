## 1. Why does `enable_diloco` exist?

Standard distributed data parallelism requires all-reducing gradients across all workers on **every single training step**. When training across geographically separated data centers or clusters connected by slow Data Center Interconnects (DCN), network latency and bandwidth bottlenecks collapse compute utilization.

DiLoCo (Distributed Low-Communication) trains independent model replicas locally for $H$ steps (e.g. 36–500 steps) using an inner optimizer, and only synchronizes replica weights periodically via an outer optimizer:

```text
Standard Data Parallelism (Frequent Communication):
Step 0 ──>[All-Reduce Grads (DCN Lag)]──> Step 1 ──>[All-Reduce Grads]...

DiLoCo (Low-Communication Distributed Training):
Replica 1: [Step 1..36 Local Training] ──┐
                                         ├──>[All-Reduce Weights & Apply Outer Update]
Replica 2: [Step 1..36 Local Training] ──┘
```

`enable_diloco` serves as the master switch for DiLoCo distributed training in MaxText.

---

## 2. Fundamentals & Mechanics

When `enable_diloco: true`:
1. Each cluster slice runs as an independent replica.
2. Replicas execute `diloco_sync_period` local training steps using their local optimizer (`opt_type`).
3. At the sync boundary, replicas compute pseudo-gradients $\Delta W = W_{	ext{local}} - W_{	ext{base}}$.
4. Pseudo-gradients are all-reduced across replicas, and an **outer optimizer** updates the base model weights.

---

## 3. Options & Defaults

| Option | Value | Meaning |
|---|---|---|
| Default | `false` | Standard synchronous training across all devices. |
| Enabled | `true` | Activates DiLoCo local replica training and periodic synchronization. |

---

## 4. Interactions & Dependencies

```text
enable_diloco: true
        │
        ├──> diloco_sync_period (Sync interval H)
        ├──> diloco_outer_lr (Outer learning rate)
        ├──> diloco_outer_momentum (Outer momentum)
        └──> dcn_diloco_parallelism (Mesh sharding across replicas)
```

---

## 5. Practical Scenarios & Failure Modes

- **Multi-Datacenter Training:** Enables training large models across two geographically separate TPU pods with 100x–1000x less cross-datacenter bandwidth demand.

---

### One-line intuition

> **`enable_diloco` enables Distributed Low-Communication training, allowing model replicas to train independently and synchronize only periodically across slow networks.**
