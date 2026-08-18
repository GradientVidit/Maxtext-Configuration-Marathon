## 1. Why does it exist?

**DiLoCo** (Distributed Low-Communication) training is a distributed optimization algorithm designed for federated or multi-cluster environments connected by low-bandwidth, high-latency networks.

In DiLoCo, multiple independent replicas perform $H$ local optimization steps (e.g. 500 steps of AdamW) entirely on their local cluster without any inter-cluster communication. After $H$ steps, an outer optimizer (such as outer momentum SGD) synchronizes the pseudo-gradients across replicas over the wide-area network.

```text
Cluster / Slice 0                       Cluster / Slice 1
 ├── Step 1 (Local AdamW)                ├── Step 1 (Local AdamW)
 ├── Step 2 (Local AdamW)                ├── Step 2 (Local AdamW)
 └── Step 500 (Local AdamW)              └── Step 500 (Local AdamW)
       │                                       │
       └───────────────┬───────────────────────┘
                       │ Every 500 steps (Outer Sync over DCN)
                       ↓
         Outer Optimizer Momentum Update
```

`dcn_diloco_parallelism` configures the number of independent DiLoCo replicas participating in outer distributed low-communication synchronization across the Data Center Network.

---

## 2. Fundamentals & Mechanics

- **Local Autonomy**: Replicas execute standard training loops with standard intra-slice parallelism (FSDP, TP, etc.) without waiting for remote nodes.
- **Outer Gradient Sync**: Replicas exchange their accumulated parameter delta $\Delta W = W_{\text{local}} - W_{\text{init}}$ across the `diloco` mesh axis.

---

## 3. Options & Configuration

| Value | Meaning |
|---|---|
| `1` (default) | Standard training; DiLoCo distributed training is disabled. |
| Integer $> 1$ | Allocates `N` independent DiLoCo islands communicating over DCN at outer step intervals. |

Default in `base.yml`:
```yaml
dcn_diloco_parallelism: 1
```

---

## 4. Interactions with Related Parameters

- **`ici_diloco_parallelism`**: Configures intra-slice DiLoCo replicas.
- **`mesh_axes`**: Includes `'diloco'`.

---

### One-line intuition

> **`dcn_diloco_parallelism` sets the number of distributed DiLoCo worker islands that train independently for hundreds of steps before syncing pseudo-gradients over the datacenter network.**
