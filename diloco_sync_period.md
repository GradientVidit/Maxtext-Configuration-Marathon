## 1. Why does `diloco_sync_period` exist?

In DiLoCo, there is a fundamental trade-off between communication frequency and optimization stability:
- If replicas synchronize too frequently (e.g. every 2 steps), network bandwidth savings disappear.
- If replicas synchronize too infrequently (e.g. every 10,000 steps), local model weights drift into completely different loss basins, and averaging them produces high loss:

```text
Replication Synchronization Interval (H):
Replica A: [Local Step 1 ──> Step 36] ──┐
                                        ├──> Synchronize & Outer Step
Replica B: [Local Step 1 ──> Step 36] ──┘
└──────────────────┬──────────────────┘
                   ▼
          diloco_sync_period = 36
```

`diloco_sync_period` sets the number of local training steps each replica takes between synchronization barriers.

---

## 2. Fundamentals & Mechanics

- Specifies the local step horizon $H$.
- Default `36` reflects the standard empirical setting from the original DiLoCo research paper.

---

## 3. Options & Defaults

| Option | Value | Meaning |
|---|---|---|
| Default | `36` | Standard DiLoCo sync period (36 local steps per outer sync). |
| Longer Interval | `100` to `500` | Extreme bandwidth reduction for very slow inter-datacenter links. |

---

## 4. Interactions & Dependencies

- Governs the invocation interval of `diloco_outer_lr` and `diloco_outer_momentum`.

---

## 5. Practical Scenarios & Failure Modes

- Setting `diloco_sync_period` excessively large ($>1000$) on high learning rates causes severe weight divergence and loss degradation upon averaging.

---

### One-line intuition

> **`diloco_sync_period` defines the number of local training steps executed on each replica between global synchronization rounds.**
