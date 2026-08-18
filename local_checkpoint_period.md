## 1. Why does it exist?

Standard checkpoint periods (configured via `checkpoint_period`) are typically set to large intervals (e.g. 5,000 to 10,000 steps) to prevent slow remote GCS write operations from disrupting training throughput.

However, when local emergency or multi-tier checkpointing is active, writing to host RAM or NVMe is orders of magnitude faster. `local_checkpoint_period` defines the **step interval** for these rapid local saves.

```text
Full GCS Period (checkpoint_period = 5000):
  Step 0 ──────────────────────────────────────────────────────────→ Step 5000 (GCS)
  
Local Period (local_checkpoint_period = 25):
  |--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--| (Local RAM)
```

`local_checkpoint_period` sets how frequently (in training steps) an emergency or multi-tier checkpoint is saved to local host storage.

---

## 2. Fundamentals & Step Timing

- **Step Cadence**: At every step $N$ where $N \pmod{\text{local\_checkpoint\_period}} == 0$, the model and optimizer state are serialized directly into `local_checkpoint_directory`.
- **Low Overhead**: Because local writes don't travel over the wide-area data center network to cloud buckets, this period can safely be set 50x to 100x more frequently than standard GCS checkpointing without causing throughput degradation.

---

## 3. Options & Configuration

| Value | Meaning |
|---|---|
| `0` (default) | Local checkpointing disabled. |
| Positive integer (e.g. `20`, `50`, `100`) | Saves a local checkpoint every `N` training steps. |

Default in `base.yml`:
```yaml
local_checkpoint_period: 0
```

---

## 4. Interactions & Strict Constraints

MaxText enforces that `local_checkpoint_period` must be positive if and only if local checkpointing is enabled:

```text
Constraint Table:
  enable_emergency_checkpoint == true OR enable_multi_tier_checkpointing == true
  └──> local_checkpoint_period MUST be > 0 (e.g. 20)

  enable_emergency_checkpoint == false AND enable_multi_tier_checkpointing == false
  └──> local_checkpoint_period MUST be 0
```

---

## 5. Practical Sizing Recommendations

- **Typical Setting**: `20` to `100` steps.
- **Trade-off Analysis**:
  - **Too Small (`< 5` steps)**: May introduce slight CPU/memory-bus contention on the host VM if serialization occurs on every iteration.
  - **Too Large (`> 500` steps)**: Increases the number of lost steps when recovering from a mid-run preemption.

---

### One-line intuition

> **`local_checkpoint_period` controls the high-frequency step interval (e.g. every 20–50 steps) for saving lightweight recovery snapshots to host-local storage.**
