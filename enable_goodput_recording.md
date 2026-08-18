## 1. Why does `enable_goodput_recording` exist?

In large-scale AI cluster operations, **Raw Uptime** is a misleading metric. A cluster may be 100% powered on and running processes, but 25% of that time could be wasted on:
- Preemption and restart recovery overhead
- Prolonged checkpoint I/O stalls
- Data pipeline starvation
- Straggler nodes waiting at barrier synchronizations

```text
Total Allocated Cluster Time (Uptime: 100%)
┌─────────────────────────────────────────────────────────────┬───────────────────────┐
│              Productive Step Computation                    │    Wasted Overhead    │
│                  (True Goodput: ~82%)                       │ (Restarts, I/O, Wait) │
└─────────────────────────────────────────────────────────────┴───────────────────────┘
```

**Goodput** (Google Cloud Goodput Library) measures the ratio of productive training computation time to total elapsed time:

$$\text{Goodput} = \frac{\text{Productive Training Time}}{\text{Total Elapsed Wall-clock Time}}$$

`enable_goodput_recording` serves as the master switch to initialize the Goodput recording client and track productive training throughput in MaxText.

---

## 2. What it actually controls

```yaml
enable_goodput_recording: false
```

- When `false` (default): Goodput tracking is disabled; no Goodput logger or client background threads are spawned.
- When `true`: MaxText hooks into step boundaries, records timestamped start/finish events, calculates productive compute fractions, and logs Goodput telemetry.

---

## 3. Options and Defaults

| Value | Goodput Recording | Overhead | Recommended Use |
|---|---|---|---|
| `false` (default) | Disabled | 0% | Local development, unit tests, short smoke runs |
| `true` | Enabled | Negligible background thread | Multi-day production runs on GKE / XPK / GCE clusters |

---

## 4. Interactions and Sub-systems

- **`monitor_goodput`**: Enables active alerting/health monitoring in addition to passive recording.
- **`goodput_upload_interval_seconds`**: Sets how frequently recorded Goodput metrics are pushed.
- **`enable_gcp_goodput_metrics`**: Exports the recorded Goodput telemetry to GCP Cloud Monitoring.

---

## 5. Practical Scenarios

- **Production LLM Cluster SLA Tracking**: Set `enable_goodput_recording: true` in production configs to quantify financial waste from preemptions and checkpoint latency.

---

### One-line intuition

> **`enable_goodput_recording` is the master switch that tracks productive training time versus wasted cluster overhead using Google's Goodput library.**
