## 1. Why does `monitor_step_time_deviation` exist?

A training run can experience "silent stalls" or partial degradation without crashing:
- A single degraded TPU chip running thermal throttling
- A host experiencing disk I/O bottlenecks during logging
- Intermittent DCN network packet loss slowing down all-gather collectives

```text
Normal Execution:
Step 100: 420ms ──> Step 101: 418ms ──> Step 102: 421ms (Stable Baseline)

Degraded Execution (Straggler Node Spike):
Step 103: 419ms ──> Step 104: 1,850ms ──> Step 105: 2,100ms (Major Deviation Detected!)
                                               │
                                               ▼ (monitor_step_time_deviation: true)
                                      Log Metric & Emit GCP Alert
```

`monitor_step_time_deviation` monitors step execution times against rolling baselines and flags anomalous variance spikes.

---

## 2. What it actually controls

```yaml
monitor_step_time_deviation: true
```

- When `true` (default): MaxText continuously tracks per-step wall-clock latency, computes statistical moving averages and standard deviations, and detects abnormal execution pauses.
- When `false`: Step latency variance tracking is disabled.

---

## 3. Options and Defaults

| Value | Behavior | Utility |
|---|---|---|
| `true` (default) | Actively monitors step duration variance | Highly recommended; catches silent stragglers and network degradation |
| `false` | Disables step latency anomaly tracking | Minimal smoke tests |

---

## 4. Interactions

- **`step_deviation_interval_seconds`**: Interval for evaluating deviation statistics.
- **`enable_gcp_step_deviation_metrics`**: Exports deviation metrics to GCP Cloud Monitoring.

---

## 5. Practical Scenarios

- **Cluster Health Debugging**: Keep `true` to quickly identify if multi-host runs are suffering from intermittent network tail-latency spikes.

---

### One-line intuition

> **`monitor_step_time_deviation` tracks execution time variance across training steps to detect hardware stragglers and network performance stalls.**
