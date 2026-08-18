## 1. Why does `monitor_goodput` exist?

While `enable_goodput_recording` passively logs timestamps and calculates efficiency ratios, cluster orchestration systems need **active monitoring** to detect when a job degrades below an acceptable throughput threshold:

```text
Cluster Health Monitor:
Goodput Rate: 92% ──(Healthy)──> Continue Training
Goodput Rate: 48% ──(Sustained Degradation / I/O Bottleneck)──> Emit Health Alert / Trigger Auto-Healer
```

`monitor_goodput` activates real-time active health monitoring over the recorded Goodput metrics stream.

---

## 2. What it actually controls

```yaml
monitor_goodput: false
```

- When `false` (default): Passive recording only (if `enable_goodput_recording: true`) without active threshold monitoring.
- When `true`: Starts an active Goodput monitoring daemon that evaluates ongoing job efficiency against baseline performance targets and reports health status.

---

## 3. Options and Defaults

| Value | Behavior |
|---|---|
| `false` (default) | Active Goodput monitoring disabled |
| `true` | Enables continuous active Goodput evaluation |

---

## 4. Interactions

- **`enable_goodput_recording`**: Must be `true` for `monitor_goodput` to operate.
- **`goodput_upload_interval_seconds`**: Defines the cadence of monitor evaluations.

---

## 5. Practical Scenarios

- **Automated Workload Slicing & XPK Operations**: Enable `monitor_goodput: true` to feed cluster-level autoscalers and workload health managers.

---

### One-line intuition

> **`monitor_goodput` enables active real-time health analysis and SLA tracking on top of recorded Goodput efficiency data.**
