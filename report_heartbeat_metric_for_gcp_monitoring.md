## 1. Why does `report_heartbeat_metric_for_gcp_monitoring` exist?

If a training job experiences a hard kernel deadlock, process hang, or silent network freeze, no error logs or crash dumps are produced. The job simply stops advancing steps while continuing to consume expensive TPU billing:

```text
Healthy Run:   [ Step 10 ] ──> [ Heartbeat ] ──> [ Step 11 ] ──> [ Heartbeat ] (Signal Alive)
Hanged Run:    [ Step 12 ] ──> (Deadlock)   ──> [ NO HEARTBEAT EMITTED ]
                                                      │
                                                      ▼ (GCP Cloud Monitoring)
                                           Trigger Auto-Restart / Alert
```

A **Heartbeat Metric** periodically emits a liveness pulse to GCP Cloud Monitoring. If the pulse disappears for longer than a threshold (e.g. 60 seconds), automated cloud watchdogs can detect the hang and restart the job.

`report_heartbeat_metric_for_gcp_monitoring` enables this periodic liveness heartbeat.

---

## 2. What it actually controls

```yaml
report_heartbeat_metric_for_gcp_monitoring: false
```

- When `false` (default): No dedicated background liveness heartbeat is emitted to GCP.
- When `true`: Spawns a background reporting thread that pings GCP Cloud Monitoring every `heartbeat_reporting_interval_in_seconds`.

---

## 3. Options and Defaults

| Value | Heartbeat Status | Use Case |
|---|---|---|
| `false` (default) | Disabled | Standard development runs |
| `true` | Enabled | Unattended, multi-day production runs with automated watchdog restart scripts |

---

## 4. Interactions

- **`heartbeat_reporting_interval_in_seconds`**: Sets heartbeat frequency (default 5s).
- **GCP IAM**: Requires Cloud Monitoring metric writing permissions.

---

## 5. Practical Scenarios

- **Automated Preemption & Hang Recovery**: Enable `report_heartbeat_metric_for_gcp_monitoring: true` on spot TPU instances paired with a Cloud Monitoring uptime alert.

---

### One-line intuition

> **`report_heartbeat_metric_for_gcp_monitoring` periodically emits a liveness pulse to Google Cloud Monitoring so watchdogs can detect deadlocks and silent job hangs.**
