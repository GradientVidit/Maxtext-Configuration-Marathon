## 1. Why does `heartbeat_reporting_interval_in_seconds` exist?

Liveness heartbeats must balance two constraints:
1. Fast enough to allow watchdog systems to detect a deadlocked training run within minutes.
2. Moderate enough to avoid overwhelming the Cloud Monitoring API quota with excessive requests.

`heartbeat_reporting_interval_in_seconds` configures the frequency of the background liveness heartbeat.

---

## 2. What it actually controls

```yaml
heartbeat_reporting_interval_in_seconds: 5
```

- Specifies the delay (in seconds) between successive heartbeat metric pings sent to GCP Cloud Monitoring.

---

## 3. Options and Defaults

| Value | Frequency | Cloud API Load | Detection Latency |
|---|---|---|---|
| `5` (default) | Ping every 5 seconds | Low | Instant (< 30s detection) |
| `15` – `30` | Ping every 15–30 seconds | Very low | Fast (< 2m detection) |

---

## 4. Interactions

- **`report_heartbeat_metric_for_gcp_monitoring`**: Must be `true` for this interval to take effect.

---

## 5. Practical Scenarios

- **Standard Watchdog Configuration**: Keep default `5` seconds.

---

### One-line intuition

> **`heartbeat_reporting_interval_in_seconds` sets the interval in seconds between periodic liveness heartbeat pings sent to GCP Cloud Monitoring.**
