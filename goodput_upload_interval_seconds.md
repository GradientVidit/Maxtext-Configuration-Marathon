## 1. Why does `goodput_upload_interval_seconds` exist?

Goodput metrics are generated every training step. Sending an HTTP/RPC request to monitoring services on every single step would create network traffic storms across thousands of TPU hosts.

Instead, Goodput metrics are aggregated locally in memory and uploaded periodically:

```text
Step 0 ──> Step 1 ──> Step 2 ... ──> [ Aggregate locally in memory ]
                                                 │
                                                 ▼ (Every 30 seconds)
                                     Upload Batch to GCP Cloud Monitoring
```

`goodput_upload_interval_seconds` defines the upload flush interval.

---

## 2. What it actually controls

```yaml
goodput_upload_interval_seconds: 30
```

- Specifies the frequency (in seconds) at which the Goodput recorder flushes aggregated metric payloads to the metrics sink (e.g. Cloud Monitoring or GCS).

---

## 3. Options and Defaults

| Value | Flush Cadence | Network Traffic | Telemetry Freshness |
|---|---|---|---|
| `30` (default) | Every 30 seconds | Very low | Near real-time |
| `10` | Every 10 seconds | Low | High-frequency |
| `60` – `120` | Every 1–2 minutes | Minimal | Batched |

---

## 4. Interactions

- **`enable_goodput_recording`**: Active only when Goodput recording is enabled.

---

## 5. Practical Scenarios

- **Default is Optimal**: Keep `30` seconds for standard production monitoring.

---

### One-line intuition

> **`goodput_upload_interval_seconds` controls how often locally aggregated Goodput metrics are flushed and uploaded to monitoring sinks.**
