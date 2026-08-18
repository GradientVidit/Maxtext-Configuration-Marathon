## 1. Why does `enable_gcp_step_deviation_metrics` exist?

Automated cluster infrastructure requires programmatic alerts when individual TPU workers slow down. 

Exporting step time deviation metrics directly to GCP Cloud Monitoring allows DevOps teams to create Cloud Monitoring Alerting Policies (e.g. "Trigger alert if step time deviation exceeds 25% for > 5 minutes"):

```text
MaxText Step Variance ──> [ enable_gcp_step_deviation_metrics: true ] ──> GCP Cloud Monitoring
                                                                                 │
                                                                                 ▼
                                                                       Alert Policy: "Node Straggler"
```

`enable_gcp_step_deviation_metrics` controls the cloud export of these step variance metrics.

---

## 2. What it actually controls

```yaml
enable_gcp_step_deviation_metrics: true
```

- When `true` (default): Pushes step duration mean, standard deviation, and anomaly flags to Google Cloud Monitoring custom metrics.
- When `false`: Disables GCP API uploads for step deviation telemetry.

---

## 3. Options and Defaults

| Value | GCP Metric Emission | Use Case |
|---|---|---|
| `true` (default) | Enabled | Production GCP workloads with automated SLA alerts |
| `false` | Disabled | Local runs, non-GCP infrastructure |

---

## 4. Interactions

- **`monitor_step_time_deviation`**: Must be `true`.
- **GCP IAM**: Requires `roles/monitoring.metricWriter`.

---

## 5. Practical Scenarios

- **Fleet Health Dashboards**: Keep `true` in all Google Cloud production environments.

---

### One-line intuition

> **`enable_gcp_step_deviation_metrics` exports step execution variance metrics to Google Cloud Monitoring for automated straggler and stall alerting.**
