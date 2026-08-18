## 1. Why does `report_performance_metric_for_gcp_monitoring` exist?

Beyond high-level Goodput ratios, infrastructure teams need to monitor raw physical hardware performance metrics in GCP Cloud Monitoring dashboards:
- Tokens per second per chip
- Step throughput (steps/sec)
- Accelerator TFLOPS utilization (MFU / HFU)

```text
MaxText Execution Loop ──> [ Compute Step TFLOPs & Token Throughput ]
                                         │
                                         ▼ (report_performance_metric_for_gcp_monitoring: true)
                               GCP Cloud Monitoring Custom Metrics
                                         │
                                         ▼
                            Real-Time Cluster Throughput Dashboard
```

`report_performance_metric_for_gcp_monitoring` controls whether raw hardware performance metrics are streamed to GCP Cloud Monitoring.

---

## 2. What it actually controls

```yaml
report_performance_metric_for_gcp_monitoring: false
```

- When `false` (default): Performance metrics are printed to stdout and TensorBoard only.
- When `true`: Publishes hardware performance timeseries directly to Google Cloud Monitoring.

---

## 3. Options and Defaults

| Value | GCP Performance Export |
|---|---|
| `false` (default) | Disabled |
| `true` | Streams throughput and MFU metrics to GCP |

---

## 4. Interactions

- **`gcs_metrics`**: Works alongside GCS metric logging to provide centralized cloud visibility.

---

## 5. Practical Scenarios

- **Enterprise Cluster Monitoring**: Set `true` to track aggregate pretraining tokens/sec across an entire fleet of TPU slices.

---

### One-line intuition

> **`report_performance_metric_for_gcp_monitoring` streams raw training throughput, token rates, and MFU metrics directly to Google Cloud Monitoring.**
