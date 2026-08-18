## 1. Why does `enable_gcp_goodput_metrics` exist?

When managing large AI training fleets across hundreds of nodes on Google Cloud Platform (GCP), operators rely on **GCP Cloud Monitoring (Stackdriver)** dashboards and alerting policies:

```text
MaxText Run ──> [ Goodput Metrics ] ──(enable_gcp_goodput_metrics: true)──> GCP Cloud Monitoring
                                                                                   │
                                                                                   ▼
                                                                        Grafana / GCP Dashboards
                                                                        PagerDuty / Alert Policies
```

`enable_gcp_goodput_metrics` authorizes MaxText to stream Goodput timeseries data directly into the project's GCP Cloud Monitoring backend.

---

## 2. What it actually controls

```yaml
enable_gcp_goodput_metrics: true
```

- When `true` (default): MaxText uses the Google Cloud Monitoring API client to push custom Goodput metric descriptors and timeseries points to GCP.
- When `false`: Suppresses GCP Cloud Monitoring API export (metrics remain local/logged to stdout).

---

## 3. Options and Defaults

| Value | GCP Export | Recommended Environment |
|---|---|---|
| `true` (default) | Streams metrics to GCP Cloud Monitoring | Production GCP environments (GKE, XPK, GCE) |
| `false` | No GCP API calls | On-premise setups, local debug, non-GCP cloud |

---

## 4. Interactions and Prerequisites

- **`enable_goodput_recording`**: Must be `true` to generate the underlying metrics.
- **GCP IAM Permissions**: Requires `roles/monitoring.metricWriter` permission on the host VM service account.

---

## 5. Practical Scenarios

- **Cloud TPU Pretraining on GCP**: Keep `true` to populate enterprise monitoring dashboards.

---

### One-line intuition

> **`enable_gcp_goodput_metrics` exports training Goodput metrics directly to Google Cloud Monitoring for fleet-wide dashboards and alerting.**
