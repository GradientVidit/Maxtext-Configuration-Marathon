## 1. Why it exists: production observability and SLA monitoring

In production LLM serving, operators need real-time, fine-grained telemetry to monitor system health, detect SLA degradations, and drive autoscaling policies:

```text
Without Prometheus Metrics (prometheus_port: 0):
[MaxText Serving Pod] ──> Opaque black box; impossible to see active request queue,
                          per-token latency spikes, or KV cache utilization in real time.

With Prometheus Metrics (prometheus_port: 9090):
┌──────────────────────────────────────┐
│        MaxText Serving Pod           │
│  - Active Requests: 42               │
│  - Prefill Latency P99: 45ms         │ ──> Prometheus Server ──> Grafana Dashboard /
│  - Inter-Token Latency P99: 12ms     │     (Scrapes :9090/metrics) Kubernetes HPA
│  - KV Cache Usage: 68%               │
└──────────────────────────────────────┘
```

Without a standard metrics scraping endpoint, integrating MaxText with standard cloud-native monitoring stacks (Kubernetes Prometheus Operator, Datadog, Grafana) requires parsing unstructured text logs, which is fragile, high-latency, and computationally expensive.

`prometheus_port` sets the TCP port on which MaxText exposes its Prometheus `/metrics` HTTP endpoint for real-time serving telemetry.

---

## 2. Mechanics: HTTP metrics server lifecycle

When `prometheus_port > 0`:

```text
 MaxText Inference Server Initialization
                   │
                   ▼
 Check: `prometheus_port: 9090` (> 0)
                   │
                   ▼
 ┌───────────────────────────────────────────────────────────┐
 │       Start Background Prometheus HTTP Server Thread      │
 │       - Binds socket: `0.0.0.0:9090`                      │
 │       - Exposes route: `http://<host>:9090/metrics`       │
 └─────────────────────────┬─────────────────────────────────┘
                           │
                           ▼
 ┌───────────────────────────────────────────────────────────┐
 │               Inference Engine Event Loop                 │
 │  - Record Time-To-First-Token (TTFT) Histogram            │
 │  - Record Inter-Token Latency (ITL) Histogram             │
 │  - Update active request Gauge & error Counters           │
 │  - Update KV Cache slot allocation Gauge                  │
 └───────────────────────────────────────────────────────────┘
```

### Key Metrics Exported:
- `maxtext_inference_request_count_total`: Cumulative counter of received requests.
- `maxtext_inference_time_to_first_token_seconds`: Histogram measuring prefill latency.
- `maxtext_inference_inter_token_latency_seconds`: Histogram measuring decode step latency.
- `maxtext_inference_active_requests`: Current in-flight concurrent requests.
- `maxtext_inference_kv_cache_utilization_ratio`: Fraction of allocated KV cache memory currently occupied ($0.0 \to 1.0$).

---

## 3. Options & Configuration

Default in `base.yml`:

```yaml
prometheus_port: 0
```

| Value | Status | Behavior | Recommended Use Case |
|---|---|---|---|
| `0` (default) | Disabled | No Prometheus HTTP server is started. Zero networking overhead. | Local development, offline training runs, microbenchmarks. |
| `> 0` (e.g. `9090`, `8000`) | Enabled | Starts an HTTP server on `0.0.0.0:<port>` exposing standard Prometheus metrics format. | **Production serving deployments**, Kubernetes GKE clusters, live monitoring dashboards. |

---

## 4. Parameter Interactions

```text
┌───────────────────────────────────────────────────────────┐
│                      prometheus_port                      │
└─────────────┬───────────────────────────────┬─────────────┘
              │ (when > 0)
              ▼
┌───────────────────────────────────────────────────────────┐
│ Monitored Systems:                                        │
│ - inference_server (exports request & latency histograms) │
│ - enable_llm_inference_pool (exports pool routing metrics)│
│ - Kubernetes Pod annotations:                             │
│   prometheus.io/scrape: "true"                            │
│   prometheus.io/port: "<port>"                            │
└───────────────────────────────────────────────────────────┘
```

- **`inference_server`**: The active serving engine continuously updates the metric gauges and histograms scraped by Prometheus.
- **Kubernetes Pod Spec**: Sizing `prometheus_port` requires adding matching container port definitions in your Kubernetes deployment manifests to allow ingress scraping.

---

## 5. Practical Scenarios & Failure Modes

### Kubernetes Horizontal Pod Autoscaling (HPA)
Configure MaxText serving pods to export metrics to Prometheus:
```yaml
inference_server: "MaxtextInterleavedServer"
prometheus_port: 9090
```
Kubernetes HPA monitors `maxtext_inference_kv_cache_utilization_ratio`. When cache utilization exceeds 80%, HPA automatically spins up additional TPU serving pods to absorb incoming traffic.

### What breaks if misconfigured:
- **Port Collision**: If `prometheus_port` is set to an already occupied port (e.g. conflicting with `jax_profiler_port` or a local gRPC port), server initialization will crash with `OSError: [Errno 98] Address already in use`.

---

### One-line intuition

> **`prometheus_port` specifies the HTTP port for exposing real-time Prometheus inference metrics (TTFT, ITL, queue depth, cache utilization), enabling cloud-native monitoring and autoscaling.**
