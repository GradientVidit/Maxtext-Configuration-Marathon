## 1. Why it exists: ephemeral test-harness execution vs. persistent serving

When developing, testing, or benchmarking MaxText's inference subsystem, starting a full persistent server process creates unnecessary testing friction:

```text
Persistent Serving Mode (inference_benchmark_test: false):
[Start MaxText] ──> [Init Weights] ──> [Open gRPC/HTTP Port] ──> [Block indefinitely listening for RPCs]
Requires external client process to send queries; blocks CI test scripts until killed.

Benchmark Test Mode (inference_benchmark_test: true):
[Start MaxText] ──> [Init Weights] ──> [Generate Synthetic Requests] ──> [Run Timed Loops] ──> [Log & Exit 0]
Fully self-contained; runs automated synthetic traffic, records timing metrics, and exits cleanly.
```

In automated regression tests, continuous integration (CI) runners, and performance validation scripts, the harness must execute end-to-end inference passes without requiring an external load generator or managing background daemon processes.

`inference_benchmark_test` is the master boolean toggle that runs MaxText in a self-contained inference benchmark test harness mode, executing synthetic traffic and exiting upon completion.

---

## 2. Mechanics: execution branch control

When `inference_benchmark_test: true`:

```text
                      Config: inference_benchmark_test: true
                                        │
                                        ▼
                      ┌───────────────────────────────────┐
                      │   Initialize JAX Mesh & Weights   │
                      └─────────────────┬─────────────────┘
                                        │
                                        ▼
                      ┌───────────────────────────────────┐
                      │  Execute Warmup Forward Passes    │
                      └─────────────────┬─────────────────┘
                                        │
                                        ▼
                      ┌───────────────────────────────────┐
                      │ Run Automated Synthetic Workload: │
                      │ - Prefill on synthetic tokens     │
                      │ - Autoregressive decode steps     │
                      │ - Measure TTFT, ITL, TPS          │
                      └─────────────────┬─────────────────┘
                                        │
                                        ▼
                      ┌───────────────────────────────────┐
                      │ Log Benchmark Results & Exit (0)  │
                      └───────────────────────────────────┘
```

Instead of binding to networking sockets (like `prometheus_port` or gRPC gateway endpoints) and waiting for remote client requests, the execution flow directly drives synthetic token inputs through the compiled inference graphs, measures timings, asserts correctness/performance thresholds, and cleanly terminates.

---

## 3. Options & Configuration

Default in `base.yml`:

```yaml
inference_benchmark_test: false
```

| Setting | Value | Behavior | Use Case |
|---|---|---|---|
| Disabled (default) | `false` | Normal operational mode (training or long-running inference server listening for requests). | Production model serving, live endpoints, interactive gateways. |
| Enabled | `true` | Executes internal synthetic benchmark passes, validates outputs, and terminates. | Automated CI/CD integration tests, hardware performance verification, post-deployment sanity checks. |

---

## 4. Parameter Interactions

```text
┌───────────────────────────────────────────────────────────┐
│                 inference_benchmark_test                  │
└─────────────┬───────────────────────────────┬─────────────┘
              │ (when true)
              ▼
┌───────────────────────────────────────────────────────────┐
│ Activates microbenchmarking & test loop parameters:       │
│ - inference_microbenchmark_prefill_lengths                │
│ - inference_microbenchmark_stages                         │
│ - inference_microbenchmark_loop_iters                     │
│ - inference_microbenchmark_log_file_path                  │
│ - inference_microbenchmark_num_samples                    │
└───────────────────────────────────────────────────────────┘
```

- **`inference_microbenchmark_*` suite**: When `inference_benchmark_test` is active, the specific prompt lengths, batch sizes, iteration counts, and logging targets are governed by these parameters.
- **`enable_model_warmup`**: Even in test mode, warmup is recommended to ensure untimed JIT compilation does not skew test metrics.

---

## 5. Practical Scenarios & Failure Modes

### CI/CD Integration Testing
Run an automated test on TPU pods before merging code changes:
```bash
python3 src/maxtext/train.py src/maxtext/configs/base.yml \
  model_name=llama2-7b \
  inference_benchmark_test=true \
  inference_microbenchmark_loop_iters=5
```
The job runs 5 iterations of synthetic inference and exits with code 0 on success.

### What breaks if misconfigured:
- **Accidental enablement in production**: Setting `inference_benchmark_test: true` in a production serving deployment will cause the serving container to run a quick synthetic test and immediately exit, causing Kubernetes pod restart loops (`CrashLoopBackOff`).

---

### One-line intuition

> **`inference_benchmark_test` switches MaxText into a self-contained test-harness mode that executes synthetic inference passes and exits, providing a clean entrypoint for CI and automated performance validation.**
