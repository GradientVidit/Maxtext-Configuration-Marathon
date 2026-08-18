## 1. Why it exists: live, on-demand profiling of running production servers

Traditional offline profiling requires stopping a process, modifying configuration flags (like `profiler_steps`), restarting the job, capturing a fixed trace, and terminating. In production serving environments, this approach is unacceptable:

```text
Offline Profiling (Destructive):
[Live Serving Pod] ──> Kill Service ──> Edit YAML (profiler_steps: 10) ──> Restart ──> Capture ──> Kill
Result: Service downtime, dropped user traffic, cannot capture transient production performance anomalies.

Live JAX Profiler Server (Non-Destructive, enable_jax_profiler: true):
[Live Serving Pod] ──> Continuously serves user traffic
                               │ (Listens on gRPC port, e.g. 9999)
                               ▲
                      TensorBoard / Xprof Profiler UI
                      "Capture Profile for 5 seconds now"
Result: Zero downtime; live capture of real-world production traffic spikes and latency bottlenecks.
```

Transient performance bugs—such as tail-latency spikes caused by specific prompt lengths, DCN network contention, or dynamic memory reallocations—only manifest under live production traffic. 

`enable_jax_profiler` starts JAX's built-in background profiler gRPC service, allowing developers and performance engineers to connect remotely with TensorBoard or Xprof to capture on-demand hardware performance traces without interrupting live serving.

---

## 2. Mechanics: JAX background profiler server

When `enable_jax_profiler: true`:

```text
 MaxText Server Boot
          │
          ▼
 Check: `enable_jax_profiler: true`
          │
          ▼
 ┌───────────────────────────────────────────────────────────┐
 │       Invoke `jax.profiler.start_server(port)`            │
 │       - Starts background gRPC service on `0.0.0.0:9999`  │
 │       - Registers hooks with XLA hardware tracing engine  │
 └─────────────────────────┬─────────────────────────────────┘
                           │
                           ▼
               Serving Production Requests
                           │
           Remote Capture Triggered via TensorBoard
                           │
 ┌─────────────────────────┴─────────────────────────────────┐
 │               Live Trace Window (e.g. 5 seconds)          │
 │ - Hardware trace: HLO Op execution on TPU Matrix Units    │
 │ - Memory trace: HBM buffer allocations & deallocations    │
 │ - Network trace: DCN all-reduce / all-gather collectives  │
 └─────────────────────────┬─────────────────────────────────┘
                           │
                           ▼
          Export Trace to GCS / TensorBoard UI
```

The background server introduces negligible CPU/HBM overhead when idle. Only when an external client (e.g. `tensorboard --logdir ...` or Cloud TPU profiler) initiates a capture session does the profiler attach tracing buffers and record hardware execution counters.

---

## 3. Options & Configuration

Default in `base.yml`:

```yaml
enable_jax_profiler: false
```

| Setting | Profiler Service State | Overhead | Recommended Use Case |
|---|---|---|---|
| `false` (default) | Profiler server is not initialized. | Zero. | Offline training runs, local testing, standard benchmark runs. |
| `true` | Runs background JAX profiler gRPC server on `jax_profiler_port`. | Negligible when idle; minor trace overhead during active capture. | **Production serving pods**, staging performance clusters, live debugging of tail-latency spikes. |

---

## 4. Parameter Interactions

```text
┌───────────────────────────────────────────────────────────┐
│                    enable_jax_profiler                    │
└─────────────┬───────────────────────────────┬─────────────┘
              │ (when true)
              ▼
┌───────────────────────────────────────────────────────────┐
│ Configured by:                                            │
│ - jax_profiler_port (sets listening port, default 9999)   │
│ - xprof_tpu_power_trace_level / profile_power_events      │
│   (adds hardware power/thermal events into the live trace)│
└───────────────────────────────────────────────────────────┘
```

- **`jax_profiler_port`**: Specifies the exact port number for the JAX gRPC profiler server.
- **`profile_power_events` & `xprof_tpu_power_trace_level`**: If power/thermal tracing is active, those hardware signals are recorded into the live profile captured through the JAX profiler server.

---

## 5. Practical Scenarios & Failure Modes

### On-Demand Remote Profiling via TensorBoard
1. Enable profiler in MaxText config:
   ```yaml
   enable_jax_profiler: true
   jax_profiler_port: 9999
   ```
2. Port-forward the pod in Kubernetes:
   ```bash
   kubectl port-forward pod/maxtext-server-0 9999:9999
   ```
3. Open TensorBoard, navigate to the **Profile** tab, select **Capture Profile**, target `localhost:9999`, and record a 5-second trace of live production requests.

### What breaks if misconfigured:
- **Port Conflict**: If another process (or another JAX worker on multi-host TPU VMs) binds to the same port, server startup will fail. Ensure ports are exposed per-host or mapped uniquely.

---

### One-line intuition

> **`enable_jax_profiler` starts JAX's background gRPC profiler server, enabling non-destructive, on-demand hardware trace capture with TensorBoard while the model is actively serving live traffic.**
