## 1. Why it exists: port management for live JAX profiling

When enabling JAX's background profiler service (`enable_jax_profiler: true`), the profiler needs a dedicated TCP port to listen for incoming gRPC capture requests from tools like TensorBoard, Xprof, or the Google Cloud TPU Profiler client:

```text
TensorBoard / Cloud TPU Profiler Client
                   │
                   │ gRPC connection to `http://<node-ip>:<jax_profiler_port>`
                   ▼
┌───────────────────────────────────────────────────────────┐
│        MaxText Serving Host (JAX Profiler Server)         │
│  - Listens on `0.0.0.0:<jax_profiler_port>`               │
│  - Captures XLA HLO execution traces on demand            │
└───────────────────────────────────────────────────────────┘
```

In multi-container Kubernetes pods, multi-host TPU VMs (where multiple host workers may run sidecars), or environments with strict networking firewall rules, default ports (e.g. `9999`) may conflict with other existing microservices or monitoring daemons.

`jax_profiler_port` configures the specific TCP port on which the JAX background profiler gRPC server listens.

---

## 2. Mechanics: socket binding and server launch

During runtime initialization (when `enable_jax_profiler: true`):

```text
 Check: `enable_jax_profiler: true`
                   │
                   ▼
 Read: `jax_profiler_port` (e.g. 9999)
                   │
                   ▼
 ┌───────────────────────────────────────────────────────────┐
 │ Call: `jax.profiler.start_server(port=jax_profiler_port)` │
 │ - Binds TCP socket: `0.0.0.0:<jax_profiler_port>`         │
 │ - Starts background worker thread                         │
 └─────────────────────────┬─────────────────────────────────┘
                           │
                           ▼
 MaxText continues normal training / serving execution loop
```

Once bound:
- The server responds to gRPC health checks on that port.
- When a `CaptureProfile` RPC arrives on this port, the server activates XLA hardware event tracing for the requested duration, packages the result into a `.xplane.pb` format, and returns it over the gRPC stream.

---

## 3. Options & Configuration

Default in `base.yml`:

```yaml
jax_profiler_port: 9999
```

| Setting | Port Value | Use Case |
|---|---|---|
| Default | `9999` | Standard JAX/TensorBoard profiling port; recognized by default in TensorBoard UI. |
| Custom port | e.g. `8444`, `9005` | Used when `9999` is occupied by another service or restricted by corporate firewall policies. |

---

## 4. Parameter Interactions

```text
┌───────────────────────────────────────────────────────────┐
│                     jax_profiler_port                     │
└─────────────┬───────────────────────────────┬─────────────┘
              │
              ▼
┌───────────────────────────────────────────────────────────┐
│ Controlled by:                                            │
│ - enable_jax_profiler (must be true to bind this port)    │
│ - prometheus_port (must be on a DIFFERENT port to avoid   │
│   socket collisions!)                                     │
└───────────────────────────────────────────────────────────┘
```

- **`enable_jax_profiler`**: Must be set to `true` for `jax_profiler_port` to take effect.
- **`prometheus_port`**: Ensure `prometheus_port` and `jax_profiler_port` do not use the same integer value; attempting to bind both to the same port will crash the process with an address collision error.

---

## 5. Practical Scenarios & Failure Modes

### Resolving Multi-Host Pod Port Conflicts
When deploying in Kubernetes with multiple background sidecars:
```yaml
enable_jax_profiler: true
jax_profiler_port: 9005
prometheus_port: 9090
```
In your Kubernetes Pod specification:
```yaml
ports:
- name: jax-profiler
  containerPort: 9005
- name: prometheus
  containerPort: 9090
```

### What breaks if misconfigured:
- **`Address already in use` error**: If the port is already bound by another process on the host, the JAX runtime raises an unhandled socket exception during startup.

---

### One-line intuition

> **`jax_profiler_port` sets the TCP listening port for JAX's on-demand profiler gRPC server, allowing remote tools like TensorBoard to connect and capture live hardware traces.**
