## 1. Why it exists: eliminating the cold-start compilation penalty

In JAX and XLA, function calls decorated with `@jax.jit` are compiled **just-in-time** upon their first invocation with concrete array shapes:

```text
First User Request (Cold Start, enable_model_warmup: false):
[User Prompt Arrives] ──> [XLA Compilation (10s – 120s!)] ──> [Forward Pass (15ms)] ──> [Response]
Result: User experiences massive latency spike, gateway timeouts (504 Gateway Timeout).

Server Startup with Warmup (enable_model_warmup: true):
[Server Boots] ──> [Execute Dummy Prefill & Decode Passes] ──> [XLA Compiles & Caches HLO]
                         │
                         ▼
[Open Server Port to Traffic] ──> [User Prompt Arrives] ──> [Fast Forward Pass (15ms)] ──> [Response]
Result: Zero user-facing compilation latency; instant sub-second response times.
```

If a production inference server immediately advertises ready health status to load balancers before executing a warmup pass, the first live client request triggers full graph compilation. For large models (e.g., Llama 3 70B across multi-host TPUs), XLA compilation can take 30 seconds to several minutes, causing connection drops, load-balancer health check failures, and severe SLA violations.

`enable_model_warmup` forces MaxText to run synthetic dummy prefill and decode passes during server initialization, pre-compiling all execution graphs before accepting live traffic.

---

## 2. Mechanics: synthetic warmup execution

During server startup (before binding network ports or entering the serving loop):

```text
 1. Initialize Weights & KV Cache Buffers
                    │
                    ▼
 2. Check: `enable_model_warmup: true`
                    │
                    ▼
 ┌───────────────────────────────────────────────────────────┐
 │               Synthetic Warmup Sequence                   │
 │                                                           │
 │ Step A: Run Dummy Prefill Graph                           │
 │   - Inputs: [Dummy Batch, Max/Target Sequence Length]     │
 │   - Forces XLA compilation of prompt attention & GEMMs    │
 │                                                           │
 │ Step B: Run Dummy Autoregressive Decode Step              │
 │   - Inputs: [Dummy Batch, Single Token] + Dummy KV Cache  │
 │   - Forces XLA compilation of decode step kernel          │
 │                                                           │
 │ Step C: Block on Completion (`jax.block_until_ready()`)   │
 │   - Ensures compiled binaries are loaded onto HBM         │
 └──────────────────────────┬────────────────────────────────┘
                            │
                            ▼
 3. Mark Server Healthy & Open Client Endpoints
```

By passing dummy token tensors matching the expected runtime shapes:
1. XLA generates and compiles the optimized HLO binaries for both prefill and generation.
2. Tensor allocations in TPU High Bandwidth Memory (HBM) are locked in.
3. Once `jax.block_until_ready()` resolves, the server marks its readiness endpoint as healthy (`/readyz` or gRPC `SERVING`).

---

## 3. Options & Configuration

Default in `base.yml`:

```yaml
enable_model_warmup: false
```

| Setting | Value | Characteristics | Recommended Usage |
|---|---|---|---|
| Disabled (default) | `false` | Fast server startup; first real request incurs compilation delay. | Fast local iteration, quick unit tests where you don't want to wait for warmup passes. |
| Enabled | `true` | Slower initial startup; perfectly smooth, instant response times on request 1. | **Mandatory for production serving**, GKE deployments, Kubernetes readiness probes, and benchmark runs. |

---

## 4. Parameter Interactions

```text
┌───────────────────────────────────────────────────────────┐
│                    enable_model_warmup                    │
└─────────────┬───────────────────────────────┬─────────────┘
              │ (when true)
              ▼
┌───────────────────────────────────────────────────────────┐
│ Warmup pass compiles graphs for shapes defined by:        │
│ - max_target_length                                       │
│ - max_prefill_predict_length                              │
│ - per_device_batch_size / num_samples                     │
│ - compute_axis_order / ar_cache_axis_order                │
└───────────────────────────────────────────────────────────┘
```

- **`max_prefill_predict_length` & `max_target_length`**: Dictates the tensor sequence dimensions passed during the dummy warmup calls.
- **`jax_cache_dir`**: If persistent compilation cache is enabled, the warmup pass reads/writes the compiled HLO to disk, accelerating future restarts.
- **Kubernetes Readiness Probes**: Ensure your Kubernetes `initialDelaySeconds` and `timeoutSeconds` are sized appropriately to allow the warmup pass to complete before Kubernetes kills the container for failing readiness checks.

---

## 5. Practical Scenarios & Failure Modes

### Production Serving on GKE
In a Kubernetes deployment:
```yaml
# MaxText Serving Config
enable_model_warmup: true
inference_server: "MaxtextInterleavedServer"
```
The Kubernetes readiness probe checks `http://localhost:<port>/ready`. Because warmup runs before the server reports ready, the GKE Ingress load balancer will only route user traffic after compilation has finished.

### What breaks if misconfigured:
- **Warmup OOM**: If the warmup shapes are configured larger than available device HBM, the server crashes during initialization rather than at runtime.
- **Probe timeouts**: If Kubernetes readiness timeout is 10s but XLA compilation takes 45s, Kubernetes may enter a restart loop. Increase pod probe timeout to accommodate warmup duration.

---

### One-line intuition

> **`enable_model_warmup` triggers synthetic prefill and decode passes at server startup, pre-compiling all XLA execution graphs so that live user requests never suffer from cold-start JIT compilation latency.**
