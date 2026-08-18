## 1. Why it exists: selecting the inference serving engine architecture

Deploying large language models for production inference on Google Cloud TPUs requires different server architectures depending on the workload characteristics and infrastructure constraints:

```text
Interleaved Server (Colocated Prefill & Decode):
┌──────────────────────────────────────────────┐
│ Single TPU Slice                             │
│ ┌──────────────────┐    ┌──────────────────┐ │
│ │  Prefill Engine  │    │  Decode Engine   │ │  ──> Resource contention between heavy
│ │  (Compute Heavy) │    │  (Memory Heavy)  │ │      prefill bursts and steady decode steps
│ └──────────────────┘    └──────────────────┘ │
└──────────────────────────────────────────────┘

Disaggregated Server (Decoupled Prefill & Decode Slices):
┌────────────────────────┐      KV Transfer      ┌────────────────────────┐
│ Prefill TPU Slice      │ ────────────────────> │ Decode TPU Slice       │
│ Sized for fast TTFT    │   (High-Speed DCN)    │ Sized for high TPS     │
└────────────────────────┘                       └────────────────────────┘
```

- **Interleaved serving**: Prefill (prompt processing) and decode (token generation) run on the same hardware devices. While simple to deploy, long prefill requests cause "preemption jitter" for ongoing decode streams, degrading Inter-Token Latency (ITL).
- **Disaggregated serving**: Prefill and decode are split across distinct, specialized TPU slices. Prefill slices handle compute-intensive prompt GEMMs, generate the KV cache, and transfer the cache across data-center networks (DCN) to decode slices optimized for high batch concurrency.

In the **JetStream** and **MaxEngine** ecosystem, `inference_server` specifies which inference server implementation and orchestration class MaxText initializes when launched as a live serving process.

---

## 2. Mechanics: server instantiation and request lifecycle

When launching MaxText in serving mode (via JetStream / MaxEngine or standalone inference entrypoints), the engine inspects `inference_server` to instantiate the corresponding server handler:

```text
                               Config: inference_server
                                          │
                  ┌───────────────────────┴───────────────────────┐
                  ▼                                               ▼
     "MaxtextInterleavedServer"                 "ExperimentalMaxtextDisaggregatedServer_8"
┌───────────────────────────────────┐               ┌───────────────────────────────────┐
│ Instantiates single-slice engine  │               │ Orchestrates two separate pools:  │
│ - Shared KV cache pool            │               │ 1. Prefill pool on `prefill_slice`│
│ - Handles incoming gRPC / HTTP    │               │ 2. Decode pool on `generate_slice`│
│ - Timeslices prefill & decode     │               │ - Manages cross-slice KV transfer │
└───────────────────────────────────┘               └───────────────────────────────────┘
```

The selected server class sets up:
1. **gRPC/REST endpoints**: Front-end request listening interfaces compatible with standard serving protocols (or `llm_inference_gateway`).
2. **Execution loop**: Continuous batching scheduling logic for forward passes.
3. **KV Cache management**: Memory allocation strategy across layers and devices (interleaved vs disaggregated buffer pools).

---

## 3. Options & Configuration

Default in `base.yml`:

```yaml
inference_server: "MaxtextInterleavedServer"
```

| Value | Server Architecture | Typical Deployment | Key Advantage |
|---|---|---|---|
| `"MaxtextInterleavedServer"` (default) | Single-slice colocated prefill + decode | Standalone GKE / TPU VM pods | Simple deployment; minimal network transfer overhead; ideal for balanced workloads. |
| `"ExperimentalMaxtextDisaggregatedServer_8"` | Disaggregated serving engine (8-way or multi-host) | Large-scale clusters with separate prefill/decode slices | Eliminates prefill-induced decode latency spikes; independent horizontal autoscaling of prefill vs decode capacity. |
| Custom server class string | User-implemented serving wrapper | Custom enterprise gateways / JetStream integrations | Allows plugging in proprietary scheduling, custom caching, or specialized token routing. |

---

## 4. Parameter Interactions

```text
┌───────────────────────────────────────────────────────────┐
│                     inference_server                      │
└─────────────┬───────────────────────────────┬─────────────┘
              │                               │
    "MaxtextInterleavedServer"     "ExperimentalMaxtextDisaggregatedServer_*"
              │                               │
              ▼                               ▼
┌───────────────────────────┐   ┌───────────────────────────┐
│Standard single mesh       │   │prefill_slice              │
│Runs on current host mesh. │   │generate_slice             │
│                           │   │stack_prefill_result_cache │
└───────────────────────────┘   └───────────────────────────┘
```

- **`prefill_slice` & `generate_slice`**: Mandatory when using disaggregated server implementations to specify the hardware types allocated for each phase.
- **`stack_prefill_result_cache`**: Highly recommended when using disaggregated servers to pack per-layer KV tensors before network serialization and transfer.
- **`enable_llm_inference_pool`**: Configures the server endpoints to register with Google Cloud's LLM Inference Gateway.
- **`prometheus_port`**: Exposes server health, request queues, and generation latencies for the active server instance.

---

## 5. Practical Scenarios & Failure Modes

### Choosing Between Interleaved and Disaggregated
- **Use `MaxtextInterleavedServer`** for low-to-medium traffic applications, single TPU slices (e.g. single v5e-8 or v6e-4), or when prompts and outputs have similar token lengths.
- **Use Disaggregated Servers** for high-volume enterprise traffic with large prompt-to-completion ratios (e.g. 8k context prompt answering in 50 tokens), where prefill would otherwise stall decode throughput.

### What breaks if misconfigured:
- **Unknown server class**: Specifying a typo in the class name causes an `AttributeError` or `ModuleNotFoundError` during server factory initialization.
- **Mismatched slice configs in disaggregated mode**: Running a disaggregated server without configuring compatible `prefill_slice` and `generate_slice` topologies leads to distributed tensor sharding incompatibilities.

---

### One-line intuition

> **`inference_server` selects the serving engine implementation—choosing between single-slice interleaved execution for simple deployments and multi-slice disaggregated serving for high-throughput, low-jitter production traffic under JetStream/MaxEngine.**
