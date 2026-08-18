## 1. Why does `elastic_timeout_seconds` exist?

During large-scale distributed execution, accelerator nodes can experience temporary communication pauses (transient network blips, host memory compaction, or brief kernel delays) that mimic node failures.

If the coordinator declared a slice dead immediately upon a single missed heartbeat, it would trigger expensive, premature mesh reconfigurations and slice shrinkages for recoverable transient delays.

Conversely, if the coordinator waited indefinitely, a permanently crashed node would hang the entire multi-slice job forever:

```text
Slice misses heartbeat:
               │
               ▼
   Wait elastic_timeout_seconds (e.g. 300s)
               │
      ┌────────┴──────────────────────────┐
      ▼                                   ▼
Node recovers within window         Node unresponsive after 300s
──> Resume normally                 ──> Declare dead, re-shard cluster elastically
```

`elastic_timeout_seconds` defines the grace period (in seconds) the coordinator waits for an unresponsive slice before officially declaring it failed and executing an elastic mesh reconfiguration.

---

## 2. Mechanics & Coordinator Lifecycle

1. The Pathways coordinator monitors periodic heartbeat pings from all active TPU slice controllers.
2. If heartbeats from Slice $X$ stop arriving, a timer starts for `elastic_timeout_seconds`.
3. If Slice $X$ responds before timeout expiry, normal execution continues without re-sharding.
4. If the timer exceeds `elastic_timeout_seconds`, Slice $X$ is severed from the cluster, surviving slices are re-meshed, and training resumes.

---

## 3. Options & Default

| Parameter | Type | Default | Options |
| :--- | :--- | :--- | :--- |
| `elastic_timeout_seconds` | `int` | `300` | Positive integer (e.g., `120`, `300`, `600`) |

---

## 4. Interactions with Related Parameters

| Related Parameter | Interaction |
| :--- | :--- |
| `elastic_enabled` | `elastic_timeout_seconds` is active only when `elastic_enabled: true`. |
| `elastic_max_retries` | Each timeout-triggered reconfiguration consumes one retry attempt from `elastic_max_retries`. |

---

## 5. Practical Guidance

| Value | Behavior | Use Case |
| :--- | :--- | :--- |
| `elastic_timeout_seconds: 300` (5 mins) | Standard default; tolerates typical host GC or transient network drops. | General multi-slice production training. |
| `elastic_timeout_seconds: 60` – `120` | Aggressive dropout; reconfigures mesh quickly to minimize idle TPU time. | Highly unstable spot environments where lost nodes rarely recover. |
| `elastic_timeout_seconds: 600` (10 mins) | Conservative; gives slow rebooting VMs ample time to rejoin. | Dedicated enterprise clusters with known maintenance windows. |

---

### One-line intuition

> `elastic_timeout_seconds` sets how long the Pathways coordinator waits for an unresponsive TPU slice before declaring it dead and reconfiguring the cluster elastically.
