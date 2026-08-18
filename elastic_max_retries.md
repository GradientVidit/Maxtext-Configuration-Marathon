## 1. Why does `elastic_max_retries` exist?

Elastic training is designed to absorb occasional hardware failures and spot preemptions.

However, if a cluster suffers from a systemic hardware defect (such as bad optical transceiver cables causing recurring link drops, or a fatal bug triggering repeated kernel panics), an elastic job could enter an **infinite retry loop**:

```text
Job shrinks to 7 slices ──> Crash ──> Reconfigures to 6 slices ──> Crash ──> Reconfigures to 5 slices ...
[Repeatedly crashes without making forward training progress, burning cloud budget]
```

To guard against runaway failures and infinite crash loops, `elastic_max_retries` establishes a hard ceiling on the total number of elastic recovery and slice-reconfiguration attempts permitted before the entire job is cleanly terminated.

---

## 2. Mechanics & Retry Counter

- **Tracking**: The Pathways coordinator maintains an internal counter $C_{\text{retries}}$ initialized to $0$.
- **Increment**: Each time a slice failure triggers an elastic recovery and mesh re-sharding sequence, $C_{\text{retries}}$ is incremented by $1$.
- **Termination**: If $C_{\text{retries}} > \text{elastic\_max\_retries}$, the coordinator aborts the run, flushes final logs, and raises a fatal exception.

---

## 3. Options & Default

| Parameter | Type | Default | Options |
| :--- | :--- | :--- | :--- |
| `elastic_max_retries` | `int` | `10` | Non-negative integer (e.g. `5`, `10`, `20`) |

---

## 4. Interactions with Related Parameters

| Related Parameter | Interaction |
| :--- | :--- |
| `elastic_enabled` | Master switch controlling elastic fault tolerance. |
| `elastic_timeout_seconds` | Each timeout event consumes one retry against `elastic_max_retries`. |
| `elastic_min_slice_count` | If surviving slices drop below `elastic_min_slice_count`, the job terminates even if retries remain. |

---

## 5. Practical Guidance

| Value | Policy | Use Case |
| :--- | :--- | :--- |
| `elastic_max_retries: 10` (Default) | Allows moderate cluster churn across multi-day pretraining runs. | Standard production baseline on large multi-slice topologies. |
| `elastic_max_retries: 3` | Strict failure threshold; stops quickly on flaky clusters. | Testing new cluster setups or debugging hardware reliability. |
| `elastic_max_retries: 25` | Highly tolerant of high-preemption spot environments. | Long-running low-cost spot capacity campaigns. |

---

### One-line intuition

> `elastic_max_retries` sets the maximum number of slice failure and recovery attempts allowed during elastic training before the job safely aborts.
