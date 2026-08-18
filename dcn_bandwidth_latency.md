## 1. Why does `dcn_bandwidth_latency` exist?

Bandwidth alone does not represent real-world wide-area networking: geographically separated clusters incur physical speed-of-light propagation latency (e.g. 50ms round-trip time between US-East and Europe).

To realistically evaluate DiLoCo synchronization overhead, the network emulator must inject artificial latency:

```text
Emulated DCN Packet Path:
Worker VM Packet ──>[ Artificial Latency Delay: 50ms ]──>[ Bandwidth Cap ]──> Network
```

`dcn_bandwidth_latency` defines the maximum artificial latency / queue delay applied by Linux `tc`.

---

## 2. Fundamentals & Mechanics

- Passed as the `latency` parameter to `tc qdisc add ... tbf`.
- Default `"50ms"` reflects standard inter-region transatlantic/transcontinental network latency.

---

## 3. Options & Defaults

| Option | Value | Meaning |
|---|---|---|
| Default | `"50ms"` | 50 millisecond emulated network delay. |
| Low Latency | `"10ms"` | Metro-area datacenter interconnect delay. |
| High Latency | `"100ms"` | Global cross-continent latency. |

---

## 4. Interactions & Dependencies

- Active only when `dcn_bandwidth_limit` is configured.

---

## 5. Practical Scenarios & Failure Modes

- Allows measuring whether DiLoCo's local step cadence ($H$) successfully amortizes a 50ms sync penalty behind compute.

---

### One-line intuition

> **`dcn_bandwidth_latency` sets the artificial network delay (e.g. `"50ms"`) used by Linux traffic control to simulate cross-region propagation latency.**
