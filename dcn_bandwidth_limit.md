## 1. Why does `dcn_bandwidth_limit` exist?

Developing and verifying distributed low-communication algorithms (like DiLoCo) inside a single high-speed datacenter cannot reveal how the system behaves when inter-replica links are throttled to realistic cross-datacenter speeds (e.g. 10 Gbps or 1 Gbps).

Rather than requiring physical cross-country hardware testbeds, MaxText integrates Linux Traffic Control (`tc`) emulation to artificially throttle network interfaces for controlled testing:

```text
Simulated DCN Throttling:
Worker VM [eth0] ──>[Linux tc Token Bucket Filter: dcn_bandwidth_limit = "10gbit"]──> Constrained Link
```

`dcn_bandwidth_limit` specifies the artificial throughput cap applied to network interfaces for DiLoCo testing.

---

## 2. Fundamentals & Mechanics

- Configures the `rate` parameter in Linux `tc qdisc` (token bucket filter).
- **Default `""` (Empty):** No throttling applied; network runs at full physical wire speed.
- Expressed as a bandwidth string, e.g. `"10gbit"`, `"1gbit"`, `"500mbit"`.

---

## 3. Options & Defaults

| Option | Value | Meaning |
|---|---|---|
| Default | `""` | No artificial throttling (full physical line rate). |
| 10 Gbps Emulation | `"10gbit"` | Simulates standard cross-datacenter WAN link. |
| 1 Gbps Emulation | `"1gbit"` | Simulates highly constrained cross-regional network. |

---

## 4. Interactions & Dependencies

```text
dcn_bandwidth_limit ──> Configures Linux tc alongside burst, latency, and interface
```

---

## 5. Practical Scenarios & Failure Modes

- **Testing Only:** Strictly a simulation tool. Never enable `dcn_bandwidth_limit` on a production run.

---

### One-line intuition

> **`dcn_bandwidth_limit` sets an artificial bandwidth limit (e.g. `"10gbit"`) via Linux traffic control to simulate slow cross-datacenter links during testing.**
