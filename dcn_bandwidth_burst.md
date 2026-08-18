## 1. Why does `dcn_bandwidth_burst` exist?

In network traffic shaping using token bucket filters (TBF), real network traffic is bursty: packets arrive in rapid trains rather than perfectly spaced intervals.

The bucket burst size determines how many bytes can be transmitted at line rate before the rate limiter enforces the average bandwidth cap:

```text
Token Bucket Filter:
Incoming Packet Train ──>[ Token Bucket Capacity: dcn_bandwidth_burst = "10mb" ]──> Rate Limiter
```

`dcn_bandwidth_burst` defines the bucket size parameter for Linux traffic control token bucket emulation.

---

## 2. Fundamentals & Mechanics

- Passes the `burst` parameter to Linux `tc`.
- Default `"10mb"` prevents artificial packet drop on initial burst transmissions during collective sync steps.

---

## 3. Options & Defaults

| Option | Value | Meaning |
|---|---|---|
| Default | `"10mb"` | 10 Megabyte burst buffer size. |
| Custom Size | `"5mb"`, `"20mb"` | Fine-tuned burst tolerance for custom traffic profiles. |

---

## 4. Interactions & Dependencies

- Active only when `dcn_bandwidth_limit` is set non-empty.

---

## 5. Practical Scenarios & Failure Modes

- Setting burst too low (e.g. `"10kb"`) with high bandwidth limits causes Linux kernel packet drops and connection resets during all-reduce bursts.

---

### One-line intuition

> **`dcn_bandwidth_burst` sets the token bucket burst capacity for Linux traffic control bandwidth throttling emulation.**
