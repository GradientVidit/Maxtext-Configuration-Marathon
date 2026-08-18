## 1. Why does `dcn_bandwidth_interface` exist?

Linux hosts have multiple network interfaces (e.g. `lo` for loopback, `eth0` for primary NIC, `gce-nic`, or specialized TPU/GPU interconnect adapters).

The traffic control throttling rules must bind to the specific network interface handling inter-VM communication:

```text
Host Network Devices:
  lo      (Loopback)          ── [No throttle]
  eth0    (Inter-VM Ethernet) ── [Linux tc Rules Applied here]
```

`dcn_bandwidth_interface` specifies the target network interface name where bandwidth throttling rules are attached.

---

## 2. Fundamentals & Mechanics

- Default `"eth0"` targets the standard default Ethernet adapter on Cloud VMs.
- MaxText executes `tc` commands targeting this interface during setup if `dcn_bandwidth_limit` is non-empty.

---

## 3. Options & Defaults

| Option | Value | Meaning |
|---|---|---|
| Default | `"eth0"` | Standard primary network interface on Linux VMs. |
| Custom NIC | `"eth1"`, `"ens4"` | Custom or secondary physical interface. |

---

## 4. Interactions & Dependencies

- Targets the interface modified by `dcn_bandwidth_limit`, `dcn_bandwidth_burst`, and `dcn_bandwidth_latency`.

---

## 5. Practical Scenarios & Failure Modes

- Specifying an interface name that does not exist on the VM (e.g. `"eth0"` on a machine named `"ens3"`) will cause traffic control setup commands to fail.

---

### One-line intuition

> **`dcn_bandwidth_interface` specifies the network interface (default `"eth0"`) to which simulated bandwidth throttling rules are applied.**
