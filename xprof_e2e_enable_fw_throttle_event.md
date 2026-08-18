## 1. Why it exists: capturing firmware-level clock throttling events in profiler traces

On modern accelerator hardware like Google Cloud TPUs, onboard microcontrollers and system firmware constantly monitor die temperatures, rail voltages, and power limits. When safe operating boundaries are approached, the firmware automatically engages **clock throttling (frequency downclocking)** to protect the silicon:

```text
Normal Operation (Full Clock Speed, e.g. 1.2 GHz):
[Matrix Multiplication: DotGeneral] ═════════════════> Finished in 3.1ms

Firmware Throttle Engaged (Throttled Clock Speed, e.g. 700 MHz):
[Matrix Multiplication: DotGeneral] ═════════════════════════════════════════> Finished in 5.8ms
                                    ▲                                       ▲
                                    └─── [FW Throttle Event: Temp > 85°C] ──┘
```

Without explicit firmware event logging, an engineer profiling slow steps in Xprof sees an unexpectedly long HLO operation duration but has no visibility into *why* the kernel ran slowly. The engineer might waste days trying to rewrite the JAX/XLA model code when the root cause was physical thermal throttling.

`xprof_e2e_enable_fw_throttle_event` enables end-to-end tracing of firmware-level throttle start/stop events directly on the Xprof / TensorBoard timeline.

---

## 2. Mechanics: firmware event hooks and timeline integration

When `xprof_e2e_enable_fw_throttle_event: true`:

```text
 Xprof Profiler Tracing Session
               │
               ▼
 ┌───────────────────────────────────────────────────────────┐
 │ Subscribe to Low-Level TPU Driver Firmware Event Queue    │
 └─────────────────────────────┬─────────────────────────────┘
                               │
               ┌───────────────┴───────────────┐
               ▼                               ▼
     Software / HLO Timeline         Firmware Event Timeline
 ┌─────────────────────────────┐ ┌─────────────────────────────┐
 │ Step 42: Forward Pass       │ │ No throttle events          │
 │ Step 43: All-Gather + GEMM  │ │ ──[THROTTLE_START (Thermal)]│
 │          (Duration Spikes!) │ │   Duration: 45ms            │
 │                             │ │ ──[THROTTLE_END]            │
 └─────────────────────────────┘ └─────────────────────────────┘
```

1. During an Xprof trace window, the profiler driver hooks into the TPU firmware event bus.
2. Whenever the hardware firmware downclocks the core clock due to thermal thresholds, current draw ($dI/dt$), or power-cap limits, a structured `THROTTLE_EVENT` timestamp is written into the XPlane protobuf.
3. TensorBoard visualizes this event as a distinct timeline track aligned with the executing JAX ops.

---

## 3. Options & Configuration

Default in `base.yml`:

```yaml
xprof_e2e_enable_fw_throttle_event: false
```

| Value | Behavior | Timeline Visualization | Recommended Use Case |
|---|---|---|---|
| `false` (default) | Firmware throttle events are not captured in profiler traces. | Compute/HLO timeline only. | Standard day-to-day training and debugging. |
| `true` | Captures firmware downclocking events with microsecond timestamps. | Adds a dedicated "Firmware Throttle" track in TensorBoard Xprof. | Investigating mysterious step-time spikes, cluster thermal imbalances, or hardware qualification on new TPU generations. |

---

## 4. Parameter Interactions

```text
┌───────────────────────────────────────────────────────────┐
│            xprof_e2e_enable_fw_throttle_event             │
└─────────────┬───────────────────────────────┬─────────────┘
              │ (requires TPU environment)
              ▼
┌───────────────────────────────────────────────────────────┐
│ Interacts directly with:                                  │
│ - profile_power_events (master power/thermal switch)      │
│ - xprof_e2e_enable_fw_thermal_event (temperature events) │
│ - xprof_e2e_enable_fw_power_level_event                  │
│ - xprof_tpu_power_trace_level                             │
└───────────────────────────────────────────────────────────┘
```

- **`profile_power_events`**: Master switch; ensure `profile_power_events: true` is enabled alongside firmware event flags.
- **`xprof_e2e_enable_fw_thermal_event`**: Pairs naturally with thermal tracing to correlate temperature spikes with the exact throttle onset timestamps.

---

## 5. Practical Scenarios & Failure Modes

### Pinpointing Hot Nodes in a TPU Pod
When running across a large TPU pod (e.g. 512 chips), a single host in a poorly ventilated rack might experience thermal throttling, slowing down the entire pod due to all-reduce synchronization barriers.
Enabling `xprof_e2e_enable_fw_throttle_event: true` immediately highlights the specific chip experiencing throttle events on the profile timeline.

### What breaks if misconfigured:
- **GPU Incompatibility**: These firmware event flags query Google TPU driver interfaces. Enabling them on GPU instances will cause profiling session errors.

---

### One-line intuition

> **`xprof_e2e_enable_fw_throttle_event` captures firmware-level hardware clock throttling events into Xprof traces, identifying when hardware downclocking caused step-time slowdowns.**
