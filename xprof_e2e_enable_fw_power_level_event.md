## 1. Why it exists: tracing dynamic power states and rail transitions

Modern AI accelerator systems (including Google Cloud TPUs) utilize dynamic power state transitions to manage electrical loads across massive data-center clusters:

```text
Low Power / Idle State (P-State 1):
[Waiting for Input / Data Pipeline Delay] ──> Power State: 120W (Reduced clock/voltages)
                                                   │
                                                   ▼ Sudden Transition
Peak Execution State (P-State 0):
[Launch Megacore FlashAttention GEMM]     ──> Power State: 480W (Max voltage rail)
```

Rapid transitions between power states cause steep electrical transients ($dI/dt$), which can induce voltage droop on the motherboard power planes. If power management firmware dynamically shifts power levels or clamps maximum wattage during execution, training steps can experience unexpected latency jitter.

`xprof_e2e_enable_fw_power_level_event` enables end-to-end tracing of firmware-initiated power-level transition events directly within Xprof profiling timelines.

---

## 2. Mechanics: logging power state changes into XPlane

When `xprof_e2e_enable_fw_power_level_event: true`:

```text
 MaxText Profiler Session Running
                │
                ▼
 TPU Driver Hooks Monitor System Management Bus (SMBus/PMBus)
                │
                ▼
 Event Detected: Power Level State Transition (e.g., Level_1 -> Level_0)
                │
                ▼
 Record Timestamped Event into XPlane Protobuf:
 ┌───────────────────────────────────────────────────────────┐
 │ Event: `FW_POWER_LEVEL_CHANGE`                            │
 │ Attributes:                                               │
 │   - from_level: "ECO / STANDBY"                           │
 │   - to_level: "BOOST_MAX"                                 │
 │   - timestamp_us: 1723984123456                           │
 └─────────────────────────────┬─────────────────────────────┘
                               │
                               ▼
 Rendered as an Event Marker on the TensorBoard Xprof Timeline
```

The recorded events allow performance engineers to correlate the exact moment when the TPU enters, leaves, or transitions between power tiers with the execution of specific JAX operations (like forward-pass GEMMs or collective communication steps).

---

## 3. Options & Configuration

Default in `base.yml`:

```yaml
xprof_e2e_enable_fw_power_level_event: false
```

| Value | Power Level Tracing | Visualized in Trace | Use Case |
|---|---|---|---|
| `false` (default) | Disabled. | No power state markers. | Standard day-to-day training and inference profiling. |
| `true` | Enabled. | Power state transition markers on the firmware timeline track. | Low-level hardware qualification, power-capping analysis, diagnosing power-ramp step latency spikes. |

---

## 4. Parameter Interactions

```text
┌───────────────────────────────────────────────────────────┐
│           xprof_e2e_enable_fw_power_level_event           │
└─────────────┬───────────────────────────────┬─────────────┘
              │ (requires TPU environment)
              ▼
┌───────────────────────────────────────────────────────────┐
│ Interacts directly with:                                  │
│ - profile_power_events (master switch)                    │
│ - xprof_tpu_power_trace_level (continuous wattage curve)  │
│ - xprof_e2e_enable_fw_throttle_event                     │
└───────────────────────────────────────────────────────────┘
```

- **`profile_power_events`**: Master switch; must be enabled to activate firmware power telemetry.
- **`xprof_tpu_power_trace_level`**: Continuous wattage sampling (`level: 1` or `2`) complements discrete power-level transition event markers.

---

## 5. Practical Scenarios & Failure Modes

### Investigating Warmup Latency & Step Ramp-up
During the first few steps of a training run, chips transition from idle states to full power states. Capturing `xprof_e2e_enable_fw_power_level_event` verifies that all chips in a pod have completed power-ramp transitions and settled into their highest-performance power states.

### What breaks if misconfigured:
- **TPU-only compatibility**: Enabling this parameter on GPU clusters will cause Xprof driver initialization errors.

---

### One-line intuition

> **`xprof_e2e_enable_fw_power_level_event` records firmware power-state transition events onto the Xprof timeline, helping engineers correlate hardware power tier changes with step execution performance.**
