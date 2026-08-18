## 1. Why it exists: the master switch for hardware power/thermal profiling

Antigravity and MaxText run across multiple hardware backends—principally Google Cloud TPUs and Nvidia GPUs. While standard software tracing (op timelines, memory allocations, communication events) is standardized via XPlane protobufs, **power and thermal telemetry relies on proprietary, hardware-specific driver interfaces**:

```text
TPU Tracing Infrastructure:
  Xprof ──> TPU Driver IOCTLs / SPI Bus ──> Power / Thermal / Throttle Events

GPU Tracing Infrastructure (Nvidia NVLink / CUPTI / NSYS):
  Xprof / NSYS ──> CUPTI Events / NVML ──> GPU Compute & Memory Events
```

If TPU-specific power and thermal event hooks are unconditionally enabled, the profiler attempts to invoke TPU driver IOCTLs on GPU nodes, causing **GPU XPlane tracing to crash or produce corrupted trace files**.

`profile_power_events` is the master boolean switch that activates TPU-specific power, voltage, and thermal event logging during Xprof profiling sessions, defaulting to `false` to maintain cross-platform safety.

---

## 2. Mechanics: gating low-level hardware tracing hooks

When initializing an Xprof profiling session:

```text
                     Start Xprof Profiling Session
                                   │
                                   ▼
                   Check: `profile_power_events: true`
                                   │
                 ┌─────────────────┴─────────────────┐
                 ▼                                   ▼
     `profile_power_events: false`       `profile_power_events: true`
  ┌─────────────────────────────┐     ┌─────────────────────────────────────┐
  │ Skip TPU power driver hooks │     │ Initialize TPU Hardware Subsystems: │
  │ Safe on GPUs and CPU hosts  │     │ - Tap SPI power bus counters        │
  │ Records compute & memory    │     │ - Hook firmware throttle events     │
  │ timeline only               │     │ - Hook thermal sensor trip points   │
  └─────────────────────────────┘     │ - Sample power rail wattage (W)     │
                                      └─────────────────────────────────────┘
```

When set to `true` on a TPU system:
- It enables the driver hooks that power `xprof_tpu_power_trace_level`, `xprof_e2e_enable_fw_throttle_event`, `xprof_e2e_enable_fw_power_level_event`, and `xprof_e2e_enable_fw_thermal_event`.
- The power and thermal event streams are multiplexed into the main XPlane execution protobuf for TensorBoard visualization.

---

## 3. Options & Configuration

Default in `base.yml`:

```yaml
profile_power_events: false
```

| Setting | Hardware Behavior | Safety Profile | When to Use |
|---|---|---|---|
| `false` (default) | Power/thermal driver hooks are disabled. | 100% safe on TPUs, GPUs, and CPUs. | Default training runs, multi-platform environments, GPU cluster profiling. |
| `true` | Activates TPU power and thermal tracing subsystems. | TPU-only. (Do not enable on GPUs). | Deep performance optimization and thermal debugging on Google Cloud TPU v4, v5e, v5p, and v6e pods. |

---

## 4. Parameter Interactions

```text
┌───────────────────────────────────────────────────────────┐
│                   profile_power_events                    │
└─────────────┬─────────────────────────────────────────────┘
              │ (Master switch - enables child flags)
              ▼
┌───────────────────────────────────────────────────────────┐
│ Child Hardware Telemetry Flags:                           │
│ - xprof_tpu_power_trace_level                             │
│ - xprof_e2e_enable_fw_throttle_event                      │
│ - xprof_e2e_enable_fw_power_level_event                   │
│ - xprof_e2e_enable_fw_thermal_event                       │
└───────────────────────────────────────────────────────────┘
```

- **Child parameters require the master switch**: Setting `xprof_e2e_enable_fw_throttle_event: true` or `xprof_tpu_power_trace_level: 1` will have no effect unless `profile_power_events: true` is also set.
- **`profiler`**: Must be set to `"xprof"` for power event collection.

---

## 5. Practical Scenarios & Failure Modes

### Full TPU Power & Thermal Diagnosis
To capture a complete hardware diagnostic trace on TPU v5p:
```yaml
profiler: "xprof"
profiler_steps: 10
profile_power_events: true
xprof_tpu_power_trace_level: 1
xprof_e2e_enable_fw_throttle_event: true
xprof_e2e_enable_fw_thermal_event: true
```

### What breaks if misconfigured:
- **GPU Profiling Crashes**: Enabling `profile_power_events: true` on an Nvidia A100/H100 GPU cluster will cause Xprof XPlane trace generation to fail with driver interface errors.

---

### One-line intuition

> **`profile_power_events` is the master toggle for TPU-specific power, thermal, and firmware throttle tracing in Xprof, disabled by default to prevent breaking GPU profiling runs.**
