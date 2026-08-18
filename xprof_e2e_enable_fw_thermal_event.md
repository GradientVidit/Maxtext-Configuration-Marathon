## 1. Why it exists: correlating chip junction temperatures with kernel execution

At high computational workloads (such as sustained 70B+ FP8 training or continuous serving), TPU silicon dies dissipate hundreds of watts. Thermal sensors distributed across the accelerator package constantly monitor junction temperatures ($T_j$):

```text
Thermal Inflow & Event Thresholds:
┌────────────────────────────────────────────────────────┐
│ Safe Operating Temperature (< 75°C)                    │
│   ──> Standard full-performance execution              │
├────────────────────────────────────────────────────────┤
│ Thermal Warning Threshold (75°C – 85°C)                │
│   ──> Emits FW_THERMAL_EVENT warning                   │
├────────────────────────────────────────────────────────┤
│ Critical Thermal Throttle Threshold (> 85°C)           │
│   ──> Firmware clamps clock frequency to protect die   │
└────────────────────────────────────────────────────────┘
```

When diagnosing performance degradation across multi-rack TPU pods, cooling variances (such as a blocked airflow baffle, liquid cooling manifold pressure drop, or hot-aisle recirculation) cause specific chips to run hotter than their neighbors.

`xprof_e2e_enable_fw_thermal_event` logs firmware-level thermal warning and temperature zone crossing events into Xprof profiling traces.

---

## 2. Mechanics: thermal sensor event logging

When `xprof_e2e_enable_fw_thermal_event: true`:

```text
 During Profiling Window:
 TPU On-Die Temperature Sensors Polled by System Firmware
                     │
                     ▼
 Sensor Crosses Configured Thermal Trip-Point / Threshold
                     │
                     ▼
 Emit Structured Firmware Event:
 ┌───────────────────────────────────────────────────────┐
 │ Event: `FW_THERMAL_EVENT`                             │
 │ Timestamp: 1723984123890 us                           │
 │ Metric: Zone Temp = 82.4°C (Threshold = 80.0°C)       │
 │ Sensor ID: TPU_DIE_CORE_TEMP_SENSOR_3                 │
 └───────────────────┬───────────────────────────────────┘
                     │
                     ▼
 Rendered as a Thermal Warning Marker in TensorBoard Xprof
```

By placing thermal events directly on the unified Xprof timeline alongside XLA HLO operations, developers can pinpoint the exact forward or backward matrix multiplication that pushed junction temperatures past thermal limits.

---

## 3. Options & Configuration

Default in `base.yml`:

```yaml
xprof_e2e_enable_fw_thermal_event: false
```

| Value | Thermal Event Logging | Impact on Profiler Trace | Recommended Use Case |
|---|---|---|---|
| `false` (default) | Disabled. | No thermal markers. | Standard profiling. |
| `true` | Enabled. | Logs thermal zone warnings and sensor trip points onto the firmware timeline. | Investigating hardware cooling issues, cluster thermal imbalances, and validating rack cooling under heavy FP8/BF16 stress tests. |

---

## 4. Parameter Interactions

```text
┌───────────────────────────────────────────────────────────┐
│            xprof_e2e_enable_fw_thermal_event              │
└─────────────┬───────────────────────────────┬─────────────┘
              │ (requires TPU environment)
              ▼
┌───────────────────────────────────────────────────────────┐
│ Interacts directly with:                                  │
│ - profile_power_events (master power/thermal switch)      │
│ - xprof_e2e_enable_fw_throttle_event (throttle reaction)  │
│ - xprof_tpu_power_trace_level                             │
└───────────────────────────────────────────────────────────┘
```

- **`profile_power_events`**: Master boolean toggle; must be set to `true` to collect firmware telemetry.
- **`xprof_e2e_enable_fw_throttle_event`**: Thermal events typically precede throttle events; enabling both allows tracing the causal chain: **High Workload $\to$ Thermal Event $\to$ Throttle Event**.

---

## 5. Practical Scenarios & Failure Modes

### Detecting Rack-Level Cooling Degradation
If training throughput unexpectedly drops by 10% during hot summer afternoons:
```yaml
profiler: "xprof"
profiler_steps: 10
profile_power_events: true
xprof_e2e_enable_fw_thermal_event: true
xprof_e2e_enable_fw_throttle_event: true
```
The resulting trace demonstrates that chips located in upper rack positions triggered thermal events during long all-reduce communication phases.

### What breaks if misconfigured:
- **GPU Cluster Usage**: TPU firmware telemetry calls will fail or corrupt trace metadata on Nvidia GPU instances.

---

### One-line intuition

> **`xprof_e2e_enable_fw_thermal_event` records on-die thermal sensor threshold crossings into the Xprof timeline, identifying hotspots and cooling issues that trigger hardware throttling.**
