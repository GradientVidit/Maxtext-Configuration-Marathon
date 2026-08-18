## 1. Why it exists: hardware-level energy and power telemetry in Xprof

When training large language models on high-density TPU pods (such as TPU v5p or TPU v6e Trillium clusters), hardware performance is deeply intertwined with **power delivery and thermal management**:

```text
Without Power Tracing (xprof_tpu_power_trace_level: 0):
Profiler Trace shows: [HLO Op: DotGeneral] ──> Duration: 4.2ms (Slow!)
Question: Was it slow because of memory bandwidth, compiler code generation,
          or did the chip throttle because power draw exceeded the rack limit?
          (Impossible to know)

With Power Tracing (xprof_tpu_power_trace_level: 1 or 2):
Profiler Trace shows:
  [HLO Op: DotGeneral]  ═════════════════════════════════════════> (4.2ms)
  [TPU Power Rail (W)]  ──/\──/\─── 280W ─── 450W ───/\───────────>
  [SPI / Current Sensor] ──────────────────── [Peak Current Alert]
Diagnosis: Chip hit power/thermal throttling during peak GEMM execution.
```

High-power workloads with dense matrix multiplies can cause sudden current transients ($dI/dt$) and voltage drops on the power rails. Understanding whether step-time jitter is caused by algorithmic inefficiencies or hardware power throttling requires sampling TPU power sensors in lockstep with the XLA execution timeline.

`xprof_tpu_power_trace_level` configures the granularity and detail level of TPU power telemetry captured during Xprof profiling sessions.

---

## 2. Mechanics: sampling TPU power rails and SPI sensors

During profiling, Xprof interfaces with the low-level TPU device driver to record power metrics into the profiler's XPlane trace buffer:

```text
                          Xprof Profiler Capture Session
                                         │
                 ┌───────────────────────┴───────────────────────┐
                 ▼                                               ▼
         Level 0: NONE                                    Level 1: NORMAL
  ┌─────────────────────────────┐                 ┌─────────────────────────────┐
  │ No power telemetry recorded │                 │ Samples board-level power   │
  │ Zero driver polling overhead│                 │ sensors at regular intervals│
  └─────────────────────────────┘                 │ (Average wattage per chip)  │
                                                  └──────────────┬──────────────┘
                                                                 │
                                                                 ▼
                                                          Level 2: SPI
                                                  ┌─────────────────────────────┐
                                                  │ High-frequency direct SPI   │
                                                  │ bus telemetry; microsecond- │
                                                  │ level current & voltage spikes│
                                                  └─────────────────────────────┘
```

1. When set to `1` (`POWER_TRACE_NORMAL`), Xprof queries driver power counters, rendering continuous power draw (Watts) curves on the TensorBoard Xprof timeline.
2. When set to `2` (`POWER_TRACE_SPI`), Xprof taps high-speed Serial Peripheral Interface (SPI) hardware telemetry buses, providing high-frequency voltage/current traces to analyze instantaneous power spikes.

---

## 3. Options & Configuration

Default in `base.yml`:

```yaml
xprof_tpu_power_trace_level: 0
```

| Level | Enum Name | Telemetry Detail | Hardware Overhead | Use Case |
|---|---|---|---|---|
| `0` (default) | `POWER_TRACE_NONE` | Disabled. No power data captured. | Zero | Standard profiling runs where only compute/memory timeline is needed. |
| `1` | `POWER_TRACE_NORMAL` | Standard board-level power sensor sampling (Watts). | Low | Investigating sustained thermal throttling, power capping, and energy efficiency. |
| `2` | `POWER_TRACE_SPI` | High-frequency SPI bus power, voltage, and current sampling. | Moderate | Deep hardware bring-up, transient $dI/dt$ power drop investigation, TPU power rail debugging. |

---

## 4. Parameter Interactions

```text
┌───────────────────────────────────────────────────────────┐
│                xprof_tpu_power_trace_level                │
└─────────────┬───────────────────────────────┬─────────────┘
              │ (requires TPU environment)
              ▼
┌───────────────────────────────────────────────────────────┐
│ Interacts directly with:                                  │
│ - profile_power_events (master switch for TPU power trace)│
│ - xprof_e2e_enable_fw_power_level_event                   │
│ - xprof_e2e_enable_fw_throttle_event                      │
│ - xprof_e2e_enable_fw_thermal_event                       │
└───────────────────────────────────────────────────────────┘
```

- **`profile_power_events`**: Master boolean toggle. Both `profile_power_events: true` and `xprof_tpu_power_trace_level > 0` are used to capture full power analytics.
- **TPU vs GPU**: Power tracing flags are **strictly TPU-specific**. Do not enable these on GPU clusters, as TPU-specific driver hooks will cause GPU XPlane tracing to fail.

---

## 5. Practical Scenarios & Failure Modes

### Diagnosing Step-Time Jitter on Dense Pods
When step time randomly spikes by 25% on a TPU v5p-512 pod:
```yaml
profiler: "xprof"
profiler_steps: 10
profile_power_events: true
xprof_tpu_power_trace_level: 1
xprof_e2e_enable_fw_throttle_event: true
```
Opening the trace in TensorBoard reveals that power consumption peaked at the facility breaker limit, triggering brief hardware throttling on specific host nodes.

### What breaks if misconfigured:
- **Running on GPUs**: Enabling `xprof_tpu_power_trace_level` on Nvidia GPU clusters will corrupt profiler event buffers and break Nvidia NSYS / XPlane tracing.

---

### One-line intuition

> **`xprof_tpu_power_trace_level` sets the granularity of TPU power telemetry in Xprof traces (0=None, 1=Normal, 2=High-Frequency SPI), revealing whether performance slowdowns stem from hardware power and voltage limits.**
