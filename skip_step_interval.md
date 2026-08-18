## 1. Why does `skip_step_interval` exist?

To detect whether a training loss or gradient norm spike is an anomalous outlier, the system must maintain a running baseline of recent statistical behavior.

If the statistical window is too small, a single noisy step skews the baseline. If it is too large, the baseline fails to adapt to natural loss reduction over the course of training:

```text
Rolling Window for Spike Baseline:

Step: [t-128] .................................. [t-1] ──> Current Step [t]
      └──────────────────┬──────────────────────────┘
                         ▼
        Compute Rolling Mean (μ) and Std (σ)
        Spike Threshold = μ + (scaling_factor * σ)
```

`skip_step_interval` sets the number of recent training steps included in the rolling statistics calculation for spike detection.

---

## 2. Fundamentals & Mechanics

- Specifies the window size $W$ of historical steps stored in rolling buffers.
- Default `128` steps provides a robust sample size for calculating sample mean and standard deviation without stale bias from early training stages.

---

## 3. Options & Defaults

| Option | Value | Meaning |
|---|---|---|
| Default | `128` | 128-step rolling window for mean and variance estimation. |
| Short Window | `64` | Fast-adapting baseline for volatile early training. |
| Long Window | `256` or `512` | Stable baseline for smooth late-stage training. |

---

## 4. Interactions & Dependencies

```text
skip_step_on_spikes: true ──> skip_step_interval ──> skip_step_scaling_factor
```

- Only active when `skip_step_on_spikes: true`.

---

## 5. Practical Scenarios & Failure Modes

- Setting `skip_step_interval` too small (e.g. `4`) makes standard deviation estimates noisy, leading to erratic false-positive step skipping.

---

### One-line intuition

> **`skip_step_interval` defines the rolling step window used to compute the mean and standard deviation for anomaly spike detection.**
