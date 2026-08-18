## 1. Why does `skip_step_scaling_factor` exist?

Loss and gradient norms naturally fluctuate between batches due to varying sequence difficulty and token distributions.

The spike detection system must distinguish between natural batch variance and true catastrophic gradient explosions:

```text
Threshold Boundary:

Metric ^
       │           * True Anomaly (Spike > μ + 6σ) -> SKIPPED
───────┼───────────────────────────────────────────── μ + 6.0 * σ
       │     /       │    /  \   /\ (Natural variance < μ + 6σ) -> ALLOWED
───────┼───/────\─/──\─────────────────────────────── μ (Mean)
       └─────────────────────────────────────────────> Steps
```

`skip_step_scaling_factor` defines how many standard deviations above the rolling mean a metric must rise to be classified as a skippable spike.

---

## 2. Fundamentals & Mechanics

A step at time $t$ with metric $M_t$ is skipped if:

$$M_t > \mu_{\text{rolling}} + \text{skip\_step\_scaling\_factor} \times \sigma_{\text{rolling}}$$

- Default `6.0` ($6\sigma$) represents a standard $p < 10^{-8}$ outlier threshold under normal distributions, ensuring only severe anomalies are intercepted.

---

## 3. Options & Defaults

| Option | Value | Meaning |
|---|---|---|
| Default | `6.0` | Strict $6\sigma$ threshold (only extreme explosions skipped). |
| Aggressive | `3.0` to `4.0` | Tighter bounds for fragile architectures or early exploration. |
| Relaxed | `8.0` to `10.0` | Ultra-conservative threshold for highly variable multimodal mixtures. |

---

## 4. Interactions & Dependencies

```text
skip_step_on_spikes: true ──> Uses skip_step_scaling_factor * σ
```

---

## 5. Practical Scenarios & Failure Modes

- Setting `skip_step_scaling_factor` too low (e.g. `1.5`) will cause normal batches with high token entropy to be skipped repeatedly, freezing training progress.

---

### One-line intuition

> **`skip_step_scaling_factor` sets the standard deviation multiplier ($K\sigma$) above the rolling mean required to trigger automatic step skipping.**
