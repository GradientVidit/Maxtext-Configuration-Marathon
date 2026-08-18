## 1. Why does `step_deviation_interval_seconds` exist?

Evaluating step time statistical deviation over too short a window causes false alarms from normal step variations (such as periodic checkpoint saves or data buffer prefetches). Evaluating over too long a window delays alerting when a true hardware slowdown occurs.

`step_deviation_interval_seconds` defines the time window cadence for aggregating step time samples and evaluating deviation metrics.

---

## 2. What it actually controls

```yaml
step_deviation_interval_seconds: 30
```

- Sets the window interval (in seconds) at which step duration statistics (mean, variance, outliers) are calculated and published.

---

## 3. Options and Defaults

| Value | Sampling Window | Alert Responsiveness |
|---|---|---|
| `30` (default) | 30-second evaluation cadence | Balanced, standard production setting |
| `10` | 10-second cadence | Rapid response; sensitive to transient spikes |
| `60` | 60-second cadence | Smooth rolling average |

---

## 4. Interactions

- **`monitor_step_time_deviation`**: Must be `true` for this interval to be active.

---

## 5. Practical Scenarios

- **Default Recommendation**: Keep default `30`.

---

### One-line intuition

> **`step_deviation_interval_seconds` sets the periodic time window in seconds for evaluating step duration anomalies.**
