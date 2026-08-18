## 1. Why does `hide_profiler_step_metric` exist?

When profiling is active, MaxText tracks internal profiler lifecycle events as metrics.

In automated metric pipelines, dashboards, or continuous integration logging, extra profiler step telemetry can clutter scalar dashboards and pollute metric schemas:

```text
Metric Output Stream:
Default:      [Loss, TFLOPS, Step_Time, Profiler_Step]
Hidden:       [Loss, TFLOPS, Step_Time]  <-- hide_profiler_step_metric: true
```

`hide_profiler_step_metric` suppresses profiler-specific step metrics from stdout logs and metric export sinks.

---

## 2. Fundamentals & Mechanics

- **`false` (Default):** Profiler step metrics are included in standard metric dictionaries.
- **`true`:** Filters out profiler step metrics before writing to GCS or stdout.

---

## 3. Options & Defaults

| Option | Value | Meaning |
|---|---|---|
| Default | `false` | Shows profiler step metric in logs. |
| Filtered | `true` | Hides profiler step metric from metric streams. |

---

## 4. Interactions & Dependencies

- Interacts with `gcs_metrics` and `log_period`.

---

## 5. Practical Scenarios & Failure Modes

- Useful for automated benchmark harnesses that parse fixed-column stdout metric tables.

---

### One-line intuition

> **`hide_profiler_step_metric` prevents internal profiler step counters from cluttering standard output and exported metric streams.**
