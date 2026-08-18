## 1. Why does `managed_mldiagnostics_on_demand_profiling` exist?

Pre-scheduled profiling (e.g. at step 100) cannot capture unexpected live anomalies (such as a sudden slowdown occurring 14 hours into a run).

Starting an on-demand profiling background RPC server allows operators to trigger live hardware trace captures from the Google Cloud Console at any arbitrary moment during training:

```text
GCP Cloud Console ("Capture Profile Now" Button)
                     │
                     ▼ (RPC Trigger)
MaxText Worker [ On-Demand Profiling Server ] ──> Captures Live XPlane Trace on the fly
```

`managed_mldiagnostics_on_demand_profiling` enables the live on-demand profiling server.

---

## 2. Fundamentals & Mechanics

- **`true` (Default when ML diagnostics is enabled):** Spawns an internal HTTP/RPC endpoint listening for manual trigger events from Cloud Diagnostics.
- **`false`:** Profiling occurs strictly based on pre-configured config step triggers.

---

## 3. Options & Defaults

| Option | Value | Meaning |
|---|---|---|
| Default | `true` | On-demand profile server active when diagnostics is enabled. |
| Disabled | `false` | Bypasses on-demand server. |

---

## 4. Interactions & Dependencies

- Active when `managed_mldiagnostics: true`.

---

## 5. Practical Scenarios & Failure Modes

- Crucial for live triage when training throughput inexplicably degrades midway through a multi-day run.

---

### One-line intuition

> **`managed_mldiagnostics_on_demand_profiling` enables an RPC server allowing engineers to trigger live hardware profiling on-demand from the Cloud Console.**
