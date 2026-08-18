## 1. Why does `dump_step` exist?

Dumping HLO modules on every step is redundant (since JIT-compiled graphs are identical across steps) and generates enormous disk I/O.

Conversely, dynamic features like input pipeline caching, dynamic shapes, or runtime state transitions might only stabilize after several warm steps:

```text
Step Timeline:
Step 0 ──> Step 1 ──> Step 2 ──>[ Step 3: Trigger HLO Dump ]──> Step 4...
                                 └───────────┬────────────┘
                                             ▼
                                      dump_step = 3
```

`dump_step` specifies the exact training step number at which HLO modules are captured.

---

## 2. Fundamentals & Mechanics

- **`-1` (Default):** Modules are dumped at compilation time before step execution begins.
- **`N > 0`:** Bypasses dumping during initial step 0 and triggers HLO serialization specifically at step $N$.

---

## 3. Options & Defaults

| Option | Value | Meaning |
|---|---|---|
| Default | `-1` | Dumps modules during initial compilation / startup. |
| Targeted Step | `N > 0` | Captures HLO specifically at training step $N$. |

---

## 4. Interactions & Dependencies

- Active when `dump_hlo: true`.

---

## 5. Practical Scenarios & Failure Modes

- Useful when diagnosing runtime recompilations that occur at specific step thresholds (e.g. at the end of warmup).

---

### One-line intuition

> **`dump_step` sets the specific training step at which HLO module dumps are captured, defaulting to `-1` for startup compilation dumping.**
