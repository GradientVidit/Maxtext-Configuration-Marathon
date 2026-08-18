## 1. Why does `colocated_python_data_input` exist?

In Google Cloud TPU architectures running under **Pathways** (Single Controller runtime), the system can either run data loading in decoupled worker processes or colocate the Python data input runtime directly inside the TPU compute worker environment.

```text
Decoupled Architecture:
[ Data Input Process ] ──(Network/RPC)──> [ Pathways Compute Worker ] ──> TPU

Colocated Architecture (colocated_python_data_input: true):
┌────────────────────────────────────────┐
│ Pathways Worker                        │
│ [ Python Data Input ] ──(Direct Memory)──> TPU │
└────────────────────────────────────────┘
```

`colocated_python_data_input: true` colocates the data input execution path directly with compute workers to eliminate RPC overhead under Pathways.

---

## 2. Mechanics & Status

This is an experimental optimization specifically developed for Pathways runtime environments on Cloud TPU.

---

## 3. Options & Default

| Parameter | Type | Default | Options |
| :--- | :--- | :--- | :--- |
| `colocated_python_data_input` | `bool` | `false` | `true` (colocate under Pathways), `false` (standard data loading) |

---

## 4. Interactions with Related Parameters

- **`colocated_python_checkpointing`**: Companion Pathways optimization for checkpoint I/O.

---

## 5. Practical Scenarios & Failure Modes

| Scenario | Symptom / Failure | Fix |
| :--- | :--- | :--- |
| **Standard JAX multihost training (Non-Pathways)** | Parameter has no effect or is unsupported | Keep default `false`. |

---

### One-line intuition

> `colocated_python_data_input` colocates the Python data loading pipeline with compute workers under the Pathways Single Controller runtime.
