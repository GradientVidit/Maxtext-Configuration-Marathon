## 1. Why it exists: Pathways-style single controller execution & elastic training

In standard JAX distributed setups, clusters execute in **Multi-Controller SPMD mode**:

```text
Multi-Controller Mode (Standard JAX SPMD, enable_single_controller: false):
┌─────────────────────────┐  ┌─────────────────────────┐  ┌─────────────────────────┐
│ TPU Host 0              │  │ TPU Host 1              │  │ TPU Host N              │
│ Python Process 0        │  │ Python Process 1        │  │ Python Process N        │
│ Coordinates Local Chips │  │ Coordinates Local Chips │  │ Coordinates Local Chips │
└─────────────────────────┘  └─────────────────────────┘  └─────────────────────────┘
──> If ANY single host dies or preempts, ALL Python processes crash. Cluster must reboot.

Single-Controller Mode (Pathways Architecture, enable_single_controller: true):
┌───────────────────────────────────────────────────────────────────────────────────┐
│                     Single Central Python Controller Process                      │
│                  Executes training script & orchestrates graph                    │
└─────────────────────────────────────────┬─────────────────────────────────────────┘
                                          │ Pathways Remote RPC Proxy
        ┌─────────────────────────────────┼─────────────────────────────────┐
        ▼                                 ▼                                 ▼
┌─────────────────────────┐   ┌─────────────────────────┐   ┌─────────────────────────┐
│ TPU Slice 0 (Workers)   │   │ TPU Slice 1 (Workers)   │   │ TPU Slice K (Workers)   │
│ Headless C++ Daemons    │   │ Headless C++ Daemons    │   │ Headless C++ Daemons    │
└─────────────────────────┘   └─────────────────────────┘   └─────────────────────────┘
──> Supports dynamic slice addition/removal, elastic scaling, and multi-slice orchestration.
```

In multi-controller mode, scaling across hundreds of hosts requires synchronizing hundreds of independent Python processes. If one worker fails or encounters a Python GIL hiccup, the entire job halts.

Google's **Pathways** system introduces a single-controller paradigm: a single client Python process coordinates thousands of remote TPU accelerators running headless C++ server daemons. This enables **elastic training**—dynamically adding or removing TPU slices mid-training without killing the Python process.

`enable_single_controller` configures MaxText to run under JAX's Pathways / Single-Controller distributed runtime.

---

## 2. Mechanics: centralized dispatch and remote worker management

When `enable_single_controller: true`:

```text
 1. Launch Single MaxText Python Process (Controller)
                         │
                         ▼
 2. Connect to Pathways Runtime Client Proxy
    - Discover available remote worker slices
    - Construct global unified device mesh across slices
                         │
                         ▼
 3. Compile Global Computation Graph (XLA)
                         │
                         ▼
 4. Dispatch Graph Segments to Remote C++ Workers
    - Workers execute compute kernels & DCN collectives
    - Controller monitors execution progress asynchronously
```

Key runtime differences in single-controller mode:
- **Data loading**: Input data pipelines (`colocated_python_data_input`) execute centrally or route through Pathways data dispatchers.
- **Checkpointing**: Uses `colocated_python_checkpointing` where the single controller coordinates checkpoint metadata.
- **Fault recovery**: If a remote worker slice fails, Pathways manages slice reconfiguration while the main Python controller remains alive.

---

## 3. Options & Configuration

Default in `base.yml`:

```yaml
enable_single_controller: false
```

| Setting | Distributed Architecture | Host Model | Recommended Use Case |
|---|---|---|---|
| `false` (default) | Multi-Controller SPMD | Every TPU VM runs an independent Python interpreter. | Standard Google Cloud TPU VM slices, standard SLURM/GKE pods, open-source JAX training. |
| `true` | Single-Controller (Pathways) | 1 Python process manages headless remote worker pools. | **Elastic training on Google Cloud**, multi-slice dynamic training, Google internal Pathways environments. |

---

## 4. Parameter Interactions

```text
┌───────────────────────────────────────────────────────────┐
│                 enable_single_controller                  │
└─────────────┬───────────────────────────────┬─────────────┘
              │ (when true)
              ▼
┌───────────────────────────────────────────────────────────┐
│ Requires Pathways-compatible configurations:              │
│ - colocated_python_checkpointing                          │
│ - colocated_python_data_input                             │
│ - enable_pathways_goodput                                 │
│ - grain_use_elastic_iterator                              │
└───────────────────────────────────────────────────────────┘
```

- **`colocated_python_checkpointing`**: Controls whether checkpoint saving is handled by the central Python process or delegated.
- **`grain_use_elastic_iterator`**: Enables the Grain data loader to dynamically adjust batch distribution when TPU slices are dynamically added or removed during elastic training.

---

## 5. Practical Scenarios & Failure Modes

### Elastic Multi-Slice Pretraining
When running an elastic pretraining run that dynamically scales from 2 slices to 8 slices depending on spot capacity:
```yaml
enable_single_controller: true
colocated_python_checkpointing: true
grain_use_elastic_iterator: true
enable_pathways_goodput: true
```
The central job controller continues running seamlessly across spot preemption events and capacity adjustments.

### What breaks if misconfigured:
- **Missing Pathways Backend**: Enabling `enable_single_controller: true` in a standard TPU VM environment without the Pathways runtime installed will fail during JAX distributed initialization with missing backend errors.

---

### One-line intuition

> **`enable_single_controller` switches MaxText from multi-process SPMD to a centralized Pathways controller architecture, enabling elastic multi-slice training and dynamic hardware scaling.**
