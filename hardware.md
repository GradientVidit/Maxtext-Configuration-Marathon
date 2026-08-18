## 1. Why does it exist?

MaxText is designed from the ground up for high-performance scale across multiple accelerator backends: Google TPUs (v4, v5e, v5p, v6e/Trillium), NVIDIA GPUs (A100, H100, B200), AMD GPUs (MI300/MI325), and CPU environments for testing.

Each hardware platform has fundamentally different:
1. Interconnect architectures (TPU ICI optical torus vs GPU NVLink / NVSwitch / InfiniBand).
2. XLA backend compiler flags and device discovery APIs.
3. Multi-process initialization routines (JAX TPU distributed runtime vs standard CUDA/MPI/NCCL multi-processing).

```text
                     MaxText Core Model Code
                                │
                      [ hardware parameter ]
                                │
     ┌──────────────┬───────────┴───────────┬──────────────┐
     ↓              ↓                       ↓              ↓
   'tpu'          'gpu'             'gpu_multiprocess'   'cpu'
 (Google TPU    (Single-host        (Multi-node GPU    (Local unit test /
  Pods / ICI)    NVIDIA/AMD)         NCCL / Slurm)      CI debugging)
```

`hardware` dictates which accelerator backend environment MaxText initializes and optimizes for.

---

## 2. Supported Hardware Modes

| Option | Architecture / Environment | Details |
|---|---|---|
| `'tpu'` (default) | Google Cloud TPUs (v4, v5e, v5p, v6e) | Leverages TPU runtime, Libtpu, ICI 2D/3D torus topologies, and Pallas TPU kernels. |
| `'gpu'` | NVIDIA or AMD GPUs (single process/node) | Initializes JAX GPU backend for single-node development and verification. |
| `'gpu_multiprocess'` | Distributed multi-node GPU clusters | Coordinates multi-host GPU clusters via NCCL and distributed JAX rendezvous. |
| `'cpu'` | Local CPU | Forces execution on host CPU cores; used exclusively for lightweight unit testing. |

Default in `base.yml`:
```yaml
hardware: 'tpu'
```

---

## 3. Platform-Specific Behaviors

```text
When hardware == 'tpu':
  ├── num_slices is auto-detected from TPU worker metadata or compile_topology
  ├── Pallas Flash/Splash attention kernels run via Mosaic TPU backends
  └── ICI mesh axis dimensions map directly onto TPU physical torus dimensions

When hardware == 'gpu' or 'gpu_multiprocess':
  ├── num_slices is forced to 1
  ├── Attention falls back to CuDNN Flash Attention via TransformerEngine ('cudnn_flash_te') or DotProduct
  └── Mesh dimensions map onto NVLink / InfiniBand ranks
```

---

## 4. Interactions with Other Parameters

- **`attention`**: When `hardware: 'tpu'`, `attention: 'flash'` or `'autoselected'` uses TPU-native Pallas kernels. On GPU, `'cudnn_flash_te'` is utilized.
- **`num_slices`**: Automatically forced to `1` when `hardware != 'tpu'`.
- **`compile_topology`**: Used for ahead-of-time (AOT) TPU compilation (e.g. `'v5e-256'`).

---

## 5. Practical Guidelines

- **Running on TPUs**: Keep default `hardware: 'tpu'`.
- **Running on GPU Slurm/Kubernetes clusters**: Specify `hardware: 'gpu_multiprocess'` and ensure standard distributed environment variables (`MASTER_ADDR`, `MASTER_PORT`, `RANK`, `WORLD_SIZE`) are exposed to JAX.
- **Local Development / CI**: Set `hardware: 'cpu'` to run fast unit tests without requiring attached accelerators.

---

### One-line intuition

> **`hardware` selects the underlying accelerator platform (`'tpu'`, `'gpu'`, `'gpu_multiprocess'`, or `'cpu'`), tailoring device initialization, interconnect topology, and kernel backends accordingly.**
