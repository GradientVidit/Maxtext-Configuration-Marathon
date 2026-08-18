## 1. Why does `use_vertex_tensorboard` exist?

Self-hosting open-source TensorBoard on local disks or reading raw event files from GCS buckets can be slow, requires running dedicated TensorBoard VM servers, and lacks enterprise collaboration features (access control, persistent sharing, unified experiment dashboards).

**Vertex AI TensorBoard** is Google Cloud's enterprise managed service that ingests TensorBoard streams directly via Google APIs without needing a separate TensorBoard server:

```text
Standard Open-Source TensorBoard:
MaxText ──> Write .tfevents to GCS ──> User runs `tensorboard --logdir=gs://...` on VM

Managed Vertex AI TensorBoard (use_vertex_tensorboard: true):
MaxText ──(Vertex AI TensorBoard Uploader)──> Managed GCP Vertex AI Experiment Console
```

`use_vertex_tensorboard` switches MaxText from writing raw GCS event files to streaming directly to Vertex AI TensorBoard.

---

## 2. What it actually controls

```yaml
use_vertex_tensorboard: false
```

- When `false` (default): MaxText writes `.tfevents` files to `<base_output_directory>/tensorboard/`. (Recommended when running via XPK).
- When `true`: MaxText configures the Vertex AI TensorBoard uploader using `vertex_tensorboard_project` and `vertex_tensorboard_region`. (Recommended for standalone GCE VM runs).

---

## 3. Options and Environments

| Environment | `use_vertex_tensorboard` | Reason |
|---|---|---|
| GCE Standalone VMs | `true` | Easy managed dashboard without running custom server |
| XPK / GKE Clusters | `false` (default) | XPK manages GCS logs natively |

---

## 4. Interactions and Requirements

- **`vertex_tensorboard_project`**: GCP Project hosting the Vertex TensorBoard instance.
- **`vertex_tensorboard_region`**: GCP Region where the Vertex TensorBoard resource exists.
- **`enable_tensorboard`**: Must be `true`.

---

## 5. Practical Scenarios

- **Running on GCE with Vertex AI**:
```yaml
enable_tensorboard: true
use_vertex_tensorboard: true
vertex_tensorboard_project: "my-gcp-project"
vertex_tensorboard_region: "us-central1"
```

---

### One-line intuition

> **`use_vertex_tensorboard` streams experiment metrics to Google Cloud's managed Vertex AI TensorBoard service instead of writing raw event files to GCS.**
