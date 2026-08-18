## 1. Why does `vertex_tensorboard_project` exist?

When using Google Cloud's managed Vertex AI TensorBoard (`use_vertex_tensorboard: true`), MaxText must know which Google Cloud Project owns the target Vertex AI TensorBoard instance.

`vertex_tensorboard_project` specifies the GCP Project ID for Vertex AI TensorBoard integration.

---

## 2. What it actually controls

```yaml
vertex_tensorboard_project: ""
```

- When empty `""` (default): MaxText attempts to infer the GCP project from the environment (e.g. `gcloud config get-value project` or instance metadata).
- When set (e.g. `"my-ai-project-123"`): Explicitly sets the target GCP project ID for creating and uploading Vertex AI TensorBoard experiments.

---

## 3. Options and Defaults

| Value | Behavior |
|---|---|
| `""` (default) | Falls back to active gcloud CLI project or VM metadata |
| `"project-id-string"` | Explicit target GCP Project ID |

---

## 4. Interactions

- **`use_vertex_tensorboard`**: Must be `true` for this parameter to be utilized.
- **`vertex_tensorboard_region`**: Paired to locate the regional Vertex AI resource.

---

## 5. Practical Scenarios

- **Cross-Project Logging**: Set `vertex_tensorboard_project: "central-ml-metrics"` when compute VMs in one project log experiments into a centralized telemetry project.

---

### One-line intuition

> **`vertex_tensorboard_project` specifies the GCP Project ID hosting the Vertex AI TensorBoard experiment resource.**
