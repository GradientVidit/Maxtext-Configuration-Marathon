## 1. Why does `gdn_conv_kernel_dim` exist?

In linear attention and recurrent architectures like Gated DeltaNet (GDN), tokens are processed recurrently along the sequence. However, pure point-wise recurrence lacks local token mixing before state transitions:

```text
Without Conv:
Token x[t] ───> Projections (Q, K, V) ───> Recurrent Delta Update

With 1D Causal Conv (Kernel size = K):
Tokens x[t-K+1 : t] ───> 1D Depthwise Conv ───> Local Contextualized Q, K, V ───> Recurrent Delta Update
```

Without a short-range convolution, query and key vectors at time step $t$ depend solely on the single current token embedding $x_t$, making it difficult for the model to capture local morphological patterns, n-grams, or punctuation boundaries before updating the recurrent memory state.

`gdn_conv_kernel_dim` specifies the temporal kernel width of the 1D causal depthwise convolution applied to the query, key, and value streams prior to the delta-rule recurrent scan.

---

## 2. Mechanics & Data Flow

Before feeding projections into the chunked parallel scan or recurrent step, GDN applies a 1D causal convolution across the sequence dimension:

```text
Sequence Input X: [Batch, Seq_Len, Hidden_Dim]
          │
          ▼
Linear Projections (Q, K, V, Gates)
          │
          ▼
   1D Causal Conv (width = gdn_conv_kernel_dim, e.g., 4)
   (Padded causally by kernel_dim - 1 steps)
          │
          ▼
Activation Function (SiLU / RMSNorm)
          │
          ▼
Chunked Parallel Scan / Gated Delta Rule Kernel
```

- **Causal Padding**: To preserve autoregressive causality, input sequences are left-padded by `gdn_conv_kernel_dim - 1` with zeros (or cached convolution state during autoregressive generation).
- **Inference State**: During step-by-step decoding, each layer maintains a circular buffer of size `gdn_conv_kernel_dim - 1` tokens per sequence to compute the convolution incrementally with $O(1)$ complexity.

---

## 3. Options & Default

| Parameter | Type | Default | Options |
| :--- | :--- | :--- | :--- |
| `gdn_conv_kernel_dim` | `int` | `4` | Positive integers (typically `2`, `4`, or `8`) |

---

## 4. Interactions with Related Parameters

| Related Parameter | Interaction |
| :--- | :--- |
| `gdn_chunk_size` | Chunked parallel scan operates on the post-convolution feature representations. `gdn_conv_kernel_dim` must be $\le$ `gdn_chunk_size`. |
| `gdn_key_head_dim` / `gdn_value_head_dim` | Convolutions are applied channel-wise (depthwise) across the head feature dimensions. |
| `use_qk_norm_in_gdn` | Query/Key normalization is applied **after** the 1D convolution and before the delta update. |

---

## 5. Practical Guidance & Failure Modes

| Scenario | Symptom / Consequence | Fix |
| :--- | :--- | :--- |
| `gdn_conv_kernel_dim: 1` | Degenerates to point-wise linear projection without local temporal receptive field; hurts language modeling perplexity. | Set to standard `4`. |
| `gdn_conv_kernel_dim > gdn_chunk_size` | Cross-chunk convolution boundaries cause pipeline bubbles and irregular halo exchange overhead during chunked training. | Keep kernel size small ($4$) relative to chunk size ($64$). |
| Generation KV/State Cache | Conv state requires caching `kernel_dim - 1` activation vectors per layer in inference runtime. | Account for minimal inference state memory per stream. |

---

### One-line intuition

> `gdn_conv_kernel_dim` defines the window size of the causal 1D depthwise convolution in Gated DeltaNet, injecting local n-gram context into Q, K, and V before linear recurrent state updates.
