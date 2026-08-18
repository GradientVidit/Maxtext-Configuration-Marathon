## 1. Why does `gradient_clipping_threshold` exist?

During LLM pretraining, anomalous tokens, data corruption, or attention logit spikes can produce catastrophic gradient magnitudes.

If an unclipped gradient with norm $10^4$ enters the optimizer, it produces enormous weight steps that permanently destroy learned embeddings and representations, resulting in irreversible loss spikes:

```text
Global Gradient Norm vs Clipping Threshold:

||g|| ^
      │            * (Anomalous Spike: ||g|| = 45.0)
      │           / Threshold ───────/───\───────────── (gradient_clipping_threshold = 1.0)
      │         /           │  /\    /       \    /      ──/──\──/─────────\──/──\────> Steps

Unclipped Update: w = w - lr * 45.0   ──> Weights blow up / NaNs
Clipped Update:   w = w - lr * (1.0 * g / ||g||) ──> Preserves direction, caps magnitude
```

`gradient_clipping_threshold` specifies the maximum allowable global $\ell_2$ norm for model gradients before scaling them down proportionally.

---

## 2. Fundamentals & Mechanics

MaxText computes the global gradient norm across all model parameters:

$$\|g\|_2 = \sqrt{\sum_{p} \sum_i g_{p, i}^2}$$

If $\|g\|_2 > \text{gradient\_clipping\_threshold}$:

$$g_{\text{clipped}} = g \cdot \frac{\text{gradient\_clipping\_threshold}}{\|g\|_2}$$

- **Direction Preserved:** The relative proportions between parameter gradients remain identical; only the step magnitude is bounded.
- **Disabled via `0.0`:** Setting `gradient_clipping_threshold: 0` disables norm clipping entirely.

---

## 3. Options & Defaults

| Option | Value | Meaning |
|---|---|---|
| Default | `1.0` | Standard clipping threshold across modern LLMs (Llama, Gemma, GPT-3). |
| Disabled | `0.0` | Disables gradient clipping completely. |
| Strict | `0.5` | Aggressive clipping for unstable architectures (e.g. deep MoE, Muon). |

---

## 4. Interactions & Dependencies

```text
Gradients (bwd pass)
        │
        ▼
gradient_accumulation_steps (Summed / Averaged)
        │
        ▼
gradient_clipping_threshold (Clipped by global L2 norm)
        │
        ▼
Optimizer Step (AdamW / Muon update)
```

- **`gradient_accumulation_steps`:** Gradients are accumulated across micro-batches first, then clipped once prior to the optimizer update.

---

## 5. Practical Scenarios & Failure Modes

- **Persistent Clipping:** If the logged gradient norm is continuously above `1.0` for hundreds of steps, the base `learning_rate` is likely too high or weight initialization scale is mismatched.
- **Disabling Clipping:** Setting `0.0` on a 70B+ pretraining run drastically increases vulnerability to sudden loss divergence from edge-case web crawl tokens.

---

### One-line intuition

> **`gradient_clipping_threshold` caps the global $\ell_2$ norm of gradients, preventing outlier batches from triggering catastrophic parameter updates.**
