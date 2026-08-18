## 1. Why does `use_qk_clip` exist?

During large-scale language model pretraining—especially when utilizing the **Muon optimizer** (or related momentum orthogonalization algorithms, as in **Kimi K2**)—Query ($Q$) and Key ($K$) activation vectors can experience transient norm explosions.

When the Euclidean norms $\|Q\|_2$ and $\|K\|_2$ surge, their dot products ($QK^T / \sqrt{d}$) produce massive attention logits, causing catastrophic loss spikes:

```text
Without QK-Clip (Muon Loss Spike Hazard):
  Step 50,000: Outlier gradient ──> ||Q|| surges to 850 ──> Logits reach 300+ ──> Attention collapses ──> LOSS SPIKE!

With QK-Clip (use_qk_clip=True, threshold τ = 100.0):
  Step 50,000: Outlier gradient ──> ||Q|| capped at 100.0 ──> Logits stay bounded ──> Zero loss spikes!
```

**QK-Clip (MuonClip)** clips the Euclidean norm of $Q$ and $K$ vectors to a maximum sphere radius $	au = \text{qk\_clip\_threshold}$:

$$Q_{\text{clipped}} = Q \cdot \min\left(1, rac{	au}{\|Q\|_2 + \epsilon}
ight), \quad K_{\text{clipped}} = K \cdot \min\left(1, rac{	au}{\|K\|_2 + \epsilon}
ight)$$

---

## 2. Options & Defaults

| Value | Behavior | Notes |
|---|---|---|
| `false` | QK-Clip disabled. Raw unclipped Query and Key vectors are used. | **Default**. |
| `true` | Enables QK-Clip norm bounding to threshold $	au$. | Supported with MLA on `dot_product` or Tokamax Splash attention. |

Default in `base.yml`: `false`

> **Implementation note:** MaxText clips Q and K *activations* during the forward pass. The original MuonClip paper rescales W_Q and W_K weight matrices *after* the optimizer step (a post-step weight norm constraint). Both approaches bound the maximum attention logit magnitude; the weight-rescaling approach never modifies gradients, while activation clipping has zero gradient when ‖Q‖ > τ (the `min(1, ...)` saturates).

---

## 3. Supported Architecture & Kernel Combinations

In MaxText, `use_qk_clip: true` is specifically integrated with:
- **Multi-Head Latent Attention (`attention_type: 'mla'`):** Applied to the reconstructed Query and Key tensors.
- **Backends:** Pure JAX `attention: 'dot_product'` or `tokamax` Splash attention kernels.

---

## 4. QK-Clip vs. QK-Norm vs. Soft-Capping

```text
Technique                 Operation                                         When it Acts
─────────────────────────────────────────────────────────────────────────────────────────────
use_qk_clip               Clips norm only when $\|z\|_2 > 	au$ (MuonClip)   Outlier prevention
qk_norm_with_scale        Strictly normalizes all vectors to unit norm      Every forward pass
attn_logits_soft_cap      Applies $c 	anh(z/c)$ to attention logits        After dot product
```

Unlike continuous normalization, QK-Clip is **completely inactive during normal steps** (when $\|Q\|_2 < 	au$) and only engages as a circuit breaker during pathological norm spikes.

---

## 5. Practical Scenarios

- **Pretraining with Muon / MuonClip Optimizer (e.g. Kimi K2):** Set `use_qk_clip: true` with `qk_clip_threshold: 100.0` to eliminate loss spikes during trillion-token pretraining runs.
- **Standard AdamW Pretraining:** Leave `use_qk_clip: false`.

---

### One-line intuition

> **`use_qk_clip=true` acts as a numerical circuit breaker that clips $Q$ and $K$ vector norms to $	au=100.0$, eliminating loss spikes during large-scale pretraining with the Muon optimizer.**
