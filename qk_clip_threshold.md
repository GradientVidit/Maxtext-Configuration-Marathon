## 1. Why does `qk_clip_threshold` exist?

When QK-Clip is enabled (`use_qk_clip: true`), the model bounds the Euclidean norm ($\| \cdot \|_2$) of Query and Key vectors.

`qk_clip_threshold` sets the maximum allowable vector norm ($	au$):

$$\text{Scaling Factor } lpha = \min\left(1.0, rac{	au}{\|v\|_2 + \epsilon}
ight)$$

$$v_{\text{clipped}} = lpha \cdot v$$

```text
Query Vector Norm ||Q||
  ▲
150│                      / Outlier Spike (Unclipped)
120│                     /
100│────────────────────/────── qk_clip_threshold (τ = 100.0)
   │                   /
 80│                  /
 60│  Normal Steps   /
 40│  (No Clipping) /
 20│               /
  0└──────────────/────────────► Step Progress
```

If $\|v\|_2 \le 	au$, $lpha = 1.0$ (the vector is untouched). If $\|v\|_2 > 	au$, the vector is scaled down to have a norm of exactly $	au$.

---

## 2. Options & Defaults

| Value | Behavior | Notes |
|---|---|---|
| `100.0` | Clips Query and Key norms to $	au = 100.0$. | **Default**. Standard value from MuonClip and Kimi K2. |
| Any float $> 0.0$ (e.g. `50.0`, `150.0`) | Custom clipping threshold radius. | Lower = tighter bound; Higher = looser bound. |

Default in `base.yml`: `100.0`

---

## 3. Selecting the Optimal Threshold $	au$

1. **Threshold too low ($	au \le 10.0$):** Forces continuous clipping on regular training steps, dampening healthy gradient dynamics and slowing convergence.
2. **Threshold too high ($	au \ge 500.0$):** Fails to engage before logits reach extreme ranges ($QK^T / \sqrt{d} > 100$), allowing numerical underflow in softmax.
3. **Empirical Sweet Spot ($	au = 100.0$):** Sits safely above normal operating norms ($\|Q\| pprox 20\text{--}60$) while cleanly catching runaway spikes ($\|Q\| > 100$).

---

## 4. Interactions with other parameters

| Related Parameter | Interaction |
|---|---|
| [[use_qk_clip]] | Parent switch: `qk_clip_threshold` is active only when `use_qk_clip: true`. |
| [[attention_type]] | Active when using MLA architectures. |

---

## 5. Practical Scenarios

- **Kimi K2 Pretraining Replication:** Use `use_qk_clip: true` and `qk_clip_threshold: 100.0`.
- **Diagnosing Training Spikes:** If a loss spike occurs at step $N$, check activation logs; if $Q/K$ norms exceed $100$, introducing QK-Clip with $	au=100.0$ will stabilize subsequent runs.

---

### One-line intuition

> **`qk_clip_threshold` specifies the maximum vector norm radius ($	au=100.0$) for QK-Clip, catching runaway attention representation spikes without altering normal gradient flow.**
