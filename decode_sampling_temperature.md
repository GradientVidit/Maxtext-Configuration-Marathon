## 1. Why does `decode_sampling_temperature` exist?

The softmax function converts raw logits $z_i$ into probabilities $P(x_i)$. Without temperature control, the sharpness of the probability distribution is fixed:

$$P(x_i) = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$

```text
Temperature T Scaling:
T → 0.0 (Cold):    Logits magnified ──> Distribution collapses to one-hot (Deterministic, conservative)
T = 1.0 (Normal):  Raw logits unscaled ──> Standard model confidence
T > 1.0 (Hot):     Logits flattened ──> Distribution approaches uniform random (Creative, chaotic)
```

`decode_sampling_temperature` ($T$) scales logit magnitudes before softmax exponentiation, controlling the entropy (randomness) of generated text.

---

## 2. What it actually controls

```yaml
decode_sampling_temperature: 1.
```

- Divides all output vocabulary logits by $T$ prior to applying softmax and categorical sampling in stochastic decoding modes (`weighted`, `topk`, `nucleus`, `composite`).

---

## 3. Options and Defaults

| Temperature ($T$) | Probability Distribution | Output Characteristics | Best Suited For |
|---|---|---|---|
| `1.0` (default) | Unscaled native logits | Standard calibrated distribution | General evaluation |
| `0.2` – `0.3` | Extremely peaked / sharp | Factual, repetitive, low randomness | Coding, math reasoning, structured extraction |
| `0.7` – `0.8` | Balanced diversity | Fluent, coherent, varied vocabulary | Conversational chat, summarization |
| `1.2` – `1.5` | Flattened / high entropy | Highly unexpected, divergent | Creative writing, brainstorming |

---

## 4. Interactions and Inactive Modes

- **Inactive in `"greedy"`**: When `decode_sampling_strategy: "greedy"`, temperature has no effect because $\aargmax(z_i / T) = \aargmax(z_i)$ for any $T > 0$.
- **Combined with `topk` / `nucleus`**: Applying temperature before Top-$P$ filtering changes how many tokens fall inside the nucleus threshold $p$.

---

## 5. Practical Scenarios

- **Code & Mathematics**: Use `decode_sampling_temperature: 0.2`.
- **Dialogue Agents**: Use `decode_sampling_temperature: 0.7`.

---

### One-line intuition

> **`decode_sampling_temperature` modulates the sharpness of the softmax probability distribution, scaling generation between rigid determinism and creative diversity.**
