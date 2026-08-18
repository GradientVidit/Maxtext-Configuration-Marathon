## 1. Why does `decode_sampling_nucleus_p` exist?

Fixed Top-$K$ filtering retains a constant number of tokens regardless of how confident the model is. When the distribution is highly peaked, Top-$K$ includes irrelevant low-probability tail tokens. When the distribution is flat, Top-$K$ prematurely cuts off valid alternatives.

**Nucleus (Top-$P$) Sampling** (Holtzman et al., 2019) dynamically sizes the candidate token pool by selecting the smallest subset $V^{(p)} \subset V$ whose cumulative probability exceeds threshold $p$:

$$\sum_{i \in V^{(p)}} P(x_i) \ge p$$

```text
Flat Distribution (Uncertain Context):    Peaked Distribution (Certain Context):
Tokens: [ A: 0.3, B: 0.3, C: 0.2, D: 0.1 ]  Tokens: [ A: 0.92, B: 0.04, C: 0.02 ]
p = 0.9 ──> Selects {A, B, C, D}           p = 0.9 ──> Selects ONLY {A} (Dynamic pruning!)
```

`decode_sampling_nucleus_p` defines the cumulative probability threshold $p$.

---

## 2. What it actually controls

```yaml
decode_sampling_nucleus_p: -1
```

- When `-1` (default): Unset / disabled.
- When set in $(0, 1.0]$ (e.g. `0.9`, `0.95`): MaxText sorts vocabulary logits in descending order, calculates the cumulative softmax sum, masks out all tokens after the cumulative mass crosses $p$, and renormalizes the remaining probabilities for sampling.

---

## 3. Options and Defaults

| Value | Meaning | Candidate Pool Dynamism |
|---|---|---|
| `-1` (default) | Disabled / unset | None |
| `0.90` | Top 90% probability mass retained | High quality, removes low-probability hallucinations |
| `0.95` | Top 95% probability mass retained | Standard default for conversational LLMs |
| `1.0` | 100% mass retained | Equivalent to unconstrained temperature sampling |

---

## 4. Interactions and Strategy Enablers

- **Active in `"nucleus"` and `"composite"`**: `decode_sampling_nucleus_p` must be set $> 0$ when `decode_sampling_strategy` is `"nucleus"` or `"composite"`.
- **`decode_sampling_temperature`**: Temperature scaling is applied before computing cumulative probabilities.

---

## 5. Practical Scenarios

- **Balanced Natural Text Generation**: Set `decode_sampling_strategy: "nucleus"`, `decode_sampling_nucleus_p: 0.9`, and `decode_sampling_temperature: 0.8`.

---

### One-line intuition

> **`decode_sampling_nucleus_p` sets the cumulative probability cutoff $p$ for nucleus sampling, dynamically pruning the probability tail.**
