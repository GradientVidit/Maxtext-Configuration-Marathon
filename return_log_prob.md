## 1. Why it exists: exposing model confidence and token probabilities

During standard text generation, an inference server computes raw logits over the entire vocabulary $V$, selects the next token via sampling or $\text{argmax}$, and returns only the generated token IDs (or decoded strings) to the user:

```text
Standard Text Generation (return_log_prob: false):
Logits: [L_1, L_2, ..., L_|V|] ──> Sample Token ──> Output: "Paris"
(Probabilities are computed internally and discarded immediately)

Log-Probability Generation (return_log_prob: true):
Logits: [L_1, L_2, ..., L_|V|] ──> Log-Softmax ──> Sample Token ──> Output: "Paris" (log_prob: -0.042)
                                                                 Top-K alternatives:
                                                                 - "London": -3.81
                                                                 - "Berlin": -4.12
```

In many production and research workflows, receiving raw text alone is insufficient:
- **Confidence Scoring & Hallucination Detection**: Evaluating whether the model was certain of a generated fact or guessing randomly.
- **Perplexity & Language Modeling Evaluation**: Scoring the likelihood of external sequences under the model.
- **Reinforcement Learning from Human/AI Feedback (RLHF / PPO / DPO)**: Computing policy log-ratios $\log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}$.
- **Constrained Beam Search & Verifier Re-ranking**: Summing log-probabilities across sequences to find the maximum-likelihood hypothesis.

`return_log_prob` instructs MaxText's inference engine to compute, preserve, and serialize the log-probabilities $\log P(w_t | w_{<t})$ for generated tokens in the response payload.

---

## 2. Mechanics: log-softmax computation and payload serialization

When `return_log_prob: true`:

```text
 1. Compute Output Logits:
    logits = model.apply(..., step_tokens)   # Shape: [Batch, Vocab_Size]
                        │
                        ▼
 2. Compute Log-Softmax:
    log_probs = jax.nn.log_softmax(logits, axis=-1)
                        │
                        ▼
 3. Extract Chosen Token Log-Probabilities:
    selected_log_prob = jnp.take_along_axis(log_probs, sampled_token_id, axis=-1)
                        │
                        ▼
 4. (Optional) Extract Top-K Alternative Log-Probs:
    top_k_log_probs, top_k_indices = jax.lax.top_k(log_probs, k)
                        │
                        ▼
 5. Attach Probabilities to Output gRPC / REST Response Payload
```

Because computing `log_softmax` over a large vocabulary (e.g. 128k–256k tokens) across all decode steps requires additional memory bandwidth and tensor serialization, keeping this disabled by default saves network payload bandwidth and compute overhead when only plain text is needed.

---

## 3. Options & Configuration

Default in `base.yml`:

```yaml
return_log_prob: false
```

| Value | Behavior | Output Data | Performance Impact | Use Case |
|---|---|---|---|---|
| `false` (default) | Discards internal probability distributions after token sampling. | Token strings / IDs only. | Lowest latency and minimal serialization payload size. | Standard user-facing chatbots and latency-critical text streaming. |
| `true` | Computes `log_softmax` and returns per-token log-probabilities. | Token strings + float log-probability values ($\le 0.0$). | Minor compute overhead in final projection; larger network response payloads. | Model evaluation, RLHF training rollouts, uncertainty estimation, classification via generation. |

---

## 4. Parameter Interactions

```text
┌───────────────────────────────────────────────────────────┐
│                      return_log_prob                      │
└─────────────┬───────────────────────────────┬─────────────┘
              │ (when true)
              ▼
┌───────────────────────────────────────────────────────────┐
│ Interacts with:                                           │
│ - multi_sampling (scores multiple candidate trajectories) │
│ - float32_logits / logits_dot_in_fp32 (numerical precision)│
│ - cast_logits_to_fp32                                     │
└───────────────────────────────────────────────────────────┘
```

- **`float32_logits` / `cast_logits_to_fp32`**: For stable and accurate log-probability computation, logits should be evaluated or cast to FP32 before `log_softmax` to prevent numerical underflow in probability tails.
- **`multi_sampling`**: Often used together to calculate cumulative sequence scores $\sum_{t} \log P(w_t)$ across $N$ sampled completions.

---

## 5. Practical Scenarios & Failure Modes

### Verifying Code Generation Confidence
When filtering generated Python functions by confidence score:
```yaml
inference_server: "MaxtextInterleavedServer"
return_log_prob: true
multi_sampling: true
```
The client receives 5 generated functions, calculates the average log-probability per token for each, and discards solutions where the model exhibited high entropy/uncertainty.

### What breaks if misconfigured:
- **Network serialization bottleneck**: In high-throughput streaming pipelines, returning full top-$K$ log-probabilities for every single token can multiply network bandwidth requirements by $10\times$–$50\times$, increasing client-side deserialization overhead.

---

### One-line intuition

> **`return_log_prob` enables computing and returning per-token log-probabilities alongside generated text, providing the confidence metrics needed for RLHF rollouts, uncertainty estimation, and candidate re-ranking.**
