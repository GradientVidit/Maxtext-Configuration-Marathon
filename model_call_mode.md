
## 1. Why does it exist?

Some parts of the MaxText codebase behave differently depending on whether the model is being **trained** or used for **inference**. These differences aren't about correctness — they're about which code paths are appropriate:

- Training: dropout is active, KV cache is not used, gradient computation is needed
- Inference: no dropout, KV cache is used, no gradients needed

`model_call_mode` lets MaxText know which context it's operating in so it can activate the appropriate code paths, especially within quantization and precision logic.

---

## 2. What it controls

From the base.yml comment:
```yaml
model_call_mode: ""  # when left as is, corresponds to training
# accepted values are "inference"
```

When set to `"inference"`:
- Certain quantization code paths that are inference-specific are activated
- The model knows not to expect gradient-related operations
- Precision/dtype handling may differ at specific points (e.g., logit casting)

When set to `""` (default):
- Training mode: gradients, dropout, no KV cache persistence

---

## 3. Options

| Value | Behavior |
|---|---|
| `""` | Training mode (default) |
| `"inference"` | Inference/serving mode |

Default in base.yml:
```yaml
model_call_mode: ""
```

---

## 4. Relationship to quantization code paths

`model_call_mode` interacts with the quantization system when using inference-specific quantization schemes like `"intmp"`. Some quantization operations only make sense in inference (e.g., static scaling, fused dequantization), and `model_call_mode="inference"` tells the quantization logic to activate those paths.

---

## 5. When to set it

**Set `"inference"` when:**
- Running decode/inference benchmarks through MaxText
- Exporting a model for inference evaluation
- Using `intmp` or other inference-oriented quantization schemes

**Leave `""` when:**
- Training or fine-tuning
- Running evaluation during training (train-eval loop where you want dropout-off semantics but still within the training framework)

---

## 6. Common confusion: training eval vs inference

During training, MaxText can run evaluation (e.g., perplexity on a held-out set). That evaluation still runs in "training mode" from MaxText's perspective — no KV cache, etc. If you want true autoregressive inference semantics (with KV cache and inference-mode quantization), set `model_call_mode: "inference"` in a dedicated inference run.

---

### One-line intuition

> **`model_call_mode` declares the execution context to MaxText's internal code paths — `""` means training (with gradient/dropout semantics), `"inference"` activates inference-specific quantization and computation paths.**
