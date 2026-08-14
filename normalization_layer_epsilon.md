
## 1. Why does it exist?

Normalization divides by a standard-deviation/RMS-like quantity. That quantity can become very small, so we add a tiny constant ε to prevent division by zero and improve numerical stability.

For RMSNorm, conceptually:

# [  
\operatorname{RMSNorm}(x)

\frac{x}{\sqrt{\operatorname{mean}(x^2)+\epsilon}}\odot\gamma  
]

So MaxText's parameter controls that:

```text
normalization_layer_epsilon
          ↓
      ε in norm
          ↓
prevents unstable division by ~0
```

---

## 2. What does it control?

It controls the epsilon used by **both RMSNorm and LayerNorm** in MaxText. ([GitHub](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/maxtext/configs/base.yml?utm_source=chatgpt.com "maxtext/src/maxtext/configs/base.yml at main · AI-Hypercomputer/maxtext · GitHub"))

Default:

```yaml
normalization_layer_epsilon: 1.e-05
```

So:

[  
\epsilon = 10^{-5}  
]

---

## 3. Why does the value matter?

It's a **numerical-stability vs. normalization-perturbation** parameter.

- **Too small:** less protection against numerical problems.
    
- **Larger:** more numerical stabilization, but ε contributes more noticeably to the denominator and therefore changes the normalization.
    

For normal values of the variance/RMS, ε is tiny and has essentially no effect. Its purpose is mainly when that quantity becomes very small.

---

## 4. Important: model-specific norms can have different epsilons

This is the subtle part.

Hugging Face model configurations, for example, can specify their own `rms_norm_eps`; current MaxText's conversion code contains models with values such as `1e-06`. ([GitHub](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/maxtext/checkpoint_conversion/utils/hf_model_configs.py?utm_source=chatgpt.com "maxtext/src/maxtext/checkpoint_conversion/utils/hf_model_configs.py at main · AI-Hypercomputer/maxtext · GitHub"))

Therefore, don't assume:

```text
MaxText default = 1e-5
```

means:

> every supported pretrained model mathematically uses 1e-5.

Model-specific configuration/conversion can matter.

If you're reproducing a pretrained model, **the normalization epsilon is part of the architecture/configuration and should match the model you're reproducing**.

---

## 5. What options does it have?

It's a floating-point value, not an enum.

Examples:

```yaml
normalization_layer_epsilon: 1.e-05
```

or:

```yaml
normalization_layer_epsilon: 1.e-06
```

There isn't a fixed list of allowed values.

---

## 6. What should you change it to?

Usually: **don't change it casually.**

If you're running a known architecture, use the epsilon specified by that model's configuration.

If you're designing/testing your own model, then this becomes a numerical-stability hyperparameter.

---

### One-line intuition

> **`normalization_layer_epsilon` is the small ε added inside RMSNorm/LayerNorm's denominator to prevent numerical instability; its value is part of the model's normalization definition.** ([GitHub](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/maxtext/configs/base.yml?utm_source=chatgpt.com "maxtext/src/maxtext/configs/base.yml at main · AI-Hypercomputer/maxtext · GitHub"))