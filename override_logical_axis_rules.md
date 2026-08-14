
## 1. First: what are "logical axis rules"?

MaxText separates:

```text
MODEL TENSOR DIMENSIONS
        ↓
logical axis names
        ↓
physical mesh axes
        ↓
TP / FSDP / DP / etc.
```

For example, a tensor dimension may have a logical name:

```text
activation_heads
```

and MaxText's default rule says:

```yaml
['activation_heads', ['tensor', 'tensor_sequence', 'autoregressive']]
```

Meaning: this logical dimension can be mapped/sharded across those physical mesh axes. ([GitHub](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/maxtext/configs/base.yml?utm_source=chatgpt.com "maxtext/src/maxtext/configs/base.yml at main · AI-Hypercomputer/maxtext · GitHub"))

So **logical axis rules are the bridge between the model's logical dimensions and the TPU device mesh.**

---

## 2. Why does `override_logical_axis_rules` exist?

MaxText already has a large default set of rules:

```yaml
logical_axis_rules:
  - ...
  - ['activation_heads', ['tensor', 'tensor_sequence', 'autoregressive']]
  - ['heads', ['tensor', 'tensor_sequence', 'autoregressive']]
  - ['mlp', ['fsdp_transpose', 'tensor', 'tensor_sequence', 'autoregressive']]
  - ...
```

These are carefully designed default sharding rules. ([GitHub](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/maxtext/configs/base.yml?utm_source=chatgpt.com "maxtext/src/maxtext/configs/base.yml at main · AI-Hypercomputer/maxtext · GitHub"))

Sometimes you want to provide your **own rules**.

The question is:

> Should my supplied rules **replace** MaxText's defaults, or should they be **combined with** them?

That's exactly what this flag controls.

---

## 3. `false` — merge

Default:

```yaml
override_logical_axis_rules: false
```

Your custom rules are **merged with** the existing rules.

Conceptually:

```text
MaxText default rules
        +
your custom rules
        ↓
combined logical-axis rules
```

This is useful when you only want to modify/add a small portion of the sharding behavior while retaining the rest.

---

## 4. `true` — replace

```yaml
override_logical_axis_rules: true
```

Now your supplied `logical_axis_rules` **replace the default rules rather than being merged with them**.

Conceptually:

```text
MaxText default rules     X
                         ↓
                  discarded

your logical_axis_rules
         ↓
      used instead
```

This is therefore a much more aggressive setting.

If you provide an incomplete rule set, you can accidentally leave tensors without the expected logical-axis mapping.

---

## 5. Why would you ever use `true`?

When you want **complete control over the sharding policy**.

For example, you're experimenting with a custom mesh:

```yaml
mesh_axes: ['data', 'fsdp', 'tensor']
```

and want a completely different mapping between logical tensor dimensions and those mesh axes.

Then:

```yaml
override_logical_axis_rules: true
logical_axis_rules: [...]
```

means:

> **"Don't combine my sharding policy with MaxText's assumptions. Use my policy."**

This is especially relevant for custom parallelism strategies / custom model architectures.

---

## 6. The distinction from `custom_mesh_and_rule`

Current MaxText also has:

```yaml
custom_mesh_and_rule: ""
```

which can replace the default mesh **and logical rules** by selecting a predefined YAML under `config/mesh_and_rule/`. ([GitHub](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/maxtext/configs/base.yml?utm_source=chatgpt.com "maxtext/src/maxtext/configs/base.yml at main · AI-Hypercomputer/maxtext · GitHub"))

So there are two concepts:

```text
override_logical_axis_rules
    ↓
"I am supplying/modifying logical axis rules;
 should they replace or merge with defaults?"


custom_mesh_and_rule
    ↓
"I want to use a predefined custom mesh + rule configuration."
```

Don't confuse them.

---

## 7. Options

Only two:

|Value|Behavior|
|---|---|
|`false`|**Merge** supplied logical-axis rules with defaults|
|`true`|**Replace** defaults with supplied logical-axis rules|

Default:

```yaml
override_logical_axis_rules: false
```

---

## One-line intuition

> **`override_logical_axis_rules` controls whether your custom sharding rules are added to MaxText's default rules (`false`) or completely replace them (`true`).** ([GitHub](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/maxtext/configs/base.yml?utm_source=chatgpt.com "maxtext/src/maxtext/configs/base.yml at main · AI-Hypercomputer/maxtext · GitHub"))

The important conceptual chain is:

```text
logical tensor dimensions
        ↓
logical axis rules
        ↓
physical mesh axes
        ↓
actual TPU sharding
```

So this parameter is ultimately a **sharding-policy override switch**.