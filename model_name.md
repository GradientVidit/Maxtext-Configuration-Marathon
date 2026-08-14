
## 1. Why does `model_name` exist?

MaxText has **one generic `base.yml`**, but different LLMs need different architecture/configuration values.

For example:

```text
Qwen3
  → hidden size
  → number of layers
  → Q/K/V heads
  → RoPE settings
  → MLP structure
  → etc.

Llama
  → different values

Gemma
  → different values

Kimi
  → different values
```

Rather than forcing you to manually specify all of those every time, MaxText uses:

```yaml
model_name: qwen3-0.6b
```

to select the appropriate **model-specific configuration overrides**.

So the fundamental purpose is:

> **`model_name` tells MaxText which model's configuration should be applied on top of the generic base configuration.**

---

## 2. What actually happens?

Conceptually:

```text
base.yml
   │
   │ generic defaults
   ↓
model_name = qwen3-0.6b
   │
   ↓
Qwen3-specific config overrides
   │
   ↓
final Config
   │
   ├── hidden dimensions
   ├── heads
   ├── layers
   ├── attention settings
   ├── RoPE settings
   └── ...
```

So `model_name` is primarily a **configuration-selection mechanism**, not a model-loading mechanism.

---

## 3. What are the options?

Unlike `run_name`, `model_name` has a **finite set of model identifiers implemented by MaxText**.

The exact available names evolve as MaxText adds models.

For example, current MaxText examples use:

```bash
model_name=kimi-k2-1t
```

for Kimi K2. ([GitHub](https://github.com/AI-Hypercomputer/maxtext/blob/main/tests/end_to_end/tpu/kimi/Run_Kimi.md?utm_source=chatgpt.com "maxtext/tests/end_to_end/tpu/kimi/Run_Kimi.md at main · AI-Hypercomputer/maxtext · GitHub"))

Other model families have their corresponding identifiers.

The important point is:

> **You cannot invent `model_name=my-awesome-model` and expect MaxText to know its architecture.**

The name has to correspond to a model configuration that MaxText knows about.

---

## 4. `model_name` does NOT mean "download this model"

This is a common misconception.

If you write:

```yaml
model_name: qwen3-0.6b
```

MaxText is **not necessarily downloading Qwen3 from Hugging Face**.

It is saying:

```text
"Configure the MaxText model implementation
according to Qwen3-0.6B's architecture."
```

Where the actual parameters come from is a separate concern:

```yaml
load_parameters_path: gs://...
```

or initialization from scratch.

For example, MaxText's Kimi example uses:

```text
model_name=kimi-k2-1t
load_parameters_path=...
```

The two serve different purposes. ([GitHub](https://github.com/AI-Hypercomputer/maxtext/blob/main/tests/end_to_end/tpu/kimi/Run_Kimi.md?utm_source=chatgpt.com "maxtext/tests/end_to_end/tpu/kimi/Run_Kimi.md at main · AI-Hypercomputer/maxtext · GitHub"))

---

## 5. Why does `base.yml` itself contain architecture parameters?

This is the subtle part.

You saw things like:

```yaml
base_emb_dim: 2048
base_num_query_heads: 16
base_num_kv_heads: 16
base_mlp_dim: 7168
base_num_decoder_layers: 16
head_dim: 128
```

in `base.yml`. ([GitHub](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/maxtext/configs/base.yml?utm_source=chatgpt.com "maxtext/src/maxtext/configs/base.yml at main · AI-Hypercomputer/maxtext · GitHub"))

So why need `model_name`?

Because those are **defaults/base values**. A model-specific configuration can override them.

Conceptually:

```text
base.yml

base_emb_dim = 2048
layers       = 16
heads        = 16
       │
       │ model_name = some_model
       ↓
model-specific overrides
       │
       ↓
final architecture
```

This lets MaxText maintain one common configuration system across many architectures.

---

## 6. What does `model_name: "default"` mean?

`default` essentially means:

> **Don't apply a model-specific architecture override; use the base configuration.**

That's why the base values themselves describe a plausible generic decoder model.

It's useful for testing/development and as the generic starting point.

---

## 7. Important: `model_name` vs `override_model_config`

These two are directly related:

```yaml
model_name: "default"
override_model_config: false
```

`model_name` selects the model configuration.

`override_model_config` controls whether you are allowed to **override model-specific parameters from the CLI/kwargs/environment**, primarily for debugging/testing. ([GitHub](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/maxtext/configs/base.yml?utm_source=chatgpt.com "maxtext/src/maxtext/configs/base.yml at main · AI-Hypercomputer/maxtext · GitHub"))

So normally:

```text
model_name
    ↓
select known model
    ↓
apply its prescribed architecture config
```

While with the override mechanism you can deliberately say, essentially:

```text
"I know this isn't the normal Qwen3 configuration;
let me change some architecture parameters anyway."
```

That is why MaxText labels it as a debugging/testing facility.

---

## 8. The most important mental model

Don't think:

> `model_name = which checkpoint?`

Think:

> **`model_name = which model architecture/configuration should MaxText instantiate?`**

Then separate the three concepts:

```text
model_name
    ↓
WHAT architecture/config?

load_parameters_path
    ↓
WHERE do the weights come from?

run_name
    ↓
WHICH experiment/run is this?
```

---

## One-line intuition

> **`model_name` selects a known model's architecture-specific configuration, which overrides the generic `base.yml` defaults; it does not itself specify where the model weights come from.** ([GitHub](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/maxtext/configs/base.yml?utm_source=chatgpt.com "maxtext/src/maxtext/configs/base.yml at main · AI-Hypercomputer/maxtext · GitHub"))

### Side note:
**`override_model_config` matters primarily when you're overriding a _specific model's prescribed configuration_.** With `model_name=default`, you're already operating on the base configuration. You can set base_emb_dim etc. directly.


