
## 1. Why does it exist?

You often want to start a new MaxText run **from an already-trained model**, rather than initialize the model randomly.

For example:

```text
Hugging Face model
       ↓
convert to MaxText checkpoint
       ↓
load_parameters_path
       ↓
MaxText model
       ↓
fine-tuning / inference
```

Or:

```text
pretrained checkpoint
       ↓
load_parameters_path
       ↓
new training run
```

So its fundamental meaning is:

> **"Initialize the model's parameters from this checkpoint."**

MaxText's documentation uses exactly this pattern for fine-tuning and RL/post-training. ([GitHub](https://github.com/AI-Hypercomputer/maxtext/blob/main/tests/end_to_end/tpu/kimi/Run_Kimi.md?utm_source=chatgpt.com "maxtext/tests/end_to_end/tpu/kimi/Run_Kimi.md at main · AI-Hypercomputer/maxtext · GitHub"))

---

## 2. What does "parameters only" mean?

This is the **most important part**.

A training checkpoint can contain much more than model weights:

```text
Full training state
├── model parameters
├── optimizer state
├── training step
├── other state
└── ...
```

`load_parameters_path` loads:

```text
model parameters
      ✓
optimizer state
      ✗
training step
      ✗
```

That's why MaxText has a separate:

```yaml
load_full_state_path: ""
```

The latter loads the **full checkpoint including optimizer state and step count**. ([GitHub](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/maxtext/configs/base.yml?utm_source=chatgpt.com "maxtext/src/maxtext/configs/base.yml at main · AI-Hypercomputer/maxtext · GitHub"))

### Mental distinction

```text
load_parameters_path
    → "Give me these weights and start my run."

load_full_state_path
    → "Continue this training job from this exact state."
```

---

## 3. What does the path point to?

Typically an **Orbax checkpoint directory**, for example:

```text
gs://my-bucket/my-model/checkpoints/1000/items
```

MaxText's current config gives both forms:

```text
.../checkpoints/items/NUMBER
```

or

```text
.../checkpoints/NUMBER/items
```

depending on checkpoint layout/version. ([GitHub](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/maxtext/configs/base.yml?utm_source=chatgpt.com "maxtext/src/maxtext/configs/base.yml at main · AI-Hypercomputer/maxtext · GitHub"))

The current documentation gives an example:

```text
gs://my-bucket/my-previous-run/checkpoints/items/1000
```

([MaxText](https://maxtext.readthedocs.io/en/latest/guides/checkpointing_solutions/gcs_checkpointing.html?utm_source=chatgpt.com "GCS bucket-based checkpointing — MaxText documentation"))

It can point to local storage too, not only GCS.

---

## 4. What happens at startup?

There's an important priority order in MaxText:

```text
1. Existing checkpoint for current run_name?
       ↓ yes
   resume it

2. Otherwise:
   load_parameters_path?
       ↓ yes
   load parameters

3. Otherwise:
   load_full_state_path?
       ↓ yes
   load full state

4. Otherwise:
   initialize from scratch
```

More precisely, MaxText says `load_parameters_path` and `load_full_state_path` are **mutually exclusive**, and the existing `run_name` checkpoint has higher priority. ([GitHub](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/maxtext/configs/base.yml?utm_source=chatgpt.com "maxtext/src/maxtext/configs/base.yml at main · AI-Hypercomputer/maxtext · GitHub"))

This is a subtle but important interaction with the parameter we just studied.

---

## 5. Example: fine-tuning

Suppose you have:

```text
gs://models/qwen3-0.6b-maxtext/...
```

and want a new fine-tuning run:

```yaml
model_name: qwen3-0.6b
run_name: qwen3_finetune_v1

load_parameters_path: gs://models/qwen3-0.6b-maxtext/checkpoints/0/items
```

Conceptually:

```text
                    pretrained weights
                           ↓
                 load_parameters_path
                           ↓
                    initialize model
                           ↓
                    new training run
                           ↓
                 qwen3_finetune_v1
                           ↓
               optimizer starts fresh
```

This is why the parameter is commonly used for fine-tuning. Current MaxText examples use it exactly this way. ([GitHub](https://github.com/AI-Hypercomputer/maxtext/blob/main/tests/end_to_end/tpu/kimi/Run_Kimi.md?utm_source=chatgpt.com "maxtext/tests/end_to_end/tpu/kimi/Run_Kimi.md at main · AI-Hypercomputer/maxtext · GitHub"))

---

## 6. What if I want to resume training?

Then `load_parameters_path` is usually **not** what you want.

Suppose training stopped at:

```text
step = 50,000
optimizer state = Adam moments at step 50,000
```

Using only:

```yaml
load_parameters_path: ...
```

would give you the **weights**, but not the optimizer state/step.

For genuine continuation, use:

```yaml
load_full_state_path: ...
```

or let MaxText automatically resume the checkpoint associated with the same `run_name`. ([GitHub](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/maxtext/configs/base.yml?utm_source=chatgpt.com "maxtext/src/maxtext/configs/base.yml at main · AI-Hypercomputer/maxtext · GitHub"))

---

## 7. What are the options?

It's simply a **string path**:

```yaml
load_parameters_path: ""
```

Empty:

```text
→ don't explicitly load parameters
```

Local:

```yaml
load_parameters_path: /path/to/checkpoint
```

GCS:

```yaml
load_parameters_path: gs://bucket/checkpoint/items
```

The checkpoint must be in a format MaxText can read—normally a **MaxText/Orbax checkpoint**, often after converting a Hugging Face checkpoint. MaxText's conversion documentation explicitly describes `load_parameters_path` as the path to the MaxText Orbax checkpoint. ([MaxText](https://maxtext.readthedocs.io/en/latest/guides/checkpointing_solutions/convert_checkpoint.html?utm_source=chatgpt.com "Checkpoint Conversion Utilities — MaxText documentation"))

---

## 8. The important interaction with `model_name`

These two answer different questions:

```yaml
model_name: qwen3-0.6b
load_parameters_path: gs://.../checkpoint
```

means:

```text
model_name
    → How should MaxText construct the model?

load_parameters_path
    → Which parameter values should be put into that model?
```

Therefore, the checkpoint's parameter shapes/configuration need to be compatible with the selected model configuration.

---

### One-line intuition

> **`load_parameters_path` loads only the model parameters from an existing checkpoint, allowing you to initialize a new MaxText run from pretrained/already-trained weights without restoring the optimizer/training state.** ([GitHub](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/maxtext/configs/base.yml?utm_source=chatgpt.com "maxtext/src/maxtext/configs/base.yml at main · AI-Hypercomputer/maxtext · GitHub"))

And remember:

```text
load_parameters_path
    = weights

load_full_state_path
    = weights + optimizer + step + full training state

run_name
    = automatic checkpoint/resume identity
```