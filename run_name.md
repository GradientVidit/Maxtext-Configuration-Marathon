
## 1. Why does `run_name` exist?

MaxText needs an **identity for a particular execution/run**.

Think:

```text
                    MaxText run
                        │
                  run_name = X
                        │
          ┌─────────────┼─────────────┐
          ↓             ↓             ↓
     checkpoints     metrics      diagnostics
```

The name becomes part of the namespace under `base_output_directory`.

For example:

```yaml
base_output_directory: gs://my-bucket/maxtext
run_name: qwen3_0.6b_exp1
```

gives MaxText a run-specific location conceptually like:

```text
gs://my-bucket/maxtext/qwen3_0.6b_exp1/
```

MaxText explicitly uses this pattern for GCS metrics and, by default, HLO/JAXPR dumps. ([GitHub](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/maxtext/configs/base.yml?utm_source=chatgpt.com "maxtext/src/maxtext/configs/base.yml at main · AI-Hypercomputer/maxtext · GitHub"))

So the fundamental purpose is:

> **`run_name` identifies which persistent MaxText run you're operating on.**

---

# 2. What does it actually control?

The most important effect is **checkpoint selection/resumption**.

MaxText's checkpoint logic currently says the priority is:

```text
1. Existing checkpoint associated with run_name
        ↓
   resume latest full checkpoint

2. load_parameters_path
        OR
   load_full_state_path

3. Initialize from scratch
```

And the reason for #1 is explicitly stated: if the job resumes after preemption/hardware failure, it loses the minimum state possible. ([GitHub](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/maxtext/configs/base.yml?utm_source=chatgpt.com "maxtext/src/maxtext/configs/base.yml at main · AI-Hypercomputer/maxtext · GitHub"))

So:

```yaml
run_name: experiment_A
```

is not merely:

```text
"put this name in my logs"
```

It can mean:

```text
"Find the persistent state belonging to experiment_A
 and resume it if one exists."
```

---

# 3. Where does it sit in the execution pipeline?

A useful simplified pipeline is:

```text
Config
  │
  ├── run_name
  │
  ├── base_output_directory
  │
  ↓
Identify persistent run
  │
  ├── checkpoint discovery
  ├── metrics location
  └── diagnostic output location
```

The checkpoint mechanism is therefore the **most important interaction** to remember.

---

# 4. What are the options?

Unlike something like:

```yaml
dtype: bfloat16
```

`run_name` has **no predefined enum**.

It's an arbitrary string:

```yaml
run_name: qwen3_test
```

```yaml
run_name: qwen3_int8_v1
```

```yaml
run_name: pretrain_8b_lr2e-4
```

All are conceptually valid.

The important property is **uniqueness and intentional reuse**.

---

# 5. The critical distinction: new run vs resume

Suppose you previously ran:

```yaml
run_name: qwen_exp1
```

and MaxText created checkpoints.

Now you change your configuration and want a **new experiment**.

Don't accidentally do:

```yaml
run_name: qwen_exp1
```

because MaxText may find the existing checkpoint and resume it.

Instead:

```yaml
run_name: qwen_exp2
```

Conversely, if your TPU job died and you **want to continue the same experiment**:

```yaml
run_name: qwen_exp1
```

That's the intended mechanism.

### Mental rule

> **Same `run_name` → potentially continue the same persistent run.**  
> **New `run_name` → separate experiment namespace.**

---

# 6. `run_name` vs `model_name`

These are easy to confuse.

```yaml
model_name: "default"
run_name: ""
```

`model_name` identifies/configures the **model**.

`run_name` identifies the **experiment/run**.

For example:

```yaml
model_name: qwen3-0.6b
run_name: qwen3_bf16_experiment_1
```

Then:

```text
model_name
    ↓
"What model am I running?"

run_name
    ↓
"Which particular run/experiment is this?"
```

This is also visible in MaxText's own examples: the Kimi K2 pretraining example uses `model_name=kimi-k2-1t` and `run_name=kimi_k2_pre_training`; its fine-tuning example uses the same model but a different `run_name`. ([GitHub](https://github.com/AI-Hypercomputer/maxtext/blob/main/tests/end_to_end/tpu/kimi/Run_Kimi.md?utm_source=chatgpt.com "maxtext/tests/end_to_end/tpu/kimi/Run_Kimi.md at main · AI-Hypercomputer/maxtext · GitHub"))

---

# 7. `run_name` vs `load_parameters_path`

This distinction is even more important.

### `run_name`

```text
Which persistent run am I operating?
```

### `load_parameters_path`

```text
Where do I get model parameters from?
```

For example:

```yaml
run_name: qwen_finetune
load_parameters_path: gs://bucket/qwen_pretrained/checkpoint
```

means roughly:

```text
Current run:
    qwen_finetune

Initial parameters:
    from qwen_pretrained checkpoint
```

That's different from simply resuming `qwen_pretrained`.

MaxText explicitly documents these as separate checkpoint-loading mechanisms. ([GitHub](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/maxtext/configs/base.yml?utm_source=chatgpt.com "maxtext/src/maxtext/configs/base.yml at main · AI-Hypercomputer/maxtext · GitHub"))

---

# 8. What happens if `run_name: ""`?

The current source deliberately uses:

```yaml
run_name: ""
```

as a **sentinel/reminder to choose a real name**, rather than as a meaningful experiment name. ([GitHub](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/maxtext/configs/base.yml?utm_source=chatgpt.com "maxtext/src/maxtext/configs/base.yml at main · AI-Hypercomputer/maxtext · GitHub"))

So in real usage, you'd normally override it:

```bash
run_name=qwen3_0.6b_exp1
```

---

# 9. What should you put in it?

I'd keep it descriptive but not ridiculously verbose:

```yaml
run_name: qwen3_0.6b_bf16
```

or:

```yaml
run_name: qwen3_0.6b_int8_w8a8
```

For a training experiment:

```yaml
run_name: qwen3_0.6b_lr2e-4
```

The key is:

**Does this name uniquely identify the persistent experiment whose checkpoint I am willing to resume?**

If yes, it's a good `run_name`.

---

# 10. One subtle but important point

`run_name` itself **doesn't change the model computation**.

It doesn't affect:

- architecture
    
- dtype
    
- sharding
    
- batch size
    
- optimizer
    
- quantization
    
- sequence length
    

It affects **run organization and persistence**, especially checkpoint discovery.

---

## The one-line intuition

> **`run_name` is the persistent identity of a MaxText run: it organizes that run's artifacts and, crucially, tells MaxText which existing checkpoint namespace to consider for automatic resumption.** ([GitHub](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/maxtext/configs/base.yml?utm_source=chatgpt.com "maxtext/src/maxtext/configs/base.yml at main · AI-Hypercomputer/maxtext · GitHub"))

### Keep this distinction in your head

```text
model_name
    → WHAT model?

run_name
    → WHICH experiment/run?

load_parameters_path
    → FROM WHERE do my initial weights come?
```

