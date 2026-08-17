
## 1. Why does it exist?

Normally, LoRA fine-tuning gives you:

```text
Base model
   +
LoRA adapter
   ↓
adapted model
```

A LoRA adapter is small because it contains only the low-rank updates, not another copy of the entire model.

For serving, you may want **many adapters for the same base model**:

```text
                    Base model
                       │
          ┌────────────┼────────────┐
          ↓            ↓            ↓
       adapter 1    adapter 2    adapter 3
       medical      coding       legal
```

`lora_input_adapters_path` provides MaxText with the **parent directory containing these adapters**.

---

## 2. What does the directory look like?

The important part of the config comment is:

```text
parent_directory/
    ├── lora_id_1/
    ├── lora_id_2/
    ├── lora_id_3/
    └── ...
```

For example:

```text
gs://my-bucket/lora_adapters/
    ├── medical/
    ├── coding/
    └── legal/
```

Then:

```yaml
lora_input_adapters_path: gs://my-bucket/lora_adapters/
```

Each subdirectory represents a **LoRA adapter identified by its `lora_id`**. ([GitHub](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/maxtext/configs/base.yml?utm_source=chatgpt.com "maxtext/src/maxtext/configs/base.yml at main · AI-Hypercomputer/maxtext · GitHub"))

---

## 3. Why "multiple adapters"?

This is different from:

```yaml
load_parameters_path: gs://.../base_model
```

which loads the **base model parameters**.

Think:

```text
load_parameters_path
        ↓
   Base model weights
        │
        │
        ├───────────────┐
        ↓               ↓
   LoRA adapter A   LoRA adapter B
```

`lora_input_adapters_path` is therefore an **adapter repository**, not a model checkpoint repository.

---

## 4. What is `lora_id`?

The subdirectory name acts as the adapter's identifier.

For example:

```text
gs://bucket/adapters/
    ├── 0/
    ├── 1/
    └── 2/
```

could represent:

```text
lora_id = 0
lora_id = 1
lora_id = 2
```

The important design is:

```text
lora_input_adapters_path
        ↓
parent directory
        ↓
   lora_id/
        ↓
adapter checkpoint
```

So the parameter itself **doesn't identify one adapter**. It identifies the **parent containing the collection**.

---

## 5. When would you use it?

Primarily when your serving/inference system needs to work with **multiple LoRA adapters on one base model**.

For example:

```text
Qwen base model
      │
      ├── customer_A adapter
      ├── customer_B adapter
      ├── customer_C adapter
      └── customer_D adapter
```

Instead of maintaining four complete model copies, you maintain:

```text
1 × base model
+
4 × tiny LoRA adapters
```

and select the appropriate adapter for a request.

This is particularly useful for **multi-tenant LoRA serving**.

---

## 6. What are its options?

It's a **path string**:

```yaml
lora_input_adapters_path: ""
```

### Empty

```yaml
lora_input_adapters_path: ""
```

→ no LoRA adapter directory is specified.

### GCS

```yaml
lora_input_adapters_path: gs://bucket/lora_adapters
```

### Local

```yaml
lora_input_adapters_path: /path/to/lora_adapters
```

The current base config specifically describes the intended input as a GCS path, while MaxText's broader LoRA tooling also supports local paths for individual HF adapters. ([GitHub](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/maxtext/configs/base.yml?utm_source=chatgpt.com "maxtext/src/maxtext/configs/base.yml at main · AI-Hypercomputer/maxtext · GitHub"))

---

## 7. Don't confuse it with `hf_lora_adapter_path`

MaxText now has both:

```yaml
lora_input_adapters_path: ""
hf_lora_adapter_path: ""
```

They serve different purposes.

### `hf_lora_adapter_path`

```text
Hugging Face LoRA adapter
          ↓
conversion/loading
```

It can point to an HF repository ID or local HF adapter path. ([GitHub](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/maxtext/configs/base.yml?utm_source=chatgpt.com "maxtext/src/maxtext/configs/base.yml at main · AI-Hypercomputer/maxtext · GitHub"))

### `lora_input_adapters_path`

```text
MaxText adapter directory
          ↓
parent/
 ├── lora_id_1
 ├── lora_id_2
 └── lora_id_3
```

So:

> **`hf_lora_adapter_path` = where an HF-format adapter comes from.**

> **`lora_input_adapters_path` = where MaxText's collection of LoRA adapters is located.**

The current LoRA documentation also distinguishes the base model checkpoint from adapter weights: `load_parameters_path` supplies the frozen base weights, while the LoRA-specific path supplies adapter weights. ([MaxText](https://maxtext.readthedocs.io/en/latest/tutorials/posttraining/lora_on_multi_host.html?utm_source=chatgpt.com "LoRA Fine-tuning on multi-host TPUs — MaxText documentation"))

---

## One-line intuition

> **`lora_input_adapters_path` points MaxText to a parent directory containing multiple LoRA adapters, organized as `lora_id` subdirectories, so the adapters can be loaded/selected without duplicating the base model.** ([GitHub](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/maxtext/configs/base.yml?utm_source=chatgpt.com "maxtext/src/maxtext/configs/base.yml at main · AI-Hypercomputer/maxtext · GitHub"))

```text
Base model
    +
lora_input_adapters_path
    ↓
{ lora_id → LoRA adapter }
```

**Important:** this is a newer/specialized LoRA-serving parameter; you don't need it for ordinary full fine-tuning or ordinary single-adapter LoRA training. The current LoRA fine-tuning workflow instead uses parameters such as `lora.enable_lora`, `lora.lora_rank`, `lora.lora_alpha`, and `lora_restore_path`. ([MaxText](https://maxtext.readthedocs.io/en/latest/tutorials/posttraining/lora_on_multi_host.html?utm_source=chatgpt.com "LoRA Fine-tuning on multi-host TPUs — MaxText documentation"))