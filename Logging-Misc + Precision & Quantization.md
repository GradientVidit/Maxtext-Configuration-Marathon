
## Run-level logging & misc

These sit between the checkpointing block and the precision/quantization block — small, easy to overlook, but worth knowing.

|Param|Purpose|Options / meaning|
|---|---|---|
|`reuse_example_batch`|Debug/benchmarking aid: repeatedly feeds the _same_ batch instead of advancing through the dataset, to isolate TPU step-time from data-loading time.|`0` = off (normal training); `1` (or truthy) = reuse the same batch every step.|
|`metrics_file`|Local file path to write scalar metrics to, for testing.|Path string, or `""` = don't write metrics locally.|
|`gcs_metrics`|Whether to also write metrics (loss, TFLOPs, etc.) to GCS under `{base_output_directory}/{run_name}/metrics/`.|`true`/`false`.|
|`enable_wandb`|Turn on Weights & Biases logging.|`True`/`False`.|
|`wandb_project_name`|W&B project name to log runs under.|String, default `"maxtext"`.|
|`wandb_run_name`|W&B run name.|String, defaults to empty (W&B will auto-name).|
|`save_config_to_gcs`|Whether to save a copy of the resolved config to GCS under `{base_output_directory}/{run_name}/`.|`true`/`false`. Handy for reproducibility — lets you see exactly what config a past run actually used.|

## Precision & Quantization

This is one of the more consequential sections for anyone doing serious pretraining work — it controls numerical precision throughout the model, and is where you'd configure post-training quantization (PTQ) or quantization-aware setups.

### Base dtypes

|Param|Purpose|Options / meaning|
|---|---|---|
|`grad_dtype`|The dtype gradients are computed/accumulated in.|String, default `"float32"` — kept at fp32 even when weights/activations are lower-precision, since gradient accumulation is sensitive to precision loss.|
|`dtype`|The dtype used for activations throughout the model.|String, default `"bfloat16"`. This is the main "what precision does the model actually run at" knob for activations (as opposed to `weight_dtype`, covered in Part 3).|
|`activations_in_float32`|Forces activations to fp32 immediately before a nonlinearity (e.g. before SiLU/GELU), regardless of `dtype`.|`true`/`false`. A targeted precision boost at a specific point where bf16 rounding is more likely to hurt.|
|`matmul_precision`|XLA's precision setting for matrix multiplies — trades speed for numerical accuracy on the actual MXU computation.|`"default"`, `"high"`, or `"highest"` (maps to `jax.lax.Precision`). `"default"` is fastest; `"highest"` uses more MXU passes for extra precision.|

### Quantization (weights/activations)

|Param|Purpose|Options / meaning|
|---|---|---|
|`quantization`|Master switch for which quantization scheme to apply to the transformer layers. Empty means no quantization — full bf16.|`""` (off/bf16), `"int8"` (dynamic-range 8-bit), `"intmp"` (mixed-precision int, inference-oriented — see `src/maxtext/configs/quantization/readme.md`), `"fp8"` (8-bit float GEMMs, NVIDIA GPUs), `"nanoo_fp8"` (8-bit float GEMMs, AMD MI300/MI325 GPUs), `"fp8_full"` (fp8 with _static_ scaling rather than dynamic).|
|`constant_bound_config`|Configures AQT's static-scaling bounds when using a quantization scheme that needs fixed (not dynamically calibrated) scale bounds.|Comma-separated floats, e.g. `'0.5, 0.5, 0.5, 0.5, 0.5, 0.5'`. Empty = not used.|
|`replicate_quant_scale`|Replicates the quantization scale factor across devices to avoid an inefficient XLA fusion pattern under 2D sharding.|`true`/`false` — a performance tuning knob, not a correctness one.|
|`quant_cfg_path`|Path to a JSON/config file specifying per-layer quantization settings when using `"intmp"`.|Path string.|
|`checkpoint_is_quantized`|Tells MaxText the checkpoint being loaded was already saved in quantized (AQT) form, so it should be read back correctly rather than assuming full-precision weights.|`true`/`false`.|
|`save_quantized_params_path`|If set, saves an on-the-fly quantized copy of the params to this path (e.g. for producing a deployable quantized checkpoint from a full-precision one).|Path string, `""` = don't save.|
|`use_qwix_quantization`|Switches the quantization backend from AQT to Qwix. **AQT is being deprecated** — the inline comment explicitly recommends setting this to `true`.|`true`/`false`. Worth defaulting to `true` on new work given the deprecation notice.|
|`use_manual_quantization`|Enables manual quantization handling for batch-split scheduling; only relevant when `use_batch_split_schedule=true` (covered in the MoE part).|`true`/`false`.|
|`weight_quantization_calibration_method`|Calibration method Qwix uses to determine quantization ranges for **weights**.|`"absmax"` (default) or other methods supported by [Qwix's `qconfig.py`](https://github.com/google/qwix/blob/dc2a0770351c740e5ab3cce7c0efe9f7beacce9e/qwix/qconfig.py#L70-L80).|
|`act_quantization_calibration_method`|Same, but for **activations**.|`"absmax"` default; same option set as above.|
|`bwd_quantization_calibration_method`|Same, but for the **backward pass** (gradients w.r.t. quantized ops).|`"absmax"` default; same option set as above.|
|`quantization_local_shard_count`|How many shards to split the quantization range-finding operation across.|`-1` = default to number of slices; positive integer overrides.|

### KV-cache quantization

|Param|Purpose|Options / meaning|
|---|---|---|
|`quantize_kvcache`|Whether to quantize the KV cache itself (separate from weight/activation quantization) — mainly an inference/serving memory optimization.|`true`/`false`, default `false`.|
|`kv_quant_axis`|Which axis (or axes) of the KV cache to quantize over. Only meaningful when `quantize_kvcache=true`.|`""` (only valid when quantization is off), `"dkv"` (quantize over the KV feature-dimension axis — better accuracy, worse compute), `"heads_and_dkv"` (quantize over both cache-heads and KV-dim axes — default, faster compute).|
|`kv_quant_dtype`|The dtype to store the quantized KV cache in.|String, default `"int8"`.|

### Misc precision-adjacent

|Param|Purpose|Options / meaning|
|---|---|---|
|`model_call_mode`|Declares what mode the model is being invoked in — affects some precision/quantization code paths internally.|`""` (default = training) or `"inference"`.|
|`weight_sparsity_n` / `weight_sparsity_m`|Define **N:M structured sparsity** on weights — at most N non-zero values in every block of M.|Both `null` by default (sparsity off). Set both to enable, e.g. `n=2, m=4` for 2:4 sparsity.|
|`weight_sparsity_update_step`|How often (in steps) the sparsity mask is recomputed once sparsity is enabled.|Integer, default `10`.|
|`weight_sparsity_start_step`|How many steps to train densely before sparsity masks kick in.|Integer, default `50`.|
|`te_comm_gemm_overlap`|Whether to use TransformerEngine's collective-GEMM-overlap algorithm to hide communication behind quantized MLP compute. Only valid with tensor-sequence parallelism.|`"disabled"` (no overlap), `"mlp"` (overlap only the MLP up/down projections), `"full"` (overlap MLP _and_ attention's QKV/output projections too).|

## Notes to self

- `dtype` (activations) vs. `weight_dtype` (Part 3) vs. `grad_dtype` are three separate knobs — a model can legitimately run bf16 activations, fp32 weights master-copy, and fp32 gradient accumulation simultaneously. Don't conflate them.
- The explicit deprecation note on `use_qwix_quantization` ("AQT will be removed... strongly recommended to set to true") is a maintainer signal worth taking seriously for any new pretraining config — AQT is legacy here.
- KV-cache quantization (`quantize_kvcache`) is a distinct axis from weight/activation quantization (`quantization`) — you can quantize the KV cache for serving-time memory savings without touching training-time weight precision at all.