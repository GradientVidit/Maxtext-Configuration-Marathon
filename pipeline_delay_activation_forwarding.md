
## 1. The overlap problem

In a pipelined forward pass, when stage N finishes a microbatch it must **send activations** to stage N+1 before it can start the next microbatch. This send/receive communication and the next microbatch's compute happen in the same loop iteration — XLA has to figure out how to overlap them. For short pipeline stages (low compute time), this overlap is hard for the compiler to schedule cleanly because the communication is a dependency of the next iteration's compute.

`pipeline_delay_activation_forwarding` decouples them: the activation forward from iteration K is delayed by one full loop iteration, so iteration K+1's compute and iteration K's communication are truly independent and XLA can freely overlap them.

---

## 2. Before / after

```text
Without delay (default):

Iteration K:
  [compute microbatch K] → [send activations K → stage+1]
  XLA must overlap send with compute of iteration K+1 = dependency chain

With delay:

Iteration K:
  [compute microbatch K] | [send activations K-1 → stage+1]
  These have no data dependency → XLA can schedule freely
```

The activation forwarding from iteration K is sent during iteration K+1's compute window — true compute/communication overlap with no compiler guessing required.

---

## 3. The cost: doubled bubble

Delaying activation forwarding adds **one extra pipeline startup phase**:

```text
Normal bubble:  (num_stages - 1) pipeline iterations
Delayed bubble: 2 × (num_stages - 1) pipeline iterations
```

The bubble fraction doubles. This is only worth it if the improved overlap efficiency outweighs the larger bubble — which typically requires long enough pipeline stages (enough compute per stage) to make the communication hide completely inside the compute window.

---

## 4. The microbatch floor

Because of the doubled startup phase, you need more microbatches to amortize it:

```text
Without delay: num_pipeline_microbatches ≥ num_stages
With delay:    num_pipeline_microbatches ≥ 2 × num_stages  (enforced as default when enabled)
```

MaxText sets this minimum automatically when `pipeline_delay_activation_forwarding=true`.

---

## 5. Options

| Value | Behavior |
|---|---|
| `false` | Default — immediate activation forwarding; XLA attempts overlap |
| `true` | Delayed forwarding; explicit compute/comm independence at cost of 2× bubble |

Default: `false`.

---

## 6. When to enable

Enable when:
- You have many pipeline stages and inter-stage communication is the bottleneck
- XLA's overlap is measurably underperforming (profileable via HLO/xplane)
- You have enough microbatches to absorb the doubled bubble (`≥ 2 × num_stages`)
- `num_layers_per_pipeline_stage` is large enough that per-stage compute is long

Don't enable when:
- The pipeline already has a large bubble (low `num_pipeline_repeats` or `num_pipeline_microbatches`)
- Per-stage compute is short — doubling the bubble hurts more than overlap helps
- You're memory-constrained — more microbatches increase peak activation memory

---

### One-line intuition

> **`pipeline_delay_activation_forwarding` shifts activation sends one iteration later to give XLA unconditionally independent compute and communication, trading a doubled bubble for guaranteed overlap — worth it only when inter-stage communication is actually the bottleneck.**
