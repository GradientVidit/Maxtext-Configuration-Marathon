## 1. Why does `grain_use_elastic_iterator` exist?

In **Elastic Training** or dynamic cluster topologies, accelerator nodes may join or leave the job dynamically without restarting the entire run. 

Standard dataset iterators assume a static cluster size $N$ and shard datasets by $ ext{worker\_id} \pmod N$. If $N$ changes, standard iterators lose their position or duplicate data.

```text
Standard Iterator: Fixed sharding (Shard = ID % N) ──> Rigid, crashes on dynamic resizing
Elastic Iterator: Dynamic coordinate negotiation    ──> Seamlessly resizes with cluster
```

`grain_use_elastic_iterator: true` enables Grain's elastic iterator, allowing dynamic worker scaling during continuous training.

---

## 2. Mechanics & Strict Constraint

> [!WARNING]
> Elastic training requires `packing: false`.

Because sequence packing aggregates variable numbers of documents into dynamic bins, checkpointing and restoring packed state across changing worker topologies is non-deterministic.

```text
grain_use_elastic_iterator: true  ===> Requires: packing: false
```

---

## 3. Options & Default

| Parameter | Type | Default | Options |
| :--- | :--- | :--- | :--- |
| `grain_use_elastic_iterator` | `bool` | `false` | `true` (enable elastic data streaming), `false` (standard static sharding) |

---

## 4. Interactions with Related Parameters

- **`packing`**: MUST be `false` when `grain_use_elastic_iterator: true`.
- **`dataset_type`**: Must be `"grain"`.

---

## 5. Practical Scenarios & Failure Modes

| Scenario | Symptom / Failure | Fix |
| :--- | :--- | :--- |
| **`grain_use_elastic_iterator: true` with `packing: true`** | MaxText raises config validation error on startup | Set `packing: false`. |

---

### One-line intuition

> `grain_use_elastic_iterator` enables dynamic cluster scaling for Grain data streams, requiring `packing: false` to ensure deterministic iterator checkpoints.
