
## 1. Background: Orbax API versions

Orbax has evolved its checkpointing API over time. The original Orbax API became what is now considered "the legacy API," and Orbax v1 is the newer, redesigned API that introduces architectural improvements.

From MaxText's base.yml:
```yaml
# Bool flag for enabling Orbax v1.
enable_orbax_v1: false
```

---

## 2. What changed in Orbax v1

The core difference is the internal architecture of how CheckpointManager coordinates saves and restores:

| Aspect | Legacy Orbax | Orbax v1 |
|---|---|---|
| API design | Older, evolved organically | Redesigned with clearer abstractions |
| Handler structure | `PytreeCheckpointHandler` etc. | Updated handler interfaces |
| Multi-host coordination | Functional but complex | More explicit and reliable |
| Future support | Maintenance mode | Active development target |

Orbax v1 is the direction the library is heading. The legacy API will eventually be deprecated.

---

## 3. Why it's `false` by default

Despite being the "newer" API, it's not yet the default in MaxText. Reasons:

- **Compatibility**: Existing checkpoints written with the legacy API need to be loadable. A flag-enabled migration path avoids breaking existing workflows.
- **Stability**: New APIs sometimes introduce regressions. The legacy API has been tested across many large-scale runs.
- **Gradual rollout**: MaxText (and Google internally) validates Orbax v1 at scale before making it default.

The comment in MaxText's config is simply `# Bool flag for enabling Orbax v1.` — minimal documentation, indicating this is a migration knob rather than a tuning parameter.

---

## 4. When to enable it

```yaml
enable_orbax_v1: true
```

Consider enabling if:
- You're experimenting with the latest Orbax features
- You're following a MaxText tutorial that explicitly uses Orbax v1
- You want to validate your workflow against the future default

Do NOT enable casually if you have existing checkpoints written with the legacy API — verify that your specific MaxText version's Orbax v1 path can still read those checkpoints.

---

## 5. Interaction with other storage flags

`enable_orbax_v1` changes the checkpoint manager/handler infrastructure, but the underlying storage format choices (`checkpoint_storage_use_ocdbt`, `checkpoint_storage_use_zarr3`) are still controlled by their respective flags regardless of which API version is used.

---

## 6. Default

```yaml
enable_orbax_v1: false
```

---

### One-line intuition

> **`enable_orbax_v1` is a migration flag that switches MaxText from Orbax's legacy checkpointing API to the newer, redesigned v1 API — off by default for stability; enabled when you want to opt into the future direction.**
