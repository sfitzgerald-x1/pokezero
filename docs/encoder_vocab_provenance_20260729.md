# Encoder vocabulary provenance: the third axis

**Date:** 2026-07-29
**Status:** consumption side latched and fail-closed. Follows #948, which fixed the export
side and corrected the polarity of the original finding.
**Base:** `dfb4d10` · Python-only change (no engine patch; the probe below runs through the
node Showdown bridge, not the Rust engine, so no engine fingerprint is load-bearing).
**Probe checkpoint:** `pz-v2-2-1m.pt`, sha256 `b0c73928ac3dc26b…`, trained on **1216** tokens.

---

## 1. The anatomy: three axes, two latched

A checkpoint's observation contract has three independent axes. Every one of them must be
resolved from the checkpoint's stamped provenance, because every one of them can differ
between the build that trained the model and the build that is now scoring it.

| axis | derived by | latched through the env? |
| --- | --- | --- |
| feature masks | `feature_masks_from_model_config` | yes — `required_masks` |
| observation spec | `observation_spec_from_model_config` | yes — `required_specs` |
| **category vocabulary** | **(did not exist)** | **NO — no parameter existed** |

Two of three. The third had a contract, and the contract was a comment:

```python
# local_showdown.py, LocalShowdownConfig.category_vocab (before)
# ... callers that pair the env with a specific model MUST pass the model's
# vocabulary here so encode-time rows match the embedding exactly (no silent row drift).
```

Nothing enforced the MUST, and **no production consumption site obeyed it.** Every one fell
through to `local_showdown.py`'s `self.config.category_vocab or gen3_category_vocabulary(root, …)`.

### Why this axis is the dangerous one

Masks and spec disagreements change the observation's **shape**, so they tend to surface as
a shape error in the forward pass — loud, immediate, hard to ship. The vocabulary is a
positional list of the **same width** whichever build wrote it. A token inserted mid-list
renumbers everything after it, and the encoder produces a perfectly well-formed tensor of
embedding rows that mean something else. Nothing crashes. The model just scores a state it
never trained on, and the run looks clean.

That asymmetry is the whole finding: **the axis that was left unlatched was the only one
whose failure is silent.**

---

## 2. Sites: what was checkpoint-anchored, what was not

Creation sites are correctly build-anchored and **must stay that way** — a new model should
stamp today's enumeration. The bug is entirely on the consumption side.

### Fixed here (consumption)

| site | spec/masks | vocab before | vocab after |
| --- | --- | --- | --- |
| `online_client.py:build_agent` | ckpt | **build** | ckpt |
| `online_client.py:build_agent_remote` | served ckpt | **build** | ckpt |
| `foulplay_bridge.py` controlled benchmark | ckpt | **build** | ckpt |
| `neural_cli.py` public-corpus profile | ckpt | **build** | ckpt |
| `local_showdown.env_config_from_checkpoint_provenance` | ckpt | **absent** | `required_vocabs`, **fail-closed** |
| `neural_cli` `_env_config_with_{matchup,mixed_history,spec}_masks` | ckpt | build fall-through | ckpt |
| `collection.env_config_with_policy_spec_masks` | ckpt | build fall-through | ckpt |
| `engine_search.py` MCTS model benchmark | ckpt | build fall-through | ckpt |
| `scenario_studio/service.py` root eval | ckpt | build fall-through | ckpt |
| `scripts/{k0_grid_h2h,mcts_acceptance_h2h,play_against_checkpoint,policy_probe}.py` | ckpt | build fall-through | ckpt |

`build_agent` is the widest: **10 probe/eval scripts** reuse `agent.vocab`
(`trait_eval.py`, `foulplay_choice_rank_probe.py`, `behavior_probe.py`, `hazard_probe.py`,
`collapse_probe.py`, `choice_sample.py`, `policy_js_divergence_probe.py`,
`hazard_blind_spot_audit.py`, `checkpoint_factors.py`, `policy_probe.py`). All inherit the
fix without their own change.

### Correctly build-anchored — left alone

`neural_cli.py:2906` (fresh `neural train`) and `:5490` (`neural iterate`) mint a **new**
`TransformerPolicyConfig.compact_category(...)`; stamping today's enumeration is the right
behaviour. Model-free encoder audits likewise.

`neural train --initial-checkpoint` was already correct — but only incidentally, via
`replace(initial_training_result.model_config, …)` carrying `category_vocab` along with
everything else, not by any deliberate latch.

### Notes for other lanes (not fixed here)

* `scripts/power_h2h.py` **does not exist**; it is referenced historically. Its successors
  are `k0_grid_h2h.py` and `mcts_acceptance_h2h.py`, both fixed above.
* `export_encoder_tables.py` and `mcts_eval/resolver.py` are **#948's** territory.
* `neural iterate` is a **hybrid**: it correctly mints a new config from the build, but then
  admits pre-existing `neural:` checkpoints (initial policy, fixed opponents, benchmark
  references) into that same env. Those are consumption participants; the latch now covers
  them, but the hybrid shape is worth remembering.
* `EngineEnvConfig` has **no vocabulary field at all** — it encodes from exported tables, so
  for engine-backed paths this provenance can only be enforced through #948's export gate.

---

## 3. Quantified: the "13 rare volatiles" premise was wrong

The expectation going in was that this would be tiny — thirteen rare volatiles, bounded
under a point. **Measured, it is not.**

Paired probe: one checkpoint, one set of states, two encodings. The same weights see the
same state encoded with its own 1216-token enumeration and with the build's 1217. Weights
and states are identical, so there is no game-sampling variance to bound — the difference is
the enumeration and nothing else.

| | |
| --- | --- |
| games / decision boundaries | 60 / **5,410** |
| encodings differed | 1,054 = **19.48 %** of decisions |
| **argmax action differed** | 201 = **3.72 %** of decisions |

### Why the premise failed

`volatile:solarbeam` was inserted at index 1204 of a **sorted** list, so it renumbers
everything alphabetically after it. That tail is not rare volatiles:

```
volatile:stockpile  substitute  taunt  torment  uproar  watersport  yawn
weather:hail  weather:raindance  weather:sandstorm  weather:sunnyday
winner:none
```

**All four weather tokens are in the shifted tail.** Attributing the divergence to specific
tokens over 2,671 decisions:

| token | appears in |
| --- | --- |
| `weather:sunnyday` | **12.0 %** of decisions |
| `volatile:substitute` | **5.5 %** |
| `weather:raindance` | **4.4 %** |

Weather and Substitute are ordinary gen3 features, not edge cases. The reachability argument
assumed the *inserted* token was rare; what matters is how common the *renumbered* tokens
are, and sorted insertion makes that a completely different question.

**Consequence:** any raw-play evaluation of a 1216-trained checkpoint on a ≥1217 build is
contaminated at roughly one action in twenty-seven. That is well above the <1 pt bound this
was expected to sit under, and it argues for re-running rather than reinterpreting affected
results. This does **not** establish a strength delta — a changed action is not necessarily
a worse one — and a head-to-head would be needed to put a number on that. What is
established is that the contamination is common, not marginal.

---

## 4. The fix, and why it is fail-closed

`category_vocab_from_model_config(config, showdown_root)` joins its two siblings as the third
derivation point. Tokens and OOV width come from the checkpoint; `showdown_root` supplies
only the **aliases**, which are a mapping *onto* rows rather than an enumeration *of* them —
each alias resolves through the vocabulary's own index and is dropped if its base is absent,
so aliases cannot shift a trained row. (The previous build path composed aliases the same
way, so only the token source changed.)

`env_config_from_checkpoint_provenance` gains `required_vocabs` with the same adopt / agree /
conflict semantics as the other two axes, plus one addition: **supplying any checkpoint
provenance without a vocabulary raises.** Mirrors #948's required-not-defaulted move. Every
valid `TransformerPolicyConfig` carries `category_vocab` (enforced in `__post_init__`), so
an absent vocabulary is always an un-updated call site, never a legitimate case. An env with
no checkpoints in play is still a no-op.

This is deliberately noisy. It caught six call sites in the test suite on the first run,
which is the point of the design.

One behavioural change worth knowing: vocabulary resolution is now **eager** at latch time
rather than lazy at first `observe()`. Fails faster; also means a harness pointed at a
stand-in `showdown_root` reaches the alias lookup during config construction.

### Naming

`env_config_from_checkpoint_provenance` now latches three axes and its name says one. Kept
deliberately — renaming churns ten call sites and buries the enforcement change — but the
authority is the signature and the fail-closed check, not the name. **Flagged for the
reviewer**: say the word and it becomes `env_config_from_checkpoint_provenance`.

---

## 5. Standing rule

> **Provenance latching must cover every enumeration axis, not the ones that have bitten us.**

The masks axis was latched after #492 bit. The spec axis was latched when dual-schema
forced it. The vocabulary axis was latched today, after it bit. In each case the axis was
knowable from the checkpoint the whole time, and in each case the gap was documented before
it was closed — this one for months, in a comment that said MUST.

Two corollaries earned here:

1. **A comment is not a latch.** `LocalShowdownConfig.category_vocab` told callers they MUST
   pass the model's vocabulary. Every production caller ignored it, and nothing noticed.
   If a contract matters, it raises; if it does not raise, assume it is being violated.
2. **A half-latch reads as a latch.** Two sites carried the comment *"the vocabulary axis
   latches with the schema"* and were half right — they latched which token **families**
   exist (`include_turn_merged`) and never the **order** within them. Order is what indexes
   the embedding. A partially-correct provenance claim is worse than none, because it stops
   the next reader from looking.
