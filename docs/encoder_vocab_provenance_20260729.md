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

### Provenance of the shift

| | |
| --- | --- |
| commit | **`b59c1ea`** — *engine_world: surface mid-charge (Solar Beam) state to world construction* |
| authored | 2026-07-28 21:51:05 -0700 |
| merged to main | **PR #909**, **2026-07-29T05:01:21Z** |
| mechanism | added `"solarbeam"` to `TRACKED_VOLATILES` (`showdown.py`); `randbat_vocab.GEN3_VOLATILES = tuple(sorted(TRACKED_VOLATILES))` then places `volatile:solarbeam` at sorted index 1204 |

The enumeration grew 1216 → 1217 at that merge, and **2026-07-29T05:01Z is the moment the
contamination window opens.** Anything evaluated before it is clean by construction.

Note the shape: the commit was about *world construction for search*, and grew the
observation vocabulary as a side effect. Nothing in it is wrong. The enumeration is simply a
shared surface that any token-set change perturbs.

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

### For harness authors: resolution is now eager

Vocabulary resolution happens at **latch time**, not lazily at first `observe()`. Deliberate
— fail-at-construction beats fail-at-first-observe — but it has one visible consequence:

**A harness pointed at a stand-in `showdown_root` now reaches the alias lookup during config
construction.** Previously a fake root survived until something actually encoded. If your
harness passes a temp dir or a fixture path, stub the alias builder:

```python
patch("pokezero.randbat_vocab.gen3_category_string_aliases", return_value={})
```

Stub *that*, not `gen3_category_vocabulary` — the tokens no longer come from the build, so
patching the build's vocabulary builder no longer intercepts this path. Three in-tree tests
needed exactly this change.

### Naming

Renamed to **`env_config_from_checkpoint_provenance`** (was `env_config_with_checkpoint_masks`).
The old name asserted one axis while the function latched three — the same failure as the rest
of this entry, in function-name form. **No compatibility alias**: an alias would let
un-updated callers keep working, which is the opposite of the point.

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

3. **Ask about the RENUMBERED tokens, not the inserted one.** The reachability argument here
   was "one rare volatile, therefore negligible" — and it was asking about the wrong set. In
   a **sorted** enumeration, an insertion's blast radius is *everything alphabetically after
   it*, and the inserted token's own rarity says nothing about that tail's. `volatile:solarbeam`
   is genuinely rare; it renumbered all four `weather:` tokens. Sorted insertion converts a
   question about one token into a question about a suffix.

   This joins **membership-not-cardinality** as a standing methodology rule: both are cases
   where the cheap summary statistic (how rare, how many) is not the quantity that governs
   the failure. When a positional structure changes, enumerate what MOVED.

---

## 6. Method note: the class label / the comment / the name

Three artifacts in this entry each described the system accurately enough to stop the next
reader from looking, and each was wrong in the same direction:

* the **comment** on `LocalShowdownConfig.category_vocab` said callers MUST pass the
  vocabulary — a real contract, unenforced, universally violated;
* the **half-true provenance comment** at two sites — *"the vocabulary axis latches with the
  schema"* — described a latch that covered families and not order;
* the **function name** `env_config_with_checkpoint_masks` — accurate about one of the three
  axes it latched.

None was a lie. Each was a *partial truth positioned where a reader looks for the whole one*,
and that is the more dangerous artifact, because a blank space invites investigation and a
plausible answer ends it. The corresponding habit: when a label, comment, or name asserts
coverage, check the coverage rather than the claim — and when you fix the mechanism, fix the
artifact that concealed it in the same change.

---

# Appendix A — Exposure inventory (list only; re-runs need separate authorization)

Scope: artifacts produced **after 2026-07-29T05:01:21Z** (the #909 merge) using a
**pre-solarbeam checkpoint** through a **build-anchored** consumption path. Everything
produced before that instant is clean by construction.

## A.0 First: 1233 vs 1216 is not a discrepancy

`docs/mcts_k0_depth_grid_20260729.md` §6.3 says the cached tables carry a "**1233**-token
vocab" against this build's "**1234**"; this entry says 1216 → 1217. **Same population, two
units.** `CategoryVocabulary.size` is `1 + len(tokens) + oov_buckets`:

```
1 pad + 1216 tokens + 16 oov = 1233 rows      (pre-#909)
1 pad + 1217 tokens + 16 oov = 1234 rows      (post-#909)
```

This matters for the inventory: it means the k0/k64 v3hist checkpoints, described only by
row count, are **confirmed pre-solarbeam** rather than unknown. Anything reporting 1233 rows
is a 1216-token model.

## A.1 Classification

| artifact | date (PDT) | checkpoint | class |
| --- | --- | --- | --- |
| `docs/audit_artifacts/hc-depth-grid-20260729/` `hc-d{1,2,4,6,8}.json` | 07-29 04:17 | `pz-v2-2-1m.pt` (1216) | **SHIFTED** |
| `docs/audit_artifacts/hc-depth-grid-20260729/calibration/vs-max-damage.json` | 07-29 04:17 | `pz-v2-2-1m.pt` (1216) | **SHIFTED** |
| `docs/audit_artifacts/hc-sims-grid-20260729/` (4 cells) | 07-29 04:17 | `pz-v2-2-1m.pt` (1216) | **SHIFTED** |
| `docs/mcts_handcrafted_leaf_depth_findings.md`; `docs/mcts_degradation_findings.md` §12 | 07-29 04:17 / 04:25 | reports the above | **SHIFTED** (inherits) |
| `docs/audit_artifacts/hc-depth-grid-20260729/control.json` (raw v raw) | 07-29 04:17 | `pz-v2-2-1m.pt` (1216) | SYMMETRIC-SHIFTED |
| `docs/audit_artifacts/k0-depth-grid-20260729/results/` (56 cells, arms a/c/ctl/k64c) | 07-29 03:50 & 04:42 | `v3hist-k0-…-2519`, `v3hist-k64-…-2657` (1233 rows = 1216 tokens) | SYMMETRIC-SHIFTED |
| `…/k0-depth-grid-20260729/summary.json`, `report.txt`; `docs/mcts_k0_depth_grid_20260729.md` §4 | 07-29 04:42 | as above | SYMMETRIC-SHIFTED (inherits) |
| `…/k0-depth-grid-20260729/reference-k64-asshipped/` (5 shards) | 07-29 04:42 | k64 @2657 | **INDETERMINATE** |
| `docs/mcts_degradation_findings.md` §9 falsifying re-bench | 07-29 02:56 | k64 @2657 | **INDETERMINATE** |
| `docs/mcts_degradation_findings.md` §11 + `scripts/mcts_seat_split.py` | 07-29 03:48 | — | re-analysis only; inherits its inputs |

### Why the hc grid splits from the k0 grid

Both ran build-anchored on post-#909 builds, but their **arm structure differs**, and that is
what decides the class.

* `hc-d<N>` cells pit `EngineMctsPolicy(leaf_eval="hp_fraction_crate", …)` — constructed with
  `dex` and `set_source` and **no network at all** — against `raw_policy = policy_from_spec(raw_spec)`,
  which is the neural checkpoint. **Only one seat is perturbed.** The measured
  handcrafted-vs-raw gap moves with the perturbation, so these need re-running.
  `vs:max-damage` is the same shape against a scripted baseline.
* k0/k64 cells are "engine MCTS vs the SAME checkpoint's raw policy" (§3), so both seats run
  the same perturbed policy: the comparison is internally consistent, absolute win rates are
  against a perturbed reference. `control.json` in the hc grid is raw-v-raw and joins this class.

**SYMMETRIC-SHIFTED is not "fine".** It means the *contrast* survives and the *level* does
not. §4.2's headline — d6 = 0.360 reproducing §9's 0.360 exactly on freshly exported tables —
is a statement about two encode sites agreeing **with each other**, which §6.3 correctly
framed as fixing crate-leaf-vs-root drift. It is not evidence that either agrees with the
**checkpoint**, and the fresh export is precisely what moved both onto the build's
enumeration. The reproduction is real; its interpretation as "vocab is not the cause" holds
only for the crate-vs-root question it was testing.

## A.2 Clean — checked and dated, not assumed

| class | newest artifact | verdict |
| --- | --- | --- |
| trait-tracking reports | `5818012` **07-25 12:22** | CLEAN — pre-dates #909 by 4 days |
| foul-play evals | `cc0d26c` **07-21 20:49** | CLEAN |
| `online_client.build_agent` lane (10 scripts) | `evals/` `29a0a6c` **07-28 01:23** | CLEAN — pre-dates the merge by ~21 h |
| `reports/c6_*.json` (engine transition census) | 07-29 05:06 | CLEAN — post-cutoff but **no model in the loop**; sim-vs-engine only |
| `runs/` gates | 2026-07-04 | CLEAN |

**MCTS acceptance: nothing to re-run — it never ran.** `docs/mcts_acceptance_rebench_plan.md`
reads *"Status: STAGED, NOT LAUNCHED"*, and a repo-wide search finds runner, report and test
but no results. `scripts/mcts_acceptance_h2h.py` received `required_vocabs` in this PR
**before its first launch**, so it will run vocab-anchored from the start.

## A.3 Two gaps worth closing regardless of re-run decisions

1. **The artifacts cannot self-report this contamination.** k0 provenance blocks record
   `tables_path`, `model_path`, `tables_masks`, `checkpoint_masks` and a `mask_drift` field —
   grep for "vocab" across every k0/hc artifact hits only `README.md`. The mask axis is
   audited in the artifact; the vocabulary axis is invisible. Adding the checkpoint's token
   count (or the `category_vocab_sha256` #948 puts in the contract) to the provenance block
   would make every future cell self-classifying.
2. **§8 already prescribes a k0/k64 re-run** — for #937/#939 and for statistical power
   (n≈400/cell) — and the vocabulary axis is **not** among its stated reasons. If that re-run
   happens on current main it will be vocab-anchored automatically; the point is that §8's
   kill criteria were written without this axis in view and should be re-read with it.
