# Gen 3 Randbats — v3 observation schema-freeze closure

**Status: CLOSED at the plan's local bar.** This is the successor to the audit
[findings ledger](silent_noop_sweep_findings.md): it coalesces the per-candidate
verdicts, the engine-lane findings, and the schema-freeze recommendation into one
clean record. The findings ledger's Wish-inclusive **V6 cluster verification
cycle** remains a stricter, ongoing check the deploy workstream maintains
separately -- it is not a gate on the freeze (the plan's Constraints require no
cluster-scale collection). Companion to the [plan](silent_noop_sweep_plan.md).

## Evidence Rules (local bar — per the plan's Constraints)

A completed layer or candidate verdict counts as evidence when it carries
**local, reproducible provenance**. No cluster-scale aggregate, immutable image
digest, or persistent-job terminal artifact is required — that bar exceeds what
the plan specifies for these layers.

| Required provenance | Purpose |
| --- | --- |
| Public-repository commit | Pins encoder, belief, parser, and audit code. |
| Showdown / engine source | Pins the reachable Gen 3 universe + simulator behavior: the vendored gen3 Showdown (`data/random-battles/gen3/`) and the gen3 poke-engine. |
| Observation schema | `pokezero.observation.v3`. |
| Reproduction | `file:line` for `COVERED`; pool/ruleset evidence for `UNREACHABLE`; a fixture or probe for an `ADD`. |

## Verdict rubric

Exactly one of **`ADD`**, **`COVERED`**, or **`UNREACHABLE`** per candidate.
`ACCEPTED-LOSS` is **not** a permitted verdict (owner rule: *"there is no such
thing as an accepted loss"*): any reachable conflation is an `ADD`; only a fact
already encoded (`COVERED`, cite the column) or one that genuinely cannot occur
in gen3 randbats (`UNREACHABLE`, cite pool/ruleset) is a non-ADD outcome.

## Completed layers (local bar)

| Layer | Status | Evidence |
| --- | --- | --- |
| 1 — Static emission inventory + handled-set diff | Complete | Reachable gen3 protocol/`-activate`/`cant` signatures enumerated vs the pool; dispatch tables in `showdown.py` / `transitions.py` / `belief.py`. |
| 2 — Census differential (E/O/C) | Complete | Emittable-vs-consumed signatures reconciled; unconsumed remainder adjudicated in the verdict table. |
| 3 — Encoding-collision audit | Complete | Public-collision driver ([#794]); no unexplained collision survived whitelist adjudication. |
| 4 — Counterfactual harm probes | Complete | Run on the surviving shortlist (confusion self-hit: opponent-move `damage_fraction` 0.270 vs true 0.170, reproduced). |
| Silent engine-mutation lane | Complete | Status/type mutation classes all COVERED or UNREACHABLE (below); 0 genuine silent status candidates. |

## Candidate verdicts (final)

### Landed ADDs (v3 series)

| Signature / fact | Column | PR |
| --- | --- | --- |
| `-fail` / `-miss` (separate non-hit flags) | `NUMERIC_TT_FAIL` 155 / `NUMERIC_TT_MISS` 110 | [#779] |
| Sleep-clause blocks self/opp | 157 / 158 | [#779] |
| Consecutive stall/Protect count | `NUMERIC_STALL_COUNTER` 159 | [#810] |
| Confusion turns-so-far | `NUMERIC_CONFUSION_TURNS` 160 | [#811] |
| Encore turns-so-far | `NUMERIC_ENCORE_TURNS` 161 | [#814] |
| Wrap partial-trap elapsed turns | `NUMERIC_WRAP_TRAP_TURNS` 162 | [#816] |
| Per-mon gender male/female | 163 / 164 | [#816] |
| Mean Look / Spider Web trap | `NUMERIC_MEANLOOK_TRAP` 165 | [#816] |
| Wish turns-to-land self/opp | 166 / 167 | [#820] |
| Confusion self-hit flag + v3 damage-attribution fix (v3-only; v2.2 frozen) | `NUMERIC_TT_CONFUSION_SELFHIT` 168 | [#826] |

v3 numeric width after #826: **169**.

### Cluster A — lost-turn / telegraph

| Signature | Verdict | Evidence |
| --- | --- | --- |
| `cant\|<reason>` (par/slp/frz/flinch/recharge/truant/attract/focuspunch/nopp) | COVERED | Each reason is a distinct non-OOV `cant:<reason>` categorical on CATEGORY_SECONDARY (`showdown.py:3883`); ids par=224/slp=227/frz=221/flinch=219/recharge=226/truant=229/attract=216/focuspunch=220/nopp=223. disable/imprison/taunt/damp are UNREACHABLE (no pool carriers) — enumerated-but-never-emitted, not conflations. |
| Yawn pending-sleep (`-start move: Yawn`) | COVERED | `volatile:yawn` in TRACKED_VOLATILES (`showdown.py:2254`), lands on the drowsy mon's active-mon token. |
| `-mustrecharge` telegraph | COVERED | The lost turn is `cant:recharge` (226); the one-turn "opponent locked" telegraph rides the deterministic immediately-preceding `move:hyperbeam` transition token. |
| Confusion self-hit lost-turn | **ADD → landed** | [#826] (see landed table). |
| `-notarget` | **UNREACHABLE** | gen3 **faint-ends-turn**: a mid-turn faint cancels the slower mon's queued action outright (verified in the real gen3 sim — slower Recover-into-Explosion does not heal, Body Slam fires no line, **no `-notarget`**). So a move never executes against an absent target in gen3 singles. The cancelled action is itself COVERED — encoded as a **NEGATED** turn-merged sub-block. |

### Cluster B — counters / misc protocol

| Signature | Verdict | Evidence |
| --- | --- | --- |
| Weather turns-remaining | COVERED | `NUMERIC_WEATHER_TURNS` 46 (move weather countdown) + `NUMERIC_WEATHER_PERMANENT` 47 (ability weather). |
| `-sethp` (Pain Split) | COVERED | Writes the public HP condition (`showdown.py:2133`); HP is always encoded. |
| Perish Song counters | COVERED | `perishN` in TRACKED_VOLATILES → `volatile:perishN` categorical tokens; each countdown value is a distinct present-set. |
| `-ohko` | **UNREACHABLE** | OHKO Clause in the gen3 `Standard` ruleset bans all OHKO moves; also 0 OHKO moves in the pool (double evidence). |
| Unconsumed `-activate` subtypes | COVERED | Full enumeration; consumed subtypes (protect/substitute/endure/pursuit/trapped/partial-trap) map to transition outcomes; dropped-but-covered (ability reveals, Destiny Bond, Trick, Heal Bell, Struggle, confusion) covered elsewhere; Bide/Magnitude/Grudge/Spite/Skill Swap/Mimic and all `item:` subtypes UNREACHABLE. |
| Destiny Bond attribution | COVERED | `destinybond` singlemove volatile — the "opponent is DB-active → mutual-KO risk" state is visible (carriers gengar/wobbuffet/qwilfish/banette). |
| BP-collapse (Baton-Pass turn-merge `fail` drop) | COVERED (was ACCEPTED-LOSS) | Re-adjudicated under the no-accepted-loss rule: already distinguished by the opponent Attack boost-stage column (Intimidate landed −1 vs blocked 0) + belief `revealed_ability` from the `[from] ability:` tag on the fail line. |

### Cluster C — silent engine-mutation

| Family | Verdict | Evidence |
| --- | --- | --- |
| 27 silent status transitions | COVERED / UNREACHABLE | Natural Cure switch-out cure (**protocol-backed in gen3 singles** — `-curestatus`; see correction below), Rest, toxic-counter reset, sleep-counter, Shed Skin, Yawn→sleep — all COVERED (belief-tracked/inferable). Guts is UNREACHABLE (boosts Attack, no status mutation — publicly inert). 0 genuine silent status candidates at HEAD. |
| 3 Kecleon Color Change type transitions | COVERED | Live type-slot encoding stamps the post-Color-Change type on the active token (`showdown.py:1395`, `3315`). |
| Ditto Transform token identity | COVERED | Encoder rewrites the token to the copied species/types/stats; the two prior divergences were an obsolete-oracle artifact, not a production gap. |
| Castform Forecast form identity | COVERED | Encoder intentionally keeps base `Castform` species + retyped active slots; the oracle demanding a form-species identity is obsolete. Not a schema item. |

## Engine-lane findings (poke-engine gen3 — OUTSIDE the observation schema)

Surfaced during the sweep; these are **search-engine fidelity gaps**, not
observation-schema items. Fixed in [#829] (a toolchain change — new patch +
differential + pin test + wheel rebuild — outside the observation freeze).

| Finding | Verdict | Evidence |
| --- | --- | --- |
| Rapid Spin clears the spinner's own hazards **through Protect** | FIXED [#829] | `remove_effects_for_protect()` (`choices.rs:20586`) strips effect fields but not `move_id`; the move-id-keyed `choice_hazard_clear` (`gen3/choice_effects.rs:343`) fired unconditionally from the hit loop (`gen3/generate_instructions.rs:2647`). Guarded on a `blocked_by_protect` signal — **not** on damage/`hit_sub`, so spin-into-Substitute still clears (verified: sub connects → hazards cleared). |
| Rapid Spin's Leech Seed + partial-trap clears **unimplemented** | FIXED [#829] | The RAPIDSPIN arm cleared only the four side conditions; a connecting gen3 Rapid Spin also frees the user's Leech Seed and partial-trap (sim-verified with a no-spin control). |
| Sibling sweep: `choice_special_effect` fired **through Protect** | FIXED [#829] | Move-id-keyed fixed/special-damage effects leaked: the old wheel dealt **Seismic Toss (80) / Super Fang (125)** damage through a Protect, plus Endeavor / Counter / Mirror Coat / Pain Split. Same `blocked_by_protect` guard. (Higher search impact than Rapid Spin — these are common gen3 lines.) |
| Sibling: `choice_after_damage_hit` (Knock Off / Thief / recharge) | No leak | Only reached under `if does_damage`, false for a gutted Protect-blocked move. |
| Sibling: Brick Break screen-break | UNREACHABLE | Not implemented in gen3 poke-engine; screens are absent from the gen3 randbats pool. |
| Toolchain: `setup_poke_engine.sh` was missing `struggle-typeless` | SYNCED [#829] | The Python wheel and the native crate were building different gen3 engines (wheel had Struggle as Normal). Both scripts now apply all four gen3 patches. |

## Plan-doc corrections (verified against the vendored gen3 engine)

- **Natural Cure** — the [plan](silent_noop_sweep_plan.md) frames NC as the
  canonical *silent* (`showCure=false`) example needing a belief-side switch-out
  clear. In gen3 **singles**, `onCheckShow` early-returns and NC emits a public
  `-curestatus` — it is protocol-backed, not silent; `showCure=false` is
  doubles-only. Verdict unchanged (COVERED); the premise is corrected.
- **`-notarget`** — listed as a seed candidate; re-verdicted **UNREACHABLE** in
  gen3 singles (faint-ends-turn cancels the mover before it can target an absent
  slot). See Cluster A.

*(These are the third and fourth catches of the "verify the current generation's
rule in the vendored engine" guardrail, after the Wish gen5-rule and the
substitute-vs-Protect nuance.)*

## Related records

- [silent_noop_sweep_findings.md](silent_noop_sweep_findings.md) -- the audit
  findings ledger, including the stricter Wish-inclusive **V6 cluster
  verification cycle** (optional, beyond the plan's local bar).
- [dead_observation_fields.md](dead_observation_fields.md) -- companion tracker
  for observation columns that encode gen3-randbats-**unreachable** mechanics
  (spare encoding; the inverse of this sweep's missing-encoding hunt).
- [silent_noop_sweep_plan.md](silent_noop_sweep_plan.md) -- the audit method.

## Completion Record

Every layer is complete at the local bar; every candidate has a final verdict;
`ACCEPTED-LOSS` appears nowhere. The one surviving observation `ADD` — confusion
self-hit — landed in [#826]. Engine-lane fidelity gaps are fixed in [#829], a
separate toolchain track outside the observation freeze.

**Schema-freeze recommendation: FREEZE `pokezero.observation.v3` at numeric
width 169** (post-#826). The Rust fold mirror and the golden-corpus regeneration
proceed against the frozen schema (spec §Coordination).

[#779]: https://github.com/sfitzgerald-x1/pokezero/pull/779
[#794]: https://github.com/sfitzgerald-x1/pokezero/pull/794
[#810]: https://github.com/sfitzgerald-x1/pokezero/pull/810
[#811]: https://github.com/sfitzgerald-x1/pokezero/pull/811
[#814]: https://github.com/sfitzgerald-x1/pokezero/pull/814
[#816]: https://github.com/sfitzgerald-x1/pokezero/pull/816
[#820]: https://github.com/sfitzgerald-x1/pokezero/pull/820
[#826]: https://github.com/sfitzgerald-x1/pokezero/pull/826
[#829]: https://github.com/sfitzgerald-x1/pokezero/pull/829
