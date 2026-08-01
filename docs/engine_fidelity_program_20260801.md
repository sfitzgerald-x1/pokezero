# Engine fidelity program — charter

## Why this exists as a separate document

The certification program asks one question: *is every difference between the
engine and the reference simulator explained?* That is a good question and a
strict gate. It is not the same question as *is the engine right*, and the two
have now visibly come apart.

Three prior adjudications turned out to rest on the engine's branch support
being incomplete, all found the same way and all within one working session:

- `limit:roll_divergent_lethality` carried the comment *"This is a limit of the
  comparison, not an engine fault"*. C27 moved 285 archive rows out of it.
- The `perish-clock` forcing proof asserted a strict zero that held only because
  the crit arm was collapsed to its average roll.
- Row `1500251:56` was adjudicated `damage_calc` — reachable under *no* legal
  roll the engine prices — and is now reachable.

Every one was filed as *not an engine problem*. Every one was, at least partly,
an engine problem. A gate that tolerates documented divergence will not surface
that class of error, because the error is in the documentation.

## The two objectives, stated separately

| | certification | fidelity |
| --- | --- | --- |
| question | is every difference explained? | does the engine's support contain every reachable sim transition? |
| shape | binary gate | asymptotic KPI |
| measure | zero unattributed rows | in-support rate per 10k games |
| passes with | 1,000+ tolerated divergences | nothing tolerated by definition |

Both are kept. Neither is read as the other. A certification PASS must never be
reported as "the engine is faithful", and a fidelity improvement must never be
reported as progress toward the gate unless it actually moves unattributed rows.

## The fidelity KPI

**In-support rate per 10k games**, tracked per build, emitted by the readout so
it cannot drift from the measurement it describes.

Measured baseline, same 60 fresh games, seed band 17,000,000:

| build | in-support | out-of-support per 10k games |
| --- | ---: | ---: |
| 51-patch `776fa1e1` | 98.649% | 10,667 |
| 52-patch `2054cf3b` (C27) | 98.734% | 10,000 |

**Shipped.** `cert_sweep_readout.py` emits a `fidelity` block carrying both
framings and its era stamp:

```json
"fidelity": {
  "in_support_rate": 0.99,
  "out_of_support_per_10k_games": 1000,
  "era": {"engine_fingerprint": "...", "differential_sha256": "..."}
}
```

Pinned by `tests/test_cert_sweep_readout_contract.py::FidelityKpiTests`.

### The KPI is comparable only within a matcher era

Corrected after C28. This document first claimed that an instrument fix cannot
move the KPI. That holds only for **classifier** changes: the KPI is
`transitions_diverged / boundaries_measured`, and both are computed by the
**differential**. A differential-side instrument fix — which the capped-heal work
is — moves the KPI directly.

Unqualified, that would let a matcher relaxation read as a fidelity gain, which
is exactly the dashboard-versus-reality failure the bucket scheme exists to
prevent.

So every KPI reading is stamped with **both** the engine fingerprint and the
differential's source hash, and may only be compared against a reading sharing
the differential hash. A matcher change opens a new KPI era and requires
re-baselining the prior engine on the new matcher before any delta is claimed.

## The three buckets

Every registered family belongs in exactly one. The current table does not
distinguish them, which is precisely what let an engine gap sit under a
comparison-limit label.

| bucket | meaning | fixing it |
| --- | --- | --- |
| **engine gap** | the support lacks a transition the sim can reach | real fidelity gain; moves the KPI |
| **instrument artifact** | the comparison itself is wrong | shrinks reported divergence and **changes nothing real** |
| **comparison limit** | no same-transition partition can reach it | permanent; must carry current-era evidence |

The middle bucket is the dangerous one. An instrument fix looks identical to
progress on a dashboard while the engine is untouched, so **each one needs a
per-shape argument** — not an aggregate improvement, and never "it lowered the
count".

## Ranking: search impact, not row count

C27's justification was never 285 rows. It was mispriced Q at KO margins: the
tree asserted a kill at full crit mass where the true kill mass was
`6.25% x (kill rolls / 16)`, and it did so at exactly the tactical margins where
search earns its keep. A low-count family that distorts searched worlds can
outrank a high-count cosmetic one.

```
rank = rows per 10k  x  is-engine-gap  x  changes-tree-pricing-at-decision-margins
```

`limit:world_sample_drag_target` (271 rows, a determinization question) is the
obvious case to test this on: fewer rows than the cosmetics above it, but
determinization error propagates into every searched world built from that
state.

## Method, per family

The C27 sequence, because it is what made C27's claims checkable:

1. **Grep before designing.** C27's mechanism already existed; the patch was a
   repair to `compare_health_with_damage_multiples`, not a new partitioner. A
   second mechanism beside an existing one is how two writers come to disagree.
2. **Measure offline first**, from retained rows the fix cannot see, and pin the
   hashes measured against.
3. **Pre-register the prediction**, class-split, with the ceiling stated as a
   ceiling. If the patch scope grows, **re-register before the measurement run** —
   scope change requires re-registration, the same rule as
   prediction-before-measurement, one level up. C27 missed this and three of its
   four per-family ceilings were exceeded.
4. **Gate on the depth-tactics suite.** Its forcing proofs re-solve from scratch
   rather than asserting recorded numbers, which is exactly why they catch a
   branch-structure change.
5. **Report the delta honestly**, including a falsified prediction.

## Time-box, and the licence to refuse

An unbounded re-litigation of every family is a crusade, not a program.

- Each family gets a stated budget before work starts.
- Anything that resists inside that budget lands as **candidate-not-finding**,
  with what was tried and what remains unknown. That is a complete outcome, not
  a failure.
- Only the verdict that *reduces* the residue has to clear a bar. The asymmetry
  is deliberate and inherited from Appendix X.

## Era discipline for adjudications

Every limit adjudication is stamped with the engine fingerprint it was derived
on, and is **mechanically re-derived on every fingerprint change**. The walk-mass
re-derivation at C27 is the template: it was cheap, it was registered in advance,
and it caught a flip that would otherwise have gone unnoticed.

**Overturned prior adjudications are findings, not cleanups.** They are recorded
as results of the change that overturned them, with the reasoning that failed.

## Sequencing

1. **Capped-heal matcher fix — critical path.** Corrected after C28: this is
   4 rows per 60 games, not 29. The rule's minority-arm scoping was the larger
   share (22 rows) and shipped as C28; its exact-magnitude sibling-carry check
   turned out not to be a defect at all and was withdrawn. The remaining
   mechanism is real: `_split_component_events` routes a component into the
   roll-scaled bucket iff its source ends with `_to_full`, so bucket membership —
   and therefore list *length* — is roll-dependent, and
   `roll_component_events_agree` rejects on length before applying any tolerance.
2. **Bucketing audit — alongside**, not instead of.
3. **One successor contract** carrying the capped-heal fix, the KPI, and C27,
   rather than a registration cycle each.
4. **Engine-gap burndown** in rank order.

## Stop conditions

- Certification PASSES: zero unattributed, every shape closed as a fix, a
  documented follow-up, or an evidence-backed limit.
- The audit table is complete, with evidence per family.
- The engine-gap bucket is empty, or every remaining family carries a
  current-era adjudication.

## Open follow-ups already registered

- The `==` orphan needs an explicit pin.
- Search-side C27 cost is unmeasured; the differential's 2.88x is close to a
  worst case because the binding branches unconditionally while the search gates
  on `DAMAGE_BRANCH_DEPTH` and `deep_ko_straddle`. The FoulPlay-power preflight
  timing math assumes pre-C27 throughput.
- The f32 loop in `compare_health_with_damage_multiples` disagrees with the
  differential's integer `base * roll // 100` on 20 of 7,484 pairs. Its own
  registered change: it moves emitted damage on every case-A hit.
- C27 is an era boundary for any eval campaign in flight.
