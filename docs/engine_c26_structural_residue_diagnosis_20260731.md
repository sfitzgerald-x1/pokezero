# C26 structural residue: pre-launch diagnosis

## Result

**The registered C26 sweep would fail, and it should not be launched as-is.**

`structural_component_count_without_supported_sibling` is registered at
predicted-zero in `reports/c26_current_engine_resweep_spec.json`. On a fresh
60-game sample of the frozen 51-patch build it fires on **29 of 64 divergent
rows — 0.48 per game**, which projects to roughly **4,800 unattributed rows** in
the registered 10,000-game sweep. The certification standard requires zero.

Most of that residue is the rule firing wider than its own stated semantics. A
real residue survives underneath, and that surviving residue — not the count —
is the diagnosis queue.

## Evidence

| population | scoping | unattributed | structural |
| --- | --- | ---: | ---: |
| fresh 60 games (64 rows) | registered | 29 | 29 |
| fresh 60 games | majority-scoped | 7 | 7 |
| fresh 60 games | majority-scoped + roll-tolerant sibling | 4 | 4 |
| retained archive (3,496 surviving rows) | registered | 883 | 776 |
| retained archive | majority-scoped | 607 | 495 |
| retained archive | majority-scoped + roll-tolerant sibling | 451 | 339 |

Reproduce with `scripts/c26_structural_residue_diagnosis.py`; the emitted
`reports/c26_structural_residue_diagnosis.json` pins the classifier hash and the
probe report it read. The probe is a **diagnostic** on seed band 17,000,000
(filed as consumed), run *after* the C26 contract was registered, so the
contract's `fresh_measurements_inspected_before_registration = 0` still holds.
It is not certification evidence: the binding measurement remains the registered
10,000-game sweep on the C26 blocks at 16,000,000, which has not run.

## Three defects in the rule, in order of impact

### 1. It fires from minority branch arms

`attribute_row` scans **every** branch arm for a component-count mismatch:

```python
if cls == "roll_scaled_component":
    for miss in misses:
        miss_obs, miss_eng = _miss_pairs(miss)
        if len(miss_obs) != len(miss_eng):
```

**22 of the 29** fresh rows have the deciding mismatch on a minority arm only —
the majority arm (median mass 84.4%) has no count mismatch at all and complains
about something a documented family explains. Seed 17000001/1 is decided by a
6.25%-mass arm.

Every neighbouring rule is majority-scoped, and I4 states the reason in its own
comment: *"A tie confined to a minority arm cannot explain the majority-arm
complaint."* The structural rule is the only documented-family-preempting rule
outside the marked new-mechanism block that reasons from a minority arm.

### 2. It precedes the documented families it overlaps

The rule sits ahead of I1, I2 and I5. **26 of the 29** fresh rows carry a
`_to_full` cap shape in the majority miss — which is I1_cap_state_shape's
literal trigger (`if "_to_full" in majority`). Those rows never reach I1.

This inverts the ordering standard this repository already recorded once, in the
Z12.6 amendment to `docs/engine_divergence_ledger_20260728.md`: *"a narrow
mechanism shape must be tested BEFORE any rule wide enough to swallow it."* Here
a wide rule precedes narrow documented families.

**Do not treat this one as a free repair.** Moving an UNATTRIBUTED rule after
attributing rules mechanically reduces the unattributed count — that is exactly
the shape of widening attribution to pass a gate. Measured on the archive, full
reordering drives the structural counter to *zero* and inflates I1 from 50 to
508, which reads as over-correction rather than a fix. Defect 1 is the
defensible minimal change; the ordering question needs its own per-shape
argument.

### 3. The sibling-carry check is stricter than the matcher

```python
if Counter(sibling_engine_components) == observed_components:
    return True
```

`_sibling_arm_carries_observed_components` compares roll-scaled components by
**exact magnitude**, while the matcher that produced the miss accepts a
roll-scaled component if it is equal, in the enumerated legal roll set, or
within ±9%. So a sibling arm carrying the observed shape at a different legal
roll reads as absent. Seed 17000025/32:

```
pct=46.88: observed=[('', -28), ('heal_to_full', 28)]  engine=[('', -30)]
pct=46.88: observed=[('', -28), ('heal_to_full', 28)]  engine=[('', -30), ('heal_to_full', 30)]
```

The second arm carries the observed component labels exactly; only the roll
differs, and the matcher already treats that roll as interchangeable. Aligning
the two tolerances removes 3 more fresh rows and 156 archive rows.

## What survives

Four fresh rows (~0.07/game → ~**670 per 10,000 games**) and 339 archive rows
survive every scoping above. Their shape is consistent: a damage component and a
`heal_to_full` / `itemleftovers_to_full` component that cancel each other, where
the engine's arm carries one component and Showdown's slice carries two (or the
reverse) — seeds 17000013/12, 17000016/38, 17000026/41, and the two-component
split at 17000037/55.

That is the same decomposition question the rejected damage-composition
experiment probed (`reports/c26_damage_composition_tail_readout.json`,
`production_code_survives: false`). It is a genuine WHAT-level gap and is the
right next investigation.

## Consequences for the certification program

1. **C26 cannot pass**, and no classifier rescoping makes it pass: ~670
   unattributed rows per 10,000 games survive the most generous defensible
   scoping. Certification requires zero.
2. **The C26 registration stays as registered.** Its contract is immutable and
   its 16,000,000 seed blocks are unspent. Nothing here amends it.
3. **Any repair is a C27.** `scripts/cert_sweep_readout.py` is pinned by
   `required_readout_sha256` in the C26 contract and by the build-source
   lifecycle, so changing one line of the classifier invalidates the C26 source
   identity. The sequence is: fix, freeze a new source identity, re-register
   contract plus calibration, then sweep — exactly as the handoff requires for
   an engine patch.
4. **Register the prediction before measuring it.** Defects 1 and 3 are
   justified from the rule's own semantics and can be pre-registered on that
   basis. Defect 2 cannot, yet; it needs a per-shape argument that does not
   reduce to "it lowers the count."

## What was not done

The registered sweep was not launched. That is a deliberate hold, not an
omission: launching would spend the reserved blocks to obtain a FAIL this
document already predicts from a sample 167 times cheaper, and any fix requires
a new registration cycle regardless. Recording the binding FAIL remains
available if a formally attested negative result is wanted.
