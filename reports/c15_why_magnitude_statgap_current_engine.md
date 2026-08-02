# C15 WHY adjudication: full magnitude and same-turn-stat population

## Scope

This is a replay-first historical adjudication plus a current-engine re-read of all 37 rows in the two registered C15 populations: 28 `CAND_unresolved_magnitude` rows and 9 `CAND_same_turn_stat_event_gap` rows. The original 16-row sample and its refutations are preserved; the exact 21-row complement was separately preregistered before any remaining repro was opened. Historical verdicts are not relabeled here. The current re-read is a 37-row diagnostic, not the full certification sweep or a certification result.

**Coverage: 37/37 (100%).**

## Current Engine Re-read

- Build: 51 patches, fingerprint `de56344dd9b40fc3f9a6b775fa75b35b04cfeb2842bd59c8b6fe0625877d854e`.
- Strict transition comparator tally: 4 diverged, 33 matched.
- The runner checks the installed Python and Rust consumers against the checked-out build before replay, and fails closed on an overlap with a separately owned historical repair lane.
- These outcomes are directional evidence for the fixed C15 rows only. The complete retained population remains the certification gate.

## Historical Prediction Score

- Rule-scored rows: 17/37.
- Confirmed: 5 (2201005/55, 2300040/84, 2300552/117, 2400451/56, 2600362/82)
- Partially supported: 7 (2000298/23, 2000431/32, 2000561/67, 2400156/29, 2401127/54, 2500576/7, 2601196/46)
- Refuted: 5 (2000261/31, 2100079/7, 2500120/60, 2600535/80, 2600657/49)
- Unscored resolution gap: 20 (2001162/120, 2100482/83, 2200369/75, 2300154/80, 2400140/9, 2400172/89, 2400342/78, 2401002/8, 2401237/14, 2500151/116, 2500297/96, 2501061/96, 2600510/111, 2600546/22, 2600546/25, 2600992/21, 2601033/129, 2601196/25, 2700218/151, 2700355/37).

- Preserved initial score: 5 confirmed, 7 partial, 4 refuted, 0 unscored.
- Preregistered remainder: 1/21 rule-scored (0 confirmed, 0 partial, 1 refuted); 20 unscored resolution gaps.

The frozen remainder rubric did not define a score for an alternative hypothesis that resolved to a confirmed defect with an exact locus: it allowed `partial` only while the row remained WHAT-level or lacked a locus, and `confirmed` only for the first-listed mechanism. The 20 fixed-point rows are therefore reported as unscored rather than retroactively widening the rubric.

The first sample's broad same-turn pre-state-stat hypothesis remains refuted. The remainder instead found one shared source mechanism across 20 rows: odd base powers modified by Torrent or Thick Fat are carried as `.5` floats in Rust, while Showdown's inherited `chainModify` rounds half-down before the damage formula.

## Historical Population Readout

- Confirmed engine defects: 22/37.
- Confirmed instrument defects: 2/37.
- Documented comparison limits: 2/37.
- Still WHAT-level: 11/37 (2000261/31, 2000298/23, 2000431/32, 2000561/67, 2100079/7, 2400156/29, 2401127/54, 2500120/60, 2500576/7, 2600657/49, 2601196/46).

## Per-Row Historical Verdicts

| Family | Row | Historical WHY status | Current strict re-read | Historical verdict |
| --- | --- | --- | --- | --- |
| CAND_same_turn_stat_event_gap | 2000261/31 | still_WHAT | matched | WHAT-level engine base-damage candidate |
| CAND_unresolved_magnitude | 2000298/23 | still_WHAT | matched | switch-choice matcher limitation |
| CAND_unresolved_magnitude | 2000431/32 | still_WHAT | matched | WHAT-level direct-damage candidate |
| CAND_unresolved_magnitude | 2000561/67 | still_WHAT | matched | switch-choice matcher limitation |
| CAND_same_turn_stat_event_gap | 2001162/120 | confirmed_engine_defect | matched | engine odd-base-power modifier rounding defect |
| CAND_same_turn_stat_event_gap | 2100079/7 | still_WHAT | matched | WHAT-level engine base-damage candidate |
| CAND_unresolved_magnitude | 2100482/83 | confirmed_engine_defect | matched | engine odd-base-power modifier rounding defect |
| CAND_unresolved_magnitude | 2200369/75 | confirmed_engine_defect | matched | engine odd-base-power modifier rounding defect |
| CAND_unresolved_magnitude | 2201005/55 | confirmed_engine_defect | matched | engine dynamic-HP timing defect |
| CAND_unresolved_magnitude | 2300040/84 | comparison_limit | diverged | roll-inherited capped residual |
| CAND_unresolved_magnitude | 2300154/80 | confirmed_engine_defect | matched | engine odd-base-power modifier rounding defect |
| CAND_same_turn_stat_event_gap | 2300552/117 | confirmed_instrument_defect | diverged | event-aware legal-set omission |
| CAND_same_turn_stat_event_gap | 2400140/9 | confirmed_engine_defect | matched | engine odd-base-power modifier rounding defect |
| CAND_unresolved_magnitude | 2400156/29 | still_WHAT | matched | WHAT-level direct-damage candidate |
| CAND_unresolved_magnitude | 2400172/89 | confirmed_engine_defect | matched | engine odd-base-power modifier rounding defect |
| CAND_same_turn_stat_event_gap | 2400342/78 | confirmed_engine_defect | matched | engine odd-base-power modifier rounding defect |
| CAND_unresolved_magnitude | 2400451/56 | confirmed_engine_defect | matched | engine Forecast weather-expiry timing defect |
| CAND_unresolved_magnitude | 2401002/8 | confirmed_engine_defect | matched | engine odd-base-power modifier rounding defect |
| CAND_unresolved_magnitude | 2401127/54 | still_WHAT | matched | WHAT-level dynamic type-effect candidate |
| CAND_unresolved_magnitude | 2401237/14 | confirmed_engine_defect | matched | engine odd-base-power modifier rounding defect |
| CAND_same_turn_stat_event_gap | 2500120/60 | still_WHAT | matched | misbucketed switch-in magnitude |
| CAND_unresolved_magnitude | 2500151/116 | confirmed_engine_defect | matched | engine odd-base-power modifier rounding defect |
| CAND_unresolved_magnitude | 2500297/96 | confirmed_engine_defect | matched | engine odd-base-power modifier rounding defect |
| CAND_unresolved_magnitude | 2500576/7 | still_WHAT | matched | WHAT-level direct-damage candidate |
| CAND_unresolved_magnitude | 2501061/96 | confirmed_engine_defect | matched | engine odd-base-power modifier rounding defect |
| CAND_unresolved_magnitude | 2600362/82 | confirmed_instrument_defect | diverged | legal-roll matcher accounting |
| CAND_unresolved_magnitude | 2600510/111 | confirmed_engine_defect | matched | engine odd-base-power modifier rounding defect |
| CAND_unresolved_magnitude | 2600535/80 | comparison_limit | diverged | documented Substitute-health comparison limit |
| CAND_same_turn_stat_event_gap | 2600546/22 | confirmed_engine_defect | matched | engine odd-base-power modifier rounding defect |
| CAND_unresolved_magnitude | 2600546/25 | confirmed_engine_defect | matched | engine odd-base-power modifier rounding defect |
| CAND_same_turn_stat_event_gap | 2600657/49 | still_WHAT | matched | misbucketed static magnitude |
| CAND_unresolved_magnitude | 2600992/21 | confirmed_engine_defect | matched | engine odd-base-power modifier rounding defect |
| CAND_unresolved_magnitude | 2601033/129 | confirmed_engine_defect | matched | engine odd-base-power modifier rounding defect |
| CAND_unresolved_magnitude | 2601196/25 | confirmed_engine_defect | matched | engine odd-base-power modifier rounding defect |
| CAND_unresolved_magnitude | 2601196/46 | still_WHAT | matched | WHAT-level direct-damage candidate |
| CAND_unresolved_magnitude | 2700218/151 | confirmed_engine_defect | matched | engine odd-base-power modifier rounding defect |
| CAND_unresolved_magnitude | 2700355/37 | confirmed_engine_defect | matched | engine odd-base-power modifier rounding defect |

## Generalization Boundary

- The odd-base-power finding is historical: it generalized only to the 20 replayed rows whose fixed-point control admitted the observation while the then-current engine rejected it. The current strict comparator records whether each retained transition now matches; it does not extend that conclusion beyond these identities.
- `CAND_same_turn_stat_event_gap` is not a mechanism: its rows include fixed-point base-power defects, an event-aware matcher omission, post-boost WHAT candidates, and noncausal/misbucketed events.
- `CAND_unresolved_magnitude` is also mixed: switch comparison limits, capped residuals, hidden Substitute HP, dynamic HP/weather timing, fixed-point base-power defects, matcher accounting, and 11 still-WHAT rows.
- `2600535/80` is a comparison limit. Public state omits remaining Substitute HP; the documented maxhp/4 materialization cannot reproduce both the observed sub break and drain heal.
- The runner derives and hashes the current historical repair-lane identities before replay. Any overlap at the same `(seed, step)` fails closed; the recorded C15 intersection is empty.

## Historical Follow-Ups

- The historical engine and timing leads are retained for provenance only. The current strict reread is the authoritative statement about these 37 identities on this build.
- The two instrument rows and two documented comparison-limit rows remain separately labeled when the strict comparator still diverges; this bounded read does not turn either into an engine patch.
- This report does not generalize its 37 identities to the full retained certification population. That claim remains reserved for the full re-sweep.

## Reproduction

```bash
PYTHONPATH=src:scripts .venv/bin/python scripts/c15_why_adjudication.py \
  --archive <retained-sweep-archive> \
  --prediction reports/c15_why_magnitude_statgap_predictions.json \
  --remainder-prediction reports/c15_why_magnitude_statgap_remainder_predictions.json \
  --out-json reports/c15_why_magnitude_statgap_current_engine.json \
  --out-md reports/c15_why_magnitude_statgap_current_engine.md
```

The JSON artifact retains every branch instruction, protocol event, legal-roll set, controlled probe, and per-row rationale.
