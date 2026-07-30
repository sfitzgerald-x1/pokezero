# C15 WHY adjudication: full magnitude and same-turn-stat population

## Scope

This is a replay-first adjudication of all 37 rows in the two registered C15 populations: 28 `CAND_unresolved_magnitude` rows and 9 `CAND_same_turn_stat_event_gap` rows. The original 16-row sample and its refutations are preserved; the exact 21-row complement was separately preregistered before any remaining repro was opened. This artifact does not relabel the certification sweep or modify the living ledger.

**Coverage: 37/37 (100%).**

## Prediction Score

- Rule-scored rows: 17/37.
- Confirmed: 5 (2201005/55, 2300040/84, 2300552/117, 2400451/56, 2600362/82)
- Partially supported: 7 (2000298/23, 2000431/32, 2000561/67, 2400156/29, 2401127/54, 2500576/7, 2601196/46)
- Refuted: 5 (2000261/31, 2100079/7, 2500120/60, 2600535/80, 2600657/49)
- Unscored resolution gap: 20 (2001162/120, 2100482/83, 2200369/75, 2300154/80, 2400140/9, 2400172/89, 2400342/78, 2401002/8, 2401237/14, 2500151/116, 2500297/96, 2501061/96, 2600510/111, 2600546/22, 2600546/25, 2600992/21, 2601033/129, 2601196/25, 2700218/151, 2700355/37).

- Preserved initial score: 5 confirmed, 7 partial, 4 refuted, 0 unscored.
- Preregistered remainder: 1/21 rule-scored (0 confirmed, 0 partial, 1 refuted); 20 unscored resolution gaps.

The frozen remainder rubric did not define a score for an alternative hypothesis that resolved to a confirmed defect with an exact locus: it allowed `partial` only while the row remained WHAT-level or lacked a locus, and `confirmed` only for the first-listed mechanism. The 20 fixed-point rows are therefore reported as unscored rather than retroactively widening the rubric.

The first sample's broad same-turn pre-state-stat hypothesis remains refuted. The remainder instead found one shared source mechanism across 20 rows: odd base powers modified by Torrent or Thick Fat are carried as `.5` floats in Rust, while Showdown's inherited `chainModify` rounds half-down before the damage formula.

## Population Readout

- Confirmed engine defects: 22/37.
- Confirmed instrument defects: 2/37.
- Documented comparison limits: 2/37.
- Still WHAT-level: 11/37 (2000261/31, 2000298/23, 2000431/32, 2000561/67, 2100079/7, 2400156/29, 2401127/54, 2500120/60, 2500576/7, 2600657/49, 2601196/46).

## Per-Row Verdicts

| Family | Row | WHY status | Verdict | Lane | Prediction |
| --- | --- | --- | --- | --- | --- |
| CAND_same_turn_stat_event_gap | 2000261/31 | still_WHAT | WHAT-level engine base-damage candidate | engine candidate; no patch locus licensed | refuted H-D2 |
| CAND_unresolved_magnitude | 2000298/23 | still_WHAT | switch-choice matcher limitation | instrument; underlying magnitude remains WHAT-level | partial H-A |
| CAND_unresolved_magnitude | 2000431/32 | still_WHAT | WHAT-level direct-damage candidate | engine candidate; no patch locus licensed | partial H-B |
| CAND_unresolved_magnitude | 2000561/67 | still_WHAT | switch-choice matcher limitation | instrument; underlying magnitude remains WHAT-level | partial H-A |
| CAND_same_turn_stat_event_gap | 2001162/120 | confirmed_engine_defect | engine odd-base-power modifier rounding defect | engine; Torrent fixed-point base-power follow-up | unscored resolution-gap S3 |
| CAND_same_turn_stat_event_gap | 2100079/7 | still_WHAT | WHAT-level engine base-damage candidate | engine candidate; no patch locus licensed | refuted H-D2 |
| CAND_unresolved_magnitude | 2100482/83 | confirmed_engine_defect | engine odd-base-power modifier rounding defect | engine; Torrent fixed-point base-power follow-up | unscored resolution-gap M5 |
| CAND_unresolved_magnitude | 2200369/75 | confirmed_engine_defect | engine odd-base-power modifier rounding defect | engine; Thick Fat fixed-point base-power follow-up | unscored resolution-gap M5 |
| CAND_unresolved_magnitude | 2201005/55 | confirmed_engine_defect | engine dynamic-HP timing defect | engine | confirmed H-B |
| CAND_unresolved_magnitude | 2300040/84 | comparison_limit | roll-inherited capped residual | instrument / documented comparison limit | confirmed H-C |
| CAND_unresolved_magnitude | 2300154/80 | confirmed_engine_defect | engine odd-base-power modifier rounding defect | engine; Thick Fat fixed-point base-power follow-up | unscored resolution-gap M5 |
| CAND_same_turn_stat_event_gap | 2300552/117 | confirmed_instrument_defect | event-aware legal-set omission | instrument | confirmed H-D1 |
| CAND_same_turn_stat_event_gap | 2400140/9 | confirmed_engine_defect | engine odd-base-power modifier rounding defect | engine; Thick Fat fixed-point base-power follow-up | unscored resolution-gap S3 |
| CAND_unresolved_magnitude | 2400156/29 | still_WHAT | WHAT-level direct-damage candidate | engine candidate; no patch locus licensed | partial H-B |
| CAND_unresolved_magnitude | 2400172/89 | confirmed_engine_defect | engine odd-base-power modifier rounding defect | engine; Thick Fat fixed-point base-power follow-up | unscored resolution-gap M5 |
| CAND_same_turn_stat_event_gap | 2400342/78 | confirmed_engine_defect | engine odd-base-power modifier rounding defect | engine; Thick Fat fixed-point base-power follow-up | unscored resolution-gap S3 |
| CAND_unresolved_magnitude | 2400451/56 | confirmed_engine_defect | engine Forecast weather-expiry timing defect | engine | confirmed H-B |
| CAND_unresolved_magnitude | 2401002/8 | confirmed_engine_defect | engine odd-base-power modifier rounding defect | engine; Thick Fat fixed-point base-power follow-up | unscored resolution-gap M5 |
| CAND_unresolved_magnitude | 2401127/54 | still_WHAT | WHAT-level dynamic type-effect candidate | engine candidate; no patch locus licensed | partial H-B |
| CAND_unresolved_magnitude | 2401237/14 | confirmed_engine_defect | engine odd-base-power modifier rounding defect | engine; Thick Fat fixed-point base-power follow-up | unscored resolution-gap M5 |
| CAND_same_turn_stat_event_gap | 2500120/60 | still_WHAT | misbucketed switch-in magnitude | instrument classification; underlying magnitude remains WHAT-level | refuted H-D2 |
| CAND_unresolved_magnitude | 2500151/116 | confirmed_engine_defect | engine odd-base-power modifier rounding defect | engine; Thick Fat fixed-point base-power follow-up | unscored resolution-gap M5 |
| CAND_unresolved_magnitude | 2500297/96 | confirmed_engine_defect | engine odd-base-power modifier rounding defect | engine; Thick Fat fixed-point base-power follow-up | unscored resolution-gap M5 |
| CAND_unresolved_magnitude | 2500576/7 | still_WHAT | WHAT-level direct-damage candidate | engine candidate; no patch locus licensed | partial H-B |
| CAND_unresolved_magnitude | 2501061/96 | confirmed_engine_defect | engine odd-base-power modifier rounding defect | engine; Thick Fat fixed-point base-power follow-up | unscored resolution-gap M5 |
| CAND_unresolved_magnitude | 2600362/82 | confirmed_instrument_defect | legal-roll matcher accounting | instrument | confirmed H-A |
| CAND_unresolved_magnitude | 2600510/111 | confirmed_engine_defect | engine odd-base-power modifier rounding defect | engine; Thick Fat fixed-point base-power follow-up | unscored resolution-gap M5 |
| CAND_unresolved_magnitude | 2600535/80 | comparison_limit | documented Substitute-health comparison limit | world/comparison limit; no engine patch | refuted M2/M4/M5 |
| CAND_same_turn_stat_event_gap | 2600546/22 | confirmed_engine_defect | engine odd-base-power modifier rounding defect | engine; Thick Fat fixed-point base-power follow-up | unscored resolution-gap S3 |
| CAND_unresolved_magnitude | 2600546/25 | confirmed_engine_defect | engine odd-base-power modifier rounding defect | engine; Thick Fat fixed-point base-power follow-up | unscored resolution-gap M5 |
| CAND_same_turn_stat_event_gap | 2600657/49 | still_WHAT | misbucketed static magnitude | engine candidate; no patch locus licensed | refuted H-D2 |
| CAND_unresolved_magnitude | 2600992/21 | confirmed_engine_defect | engine odd-base-power modifier rounding defect | engine; Thick Fat fixed-point base-power follow-up | unscored resolution-gap M5 |
| CAND_unresolved_magnitude | 2601033/129 | confirmed_engine_defect | engine odd-base-power modifier rounding defect | engine; Thick Fat fixed-point base-power follow-up | unscored resolution-gap M5 |
| CAND_unresolved_magnitude | 2601196/25 | confirmed_engine_defect | engine odd-base-power modifier rounding defect | engine; Thick Fat fixed-point base-power follow-up | unscored resolution-gap M5 |
| CAND_unresolved_magnitude | 2601196/46 | still_WHAT | WHAT-level direct-damage candidate | engine candidate; no patch locus licensed | partial H-B |
| CAND_unresolved_magnitude | 2700218/151 | confirmed_engine_defect | engine odd-base-power modifier rounding defect | engine; Thick Fat fixed-point base-power follow-up | unscored resolution-gap M5 |
| CAND_unresolved_magnitude | 2700355/37 | confirmed_engine_defect | engine odd-base-power modifier rounding defect | engine; Thick Fat fixed-point base-power follow-up | unscored resolution-gap M5 |

## Generalization Boundary

- The odd-base-power finding generalizes only to the 20 replayed rows whose exact fixed-point control admits the observation while the current engine set rejects it. It does not absorb the other 17 rows.
- `CAND_same_turn_stat_event_gap` is not a mechanism: its rows include fixed-point base-power defects, an event-aware matcher omission, post-boost WHAT candidates, and noncausal/misbucketed events.
- `CAND_unresolved_magnitude` is also mixed: switch comparison limits, capped residuals, hidden Substitute HP, dynamic HP/weather timing, fixed-point base-power defects, matcher accounting, and 11 still-WHAT rows.
- `2600535/80` is a comparison limit. Public state omits remaining Substitute HP; the documented maxhp/4 materialization cannot reproduce both the observed sub break and drain heal.
- No row overlaps patches 42-44 or active world-lane rows at the same `(seed, step)`; the shared seed 2000431 is explicitly recorded as a different step.

## Banked Follow-Ups

- Engine fixed-point: replace Torrent's floating `*= 1.5` in `abilities.rs::ability_modify_attack_being_used` and Thick Fat's `/= 2.0` in `abilities.rs::ability_modify_attack_against`; audit sibling Blaze, Overgrow, and Swarm arms.
- Engine timing: inspect `generate_instructions.rs::before_move -> choice_effects::modify_choice` for Flail/Reversal BP after earlier same-turn damage (`2201005/55`), and `abilities.rs::update_forecast` plus the weather-expiry call ordering for `2400451/56`.
- Instrument: in `engine_transition_differential.py::evaluate_boundary_strict -> roll_components_agree`, derive event-aware legal rolls after same-turn stat changes (`2300552/117`) and post-switch branch legality (`2600362/82`).
- Comparison limits: keep capped residual handling in `roll_components_agree` (`2300040/84`) and `_build_side_spec`'s `substitute_health = maxhp // 4` approximation (`2600535/80`) explicit; do not turn hidden Substitute HP into a deterministic engine patch.
- Remaining WHY: carry the 11 exact unresolved identities above into a focused source lane rather than inferring from family or ratio.

## Reproduction

```bash
PYTHONPATH=src:scripts .venv/bin/python scripts/c15_why_adjudication.py \
  --archive <retained-sweep-archive> \
  --prediction reports/c15_why_magnitude_statgap_predictions.json \
  --remainder-prediction reports/c15_why_magnitude_statgap_remainder_predictions.json \
  --out-json reports/c15_why_magnitude_statgap_results.json \
  --out-md reports/c15_why_magnitude_statgap_report.md
```

The JSON artifact retains every branch instruction, protocol event, legal-roll set, controlled probe, and per-row rationale.
