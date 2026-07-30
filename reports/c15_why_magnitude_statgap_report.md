# C15 WHY adjudication: magnitude and same-turn-stat samples

## Scope

This is a replay-first adjudication of the 16 fixed samples registered before measurement. It does not relabel the certification sweep, modify the ledger, or claim a family-level WHY where samples disagree.

## Prediction Score

- Confirmed: 5/16 (2201005/55, 2300040/84, 2300552/117, 2400451/56, 2600362/82)
- Partially supported: 7/16 (2000298/23, 2000431/32, 2000561/67, 2400156/29, 2401127/54, 2500576/7, 2601196/46)
- Refuted: 4/16 (2000261/31, 2100079/7, 2500120/60, 2600657/49)

The main correction is negative: the engine does apply a same-turn Calm Mind before the opposing hit. The broad pre-state-stat hypothesis is therefore refuted, not promoted to an engine patch.

## Per-Row Verdicts

| Family | Row | Verdict | Lane | Prediction |
| --- | --- | --- | --- | --- |
| CAND_same_turn_stat_event_gap | 2000261/31 | WHAT-level engine base-damage candidate | engine candidate; no patch locus licensed | refuted H-D2 |
| CAND_unresolved_magnitude | 2000298/23 | switch-choice matcher limitation | instrument; underlying magnitude remains WHAT-level | partial H-A |
| CAND_unresolved_magnitude | 2000431/32 | WHAT-level direct-damage candidate | engine candidate; no patch locus licensed | partial H-B |
| CAND_unresolved_magnitude | 2000561/67 | switch-choice matcher limitation | instrument; underlying magnitude remains WHAT-level | partial H-A |
| CAND_same_turn_stat_event_gap | 2100079/7 | WHAT-level engine base-damage candidate | engine candidate; no patch locus licensed | refuted H-D2 |
| CAND_unresolved_magnitude | 2201005/55 | engine dynamic-HP timing defect | engine | confirmed H-B |
| CAND_unresolved_magnitude | 2300040/84 | roll-inherited capped residual | instrument / documented comparison limit | confirmed H-C |
| CAND_same_turn_stat_event_gap | 2300552/117 | event-aware legal-set omission | instrument | confirmed H-D1 |
| CAND_unresolved_magnitude | 2400156/29 | WHAT-level direct-damage candidate | engine candidate; no patch locus licensed | partial H-B |
| CAND_unresolved_magnitude | 2400451/56 | engine Forecast weather-expiry timing defect | engine | confirmed H-B |
| CAND_unresolved_magnitude | 2401127/54 | WHAT-level dynamic type-effect candidate | engine candidate; no patch locus licensed | partial H-B |
| CAND_same_turn_stat_event_gap | 2500120/60 | misbucketed switch-in magnitude | instrument classification; underlying magnitude remains WHAT-level | refuted H-D2 |
| CAND_unresolved_magnitude | 2500576/7 | WHAT-level direct-damage candidate | engine candidate; no patch locus licensed | partial H-B |
| CAND_unresolved_magnitude | 2600362/82 | legal-roll matcher accounting | instrument | confirmed H-A |
| CAND_same_turn_stat_event_gap | 2600657/49 | misbucketed static magnitude | engine candidate; no patch locus licensed | refuted H-D2 |
| CAND_unresolved_magnitude | 2601196/46 | WHAT-level direct-damage candidate | engine candidate; no patch locus licensed | partial H-B |

## Generalization Boundary

- `CAND_same_turn_stat_event_gap` is not one mechanism: one sample is an event-aware legal-set omission, two are post-boost one-point candidates, and two are misbucketed non-stat rows.
- `CAND_unresolved_magnitude` is not one mechanism: the samples include switch-choice matcher misuse, a roll-inherited capped residual, dynamic Flail HP timing, Forecast timing, and static magnitude candidates.
- Only the row-level mechanisms above are adjudicated. The remaining unsampled rows stay WHAT-level until replay establishes a shared WHY.
- No sampled row overlaps patches 42-44 or the active world-lane rows at the same `(seed, step)`; the one shared game is explicitly recorded in the JSON artifact.

## Reproduction

```bash
PYTHONPATH=src:scripts .venv/bin/python scripts/c15_why_adjudication.py \
  --archive <retained-sweep-archive> \
  --prediction reports/c15_why_magnitude_statgap_predictions.json \
  --out-json reports/c15_why_magnitude_statgap_results.json \
  --out-md reports/c15_why_magnitude_statgap_report.md
```

The JSON artifact retains every branch instruction, protocol event, legal-roll set, controlled probe, and per-row rationale.
