# investment gate — `required_pin_strikes` 1 vs 2 (2026-08-02)

Paired run of `scripts/investment_gate.py` deciding the default corroboration
requirement for the defender-side investment inference
(`InvestmentConfig.required_pin_strikes`). Both arms are the **same replays**: same
harness, same policy (move-biased random-legal, `move_bias=0.75`, both seats), **seed 11**,
**120 games**, randbats **source hash `f9e35e1f`**. Only `--required-pin-strikes` differs,
so every difference below is attributable to it.

- `k1/summary.json` — `required_pin_strikes=1` (the new default)
- `k2/summary.json` — `required_pin_strikes=2` (the previous default)

## Result

| | k=1 | k=2 |
|---|---|---|
| `hp_value` | 54 preds, 54 TP, 0 FP, precision 1.000 | 10, 10, 0, 1.000 |
| `hp_class` | 52, 52, 0, 1.000 | 10, 10, 0, 1.000 |
| `defense` | 40, 40, 0, 1.000 | 6, 6, 0, 1.000 |
| blocked mons (hp / defense) | 0 / 0 | 0 / 0 |
| HP conclusion rate on mixed-family mons | 31.4% | 5.8% |

Both arms PASS the gate on precision and calibration; the strike-level ledger is identical
(5435 assessed, 3425 clean, 64 HP pins, 46 defense pins) because `required_pin_strikes` only
governs when an axis FREEZES, never how a strike is assessed.

## Why the default moved to 1

The lattice test is **deductive**, not evidential. Our attacker stats are exactly known, so
each candidate defender variant admits exactly 16 legal per-hit damage values; an observed
damage that misses all of them by more than `rejection_margin_hp` makes that variant
*impossible*. Corroboration cannot make an exclusion more sound. A second strike buys only a
window in which a later contradicting strike can still block the axis — and the axis ledger
already blocks on contradiction whenever it arrives before the freeze. Empirically that
window blocked nothing (0 blocked mons under both settings) while costing 5.4x HP and 6.7x
defense coverage.

## Relation to `runs/investment-gate-2026-07-04/`

That archived run is also seed 11 / 120 games at `required_pin_strikes=2`, but it was
produced from a **different checkout** against randbats source hash `e648ed6a`, not
`f9e35e1f`. Its k=2 numbers therefore differ slightly (13/13 `hp_value`, 9/9 `defense`, and a
different `margin-ambiguity` count) and are **not** the right comparison point for this
decision. The k=2 arm here was re-run on the current source so both arms of the comparison
share a source hash.
