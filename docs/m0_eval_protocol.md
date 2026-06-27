# M0 evaluation protocol — note to the in-flight root-PUCT work

_2026-06-26._ The `root-puct-*` probes are the right direction and are effectively **M0** of
[`selfplay_mcts_roadmap.md`](selfplay_mcts_roadmap.md): *does test-time search lift the net past the
~0.52 vs-max-damage plateau?* The harness looks good — three changes make the result conclusive
instead of noise.

## 1. Evaluate at ≥300–400 games, not 8–16
The current probes run 8–16 games vs max-damage, where sampling variance is ±15–35%: net-alone
reads anywhere from 0.12 to 0.38 across probes purely from noise, and +PUCT swings 0.25→0.625 the
same way. None of that is signal. **The M0 gate (net+MCTS clears ~0.60 vs max-damage) is only
meaningful at ≥300 games.** Treat 8-game runs as wiring checks, not strength evidence.

## 2. Fix the value head first — it gates search
The `value-calibration` probe reports **sign accuracy ≈ 0.72, ECE ≈ 0.19, MAE ≈ 0.81** — a mediocre
value head. MCTS leaf evaluation is bounded by value quality, so a weak value head is the most
likely reason PUCT helps inconsistently (e.g. leaf2 nudged 0.19→0.31 but selection/value-gate
probes showed no lift). Improve value calibration (roadmap WS-E) and re-measure before concluding
anything about whether search helps.

## 3. Use a base net worth searching from
`teacher-bc-64` is a tiny imitation net sitting near the imitation ceiling. M0 should run on a
*decent* net (a real self-play/PPO checkpoint), so we're testing search on something worth
searching from — not amplifying a weak prior.

## M0 go/no-go
On a decent net + a calibrated value head, at **≥300 games**: net+MCTS must clear **~0.60 vs
max-damage** to justify scaling (then proceed to WS-B fleet scaling). If it does not, fix the
operator — search depth, value head, or DUCT for the simultaneous-move handling — **before**
spending fleet compute. Search is the load-bearing bet; prove it cheaply but rigorously first.
