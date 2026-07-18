# Checkpoint trait-tracking — findings

Implementation of `docs/checkpoint_trait_tracking_plan.md`. Behavioral traits measured per
checkpoint over cumulative-games milestones across the tracked lineages, from self-play (every
milestone) and foul-play (500k + a frontier per lineage). Every metric is derived from the
omniscient Showdown protocol log; the machinery, gates, and per-metric definitions live in
`scripts/trait_*.py`. Regenerate any time with `scripts/trait_extract_all.sh`. The rendered,
self-contained report is committed alongside this file at
[`checkpoint_trait_tracking_report.html`](checkpoint_trait_tracking_report.html) (a static
snapshot — open it directly in a browser; no server needed).

**Scope.** Three active lineages are tracked: **m50-ep7, l200-ep7-wu75, v22-lr3m**, plus
**v22-flat2m**, a *fork* of v22-lr3m (see below). The two `seq` lineages (m50-seq, l200-seq)
stalled at 1000k and are no longer tracked — they are excluded from the report and this doc
(`REPORT_EXCLUDE_LINEAGES` in `trait_report.py`; their metrics remain on disk, so it is reversible).

**Forked lineages.** A lineage's legs are *continuations* on one cumulative-games axis. A **fork**
is not a continuation: it branches from a shared ancestor and is its own entity from the fork point
on. `v22-flat2m` (run `emeta-v2-2-flat2m-belief`) forks from v22-lr3m at **2,000,000 games** — the
flat-LR twin against the `lr3m` schedule — and is tracked separately; it does not match the
`emeta-v2-2-lr3m-*` pattern, so the two never merge. A fork legitimately has no history below its
fork point, which required two fixes: the milestone grid now **skips** milestones no leg trained
through (it previously fell back to the nearest leg and clamped the iteration to 1, inventing ~20
pre-fork checkpoints), and **G0** only demands a sha-pinned 500k from lineages that actually span
500k. Both are covered by `tests/test_trait_inventory.py`.

> **v22-flat2m now has its first grid points** at 2100k and 2200k (it crossed the fork's first
> 100k boundary). Early read of the anneal-vs-flat A/B at the shared 2200k milestone: the flat-LR
> twin runs slightly longer games (47.3 vs 43.4 turns) and uses more Substitute (3.51 vs 2.91/sg)
> and stat-boost (3.59 vs 2.64/sg); Solar-Beam-in-sun and toxic are near-identical. Two points is
> too few to conclude — noted to watch as it extends.

Lineages are resolved from run-directory names by pattern (`trait_inventory.py`), which absorbs
continuation legs automatically; run names drift as new legs are added, so the inventory is re-run
each refresh and G0 is re-checked — it passes, and the tracked lineages resolve cleanly.

**Data.** 86 metric sets. Self-play at every 100k milestone per lineage (2000 games/milestone,
5000 at 500k) — **80 checkpoints**, following the active lineages to their current frontiers:
v22-lr3m 100k→2900k (29 pts), m50-ep7 →2700k (27), l200-ep7-wu75 →2200k (22), and v22-flat2m
(2100k–2200k, the fork). Foul-play (~950–1000 games, FoulPlay search at 1000 ms/move) at 500k and a frontier
per lineage — 6 checkpoints. **Foul-play was not re-run for the latest refreshes, so its
checkpoints trail the self-play frontiers badly** (m50-ep7 foul-play is @1000k while self-play now
reaches 2700k); the foul-play panel and the trait↔win-rate correlations describe those specific
older checkpoints, not the current frontier. Self-play and foul-play are kept separate and never
merged. The observation unit is the behavioral-seat-game (self-play has two behavioral seats,
foul-play one), so rates are comparable across the two.

## What changes over training (self-play)

Each point is one checkpoint; no aggregation.

- **Conditional move use is *learned*, not innate.** Early checkpoints fire conditional moves
  blindly; later ones gate them on the condition that makes them good:
  - *Solar Beam in sun* — the sharpest signal in the dataset, and all three tracked lineages
    develop it. Start → frontier: m50-ep7 **25.4%→100.0%** (2700k), l200-ep7-wu75 **31.1%→98.8%**
    (2200k), v22-lr3m **23.5%→99.7%** (2900k). It converges and stays converged (v22-lr3m reads
    95.1% at 2600k, 99.7% at 2900k — that wobble is a small denominator, ~270–320 uses/checkpoint).
    *These figures were corrected — see "Measurement corrections" below; earlier drafts reported a
    much lower start (~14–20%) and even an impossible 100.4%.*
  - *Phazing when justified* (enemy boosted or behind a Substitute) rises off ~0% early, but
    **unlike Solar Beam it does not converge** — it stays volatile and non-monotonic. At the
    current frontiers: l200-ep7-wu75 51.8% (2200k), m50-ep7 48.4% (2700k), v22-lr3m 35.6% (2900k) —
    yet at earlier checkpoints l200-ep7-wu75 read 63.7% (1300k) then 31.7% (1900k), and v22-lr3m read
    67.6% (2700k). The ordering reshuffles refresh to refresh. Phazing is
    rare (only ~70–270 uses per checkpoint vs 175–669 for Solar Beam), but those swings are ~6×
    the binomial SE, so this is real fluctuation rather than small-n noise. **Do not quote a
    frontier value as "the" rate for a lineage** — read the trajectory. Only the early rise off
    ~0% is solid.
- **Early toxic-spam collapses.** v22-lr3m opens at 5.41 Toxic/seat-game at 100k and settles to
  ~2.3. Very early checkpoints lean on status as a crutch.
- **The setup/utility toolkit is picked up over training.** stat-boost, Substitute, Spikes, and
  healing (excl. Rest) all rise from near-zero at 100k and plateau.
- **The weakest checkpoints can't close games.** v22-lr3m@100k times out (stalls to the turn cap)
  in ~50% of its self-play games; its *decided* games average ~57 turns. Timeout rate falls to ~0
  by 300k. avg-turns is reported over decided games only, with timeout rate as its own trajectory —
  a checkpoint that cannot win is a distinct failure mode from one that wins slowly.

## Skilled-use traits (added this round)

A second batch of traits, all derived from the same omniscient log (no re-eval). Each pairs a raw
usage rate with a *conditional* that measures whether the move is used **well**:

- **Priority moves** (Quick Attack / Extreme Speed / Mach Punch). Three columns: usage, *used vs a
  faster opponent*, and *KO rate*. All three rise with training and tell a consistent story — the
  policy learns priority is a revenge/pick-off tool, not filler. Start → frontier: *vs-faster*
  m50-ep7 7.7%→14.4%, l200 4.7%→12.6%, v22-lr3m 4.2%→13.6%; *KO rate* climbs harder still (v22-lr3m
  18.8%→46.9%, l200 18.3%→36.6%). "Faster" is read from the omniscient turn order (who moves first
  in a priority-0 turn = who is effectively faster, including paralysis/weather), so it needs no
  speed stats.
- **Destiny Bond success** (a use that drags the attacker down). Sits ~9–20% and is noisy across
  refreshes; no clean training trend. Low volume — read the trajectory, not a single point.
- **Status-only moves.** An aggregate of the dedicated status moves (paralysis, sleep, burn via
  Will-O-Wisp, Toxic, Yawn); secondary statuses riding on attacks (Body Slam, etc.) are excluded.
  **Knock Off** is in the gen3-randbats pool and is tracked; **Thief is not** (0 carriers) and is
  deliberately skipped.
- **Intentional weather** (Sunny Day / Rain Dance) as use-rate *conditioned on carrying the move*.
  Rises with training (Sunny Day 0.08–0.48 → ~1.2–1.3 uses/seat-game when in the pool) — later
  checkpoints actually press their weather rather than sitting on it.
- **Setup payoff.** Reversal/Flail average BP (higher = fired at lower HP, the correct timing) and
  Belly Drum average opponent-KOs-after (the payoff of the setup). Both are low-volume in randbats;
  tracked for the trend, quoted with sample sizes in the report.
- **Intimidate activations / game**, among games where an intimidator is on the team. Settles at
  ~1.9–2.5 across lineages (re-pivoting the intimidator in to re-drop Attack). Computed over
  **decided games only** — see corrections below.
- **Absorb switch-in reads / game** (Volt Absorb / Water Absorb heal, Flash Fire boost) triggering
  on the switch-in turn — a hard read, distinct from the type-immunity switch-in (a full-HP absorb
  legitimately counts as both). Rises sharply, 0.06–0.09 → ~0.9–1.4, so later checkpoints
  increasingly switch absorbers into the move they wall.
- **Average toxic stage reached** on badly-poisoned active mons (peak counter before the mon
  switches out / cures / faints; the counter resets on switch, per gen3). Reads the "does it pivot
  toxiced mons out to preserve HP" question directly, and the lineages **diverge**: l200-ep7-wu75
  falls 3.15 → 2.34 over training (learns to switch them out), m50-ep7 stays flat ~2.9–3.1, and
  v22-lr3m *rises* 1.07 → 2.83 (its 100k mons faint so fast they never escalate). Not a single story
  — a genuinely lineage-dependent behavior.
- **Boom blocks** — enemy Explosion/Self-Destruct neutralized by Protect, an absorbing Substitute,
  or a Ghost/type immunity, over booms faced. Rare (~0–11%), reported with the boom-faced count.

**Ability-gated traits carry a gating caveat.** The captured events hold species + moves but not
abilities, so Intimidate and the absorb abilities gate "on the team" via *protocol-detected
presence* (the ability fired at least once). This is essentially exact for Intimidate (an
intimidator almost always triggers when it sees play) but slightly under-counts presence for an
absorber never hit by its type, nudging that rate up. `trait_eval` now captures per-mon abilities,
so a future re-eval gates exactly (`trait_extract` prefers moveset abilities when present); the
existing frontier uses the fallback.

## Strength vs FoulPlay improves with training

| lineage | 500k | frontier | |
|---|---|---|---|
| m50-ep7 | 0.198 | **0.353** @1000k | |
| v22-lr3m | 0.307 | **0.420** @1900k | best at frontier |
| l200-ep7-wu75 | 0.319 | 0.295 @800k | **not significant** (z=1.13) |

l200-ep7-wu75 is the one lineage that appears to get *worse*. It does not survive a test: a
two-proportion z-test on 1000 vs 978 games gives **z=1.13** (needs |z|>1.96 at α=0.05). Do not
report it as a regression — it is noise.

## Trait ↔ win correlation — read the per-game charts, not the aggregate

Two correlations are rendered, and they disagree. **The per-game one is the trustworthy one.**

### Per-game (n = games) — the real answer

For each game: x = the seat's count of the trait *in that game*, y = 1 if that seat won.
Point-biserial across games, computed *within* each checkpoint, then aggregated by reporting the
mean r and the min..max range across checkpoints. Self-play is a **paired design** — both seats
are the same policy in the same game, so comparing winner against loser holds policy strength and
game length fixed by construction (a game-level quantity has no within-game variance and correctly
falls out at r≈0). **80** self-play checkpoints / **334,034** decided seat-games; 6 foul-play /
5,864.

**Headline: per-game behavior barely predicts winning.** Every effect is |r| ≤ 0.11 — under ~1% of
outcome variance. Only these are sign-consistent across *every* checkpoint (consistency across
independent checkpoints is the evidence, not any single r), and the effects have been stable as the
self-play sample grew across successive refreshes:

| trait | self-play (80 ckpts) | vs FoulPlay (6 ckpts) |
|---|---|---|
| Substitute | **−0.112** (all 80) | **−0.081** (all 6) |
| healing (excl Rest) | **−0.068** (all 80) | **−0.060** (all 6) |
| immunity switch-in | +0.041 (not consistent) | **+0.058** (all 6) |
| phaze when justified | — | **−0.025** (all 6) |

Substitute and healing tracking *losses* is almost certainly **reverse causality** — you Substitute
and heal when you are behind — not evidence that they lose games. Read these as association within
game context. The clean version would be a per-decision propensity analysis, which is a much
bigger lift.

## Measurement corrections

Two bugs were found by chasing a hypothesis that the v22-lr3m Solar-Beam dip reflected *learned
weather counterplay* (the opponent removing sun). **That hypothesis was not supported** — sun is
overwritten only ~16–22 times per 2000 games (~1% of games), the rate is flat from 1500k→2700k, and
every no-sun Solar Beam had `weather=None` rather than rain/sand. But the investigation exposed two
real defects in how the rates were computed:

1. **Locked continuations were counted as decisions.** A no-sun Solar Beam charges
   (`|move|…|[still]` + `-prepare`) and is re-emitted the next turn as `[from] lockedmove` — a
   forced continuation the policy never chose. Counting that second line double-counted the move,
   and since *only no-sun beams ever charge*, the double-count landed entirely on the no-sun side
   and deflated the in-sun rate. Now skipped (`lockedmove` lines are not decisions). Impact is
   narrow: only 12 of 41,008 `|move|` lines in a shard, **all Solar Beam** (15.4% of its count).
2. **Conditional rates mixed gated and ungated counters.** `move_category_extras` counts every
   occurrence, but `move_categories[*].total_uses` is gated on the acting seat's *moveset carrying*
   the move. A move used but not carried (Metronome/Mimic) lifts the numerator and not the
   denominator — which produced a literally impossible **100.4%** "Solar Beam in sun". All
   conditional rates now divide an **ungated pair** (e.g. `sun/(sun+nosun)`,
   `justified/(justified+neutral)`, `bp_stat_or_sub/bp_switch`). A regression test asserts no rate
   can exceed 100%.

Both fixes raise the *early* Solar-Beam-in-sun baseline (early checkpoints fire many no-sun beams,
which were the double-counted ones), so the learning curve is real but less dramatic than first
reported: ~23–31%→97–99%, not ~14–20%→100%. Anything computed as a ratio in this report should be
checked for the gated/ungated trap before being trusted.

3. **Per-game ability counts were inflated by timeout stalls.** `intimidate_activations_per_game`
   read **38.8** at v22-lr3m@100k — off the scale of every other checkpoint (~1–2.5). The cause is
   the same weak-checkpoint pathology as the old `avg_turns=485`: 49.6% of that checkpoint's games
   time out at ~1000 turns, and a stalled game pivots an intimidator in and out endlessly (~32k
   activations across the sample). Per-game counts (Intimidate, absorb reads) are now computed over
   **decided games only**, exactly as `avg_turns` is. That drops 100k to 8.8 — still elevated,
   because even its *decided* games run ~57 turns and genuinely pivot a lot; `avg_turns` is reported
   alongside so this reads as game-length, not a strategy signal. Ratio metrics (priority/Destiny
   Bond/boom rates) are self-normalizing and were left over all games.

**A trap worth recording:** the first cut of this chart ranked `forced_switch` first at r≈−0.65,
sign-consistent everywhere. It is circular: a forced switch happens *because your mon fainted*, so
its count is essentially "mons lost" and correlating it with losing restates the outcome. It is now
excluded, with a test (`test_outcome_definitional_traits_excluded`) so it cannot creep back. Any
trait added to `PER_GAME_TRAITS` must be a *chosen* behavior.

### Aggregate (n = checkpoints) — kept, but weak

One point per checkpoint: the trait and win rate are both aggregated over that checkpoint's
foul-play games (same population). n=6 across a 0.198–0.420 win-rate spread (dropping the two seq
lineages removed 4 checkpoints). It is **confounded by overall checkpoint strength** — better
models do more of everything effective *and* win more — so it largely measures "this trait tracks
being a stronger model", not any contribution to winning.
Its instability is documented: earlier cuts of this same chart (n=5, then n=4 with m50-ep7 held
out, then switching the trait source from self-play to foul-play games) produced *sign flips* on
`focus punch success`, `sleeping mon out`, and `spikes`. Treat it as descriptive only; where it and
the per-game chart disagree, believe the per-game chart.

`CORR_EXCLUDE_LINEAGES` is now empty: m50-ep7 was held out while foul-play existed only at 500k
(one point at 0.198 against a 0.31–0.34 cluster = high leverage at n=5). With foul-play at each
frontier its two points sit on a continuum, so excluding it would only discard win-rate spread.

## Per-trait conditional definitions

- **explosion/self-destruct**: one combined "boom" category (not split by move).
- **focus punch success rate**: landed (not disrupted) / total bot attempts.
- **opp focus-punch disrupted**: opponent Focus Punches the bot broke / opponent attempts — vs
  FoulPlay the bot lands ~90% of its own Focus Punches but disrupts only ~12% of FoulPlay's.
- **BP w/ stat or sub**: Baton Passes that actually carry a stat boost or Substitute (checked on
  the outgoing mon at pass time), not every BP switch.
- **priority vs faster opp**: priority uses where the opponent's active mon was observed to move
  first in a prior priority-0 turn this matchup (effective speed, including paralysis/weather).
  **priority KO rate**: priority uses whose target faints immediately (a residual-poison faint after
  the hit does not count).
- **destiny bond success**: uses where the DB `-activate` fired (it dragged the attacker down) / uses.
- **status-only moves**: dedicated status moves only (paralysis, sleep, Will-O-Wisp burn, Toxic,
  Yawn); a secondary status on an attacking move is not a status *choice* and is excluded.
- **intimidate / game** and **absorb switch-in reads / game**: per **decided** game, among games
  where the ability is on the team (moveset abilities when captured, else protocol-detected presence).
  An absorb read on the switch-in turn also counts as an immunity switch-in (it is both).
- **avg toxic stage**: mean peak badly-poison counter per episode; an episode ends on switch-out
  (counter resets in gen3), cure, faint, or game end.
- **boom blocks**: enemy Explosion/Self-Destruct the bot neutralized (Protect that turn / an up
  Substitute absorbing it / a Ghost or type immunity) / enemy booms faced.
- **win rate** is only reported where there is a real opponent. In self-play the bot drives both
  seats, so p1's rate is ~0.5 by construction and is omitted.

## Caveats

- **Foul-play covers 500k + frontier, not every milestone.** Trajectories over training are
  self-play. Extending foul-play to every milestone is gated on compute (~40 s/game).
- **Rare conditionals are noisy at low volume.** rapid-spin-when-spikes-down and solar-beam-in-sun
  ride on small denominators at early checkpoints. Denominators are surfaced (`carrier_rate`,
  category totals).
- **A per-game memory climb** (~28 MB/game, framework-side, independent of env lifetime) caps
  shards at 500 games / 24 GiB; production is sharded accordingly. Documented in `trait_eval.py`.
- **gen3-randbats coverage.** Screen-conditioned traits are absent from the pool by construction
  and are not tracked.
