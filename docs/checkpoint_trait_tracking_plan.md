# Checkpoint trait tracking: behavioral traits over training time

Status: plan for review, 2026-07-13. Agent-executable; every stage names its inputs,
outputs, validation gates, and metric definitions before any compute is spent.

## The question

Strength curves (max-damage / foul-play win rates) tell us *how much* checkpoints
improve but nothing about *what they learn to do*. This plan builds a longitudinal
behavioral record — move-usage profiles, tempo, switching discipline, resource
management — across five lineages at fixed game-count milestones, so that recipe
effects (ep7 vs ep5, hot vs annealed LR, 50M vs 200M vs 10M) become visible as
behavioral divergence, not just win-rate deltas. It complements the
strategy-diversity fingerprint plan (which asks whether equal-strength checkpoints
differ *from each other* on fixed corpora); here we ask how each lineage's behavior
*moves through time*, and which behaviors correlate with winning.

## Lineages under study

Disjoint runs from the same lineage are merged and treated as one entity on a
single cumulative-games axis (continuations resume with `--completed-games`, so
game counts are already lineage-global).

| key | run-id pattern | known legs (ordered) | span | notes |
|---|---|---|---|---|
| `m50-ep7` | `metamon-m50-.*-lr10m-ep7` | `metamon-m50-2m-lr10m-ep7` | 0→2M (running) | 50M, ep7, 10M-LR runway |
| `l200-ep7-wu75` | `metamon-l200-.*-lr10m-ep7-wu75` | `…-500k-lr10m-ep7-wu75`, `…-2m-lr10m-ep7-wu75` | 0→2M (running) | 200M, ep7, 75k warmup |
| `v22-lr3m` | `emeta-v2-2-lr3m-.*` | `…-500k-belief`, `…-1m-belief`, `…-2m-belief` | 0→2M (running) | 10M small, hot 3M schedule |
| `m50-seq` | `metamon-m-50m-.*-seq-20260710` | `…-500k-seq-…`, `…-1m-seq-…` | 0→1M (complete) | 50M, ep5, 3M schedule |
| `l200-seq` | `metamon-l-200m-.*-seq-20260710` | `…-500k-seq-…`, `…-1m-seq-…` | 0→1M (complete) | 200M, ep5, spiral-then-recovery |

Phase 0 resolves the patterns against shared experiment storage authoritatively
(naming eras vary); the table above is the expected resolution. Lineages still
training grow new milestones; the pipeline appends rather than recomputes.

## Conventions

- **Milestone grid.** Every 100k games from 100k to the lineage frontier. A
  milestone G maps to the leg covering G and iteration `ceil((G − leg_offset) /
  games_per_iteration)`; actual games are recorded next to the nominal label
  (skew ≤ one iteration, ~1,600 games). Games-per-iteration is read from each
  leg's run summary, never assumed.
- **Checkpoint identity.** Every evaluated checkpoint is pinned as (run id,
  iteration, sha256 of `transformer-policy.pt`). Checkpoints cannot be
  regenerated: if a milestone's iteration directory was pruned, substitute the
  nearest retained iteration and label the substitution; never silently shift.
- **Self-play vs FoulPlay stats are never merged.** Every metric exists twice.
  In self-play both seats are the same policy; per-seat statistics are pooled
  (doubling n) with game-level clustering respected in CIs. In FoulPlay games
  only the bot seat contributes behavioral metrics (opponent seat contributes
  only the opponent-PP metric), and the FoulPlay config (search ms, version) is
  pinned in the artifact.
- **"When present on a moveset" denominators.** Move-category rates are
  conditional: for category C, the denominator is games in which ≥1 bot-side
  mon carries a C move (games-present), and we report (a) uses per
  game-present, (b) fraction of carriers that used it ≥1 time. This separates
  "the generator didn't deal the move" from "the policy ignores it".
- **Statistics.** Wilson 95% CIs for proportions; 1,000-resample game-level
  bootstrap for per-game means; species-vector entries with <50 team-instances
  flagged low-n; with ~150 species tested, per-species CIs are reported with a
  Benjamini-Hochberg q-value column so ~7 expected false 95% hits don't get
  narrated as signal.
- **Seeds.** Trait-eval games draw from a new reserved seed block registered in
  the deploy repo's seed-reservation file (disjoint from capstone and diversity
  blocks) so no cross-experiment seed reuse occurs.

## Phase 0 — inventory and feasibility (no GPU compute)

Inputs: shared experiment storage; each leg's run summary.
Outputs: `traits/inventory.json` — per lineage: resolved legs, games/iteration,
retained iteration list, milestone→(leg, iteration, sha256) table, and for each
milestone a data-source verdict for Phase 1 (see below).

1. Resolve the five patterns to concrete legs; verify continuity (leg N+1's
   `--completed-games` equals leg N's terminal games; same architecture flags).
2. Enumerate retained iteration checkpoints per leg; emit the milestone map with
   substitutions marked.
3. Probe archived self-play data recoverability for Phase 1: for one iteration
   of each lineage, inspect the training cache and (where present) the 50k-game
   trajectory dumps, and answer: can we recover per-turn (move name, switch,
   turn count, forced-vs-voluntary) for the games generated by a given
   checkpoint? Training caches are tensorized; recovery requires decoding action
   indices against the action-space mapping plus enough state to distinguish
   forced replacements. The verdict is per-lineage USE-ARCHIVE or REGENERATE.
   Ambiguity resolves to REGENERATE — archived-data archaeology must not become
   the critical path; hi-fi eval games are vs scripted benchmarks, not
   self-play, and are never used here.

Gate G0: inventory.json exists, every lineage has a 500k checkpoint pinned by
sha, and each milestone has a data-source verdict.

## Machinery (built once, generic over checkpoints)

Three public-repo scripts (cluster-agnostic; k8s job wrappers mirroring the
hi-fi shard sizing live in the private deploy repo):

1. **`scripts/trait_eval.py` — game generation with full event capture.**
   `--checkpoint <path> --opponent {self,foulplay} --games N --search-ms 100
   --seed-block <name> --out <dir>`. Drives the local showdown bridge
   (`battle_bridge.mjs`) with a server-side (omniscient) per-turn event tap and
   writes `events-<shard>.jsonl.gz`. Works with **any** checkpoint — the 500k
   restriction below is scope, not machinery. Inspection is **omniscient
   oracle**: the tap reads full simulator state for both seats every turn — PP
   tables (text logs don't carry PP, and Pressure doubles decrement), exact
   statuses, boosts, and hidden state — and immunity / Focus Punch outcomes
   come from resolved simulator effects, not inference. Owner has approved
   storing these higher-fidelity per-game records for trait-eval games; the
   full event logs are the canonical retained artifact.
2. **`scripts/trait_extract.py` — pure function events → metrics.**
   Versioned metric definitions (`metrics_version` in output); adding a trait
   later means re-running extraction over stored events, not regenerating
   games. Output: `metrics.json` per (checkpoint × opponent-mode).
3. **`scripts/trait_report.py` — aggregation + report.** Builds the
   lineage×milestone matrices and renders a self-contained no-CDN HTML report
   (same conventions as `diversity_report.py`): per-trait time series per
   lineage, cross-lineage overlays at matched game counts, and 500k deep-dive
   pages with self-play and FoulPlay columns side by side. Index page links all
   runs of the analysis.

**Event schema (sufficiency contract).** Per turn and per seat: turn number;
active species (both sides); chosen action (move name / switch target /
forced-replacement flag); move resolution flags (executed, failed, disrupted,
immune, blocked-by-substitute); non-volatile status of both actives; volatile
state (substitute up, per-side); boost stages (both actives); weather; side
conditions (spikes layers per side); full PP table per side (mon, slot,
pp-remaining). Per game: winner seat, remaining mons per side at end, last
active mon per side, total turns, team rosters (species + movesets per side).
Each metric below cites only these fields; any metric that would need more
fails the design review, not the implementation.

Gate G1 (machinery validation): 50-game self-play smoke on one 500k checkpoint;
hand-verify against raw logs — one turn-count, one pivot count, one PP
exhaustion, one immunity switch-in, one Focus Punch disruption. All five must
reconcile exactly before any full-scale run.

## Phase 1 — longitudinal basics, every 100k milestone

Scope: all lineages, every 100k milestone, self-play only.
Source: per Phase 0 verdict — USE-ARCHIVE lineages read the games generated by
the milestone checkpoint (the collect phase of the following iteration, ~1,600
games); REGENERATE lineages get 1,000 fresh self-play games from
`trait_eval.py` (archived and regenerated sources are labeled in the report and
never mixed within a lineage).

Metrics per milestone:
1. **Top 5 moves and rate of use** — share of all move-selections (both seats
   pooled) per move name; report top 5 with share and uses/game, plus the full
   distribution in the artifact (top-5 is presentation, not storage).
2. **Average turns per game.**
3. **Average pivots per game** — voluntary switches: switch chosen as the
   turn's action **plus Baton Pass-initiated switches** (owner: Baton Pass is
   a switch for pivot purposes); forced post-faint replacements excluded. The
   artifact stores pivots-excluding-BP as a secondary column since it is free
   to compute.

Gate G2: every (lineage × milestone) cell has metrics.json with n ≥ 900 games
(archive cells: n ≥ 1,400); report renders with no empty cells except
explicitly-marked pruned-checkpoint substitutions.

## Phase 2 — deep behavioral evals at 500k checkpoints

Scope for the first pass: **the five 500k checkpoints only** (one per
lineage). Volume: **5,000 self-play games + 1,000 FoulPlay games** per
checkpoint, stats kept separate throughout. The machinery is
checkpoint-generic, so extending to later 500k multiples (1M, 1.5M, 2M) is a
scheduling decision the owner makes after reading the first-pass results — no
standing cadence is armed.

### 2a. Winning-team species vector

For every species in the gen3 randbats pool: team-instances, winning-team
instances, and the win-correlation delta — self-play: `P(win | s on team) −
0.5`; FoulPlay: `P(bot win | s on bot team) − P(bot win overall)`. Wilson CIs,
BH q-values, low-n flags. Artifact stores the full vector; the report shows
top/bottom 15 with CIs.

### 2b. Move-category usage (per category: games-present denominator, uses per game-present, carrier-use fraction)

The concrete move lists are frozen in Phase 0 by intersecting these category
definitions with the actual gen3 randbats move pool (from the generator data),
and the frozen JSON ships with the artifact:

1. **Stat-boosting moves** — Swords Dance, Dragon Dance, Calm Mind, Bulk Up,
   Agility, Amnesia, Barrier, Acid Armor, Iron Defense, Cosmic Power, Tail
   Glow, Belly Drum, Growth, Meditate, Curse when used by a non-Ghost (Ghost
   Curse is not a boost). Focus Energy and Psych Up excluded.
2. **Healing moves excluding Rest** — Recover, Softboiled, Milk Drink, Slack
   Off, Synthesis, Morning Sun, Moonlight; Wish included but broken out as its
   own row (delayed heal). Ingrain excluded (residual). Drain attacks excluded.
3. **Rest** — its own row.
4. **Weather moves** — Sunny Day and Rain Dance only (owner: Hail cannot be
   set in gen3 randbats, and Sandstorm arrives only via Sand Stream). The
   Phase 0 pool intersection confirms the absence of move-set Sandstorm/Hail
   mechanically; if either unexpectedly appears in the pool, flag rather than
   silently add rows. Ability weather (Drought, Drizzle, Sand Stream) never
   counts.
5. **Phazing** (Roar, Whirlwind) — with a breakdown at time-of-use: opponent
   active had ≥1 positive stat stage or an active Substitute ("justified") vs
   neither ("neutral").
6. **Spikes.**
7. **Rapid Spin, only when own side has Spikes down** — the counted event is a
   spin with ≥1 Spikes layer on the spinner's side; total spins regardless of
   hazards stored as an auxiliary column for contrast.
8. **Toxic.**
9. **Paralysis-inducing status moves** — Thunder Wave, Stun Spore, Glare.
   Secondary-effect paralysis (Body Slam, Thunderbolt, …) never counts.
10. **Sleep moves** — Sleep Powder, Spore, Hypnosis, Lovely Kiss, Sing, Grass
    Whistle. Yawn excluded.
11. **Yawn** — its own row.
12. **Explosion + Self-Destruct** — combined row plus per-move split.
13. **Baton Pass** — with two context breakdowns at time-of-use: (a) caller
    had ≥1 positive stat stage active (a boost being passed), (b) caller had a
    Substitute up (a sub being passed). Reported alongside the plain use rate
    so boost-passing / sub-passing emerge as distinct learned behaviors.
14. **Substitute.**
15. **Focus Punch** — attempts, plus breakdown: executed vs disrupted (took a
    damaging hit during focus); success rate = executed/attempts. Executions
    into a Substitute or an immune target still count as executed (the focus
    succeeded; targeting is a separate skill).
16. **Leech Seed.**
17. **Solar Beam** — split by sun active vs not at the selection turn
    (sun-active beams are one-turn; charge-turn beams are the error case).

### 2c. Switch behavior

1. **Immunity switch-ins** — switch action where the incoming mon takes zero
   effect from the opponent's move that turn via type immunity (Ground vs
   Electric incl. Thunder Wave, Flying/Levitate vs Ground, Ghost vs
   Normal/Fighting, Dark vs Psychic, Steel vs Poison/Toxic) or absorbing/immune
   ability (Volt Absorb, Water Absorb, Flash Fire, Levitate, Soundproof).
   Counted on materialized immunity (the move resolved into the switch-in with
   no effect), reported per game and per 100 switches, with a type-vs-ability
   split.
2. **Switching a sleeping mon out** (voluntary switch while active mon asleep).
3. **Switching a frozen mon out** (voluntary switch while active mon frozen).
4. **Status-absorber switch-ins** — switching an already-statused mon in on an
   opposing status-inflicting move (the move resolved into the switch-in and
   was blocked by existing status).

### 2d. Resource and endgame metrics

1. **Bot-side PP exhaustions per game** — count of bot move-slots reaching 0 PP.
2. **Opponent-side PP exhaustions per game** — same, opponent seat (in
   self-play this is the mirror seat; in FoulPlay games it is FoulPlay's team,
   read from simulator state).
3. **Average bot mons alive on wins.**
4. **Average opponent mons alive on bot losses.**
5. **Top 5 last-active bot mons on wins** — the "closer" ranking, with counts.

Gate G3: all five checkpoints × two opponent modes have metrics.json with the
full n (5,000 / 1,000); report deep-dive pages render; the three cross-checks
(species-vector totals = 2×games for self-play, PP events ≥ 0 monotone,
switch-event counts ≤ total switches) all pass.

## Compute budget

- Phase 1 regeneration (worst case, all REGENERATE): ~45 milestone cells now ×
  1,000 self-play games ≈ 28 collect-iterations of volume, scheduled at low
  priority behind training runs; grows by one cell per lineage per 100k.
- Phase 2 now: 5 × 5,000 self-play (≈ 16 collect-iterations of volume, run as
  sharded CPU/GPU jobs mirroring collect parallelism) + 5 × 1,000 FoulPlay
  games (equal to one hi-fi foul-play battery per checkpoint; reuse the hi-fi
  GPU shard sizing: ≤8 shards, 4 CPU / 16 Gi, standard deadlines).
- Nothing here touches the training namespace's GPU quota while a training run
  is in its train step; wrappers submit at the same priority class as hi-fi
  evals.

## Execution order

Phase 0 → G0 → machinery + G1 (50-game smoke) → **Phase 2 on the five 500k
checkpoints** (validates the new machinery on the highest-value question) → G3
→ Phase 1 sweep of the 100k grid (archive-first, regen where verdicts say so)
→ G2 → report publication. The 100k Phase-1 grid keeps appending as the three
live lineages grow; deep-battery extension past 500k awaits an owner call.

## Defaults adopted (flag if wrong)

Owner rulings already incorporated: pivots include Baton Pass switches; Baton
Pass gets boost-passing and sub-passing breakdowns; weather rows are Sunny
Day/Rain Dance only (no move-set Hail/Sandstorm in gen3 randbats); PP and all
hidden state read omnisciently with full event logs retained; first pass runs
500k checkpoints only.

Remaining defaults:
1. Phase 1 regeneration size: 1,000 self-play games per milestone cell.
2. Wish counts as healing but broken out; Ingrain excluded.
3. Ability immunities count in 2c-1 with a type/ability split.
4. FoulPlay games at 100 ms search, matching the training/hi-fi convention.
5. Focus Punch executions into Substitute/immune targets count as executed.
