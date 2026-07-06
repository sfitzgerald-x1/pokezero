# Next-train readiness plan

Status: plan of record, 2026-07-04. Everything that must land between the
design freeze (#491/#493 merged) and the launch of the next training
wave, organized as five workstreams with gates, plus the launch
checklist and the open decisions. Companion docs:
[`observation_compression_design.md`](observation_compression_design.md)
(the frozen spec), [`gen3_interaction_inventory.md`](gen3_interaction_inventory.md),
[`wse_shaping_and_coverage_design.md`](wse_shaping_and_coverage_design.md),
[`research_synthesis.md`](research_synthesis.md) (theses & strategy stack),
[`eval_opponents.md`](eval_opponents.md).

## Context the plan is built on (this week's measured facts)

512d broke the foul-play band (14.7% @ ~405k) on tactics alone — its
value head is in-cluster (0.5226) and *more* hazard-blind than smaller
models (self-ΔV 0.1% of spread). Capacity climbs within the meta;
coverage/shaping/attribution remain the only route to the strategy
tier. E0-oracle: search adds +4.0 pts at its ceiling with a 0.515 head
(p≈0.056) — real but thin; search spend stays paused. The observation
spec is frozen and review-hardened; the belief train/eval mismatch is
fixed (#492) with provenance stamping from the next trained checkpoint.

## WS-1 — Implementation package (the one-way door)

Four PRs, in order, all against the frozen spec + corrections layer:

- **A — exact-state belief layer**: PP ledger (catalog max PP, our-side
  Pressure ×2, Sleep-Talk-charges-caller, Transform-instance scoping),
  non-proc pruning (no-Leftovers per damaged end-of-turn; no-Lum on
  stuck status; no-pinch-berry **end-of-turn gated**; Mud Shot
  Shield-Dust rule), sleep counters (Rest flag candidate-conditioned on
  Early Bird; live sleep-clause bit), weather/screen durations + source,
  pending-Wish counters, turns-in-battle, Natural Cure & Early Bird
  eliminations, Trick item-mutation with pruning freeze, and
  **trapper-alive flags** (revealed Shadow Tag/Arena Trap/Magnet Pull
  persists while the trapper is benched, cleared on its faint —
  approved in the original feature review but omitted from the design
  doc's exact-state list; this plan is the corrective record).
  *Gate: unit fixtures per rule + replay of the 5-game audit corpus
  asserting every rule fires where the logs say it should.*
- **B — extraction functions**: transition tokens (canonical schema
  incl. `called`/`transformed`/`pursuit-intercept`/outcome enum/context
  trio/positional pair; one token per declared action) and tendency
  stats ((count, opportunity) pairs; prediction-channel routing).
  *Gate: pure-function tests against fixture lines + audit logs; the
  two open sim experiments (Pursuit KO-intercept, self-secondaries on
  sub hits) resolved as fixtures.*
- **C — observation spec v-next (the single break)**: window=1 +
  K=128 transition-token slots + stats token + exact-state fields +
  **reserved zero-masked Tier-2 slots** + computed expected stats
  (exact 4, variant-conditioned HP/Atk); 24 recent-event tokens
  dropped; checkpoint-compat guards; feature masks for ablation arms.
  *Gate: encode/decode round-trip; old checkpoints load-and-refuse
  cleanly; provenance stamps carry the new spec version.*
- **D — Tier-2 residual/CB (mask-flipped only on gate pass)**:
  poke-engine expected-damage queries, whitelist, two-strike CB,
  crit/outcome conditioning, Solar Beam rain/sand/hail, Flash Fire
  volatile modifier.
  *Gate: precision ≈ 1.0 for the CB bit against ground-truth items
  from omniscient controlled-game logs; residual calibration ≈ 0-mean
  roll-shaped on known-set control games. Ships zero-masked if the
  gate fails — launch does NOT wait for D.*

**Standing condition, stated explicitly: the wave trains and evaluates
belief-on** (`POKEZERO_BELIEF_SET_SOURCE=1` everywhere) — the new spec
makes candidate-set features standard input, and the twin-pool
experiment (WS-3) is what validates the interpretation of belief-on
reads, not whether to enable them.

## WS-2 — Deploy side (private repo)

Image rebuild from post-merge main (picks up #492's set-source fix +
the new spec); `POKEZERO_BELIEF_SET_SOURCE` into the foul-play
probe/benchmark pods (training pods already have it); verify
provenance fields appear in the first new-image checkpoints; keep the
current four arms on the old image to completion (no mid-run
switchover — probe comparability).

## WS-3 — Measurement readiness (before, not after, launch)

1. **Pools**: freeze `pool-self-v1` (done — the belief-1.5m held-out
   pool); build **`pool-fp-v1`** from captured controlled foul-play
   games (capture patch + encode); re-derive the Pearson bar on it.
2. **Probe suite wired into the milestone cron**: cross-pool Pearson
   (gate read), ΔV probe, behavior probes, **spin-block puzzle** (new;
   fixture-built), meta-histograms (move-class usage — the
   differentiation curve). Every ~100k milestone, both pools.
3. **Ecology watchdogs** (the 4L lesson): game-length drift (+50% from
   the 25–30-turn band) and matched-milestone strength regression as
   automated alarms/kill criteria.
4. **Twin-pool re-encode experiment** (cheap, pre-launch): re-encode
   identical protocol lines with set-source on/off and score both —
   definitively separates feature-presence effects from
   game-distribution effects before the new spec bakes belief-on
   observations into everything.

## WS-4 — Wave composition (decision required)

Anchor: 512d, new spec. **Standard budget: every test arm runs to
500k** (decided 2026-07-04) — matching the bake-off yardstick so all
reads compare at matched milestones against the historical controls;
volume beyond 500k is exclusively the designated continuation run
(open decision 2), never a test arm. Controls are **historical** — the
completed width-wave curves at matched milestones (no GPU spent
re-running A/A8; the spec's masked-config keeps within-wave ablation
cheap). Candidate slots for ~4 concurrent arms:

Eval-target constraint: new strength reads use current-family v2+ checkpoints
and matched milestones only, where v2+ means observation-schema v2 or newer
rather than older no-belief/pre-v2 families. Random/simple remain plumbing
checks, not strength gradients. Legacy-family checkpoints at any milestone are
historical context rather than opponents to evaluate against; calibration pools
can still be used for value/diagnostic reads when labeled as such.

| slot | arm | tests |
|---|---|---|
| 1 | **E-512d vanilla** (K=64, transition tokens + stats + exact-state) | compression at width vs historical 512d curve |
| 2 | **E-512d + dense shaping** (#487 arm 1: status-delta + hp/faint) | tripod leg 2; ΔV self-response is the primary read |
| 3 | **E-512d + coverage pool** (#487 arm 2 with cross-arm opponents) | tripod leg 3; Spikes argmax + meta-histograms |
| 4 | wildcard — pick one: metamon-S continuation (steepest climber), R-NaD-lite regularization arm (anti-collapse, pre-depth), or K=16 mask ablation | decision below |

Gates at 250k (from #487): ΔV movement + argmax movement without
matched-milestone foul-play regression; watchdogs live from 0k.

## WS-5 — Reads that inform but do not block

E0-ladder rung on 512d-500k when it finishes (predict: similar absolute
delta on higher base); full per-head E1 on the wave's best head at
250k; FP-ladder rung calibration (FP-10/50 one-off).

## Launch checklist (ordered)

1. PRs A → B → C merged, full suite green, provenance verified.
2. Image rebuilt + deploy env wiring; smoke: 5 controlled games on the
   new image with belief-on, warnings clean.
3. Twin-pool experiment read; pool-fp-v1 frozen; probe cron live with
   watchdogs.
4. Wave slots decided (WS-4 table); arm configs written to deploy repo.
5. D's gate run attempted once (ships masked either way).
6. Launch; first milestone read at 50k confirms probes fire end-to-end.

## Horizon — the program beyond this train (triggers, not dates)

Everything tiered/gated past the next wave, consolidated from the
design docs so it stops living in "deferred" footnotes. Six tracks:

**H1 — Search revival** *(paused per E0's thin +4.0)*: rerun E0-oracle
with the wave's best head (trigger: a head passes the ΔV ≥10%-of-spread
bar) and with belief-2-3m-class heads as the free upgrade; then
**matrix-game root search** (Simultaneous-AlphaZero-style per-node
solves replacing opponent-action scenarios — the soundness fix);
then depth-limited re-solving; endgame **PBS/ReBeL** over the belief
engine (the closed-universe bet). **Expert-iteration stages** stay
gated on a decisive E0 margin — search amplifies, never teaches.

**H2 — Population/diversity** *(beyond the wave's cross-arm pools)*:
**self-derived z conditioning** (descriptor pseudo-rewards over the
stats vocabulary the new spec already computes — hazard/status/switch
rates; the AlphaStar mechanism minus the human source); then
exploiter-lite arms / PSRO-proper with `f_hard` matchmaking (trigger:
cross-arm pools move ΔV/argmax but plateau before the strategy tier).

**H3 — Value-head machinery**: **multi-γ heads** (trigger, per #487:
shaping moves ΔV but degrades terminal calibration); **defender-side
ability inference** (the symmetric Tier-2 extension, cheap once D
exists); **Tier-3 evidence-weighted posteriors** (requires a behavior
likelihood model — i.e., the quarantined opponent world-model below).

**H4 — Opponent world-model** *(the one sanctioned human-data use)*:
train a "what would they click" model from the PokéChamp 500k+
high-Elo corpus, for search-time opponent priors (opens the spin-block
branch that self-play blindness keeps shut) and exploiter training —
never as policy teacher. Companion: **pool-human-v1** via
`pokezero-replay-import` as the third eval pool.

**H5 — Architecture falsification & depth retry**: the
**attention-knockout probe** on existing checkpoints (an afternoon;
decides whether attention is load-bearing at 2 layers); the
**set-encoder control arm** (thesis-style, attention-free, games/hour-
matched) if the knockout is ambiguous; **3L depth retry** only with the
full stabilizer bundle (QK-LN, depth-scaled init, warmup, logit cap)
AND an equilibrium guard (R-NaD-lite or pool share) — the 4L
postmortem's two-front lesson.

**H6 — Scale & debts**: the 512d **volume continuation to ~3M**
(decision 2 below — T6's unfinished axis); muP if width scaling
continues past 512d; small debts: `online_client` set-source threading
(same fix shape as the bridge), finetune base-hash cross-check,
FP-ladder rung calibration, metamon OOD-rung calibration / gen3-OU
side yardstick from the eval registry's future-use list.

## Open decisions (Scott)

1. WS-4 slot 4: metamon-S continuation vs R-NaD arm vs K=16 mask.
2. Whether the 512d → 1M/3M volume continuation (T6's unfinished axis)
   starts on the old spec now or waits to ride the new spec (waiting
   costs weeks of volume; starting costs comparability with the wave).
3. Curation cron re-creation (still pending from the original handoff)
   — the probe suite in WS-3 wants to ride it.
4. Open PRs to disposition: #485/#486 (runbook/probe branches — merge
   or fold into the new-spec world), #488 (research package), #489
   (eval registry).
