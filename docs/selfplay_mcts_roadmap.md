# Roadmap: self-play + test-time MCTS toward ladder-competitive play

Status: planning. This is the multi-workstream plan for getting pokezero from "coherent but
plateaued" to genuinely strong Gen 3 random-battle play, learned from first principles via
self-play (see [`goals.md`](goals.md)). It is written so independent agents can each own a
workstream in parallel.

## The proven recipe we are following

The MIT thesis *"Winning at Pokémon Random Battles Using Reinforcement Learning"* (Jett Wang,
2024) reached **rank 8 / 1693 Elo** on the official gen4 random-battle ladder — the best known
non-human result — with a recipe that is directly applicable and was achieved on **modest
compute** (one GPU + ~80 CPU workers, ~3M self-play battles, ~4 days):

- **Train a policy/value net via PPO self-play — *without* MCTS in the loop.** The thesis explicitly
  diverges from AlphaZero here: the simulator is too slow to generate MCTS-improved targets at the
  scale a net needs to converge. PPO self-play is the training engine.
- **Add MCTS only at inference**, as a *policy-improvement operator* on top of the trained net.
- **Determinize hidden information**: sample the opponent's hidden set → the rollout becomes a
  (near) perfect-information game.
- **Restore the Markov property** by encoding multi-turn effect durations into the observation.

This de-risks the direction: it is an engineering+scale problem with a known-good shape, not a
research gamble. Our job is to reproduce it for Gen 3 on our stack and push past it.

## Current assets (what already exists)

- **Warm-pooled sim** (`local_showdown.py`, `scripts/battle_bridge.mjs`): battle-id-keyed bridge,
  ~0.4 ms/turn warm, byte-identical battles, ~2× collection throughput. Collectors reuse one env.
- **Replay-from-root prefix harness** (`replay_branching.py`): rebuilds a battle branch point from
  the original seed + recorded action prefix, with real-sim equivalence tests modulo Showdown
  timestamp lines, submits explicit divergent branch actions, then rolls out from the resulting
  state with normal policy semantics.
  `rollout_cli replay-benchmark` measures prefix-replay latency before considering snapshot/restore.
- **Flat branch-search harness** (`search.py`): enumerates legal root actions, evaluates each via
  replayed branch rollout, and selects by terminal outcome. This is a deliberately small search
  stepping stone before value-guided MCTS.
- **Value-guided branch scoring** (`search.py` + transformer value helper): enumerates legal root
  actions through the replay branch harness and scores non-terminal post-branch states with a value
  function over the player's observation history. This is the bridge from flat terminal rollout
  search toward PUCT/MCTS leaf evaluation.
- **Root PUCT branch scorer** (`search.py` + transformer prior helper): combines policy-head priors
  with value-branch scores at the search root, establishing the first PUCT-style selection layer
  before deeper tree expansion. `neural_cli root-puct-benchmark` evaluates that scorer on sampled
  rollout prefixes to measure action deltas and search latency, while
  `neural_cli root-puct-counterfactual` replays the recorded and selected branches forward for a
  first retrospective outcome signal before changing live rollout control.
- **Context-aware root-PUCT policy adapter** (`PolicyContext`, `RootPUCTSearchPolicy`): the rollout
  driver can now call policies with player-local decision context, and the root-PUCT adapter can
  select actions through a separate branch env without mutating the live rollout. Simultaneous-turn
  opponent actions are supplied by an explicit planner hook; `greedy_opponent_action_planner` can
  drive that hook from player-local opponent-action priors such as the transformer's auxiliary
  opponent-action head. In self-play/search benchmarks, the harness can additionally mask planned
  opponent actions against requested-player legal masks to avoid invalid branch submissions; this is
  recorded as search metadata. Because the mask includes opponent legal-action availability, this is
  a privileged benchmark safety guard and is not a substitute for full hidden-information
  determinization.
  The play benchmark also has an opt-in conservative value gate (`--min-value-improvement`) that
  keeps the raw policy-prior action unless the configured root-selected action beats it by a configurable
  value margin; aggregate diagnostics report gate uses/checks so M0 runs can tell whether the knob
  is actually changing decisions.
  The same benchmark can now switch root selection between current PUCT-score selection and pure
  branch-value selection (`--selection-mode puct|value`), which isolates whether the near-term M0
  issue is the branch value signal or the PUCT prior/score mixer.
  It can also run opt-in bounded leaf continuations (`--leaf-rollout-rounds`) before evaluating a
  root candidate, using real simulator steps and raw checkpoint policies to test whether short
  rollout leaves produce a better root value signal than immediate one-ply value-head evaluation.
  `--leaf-rollout-rounds-sweep` can run multiple leaf depths in one same-seed benchmark artifact so
  depth comparisons are less error-prone than separate one-off commands.
  `neural_cli root-puct-play-benchmark` compares raw checkpoint play against root-PUCT checkpoint
  play over full games on the same fixed-opponent matchups.
  A first local smoke run against `max-damage` proved the full-game path executes end to end with a
  current-schema checkpoint: 4 total games at `--games 1`, active-search decisions around
  3-4 decisions/s, and no runtime crash. This is harness evidence only, not strength evidence: the
  tiny 4-game-trained smoke checkpoint lost to `max-damage`, and one root-PUCT orientation capped at
  the 50-decision smoke limit.
  A first small current-schema probe then trained a 64-game scripted-teacher BC checkpoint and ran
  `--games 4` against `max-damage`: raw scored 3/8, root-PUCT scored 2/8, both with zero capped
  games. This is too small to decide strength, but it is negative search-lift evidence for the
  current shallow operator and confirms active-search throughput around 3 decisions/s.
  Benchmark JSON now includes root-PUCT decision diagnostics so follow-up M0 runs can see search
  counts, fallbacks, candidate counts, selected values/scores, and per-decision search latency.
  After adding opponent-action legality masking, a repeat `--games 4` probe on the same 64-game
  teacher-BC checkpoint showed the known invalid-action fallback was resolved (`211` root-PUCT
  searches, `0` fallbacks), but still did not show strength lift: raw and root-PUCT both scored
  1/8 vs `max-damage`, with zero capped games and active-search throughput around 2.7-2.8
  decisions/s. This points back to operator/value quality rather than submit-validity as the next
  M0 bottleneck.
  A follow-up probe with the conservative value gate on the same 64-game teacher-BC checkpoint and
  a fresh fixed seed range also failed to show lift: at `--min-value-improvement 0.0`, the gate made
  no changes (`0/244` uses/checks) and raw/root-PUCT both scored 3/8 vs `max-damage`; at
  `--min-value-improvement 0.25`, the gate changed a few decisions (`8/245`) but raw/root-PUCT still
  both scored 3/8. Both runs had zero fallbacks and zero capped games, so the next M0 work should
  focus on improving the search/value signal itself rather than only adding conservative vetoes.
  A value-selection probe (`--selection-mode value`) on the same checkpoint and seed range also
  scored 3/8 vs `max-damage`, matching raw and same-seed PUCT despite selecting from a different
  value/score profile. That makes "PUCT prior mixing alone" a weaker explanation for the current
  miss; the more likely M0 bottlenecks are value-head quality, one-ply depth, and opponent/chance
  modeling.
  A value-calibration pass on the same checkpoint's own 64-game training rollouts reinforces the
  value-quality concern even before testing generalization: over 3,809 in-sample examples, the value
  head measured MSE 0.752, MAE 0.810, sign accuracy 0.722, and expected calibration error 0.187.
  That is too noisy to trust as the only one-ply search leaf signal.
  A first bounded-rollout-leaf probe (`--leaf-rollout-rounds 1`) on the same checkpoint and
  max-damage seed range gave a small positive but still far-from-gate result: raw scored 3/8 and
  root-PUCT with one rollout leaf round scored 4/8, with zero capped games and zero search
  fallbacks. The run recorded 252 root-PUCT searches, actual candidate leaf rounds split across
  0-round immediate terminals and 1-round continuations, and active-search throughput around
  2.2-2.4 decisions/s. This is exploratory evidence that replacing immediate one-ply leaves may
  matter, but it is not enough to clear M0 or justify scale; the next read should either enlarge
  this leaf-depth probe or improve value/opponent modeling further.
  A same-seed depth-2 follow-up moved in the same direction: raw again scored 3/8, while
  root-PUCT with `--leaf-rollout-rounds 2` scored 5/8, with zero capped games and zero search
  fallbacks across 255 root-PUCT searches. Candidate leaf diagnostics showed 9 immediate terminal
  branches, 20 one-round continuations, and 1,419 two-round continuations; 80 candidate branches
  reached rollout terminals and 1,359 still required truncated leaf value evaluation. Active-search
  throughput stayed around 2.1-2.3 decisions/s. The sample is still too small for a strength claim,
  but the one-step and two-step probes now both point toward leaf depth/value-target quality as a
  better next M0 lever than more PUCT-score mixing.
  A larger same-seed sweep using the new `--leaf-rollout-rounds-sweep 0/1/2` harness then ran
  `--games 8` per mirrored matchup against `max-damage` on a fresh seed range. It preserved the
  same direction but still did not clear M0: raw scored 3/16, leaf0 scored 4/16, leaf1 scored 4/16,
  and leaf2 scored 5/16, with zero capped games and zero search fallbacks in every searched row.
  Leaf2 ran 436 root-PUCT searches at roughly 2.3-2.4 active-search decisions/s; candidate leaf
  diagnostics recorded 23 immediate terminal branches, 51 one-round continuations, 2,473 two-round
  continuations, 101 rollout-terminal candidate leaves, and 2,423 truncated value-head leaves. This
  strengthens the case that deeper leaf evaluation is the right M0 lever, but the absolute score
  remains much too low; next work should either run a larger depth sweep for variance reduction or
  improve the leaf evaluator/opponent model before spending larger training compute.
- **Value-head calibration report** (`value_calibration.py`, `neural_cli value-calibration`):
  measures MSE/MAE/bias/sign accuracy and predicted-value calibration bins against rollout return
  targets; this is the first WS-E metric before using the value head for MCTS leaf evaluation.
- **Entity-token transformer policy+value net** (`neural_policy.py`) — richer than the thesis's
  3-layer MLP; already has policy, value, and opponent-action heads.
- **Public belief engine** (`belief.py`) — narrows the opponent's hidden set from observable facts;
  a better basis for determinization than ad-hoc set sampling. It now exposes bounded
  player-relative opponent determinizations for search, preserving unknowns instead of inventing
  hidden facts.
- **Self-play iterate loop** (`neural_cli iterate` → `neural_selfplay.py`, `selfplay.py`,
  `collection.py`) — collect → train → benchmark, with promotion gates (`evaluation.py`).
- **Benchmark harness** (`collection.benchmark_rollouts`, `neural_cli benchmark`) — vs
  random/simple/max-damage baselines.
- **Online ladder client** (`online_client.py`) — can play a checkpoint against the live server.
- **Raw-facts observation** with Markov-restoring encodings (turn count, future-sight, toxic stage,
  screens) already present (`showdown.py`).

## Where we stand / why we plateau

~20 prior training-method variants converged to ~0.46–0.52 vs max-damage (the imitation ceiling).
Pure searchless self-play risks settling into a mediocre local equilibrium. The fixes — a history
pool, exploration pressure, and (above all) a search improvement operator — are exactly the recipe
above, and are all knowledge-free / on-mission.

## Strategy hypothesis & go/no-go gates

**Core hypothesis (unverified):** prior self-play stalled at ~0.52 because we never combined (a)
enough scale, (b) a real opponent *league*, (c) exploration pressure, and — decisively — (d) a
**search improvement operator**. The recipe above asserts all four break the plateau; gen4→gen3
transfer is also assumed. Treat this as a hypothesis to test, not a given.

**Search is the load-bearing bet, so prove it first and cheaply** (see M0): the thesis's strength
came from *net + MCTS*, and net-alone may well re-plateau at ~0.52. Do **not** over-invest in
scaling PPO before demonstrating that search lifts a modest net past the plateau. WS-D does **not**
require a strong net — search improves any decent one — so it should not be gated on a great M1.

**Go/no-go gates:**
- **M0 gate:** on a cheap/early net, net+MCTS must clear ~0.60 vs max-damage (well past the 0.52
  plateau). If search does *not* move the needle here, scaling PPO will not save us — stop and
  rethink the operator (deeper search, better value head, DUCT) before spending fleet compute.
  The first smoke measurement only validates the plumbing; the real M0 read still requires a
  current-observation-schema checkpoint with meaningful strength. Older local checkpoints trained
  before the latest observation features can fail with numeric-feature shape mismatches and should
  be retrained or skipped for M0 evidence.
  The first 64-game-BC probe did **not** show lift, so the next M0 step should improve the search
  operator/value quality or run against a meaningfully stronger current-schema checkpoint before
  spending larger benchmark samples.
- **M1 gate:** the per-iteration strength curve must *rise* over ≥10 league iterations; a multi-
  iteration flatline = stuck → lean on search and revisit league diversity + exploration.

---

## Workstreams (parallelizable)

Each workstream lists scope, concrete steps, deliverables, acceptance criteria, and dependencies.
WS-A, WS-C, WS-E, WS-F can start in parallel today; the first milestone (M0) proves search on a
*modest* net before WS-B commits fleet compute to scaling. WS-D depends on WS-C + WS-E + a decent
(not great) net from WS-A.

### WS-A — Self-play PPO training loop (the RL engine)
**Owner goal:** a robust PPO self-play loop that reliably *climbs* on a fixed strength yardstick,
with anti-stagnation machinery.

Steps:
1. Audit/solidify the PPO path in `neural_cli iterate` / `neural_selfplay.py`: advantage estimation,
   value-head loss weighting, entropy bonus, capped-game return, gradient/clip settings.
2. **History/league opponent pool — diversity, not just recency:** sample opponents from a bounded
   set of *past* checkpoints (not just the latest) to kill non-transitive cycling and forgetting.
   Crucially, guard pool *diversity*: a pool of near-identical aggression-exploiters (the failure
   mode we already hit) induces no learning pressure. Add a behavioral-diversity check and/or a
   dedicated exploiter agent folded back into the pool. Wire through the existing promotion registry
   / historical-opponent plumbing.
3. **Exploration pressure:** expose and tune entropy coefficient + collection temperature; ensure
   collection samples (not greedy) so the policy keeps exploring.
4. **Fixed-yardstick eval every iteration** (see WS-F) and persist the strength curve.
5. Remove imitation as a *crutch*: support cold self-play from a weak/random init as the on-mission
   path; keep the scripted-teacher bootstrap only as an optional warm-start/control, clearly flagged.

Deliverable: `neural_cli iterate` that trains a net via league self-play with exploration and a
per-iteration strength curve.
Acceptance: strength vs the fixed yardstick **rises** across ≥10 iterations (not a flatline);
no degenerate-collapse (capped-game rate bounded).
Touches: `neural_selfplay.py`, `selfplay.py`, `collection.py`, `neural_cli.py`.

### WS-B — Distributed scaling (parallel collection → central train)
**Owner goal:** turn one-box self-play into a CPU fleet hitting the thesis's ~3M-battles budget in
days. Collection is the CPU bottleneck; fan it out.

Steps:
1. **Collection/train split:** make collectors emit rollout JSONL to shared storage keyed by
   iteration + shard; a central step trains on the aggregated shards and publishes the next
   checkpoint; collectors pick up the new checkpoint. (This is the distributed form of the existing
   collect→train loop; the *code* for sharded collection + aggregation lives in the tracked repo.)
2. **Iteration controller:** a loop that, per iteration, launches N collector shards against the
   current checkpoint, waits, runs the central train, and advances.
3. **On-policy consistency (critical — PPO is on-policy):** use **synchronous iterations with a
   barrier** — every collector shard uses checkpoint N; train N→N+1 only after all shards finish.
   Do **not** mix rollouts collected under different checkpoints into one PPO update; stale rollouts
   degrade PPO. (If we later want asynchronous collection, switch to a staleness-tolerant objective
   — out of scope for v1.)
4. **Data pipeline at scale:** rollout JSONL is ~TB-scale at 3M battles (≈215 MB / 200 games).
   Design shard layout, cross-shard shuffle for training, and a retention policy (keep recent
   iterations, prune old) so storage and train-time I/O stay bounded.
5. **Hardware split:** collection is CPU (the fleet); the **central train step benefits from a GPU**
   (the thesis trained the net on one GPU). Provision the train step accordingly — collection stays
   CPU-only, training is not.
6. **Fleet deployment** (a CPU pod fleet): container image, parallel-collection manifests, shared
   storage, and the iteration controller. All environment/location specifics are deliberately kept
   **out of this (public) repo**.
7. Throughput target: enough parallel CPU to reach ~3M battles in single-digit days.

Deliverable: sharded-collection + central-train code (tracked); the fleet deployment itself is kept
out of this repo.
Acceptance: end-to-end iteration runs across many workers; aggregate games/hour scales ~linearly
with workers; identical rollout records to the single-box path (equivalence test).
Touches: `collection.py`, `neural_selfplay.py`, `rollout_cli.py` (tracked). Deployment manifests are
kept out of this repo.

### WS-C — Battle forking / snapshot-restore (the MCTS enabler)
**Owner goal:** explore alternative lines from a battle position — the prerequisite for MCTS. Pick
the *simplest* mechanism that meets the per-move search budget; do not assume snapshot/restore.

Steps:
0. **Verify how the thesis did rollouts first.** It ran MCTS over the Showdown sim, so branch
   exploration is solved prior art — match its approach (snapshot/restore vs replay-from-root)
   before inventing. This is the single highest-leverage de-risking step in the whole plan.
1. **Prefer replay-from-root if the warm sim makes it cheap enough.** With determinization, each
   rollout re-simulates from the battle's recorded line + a sampled opponent set; warm sim
   (~0.4 ms/turn) may make this fast enough for shallow search and avoids state serialization
   entirely. Validate the per-move cost against a realistic search budget.
2. **Only if replay-from-root is too slow:** build snapshot/restore — investigate `BattleStream`
   serialization (`Battle.toJSON()` / restart-from-state) at our pinned commit, then extend
   `battle_bridge.mjs` (already battle-id-keyed) with `snapshot {battleId}` and
   `fork {fromBattleId,newBattleId,state}`; expose `LocalShowdownEnv.snapshot()/fork()`.
3. Validate either path: explore divergent lines from turn N and confirm each is byte-identical to a
   from-scratch battle that took the same actions (modulo timestamps).

Deliverable: a forking/rollout mechanism (replay-from-root preferred) + equivalence tests.
Acceptance: divergent lines are deterministic and identical to ground-truth replays; per-move rollout
cost fits the search budget.
Touches: `local_showdown.py`, and `scripts/battle_bridge.mjs` only if snapshot/restore is needed.
Risk: this gates all of WS-D — validate the mechanism in days, not weeks. Last-resort fallback is a
learned/in-process model (much larger effort).

### WS-D — Test-time MCTS (the policy-improvement operator)
**Owner goal:** a search-augmented policy that measurably beats the raw net, mirroring the thesis.
Depends on WS-C (forking) + a *decent* net from WS-A (not a great one) + a **well-calibrated value
head** (WS-E). MCTS leaf evaluation is bounded by value quality: a noisy value head makes search
*worse* than the raw policy, so value calibration is a hard prerequisite, not a nicety.

Steps:
1. MCTS skeleton over the forkable sim, guided by the net (PUCT-style: prior from policy head, value
   from value head; back up values).
2. **Determinization:** at the search root, sample the opponent's hidden set from the belief engine
   (`belief.py`) → search a (near) perfect-information instance; average over a few sampled sets.
3. **Chance handling — damage-roll grouping:** collapse damage outcomes to KO / no-KO branches (per
   Foul Play) instead of all 16 rolls; optionally best/worst/avg-case chance aggregation
   (*-minimax-style) as a knob.
4. **Opponent move during search:** start with the thesis's approach (opponent plays the net
   policy); leave a hook for DUCT (decoupled UCB, true simultaneous-move handling) as an upgrade.
5. Search budget / time control; integrate as an alternate `select_action` (net+MCTS).

Deliverable: a net+MCTS policy usable in benchmark and ladder play.
Acceptance: net+MCTS **beats net-alone** by a clear margin on the fixed yardstick and head-to-head.
Touches: new search module (e.g. `search/mcts.py`), `belief.py`, `local_showdown.py` (fork API),
`neural_policy.py` (priors/value), `policy.py` (a search-policy adapter).

### WS-E — Value-head calibration + observation/belief support (on the MCTS critical path)
**Owner goal:** a value head good enough to guide MCTS, a Markov-complete observation, and a clean
belief-sampling API for the searcher. Not "lighter" — WS-D's search quality is bounded by the value
head, so this gates M0/M3.
Steps: audit and improve value-target construction (terminal return, discount, capped-game value)
and measure value-head **calibration** (predicted vs realized outcome); confirm multi-turn-effect
duration encodings are complete; expose a clean belief-determinization (opponent-set sampling) API.
Deliverable: a calibration metric + improved value targets + a belief-sampling API.
Acceptance: value-head calibration is good enough that net+MCTS > net-alone (verified jointly in M0);
WS-D can request sampled opponent sets.
Touches: `showdown.py`, `belief.py`, `dataset.py`, `randbat_vocab.py`.

### WS-F — Evaluation, strength tracking, and ladder
**Owner goal:** a *fixed* yardstick to detect climbing vs stagnation, and a path to human-relative Elo.
Steps:
1. Per-iteration eval vs a frozen set: max-damage + a few frozen past checkpoints; persist the curve.
2. A larger eval (≥300–400 games) for low-variance strength reads at milestones.
3. Ladder path: use `online_client.py` to play checkpoints on the live server for human-relative Elo
   (the ultimate goal).
Deliverable: strength-curve tracking + a ladder eval runbook.
Acceptance: a flat multi-iteration curve reliably signals stagnation (→ add search); a rising curve
confirms progress.
Touches: `collection.py`, `evaluation.py`, `neural_cli.py`, `online_client.py`.

---

## Sequencing & milestones

**Ordering principle: prove the load-bearing bet (search) cheaply *before* spending fleet compute on
scale.** Search improves any decent net, so it must not wait for a fully-scaled M1.

- **Now (parallel):** WS-C (verify forking/rollout — the riskiest unknown), WS-E (value-head
  calibration + belief-sampling API), WS-A (league self-play to produce a *modest* net), WS-F (fixed
  yardstick). WS-B (full fleet scaling) can be scaffolded but is **not** on the critical path to M0.
- **M0 — Prove search lifts a modest net (the de-risking gate):** WS-C + minimal WS-D + WS-E on a
  cheap/early WS-A net → **net+MCTS clears ~0.60 vs max-damage** (past the 0.52 plateau). Pass →
  scale. Fail → fix the operator (search depth / value head / DUCT) before any fleet compute.
- **M1 — Scaled self-play net:** WS-A + WS-B + WS-F → a league-trained net on the fleet at ~thesis
  scale, with a *rising* strength curve (M1 gate).
- **M2 — Full MCTS:** harden WS-D (determinization over multiple sampled sets, roll-grouping, search
  budget; DUCT if opponent-as-policy limits strength) on the scaled net.
- **M3 — Search beats net at scale:** net+MCTS clearly beats net-alone and baselines at the larger
  eval (≥300–400 games).
- **M4 — Ladder:** WS-F ladder path — measure human-relative Elo; iterate.

## Anti-stagnation guardrails (apply throughout)
- League/history-pool opponents (WS-A) — not just the latest self.
- Exploration pressure (entropy/temperature) so the policy doesn't collapse early.
- Fixed-yardstick strength curve (WS-F) — the early-warning signal for local minima.
- Search (WS-D) as the ultimate improvement operator — the thing that pulls the policy out of a
  searchless local optimum.

## Open questions / risks
- **Forking/rollout mechanism (WS-C)** is the biggest unknown; validate in days. Prefer
  replay-from-root over snapshot/restore; last-resort fallback is a learned/in-process model.
- **Value-head quality (WS-E)** is a hard MCTS dependency — a noisy value makes search worse than the
  raw policy. Measure calibration, don't assume it.
- **On-policy staleness (WS-B):** distributed PPO must use synchronous, single-checkpoint iterations
  or it degrades — do not mix checkpoints in one update.
- **Sim speed for search:** warm sim is ~0.4 ms/turn, but MCTS multiplies sim calls; per-move budget
  matters (the thesis worked within a 10 s/move ladder timer).
- **Simultaneous moves:** start with opponent-plays-policy (simple); upgrade to DUCT if it limits
  strength vs stronger opponents.
- **In-loop MCTS (true AlphaZero) is a research *stretch*, not near-term.** The thesis avoided it for
  sim-speed reasons, and in-loop MCTS for *simultaneous-move* games is genuinely hard. The validated
  path is PPO-self-play-then-test-time-MCTS; treat in-loop as a later experiment, not a milestone.
- **Plateau-break + transfer are hypotheses:** prior self-play stalled at 0.52, and the thesis was
  gen4, not gen3 (different mechanics: type-based phys/spec split, gen3 sleep/ability set). Hold the
  go/no-go gates.
- **Ladder eval is noisy/slow and the online client (`online_client.py`) is young** — it needs
  reconnect/timeout hardening before ladder Elo is a trustworthy signal.
- **Compute:** thesis hit rank 8 on ~3M battles / one GPU / ~80 CPU / 4 days — our budget target.

## References
- MIT thesis (PPO self-play + test-time MCTS; rank 8 gen4) — the blueprint.
- Foul Play (DUCT + damage-roll grouping). Technical Machine (expectiminimax). *-Minimax / MCMS
  (best/worst/avg chance). metamon (offline-RL human-level). See `docs/max_damage_exploration_learnings.md`
  for the plateau analysis that motivates the search direction.
