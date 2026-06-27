# Learning Architecture Exploration

This document is a working brainstorm for the *post-linear* learning architecture:
the model and training algorithm that should replace the dependency-free linear
softmax baseline once the CPU harness is trusted. AlphaGo Zero / AlphaZero is the
stated inspiration, so this doc starts from that lineage and then adapts it to the
ways Gen 3 random battles differ from Go.

This file now doubles as a **goal specification**: point `/goal` at it to drive the
work below. Follow `docs/autonomous_task_loop.md` for the per-task loop (Claude executor,
Codex GPT-5.5 xhigh reviewer, serial, merge-on-sign-off) and the conventions in
`docs/goal_cpu_past_simple_legal.md`. The sections after the goal are the supporting design
menu the hypotheses draw from.

## Active Goal: Beat `max-damage` > 80% From Self-Play

Produce a CPU-trained (GPU for scale) neural policy that **beats the `max-damage` baseline
more than 80% of the time**, reached by *self-play learning* rather than imitation alone.

### Where we are
- The compact entity-token transformer, trained supervised (behavior-cloning + value) on
  64 teacher games, scores ~0.80 vs `simple-legal` and ~0.94 vs `random-legal` — but it has
  **not** beaten `max-damage`, which itself scores 0.875 vs `simple-legal` (a tougher bar
  than our current model). See `docs/cpu_self_play_roadmap.md`.
- The category embedding was retired from a 1,000,001-row hash to a compact ~795-id
  vocabulary, so the model dropped from ~128M params (513 MB) to ~1.3M params (~5 MB) with
  no loss in benchmark strength. **The trunk is now ~270K params of ~1.3M total — there is
  large headroom to grow the network and add structure while staying small enough for
  CPU proof-of-life and a consumer GPU.** This is what makes the hypotheses below affordable.
- The RL infrastructure (PPO objective, eval-only benchmark-reference, TensorBoard) is built and
  a first self-play pilot ran but **plateaued vs `max-damage` at ~0.24** — for reasons that are
  about the *loop*, not the thesis. See "Findings So Far & The Self-Play Concern" below before
  reading the hypotheses.

### Definition of Done
- A checkpoint beats `max-damage` in a mirrored shared-opponent benchmark with **Wilson 95%
  lower bound > 0.80** over **>= 240 games per seed band**, **reproduced across two
  independent seed bands**, with capped-game rate <= 0.05.
- It still beats `random-legal` and `simple-legal` decisively (no regression).
- The winning checkpoint was produced by **self-play training** (model-free RL), not pure
  behavior cloning. `max-damage` is an **eval-only** opponent: it must **not** appear in the
  training opponent pool or as the initial policy (enforced by `reject_eval_only_specs`), so
  beating it measures genuine generalization, not overfitting to the eval.
- Every claim is backed by a targeted shared-opponent benchmark (not wrapper aggregates).

### Why `max-damage` is the right bar — and why it is beatable
`max-damage` is strong but **deterministic and exploitable**: it always clicks the
highest-damage move on its current active Pokemon, and it **never switches, never uses
status/recovery/setup, and never plays around your switches**. A competent learned policy
should exploit exactly that: switch into resists/immunities so its big move is wasted, set
up (Swords Dance / Calm Mind) on a mon it cannot threaten, and stall/status it. Crucially,
these are the *same skills self-play is supposed to discover* (switch timing, opponent
prediction, setup, HP preservation) — so `max-damage` is both a meaningful bar and a clean
signal that self-play is learning real tactics. >80% is an ambitious-but-reachable target
against a no-switch/no-status opponent.

## Findings So Far & The Self-Play Concern (first pilot)

The RL infrastructure now exists end-to-end: a PPO clipped-surrogate training objective
(PR #208), an eval-only `--benchmark-reference-policy` so each iteration is scored vs
`max-damage` without it entering training (PR #209), and TensorBoard scalars including
`winrate/max-damage` (PR #210). The first BC-bootstrapped PPO self-play pilot has run on CPU.
**It plateaued well below the bar — and the diagnosis matters more than the number.**

- **Setup:** rebuilt a randbat-dex-vocab behavior-cloning bootstrap from the scripted teacher
  (0.96 vs `random-legal`, 0.78 vs `simple-legal`, **0.21 vs `max-damage`**), then ran 10
  iterations of PPO self-play against a diverse pool (self + `simple-legal` + `scripted-teacher`),
  150 games/iteration.
- **Result:** the policy kept improving against its pool-mates but **plateaued vs `max-damage`
  at ~0.24 (best 0.30), no upward trend.**

### Why it plateaued — and why this does *not* yet refute self-play

The pilot manifests show the loop **barely did self-play, and what little it did froze fast**.
This is the central concern to fix before drawing any conclusion about whether self-play can
beat `max-damage`:

1. **No mirror matches early.** This loop's "self-play" is *current-vs-(fixed pool + promoted
   history)*, and the history pool starts empty — a self snapshot only enters after a checkpoint
   is *promoted*. So iterations 1–5 trained **only against `simple-legal` + `scripted-teacher`**:
   the same conventional distribution the BC bootstrap already cloned. There was no
   current-vs-current game at all until iteration 6.
2. **The collector froze.** The candidate advanced only at iterations 3 and 5; from iteration 6
   on, candidate-vs-incumbent sat at ~0.45–0.50 (`failed_to_beat_incumbent` every time). 0.5 vs
   its own snapshot *is* the self-play fixed point — the arms race stalled almost immediately and
   the policy just jittered inside the BC basin.
3. **No search + weak exploration.** Unlike AGZ (whose escalation comes from MCTS-amplified
   targets over millions of games), this is model-free policy gradient on a small net exploring
   only via softmax sampling + a tiny entropy bonus. The population never *generates*
   `max-damage`'s hyper-aggressive "never switch, always nuke" corner strategy — and you cannot
   learn to counter a strategy your self-play never produces.
4. **Self-play optimizes its own equilibrium, not robustness to an external exploiter.** A
   converged self-play policy is robust against the strategies it *encountered*; it carries no
   guarantee of beating an out-of-distribution fixed opponent that plays a specific
   exploitable-but-effective style the population never explored. (Well-known self-play failure
   mode: strong vs self, still loses to a simple unseen exploiter.)
5. **Tiny scale.** 150 games/iteration × 10 iterations on a ~1.3M-param net is far short of what
   a model-free self-play arms race needs to discover and stabilize new tactics.

**Conclusion: H1 is not refuted — we never ran a genuine self-play arms race.** The plateau is
consistent with "the loop reinforced the bootstrap distribution and froze," not with "self-play
cannot beat `max-damage`."

### Revised plan: fix self-play *first*, seed aggression only as a fallback

Before concluding self-play is insufficient (and falling back to curriculum), make the loop
actually function as an arms race and re-test the emergence thesis directly:

- **Mirror matches from iteration 1** — include current-vs-current self-play in collection from
  the start, instead of waiting for a promotion to seed the history pool.
- **Real exploration** — expose/raise collection temperature and entropy so the policy explores
  outward instead of jittering in the BC basin; consider a softer advancement gate or a small
  population so the collector keeps moving.
- **Scale the run** — more iterations and, once it helps, a bigger trunk (H5), watching
  `winrate/max-damage` live on TensorBoard.

**Only if a genuine self-play run still flatlines** do we seed a `max-damage`-*like* aggressive
opponent into the **training** pool (H8 as curriculum), with the real `max-damage` held strictly
eval-only. That is the warranted fallback, but it is curriculum design, not emergence — and it
risks overfitting to the aggressive style rather than learning general tactics, so it is the
second choice, not the first. The honest experiment is: does a *real* self-play arms race
discover the counter to relentless aggression on its own?

## Hypotheses (ranked) — what should move win-rate vs `max-damage`

Each is a falsifiable hypothesis with an experiment and the signal that confirms/refutes it.
They draw on the design menu in the later sections (candidate network bodies, operators,
privileged supervision, etc.).

- **H1 — Self-play RL is the necessary unlock (highest expected impact).** Supervised BC
  caps at imitating the teacher; exceeding `max-damage` requires learning from *outcomes*.
  The PPO loop is built and a first pilot ran, but it **did not actually test H1** — the
  collector froze and the first half had no mirror matches (see "Findings So Far & The
  Self-Play Concern"). *Experiment (corrected):* run a **genuine self-play arms race** —
  current-vs-current mirror matches from iteration 1, real exploration (temperature/entropy),
  and enough iterations that the collector keeps advancing — benchmarking vs `max-damage` each
  iteration. *Signal:* vs-`max-damage` win rate climbs past 0.80. *Refuted only if* it
  plateaus well below 0.80 **while the collector is still advancing and exploring** (i.e. a
  healthy arms race that still can't beat the external exploiter) — not, as in the first
  pilot, while frozen at a self-play fixed point.
- **H2 — Temporal memory enables anticipatory switching.** A recurrent core (LSTM/GRU/small
  state-space) over per-turn embeddings should beat the fixed 4-turn window because exploiting
  `max-damage` is a *sequential* read (it keeps clicking the same predictable move). Now cheap
  given the param headroom. *Experiment:* ablate recurrent vs windowed under the same RL setup.
  *Signal:* higher vs-`max-damage` win rate and better long-game switch timing. Early evidence:
  an opt-in GRU temporal aggregator is implemented, but a same-data teacher-BC/value-selection
  probe did **not** beat the mean-pooled current-schema checkpoint (142/600 vs `max-damage`
  versus 150/600 for mean; worse held-out value ECE/MAE/sign). A follow-up same-shape
  always-advance PPO run gave only a small/noisy lift over mean pooling (166/600 vs 159/600
  against `max-damage` on the same seed bands), still nowhere near the required base-net strength.
  Treat this as evidence that temporal capacity is available and may help slightly, but the next
  bottleneck is the self-play training signal/opponent pool rather than GRU plumbing.
- **H3 — Opponent-action prediction (privileged self-play supervision) sharpens switches.**
  Train the opponent head against self-play ground truth. Against deterministic `max-damage`
  the opponent's next move is near-perfectly predictable, so the agent can pre-switch to a
  resist/immunity. *Experiment:* sweep the opponent-loss weight (incl. 0); measure
  vs-`max-damage` win rate and opponent-prediction accuracy. *Signal:* win-rate gain that
  tracks prediction accuracy.
- **H4 — Pointer/attention action head over move semantics.** Replace the positional
  9-slot head with attention over the action-candidate tokens (move type / base power /
  category / status), so move choice reasons about *effects* not slot index. *Experiment:*
  ablate pointer head vs positional under fixed training. *Signal:* better tactical selection
  (more setup/switch lines, higher win rate).
- **H5 — Grow the trunk now that it is affordable.** Embedding dim 128 -> 256, layers 2 ->
  4-6, heads 4 -> 8 keeps the model at a few M params (still tiny vs the old 128M). *Experiment:*
  width/depth sweep, watching CPU inference cost and overfitting. *Signal:* higher ceiling
  on vs-`max-damage` win rate without inference becoming the bottleneck.
- **H6 — Belief-conditioned inputs from the known finite sets.** Feed the deterministic
  randbat belief posterior (possible moves/items/abilities per revealed slot) so switch/threat
  assessment uses what a player could infer. *Experiment:* ablate belief features. *Signal:*
  safer switches, fewer losses to a revealed coverage move.
- **H7 — Reward shaping + value head for long-horizon credit.** Faint/HP-fraction-differential
  shaping plus a trained value head should speed RL convergence to stall/setup lines (the win
  often comes many turns after the decisive switch). *Experiment:* ablate shaping/value weight.
  *Signal:* faster, more stable climb vs `max-damage`.
- **H8 — Diverse opponent pool / light league for robustness.** Train against self + frozen
  checkpoints + random/simple/scripted-teacher (never `max-damage`). *Experiment:* pool-
  composition ablation. *Signal:* generalization — beating the *unseen* `max-damage` rather
  than overfitting any single opponent. *Note:* the first pilot used exactly this conventional
  pool and did **not** transfer to `max-damage` (~0.24) — but with a frozen collector and no
  early mirror matches, so it tested the *loop*, not the pool. The open question is whether a
  healthy self-play arms race over this pool generates aggression on its own, or whether the
  pool must be *seeded* with a `max-damage`-like aggressor (curriculum) for the equilibrium to
  account for relentless aggression.

## Recommended Experiment Sequence (goal task order)

1. **Build the model-free RL self-play loop (H1). [DONE — infra built, first pilot plateaued.]**
   The PPO objective, eval-only benchmark-reference, and TensorBoard are merged, and a CPU
   proof-of-life ran. It plateaued vs `max-damage` because the loop barely self-played (see
   "Findings So Far"). **Next: make it a real arms race** — current-vs-current mirror matches
   from iteration 1, exploration knobs (temperature/entropy), and a longer run — then re-read
   `winrate/max-damage`. Only if a genuine arms race still flatlines, fall back to seeding a
   `max-damage`-like aggressive *training* opponent (H8 as curriculum; real `max-damage` stays
   eval-only).
2. **Add temporal memory + opponent-prediction + pointer head (H2, H3, H4)** on top of the RL
   loop, one ablation at a time, each gated by a benchmark vs `max-damage`.
3. **Scale the trunk and tune (H5, H7)** once the structure helps, watching CPU inference cost
   (the env is already the throughput bottleneck — see Infrastructure) and promote to GPU for
   the longer strength runs.
4. **Belief inputs and pool diversity (H6, H8)** as supporting gains and robustness.
5. **Confirm and reproduce** any checkpoint that clears 80% (two seed bands, clean capped
   health), per the Definition of Done.

Treat each step as a goal-loop task: pick the next hypothesis, implement on a `scott/` branch,
benchmark the candidate vs `max-damage` + `random-legal` + `simple-legal` across two seed
bands, record the result in `docs/cpu_self_play_roadmap.md`, get Codex sign-off on code PRs,
merge, and continue. Stop and report if the next step needs GPU, a missing prerequisite, or if
evidence shows a hypothesis is a dead end (record it as negative evidence and move to the next).

### Constraints
- CPU-first for proof-of-life (does the signal move?); GPU + env throughput for strength runs.
  The Showdown harness is the throughput bottleneck (~91% per-turn sim) — see Infrastructure;
  `--workers` defaults to 16.
- `max-damage` is eval-only and must never enter training (initial policy or opponent pool).
- No strength claim without a targeted shared-opponent benchmark; validation/imitation
  accuracy is health only, never a strength signal.

## Why The Linear Baseline Is Not Enough

The linear softmax policy was always scaffolding (`docs/first_iteration_design.md`,
"Deviations From Original Plan"). It hashes the observation into fixed features and
learns independent per-action weights. It cannot:

- model interactions between board state, the active matchup, and a candidate move
- represent opponent uncertainty as anything richer than bucketed counts
- share a representation between the policy, the value estimate, and opponent
  prediction (the linear "opponent head" is provably inert because it has its own
  weights over the same hashed features)
- generalize across move/Pokemon identities by *meaning* rather than by hashed slot

Empirically, the CPU roadmap shows the loop runs and is auditable, but no linear
checkpoint has clearly beaten the teacher-bootstrap reference. That is the expected
ceiling of a linear model on a game this combinatorial. The next gain has to come
from representation + a stronger policy-improvement operator, not more tuning.

## What "AlphaGo Zero As Inspiration" Actually Means Here

AlphaGo Zero's recipe has five load-bearing parts:

1. A single deep network with a **policy head and a value head** over a shared trunk.
2. **Self-play** with the current best network generating its own training data.
3. **MCTS** (PUCT) used at move time as a *policy-improvement operator*: search
   produces a better action distribution than the raw policy head.
4. Training targets = **MCTS visit-count distribution** (for policy) and the
   **final game outcome** (for value).
5. A **cheap, perfect, deterministic forward model** (the rules of Go) that MCTS can
   expand millions of times.

Parts 1, 2, and the outcome-trained value head transfer cleanly and we already have
the surrounding infrastructure (self-play loop, frozen-checkpoint opponent pool,
promotion gates, benchmarking, audits). Parts 3-5 are where Pokemon breaks the
assumptions, and that is the real design problem.

### Go vs Gen 3 Random Battle

| Property | Go (AGZ) | Gen 3 randbat (PokeZero) |
| --- | --- | --- |
| Information | Perfect | **Imperfect** — hidden team, moves, items, abilities, exact HP/EVs |
| Dynamics | Deterministic | **Stochastic** — damage rolls (0.85–1.0), accuracy, crits, secondary effects, speed ties |
| Turn order | Alternating | **Simultaneous** choices on most turns (asymmetric only on forced switches) |
| Forward model | Cheap, perfect, in-process | **Subprocess Showdown BattleStream**; slow to fork; and an agent-side model would have to *invent* hidden state to step |
| State space | Fixed 19×19 board | Variable revealed entities + a *distribution* over opponent realities |

The consequence is blunt: **a literal AlphaGo Zero loop is not runnable as-is.** PUCT
MCTS needs a fast, perfect, deterministic simulator the agent can query thousands of
times per move. We have none of those three properties. So the design question is not
"AlphaZero or not" — it is "which AlphaZero ideas do we keep, and what replaces MCTS
as the policy-improvement operator until/unless real search becomes feasible."

The part of AlphaGo Zero we actually want is **the self-play learning loop**: a policy
that improves by repeatedly playing itself and learning from the outcomes, with no
hand-coded strategy. MCTS was *how* AGZ produced an improvement signal; it is not the
goal. The goal here is "learn to play Gen 3 randbats from self-play." That keeps the
self-play/promotion/value-head machinery and treats search as an optional accelerator,
not a requirement.

## The Random-Battle Advantage: A Known, Finite Set Space

Gen 3 randbats are a deliberately *easier* learning target than human-teambuilt formats,
and the architecture should exploit exactly why.

In open formats, a revealed Pokemon could carry almost any spread of moves, item, ability,
EVs, and nature — the hidden space is effectively unbounded and human-chosen. In randbats,
every Pokemon is drawn from a **known generator with a small, enumerable set of possible
moves/items/abilities.** That changes the nature of the hidden-information problem:

- Opponent uncertainty is a **posterior over a finite, listable set of realities**, not an
  open-ended guess. The deterministic belief tracker (`docs/first_iteration_design.md`,
  "Belief Tracking") already collapses that list from public events.
- Therefore "model the opponent" can be a **classification head over candidate sets** for
  each revealed slot, not generative uncertainty. The network learns the *behavioral*
  narrowing on top of the rules-based narrowing the tracker already does.
- The small hidden space is what makes everything below tractable: belief supervision,
  opponent-action prediction, and any future determinized search all get cheaper because
  the branching over hidden realities is bounded and known up front.

This is the single biggest reason to be optimistic that self-play can work here without
the full AlphaZero search apparatus.

## Temporal Modeling And Opponent Inference

A first-class requirement: the model must use **history to infer hidden cards**, because in
this game a turn's best move often depends on what the opponent's earlier behavior implies
about their unrevealed moves/sets.

Motivating example: if the opponent repeatedly switches *out* in front of one of your
Pokemon, that is strong evidence their active mon cannot threaten it. The skilled response
is to read the switch coming, predict *what they switch to*, and punish the predicted
incoming Pokemon — not to react one turn late. A purely per-turn model cannot represent
this; the signal lives in the sequence of past decisions.

Implications for the architecture:

- **Carry belief state across turns.** A fixed history window (the current scaffold uses 4)
  may be too short for patterns that develop over a long game. Prefer a **recurrent core**
  (LSTM / GRU / small state-space model, AlphaStar-style) or full-game causal attention so
  accumulated evidence persists for the whole battle.
- **Make opponent inference an explicit objective.** Keep the existing opponent-next-action
  head and add an **opponent-set / belief head** (which candidate set, which moves remain
  possible). These heads pressure the recurrent state to actually encode "what have I
  learned about the opponent so far."

## Privileged Self-Play Supervision

Self-play makes the hard part cheap. Because we control both sides, **the true hidden state
of the opponent is known at training time**, even though the policy only sees public
information at inference. So the opponent-action and opponent-set/belief heads can be
**supervised directly against ground truth** during self-play, instead of being learned
purely indirectly through reward.

This converts the central imperfect-information difficulty — inferring hidden cards from
behavior — into a supervised auxiliary loss riding on top of the RL objective. It is the
asymmetric-information trick used by strong imperfect-info agents (e.g. AlphaStar's use of
privileged signals), and it is *unusually* effective here because the finite randbat set
space makes the supervision target small and well-defined. The policy/value path stays
strictly public-information-only so the deployed agent never depends on hidden state.

## Design Tension: Explicit Belief vs. Emergent Inference

Open and explicitly *not yet decided*. The question "do we need an opponent-belief system"
hides three separate knobs with different answers:

1. **Belief as input** — feed the deterministic tracker's candidate-set features into the
   net. This is just exposing known, legal game structure (analogous to AGZ getting the
   board, not hand-crafted features). Low controversy; keep it.
2. **Belief as auxiliary output** — a head that predicts the opponent's hidden set / next
   action, supervised against self-play ground truth. This is the contested knob.
3. **Belief as a hard-coded module** — a Bayesian engine the policy must consume. Most
   committed and least "Zero"; rejected for now because it bakes in our decomposition.

The contested knob (2) is a genuine open tension, not a decision:

- **Case for emergence:** end-to-end RL agents (AlphaStar, OpenAI Five) developed implicit
  belief in their recurrent state with no belief head — the hidden state *is* the belief
  state, shaped by the win/loss gradient. Forcing prediction of the literal set may waste
  capacity on facts that do not affect winning, when a coarser task-relevant latent
  ("this mon is passive, expect a pivot") would serve better.
- **Case for explicit supervision:** terminal reward is a long, noisy credit-assignment
  chain; an auxiliary loss gives a dense, immediate, *correct* gradient every turn, and the
  signal is free and exact here thanks to self-play ground truth plus the finite set space.
  It is also *measurable*: a belief probe shows whether inference is actually improving,
  turning an otherwise black-box failure into a diagnosable one.

**Resolution: treat it as an ablation, not an architecture commitment.** The temporal
*capacity* to infer (recurrent state or history attention) is required either way and is
the real commitment. The belief head is a single tunable loss weight: set it positive to
accelerate and instrument early training, then anneal toward zero to let the policy rely on
whatever internal representation actually wins. This honors the emergence hypothesis (the
weight can go to zero) while not discarding an unusually clean free signal. Decide it by
training with and without the aux loss and comparing both win rate and belief-probe
accuracy — the cheapest A/B test in the project.

Bitter-lesson note: predicting the opponent's true set is predicting a *fact the
environment provides*, not encoding a human heuristic about how to play. "If they switch
twice, assume X" would violate the bitter lesson; "predict the hidden state we can observe
in self-play" does not.

## Decision Axes

Five mostly-independent choices:

1. **Network body** — how we encode the observation.
2. **Imperfect-information handling** — what we do with opponent uncertainty.
3. **Policy-improvement operator** — model-free RL vs search vs learned-model search.
4. **Training targets / losses** — outcome, advantage, visit counts, auxiliaries.
5. **Inference cost / deployment** — must stay fast for high-throughput self-play and
   should eventually fit a consumer GPU.

## Candidate Network Architectures

### A. Refined entity-token transformer (extends the current scaffold) — recommended base
`neural_policy.py` already implements `EntityTokenTransformerPolicy`: categorical +
numeric + token-type + history-position embeddings, a `TransformerEncoder`, and
policy/value/opponent-action heads. This is the right family. Refinements to explore:

- **Set-invariant entity tokens.** Treat the 6 self and up to 6 opponent Pokemon as a
  set (no positional bias beyond "active"), so the model generalizes across team order.
- **Belief-conditioned opponent tokens.** Feed the deterministic belief tracker's
  candidate-set distribution per revealed slot, not just bucketed counts. Start from
  the existing compact features; consider explicit per-fact embeddings later.
- **Pointer/attention action head.** Replace the positional linear-over-9-slots head
  with attention over the `action_candidate_tokens`, so a move logit comes from the
  move's *semantics* (type, BP, status, target) rather than its slot index. This is the
  single biggest generalization win over the linear baseline.
- **Heads:** policy + scalar value (win prob), opponent-action aux (already present),
  optional belief-prediction aux (predict masked opponent facts) to pressure the trunk
  to actually use uncertainty.

### B. Deep Set / GNN over entities
Permutation-invariant, cheaper than full attention. Likely subsumed by a small
transformer; keep as a fallback if transformer inference is too slow on CPU.

### C. Recurrent core (AlphaStar-flavored)
An LSTM/GRU/state-space core over per-turn embeddings instead of (or on top of) a fixed
history window. Better for long games and partial observability; pairs naturally with
model-free RL. Tradeoff: harder to batch, statefulness complicates the rollout format.

### D. MuZero-style latent dynamics model
Learn a dynamics model in latent space and search over *that*, sidestepping the slow,
hidden-information real simulator. Discussed under training operators below; it is a
network-architecture choice too (representation + dynamics + prediction networks).

## Candidate Policy-Improvement Operators

This is the crux — what plays the role MCTS plays in AGZ.

### 1. Model-free deep RL (PPO / V-trace / IMPALA-style) — recommended first
No search at inference. The network *is* the policy; self-play + policy-gradient
updates improve it. `docs/first_iteration_design.md` already names PPO as the V0
training approach. AlphaStar is the precedent that a large imperfect-information,
stochastic game can be driven to superhuman strength by model-free RL + league play
without per-move tree search.
- **Pros:** reuses every existing system (rollout JSONL, promotion, audits, opponent
  pool); CPU-smoke-friendly; cheap inference; no fast forward model required.
- **Cons:** sample-hungry; not literally "AlphaZero"; value/credit assignment is harder
  with sparse terminal reward (mitigated by the shaping already specified).

### 2. AlphaZero-adapted search (the literal inspiration, made to fit)
Keep MCTS but fix each broken assumption:
- **Imperfect info →** Information-Set MCTS / **determinized** search: sample N opponent
  teams/sets from the belief tracker, run search per determinization, aggregate. We
  already have a deterministic belief engine, which makes this unusually tractable.
- **Stochasticity →** explicit **chance nodes** (expectimax over damage/accuracy/secondary
  outcomes) or sampled stochastic MCTS.
- **Simultaneous moves →** **decoupled PUCT** or regret-matching (CFR-style) at
  simultaneous nodes, or solve the small per-node matrix game.
- **The blocker:** MCTS needs to expand many nodes per move, and each expansion needs a
  forward model. The current Showdown BattleStream is a subprocess — far too slow to
  fork thousands of times per decision. This is the gating constraint, not the math.
  Mitigations: shallow depth with value-net leaf evaluation; an **in-process / batched
  simulator**; or a **learned dynamics model** (→ option 3).

### 3. MuZero / Stochastic MuZero
Learn the dynamics in latent space and run MCTS over the learned model, so we never need
a fast queryable real simulator and never have to hand-handle hidden state during search.
Stochastic MuZero adds chance outcomes.
- **Pros:** most faithful to the "Zero" spirit *given* our constraints; no fast real sim
  needed; learns its own abstraction of hidden info.
- **Cons:** the largest research + engineering lift; hard to debug; stochastic +
  imperfect-info MuZero is near the frontier. High risk for a proof of concept.

### 4. Hybrid: model-free policy + decision-time value search
Train the value+policy network model-free (option 1), then bolt on a *shallow*
value-net search (e.g., 1–2 ply determinized expectimax / decoupled search) **only at
evaluation/decision time** for a strength boost. This is the cheapest way to test
whether search helps at all in this game before committing to a full AlphaZero training
loop.

## Recommended Staged Path

- **Stage 0 (done):** Linear baseline validates the harness end to end.
- **Stage 1 — temporal self-play RL.** Build the recommended body — a per-turn
  **set-encoder** over entity tokens feeding a **recurrent core across turns** — and train
  it in the self-play loop with model-free RL (PPO/V-trace), value head trained to game
  outcome. Add the **opponent-action and opponent-set/belief heads supervised with
  privileged self-play ground truth** from the start, since temporal opponent inference is
  the point, not a later nicety. Reuse promotion/benchmark/audit. Keep it CPU-smoke-able;
  scale on GPU. **Highest value, lowest risk; unblocks real representation learning.**
- **Stage 2 — strengthen representation.** Pointer action head (logits from move
  semantics), richer belief-conditioned opponent tokens over the enumerated candidate
  sets, longer temporal context, and tuning of the belief auxiliary losses.
- **Stage 3 — decision-time search as a bolt-on.** Add shallow determinized value-net
  search at eval time and measure the strength delta vs the pure policy. This is the
  first concrete AGZ-flavored experiment and it is low-commitment: it answers "does
  search even help here?" before we pay for a search training loop.
- **Stage 4 — full search-based training (research).** Only if Stage 3 shows search
  materially helps *and* we have a fast enough forward model: IS-MCTS visit-count policy
  targets (AlphaZero-adapted) and/or Stochastic MuZero over a learned model.

Rationale for model-free first: (a) the original design already targets PPO; (b) no fast
forward model exists today, so MCTS is blocked on infrastructure regardless; (c) it
reuses all existing infra and stays CPU-smoke-friendly; (d) it de-risks the
representation before we invest in search. AlphaGo Zero ideas then enter incrementally
exactly where they pay off, instead of forcing a search loop the simulator can't support.

## Model Infrastructure

Resist introducing new frameworks; most of the stack is already chosen.

- **Framework: PyTorch.** Already the `[neural]` extra (`neural_policy.py` is an
  `nn.Module` with a `TransformerEncoder`). No reason to reach for JAX or anything else for
  a proof of concept.
- **RL algorithm: a lightweight, bespoke PPO/V-trace loop** (CleanRL single-file spirit),
  *not* a heavy framework (RLlib / SB3 / TorchRL). Our shape — a custom recurrent
  multi-head model, privileged auxiliary losses, and an already-built
  league/promotion/benchmark/audit system — is exactly what heavy frameworks fight. A few
  hundred lines of explicit PPO is more controllable than bending a framework to fit.
- **Reuse the existing harness.** The iterate→collect→train→benchmark→promote loop, the
  frozen-opponent pool, and the audits are assets. Wire a new *learner* into them; do not
  replace them.
- **Two concrete data-path changes** for RL: rollout records must also store the
  **behavior-policy log-probs and value estimates** (for PPO importance weighting), and
  collection must keep **each game's steps contiguous** (no cross-game shuffling) so the
  recurrent core can backpropagate through time.
- **Write it device-agnostic from day one** (`.to(device)`, no hard-coded CPU/GPU
  assumptions). This is the single most important decision for the CPU→GPU question: it
  makes "promote to GPU" a flag, not a port.

## CPU Proof-Of-Life vs. GPU For Strength

Separate two questions that have different bottlenecks and different hardware answers:
"prove the design works" vs. "train something strong."

**This workload is environment-bound, not compute-bound, at proof-of-concept scale.** The
model is small (millions of params), but every step is a subprocess round-trip to Showdown
BattleStream across a 30–60-turn game. The simulator dominates wall-clock, not the network
(the CPU loop already sustains roughly 0.6–2 games/s). A GPU at this stage would sit
starved behind a slow env, so it does not even attack the real bottleneck yet.

### CPU is sufficient — and correct — for proving the design out

Validating correctness and learning signal needs *iterations and signal*, not throughput,
so it belongs on CPU and fits the existing CPU-first philosophy. Promote to GPU only after
these proof-of-life criteria pass:

- gradients flow, losses descend, no NaNs, and seat-symmetry holds
- the model can **overfit a handful of games** (basic plumbing sanity)
- it **beats `random-legal`, then `simple-legal`** (first real learning signal)
- **belief-probe accuracy rises over the course of a game** (the direct test for the
  explicit-vs-emergent belief tension above)
- PPO is stable across iterations and the promotion gate behaves as expected

These are order 10^4–10^5 games — hours to a couple of days on CPU. This mirrors the
run-health-vs-strength split the CPU roadmap already uses: a passing proof-of-life run
proves the *design*, not strength.

### GPU (plus env throughput) is what buys strength

Reaching real randbat strength is sample-hungry — conservatively 10^6–10^7+ games, which
is weeks-to-months on CPU and not viable. But **a GPU alone does not fix it, because the
env is the bottleneck.** The strength-scale architecture is the classic **actor–learner
split**: many CPU env workers (subprocess Showdown) generating rollouts in parallel feeding
**one GPU learner** doing batched forward/backward. GPU accelerates the learner and batched
inference; CPU parallelism (or a faster/in-process engine) feeds it. Buying a GPU without
scaling env throughput leaves it idle.

### Sequencing and the promotion trigger

1. **CPU proof-of-life** — validate architecture and learning signal against the criteria
   above. No GPU.
2. **Promotion trigger** — once those criteria pass *and* `games/s` vs `net-forward/s` is
   instrumented to confirm where the bottleneck actually is.
3. **GPU + env parallelism together** — actor–learner for the strength run, likely with a
   bigger model and full-game attention.

The clean mental model: **CPU proves the design is correct; GPU plus env throughput is what
buys strength.** Because the bottleneck today is the simulator, the *first* scaling lever
after proof-of-life may be "speed up / parallelize the env," not "buy a GPU" — measure
before committing GPU budget.

## Other Infrastructure Implications

- **Forward-model speed is also the gate for any search.** A subprocess-per-step simulator
  cannot back MCTS. Before Stage 3/4, we likely need an in-process or batched Gen 3 engine,
  or a learned dynamics model. The same env-throughput work that feeds a GPU learner is
  what later makes search feasible — worth scoping once, used twice.
- **The GPU training path is unvalidated** (`first_iteration_design.md`, "Not implemented
  yet"). The promotion step above is where it must be validated.
- **Consumer-GPU target** constrains model size (`embedding_dim`, layers, heads) — keep the
  architecture sweepable and small by default.
- **Inference must stay fast** to preserve high-throughput self-play (`docs/goals.md`).
- Reward shaping, the 250-turn cap, and capped-game scoring are already specified and carry
  over unchanged.
- **Observability uses TensorBoard for now** (local, one dependency); see the Observability
  section.

## Observability

Live observability matters increasingly as runs move from minutes to multi-hour CPU
proof-of-life and then GPU strength runs: point-in-time CLI snapshots
(`cpu-long-run-report`, `compare`) are not enough once you need *curves over time* —
win-rate vs iteration, the RL losses, belief-probe accuracy, `games/s`, RSS, and promotion
events — to catch a bad run early instead of reading a manifest after the fact.

**Goal: use TensorBoard for now; do not build a custom browser server yet.** TensorBoard
pairs natively with PyTorch, adds one dependency, runs fully local (no external service,
nothing published off-machine), and provides live scalar curves and run comparison in a
browser for free. That covers the proof-of-life phase.

The one piece of enabling work worth doing early is **decoupling metric emission from any
viewer**: have the training loop write append-only scalar streams (TensorBoard event files,
and/or metrics JSONL alongside the manifests already persisted). Any viewer then reads from
that stream, so the trainer is never coupled to a UI. Runs already persist structured JSON,
so this is a small step.

Deliberately deferred:

- **Weights & Biases** — richer (system metrics, run tables, sweeps) and a good fit at the
  GPU/scale phase, but it is a *hosted external service*: logging to it publishes run data
  off-machine. Revisit at scale as a deliberate choice, not a default. Self-hosted MLflow or
  Aim are local alternatives if a server is wanted without building one.
- **A custom browser server** — justified only once Pokemon-specific views are needed that
  generic tools cannot render: live battle replay of in-progress self-play games, per-turn
  belief-probe visualization (watching the opponent-set posterior sharpen as evidence
  accumulates — a strong debugging tool for the temporal-inference work), and a
  promotion/league graph of the opponent pool. Build it only when the generic tools
  demonstrably fall short, not before.

## Open Questions

- Is decision-time search worth it in a stochastic, imperfect-information game, or does a
  strong model-free policy capture most of the value? (Stage 3 is designed to answer this.)
- Do we build an in-process/batched simulator, or learn a dynamics model, to make any
  search feasible?
- Determinization count vs compute budget for IS-MCTS.
- Simultaneous-move handling: decoupled PUCT vs regret matching vs per-node matrix solve.
- Temporal context: windowed transformer (current) vs recurrent core.
- Value target: pure win/loss vs the existing shaped reward.
- Belief representation inside the net: bucketed facts (current) vs explicit masks vs
  learned set embeddings.
- Online RL vs the current iterate-train-promote loop: does PPO fit the existing
  manifest/promotion model, or does it need a new training driver?
- Is the first post-proof-of-life scaling lever a GPU learner or a faster/parallel env?
  (Decide from measured `games/s` vs `net-forward/s`, not a priori.)
- In-process/batched Gen 3 engine vs. staying on subprocess Showdown: when is the env
  rewrite worth it, given it unblocks both GPU-scale throughput and future search?

## Related Documents

- `docs/first_iteration_design.md` — environment boundary, observation shape, action
  space, and the original PPO/transformer target.
- `docs/bootstrap_strategy.md` — teacher bootstrap vs cold self-play data strategy.
- `docs/cpu_self_play_roadmap.md` — current CPU loop state and what is empirically left.
- `docs/goals.md` — project-level goals and throughput/GPU constraints.
