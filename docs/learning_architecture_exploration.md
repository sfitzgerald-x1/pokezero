# Learning Architecture Exploration

This document is a working brainstorm for the *post-linear* learning architecture:
the model and training algorithm that should replace the dependency-free linear
softmax baseline once the CPU harness is trusted. AlphaGo Zero / AlphaZero is the
stated inspiration, so this doc starts from that lineage and then adapts it to the
ways Gen 3 random battles differ from Go.

Status: exploration only. Nothing here is committed implementation. The intent is
to agree on a staged path and a first concrete neural training target.

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
- **Stage 1 — neural representation + model-free RL.** Wire the existing entity-token
  transformer into the self-play loop with PPO/V-trace, value head trained to game
  outcome, opponent-action aux on the shared trunk. Reuse promotion/benchmark/audit.
  Keep it CPU-smoke-able; scale on GPU. **Highest value, lowest risk; unblocks real
  representation learning.**
- **Stage 2 — strengthen representation.** Pointer action head, belief-conditioned
  opponent tokens, longer/recurrent temporal context, belief-prediction aux loss.
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

## Infrastructure Implications

- **Forward-model speed is the real gate for any search.** A subprocess-per-step
  simulator cannot back MCTS. Before Stage 3/4, we likely need an in-process or batched
  Gen 3 engine, or a learned dynamics model. Worth scoping early even though Stage 1
  doesn't need it.
- **The GPU path is unvalidated** (`first_iteration_design.md`, "Not implemented yet").
  Neural RL needs it for scale; Stage 1 should include a validated GPU training path.
- **Consumer-GPU target** constrains model size (`embedding_dim`, layers, heads) — keep
  the architecture sweepable and small by default.
- **Inference must stay fast** to preserve high-throughput self-play (`docs/goals.md`).
- Reward shaping, the 250-turn cap, and capped-game scoring are already specified and
  carry over unchanged.

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

## Related Documents

- `docs/first_iteration_design.md` — environment boundary, observation shape, action
  space, and the original PPO/transformer target.
- `docs/bootstrap_strategy.md` — teacher bootstrap vs cold self-play data strategy.
- `docs/cpu_self_play_roadmap.md` — current CPU loop state and what is empirically left.
- `docs/goals.md` — project-level goals and throughput/GPU constraints.
