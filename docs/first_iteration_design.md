# First Iteration Design

PokeZero's first iteration targets Gen 3 random battles with player-knowable observations, randbat-specific belief tracking, fast policy inference, and self-play refinement.

## Current Status

As of June 2026, the repo has moved from design exploration into a working CPU-first harness. The implemented stack can run local Gen 3 random-battle games through a Showdown BattleStream bridge, normalize each side into player-relative observations, collect trajectories, train a small dependency-free linear policy baseline, evaluate checkpoints, iterate self-play runs, resume interrupted runs, and inspect run manifests.

Implemented:

- Fixed 9-action Gen 3 singles action space with move and switch slots.
- Player-relative Showdown normalization where `self_*` and `opponent_*` are stable regardless of raw `p1` or `p2` seat.
- Local Showdown environment backed by the built Pokemon Showdown simulator.
- Rollout collection to JSONL with terminal outcome, capped-game marker, opponent action metadata, and policy identifiers.
- Dataset streaming with left-padded temporal windows, terminal-winner-derived returns, and batch containers for training examples.
- Random legal, simple legal, and initial scripted-teacher baseline policies.
- CPU-only masked linear softmax baseline with behavior-cloning and reward-weighted objectives.
- Linear auxiliary opponent-action prediction head trained from recorded opponent move/switch labels.
- Linear checkpoint save/load with version-tag compatibility checks.
- Baseline rollout benchmarking and checkpoint benchmarking.
- Scripted-teacher bootstrap workflow that collects teacher-only train/validation rollouts, includes teacher-mirror states by default, runs strict-teacher preflight, trains a linear behavior-cloning checkpoint, benchmarks it, and records a manifest.
- Self-play iteration harness with current-policy-only training data, held-out validation data, frozen historical opponent checkpoints, checkpoint warm starts, per-iteration manifests, resumable runs, parallel collection workers, auto-promotion, and run reporting.
- Configurable promotion gate CLI over bootstrap and self-play manifests using per-opponent benchmark win rates, incumbent-delta checks, minimum game counts, capped-game rates, and teacher-degradation counters.
- Promotion registry verification that checks registry sequence integrity, promoted checkpoint existence, embedded passing gate results, and stored artifact checksums.
- Append-only promotion registry for recording gate-passing checkpoints, optionally copying them into a managed artifact directory, defaulting incumbent gates to the latest promoted policy, refreshing promoted self-play opponents during long runs, and filtering historical opponents to promoted checkpoints.
- Source-backed Gen 3 randbat belief sidecar for local battle inspection from public information.
- Compact public-belief observation features for revealed opposing Pokemon, including surviving candidate count, uncertainty, possible ability/item/move counts, bucketed per-value possible-fact features, and revealed ability/item flags.

Partially implemented:

- Temporal context exists in dataset windows, linear feature hashing, and the transformer scaffold; PPO-style online training is not implemented.
- Belief tracking is wired into the observation path as compact summaries, but full explicit ability/item/move masks and neural-policy-specific belief embeddings are not implemented yet.
- Opponent-action prediction exists in the linear baseline and transformer scaffold as an auxiliary supervised head.
- A PyTorch-backed entity-token transformer scaffold exists behind the optional `neural` extra, including `neural:<checkpoint>` policy-spec loading, a neural benchmark CLI, and a first neural self-play iteration command that trains transformer checkpoints from accumulated rollout records with promotion-gated collector advancement, warm starts, resume support, and optional promotion-registry artifact management.
- Evaluation exists against fixed baselines and promoted historical checkpoints, with configurable absolute-floor and incumbent-delta promotion gates plus a promotion registry; long-run experiment criteria are still informal.
- CPU-only run audit CLI over linear and neural self-play manifests that checks latest benchmark health, capped-game rates, average decision-round length, same-opponent benchmark regression from previous best, and trailing promotion failures.
- Optional post-iteration audit enforcement for linear and neural self-play runs, so long CPU experiments can stop after a bad manifest is written instead of continuing unattended.
- Capped games are recorded and surfaced in reports; self-play CLI training now defaults them to a mild double-loss return.

Known limitations:

- Checkpoint compatibility is guarded by hand-maintained schema/version tags, not content-derived feature fingerprints. Feature changes still need deliberate version bumps.
- Parallel collection caches immutable linear models per collection call, but larger checkpoints and high worker counts still need memory profiling before long unattended runs.
- Held-out validation metrics measure imitation fit against rollout labels, not policy strength. Benchmark win rate and capped-game rate remain the quality signals for promotion decisions.
- Neural iteration can use the shared promotion registry/gate path, and run-level audits can flag obvious benchmark/capped-rate/promotion regressions; useful long-run threshold settings still need empirical validation.
- The scripted teacher uses local Showdown dex metadata plus first-pass context heuristics for utility moves and safer switching. It is a bootstrap data source, not the intended long-term policy, and it still lacks hazards and deeper sequence planning.
- Current observation belief features are compact bucketed facts and counts, not full explicit masks. Detailed candidate variants and evidence logs remain sidecar-only to avoid bloating every trajectory record.

Not implemented yet:

- PPO-style online actor-critic training.
- Validated GPU training path.
- Large-scale experiment orchestration across multiple machines.
- Empirically validated long-run benchmark thresholds or richer managed checkpoint lifecycle tooling.

## Deviations From Original Plan

The design below still describes the target direction, but the implementation deliberately inserted a dependency-free linear-policy phase before the transformer/PPO phase. This was not intended as the final learning algorithm. It validates the environment boundary, observation serialization, trajectory format, checkpoint compatibility, self-play loop, resume behavior, parallel collection, and reporting before adding a heavier training framework.

The current docs now separate the cold self-play baseline from the imitation-bootstrap path in `docs/bootstrap_strategy.md`. The implementation supports both at the harness level: cold runs can start from `random-legal`, while bootstrap runs can generate a scripted-teacher checkpoint through `bootstrap_cli teacher`, then start self-play from that checkpoint via `--initial-policy linear:<checkpoint>` and carry held-out validation JSONL with `--validation-data`.

The Gen 3 belief work also started as a read-only sidecar and deterministic public-information engine rather than being immediately embedded into a learned policy. Compact belief facts and counts are now included in `PokeZeroObservationV0`; detailed candidate variants and evidence logs remain sidecar-only until a neural input format needs explicit masks or richer set-branch embeddings.

Parallelization has started at the single-machine collection layer with `--workers`. It is not yet a distributed rollout system.

## Goals

- Build a fresh PokeZero training stack. Metamon is prior art for observation and evaluation ideas; PokeZero owns its implementation.
- Start with Gen 3 random battles before expanding to other formats.
- Preserve uncertainty by exposing only information a player could know.
- Use Gen 3 random battle set data to track possible opponent realities.
- Keep inference fast enough to support high-throughput rollouts.
- Use temporal context so previous turns influence future predictions and decisions.

## Environment Boundary

The simulator owns complete battle state, including hidden teams, unrevealed moves, items, and abilities. The policy receives an information state derived from public events, its own side, legal choices, and a deterministic belief tracker.

The environment boundary is responsible for converting Showdown protocol state into a player-relative view. Raw Showdown seats such as `p1` and `p2` are transport/provenance metadata, not the model's primary frame. For any call to `observe(player)`, `self_*` tokens must always describe the observing player, `opponent_*` tokens must always describe the other side, and the legal-action mask must describe only that player's current request.

This normalization is mandatory for self-play and two-bot evaluation. The same battle position observed by opposite players should produce mirrored player-relative observations, not inputs where the meaning of `p1` and `p2` changes under the model. Raw protocol identifiers may be retained in debug metadata so selected actions can still be submitted back to the correct Showdown side.

The first simulator integration should be a thin PokeZero environment API over Showdown-compatible battle execution:

- `reset(seed, format)` starts a Gen 3 random battle.
- `observe(player)` returns `PokeZeroObservationV0`.
- `legal_actions(player)` returns the 9-action mask.
- `step(actions)` submits simultaneous player choices and advances the battle.
- `terminal()` reports normal win/loss or capped-game termination.

The environment must model Pokemon as a simultaneous-choice game. On standard turns, both players choose before resolution. On forced-switch or asymmetric request turns, the legal-action mask should expose only the choices available to the requested side.

The environment contract should allow both blocking and async adapters. A local subprocess backend can implement the blocking protocol; an event-loop-backed client can implement the async protocol with the same observation and step semantics.

## Belief Tracking

The belief tracker is part of the environment. It updates from public battle events and known Gen 3 random battle set data. It should track candidate opponent realities per revealed slot, including possible moves, abilities, and items.

Example: if an opposing Arcanine can have either Intimidate or Flash Fire, and no Intimidate event occurs when it enters, the tracker collapses the ability belief toward Flash Fire and exposes that reduced belief on later turns.

V0 should treat belief tracking as an explicit engineering workstream. The implementation must extract or mirror the Gen 3 randbat set-generation data needed to produce candidate sets and probability features.

## Observation Shape

The first observation format, `PokeZeroObservationV0`, is a fixed-shape structured token package:

- `field_token`: turn number, weather, hazards, screens, request type, forced-switch flag, and turn-cap progress.
- `self_pokemon_tokens[6]`: own team state, including species, HP, status, boosts, item, ability, moves, and active/fainted flags.
- `opponent_pokemon_tokens[6]`: visible opponent state plus belief features for revealed or inferred slots.
- `action_candidate_tokens[9]`: four move slots and five switch slots, each carrying the semantics of the currently available option.
- `recent_event_tokens[24]`: compact public events from recent turns, padded when fewer events are available.
- `legal_action_mask[9]`: valid choices for the current request.

Opponent belief features currently include bucketed per-value features for possible abilities, items, and moves; possible-fact counts; revealed move count; surviving set count; revealed ability/item flags; and a compact uncertainty scalar. A later neural observation format can replace these bucketed facts with explicit masks or set-branch embeddings. The actor input is restricted to player-knowable state.

Token feature widths are uniform for batching. Token sections may use different subsets of the categorical and numeric columns; unused columns are padding.

Observation builders must use deterministic player-relative ordering:

- self team tokens follow the observing player's canonical team order from the request side.
- opponent tokens follow stable revealed-slot order, with padded unknown slots when fewer than six opposing Pokemon are known.
- action candidate tokens describe the observing player's current legal move and switch candidates.
- recent event tokens should be normalized to self/opponent actor roles where possible, instead of requiring the model to learn whether `p1` or `p2` is itself on that turn.

## Action Space

Gen 3 singles random battles use a fixed 9-action policy head:

- `0..3`: active move slots in Showdown request order.
- `4..8`: switch slots in canonical team order, excluding the active Pokemon position.

Invalid moves, disabled moves, unavailable switch slots, trapped switch options, and fainted switch targets are masked by `legal_action_mask`.

The policy head is positional over the 9 slots. Action candidate tokens provide the current slot semantics, so the model can evaluate the current move or switch occupying each position.

Switch actions are dense candidates over the non-active team members in canonical team order. A switch logit is meaningful through its action-candidate token, which identifies the target Pokemon for that request.

## Model Shape

The first neural model should use a small transformer over entity, action, and history tokens. The initial PyTorch scaffold lives behind the optional `neural` extra so the base rollout/evaluation harness stays lightweight.

Inputs:

- categorical ids for species, moves, items, abilities, statuses, weather, and token types
- numeric features for HP fractions, boosts, turn progress, set counts, and uncertainty
- attention mask
- legal action mask

Outputs:

- policy logits over the 9 action slots
- scalar value estimate
- opponent-action prediction logits for auxiliary training

The current linear baseline ships an optional supervised opponent-action auxiliary head trained from recorded opponent move/switch labels, off by default. Because the linear policy and aux head use independent weights over fixed hashed features, the head shares no representation with the policy and cannot influence play in the linear baseline — it is scaffolding that validates the label pipeline and metrics. The transformer scaffold keeps this head and predicts the opponent's chosen move or switch slot from the same information state; once wired into training and self-play, it will share the trunk representation so the auxiliary task can shape temporal prediction and belief use.

## Training Approach

V0 should use online actor-critic training with PPO-style updates. Training starts with a bootstrap phase against fixed opponents:

- random legal policy
- simple legal-action policy with basic switch participation
- scripted-teacher bootstrap policy
- frozen early checkpoints once the first policy can complete games reliably

Once the policy can reliably complete games, self-play should use a pool of frozen checkpoints. This keeps training pressure broader than the current policy matchup without baking damage-calc or handcrafted battle heuristics into the main training loop.

Trajectory records should include observations, legal masks, selected actions, opponent actions, rewards, terminal outcome, capped-game marker, and checkpoint/opponent identifiers.

## Rewards And Termination

Games end normally through the simulator or at a provisional 250-turn cap. Capped games must be recorded distinctly from normal wins and losses.

V0 should use terminal win/loss reward plus lightweight shaping to reduce sparse-credit problems:

- faint differential
- HP fraction differential
- small penalty for turns consumed after the cap-progress threshold

The self-play CLI currently uses a provisional capped-game return of `-0.25` for both players. Stronger double-loss or explicit stall penalties remain experiments.

## Evaluation

Progress should be measured against fixed benchmark opponents and historical self-play checkpoints.

V0 evaluation should include:

- random legal policy
- simple legal-action baseline
- frozen historical PokeZero checkpoints

Each evaluation run should report win rate, average turns per game, capped-game rate, average decision latency, and throughput in completed games per hour.

Harness validation should include replay-style cases that prove side ownership is normalized correctly:

- a bot configured with one default seat but actually seated as the other side still observes itself under `self_*`.
- two bots observing the same turn receive mirrored `self_*` and `opponent_*` sections.
- selected policy actions are translated back to the detected Showdown seat before submission.
- hidden opponent state never appears in the normalized observation.

## Open Questions

- What capped-game reward works best once self-play begins?
- What throughput target should define a useful first rollout loop?
- What consumer GPU class should constrain the first model size?
- How much reward shaping is enough before it starts distorting the win objective?
- Which non-handcrafted baseline should be the first "must beat" milestone?
