# First Iteration Design

PokeZero's first iteration targets Gen 3 random battles with player-knowable observations, randbat-specific belief tracking, fast policy inference, and self-play refinement.

## Current Status

As of June 2026, the repo has moved from design exploration into a working CPU-first harness. The implemented stack can run local Gen 3 random-battle games through a Showdown BattleStream bridge, normalize each side into player-relative observations, collect trajectories, train a small dependency-free linear policy baseline, evaluate checkpoints, iterate self-play runs, resume interrupted runs, and inspect run manifests.

Implemented:

- Fixed 9-action Gen 3 singles action space with move and switch slots.
- Player-relative Showdown normalization where `self_*` and `opponent_*` are stable regardless of raw `p1` or `p2` seat.
- Local Showdown environment backed by the built Pokemon Showdown simulator.
- Rollout collection to JSONL with terminal outcome, capped-game marker, opponent action metadata, and policy identifiers.
- Dataset streaming with left-padded temporal windows for training examples and batches.
- Random legal and simple legal baseline policies.
- CPU-only masked linear softmax baseline with behavior-cloning and reward-weighted objectives.
- Linear checkpoint save/load with schema compatibility checks.
- Baseline rollout benchmarking and checkpoint benchmarking.
- Self-play iteration harness with current-policy-only training data, frozen historical opponent checkpoints, checkpoint warm starts, per-iteration manifests, resumable runs, parallel collection workers, and run reporting.
- Source-backed Gen 3 randbat belief sidecar for local battle inspection from public information.

Partially implemented:

- Temporal context exists in dataset windows and linear feature hashing, but the end-state model has not been implemented.
- Belief tracking exists as a sidecar/debug system and observation feature source, but it is not yet a fully trained model input in a neural policy loop.
- Evaluation exists against fixed baselines and historical checkpoints, but benchmark gates and long-run experiment criteria are still informal.
- Capped games are recorded and surfaced in reports, but capped-game reward/scoring policy is still unresolved.

Not implemented yet:

- Transformer/entity-token policy model.
- Value head, opponent-action auxiliary head, or PPO-style online actor-critic training.
- GPU training path.
- Large-scale experiment orchestration across multiple machines.
- Formal benchmark thresholds for promotion, regression detection, or checkpoint pool curation.

## Deviations From Original Plan

The design below still describes the target direction, but the implementation deliberately inserted a dependency-free linear-policy phase before the transformer/PPO phase. This was not intended as the final learning algorithm. It validates the environment boundary, observation serialization, trajectory format, checkpoint compatibility, self-play loop, resume behavior, parallel collection, and reporting before adding a heavier training framework.

The Gen 3 belief work also started as a read-only sidecar and deterministic public-information engine rather than being immediately embedded into a learned policy. That has been useful for validating randbat set inference and player-relative state without coupling early training to belief-model complexity.

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

Opponent belief features should include possible ability, item, and move masks; revealed move masks; surviving set count; and a compact uncertainty scalar. The actor input is restricted to player-knowable state.

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

The first model should use a small transformer over entity, action, and history tokens.

Inputs:

- categorical ids for species, moves, items, abilities, statuses, weather, and token types
- numeric features for HP fractions, boosts, turn progress, set counts, and uncertainty
- attention mask
- legal action mask

Outputs:

- policy logits over the 9 action slots
- scalar value estimate
- opponent-action prediction logits for auxiliary training

The opponent-action prediction head should predict the opponent's chosen move or switch slot from the same information state. This auxiliary task should help train temporal prediction and belief use.

## Training Approach

V0 should use online actor-critic training with PPO-style updates. Training starts with a bootstrap phase against fixed opponents:

- random legal policy
- simple legal-action policy with basic switch participation
- frozen early checkpoints once the first policy can complete games reliably

Once the policy can reliably complete games, self-play should use a pool of frozen checkpoints. This keeps training pressure broader than the current policy matchup without baking damage-calc or handcrafted battle heuristics into the main training loop.

Trajectory records should include observations, legal masks, selected actions, opponent actions, rewards, terminal outcome, capped-game marker, and checkpoint/opponent identifiers.

## Rewards And Termination

Games end normally through the simulator or at a provisional 250-turn cap. Capped games must be recorded distinctly from normal wins and losses.

V0 should use terminal win/loss reward plus lightweight shaping to reduce sparse-credit problems:

- faint differential
- HP fraction differential
- small penalty for turns consumed after the cap-progress threshold

The exact capped-game training reward remains an experiment. Candidate treatments are tie, double loss, or explicit stall penalty.

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
