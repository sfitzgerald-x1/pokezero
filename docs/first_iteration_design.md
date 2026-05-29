# First Iteration Design

PokeZero's first iteration targets Gen 3 random battles with a self-play training loop, player-knowable observations, and fast policy inference.

## Goals

- Generate training games through repeated self-play.
- Preserve uncertainty by exposing only information a player could know.
- Use Gen 3 random battle set data to track possible opponent realities.
- Keep the action space compact and inference fast enough for high rollout throughput.
- Support temporal context so previous turns influence future predictions and decisions.

## Environment Boundary

The simulator owns complete battle state, including hidden teams, unrevealed moves, items, and abilities. The policy receives an information state derived from public events, its own side, legal choices, and a deterministic belief tracker.

The belief tracker is part of the environment. It updates from public battle events and known random battle set data. For example, if an opposing Arcanine can have either Intimidate or Flash Fire, and no Intimidate event occurs when it enters, the tracker collapses the ability belief toward Flash Fire and exposes that reduced belief on later turns.

The policy should learn strategy under uncertainty. The environment should maintain the legal, player-knowable uncertainty.

## Observation Shape

The first observation format, `PokeZeroObservationV0`, should be a fixed-shape structured token package:

- `field_token`: turn number, weather, hazards, screens, current request type, forced-switch flag, and turn-cap progress.
- `self_pokemon_tokens[6]`: own team state, including species, HP, status, boosts, item, ability, moves, and active/fainted flags.
- `opponent_pokemon_tokens[6]`: visible opponent state plus belief features for revealed or inferred slots.
- `action_candidate_tokens[9]`: four move slots and five switch slots, each carrying the semantics of the currently available option.
- `recent_event_tokens[16..32]`: compact public events from recent turns.
- `legal_action_mask[9]`: valid choices for the current request.

Opponent belief features should include possible ability, item, and move masks; revealed move masks; surviving set count; and a compact uncertainty scalar. The actor input should never include hidden simulator truth directly.

## Action Space

Gen 3 singles random battles use a fixed 9-action policy head:

- `0..3`: active move slots.
- `4..8`: switch slots.

Invalid moves, disabled moves, unavailable switch slots, trapped switch options, and fainted switch targets are masked by `legal_action_mask`.

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
- optional opponent-action prediction head for auxiliary training

The action head should score action candidate tokens so each logit is tied to the current move or switch option in that slot.

## Training Loop

Self-play workers generate trajectories containing observations, legal masks, selected actions, rewards, and terminal outcomes. Games end normally through the simulator or at a provisional 250-turn cap.

For the first iteration, capped games should be recorded distinctly from normal wins and losses. The exact training reward for capped games remains an experiment, with tie, double loss, and explicit stall penalty all viable candidates.

Parallel rollout workers should be explored early because game length is nondeterministic and throughput will depend on keeping many games in flight.

## Open Questions

- What reward should capped games receive during training?
- What initial event-history length gives enough temporal signal without slowing inference?
- What throughput target should define a useful first self-play loop?
- What consumer GPU class should constrain the first model size?
- Should opponent-action prediction be part of the first training objective or added after the policy/value loop works?
