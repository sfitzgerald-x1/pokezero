# Goals

PokeZero aims to train a model to play Pokemon Showdown Gen 3 random battles through repeated self-play.

## Project Goals

- Train through self-play, where agents repeatedly battle each other and learn from the resulting games.
- Use temporal context so previous turns bias future predictions and decisions.
- Keep turns relatively fast so self-play can generate a high volume of games, even though game length is nondeterministic.
- Add a provisional 250-turn cap to discourage stalled play and losing-position turn cycling.
- Use parallel self-play collection to increase sample generation throughput.
- Prefer an eventual model and runtime that can run on smaller consumer-grade GPUs when feasible.

## Open Questions

- Is the provisional capped-game penalty strong enough, or should capped games become a stronger double loss or explicit stall penalty?
- How much temporal context should be encoded in the model versus supplied by the environment?
- What throughput target is acceptable beyond the initial single-machine parallel collector?
- Which consumer GPU class should be treated as the target constraint?
