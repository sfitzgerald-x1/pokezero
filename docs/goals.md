# Goals

PokeZero aims to train a model that learns to play Pokémon Showdown Gen 3 random battles **entirely
on its own, from first principles, through self-play** — with the ultimate aim of being competitive
on the live ladder. AlphaGo Zero / AlphaZero is the inspiration: strength should emerge from the
agent playing itself, not from imitating a teacher or from hand-built strategy.

## Project Goals

- Train through self-play, where agents repeatedly battle each other and learn from the resulting games — the agent discovers strategy on its own, not by copying a teacher.
- Use temporal context so previous turns bias future predictions and decisions.
- Keep turns relatively fast so self-play can generate a high volume of games, even though game length is nondeterministic.
- Add a provisional 250-turn cap to discourage stalled play and losing-position turn cycling.
- Use parallel self-play collection to increase sample generation throughput.
- Prefer an eventual model and runtime that can run on smaller consumer-grade GPUs when feasible.

## Non-Goals

- **Imitation as the source of strategy.** The scripted teacher, behavior cloning, and DAgger are at
  most optional early-signal scaffolding and are not the path to the final policy. The agent should
  ultimately reach strength without them.
- **Engineering an opponent roster.** Strength should come from self-play (the agent and its past
  selves), not from curating a set of fixed opponents to train against. The fixed baselines
  (`random-legal`, `simple-legal`, `max-damage`, `aggressive-damage`) are evaluation yardsticks, not
  training targets.

## Open Questions

- Is the provisional capped-game penalty strong enough, or should capped games become a stronger double loss or explicit stall penalty?
- How much temporal context should be encoded in the model versus supplied by the environment?
- What throughput target is acceptable beyond the initial single-machine parallel collector?
- Which consumer GPU class should be treated as the target constraint?
