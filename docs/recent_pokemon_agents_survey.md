# Recent Pokémon agent work (2024 – mid-2026) — survey notes

Status: reference notes, 2026-07-03. Companion to
[`no_human_data_selfplay_context.md`](no_human_data_selfplay_context.md) and
[`alphastar_training_context.md`](alphastar_training_context.md). Scope:
published agents/benchmarks for competitive Pokémon battling from the last
two years, read for what transfers to PokeZero's from-scratch randbats line.

## The headline event: the PokéAgent Challenge (NeurIPS 2025)

([site](https://pokeagent.github.io/), [paper, Mar 2026](https://arxiv.org/abs/2603.15563)) —
100+ teams, two tracks (battling + RPG speedrunning), now a standing
leaderboard. The battle-track results are a field-wide natural experiment on
method choice, and they broke cleanly:

- **RL and search agents beat LLM agents, consistently.** The paper calls
  battling "nearly orthogonal to standard LLM benchmarks." The generalist-LLM
  line (PokeLLMon-style) lost to specialists across the board.
- **Gen 9 OU was won by FoulPlay** — root-parallelized MCTS under imperfect
  information. Our external yardstick bot is the literal reigning
  NeurIPS-challenge champion in the hardest open format. Two implications:
  (a) the 3–9% win-rate band our arms sit in is measured against
  state-of-the-art search, not a weak heuristic — absolute numbers should be
  read accordingly; (b) search under hidden information demonstrably wins at
  the top of this domain, which supports the project's search phase in
  principle (E0 measures whether *our value heads* are ready for it, a
  different question).
- **Gen 1 OU was won by an extended Metamon RL baseline** (PA-Agent); 13 of
  16 qualifying slots went to teams extending public RL baselines.
- **Porygon2AI (#8, Gen 9 OU) used AlphaStar-style league training** —
  the league/population machinery has already been ported to Pokémon by an
  independent team and placed top-10 against the field. Evidence that the
  population-diversity direction is not speculative here.
- **Released resource:** checkpoints for **30 agents spanning the skill
  ladder** (compact RNNs to 200M-param transformers). A free graded
  evaluation gauntlet — a much finer strength ladder than our current
  max-damage → foul-play jump, usable without any training-data
  contamination.

## Metamon (RLC 2025 → 2026 updates)

([paper](https://arxiv.org/abs/2504.04395), [repo](https://github.com/UT-Austin-RPL/metamon),
[site](https://metamon.tech/)) — offline RL → self-play fine-tuning on gens
1–4, top 10% of active ladder players. Already the source of our WS-E
dense-shaping template. New-to-us findings worth stealing:

- **In-context opponent adaptation without search:** their sequence models
  "adapt to their opponent based solely on their input trajectory." That is
  the implicit-memory counterpart of our explicit belief engine — and their
  strength without any search at all is a data point that a good enough
  policy/representation carries far in this game.
- **Synthetic self-play data beat realistic data:** retraining on
  "intentionally unrealistic synthetic self-play datasets that do not
  attempt to recreate the teams and opponents seen in online battles"
  *improved* online performance against humans. Directly relevant to the
  no-human-data thesis: distribution-matching human play is not what made
  their agents strong, broad coverage was. This is our coverage-curriculum
  argument stated as their empirical result.
- Scale note: 5M human + 20M self-play trajectories, models to 200M params —
  one to two orders beyond our current runs. Their curves say capacity +
  data kept paying; our 3m-belief Pearson plateau at ~2m games may be a
  small-model ceiling, not a task ceiling.

## PokéChamp (ICML 2025 spotlight)

([paper](https://arxiv.org/abs/2503.04094), [repo](https://github.com/sethkarten/pokechamp)) —
minimax search where an LLM fills three modules: action sampling, opponent
modeling, and leaf value estimation. ~1300–1500 Showdown Elo (top 10–30%),
no LLM training. Lost to specialist RL/search in the challenge, so not a
strength template — but two resources matter:

- **Dataset: 3M+ real battles including 500k+ high-Elo games.** Under our
  human-data quarantine this is *opponent-model and evaluation* material:
  exactly what's needed to train the "what would a human play" world model
  for search-time opponent modeling — the one role human data is allowed in.
- **Skill-specific battle puzzles/benchmarks** — curated decision points
  testing particular competencies. The same idea as our ΔV/behavior probes,
  pre-built; worth importing as additional feature-specific probes rather
  than relying on aggregate win rate alone.
- Design echo: their explicit opponent-model module inside minimax is the
  architecture-level version of the quarantine — opponent model as
  environment, not teacher.

## VGC-Bench (AAMAS 2026)

([paper](https://arxiv.org/abs/2506.10326), [repo](https://github.com/cameronangliss/VGC-Bench)) —
doubles/VGC benchmark: 700k+ human battle logs, baselines spanning
heuristics, LLMs, behavior cloning, and **empirical game-theoretic
multi-agent RL (self-play, fictitious play, double oracle/PSRO)**. Key
result: in single-team mirror matches their agent **beat a professional VGC
competitor** — but as team diversity grows, the single-team winner degrades
and becomes *more exploitable*, while generalization-oriented training
trades peak strength for robustness.

That is our ecology/coverage finding measured independently in a sibling
format: **self-play equilibria are narrow, and narrowness = exploitability
once the opponent distribution widens.** It also establishes precedent for
PSRO-family methods in Pokémon specifically. Randbats blunts the
team-diversity axis (the generator diversifies teams for us — a structural
advantage of the format Scott's framing already identified), but the
strategic-diversity axis is the same problem.

## LLM-agent line (context only)

PokeLLMon ([2024](https://arxiv.org/abs/2402.01118)) reached "human parity"
in gen-8 randbats with in-context feedback + knowledge augmentation; later
zero-shot LLM evaluations ([Dec 2025](https://arxiv.org/abs/2512.17308))
added breadth. The challenge results settled this line's ceiling for now:
prompted general reasoning loses to specialized training in this domain.
One reusable nugget: PokeLLMon's documented *panic-switching* pathology
(consecutive defensive switches under threat) is a cheap behavioral probe we
could add alongside the hazard probes.

## What this changes for PokeZero

1. **Yardstick reframe:** foul-play is the reigning open-format champion.
   Beating it is a legitimately high bar; progress reads should say
   "vs SOTA search bot," and the 30-agent challenge ladder should be added
   as intermediate rungs (free, no training contamination).
2. **Method bet validated at both ends:** RL specialists won Gen 1, search
   won Gen 9 — our RL-foundation + search-topper plan occupies exactly the
   winning quadrant. The LLM line is not a competitor for strength.
3. **League precedent in-domain:** Porygon2AI's AlphaStar-style league
   placing #8 removes the "will population methods even work in Pokémon"
   uncertainty from the cross-arm/PSRO-lite plan.
4. **Coverage > distribution-matching, empirically:** Metamon's synthetic
   self-play result and VGC-Bench's diversity/exploitability trade-off both
   independently support the coverage-curriculum direction over imitating
   realistic play — consonant with the no-human-data constraint.
5. **Resources to import without violating the quarantine:** PokéChamp's
   500k high-Elo games (opponent world-model + eval), its skill puzzles and
   PokeLLMon's panic-switch pathology (feature-specific probes), the
   challenge checkpoint ladder (graded evaluation).
6. **Scale honesty:** the strongest published RL results run 10–100× our
   trajectory and parameter budget. Architecture sophistication saturated
   our value metric today, but *capacity + game volume* is the axis the
   field's results say keeps paying — relevant when weighing another
   architecture arm against simply longer/bigger runs.
