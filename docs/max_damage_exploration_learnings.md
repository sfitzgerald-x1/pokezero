# Beat-max-damage exploration — learnings

Goal: a self-play-trained neural policy that beats the `max-damage` baseline **>80%** in Gen 3
random battles. This doc records what an overnight exploration sweep tried, what it achieved, and
the decisions that came out of it. `max-damage` is **eval-only** (never trained against directly).

## Starting point

The earlier agent sat **flat at ~0.24 vs max-damage** while already beating random/simple — i.e.
the wall was never the weak baselines; it was learning the signal that punishes a rigid maximizer.

## What was tried (overnight sweep, `runs/max-damage-goal-local-20260625/`)

All on top of the merged observation overhaul (#217):

- **Aggressive scripted teacher** — status-pressure scoring, an active-danger switch bonus, and a
  deterministic tie-break, to demonstrate/train against a stronger, more rigid opponent.
- **DAgger** (dataset-aggregation imitation) across several rounds vs the aggressive baseline.
- **Objectives** — `reward-weighted` (filtered) behavior cloning, and **PPO warm-started from the
  DAgger model**, alongside plain BC.
- **Switch-focused auxiliary losses** — upweighted switch-action CE, a move-vs-switch
  "action-family" head, a conditional switch-target head, and family-gated action selection.
- **Precomputed "matchup features"** on the action tokens — type effectiveness, STAB, expected
  power (bp×eff×STAB×acc), and estimated damage fractions (observation slots 38–47).

## Results (80 games per matchup unless noted)

| Run | vs max-damage | vs random | vs simple |
|---|---|---|---|
| ppo-from-dagger2-lowlr-512 (iter 4) | **0.525** | ~0.99 | ~0.96 |
| ppo-from-dagger2-lowlr-512 (iter 1→3) | 0.500 → 0.500 → 0.512 | | |
| switch-status-bc (160 games) | 0.519 | | |
| matchup-features-direct-bc-512 | 0.512 | | |
| danger65-bc-512 | 0.487 | 0.98 | 0.89 |
| expected-power-status75-bc-512 | 0.475 | 1.00 | 0.89 |
| matchup-features-direct-bc-1024 | 0.475 | 0.95 | 0.91 |
| ppo-from-matchup-bc-512-vs-aggressive | 0.463 | 0.99 | 0.91 |

## Key findings

1. **Real progress, but a plateau, not a breakthrough.** vs-max-damage moved from ~0.24 to a tight
   **~0.46–0.52 band across ~20 varied methods**. When that many different approaches converge to
   the same number, it's a ceiling — not active climbing. (vs random/simple at 0.9–1.0 is table
   stakes; it is not a progress signal and should not be read as one.)
2. **The ceiling is the imitation ceiling.** The scripted teacher itself only beats max-damage
   ~0.57, and **you cannot out-clone your teacher.** Imitation (BC/DAgger) caps the policy near the
   teacher's strength; the models land slightly *below* it (~0.50). Exceeding max-damage requires
   *exceeding the teacher*, which only RL discovering exploitation (or search) can do.
3. **PPO-from-DAgger is the only line that could break the ceiling — and it barely moved.**
   0.500 → 0.525 over 4 iterations is **inside the ±5.6% noise of an 80-game eval**, and still
   below the teacher. Inconclusive: needs more iterations *and* a larger eval (≥300–400 games) to
   tell a real climb from noise.
4. **The precomputed matchup features did not help.** The runs with effectiveness/STAB/expected-
   power/estimated-damage features (`matchup-features-*`, `expected-power-*`) did **not** beat the
   plain raw-facts DAgger/PPO runs. This is direct evidence *for* the raw-facts principle: the
   shortcut added brittleness (it ignores ability/item immunities) without buying accuracy.

## Decisions

- **Removed the matchup-feature precompute** from the observation (slots 38–47 and their helpers),
  restoring the clean raw-facts encoding from #217. The principle is now stated as a **hard rule**
  in `docs/observation_input_shape.html`: no precomputed type effectiveness, STAB, expected power,
  damage estimates, or matchup/threat summaries — the model must learn these from raw facts. The
  removed code is preserved in this branch's history (the pre-cleanup snapshot commit).
- **Kept the training-method work** (aggressive teacher, DAgger, reward-weighted/PPO objectives,
  switch/action-family/switch-target auxiliary losses, family-gated selection, aggressive-damage
  baseline). These *teach* the model from the raw observation — the opposite of precomputing — and
  are the part that moved 0.24 → ~0.50.

## Next directions

1. **Make the RL line measurable.** Run PPO from the best DAgger init for ~10–20 more iterations
   with vs-max-damage eval at ≥300–400 games. Decision rule: if it doesn't clearly clear ~0.55–0.60
   in that window, "imitate-an-aggressor then PPO" is capped.
2. **If capped, pivot to search.** AlphaZero-style MCTS over the real Showdown simulator, using the
   belief engine to determinize hidden info. Higher ceiling than a single-pass policy, and it
   reuses two assets we already have (a fast perfect sim + the belief engine). This — not a richer
   precomputed observation — is the most promising path to actually exceeding max-damage.
