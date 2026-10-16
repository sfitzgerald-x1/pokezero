# Root-prior exactness: the native search root reproduces the policy's observation

Date: 2026-10-16. Scope: the incumbent engine search on main, gen3randombattle, champion
checkpoint `v4prod-enthalf-15m-20260807084346-iteration-9375`.

## Question

The search-over-raw exploration panels (r3, r4, r5 of October 15 and 16) recorded, for 268
roots, both the raw policy's masked distribution and the native search's root prior over the
same legal actions. The two disagreed on the argmax on 53 roots (20 percent), with total
variation 0.42 on disagreement roots, and on 46 of the 54 search deviations from raw the search
had simply followed its native prior. Either the native root encoding on main differs from the
Python observation, or the collector reconstructed those roots differently from the live
boundary. This note settles which.

## Method

Two tools, both on main and both replaying the exact source games from their seeds:

- `scripts/regen_root_corpus.py` replays seeded raw-versus-raw source games with the same env
  configuration and deterministic masked-argmax champion the exploration collector uses, and
  writes them as a golden corpus (public materialization payload, true teams, observation
  arrays, fold rows). It matches each archived exploration root to its regenerated row by
  observation-array equality.
- `scripts/live_root_prior_check.py` replays the same games and, at every p1 boundary, also
  runs the production `EngineMctsPolicy` (model priors, model leaf, override telemetry) under
  three world variants without touching the game: belief-sampled four worlds (the collector's
  configuration), belief-sampled one world, and a fixed override with the true teams. It compares
  each variant's root prior with the raw policy's distribution.

The existing `scripts/leaf_root_parity.py` then byte-diffs the depth-0 native encoding built
from each regenerated row's payload and true teams against the recorded Python arrays.

## Results

Regeneration is exact. All 276 archived roots of the three runs (62, 152, 62) matched a
regenerated row byte for byte, over 3,607 regenerated decisions in 42 games.

Depth-0 parity over the regenerated corpora:

| Corpus | Rows | Constructible | Exact | Divergent | Skipped (world unsupported) |
| --- | ---: | ---: | ---: | ---: | ---: |
| r3 (9 games) | 821 | 701 | 680 | 21 | 120 |
| r4 (24 games) | 2,030 | 1,577 | 1,570 | 7 | 453 |
| r5 (9 games) | 756 | 664 | 664 | 0 | 92 |

None of the 53 disagreement roots is divergent. Two of the 28 divergent rows are archived roots
(`exploration:1612591941:2` and `:5`, rounds 9 and 10 of that game); both are agreement roots in
the collector's receipts. The 28 rows are one family: action-block columns (`CATEGORY_PRIMARY`,
`NUMERIC_MOVE_PP_FRACTION`, `NUMERIC_PRESENT`, the legal mask) on p2 rows 5 to 16 of game
2057574279 and seven p1 rows of game 1612591941; an existing move-surface class, recorded here
for the fidelity ledger and not pursued. The skips are worlds the constructor refuses from the
payload (`materialization_blocker`, `pending_baton_pass`, `volatile_unsupported`,
`substitute_health_unknown`); the live search falls back on those roots.

Live root prior versus raw policy, game 1505790335 (the game holding two of the twelve worst
disagreement roots, the asleep Girafarig choosing between rest and psychic):

| Variant | p1 decisions | Argmax agreement | Max total variation |
| --- | ---: | ---: | ---: |
| sampled, 4 worlds | 52 | 52 | 0.0000 |
| sampled, 1 world | 52 | 52 | 0.0000 |
| true teams | 52 | 52 | 0.0000 |

At the two Girafarig roots the live native prior is `rest 0.9813` and `rest 0.9789`, equal to the
Python prior to four decimals; the collector's receipts for the same roots recorded a native
prior of `psychic 0.97` and `psychic 0.96`.

Panel-wide live check, all 42 games, 1,801 p1 decisions (one is a forced single-action root,
for which the native side reports no mappable arm and which counts as exact):

| Variant | Measured | Argmax agreement | Exact (TV ≤ 0.001) | Max TV | Games with any TV > 0.001 |
| --- | ---: | ---: | ---: | ---: | ---: |
| sampled, 4 worlds | 1,801 | 1,798 | 1,649 | 0.48 | 11 |
| true teams | 1,801 | 1,799 | 1,651 | 0.09 | 11 |

For comparison, the collector's own receipts on the 268 archived roots: argmax agreement 215,
median TV on disagreement roots 0.42.

Residuals in the live path, all small and none matching the collector's pattern:

- Two near-tie flips (seeds 945994609 round 30 and 1122319173 round 25): the Python policy's
  top two actions sit within 0.03 and 0.006 of each other and the native prior moves them by
  about 0.05, identically under sampled and true-team worlds. The same small shift, at most 0.09
  TV, appears on about 150 decisions across 11 games without changing the argmax. It is
  independent of the world sample, so it is an encoding or numeric nuance of the root row, not
  hidden state.
- Game 1612591941, rounds 9 to 17: under belief-sampled worlds two native arms carry no action
  index and the remaining mass is renormalized, giving TV up to 0.48 and one argmax flip (round 9,
  archived root `exploration:1612591941:2`, an agreement root in the receipts); under the true
  teams the same rows agree on the argmax with TV at most 0.017. These are the same seven rows the
  parity gate flags in the action-block columns. An existing move-surface class on that game,
  visible to both tools.

## Conclusion

On main, the native search root is, to within the small residuals above, a function of the
policy's own observation: the belief sample does not reach the root encoding except through the
move-surface class on one game, and the depth-0 encoder reproduces the Python arrays on every
disagreement root. The 20 percent root-prior disagreement in the exploration
panels, and therefore the search-versus-raw contrasts measured so far, are collector-side, not a
property of main's engine. The collector ran from branch commit `5752cd32`, which is not an
ancestor of main and whose `engine_search.py` and `local_showdown.py` differ from main by several
hundred lines each, and it selected from an archived reconstruction of the root (the actor-timed
job rebuilt from the sealed snapshot) rather than at the live boundary. This note does not
separate those two; the live check does, once run against that branch. The raw selector in the
same runs consumed a faithful observation, which is why its priors match the live ones. The live
check and the receipts read the same quantity, `engine_mcts.override.root_allocation.arms[].model_prior`,
and the live check used the r4 entry's own TorchScript export, encoder tables, checkpoint and
environment configuration digest.

Any collector that selects actions away from the live boundary, or from a branch that modifies
the engine path, must pass this gate before its results are read: total variation at or below
0.001 between its native root prior and the live `EngineMctsPolicy` root prior on every root of
a fresh panel, measured with `scripts/live_root_prior_check.py`, whose exit status enforces it.
The simplest way to pass it is to select at the live boundary, as the check does, and run
continuations from the sealed snapshot afterwards.

The live numbers above were produced with the belief sampler reseeded from the game seed at
every boundary (`--world-rng per-decision`, the reproducible default), so the sampled variant
draws the same worlds at every root of a game; the true-team variant, which samples nothing,
corroborates it. All four live-check shards were produced by the committed script; earlier shards from
pre-commit revisions, which differed only in crash handling and output format, gave the same
per-game counts.

## Commands

```
PYTHONPATH=src python scripts/regen_root_corpus.py --checkpoint <ckpt.pt> --showdown-root <showdown> \
    --run-dir <exploration run dir> --seeds <seeds...> --out <corpus>
PYTHONPATH=src:scripts python scripts/leaf_root_parity.py --corpus <corpus> --tables <encoder_tables.json> --json <report>
PYTHONPATH=src python scripts/live_root_prior_check.py --checkpoint <ckpt.pt> --model-ts <model_ts.pt> \
    --tables <encoder_tables.json> --showdown-root <showdown> --seeds <seeds...> --out <report>
```

Unit tests for the pure helpers: `python -m unittest tests.test_root_prior_exactness_tools`.
