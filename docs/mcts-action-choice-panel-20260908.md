# MCTS action-choice panel on corrected batched backup

Date: 2026-09-08. This is a small deterministic mechanics panel on the
corrected baseline that merged PR #1337 (`df4e3ce15ee69f922f6ae1b81c7b5e9861828319`).
It tests the real decision/chance traversal and the collect-then-finalize
schedule. It is not a full model-path timing run, a multi-world result, or a
playing-strength result.

## Contract

Each cell is evaluated for both acting seats and collection batches 1, 2, 8,
and 64. The opponent reply is restricted only where needed to make the
counterfactual explicit; it is not an opponent reply selected after observing
the search result. Nonterminal leaves use deferred rows, so a terminal branch
still bypasses the leaf callback exactly as it does in the production model
path. Each row reports the current visit-max action, completed root visits,
deferred leaf evaluations, terminal branches, the action's exact simple regret,
and its Q error against the declared payoff. Deferred leaf evaluations are the
actual callback work in this deterministic driver; they are reported separately
from visits rather than treating equal simulation counts as equal model work.

The same completed tree also feeds a shadow acting-seat Q-max selector. It
considers only visited legal actions, resolves exact Q ties by visits and then
stable action order, and never changes production selection. Its regret and
whether it selected a lower-visit arm are recorded separately.

## Declared fixtures and result

| Fixture | Known acting-seat payoffs | Fixed visits | Scheduler work: deferred leaves / terminal branches | Result for both seats and all batches |
| --- | --- | ---: | ---: | --- |
| Immediate win + deferred alternative | `seismictoss=1`, `toxic=0.5255`; opponent `splash` | 128 | 2 / 1 | Visit-max and shadow Q-max choose `seismictoss`; regret and selected-action Q error are 0. Toxic expands two deferred chance leaves; the terminal Toss has no leaf row. |
| Nested, variable two-turn value | `seismictoss→seismictoss=1`, `splash→seismictoss=0.75`; opponent `splash` | 256 | 5 / 1 | Both choose `seismictoss`, regret 0. Its finite-tree Q is below the solved 1.0 target, documenting remaining finite-work value error rather than hiding it. |
| Rare terminal decoy | `splash=0.5`, `tackle=0.25`; Tackle has 4 terminal KO rolls of 16 and controlled non-KO leaves at 0 | 256 | 2 / 1 | Both selectors choose `splash`; regret and Q error are 0. Thus Q-max does not win the panel merely by chasing a low-visit lucky KO. |
| Equal nonterminal control | `ember=0.5`, `tackle=0.5` | 128 | 16 / 0 | Visit-max and stable Q-max may choose different tied actions, but both have regret and Q error 0. |

The nested selected-action Q and its absolute error, identical up to floating
rounding for the two seat orientations, were:

| Batch | Selected Q | Exact target | Absolute error |
| ---: | ---: | ---: | ---: |
| 1 | 0.941817 | 1.0 | 0.058183 |
| 2 | 0.939860 | 1.0 | 0.060140 |
| 8 | 0.929777 | 1.0 | 0.070223 |
| 64 | 0.887006 | 1.0 | 0.112994 |

This is an important boundary: corrected backups retain the right action here,
but a fixed finite budget does not make every nested root Q exact. The panel
therefore reports value error rather than treating a correct recommendation as
proof of full convergence.

## One shadow-selector observation

One separate, predeclared harsh-prior row uses the same immediate-win fixture
with `toxic=0.7` and `seismictoss=0.3` priors, 128 visits, and batch 64. In
both seat orientations, visit-max selects `toxic` (simple regret `0.4745`),
whereas shadow Q-max selects `seismictoss` (regret `0`). The Q-max arm has 52
visits, fewer than the visit-max arm, so this is a real completed-visit lag,
not a tie-break artifact.

That is a useful mechanism observation, not a production recommendation. The
rare-decoy and equal controls pass, but the panel remains synthetic and
one-world.

## Raw Q-max is parked

The required deep noisy-continuation control was run on 2026-09-10. It uses no
terminal branches and declares the exact root payoffs in advance: `splash=0.55`
and `tackle=0.40`. With 64 visits, batch 1, depth 2, and fixed priors of 0.9 /
0.1, the reference leaf pass makes both visit-max and shadow Q-max choose
`splash` for both acting seats.

The control then changes only Tackle's deep, nonterminal continuation estimate
to 0.75. Visit-max still selects `splash`, with zero regret. Raw shadow Q-max
instead selects the 10-visit `tackle` arm at Q `0.632018`, despite its declared
payoff of `0.40`: simple regret `0.15` and Q error `0.232018` for both seats.

This falsifies raw acting-seat Q-max as a selector candidate at finite work. It
remains telemetry only and is not eligible for a representative-position or
head-to-head promotion. The harsh-prior terminal example is retained as a
diagnostic of visit lag, not evidence that Q-max improves MCTS play.

## Reproduction

Use the vendored pinned engine build described by the search test instructions,
then run:

```sh
PYO3_PYTHON="$SEARCH_TEST_PYTHON" CARGO_BUILD_JOBS=4 cargo test \
  --manifest-path rust/pokezero-search/Cargo.toml \
  predeclared_action_choice_panel_has_explicit_regret_and_value_error -- --nocapture

PYO3_PYTHON="$SEARCH_TEST_PYTHON" CARGO_BUILD_JOBS=4 cargo test \
  --manifest-path rust/pokezero-search/Cargo.toml \
  deep_noisy_q_control_falsifies_raw_q_max_for_both_seats -- --nocapture
```

The output includes every per-seat/per-batch row. The test is deterministic;
it supplies no model checkpoint and must not be cited as model inference timing
or as evidence of playing strength.
