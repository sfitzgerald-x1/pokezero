# Fold-state closure probe (track B prerequisite)

Status: 2026-07-18. The prerequisite probe demanded by
`test_time_search_plan_v3.md` ("Encoder contract", "Schema v2"): before schema
v2 can store an exported fold state + inter-decision event slice per corpus
row, the production transition/tendency fold must be shown to be
**deterministic and prefix-closed** — i.e. expressible as
`advance(fold_state, events) -> (fold_state', products')` such that advancing
slice-by-slice reproduces, at every decision boundary, exactly what the batch
path computes from the full prefix.

Scope of "the fold" (what production actually runs per observe, verified):

- `normalize_for_player` (`showdown.py:1073-1085`) calls
  `extract_transition_products` (`turn_merged.py:214`) → **one**
  `_fold_replay(replay, ...)` (`transitions.py:498`) over **all** raw public
  lines, then `_merge_fold` + `_tendency_stats_from_fold`. No persistent fold
  state exists anywhere in production — the full stream is re-folded on every
  observe. (Both the local env observe path `local_showdown.py:1240-1291` and
  the online client go through `normalize_for_player`.)
- The env then layers Tier-2 annotations onto the freshly re-extracted stream:
  `Tier2LiveTracker.annotate` (`tier2.py:1170`), `InvestmentLiveTracker.observe`
  (`investment.py:751`), and the merged-stream join
  `annotate_turn_merged_tokens` (`turn_merged.py:235`,
  called at `local_showdown.py:1283-1291`).
- The observation-visible products of all of the above (verified against the
  v2.2 encoder):
  1. the **merged transition block**: `state.turn_merged_tokens[-budget:]`
     (`showdown.py:2817-2818`), annotated;
  2. the **TendencyStats** dataclass (stats token, `showdown.py` stats block);
  3. the **Tier-2 pinned surfaces**, derived from the FULL (untruncated)
     annotated per-action stream: `tier2_cb_pinned_species`
     (`showdown.py:1233-1246`) and `tier2_investment_pinned`
     (`showdown.py:1247-1268`);
  4. the attention-mask fill count `min(len(stream), budget, count)`
     (`showdown.py:3450-3461`) — needs only the min-capped length.

## Component verdicts

Verdict vocabulary: **LEFT-TO-RIGHT** (plain running state, carryable),
**BOUNDED-LOOKBACK** (needs a bounded buffer of recent lines/turns),
**NEEDS-FULL-CARRY** (state that must be carried whole — noted with its bound),
**VIOLATION** (lookahead across a decision boundary / non-carryable global
state — none found).

| # | Component | Evidence | Verdict |
|---|---|---|---|
| 1 | `_fold_replay` main-loop locals (`side_condition_counts`, `weather`, `turn_number`, `hp_fraction`, `occupant`/`_StayRecord`, `transformed`, `pending_baton_pass`, `pending_faint_replacement`, `lead_seen`, `pending_charge`) | `transitions.py:498-544` | LEFT-TO-RIGHT — plain per-line state, carried verbatim. |
| 2 | Window lifecycle + token emission | `transitions.py:527-537, 895, 899-924` | LEFT-TO-RIGHT with one open window carried. A closed window is immutable (no handler touches `windows[]`; the batch post-pass exception is #3, resolved at open time). The batch fold's trailing `close_window()` (`:895`) means a boundary's product includes the open window **virtually closed**; the incremental form must compute boundary views without baking them (the same window keeps accumulating after a mid-chunk boundary, e.g. a forceSwitch pause inside the killing move's chunk — batch at the next boundary re-attributes those lines to the same window). |
| 3 | `_flag_pursuit_intercepts` | `transitions.py:939, 942-972` | BOUNDED-LOOKBACK — SAFE. Backward scan from the Pursuit window's `event_index`, breaking at the first `_PURSUIT_SCAN_BOUNDARY` line (`move/switch/drag/replace/cant/turn/upkeep`). Finding: blank `\|` chunk separators are **not** in the boundary set, so the scan may cross them — the incremental ring buffer therefore clears on boundary lines only, never on `\|` separators. Equivalent computed at window-open time: the buffer holds exactly the lines the batch scan would visit before its break. Post-pass in batch for convenience only. |
| 4 | `opportunity_turns` + the other `_tendency_stats_from_fold` aggregates | `transitions.py:431-495` | LEFT-TO-RIGHT (reducible). Not stored state — derived by iterating all (token, window) pairs. The `(side, turn)` dedupe is a last-counted-turn scalar (turns are monotone, tokens in order; only the opponent side's count is ever read). `is_decision` inputs (`kind`, `called`, `locked_continuation`, `voluntary_switch`, `action`) are fixed at window open; `blocked_on_our_attack` reads `damage_outcome`, final only at close → counters accumulate at window close, with the open window contributing virtually at a boundary (per #2). `pursuit_intercept` is open-time in the incremental form (#3). |
| 5 | `mon_counters` | `transitions.py:519-524, 562-567, 588-590, 649-651` | LEFT-TO-RIGHT — bounded (`(side, base species)` → 3 ints, ≤ 12 revealed mons), increment-only, mutated at line time (not token time). Carried verbatim. |
| 6 | `weather_reveals` | `transitions.py:817-826` (producer), `:475-483` (consumer) | LEFT-TO-RIGHT (reducible). Appended per `-weather` line in event order, but the only consumer reduces to `{weather: OR(from_ability)}` per side then sorts — order-independent → carried as a bounded dict. |
| 7 | `turn_start_occupants` / `completed_turns` / `fainted_turns` | producers `transitions.py:552-567, 553-558, 570-573, 793-795`; consumers `turn_merged.py:451-459` (`consumption_confirmed` for the group's own turn), `turn_merged.py:513-518` → `_missing_sub_block` `turn_merged.py:675-683` (occupants of the group's own turn). Grep-verified: no consumer outside `turn_merged.py`. | BOUNDED-LOOKBACK — the merge only ever queries the **current group's turn** (and the just-flushed turn at a `\|turn\|` line). The NEGATED-gate inputs freeze when the group finalizes: `\|turn\|N+1` adds `N` to `completed_turns` before any turn-N+1 window exists, and `fainted_turns` can no longer gain `N` afterwards. Carrying a last-2-turns slice of each map is sufficient; the full maps are also tiny (O(turns)) if ever needed. |
| 8 | `_merge_fold` (turn-merged layer) | `turn_merged.py:413-461` | LEFT-TO-RIGHT with bounded staging. Groups are contiguous same-turn window runs; turn numbers are non-decreasing, so a group is complete at the next `\|turn\|`/`\|win\|` line and its merged tokens are immutable from then on (inputs per #7 frozen at the same moment). Incremental form: flush the pending group at `\|turn\|`/`\|win\|` (after `completed_turns.add`, before the turn number updates — matching batch handler order `transitions.py:556-567`), keep a bounded tail of finalized merged tokens, and virtually merge the open group (+ virtually-closed open window) at boundaries. The lead pass consumes only the initial LEAD-reason run (`turn_merged.py:419-442`) → one carried `lead_done` flag. Assumption made explicit: a repeated `\|turn\|N` line would split a batch group in the incremental form — impossible in engine-emitted protocol (turn numbers strictly increase); noted, not guarded. |
| 9 | Tier-2 annotation layer: `Tier2LiveTracker` + `InvestmentLiveTracker` + `annotate_turn_merged_tokens` (the pinned-CB bit, numeric col 138 `NUMERIC_TIER2_CB_PINNED`, and the Tier-2/investment sub-block fields) | `tier2.py:294-329, 1109-1234`; `investment.py:714-799`; `turn_merged.py:235-289`; pinned surfaces `showdown.py:1233-1268` | LEFT-TO-RIGHT — VERIFIED streaming. Both trackers consume each protocol line exactly once (`_IncrementalContextFold.process`, `tier2.py:325-329`) and assess each token once, monotonically from `_assessed_until`; per-index conclusions are immutable once set (Tier-2 assesses a strike at the first boundary that sees it; investment codes are as-of-strike). Carried tracker state, enumerated: context-fold dicts (occupant/status/boosts/pending_bp/flash_fire/transformed/type_changed/ability_overridden/item_mutated/hp/side_counts) + `contexts`/`token_line_indices` (**O(actions), but only indices ≥ `_assessed_until` are ever read again — prunable to a bounded frontier**), `_assessed_until`, `_residuals`, `_cb_turns`, `_cb_non_ko`, `_cb_bit_indices` (tier2), `_state` conclusions + `token_codes`, `_defender_levels` (investment); `_stats_cache`/`_spread_cache` are pure caches, not state. The trackers additionally need runtime dependencies that are not fold state (the caller's belief engine, dex, own team, CB whitelist). The annotation **join** is a positional cursor walk over the full streams — incrementalized with per-merged-token representative-index bookkeeping; the pinned surfaces are monotone reductions (CB: species set, add-only, `cb_bit` never retracts; investment: last-annotated-index-per-species map). |
| 10 | Observe-path consumers | `showdown.py:1073-1085, 2817-2818, 1233-1268, 3450-3461`; `local_showdown.py:1240-1291` | Verified: everything the observation reads is reproducible from the bounded state above — merged tail (last `budget` ≤ 128 rows), TendencyStats, pinned reductions, min-capped stream length. |

**No VIOLATIONS found.** The two near-misses, made explicit:

- **Open-window mutation across mid-chunk boundaries** (#2): a decision
  boundary can occur while a window is still open (forceSwitch pause inside
  the acting move's chunk — hazard sack, Explosion double-faint, Baton Pass
  completion). Lines arriving after the boundary may still attach to that
  window. Any incremental design that finalizes the open window's token or its
  tendency contributions at the boundary diverges from the batch re-fold at the
  *next* boundary. Resolution: the carried state keeps the window open;
  boundary products are computed from a **virtual close** that never mutates
  carried state. This is a design constraint, not a closure violation — batch
  at each boundary sees exactly the virtual-close view.
- **`_flag_pursuit_intercepts` crosses blank separators** (#3): the scan
  boundary set excludes the blank `|` chunk separator, so a lookback buffer
  keyed on "lines since the last chunk separator" would be wrong. The correct
  buffer clears on `_PURSUIT_SCAN_BOUNDARY` line types only.

## The incremental accumulator (implemented)

`src/pokezero/transitions_fold.py` — `FoldState` with
`FoldState.initial(perspective_slot=...)`,
`advance(raw_lines_slice) -> (new_state, FoldProducts)` (pure; `|t:|`
wall-clock lines filtered per the schema-v2 byte-determinism rule),
`apply_annotations(overlay)` (the tracker join), and
`to_payload()`/`from_payload()` (JSON-safe, deterministic). The batch fold in
`transitions.py`/`turn_merged.py` is untouched and serves as the differential
oracle; the merge itself reuses `turn_merged._merge_turn` verbatim so the two
paths cannot drift on merge semantics.

Carried state (serialized): the main-loop locals (#1), the open window (#2),
the pursuit ring buffer (#3), the tendency counters + opportunity dedupe
scalar (#4), `mon_counters` (#5), the reveal dict (#6), last-2-turn slices of
occupants/completed/fainted (#7), the pending window group + `lead_done` +
bounded finalized-merged tail + expansion cursor (#8), and the annotation
overlay + representative-index map + pinned aggregates (#9). Token tail bound
defaults to 512 per-action tokens (≥ the v2/v2.1 encode budget of 128 and ≥
the flatten expansion of the merged tail); merged tail bound defaults to 128
(= `TRANSITION_TOKEN_COUNT`, the maximum any spec/budget can read).

## The differential closure proof (results)

`tests/test_transitions_fold.py` drives real `LocalShowdownEnv` games —
10 random `gen3randombattle` seeds plus all 10 curated scenario games from
`pokezero.golden_corpus_scenarios` (Pursuit, Baton Pass, RestTalk, Explosion
double-faints, Truant, Transform, recharge, screens, sand+Shedinja, toxic
stall) — and at EVERY decision boundary, for BOTH perspectives, asserts
dataclass-equality between the batch fold over the full prefix and the
incremental fold advanced slice-by-slice, on every observation-visible
product: the merged tail, the per-action tail, both totals, and TendencyStats.
A separate battery does the same for the annotated surfaces (Tier-2 trackers
active) including the pinned reductions, plus serialization round-trips
mid-game (resume-from-payload must converge identically).

RESULTS (2026-07-18, this machine, vendored Showdown at
`/Users/scott/workspace/pokerena/vendor/pokemon-showdown`):

- Random games (seeds 9001-9010): 10 games, 881 decision boundaries ×
  2 perspectives = **1,762 differential checks — all products equal**, with a
  mid-game serialize→canonical-JSON→resume round-trip at boundary 7 of every
  game (all later boundaries prove resume convergence).
- Scenario games: 10 scenarios, 153 boundaries × 2 perspectives = **306
  differential checks — all equal** (truant_slaking:6, ditto_transform:14,
  encore_wobbuffet:24, hyperbeam_recharge:6, baton_pass_boundary:21,
  wish_boundary:24, sand_shedinja:9, resttalk_snorlax:17, screens_jirachi:18,
  toxic_stall:14; round-trip at boundary 3).
- Annotated surfaces (tier2 residuals + investment masks on, belief set
  source on; seeds 17001-17003 + pin-bearing seeds 17007/17012/17013):
  6 games, **535 boundaries**, the acting player's tracker overlay (up to 27
  annotated indices) applied per boundary — annotated per-action tail,
  annotated merged tail, and TendencyStats equal to the production env state
  at every boundary. The pinned surfaces are bound to PRODUCTION, not a
  test-local reduction: `tier2_cb_pinned_species`/`tier2_investment_pinned`
  are locals of `observation_from_player_state`, so the check reads where they
  land — the per-opponent-mon `NUMERIC_TIER2_CB_PINNED` /
  `NUMERIC_TIER2_INVESTMENT_PINNED` columns of the encoded observation —
  and is non-vacuous by assertion: 138 boundaries carried a real pinned CB
  conclusion and 48 a pinned investment conclusion. Round-trip at boundary 9.
- Synthetic battery (no Showdown): line-by-line advance vs the batch fold on
  EVERY prefix of a hand-built log (41 prefixes × 2 perspectives — mid-chunk
  cuts included), Pursuit-intercept incremental flagging, `|t:|` filtering,
  advance purity, and annotation-overlay round-trips.

This differential IS the prefix-closure proof the plan requires: the batch
fold over every prefix equals the incremental fold over the slices, so the
fold is (state + slice)-closed and the state is exportable.

## Perf note (stretch measurement, same machine)

Measured on the seed-9001 random game (101 boundaries, 878 protocol lines),
p1 perspective, comparing what production pays per observe (batch
`extract_transition_products` over the full prefix) against
`FoldState.advance` on the inter-boundary slice:

- batch-per-observe total across the game: **117.9 ms** (grows quadratically
  with game length — each later boundary re-folds the whole prefix);
- incremental total: **8.4 ms** with the pure (clone-per-advance) API,
  **6.7 ms** in-place — **14-18× cheaper end-to-end**, and O(slice) per
  boundary instead of O(prefix), so the gap widens on stall games.
- Final serialized fold state: **225,802 bytes** of canonical JSON
  (dominated by the 512-token action tail + 128-token merged tail; the
  repetitive keys compress ~10× under gzip for corpus storage, and both tail
  limits are per-state constructor knobs if schema v2 wants smaller rows).

Numbers are from `tests/test_transitions_fold.py`'s perf harness
(`POKEZERO_FOLD_PERF=1 .venv/bin/python -m unittest
tests.test_transitions_fold.FoldPerfNote`); they will vary with hardware but
the asymptotic shape is the point.
