# Leaf observation column source map (engine-swap capstone)

Status: 2026-07-19, post adversarial review (PR #730 fixes F1-F5 landed).
The contract for real per-outcome model observations at search leaves (plan
v3 "Integration endgame"; owner-decided architecture).
Implementation: `rust/pokezero-search/src/leaf.rs` (`LeafEncoder`),
`src/encoder.rs` (`write_history_cells` — native fold-product consumption),
`src/model.rs` (`search_batched_multi_encoded` — the end-to-end loop).
Validation: `scripts/leaf_root_parity.py` (depth-0 root-parity gate),
`scripts/leaf_vs_reality.py` (the one-branch leaf-vs-reality differential —
the only gate that exercises evolve-on-change, the delta/line families,
per-branch recompute, and self-order evolution; the depth-0 gate
structurally cannot fire any of them),
`scripts/validate_rust_encoder.py --backend rust-fold` (full-surface gate),
`tests/test_leaf_encoder.py` (committed-sample nets).

## The architecture (owner-decided; do not relitigate)

A leaf observation is the ROOT observation EVOLVED per branch:

- **FOLD-DERIVED** — history/tendency columns come from the branch's advanced
  `FoldState`: shared root prefix + appended synthesized tokens, NO freezing
  of history columns (a leaf whose transition tokens show simulated turns its
  tendency columns ignore is internally inconsistent / OOD).
- **ENGINE-STATE-DERIVED** — state columns recompute from the ENGINE
  post-state of that branch.
- **WORLD-CONSTANT** — belief-epistemic features (unrevealed opponent
  facts, the sampled world's fixed assignments) are per-world constants
  fixed at the root: legitimately root-frozen because they are epistemic,
  not history.

Three refinements the two gates forced, all inside the ENGINE-STATE-DERIVED
class:

- **Evolve-on-change** (root-parity-forced): parser/ledger-authored strings
  (team `condition`, belief `condition`/`status`) stay byte-frozen at their
  root values until the engine actually moves that mon's (hp, status) during
  a branch. Rationale: the recorded ledger legitimately holds conventions
  the payload-built engine world cannot see — fainted mons keep their last
  status, and recorded ledger/payload skews (e.g. a Refresh-cured paralysis
  still shown in the ledger's condition, 10/1028 golden rows) must reproduce
  as recorded, not as re-derived.
- **Line-driven metadata** (differential-forced, review F1/F4): several
  ledger counters are functions of the PROTOCOL LINES, not of engine state,
  because the engine ticks per end-of-turn run while the parser/ledger ticks
  per LINE — a faint-pending ply runs the engine's EOT but emits no `|turn|`
  (toxic_stall repro). `LeafMeta` replays the parser's own rules over the
  branch's synthesized lines, chained per branch exactly like the fold:
  toxic stage (=1 on `|-status|tox`, +1 per `|turn|` cap 15, reset on the
  side's switch, cleared on `|-curestatus|`; a status REPLACEMENT like
  Rest's tox→slp KEEPS the stage — golden-proven), active stints
  (`turns_active`: reset on switch lines, +1 per `|turn|`), per-mon sleep
  counts (`sleep_turns` = observed `|cant …|slp` lines since the
  `|-status|slp`, keyed per mon so a sleeper that faints later keeps its
  count), self-team display order (Showdown's exact switch-SWAP-with-slot-0
  `switchIn` semantics), and the fresh-active flag (below).
- **Delta families** (state-snapshot-based): opponent `move_uses` evolves as
  root ledger + engine PP consumed since root — but see the PP finding
  below; sleep-clause-used follows the ledger's HOLDER semantics
  (golden-proven: the flag marks the side that INFLICTED sleep): it engages
  when the side's OPPONENT gains a non-Rest sleeper and RELEASES when that
  sleeper leaves play (faint/wake — double-KO repro).

## The map

Production construction sites are `src/pokezero/showdown.py` unless noted.
Classes: **F** = fold-derived, **E** = engine-state-derived, **W** =
world-constant, **C** = static contract (layout constants, invariant).

### Token 0 — field (`_encode_field_token` :1977, `_encode_field_exact_state` :2017)

| surface | class | leaf source |
|---|---|---|
| `CATEGORY_PRIMARY` request_kind | E | `move` / `force_switch` from engine force_switch + active hp |
| `CATEGORY_SECONDARY` weather id | E | engine `state.weather.weather_type` (SUN→sunnyday, RAIN→raindance, SAND→sandstorm, HAIL→hail) |
| `CATEGORY_ROLE`, `NUMERIC_PRESENT` | C | constants |
| `NUMERIC_{SELF,OPP}_HAZARDS/SCREENS` (:2058) | E | engine `side_conditions` (spikes layers; reflect/lightscreen/safeguard/mist as booleans) |
| `NUMERIC_TURN_COUNT` | E | leaf turn = root turn + completed simulated turns (`RenderedEvents.turn_completed`; the engine has no turn counter) |
| `NUMERIC_{SELF,OPP}_FUTURE_SIGHT` (:1592,:1631) | E | engine `side.future_sight.0` |
| `NUMERIC_{SELF,OPP}_SLEEP_CLAUSE` | E-delta | ledger HOLDER semantics: engages on a new non-Rest sleeper on the side's OPPONENT, releases when that sleeper leaves play, else root value |
| `NUMERIC_WEATHER_TURNS`, `NUMERIC_WEATHER_PERMANENT` (:1127) | E+turn | permanent (engine −1) pinned at 5; same-as-root weather ticks by the PARSER formula (root remaining − completed simulated turns — the engine decrements per EOT run, incl. turn-less faint plies); weather set in-branch uses the engine counter (set-ply granularity, documented) |
| timed side-condition turns (`_timed_condition_turns` :1143) | E | ACTIVE counts from engine `side_conditions`; SET TURNS root-frozen — remaining = duration − (leaf turn − set turn) keeps ticking through simulated turns. A screen SET inside a branch has no set-turn (duration column stays 0 for it) — documented approximation; gen3 randbats has no screens in the pool |
| `NUMERIC_{SELF,OPP}_WISH_PENDING` (:1155) | E | engine `side.wish.0 != 0` |

### Tokens 1–6 self team / 7–12 opponent team (`_encode_pokemon_tokens` :2192)

| surface | class | leaf source |
|---|---|---|
| SELF-TEAM DISPLAY ORDER (token/switch-slot/mask index assignment) | E (line-driven) | golden observations order the self team ACTIVE-FIRST per the request; each switch during a branch SWAPS the incoming mon with slot 0 — Showdown's exact `switchIn` position semantics (sim/battle-actions.ts) — replayed over the synthesized lines (`evolve_self_order`). **Review F2: pre-fix the leaf kept root order, landing switch tokens and mask bits on wrong indices at ~26% of boundaries; post-fix the permutation class is gone (differential evidence below)** |
| `CATEGORY_PRIMARY` species, `CATEGORY_TYPE_*`, base stats, level | W | root (identity; Transform is fail-closed at world construction and its in-branch application is an accepted engine-model deviation) |
| `condition` → `NUMERIC_HP_FRACTION`, `NUMERIC_LEGAL` (fainted), status categorical (:3125) | E (evolve-on-change) | engine (hp, maxhp, status) when moved since root, else the root parser/ledger string byte-frozen; a mon FAINTING in-branch drops its actual-HP stat entry (requests report "0 fnt" with no max HP — `_max_hp_from_condition`) while its five request stats remain |
| `NUMERIC_ACTIVE` | E | engine `active_index` (a fainted, not-yet-replaced active stays marked active — request semantics) |
| boosts (`_encode_active_boosts` :2065) | E | engine side boost fields (atk/def/spa/spd/spe/accuracy/evasion) |
| volatiles (`_encode_active_volatiles` :2075) | E | engine volatile bitset filtered/mapped to `TRACKED_VOLATILES` ids (leaf.rs `VOLATILE_MAP`); engine-only mechanics volatiles dropped; CURSE Ghost-gated (review F5: the gen3 engine applies the base Curse choice's USER volatile with no Ghost split — the spurious non-Ghost volatile is dropped; Ghost-curse TARGET placement stays an engine-model deviation) |
| `NUMERIC_TOXIC_STAGE` (:1613) | E (line-driven) | **review F1**: the parser stage is |turn|-line-driven (set 1 on `-status tox`, +1 per `|turn|` cap 15, reset on switch, cleared on curestatus — Rest's status REPLACEMENT keeps it), replayed by `LeafMeta`; a state-side guard clears only full cures the mapper cannot render (engine active alive at status NONE). The pre-fix engine-counter arithmetic was 1 low on fresh applies and 1 high on re-applies (arr69/arr82) — both repro shapes unit-tested, the differential's toxic family is zero |
| belief facts: possible abilities/items/moves, revealed flags+counts, `candidate_set_count`, `NUMERIC_UNCERTAINTY`, `candidate_variants` → expected-stat ranges (:2492) | **W** | root, byte-frozen (epistemic); the SELF ledger is NOT epistemic — a first-time-active self mon (e.g. the replacement after a faint) gets a synthesized minimal entry, matching production's ledger growth |
| exact-state ledger: `NUMERIC_SLEEP_TURNS`, `NUMERIC_REST_SLEEP`, `NUMERIC_WAKE_KNOWN` (:2348) | E (line-driven) | `sleep_turns` = observed `|cant …|slp` lines since the `|-status|slp`, per MON (belief.py:91 semantics; a sleeper that faints after its cants keeps them); `rest_sleep` from engine rest counters; `WAKE_KNOWN` derives from W ability facts |
| `NUMERIC_TURNS_ACTIVE` (:2348) | E (line-driven) | **review F4 fixed**: the ledger's per-stint counter (reset on switch-in — Showdown `activeTurns = 0` — +1 per `|turn|`) is replayed by `LeafMeta` stints. Pre-fix it was root-frozen, stale by exactly the completed-turn count on ~85-95% of multi-turn leaves (self 724 + opp 588 boundary hits in the reviewer's differential); post-fix the family is zero |
| `NUMERIC_TRAPPER_ALIVE` | W+E | ability certainty is W; alive/active bits are E |
| `NUMERIC_SUB_HP_FRACTION` (:2472) | E | engine SUBSTITUTE volatile + the production maxhp/4 approximation (the model was trained on the approximation, not the engine's true `substitute_health`); whether a hit BREAKS the sub is roll-envelope + sub-health-approximation dependent |
| opponent revealed-move PP fractions + validity (:2421) | W + E-delta (**effectively root-frozen above PP 10**) | revealed set is W; `move_uses` = root uses + engine PP consumed since root — **review F3 correction: the engine only emits `DecrementPP` when pp < 10 (generate_instructions.rs:2207 optimization), so PP-derived cells (opponent move_uses AND the self action-token `NUMERIC_MOVE_PP_FRACTION`) do NOT track in-branch consumption above PP 10: exact below 10, root-frozen at/above. Differential measurement: 478/775 driven golden-v2 boundaries show a PP-family divergence one boundary ahead (the dominant expected class)** |
| tendency triple `NUMERIC_MON_*` (:2643) | **F** | fold products `opponent_mon_tendencies` |
| pinned Tier-2 `NUMERIC_TIER2_CB_PINNED` / `_INVESTMENT_PINNED` (:1233–1269, :2334) | **F** | fold products `cb_pinned_species` / `investment_pinned` (running state, truncation-robust) |

### Tokens 13–21 actions (`_encode_action_tokens` :2958, `_action_candidate_metadata` :3054)

| surface | class | leaf source |
|---|---|---|
| move ids, mechanics, PP fractions | E | engine active's move surface (engine slot order = sampled = request order, root-parity-proven; hidden power renders as plain `hiddenpower`); PP from engine (but see the F3 pp≥10 freeze above), maxpp from dex; a recharging active (MUSTRECHARGE) presents the production request shape: one legal PP-less `recharge` pseudo-move, switching disallowed |
| disabled / `NUMERIC_ACTIVE` | E (line-guarded) | engine `Move.disabled`, EXCEPT a fresh switch-in (line-tracked: switched and no `|move|` since): choice locks reset on switch but the world seeds benched mons with their last stint's cached disabled bits and the engine never re-enables them on a branch switch (`use_last_used_move` is off in constructed worlds) — Choice-Band-Nidoking repro |
| legal bits + switch candidates | E | the engine's OWN option surface (`get_all_options`) mapped through the canonical switch map over the EVOLVED team ordering; fresh switch-ins bypass the stale-lock option restriction (pp>0 ⇒ legal) |
| `legal_action_mask` (:3227) | E | same. **Honest history (review item 3a): at depth 0 the engine option surface reproduced every recorded request mask (root-parity), but PRE-FIX the mask did NOT reproduce reality at switch-reached leaves — the self-team permutation put switch bits on wrong indices (~26% of boundaries) and stale choice locks killed move bits on fresh switch-ins. POST-FIX the differential's mask families are zero outside the documented engine-model tags (transform/recharge/encore/baton-pass)** |

### Token 22 stats (`_encode_stats_token` :2665)

All counters (**F**): fold products `tendency_stats` (switch counts, decision
opportunities, blocked-on-our-attack, pursuit-intercept predictions, my
switch turns, weather-reveal pairs). Role/presence: C.

### Tokens 23–150 transitions (`_encode_turn_merged_transition_tokens` :2794)

Everything (**F**): the branch's advanced fold's `turn_merged_tokens` tail
(budget-truncated), both sub-blocks, spikes/weather collapse fields, Tier-2
annotations — written natively from `ProductsData` in
`encoder.rs::write_history_cells` (no `products_payload` crossing).

### Masks (`_token_type_ids` :3428, `_attention_mask` :3440)

`token_type_ids`: C. `attention_mask`: team extents E/W (root membership),
stats visibility C(mask)+F(presence), transition extent **F** (filled
turn-merged rows). `history_mask`: C (window 1).

## Epistemic asymmetries (documented, by design)

- **Opponent team membership is root-frozen.** A branch that switches in a
  never-revealed opponent mon shows it in the transition tokens (the fold
  sees the synthesized switch), but no opponent-team token materializes for
  it — materializing one would present the sampled world as revealed fact.
  Its within-branch state (hp/status) rides the history tokens only.
- **In-branch Transform / newly-set screens' timers** inherit the engine's
  model limits (world construction fail-closes on the corresponding ROOT
  states; see `belief_edge_case_matrix`).

## Root-parity gate results (2026-07-19, the decisive validation)

`scripts/leaf_root_parity.py`: per corpus row, world from the recorded public
payload + TRUE teams (fidelity-harness machinery: true-override,
recharge/Truant flags), ZERO branch steps, `encode_leaf` on the untouched
root state, byte-diff of all five arrays against the recorded golden arrays.

| corpus | rows | driven | exact | divergent | skips (world fail-closed) |
|---|---|---|---|---|---|
| golden-v2 | 1028 | 1015 | **1015 (100%)** | 0 | 13 (8 encore_move_unknown, 3 pending_baton_pass, 2 self_request_state_unsupported) |
| golden-v2-scenarios | 290 | 235 | **235 (100%)** | 0 | 55 (15 encore, 2 baton-pass, 12 self_moveset_mismatch, 26 self_request_state_unsupported) |

Every skip is an `EngineWorldUnsupported` fail-closed reason — positions the
branch simulator itself cannot search today (the same wall as
engine-search fallback), not encoder gaps. WORLD-CONSTANT columns matched
exactly under true-override; no history-column divergence (fold cells were
separately proven byte-exact over ALL 1318 rows via
`validate_rust_encoder.py --backend rust-fold`).

## Leaf-vs-reality differential (2026-07-19, the review-mandated gate)

`scripts/leaf_vs_reality.py`: per same-seat boundary (row n → n+1), build the
world at row n (true teams), drive the RECORDED joint actions through
`branch_events` selecting the reality-consistent branch (fidelity matcher +
a strict faint-pattern filter — the roll-scaled HP tolerance can conflate a
KO'd active with a 1-HP one, which an observation diff cares about),
advance a fresh copy of row n's fold + overlay, evolve order/metadata from
the synthesized lines, `encode_leaf`, byte-diff against ROW N+1's golden
arrays. This is the only gate that fires evolve-on-change, the line/delta
families, and per-branch recompute (all root values at depth 0).

Results after the F1-F5 fix wave (exit gates on the defect classes):

| corpus | boundaries | driven | exact | divergent | **state** | **turn** | fold | epistemic | engine_pp | engine_roll | engine_model | ledger_skew |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| golden-v2 | 1008 | 771 | 78 | 693 | **0** | **0** | 440 | 322 | 663 | 313 | 9 | 1 |
| golden-v2-scenarios | 270 | 183 | 9 | 174 | **0** | **0** | 108 | 49 | 173 | 97 | 16 | 0 |

(Class columns count divergent boundaries touching that class; a boundary
can carry several. Skips mirror the fidelity gate: world fail-closed 13/52,
no_branch_match 224/28 — dominated by roll-collapse outcomes reality didn't
take — plus 7 action_unmapped scenario rows.)

Every non-defect class is documented above: `fold` inherits the fidelity
gate's (b)/(c) classes; `epistemic` is the root-frozen belief surface
meeting row n+1's new reveals (by design); `engine_pp_model` is the F3
pp≥10 freeze; `engine_roll` is the damage-roll collapse envelope (HP
fractions, sub breaks, pinch-berry thresholds); `engine_model` is the
tagged vendored-engine deviations (Transform empty delta, Encore volatile
not applied, recharge consumed a ply early, Baton-Passed saved moves never
resolving, in-branch screen set-turns); `ledger_skew` is the recorded
production inconsistency (ledger condition strings keep stale status
suffixes through cures — the same class root-parity documents at ~1%).

Pre-fix defect counts, for the record: toxic ~60 boundaries (F1 both
directions), self-team permutation ~26% of switch-reached leaves (F2),
turns_active 724+588 hits (F4), plus stale choice locks, sleep-clause side
inversion, faint-ply weather/toxic escalation, per-mon sleep counts, fainted
actual-HP stats, and self-ledger growth — all found by this differential,
none visible to the depth-0 gate.

## /100 opponent-HP base decision (forward caveat #2, resolved)

Evidence: the local harness feeds the **omniscient** stream
(`local_showdown.py` `_apply_event`, stream == "omniscient") — exact HP for
BOTH sides; corpus event slices confirm (`|-damage|p2a: Deoxys|148/196` as
seen by either seat). So in the entire local domain — self-play training
data, the golden corpus, the paired FoulPlay harness where the 200-seed read
runs — the root fold consumes TRUE-base fractions and the mapper's
true-`cur/maxhp` rendering is already distributionally identical. **Default:
exact base, no change.**

For ladder (player-view) deployments the opponent side arrives as `X/100`
under Showdown's HP Percentage Mod (`sim/pokemon.ts getHealth`:
`ceil(100*hp/maxhp)`, 100 shown as 99 while damaged). The mapper now takes
`EventContext.hp_percent` per side and renders that side's conditions with
the exact formula (`events.rs hp_percent_condition`; unit-tested against the
formula's edge cases), so leaf-synthesized fractions land on the same /100
grid the root fold consumed. `tests/test_leaf_encoder.py::HpPercentGridTest`
pins the two grids apart. Nicknames (forward caveat #1) remain cosmetic:
fold occupant tracking is details-based (species either way); revisit only
if a live ladder integration consumes idents directly.

## End-to-end bench (2026-07-19, `scripts/bench_leaf_search.py`)

Full loop per model row: root state → tree → branch → synthesized events →
per-branch Rust fold advance (chained via `BranchSeam` parent keys) → native
encode → batched TorchScript eval → exact-expectation backup. Random-weights
artifact at the real v2.2 shape (151×51 cat / 155 num, embedding 64, 1
layer), 3 mid-game golden-v2 positions, batch 16, Apple M-series CPU:

| position | depth | sims | stub sims/s | encoded sims/s | overhead | µs/eval | lossy |
|---|---|---|---|---|---|---|---|
| 1000#r5 | 2 | 256 | 2620 | 1406 | 1.86× | 125 | 0 |
| 1000#r5 | 2 | 1024 | 4025 | 1881 | 2.14× | 149 | 0 |
| 1000#r5 | 3 | 1024 | 3843 | 1757 | 2.19× | 153 | 0 |
| 1001#r4 | 2 | 1024 | 3173 | 1363 | 2.33× | 172 | 110 |
| 1001#r4 | 3 | 1024 | 3009 | 1212 | 2.48× | 182 | 153 |
| 1002#r4 | 2 | 1024 | 3121 | 1561 | 2.00× | 132 | 0 |
| 1002#r4 | 3 | 1024 | 2594 | 1472 | 1.76× | 116 | 0 |

Readings: the real-observation overhead is **~115–190 µs per model eval**
(state clone + event render + fold clone/advance + JSON rewrite + encode +
tensor marshaling) — 1.8–2.6× elapsed vs the #716-style template stub at
these budgets, i.e. **~1.2–1.9k model-priced sims/s with real observations**
(vs the stub's 2.6–4.0k) at 0.5–0.9 s per 1024-sim decision. MPS shows the
same per-eval overhead band (the encode is CPU-side either way). Lossy
renders (position 1001, a Rest/Sleep-Talk position: 4–6% of evals) are the
mapper's documented ambiguity classes, counted in the report
(`lossy_renders`), rendered fold-safe, never silently mis-attributed.
Argmax: the encoded search differentiates positions the constant-template
stub cannot (per-position stable choices; random weights — no strength
claim).

Optimization headroom (not taken here, honestly deferred): the encode path
re-clones the root JSON per leaf and re-hashes ~7.7k vocab strings; interning
categorical rows and reusing grids should cut the per-eval overhead several
fold before the paired read if wall-clock parity at fixed budget matters.

Memory note: `fold_by_branch` (per-branch fold + order + metadata records in
the encoded search) is never pruned within a search — it grows with the
priced-branch count (fold states are ~O(100KB) worst case late-game). Fine
at the intended ≤8192-sim budgets; revisit before very large searches.

## Remaining before the 200-seed paired FoulPlay read

1. **Prior/action mapping** (track D residue): map policy-head priors onto
   engine `MoveChoice`s at decision nodes (the encoder now produces the
   observation and the legal mask; priors are still uniform in-tree).
2. **`search.py` integration**: swap the branch simulator behind the
   existing search-policy interface (root belief worlds → per-world
   `LeafEncoder` + root fold export) and aggregate across worlds at the root.
3. **Root fold export at live decision boundaries**: production recomputes
   the fold per observe; the live client needs the incremental fold state
   handed to the crate (the corpus proves the payload codec both ways).
4. Re-price batch size / virtual-loss settings under real observation costs
   (keep batch ≪ sims; docs/crate_search_design.md review caveats).
