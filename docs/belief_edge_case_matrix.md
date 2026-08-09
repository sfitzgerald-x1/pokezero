# Mask/belief edge-case audit vs the engine-world construct (2026-07-18)

Owner-directed audit: every edge case deliberately handled in the legacy
legal-mask + public-belief stack (mined from git history, code comments, and
test names), cross-checked against the NEW engine-world construction path
(`engine_world` + `engine_search` signals + the witnessed sampler fallback).

Verdict legend: **SAFE-CLOSED** = fails closed (search falls back; no wrong
world) · **HANDLED** = constructed correctly, tested · **FIXED-NOW** = hole
found by this audit, fixed in this change · **N/A** = not reachable by the
construct · **NOTE** = residual risk documented.

| Edge case (legacy handling) | New-construct verdict | Test |
|---|---|---|
| `maybeTrapped` mask divergence (obs mask fail-closes; engine-equivalence spike allows) | SAFE-CLOSED — any raised self request flag fails construction closed; the engine derives its own trapping. NOTE: a future native mask must pick the observation convention (fail-closed) explicitly | `test_fail_closed_taxonomy` (request flags) |
| Toxic counter residual-boundary conversion | HANDLED — ordinary requests apply −1; stage zero is admitted only after same-seat active faint → upkeep → non-Baton-Pass toxic replacement; internal 16 preserves an already-saturated simulator stage 15; unknown active-Toxic provenance fails closed | `ToxicStageWorldTest` latch/boundary tests + Rust root-to-leaf pins |
| Transform (Ditto): copied identity, revert-on-switch, PP suppression | FIXED (prior change this branch): both seats fail closed; revert clears the block | live Ditto tests ×2 + revert assertion |
| **Shedinja fixed HP = 1** | **FIXED-NOW** — construct computed maxhp 164 via the raw formula; a "1/1" condition fraction-scaled to an unkillable 164-HP Shedinja (silent wrongness). Base-HP-1 pin now mirrors the generator | `test_shedinja_maxhp_is_pinned_to_one` |
| **Recharge (Hyper Beam)** | **FIXED-NOW** — self seat was already safely closed (request `trapped` flag; costless: forced turns have no decision). Opponent seat searched worlds gave the recharging mon a FREE MOVE (silent wrongness). Now: turn-exact signal from round-indexed public actions (+ miss check; fail-open if the record is unavailable) → engine `MUSTRECHARGE` volatile (verified: restricts to "No Move") | `test_recharging_slot_gets_mustrecharge_volatile` + `RechargeSignalTests` (anchor-required miss check, species continuity, scrolled-window fail-open) |
| **Trick / Knock Off item mutation** | **HANDLED (removal + swap)** — belief exposes `item_mutated` + `item_removed` + `current_public_item`: a REMOVAL (Knock Off, or an item-taking Trick — `-enditem ... [silent] [from] move: Trick`) leaves the mon publicly ITEMLESS → `removed_item_species` clears the sampled item; a SWAP is fully public in gen3 (a successful Trick emits one `-item`/`-enditem` per mon, each naming the holder and its resulting item — proven from `data/moves.ts` trick onHit + live probes both directions) → `current_item_overrides` substitutes the protocol-confirmed CURRENT item on BOTH mons of the exchange (self seat included: it never walled, it was silently stale). Fail-closed remains for a mutation with NO confirmed current item (unaudited `-item`/`-enditem` move sources — Thief/Covet pool-change guard) and for contradictory removal+override state (`item_state_conflict`). KO→Trick is unreachable (gen≤4 `itemKnockedOff` gate; probed `\|-fail\|`); Trick→KO and Trick→eat compositions end in removal | belief swap/removal/composition/hardening tests · signal units · `TrickSwapOverrideLiveTests` + `KnockOffRemovalLiveTests` (real protocol end-to-end) · trick scenario sweep |
| **Berry / herb consumption (`-enditem ... [eat]`, plain)** | **FIXED-NOW** — a publicly EATEN berry (own or Tricked-on) previously left no belief mark: sampled worlds handed the berry back (silent wrongness — the mon could "re-eat" it in search). Consumption now sets `item_removed` WITHOUT `item_mutated` (the eaten item still pins variant matching when unmutated) → same `removed_item_species` clearing. Verbatim shapes probed: `\|-enditem\|SLOT\|Petaya Berry\|[eat]` (+ `-boost ... [from] item:`), Chesto-Rest `[eat]` + `-curestatus` | belief eat tests (own / Tricked / Chesto) · signal unit (removed-without-mutation) · `berry_eat_chesto` + `trick_berry_pinch` scenario sweep |
| Baton Pass volatile whitelist + fail-closed markers (sub HP, leech source) | HANDLED — payload `materializationBlockers` → `materialization_blocker` fail-close; pending BP boundary fails closed | `test_fail_closed_taxonomy` |
| Encore: sole real move-locker, derivable lock, NO duration counter | HANDLED (prior change this branch) — lock derived (self: disabled pattern; opp: last public move), engine restriction pinned, no invented counter | encore tests + engine pin |
| Taunt/Disable/Torment/Imprison | N/A in pool (inventory-certified: never gate a move — re-counted 2026-08-09: `taunt` appears on 0 of 1682 gen3 randbats variants). **Taunt is now HANDLED anyway**, because the corpus scenarios reach it: gen3's clock is FIXED (`data/mods/gen3/moves.ts` `duration: 2`, `durationCallback: undefined`, and gen4's plain `onStart` — which gen3 inherits — drops the modern already-moved bump), so it is exact rather than hidden, and it is in `_SUPPORTED_VOLATILES` with `volatile_status_durations['taunt'] = 1`. Disable/Torment/Imprison still fail closed | `test_engine_world_taunt.py` + `struggle_taunt_stall` scenario; the other three: allow-list default |
| Struggle / recharge pseudo-moves vs self-moveset guard | HANDLED — pp-less request rows never enter `known_pp` (reviewer-verified vs Showdown source); cannot false-fire | #707 review evidence |
| **Struggle-only request vs the retained self moveset** | **FIXED-NOW** — the same pp-less drop that keeps the pseudo-move out of `known_pp` also made `actor_move_states_from_request_history` SKIP the request, so `sides[self].pokemon[].moves` stayed pinned to the last pp-bearing request and advertised a usable move while `selfActiveMoves` (current request) said `[]` — one payload, two views, two different requests (measured live: `sunnyday pp1 disabled:false` vs `[]`). `_apply_struggle_only_move_state` now marks the ACTIVE row's moves `disabled` at the payload boundary, restoring the verdict `Pokemon.getMoves` computed (`hasValidMove == false`) before substituting the Struggle row. Boundary-scoped on purpose: the same marking in the FOLD rode a switched-out mon onto the bench with full PP and no legal move. PP stays one use too generous (the request carries none). Struggle is now MEASURED rather than source-only. The former RESIDUAL here — with no legal switch (trapped, or last-mon PP stall) the engine emits `MoveChoice::None` and `_map_choices` translated only to `recharge`, so the decision still missed — is CLOSED: the forced-no-move token now also resolves to the request's substituted `struggle` candidate, gated on it being the request's only legal move and on there being no legal switch. Captured before/after on `bp-trap-lastmon2` d20 (20 `choices_unmapped` refusals → 0). BOTH ROUTES NOW REACH IT: the TAUNT route used to refuse EARLIER and under a different class — `no_worlds_constructed`, `world_failures {"volatile_unsupported: side 'p1': ['taunt']"}` — and `engine_world._SUPPORTED_VOLATILES` now admits `taunt` (counter seeded at 1 tick elapsed), so `struggle_taunt_stall` is searchable: 9 refusals → 0, 30/102 worlds constructed → 48/48. On the `legal == ['struggle']` shape (lone all-status Blissey vs a Taunting Smeargle) 12 refusals → 0 with all 12 Struggle-only decisions still present, i.e. the two fixes compose. NOT a production-rate claim: Taunt is on 0 of 1682 gen3 randbats variants, so this class is corpus/customgame-only | `test_struggle_only_move_state.py` (predicate · row scope · bench leak · both payload views live · world construction) + `test_engine_world_taunt.py` (construction · counter · engine fidelity) + `struggle_taunt_stall` scenario |
| Locked moves (Thrash/Outrage, `[from] lockedmove`) | SAFE-CLOSED — `lockedmove` volatile not in allow-list; belief PP semantics live upstream | allow-list default |
| Trap abilities (Wobbuffet/Dugtrio/Magneton/Nosepass) | HANDLED — engine models gen3 ability trapping natively; sampled singleton abilities reproduce it (POC review verified) | POC review probe |
| Mean Look / partial trap (`-activate move: Wrap` shape, audit bug C2) | SAFE-CLOSED — arrives as unsupported volatile → fail-close | allow-list default |
| Sleep Clause holder | N/A — sampling never invents a second sleeper (hidden mons sample healthy; statuses copied only from public rows) | — |
| Sleep/rest counters, Early Bird | NOTE — `approximate_sleep_turns` opt-in flattens Rest (documented tradeoff); exact fix = public counter plumbing | flag tests |
| Natural Cure / Lum non-proc pruning / Shield Dust / Intimidate elimination | N/A — belief-side candidate pruning, upstream of sampling; consumed via candidate variants | belief tests |
| Pressure ×2 PP ledger, caller-charging, transform-no-charge | N/A today — opponent PP is catalog-full (documented exemption). MUST use the ledger when PP modeling lands | exemption note |
| Pursuit mid-switch (no forceSwitch cycle) | N/A — turn-structure concern; construct reads request-boundary payload state only | — |
| Roar/Whirlwind phazing (drag ≠ BP) | HANDLED — payload boosts/volatiles reflect the drag reset; force-switch boundaries construct | force-switch test |
| Wish (interrupted/expired) | HANDLED — pending-only payload + carrier-independent engine semantics (amount ignored by engine; deviation documented) | wish tests |
| Deferred simultaneous-turn opponent action (Baton Pass boundary) | FIXED-NOW — self-pending BP constructs: passer `baton_passing`+`force_switch`, opponent commitment sampled into the engine's saved-move field, but review probes show the gen3 build does NOT resolve it (fail-soft under-model; field kept for forward compat). The boundary itself searches — recipient choice with boosts passing — which was the fallback cost. Opponent-pending shapes stay closed. Bench: 0.0% fallback (original seed set — see baseline note below) | BP boundary tests |
| Unown formes / Deoxys formes | HANDLED — Unown collapsed (with party-species consistency); Deoxys formes are real dex entries | forme test |
| Screens presence vs turns-remaining | HANDLED — turns derived from set turns; expiry validated multi-turn | screens tests + S4 |
| Leftovers/pinch residual ORDER | HANDLED — engine build patched (order-5/10 split), differential + pins | #686 gates |
| Future Sight / Doom Desire | SAFE-CLOSED — pending strike fails closed | taxonomy test |
| **Truant (Slaking)** | **FIXED-NOW** — engine models the loaf alternation natively but the construct never seeded the phase: every sampled world had Slaking about to act (over-valued both seats). Phase is publicly derivable (acted last round → loafs now); signal from round-indexed public actions seeds the engine's TRUANT volatile. Fail-open without clear evidence | truant scenario sweep |

## Residual known gaps (accepted, documented)
- Recharge signal fails OPEN when the round record is unavailable OR the
  move-line anchor has scrolled out of the 24-line event window (cannot
  verify hit/miss -> no lock; review-hardened). Missed Hyper Beam and
  replaced actives (species continuity) correctly produce no lock.
- Baton Pass: the opponent's committed move is NOT resolved by the gen3
  engine after the pass (probe-confirmed engine limitation) — recipient
  enters unharmed; optimistic, fail-soft, documented.
- Opponent PP exemption stands until PP modeling adopts the belief ledger.
- Sleep/rest approximation per the POC tradeoffs doc.
- A future NATIVE legal-mask implementation (the crate) must adopt the
  observation mask's fail-closed `maybeTrapped` convention — the
  engine-equivalence spike's permissive convention is the wrong reference.

## Scenario suite (owner-directed, 2026-07-18)

Random-seed games exercise edge cases by luck; `pokezero.golden_corpus_scenarios`
scripts 10+ deterministic `gen3customgame` scenarios (Truant, Transform, Encore,
Hyper Beam recharge, Baton Pass boundary, Wish, sand+Shedinja, RestTalk,
screens, toxic stall) through the SAME corpus capture machinery, plus a
fallback-detection sweep driving the engine-search policy over every scenario
(true-override injected worlds — the sampler's catalog cannot cover custom
games). Sweep result: every decision searched or failed closed with a known
taxonomy reason; zero unmapped choices. Bonus finding: screen moves are
outside the closed randbats vocabulary (they don't exist in the pool), so
scenario rows exercise the encoder's OOV safety-net path — a validation case
the random corpus can never produce.

## Fallback alerting (owner-directed, 2026-07-18)

Every decision-level fallback is now LOUD, three tiers:
1. `EngineSearchFallbackWarning` (Python warning — visible in test output;
   escalate to hard errors with `warnings.simplefilter("error", ...)`),
2. a structured WARNING on the stable logger
   `pokezero.engine_search.fallback` carrying battle id, round, seat,
   reason, and the per-decision world-failure delta,
3. `EngineMctsConfig(strict_fallbacks=True)` → `EngineSearchFallbackError`
   for sweeps/CI that require zero, and the bench CLI's
   `--fail-on-fallback` flag exits nonzero with a stderr banner.
Baseline note (revised 2026-07-19, PR #737; attribution corrected same day
from the bench logs): the original "0.0% fallback" 15-game bench was
seed-lucky. On the fresh seed set (7000-7014) the rate is 15.2% (60/394
decisions), identical under the HpFraction control, so the pipeline adds
none of its own. Per-battle wall composition (from the per-decision
world-failure deltas): seed 7013 = 48/60 decisions, `public_effect_blocked`
via a GENUINE Trick swap (opponent Furret holding a Tricked Petaya Berry) —
fail-closed BY DESIGN and expected to stay closed even after the
Knock-Off-removal recovery lands; seed 7010 = 7/60,
`self_request_state_unsupported` (request flags, unrelated to items); seeds
7005/7014 = 5/60, `volatile_unsupported: flashfire` (an unseeded but
publicly-derivable volatile — a knockable wall, same shape as the
Truant/MUSTRECHARGE seeding). No Transform wall occurred on these seeds,
and the hidden-power IV mismatches were world-attempt-level only (never
cost a decision). Alerts remain worth a look, judged against this
per-reason taxonomy rather than a zero baseline.
Update (same day, removal recovery landed): the Knock-Off-removal recovery
shipped (belief `item_removed` → `removed_item_species` signal →
engine_world item clearing). Same-seed re-run on the landed code: model arm
15.2% (60/394) with the identical per-battle composition above — 7013's
Trick swap stays closed as predicted — and the new
`removed_item_decisions` telemetry reads 0 on this band (no organic
Knock-Off wall on seeds 7000-7014). The removal path is proven by the live
end-to-end test (`KnockOffRemovalLiveTests`) and a directed paired repro on
one post-Knock-Off state: pre-fix 0 searched / 1 fallback
(`public_effect_blocked`), landed 1 searched / 0 fallbacks.
Update (2026-07-20, Trick-swap override + consumption landed): the item
walls are gone. Same-seed model-arm re-run (base = #744 main): 351/363
searched, 3.3% fallback (was 15.2%) — 7013 48 → 0 (item_override_decisions
13, then removed_item_decisions 12 as the Tricked Petaya Berry was eaten:
the override→removal transition live), 7010 request flags 7 → 7 and
7005/7014 flashfire 5 → 5 (both unchanged, per-reason attributed), zero
public_effect_blocked anywhere, fold cross-check 363/363, wins 15/15.
Paired HP control: main 66/423 = 15.6% (7013 = 44) → branch 22/389 = 5.7%
(7013 = 0; only 7010's request-flag fallbacks remain, identical to main).
Berry consumption now surfaces organically (removed_item_decisions 51 on
the band — worlds no longer hand eaten berries back). Remaining walls,
re-ranked: (1) request-state flags (7010); (2) flashfire/confusion
volatile seeding (7005/7014 — in flight on the parallel absorb PR).
