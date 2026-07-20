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
| `maybeTrapped` mask divergence (obs mask fail-closes; engine-equivalence spike allows) | REVISED (walls audit 2026-07-19) — WORLD CONSTRUCTION fails open with no restriction: each sampled world's own opponent ability decides trapping in-engine. The OBSERVATION mask remains the sole fail-closed action authority (`showdown._switching_allowed` drops all switches on trapped/maybeTrapped; `engine_search._map_choices` clamps search output to mask-legal candidates — an open world shapes values but can never emit an unoffered action). NOTE unchanged: a future native mask must pick the observation convention (fail-closed) explicitly | `test_maybe_trapped_constructs_without_restriction` |
| `trapped` request flag (server truth: Shadow Tag / Arena Trap / Magnet Pull / partial trap / Spider Web-class) | FIXED-NOW (walls audit 2026-07-19) — was a blanket construction wall (45.9% of the 100-seed band's fallbacks). Now fails open: native causes re-derive in-world (sampled trap ability, seeded `partiallytrapped` — gen3/state.rs `Side::trapped`); when THIS world shows NO native cause the self side gets the engine's external `force_trapped` (pessimistic-only catch-all — a trapped mon can never escape in search). Locked pseudo-move shapes (empty request move list: recharge / two-turn charge) stay closed: forced 1-action turns, a world would mis-model the charge. `maybeDisabled`/`maybeLocked` (Imprison-only, absent from pool) still fail closed | trapped-flag tests ×5 + `shadowtag_wobbuffet`/`wrap_partialtrap` scenarios |
| Confusion / Destiny Bond volatiles (unseeded family B) | FIXED-NOW (walls audit 2026-07-19) — allow-listed, seeded verbatim: `destinybond` engine-exact (KO reflection, removal on next non-DB move, preserved on consecutive DB); `confusion` engine-native 50/50 self-hit + exact Gen 3 self-damage, hidden 2-5-turn countdown has no engine counterpart → documented no-expiry pessimistic approximation (sleep precedent). `attract` stays FAIL-CLOSED: the engine accepts but wholesale-ignores the volatile (inert no-op, not a duration gap) — owner decision pending (inert allow-list vs vendored-engine patch vs stay closed) | `confusion_lifecycle`/`destinybond_reflection` scenarios + allow-list tests |
| Toxic counter −1 residual-boundary offset | HANDLED — the payload applies the offset (`_materialization_toxic_stage`); construct consumes it verbatim | `test_toxic_stage_maps_to_toxic_count` + `test_local_showdown` offset tests |
| Transform (Ditto): copied identity, revert-on-switch, PP suppression | FIXED (prior change this branch): both seats fail closed; revert clears the block | live Ditto tests ×2 + revert assertion |
| **Shedinja fixed HP = 1** | **FIXED-NOW** — construct computed maxhp 164 via the raw formula; a "1/1" condition fraction-scaled to an unkillable 164-HP Shedinja (silent wrongness). Base-HP-1 pin now mirrors the generator | `test_shedinja_maxhp_is_pinned_to_one` |
| **Recharge (Hyper Beam)** | **FIXED-NOW** — self seat was already safely closed (request `trapped` flag; costless: forced turns have no decision). Opponent seat searched worlds gave the recharging mon a FREE MOVE (silent wrongness). Now: turn-exact signal from round-indexed public actions (+ miss check; fail-open if the record is unavailable) → engine `MUSTRECHARGE` volatile (verified: restricts to "No Move") | `test_recharging_slot_gets_mustrecharge_volatile` + `RechargeSignalTests` (anchor-required miss check, species continuity, scrolled-window fail-open) |
| **Trick / Knock Off item mutation** | **HANDLED (removal) / SAFE-CLOSED (swap)** — belief exposes `item_mutated` + `item_removed` (removal-vs-swap): a REMOVAL (Knock Off, or an item-taking Trick that returned nothing) leaves the mon publicly ITEMLESS — `removed_item_species` signal → engine_world clears the sampled item (spread stays the original assignment's; rule-outs stay frozen upstream). A live SWAP (holder carries an item that is not the sampled assignment) still fails closed; follow-up: the post-swap CURRENT item is publicly revealed, so worlds could substitute it instead | belief removal/swap/Trick-after-KO tests · signal units · `KnockOffRemovalLiveTests` (real protocol end-to-end) |
| Baton Pass volatile whitelist + fail-closed markers (sub HP, leech source) | HANDLED — payload `materializationBlockers` → `materialization_blocker` fail-close; pending BP boundary fails closed | `test_fail_closed_taxonomy` |
| Encore: sole real move-locker, derivable lock, NO duration counter | HANDLED (prior change this branch) — lock derived (self: disabled pattern; opp: last public move), engine restriction pinned, no invented counter | encore tests + engine pin |
| Taunt/Disable/Torment/Imprison | N/A in pool (inventory-certified: never gate a move); volatile allow-list fails closed anyway | allow-list default |
| Struggle / recharge pseudo-moves vs self-moveset guard | HANDLED — pp-less request rows never enter `known_pp` (reviewer-verified vs Showdown source); cannot false-fire | #707 review evidence |
| Locked moves (Thrash/Outrage, `[from] lockedmove`) | SAFE-CLOSED — `lockedmove` volatile not in allow-list; belief PP semantics live upstream | allow-list default |
| Trap abilities (Wobbuffet/Dugtrio/Magneton/Nosepass) | HANDLED — engine models gen3 ability trapping natively; sampled singleton abilities reproduce it (POC review verified) | POC review probe |
| Mean Look / partial trap (`-activate move: Wrap` shape, audit bug C2) | REVISED (walls audit 2026-07-19) — `partiallytrapped` allow-listed: engine-native switch restriction, 1/16 max-HP residual, trapper-switch release (hidden 2-5-turn duration = no-expiry approximation). SELF-side Mean Look / Spider Web (victim volatile not parser-tracked) covered via the `trapped` flag + `force_trapped` catch-all. OPPONENT-side Mean Look / Spider Web remains a PRE-EXISTING silent optimistic under-model (worlds let the victim switch; unchanged by this fix — follow-up: track the `-activate\|<mon>\|trapped` shape → per-side force_trapped) | `wrap_partialtrap` scenario + allow-list tests |
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
- Confusion / partial-trap no-expiry approximation (walls audit): the server's
  hidden 2-5-turn counters have no gen3 engine counterpart, so seeded worlds
  hold the volatile until switch/release — pessimistic-only, sleep precedent.
- Opponent-side Mean Look / Spider Web: the victim's `trapped` volatile is not
  parser-tracked, so worlds where WE trap the opponent still let it switch —
  pre-existing optimistic under-model, unchanged by the walls fix (the SELF
  side is covered via the request flag + `force_trapped`).
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
