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
| Toxic counter −1 residual-boundary offset | HANDLED — the payload applies the offset (`_materialization_toxic_stage`); construct consumes it verbatim | `test_toxic_stage_maps_to_toxic_count` + `test_local_showdown` offset tests |
| Transform (Ditto): copied identity, revert-on-switch, PP suppression | FIXED (prior change this branch): both seats fail closed; revert clears the block | live Ditto tests ×2 + revert assertion |
| **Shedinja fixed HP = 1** | **FIXED-NOW** — construct computed maxhp 164 via the raw formula; a "1/1" condition fraction-scaled to an unkillable 164-HP Shedinja (silent wrongness). Base-HP-1 pin now mirrors the generator | `test_shedinja_maxhp_is_pinned_to_one` |
| **Recharge (Hyper Beam)** | **FIXED-NOW** — self seat was already safely closed (request `trapped` flag; costless: forced turns have no decision). Opponent seat searched worlds gave the recharging mon a FREE MOVE (silent wrongness). Now: turn-exact signal from round-indexed public actions (+ miss check; fail-open if the record is unavailable) → engine `MUSTRECHARGE` volatile (verified: restricts to "No Move") | `test_recharging_slot_gets_mustrecharge_volatile` + signal units |
| **Trick / Knock Off item mutation** | **FIXED-NOW** — belief exposes `item_mutated`; sampled items are frozen to the ORIGINAL assignment, so a mutated holder mismatches reality. Fail closed on any opponent `item_mutated` (2 Trick sets in the pool — rare) | signal unit |
| Baton Pass volatile whitelist + fail-closed markers (sub HP, leech source) | HANDLED — payload `materializationBlockers` → `materialization_blocker` fail-close; pending BP boundary fails closed | `test_fail_closed_taxonomy` |
| Encore: sole real move-locker, derivable lock, NO duration counter | HANDLED (prior change this branch) — lock derived (self: disabled pattern; opp: last public move), engine restriction pinned, no invented counter | encore tests + engine pin |
| Taunt/Disable/Torment/Imprison | N/A in pool (inventory-certified: never gate a move); volatile allow-list fails closed anyway | allow-list default |
| Struggle / recharge pseudo-moves vs self-moveset guard | HANDLED — pp-less request rows never enter `known_pp` (reviewer-verified vs Showdown source); cannot false-fire | #707 review evidence |
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
| Deferred simultaneous-turn opponent action | SAFE-CLOSED — payload validation rejects; boundary fails closed (the one remaining bench fallback) | taxonomy test |
| Unown formes / Deoxys formes | HANDLED — Unown collapsed (with party-species consistency); Deoxys formes are real dex entries | forme test |
| Screens presence vs turns-remaining | HANDLED — turns derived from set turns; expiry validated multi-turn | screens tests + S4 |
| Leftovers/pinch residual ORDER | HANDLED — engine build patched (order-5/10 split), differential + pins | #686 gates |
| Future Sight / Doom Desire | SAFE-CLOSED — pending strike fails closed | taxonomy test |

## Residual known gaps (accepted, documented)
- Recharge signal fails OPEN when the round record is unavailable (rare;
  pre-fix behavior). Missed Hyper Beam correctly produces no lock.
- Opponent PP exemption stands until PP modeling adopts the belief ledger.
- Sleep/rest approximation per the POC tradeoffs doc.
- A future NATIVE legal-mask implementation (the crate) must adopt the
  observation mask's fail-closed `maybeTrapped` convention — the
  engine-equivalence spike's permissive convention is the wrong reference.
