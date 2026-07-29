# Prediction, recorded BEFORE the 300-game differential run

Base: cycle-nine, engine fingerprint 887a722dd2d6cd9b (33 patches, UNCHANGED — this is a
pure parser/engine_world change, no vendored patch).
Population: c9's 108 outside-limits rows, seeds 1500000-1500299, strict matcher.

1. All **11** encore_redirect_gap rows clear  ->  108 - 11 = **97** outside-limits rows.
   Basis: 5 of the 11 already verified closed individually (1500099/49, 1500123/38,
   1500136/27, 1500161/44, 1500232/54), each -1 divergence at an identical measured
   boundary count. The other 6 are the same mechanism.

2. Known way a row could NOT clear: if the observed last move is absent from the
   constructed moveset, `_resolve_encored_move_index` returns None and the seed is
   deliberately left None. Judged unlikely here because the encored move was just used
   PUBLICLY, so it is revealed by construction — but it is the failure mode to look for
   if the count comes in under 11.

3. **Other families: expect ZERO change.** `last_used_move` has exactly two consumer
   families in the gen3 engine — Encore (option filter, onStart failure guard, same-turn
   redirect, PP-exhaustion end) and Fake Out. Fake Out is NOT in the gen3 randbats pool,
   so it cannot fire in this population. Specifically including the 5 adjudicated
   `limit:roll_divergent_lethality` rows and the 3 `limit_not_established_keeps_label`
   rows: their branch sets are priced by damage rolls, which read no last_used_move.
   The already-encored path is byte-unchanged (the new seeding is an `elif`).

   The one mechanical difference elsewhere: the engine's record site dedupes
   (`if last_used_move == used_move { return; }`), so a side that repeats a move now
   emits FEWER SetLastUsedMove instructions. That is not an HP component and must not
   move any row; if it does, that is a finding.
