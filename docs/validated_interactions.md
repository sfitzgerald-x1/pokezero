# Validated-interactions registry — gen3 encoder (2026-07-20)

A living registry of gen3 interactions driven through the **real sim** (scripted
`gen3customgame`), with the resulting protocol captured and the **encoded observation
verified** (or a bug flagged). Each row: interaction · protocol shape · expected
encoding · verdict · regression scenario.

Verdict legend:
- **VERIFIED-CORRECT** — driven live; the encoder represents it correctly.
- **BUG-FOUND** — driven live; the encoder mis-encodes it. **Flagged; NOT fixed here**
  (a fix is a separate PR so this registry stays a clean audit surface).
- **ACCEPTED-APPROX** — a deliberate, documented approximation (no fix warranted).

Regression scenarios live in `pokezero.golden_corpus_scenarios.interaction_registry_specs()`
and are asserted by `tests/test_interaction_registry.py` (live-gated on a built Showdown
checkout; well-formedness always runs). The two bugs are wired as `expectedFailure`
asserting the CORRECT behavior — they flip to xpass the day the encoder is fixed.

## ✅ FIXED (was: in-battle retype family — loud flag)

**STATUS (2026-07-20): FIXED** by the `scott/in-battle-retype-fix` PR (In-battle retype
encoder fix: Castform Forecast + Kecleon Color Change type slots). The parser now tracks a
per-active-slot **live type override** and the encoder overrides the **type slots only** from
the forme / `typechange` payload — the base species token is left unchanged (retyped Castform
formes / Kecleon are OOV for the species vocab). Cleared on switch-out (both effects revert on
leaving the field). The two registry tests below now pass as normal assertions (formerly
`expectedFailure`). Audit detail is preserved unchanged for the record.

Two **genuine encoder bugs**, same root cause. **Both species are in the gen3 randbats
pool** (`castform`, `kecleon` in `sets.json`), so this is in-distribution, not a
customgame-only curiosity.

| interaction | protocol shape (captured live) | expected encoding | verdict | scenario |
|---|---|---|---|---|
| **Castform Forecast retype** (the flagged case) | `\|-weather\|SunnyDay` then `\|-formechange\|p2a: Castform\|Castform-Sunny\|[msg]\|[from] ability: Forecast` | active-mon `CATEGORY_TYPE_1` = **Fire** (Rainy→Water, Snowy→Ice) while the forme holds | **FIXED** (was BUG-FOUND — encoded **Normal**). `showdown.py` now consumes `-formechange`, tracks the forme's type in a live override, and encodes it into the type slots for **both** self and opponent tokens (forme resolved via the dex, like Deoxys; explicit map fallback). | `castform_forecast_formechange` |
| **Color Change retype** (Kecleon) | `\|-start\|p1a: Kecleon\|typechange\|Psychic\|[from] ability: Color Change` (payload = new type) | active-mon `CATEGORY_TYPE_1` = the last hit's type (here Psychic) | **FIXED** (was BUG-FOUND — encoded **Normal**). The `typechange` payload type is now written to the type slots (mono-type; TYPE_2 padded), persisting until the next hit or switch-out. | `colorchange_kecleon` |

**Repro (Castform):**
```
env.reset_with_start_override(seed=999, override=<Charizard[Sunny Day] vs Castform[Forecast]>)
env.step({p1: Sunny Day, p2: Return})     # sun up -> Forecast fires -> Castform-Sunny (Fire)
obs = env.observe("p1")                     # p1 sees Castform as opponent-active
obs.categorical_ids[OPPONENT_POKEMON_TOKEN_OFFSET][CATEGORY_TYPE_1]
    == vocab.encode("type:Normal")          # TRUE  (BUG: should be type:Fire)
```
(Self side identical: `p2`'s own Castform token also encodes Normal.)

**Blast radius / mitigation (why it is real but bounded):**
- The **type feature slot is plainly wrong** (Normal) with **no "type unreliable" bit**
  on the mon token. The net reads Castform as Normal-typed for both offense (STAB) and
  defense (weaknesses/immunities) while the forme holds.
- Partial mitigation: the **Tier-2 residual layer** (`tier2.py`, default-ON
  `tier2_residuals`) sets `type_changed` on `-formechange`/`typechange` and **disqualifies
  the species-derived damage residual** (`disqualifiers=("type-changed",)`, `tier2.py`
  L1330 / `investment.py` L391), so historical damage-inference annotations for that mon
  are NOT poisoned. But this protects only the residual columns; the base type feature the
  net sees on the active/team tokens stays Normal, and `tier2_investment` is default-OFF.
- If the parser were fixed to update the species to `Castform-Sunny`, note `species:castform-sunny`
  / `species:castform-rainy` / `-snowy` are **outside the enumerated randbat vocab** (they
  hash to the OOV safety-net row) — so a correct fix should update the **type slots** from the
  forme (and/or the `typechange` payload) rather than swap the species token. Same for Kecleon.

**Scope:** 2 of 220 species; retype only while the trigger holds (weather for Castform —
its pool set has no self-weather move, so it fires off opponent Sunny Day / Rain Dance,
both in-pool; last-hit type for Kecleon). Bounded but real, and squarely the
"un-enumerated sub-case" tail this audit targets.

## VERIFIED-CORRECT — driven live this audit

| interaction | protocol shape | expected encoding | scenario / evidence |
|---|---|---|---|
| **Deoxys forme** (Attack/Defense/Speed) | `\|switch\|p1a: Deoxys\|Deoxys-Attack\|241/241` (real dex entry in DETAILS) | distinct base stats per forme | VERIFIED: Deoxys-Attack base_atk feat **0.900**/def 0.100; Deoxys-Defense **0.350**/0.800; distinct species ids. `deoxys_forme_swap` |
| **Intimidate** (switch-in) | `\|-ability\|p2a: Salamence\|Intimidate\|boost` + `\|-unboost\|p1a: ...\|atk\|1` | victim active atk boost = **−1/6 ≈ −0.1667** | VERIFIED. `intimidate_switchin` |
| **Trace** | `\|-ability\|p1a: Porygon2\|Intimidate\|Trace\|[from] ability: Trace\|[of] p2a: Salamence` | traced ability marks `ability_overridden` (residuals disqualified) | VERIFIED (tier2 `ability_overridden`). `intimidate_switchin` (Porygon2 partner) |
| **Belly Drum** | `\|-setboost\|p1a: Snorlax\|atk\|6\|[from] move: Belly Drum` | active atk boost = **+1.0** (6/6) | VERIFIED. `bellydrum_snorlax` |
| **Spikes stacking** | `\|-sidestart\|p2: P2\|Spikes` ×3 (one per layer) + `\|-damage\|...\|[from] Spikes` | self-hazard field feat = **layers/3** (→1.0 at 3) | VERIFIED. `spikes_stack` |
| **Substitute** | `\|-start\|P\|Substitute` … `\|-end\|P\|Substitute` | active volatile col = `volatile:substitute` | VERIFIED (id present the turn after cast). `substitute_focuspunch` |
| **Weather permanence** (Sand Stream vs move weather) | `\|-weather\|Sandstorm\|[from] ability: Sand Stream\|[of] p2a: Tyranitar` vs `\|-weather\|SunnyDay` | ability weather → `WEATHER_PERMANENT=1.0`; move weather → timed | VERIFIED. `sand_stream_permanence` |
| **Roar / Whirlwind drag** | `\|drag\|p2a: Snorlax\|...` after opponent boosted | dragged-in mon's boosts **reset to 0** (drag ≠ Baton Pass) | VERIFIED (opp spa boost 0.0 post-drag). `roar_drag_reset` |
| **Future Sight / Doom Desire** | `\|-start\|p1a: Alakazam\|Future Sight` (on USER; lands on OPPONENT side) | pending strike on opponent side → `OPP_FUTURE_SIGHT` > 0 from caster view | VERIFIED (parser `_update_future_sight`). `future_sight_pending` |
| **Perish Song** | `\|-start\|P\|perish3\|[silent]` then `perish3→0` each turn + `\|-fieldactivate\|move: Perish Song` | per-mon `volatile:perishN` (tracked) | VERIFIED (perish0-3 in `TRACKED_VOLATILES`). `perish_song` |
| **Counter / Mirror Coat** | fixed/callback damage; `\|-immune\|p2a: Tyranitar` (Mirror Coat is Psychic-typed → Dark immune) | fixed-damage move (no set info); effectiveness via `-immune` | VERIFIED (drove Wobbuffet). `counter_mirrorcoat` |
| **Heal Bell** | `\|-activate\|P\|move: Heal Bell` + `\|-curestatus\|P\|tox\|[silent]` (per-mon, NOT `-cureteam`) | status cleared via `-curestatus` | VERIFIED live (census fact: gen3 never emits `-cureteam`). — |

## VERIFIED-CORRECT — covered by existing scenarios/tests (referenced, re-confirmed)

These already ship with dedicated scenarios/tests (`golden_corpus_scenarios.scenario_specs()`
+ `belief_edge_case_matrix.md`); this audit re-confirmed the protocol shape and encoding
path rather than duplicating them.

| interaction | encoding | existing coverage |
|---|---|---|
| **Ditto Transform** (opponent) | `enc_species = transform_species`; types/stats from copied identity; base HP stays Ditto's | `ditto_transform` scenario + belief `transform_species`; `test_transformed_ditto_encodes_target_stats_but_original_hp` |
| **Ditto Transform** (SELF) | same copied identity/types/base-stats on OUR own transformed Ditto (base HP stays Ditto's); transform flag read from the self EXACT belief, not the self-side-absent set-source belief | FIXED (was ditto/Normal/48-across on every self-transformed decision, incl. re-transform after switch-out; opponent path was already correct). `test_self_ditto_transform_surfaces_target_identity` + `test_self_ditto_retransform_after_switch`. Rust LEAF encoder (search path) still needs the same self-transform fix — follow-up (cf. HP #758) |
| **Baton Pass** (boosts/sub/leech pass) | `_BATON_PASS_TRANSFERRED_VOLATILES` carried; boosts NOT reset on BP `\|switch\|` | `baton_pass_boundary` scenario |
| **Trick + Knock Off** item mutation | belief `item_mutated`/`item_removed`/`current_public_item`; current-item override both seats | `trick_swap_exchange` / `trick_berry_pinch` scenarios + belief tests |
| **Berry / herb consumption** | `-enditem ... [eat]` → `item_removed` without mutation | `berry_eat_chesto` scenario |
| **Wish** | pending-only field bit (`SELF/OPP_WISH_PENDING`) | `wish_boundary` scenario |
| **Encore** | derived lock (no invented counter); volatile pinned | `encore_wobbuffet` scenario |
| **Rest + Sleep Talk** | sleep status + approximate-sleep-turns (documented tradeoff) | `resttalk_snorlax` scenario |
| **Truant** | loaf phase seeded from round-indexed public actions → TRUANT volatile | `truant_slaking` scenario |
| **Recharge (Hyper Beam)** | `MUSTRECHARGE` volatile from request pseudo-move + round-indexed actions (NOT `-mustrecharge` line) | `hyperbeam_recharge` scenario |
| **Explosion / Self-Destruct self-KO** | double-faint → double cold-replacement; defense-halving is Tier-2's problem | `test_explosion_fixture` (seed-148 turn-7 double-faint) |
| **Pursuit-on-switch** | `pursuit_intercept` transition-token flag; KO-continuation | `transitions.py` pursuit flag + `test_transitions` |
| **Flash Fire / Volt Absorb / Water Absorb** | `-start`/`-immune`/`-heal [of]` absorb shapes | `flashfire_houndoom` / `voltabsorb_lanturn` / `waterabsorb_quagsire` scenarios |
| **Ghost vs non-Ghost Curse** | Ghost: target `-start Curse` + self HP cut; non-Ghost: boost events | `ghost_curse` scenario + Snorlax Curse |
| **Screens (Reflect/Light Screen)** | `-sidestart` + per-side timed-turn features | `screens_jirachi` scenario |
| **Sand + Shedinja / Wonder Guard** | Shedinja base-HP-1 pin; Wonder Guard `-immune` | `sand_shedinja` scenario |
| **Attract immobilization** | `attract` volatile + patch; searches (not walled) | `attract_snorlax` scenario |

## ACCEPTED-APPROX / N-A-in-randbats

| interaction | disposition | rationale |
|---|---|---|
| **Focus Punch** focus marker | ACCEPTED-APPROX | `-singleturn\|move: Focus Punch` not consumed; the move resolves normally and PP/legality come from the request — no cross-turn belief state to lose |
| **Endeavor** | ACCEPTED-APPROX | fixed/callback damage (sets target HP to attacker's); carries no set info, excluded from CB by construction (like Seismic Toss) |
| **Present** | ACCEPTED-APPROX / N/A-randbats | mixed damage/heal with empty-target `\|move\|...\|\|[still]`; Delibird is not on any gen3 randbats set → out of distribution; the move token parses (empty target tolerated) |
| **Beat Up** | N/A-randbats | absent from the gen3 randbats pool (inventory-certified; no team-dependent powers) |
| **Taunt / Disable / Torment / Imprison** | N/A-randbats | not on any gen3 randbats set (inventory-certified); the volatiles ARE in `TRACKED_VOLATILES` so they encode if they ever appear (customgame), and the engine-world allow-list fails closed |
| **`\|tie\|`, `-notarget`, `-singleturn`, `-fieldactivate`, `-endability`** | ACCEPTED gaps | see `protocol_coverage_matrix.md` — benign / no functional state loss |

## Registry summary

- Rows verified this way: **VERIFIED-CORRECT = 26** (12 driven-live-here + 14
  existing-scenario re-confirmations), **BUG-FOUND (now FIXED) = 2** (Castform Forecast,
  Kecleon Color Change — same root cause), **ACCEPTED-APPROX / N-A = 6**.
- **Castform verdict: FIXED** (was BUG-FOUND) — in-battle `-formechange` / `typechange`
  retype is now reflected in the observation's type slots via a per-active-slot live type
  override (`scott/in-battle-retype-fix`); the base species token is left unchanged.
  Note: the **Rust leaf encoder** (search path) still needs a separate retype realignment
  (a search-side follow-up, like the Hidden Power Rust realignment #758) — the corpus
  rust-parity gate is unaffected because the golden sample contains no retype states.
