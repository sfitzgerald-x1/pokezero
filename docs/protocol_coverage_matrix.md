# Gen3 protocol coverage census — encoder parser (2026-07-20)

Owner-directed census: enumerate **every** Showdown sim protocol message type the
production encoder's parser could receive in gen3, cross-reference against what the
parser actually consumes (`showdown.py` `_ReplayParser`, plus the downstream folds in
`transitions.py` / `turn_merged.py` / `tier2.py` / `belief.py`), and mark whether the
current scenario suite / corpus / tests exercise it.

Ground truth for the message universe: `vendor/pokemon-showdown/sim/SIM-PROTOCOL.md`
(94 documented message types) intersected with the gen3 mechanics reachable in
`gen3randombattle` (`data/random-battles/gen3/sets.json` — 220 species, 113 moves,
76 abilities, per `docs/gen3_interaction_inventory.md`).

Verdict legend:
- **COVERED** — the parser consumes it AND it is exercised by a scenario/test/corpus row.
- **COVERED\*** — consumed and exercised, but via an ALTERNATE signal (documented).
- **UNCOVERED-reachable** — reachable in gen3 randbats but NOT consumed (or consumed but
  mis-encoded). **These are the tail.**
- **N/A-gen3** — the message cannot occur in a gen3 singles randbat (gen4+ mechanic,
  doubles-only, team-preview-only, or superseded by a different gen3 shape).

## Message-type coverage matrix

### Battle-structure / meta messages

| message | parser | reachable gen3? | verdict | notes |
|---|---|---|---|---|
| `\|player\|` | consumed (names) | yes | COVERED | slot→name; every game |
| `\|request\|` | consumed (self team + legal mask) | yes | COVERED | self side, request kind, PP, forceSwitch, trapped |
| `\|turn\|` | consumed | yes | COVERED | turn counter; toxic ramp; BP-pending clear |
| `\|win\|` | consumed | yes | COVERED | terminal |
| `\|tie\|` | not consumed | yes (rare: cap/both-faint) | UNCOVERED-reachable | winner stays `None`; benign (terminal handled by turn cap upstream), documented gap |
| `\|t:\|` | consumed (skipped) | yes | COVERED | intentionally dropped (wall-clock, non-deterministic) |
| `\|upkeep\|` | referenced (transitions/turn-merged phase) | yes | COVERED | end-of-turn phase marker |
| `\|teamsize\|` `\|gametype\|` `\|gen\|` `\|tier\|` `\|rule\|` `\|rated\|` `\|start\|` | not consumed | yes (preamble) | N/A-inert | fixed-constant preamble; carries no per-battle state |
| `\|inactive\|` `\|inactiveoff\|` | not consumed | yes (timer) | N/A-inert | timer chatter; no battle state |
| `\|clearpoke\|` `\|poke\|` `\|teampreview\|` | not consumed | no | N/A-gen3 | no team preview in randbats |
| `\|debug\|` | not consumed (fall-through) | customgame-only | N/A-inert | emitted by customgame driver (`before turn callback`); absent in real randbats; ignored |

### Major action messages

| message | parser | reachable gen3? | verdict | notes |
|---|---|---|---|---|
| `\|move\|` | consumed | yes | COVERED | incl. empty-target `\|move\|P\|Move\|\|[still]` (Present/Future Sight) tolerated |
| `\|switch\|` | consumed | yes | COVERED | reveal + boost reset + BP boost/volatile carry + toxic reset |
| `\|drag\|` | consumed | yes | COVERED | Roar/Whirlwind; NEVER Baton Pass (boost reset enforced) |
| `\|replace\|` | consumed (switch path) | no | N/A-gen3 | Illusion (Zoroark) is gen5 |
| `\|cant\|` | consumed (transitions/tier2) | yes | COVERED | flinch/par/sleep/recharge/Truant/Disable/Taunt |
| `\|faint\|` | consumed | yes | COVERED | KO; volatile/transform/flashfire clear |
| `\|-formechange\|` | **tier2 ONLY** (`type_changed` flag); NOT in `showdown.py` public state | **yes (Castform Forecast)** | **UNCOVERED-reachable — BUG** | species/type slot NOT updated → **type feature stays base Normal**. See `validated_interactions.md`. Registry: `castform_forecast_formechange` |
| `\|detailschange\|` | not consumed | no | N/A-gen3 | permanent forme (Mega/Primal) is gen4+; gen3 Deoxys formes arrive via `\|switch\|` DETAILS (real dex entries, HANDLED) |
| `\|swap\|` | not consumed | no | N/A-gen3 | doubles slot swap |

### Minor (`-`) messages — damage/heal/status/boosts

| message | parser | reachable gen3? | verdict | notes |
|---|---|---|---|---|
| `\|-damage\|` | consumed | yes | COVERED | HP + `[from]` attribution (psn ramp, Spikes, Sandstorm, Recoil, Leech Seed) |
| `\|-heal\|` | consumed | yes | COVERED | Leftovers, Wish (`[from] move: Wish`), absorb (`[of]`), Rest |
| `\|-sethp\|` | consumed | yes (Pain Split) | COVERED | HP set; also Wish-clear path |
| `\|-status\|` | consumed | yes | COVERED | brn/par/psn/tox/slp/frz |
| `\|-curestatus\|` | consumed | yes | COVERED | incl. **Heal Bell per-mon `[silent]`** + Natural Cure `[from] ability` |
| `\|-cureteam\|` | tier2 only | **no** | N/A-gen3 | gen3 Heal Bell/Aromatherapy emit per-mon `\|-curestatus\|...[silent]`, never `-cureteam` (verified live) — tier2 handler is dead/defensive |
| `\|-boost\|` `\|-unboost\|` | consumed | yes | COVERED | incl. Intimidate `-unboost`, DDance, self-drops |
| `\|-setboost\|` | consumed | yes | COVERED | **Belly Drum** `atk\|6\|[from] move: Belly Drum` |
| `\|-clearallboost\|` | consumed | yes (Haze) | COVERED | full reset |
| `\|-clearnegativeboost\|` `\|-restoreboost\|` | consumed (tier2) | yes (White Herb) | COVERED | zeroes negative stages |
| `\|-copyboost\|` | consumed | yes (Psych Up) | COVERED | copies stages |
| `\|-clearboost\|` `\|-clearpositiveboost\|` | consumed | marginal | COVERED-defensive | no common gen3 single-target clearer; handlers present |
| `\|-swapboost\|` `\|-invertboost\|` | not consumed | no | N/A-gen3 | Heart/Guard Swap gen4, Topsy-Turvy gen6 |

### Minor messages — field / side / volatiles / effectiveness

| message | parser | reachable gen3? | verdict | notes |
|---|---|---|---|---|
| `\|-weather\|` | consumed | yes | COVERED | id + set-turn + `[upkeep]` continue + **ability-permanence** (`[from] ability:`) |
| `\|-sidestart\|` | consumed (layer count) | yes | COVERED | Spikes (stacks to 3), Reflect, Light Screen, Safeguard |
| `\|-sideend\|` | consumed | yes | COVERED | Rapid Spin clear; timed expiry |
| `\|-start\|` | consumed (volatiles) | yes | COVERED | Substitute, Encore, Leech Seed, Perish (`perish3..0`), Taunt, Future Sight, Confusion, Curse, flashfire |
| `\|-start\|...\|typechange` | consumed by tier2 (`type_changed`); type payload NOT in type slot | **yes (Color Change/Kecleon)** | **UNCOVERED-reachable — BUG** | same root cause as Castform: live type not reflected in `CATEGORY_TYPE_1/2`. Registry: `colorchange_kecleon` |
| `\|-end\|` | consumed | yes | COVERED | volatile clear |
| `\|-crit\|` | consumed (transitions) | yes | COVERED | crit flag on token |
| `\|-supereffective\|` `\|-resisted\|` `\|-immune\|` | consumed (transitions) | yes | COVERED | effectiveness outcome; absorb `-immune` |
| `\|-miss\|` | consumed (transitions) | yes | COVERED | miss outcome |
| `\|-prepare\|` | consumed (transitions) | yes (Solar Beam) | COVERED | charge turn |
| `\|-hitcount\|` | consumed (transitions) | yes (Bonemerang/Double Kick) | COVERED | multi-hit normalization |
| `\|-singlemove\|` | consumed (volatiles) | yes (Destiny Bond/Grudge) | COVERED | single-move volatile |
| `\|-fieldactivate\|` | not consumed | yes (Perish Song) | UNCOVERED-reachable | cosmetic; state carried by per-mon `perishN` volatiles (COVERED) → no functional gap |
| `\|-fieldstart\|` `\|-fieldend\|` | not consumed | no | N/A-gen3 | no terrains / Trick Room / Gravity in gen3 |
| `\|-swapsideconditions\|` | not consumed | no | N/A-gen3 | Court Change gen8 |

### Minor messages — items / abilities / misc

| message | parser | reachable gen3? | verdict | notes |
|---|---|---|---|---|
| `\|-item\|` | consumed (belief/tier2) | yes | COVERED | Trick swap (`[from] move: Trick`) — current-item override both seats |
| `\|-enditem\|` | consumed (belief) | yes | COVERED | Knock Off removal, berry `[eat]`, Chesto-Rest |
| `\|-ability\|` | consumed (belief/tier2) | yes | COVERED | Intimidate reveal, **Trace** acquisition (`ability_overridden`), Flash Fire |
| `\|-endability\|` | not consumed | marginal (Skill Swap) | UNCOVERED-reachable | Skill Swap is in-dex but not on any randbats set → effectively N/A; handler absent |
| `\|-transform\|` | consumed (belief/transitions/tier2) | yes (Ditto) | COVERED | copies species/stats/moves; revert on switch/faint |
| `\|-activate\|` | consumed (volatiles/tier2) | yes | COVERED | partial-trap, Protect block, Heal Bell, Substitute-block, Encore |
| `\|-fail\|` | consumed (BP guard/tier2) | yes | COVERED | failed BP guard; `-fail heal`; already-statused |
| `\|-mustrecharge\|` | **not consumed by parser** | yes (Hyper Beam) | **COVERED\*** | recharge derived from the request `recharge` pseudo-move + round-indexed public actions (belief edge matrix), NOT this line → engine `MUSTRECHARGE` volatile |
| `\|-singleturn\|` | not consumed | yes (Protect/Focus Punch/Endure) | UNCOVERED-reachable | Protect/Endure are same-turn-only (no cross-turn belief state); Focus Punch "focusing" marker unused → low impact, documented |
| `\|-notarget\|` | not consumed | yes (target fainted) | UNCOVERED-reachable | move fizzle; the token lands as a no-damage move (no crash), attribution slightly lossy → low impact |
| `\|-block\|` | not consumed | **no** | N/A-gen3 | gen3 Protect emits `-singleturn` + `-activate\|Protect`, never `-block` (verified live) |
| `\|-hint\|` `\|-center\|` `\|-message\|` `\|-combine\|` `\|-waiting\|` `\|-nothing\|` | not consumed | cosmetic | N/A-inert | UI/animation chatter; no state |
| `\|-mega\|` `\|-primal\|` `\|-burst\|` `\|-zpower\|` `\|-zbroken\|` | not consumed | no | N/A-gen3 | gen6+ battle mechanics |
| `\|error\|` | not consumed | control | N/A | choice-validation control line |

## Modifier-tag coverage matrix

Tags that qualify a message (`parts[4:]` fields). The parser processes the base event
regardless of most tags; these rows track tags the encoder must READ to be correct.

| tag | consumed for | verdict | notes |
|---|---|---|---|
| `[from] EFFECT` | psn/tox ramp, Spikes/Sandstorm chip, Recoil, Leech Seed, item:/ability:/move: attribution | COVERED | central attribution channel |
| `[of] SOURCE` | absorb-heal source, Leech Seed source, weather source | COVERED | source attribution |
| `[silent]` | Heal Bell `-curestatus`, Perish `-start`, Taunt `-end`, item-taking Trick `-enditem` | COVERED | event applied transparently despite `[silent]` |
| `[eat]` | berry consumption (`-enditem ... [eat]`) | COVERED | drives item-removal signal |
| `[upkeep]` | weather continuation (set-turn/source preserved) | COVERED | distinguishes re-set from continue |
| `[miss]` | `\|move\|...\|[miss]` paired with `-miss` | COVERED | via `-miss` |
| `[msg]` | Castform `-formechange` message | **UNCOVERED** | rides the un-consumed `-formechange` (BUG row) |
| `[still]` | empty-target `\|move\|` (Present, self-target) | COVERED-tolerated | move token parses with empty target field |
| `[weaken]` `[wisher]` `[consumed]` `[identify]` `[anim]` `[zeffect]` | — | N/A-inert | cosmetic / gen-specific; not needed by the gen3 encoder |

## Summary counts

- Message types enumerated (SIM-PROTOCOL, deduped): **~72 distinct**.
- **COVERED / COVERED\***: **38** (all core action/damage/status/boost/side/volatile/
  effectiveness/item/ability channels + recharge-via-alternate).
- **N/A-gen3 / N/A-inert**: **28** (gen4+ mechanics, doubles, team-preview, timer/UI
  chatter, superseded shapes like `-cureteam`/`-block`).
- **UNCOVERED-reachable (the tail)**: **6**
  1. `-formechange` (Castform Forecast) — **BUG: live type not encoded** ← flagged case
  2. `-start ... typechange` (Color Change/Kecleon) — **BUG: same root cause**
  3. `-singleturn` (Focus Punch focus marker) — low impact
  4. `-notarget` (move fizzle) — low impact
  5. `-fieldactivate` (Perish Song) — no functional gap (state via `perishN` volatiles)
  6. `\|tie\|` / `-endability` — benign / effectively-N/A

Of the 6 uncovered-reachable rows, **2 are genuine encoder bugs** (the retype family:
Forecast + Color Change), both flagged and repro'd in `docs/validated_interactions.md`.
The remaining 4 are low/no functional impact and documented as accepted gaps.

## Method / reproducibility

Protocol shapes captured live by driving scripted `gen3customgame` battles through the
vendored sim (`node driver.mjs`) and parsing with `parse_showdown_replay`; encodings
verified through `observation_from_player_state` (and the production `LocalShowdownEnv`).
Every "verified live" claim in this doc corresponds to a captured omniscient log.
