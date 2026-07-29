# Engine divergence ledger — 2026-07-28

Authoritative, **measured** baseline of what still differs between the vendored
gen3-patched Rust poke-engine and the reference Pokemon Showdown simulator, as
of `origin/main` @ `941c86d`. This is the zero-divergence program's starting
line: every row was produced by running a harness on this machine today, and
every claim carries its repro.

Nothing here was fixed. Measurement only.

> ## ⚠ THIS IS A PRE-FIX BASELINE
>
> Every number in this document was measured against `941c86d`, when the
> vendored engine carried **5** gen3 patches. While this ledger was being
> written, `main` merged the fixes for D1, D2 and D3 and the engine now carries
> **11** gen3 patches, listed in the shared
> `third_party/poke-engine-gen3-patches.txt` that both the wheel builder
> (`setup_poke_engine.sh`) and the crate vendorer
> (`vendor_poke_engine_src.sh`) read (`096c4c1` — which also closes the
> wheel-drift hazard flagged in §0).
>
> **These rates are the "before" column. Do not quote them as current.**
> Re-run §1's harnesses against current `main` to get the "after".

### Fixed since this baseline

| Ledger row | Fix | Landed |
| --- | --- | --- |
| **D1** Spikes 2/3-layer damage | `5696ac1` *poke-engine gen3: Spikes layer fractions (1/8, 1/6, 1/4)* | PR #874 (`7ce6fc9`), re-merged #876 (`a991eb7`) |
| **D2** faint-ply residual deferral | `3207b63` *poke-engine gen3: defer end-of-turn residuals past a forced replacement* | PR #874 (`7ce6fc9`) / #877 (`ab21b25`) |
| **D3** Transform (the move) | `eb1272e` *implement Transform (Ditto) in rollouts* + `234d011` adapter base-form carry + `b1a671d` State-serialization fixed point | PR #878 (`2f17cff`) |
| **D3** Transform (world constructor) | `046f58f` *engine-world: express Ditto Transform instead of failing closed on it* | PR #872 (`94c2498`) |
| — | `31b8130` *bound the CONFUSION duration and carry it on Baton Pass* (found independently, not on this ledger) | PR #875 (`0a792b2`) |
| — | `e7db150` *carry the Perish Song counter through Baton Pass* | PR #874 |

Both D1 and D2 shipped with Showdown differential scenarios and regression pins
(`7872cc9`, `ba2f747`), so the classes this ledger measured are now gated.

**Still open at time of writing:** PR #880 (trapper Baton-Pass bug, in the fix
loop) and PR #881 (fixture refresh, queued last). Rows **D7–D11** below were
*not* addressed by the merged PRs and remain live.

## 0. Build provenance (what "the engine" means in this document)

| Component | Value |
| --- | --- |
| pokezero commit | `941c86d` (worktree `scott/divergence-ledger-20260728`) |
| poke-engine | 0.0.47 sdist + **five** gen3 patches, in `scripts/vendor_poke_engine_src.sh` order: residual-order, attract, struggle-typeless, rapidspin-fidelity, ability-fidelity (`--fuzz=0`) |
| Python wheel | rebuilt today via `scripts/setup_poke_engine.sh` into a **dedicated** venv (`.venv` of this worktree) |
| Native crate | `rust/pokezero-search` built via `maturin build --release` against `third_party/poke-engine-src/` |
| Showdown | `~/workspace/pokerena/vendor/pokemon-showdown` (Node v22.22.2) |

**Provenance warning that materially changed this run.** The shared engine venv
(`~/workspace/agents/pokezero-engine/.venv`) carries a **stale** wheel: its
`poke_engine.pyi` is missing the `gender` parameter that the current
`poke-engine-gen3-ability-fidelity.patch` adds, because that checkout sits at
`35c4158`, before the patch existed. Any fidelity number taken from that venv
measures a different engine than `main` builds. All numbers below come from a
freshly built wheel; reproducing them requires the same rebuild.

Pre-flight gates, both green before any harness result was trusted:

```
python -m unittest tests.test_engine_attract_immobilization tests.test_engine_gen3_abilities \
  tests.test_engine_rapidspin_fidelity tests.test_engine_residual_order \
  tests.test_engine_struggle_typeless tests.test_poke_engine_adapter \
  tests.test_poke_engine_backend tests.test_poke_engine_outcomes tests.test_engine_world \
  tests.test_poke_engine_legal_actions
  -> Ran 263 tests in 4.4s, OK

cd rust/pokezero-search && cargo test --release
  -> 28 passed; 0 failed
```

## 1. Harness inventory — exact invocations, N, runtime

All commands run from the repo root with `PYTHONPATH=src` and this worktree's
`.venv/bin/python`; `SHOWDOWN=~/workspace/pokerena/vendor/pokemon-showdown`.

| # | Harness | Invocation | N | Runtime | Result |
| --- | --- | --- | --- | --- | --- |
| H1 | `pokezero.engine_fidelity` (one-turn curated) | `python -m pokezero.engine_fidelity --showdown-root $SHOWDOWN --out oneturn.json` | 15 mechanics x 8 seeds = 120 turns | 35.9 s | **15/15 clean** |
| H2 | `pokezero.engine_fidelity_multiturn` (curated trajectories) | `python -m pokezero.engine_fidelity_multiturn --showdown-root $SHOWDOWN --out multiturn.json` | 6 cases x 4 seeds, 3–7 steps = 24 trajectories | 8.9 s | **6/6 clean** (24/24 seed trajectories, all expected traces) |
| H3 | `scripts/rapidspin_differential.py` | `python scripts/rapidspin_differential.py --showdown-root $SHOWDOWN --seeds 60` | 7 scenarios x 60 seeds = 420 sim runs | 109.4 s | **PASS** (all 7 scenarios agree) |
| H4 | `scripts/attract_differential.py` | `python scripts/attract_differential.py --showdown-root $SHOWDOWN --seeds 200` | 2 scenarios x 200 seeds = 400 sim runs | 98.5 s | **PASS** (within 4σ; activate line 200/200 both scenarios) |
| H5 | `pokezero.engine_search` (fallback census) | `python -m pokezero.engine_search --showdown-root $SHOWDOWN --games 60 --seed-start 930000 --worlds 4 --search-time-ms 20 --out engine_mcts_fallback.json` | 60 games, 1436 decisions, 6296 world constructions | 133 s | fallback **3.20 %** (46/1436); 736 world-construction failures |
| H6 | `scripts/engine_transition_differential.py` (**new**, game-level) | see §3.1 | 500 (approx-sleep) + 400 (strict) + 60 (attribution) games | see §3.1 | **8.16 divergent transitions/game** (§3.2) |
| H7 | `scripts/oracle_differential.py` (encoder oracle, not engine) | `python scripts/oracle_differential.py --games 300 --seed 940000 --max-steps 200 --json oracle_300.json` | 300 games | see §3.4 | see §3.4 |

`scripts/fidelity_gate_events.py` was **not** run: it requires a prebuilt
golden-corpus-v2 fold sidecar (`--corpus`), and no such corpus exists in this
checkout. It measures the instruction→event *mapper*, not engine mechanics, so
it is not on this ledger's critical path; building a corpus for it is a
follow-up, not a blocker.

### 1.1 Why a new harness (H6) was written

`docs/engine_fidelity_findings.md` lists the tier-2 real-game sweep — "replay
recorded decision points through `engine_world` and check each observed
Showdown outcome lies in the engine's branch support" — under **Next**. It did
not exist. H1/H2 are curated-fixture harnesses with hard-coded teams, scripted
choices and 4–8 seeds; they cannot be scaled to N games and cannot produce a
per-game divergence rate, which is exactly what the 10,000-game acceptance
criterion needs. `scripts/engine_transition_differential.py` fills that gap:

* plays whole `gen3randombattle` games in the real Node sim with uniform-random
  legal actions;
* at **every** full decision boundary rebuilds the engine state through the
  **production** constructor (`engine_world.world_battle_spec`) with the game's
  true packed teams as a fixed `BattleStartOverride` — no belief sampling, and
  the public item / Transform / Encore signals derived by the production
  `EngineMctsPolicy._public_effect_signals`;
* gates on an **exact** pre-state match (HP, status, weather, side-condition
  presence) before scoring the transition, so constructor error is never
  charged to the engine's transition model;
* checks that the transition Showdown actually took lies inside
  `generate_instructions`' branch support for the same joint action.

## 2. Divergence ledger

Rates are per *measured* boundary unless a per-game column says otherwise.
`n/game` = divergent transitions per game in the 2000-game strict sweep (§3).

| # | Mechanic | Divergence | Rate | Repro | Status |
| --- | --- | --- | --- | --- | --- |
| D1 | **Spikes, 2 and 3 layers** | Engine deals `maxhp * layers / 8`; gen3 Showdown deals `[0,3,4,6][layers] * maxhp / 24` (1/8, **1/6**, **1/4**). Engine over-damages by 1.5x at both 2 and 3 layers. | **0.42 / game** (approx, 500 g) · **0.22 / game** (strict, 400 g); **0.37–0.53 %** of measured boundaries | `third_party/poke-engine-src/src/gen3/generate_instructions.rs:254` vs `pokemon-showdown/data/moves.ts` spikes `damageAmounts = [0, 3, 4, 6]`; direct probe in §2.1; game repro seed 900001 (steps 73/76/85/101) | NEW at `941c86d`; **FIXED** by `5696ac1` (PR #874/#876) |
| D2 | **End-of-turn residuals on a faint ply** | When a Pokemon faints mid-turn, Showdown emits **no** residual block and no `|upkeep|`: Leftovers/toxic/sand/Leech Seed are deferred until *after* the forced replacement switches in. poke-engine runs the full residual block in the **same** instruction set as the faint. | **4.26 / game** (approx, 500 g) · **3.16 / game** (strict, 400 g); **5.33 % / 5.39 %** of measured boundaries — cross-validated on disjoint seeds; 52–68 % of all divergence | trace in §2.2 (seed 911003) | NEW at `941c86d`; **FIXED** by `3207b63` (PR #874/#877) |
| D3 | **Transform (the move)** | The gen3 engine has **no** implementation of Transform. `Choices::TRANSFORM` exists in the shared move table but `grep -rn TRANSFORM third_party/poke-engine-src/src/gen3/` returns **nothing**. Clicking `transform` returns two 50 % branches with **empty** instruction lists — Ditto copies nothing and the state is unchanged. `PokemonVolatileStatus::TRANSFORM` is accepted by the `Side` constructor and stored, then wholly ignored. | 74 % of world-construction failures in H5 (576/736); ~0 in H6 (the true-team override never has to express it) | probe in §2.3 | **FIXED** by `eb1272e`+`234d011`+`b1a671d` (PR #878) and `046f58f` (PR #872) |
| D4 | **Rapid Spin hazard-clear through Protect** | Claimed: `remove_effects_for_protect` clears effect fields but not `move_id`, so the move-id-keyed gen3 hazard clear still fires through Protect. | 0/60 seeds — no divergence | `scripts/rapidspin_differential.py --seeds 60`, scenario `protect` | **fixed-since** |
| D5 | **Leech Seed removal on switch-out** | Claimed unimplemented. | 0 — engine emits the removal | probe in §2.4 | **cannot-reproduce** |
| D6 | **Partial trap (Wrap-class) removal** | Claimed unimplemented, both directions. | 0 — both directions handled | probe in §2.4 | **cannot-reproduce** |
| D7 | **`trapped` request flag vs sampled world** | Showdown reports the acting mon `trapped`, but the belief-sampled world's foe has a non-trapping ability, so the engine world cannot reproduce the constraint and fails closed. | 26 % of H5 world-construction failures (160/736) | H5 `world_failure_reasons` | **confirmed-current** (belief/coverage gap, not an engine mechanic bug) |
| D8 | **Hidden sleep-turn counters** | Sleep turns remaining are not public. Strict construction fails closed (`status_unsupported`); the approximation exists but is a guess. | 31 % of full-round boundaries unmeasurable under strict mode (§3.2) | H6 counter `skip:world_unsupported:status_unsupported` | **confirmed-current** (observability gap, not an engine bug) |
| D9 | **Encore duration** | gen3 Showdown rolls 3–6 turns; the engine applies the volatile with `volatile_status_durations.encore` stuck at 0 and never expires it. | not separately rated (Encore fails closed in world construction) | `docs/engine_fidelity_findings.md` §"Engine caller-contract sharp edges" 3; still true in the vendored source | **confirmed-current** (pre-existing, documented) |
| D10 | **Wish heal amount** | Engine heals the *resolving active's* maxhp/2; gen3 heals the *caster's* maxhp/2. | unmeasured (needs a caster-switches-out fixture) | `docs/engine_fidelity_findings.md` §"Known engine deviation" | **confirmed-current** (pre-existing, documented, low severity) |
| D11 | **Sub-band damage error** | The matcher's ±16 %-of-this-turn's-damage band cannot see a systematic damage bias smaller than the band. | unmeasurable by construction | `docs/engine_fidelity_findings.md` "Scope clarification" | **open measurement gap** |

### 2.1 D1 repro — Spikes layer damage

Source, `third_party/poke-engine-src/src/gen3/generate_instructions.rs:252-255`:

```rust
if side.side_conditions.spikes > 0 && switched_in_pkmn.is_grounded() {
    let dmg_amount = cmp::min(
        switched_in_pkmn.maxhp * side.side_conditions.spikes as i16 / 8,
```

Showdown, `data/moves.ts` (`spikes.condition.onSwitchIn`):

```ts
const damageAmounts = [0, 3, 4, 6]; // 1/8, 1/6, 1/4
this.damage(damageAmounts[this.effectState.layers] * pokemon.maxhp / 24);
```

Direct engine probe (maxhp 240):

| layers | engine damage | Showdown damage | ratio |
| --- | --- | --- | --- |
| 1 | 30 (0.1250) | 30 (0.1250) | 1.00 |
| 2 | **60 (0.2500)** | 40 (0.1667) | **1.50** |
| 3 | **90 (0.3750)** | 60 (0.2500) | **1.50** |

Live-game repro (seed 900001, boundary 76): Jirachi (maxhp 266) switches into
two Spikes layers. Showdown `|-damage|p2a: Jirachi|222/266|[from] Spikes`
(−44 = 266/6) then Leftovers `238/266`. The engine's only branch lands at 216
(= 266 − 66 + 16), i.e. it charged 266/4.

Spikes is heavily on-distribution in gen3 randbats (Skarmory, Forretress,
Cloyster, Deoxys-Defense …), and two layers is the common stall-game state, so
this is not a corner.

### 2.2 D2 repro — residual deferral across a faint

Seed 911003, two consecutive plies of the same Showdown battle
(`|request|`/`|t:|` lines stripped):

```
### FAINT PLY (both seats acted)
|move|p1a: Dragonite|Hidden Power|p2a: Corsola
|-resisted|p2a: Corsola
|-damage|p2a: Corsola|229/266
|move|p2a: Corsola|Ice Beam|p1a: Dragonite
|-supereffective|p1a: Dragonite
|-damage|p1a: Dragonite|0 fnt
|faint|p1a: Dragonite                       <- turn ends here: no residuals, no |upkeep|, no |turn|

### NEXT PLY (p1 force switch only)
|switch|p1a: Suicune|Suicune, L74|270/270
|-ability|p1a: Suicune|Pressure|[silent]
|-heal|p2a: Corsola|245/266|[from] item: Leftovers   <- the deferred residual block
|upkeep
|turn|19
```

poke-engine resolves the whole turn — including the residual block — inside the
single `generate_instructions` call that produced the faint. Consequences:

1. the engine's post-state at a faint ply already carries residuals Showdown
   has not applied yet, so **every KO turn is a divergent transition** under a
   ply-for-ply comparison;
2. in Showdown the **replacement** is on the field when the residuals run, so
   the incoming mon is exposed to sand/hail chip and Leftovers; in the engine
   the fainted mon is still in the slot;
3. the two sims **reconverge** after the force-switch ply, which is why this
   never showed up in the curated harnesses (none of them script a faint
   followed by a scored step) and why it does not corrupt search value
   materially — but it *does* mean an engine-as-environment driver would emit
   states the real simulator never visits.

### 2.3 D3 repro — Transform

```
grep -rn "TRANSFORM" third_party/poke-engine-src/src/gen3/     -> (no matches)
grep -n  "TRANSFORM" third_party/poke-engine-src/src/choices.rs -> 18012, 18014, 20366 (move table + enum only)
```

```python
>>> br = poke_engine.generate_instructions(state, "transform", "splash")
TRANSFORM branches: 2
   pct 50.0 insts []
   pct 50.0 insts []
   ditto after: id=DITTO types=('NORMAL','TYPELESS') atk=100 moves=['TRANSFORM','TACKLE','NONE','NONE'] vols=set()
>>> pe.Side(..., volatile_statuses={'TRANSFORM'}).volatile_statuses
{'TRANSFORM'}          # accepted and stored, zero behavioural references in gen3
```

This is distinct from PR #872 (`scott/engine-world-transform`, `046f58f`),
which teaches the **constructor** to express an already-transformed state. That
PR removes the fail-closed wall; it does not give the engine the Transform
*move*, so a Ditto in a search rollout still cannot click Transform and a
transformed Ditto in a k=0 engine environment cannot be produced at all.

Fallback attribution (H5, 736 world-construction failures over 60 fresh games,
seeds 930000-930059):

| share | reason |
| --- | --- |
| 52.2 % | `public_effect_blocked: slot 'p2': active transformed into Slowking` |
| 13.0 % | … `into Exploud` |
| 8.7 % | … `into Sneasel` |
| 26.1 % | `self_request_state_unsupported: ['trapped']` (12 distinct foe abilities) |

Transform = **73.9 %**, `trapped` = **26.1 %**, nothing else. `Trick`,
`flashfire`, `maybeTrapped` and Knock-Off — the causes named in `604c4c7` — did
**not** appear on these seeds; `removed_item_decisions = 293` shows the
item-removal path is exercised heavily and no longer walls, so PR #871's
discharge fix is holding.

Measured fallback rate is **3.20 %**, not the 2.45 % on record. The difference
is seed set and world budget (`--worlds 4 --search-time-ms 20`), not a
regression: the residual is the same two causes.

### 2.4 D5/D6 repro — Leech Seed and partial trap on switch

Direct engine probes (all on the freshly built wheel):

| scenario | engine instructions | verdict |
| --- | --- | --- |
| seeded mon switches out | `RemoveVolatileStatus SideOne: LEECHSEED`, `Switch SideOne: P0 -> P1` | removal implemented |
| trapped victim switches out | volatiles empty after switch | removal implemented |
| **trapper** switches out | `RemoveVolatileStatus SideOne: PARTIALLYTRAPPED`, `Switch SideTwo: P0 -> P1` | victim correctly freed |

Source: the switching side's own volatiles fall through the `_ => false`
catch-all in `remove_volatile_statuses_on_switch`
(`third_party/poke-engine-src/src/gen3/state.rs:688-750`; `LEECHSEED` is
retained only under Baton Pass, which is correct for gen3), and the opposite
side's `PARTIALLYTRAPPED` is removed during switch generation
(`gen3/generate_instructions.rs:171-186`).

Note on the trapping *option surface*: `generate_instructions` will happily
resolve a switch supplied for a `PARTIALLYTRAPPED` mon — it does **not**
validate legality — but `Side::trapped()` (`gen3/state.rs:431`) does include
`PARTIALLYTRAPPED` and *is* consulted by `get_all_options`
(`gen3/state.rs:615/643`), so the engine's own enumeration is correct. Only
`root_get_all_options`' extra filter is keyed on the separate `force_trapped`
flag.

### 2.5 Caller-contract findings (not engine bugs, but they break callers)

* **gen3 switch choices are BARE species ids.** `MoveChoice::from_string`
  (`third_party/poke-engine-src/src/gen3/state.rs:51-75`) resolves a switch by
  matching `pkmn.id` directly; the `"switch <species>"` form raises
  `ValueError: Invalid move for sN`. `pokezero.engine_fidelity_multiturn.
  engine_step_choices` (`src/pokezero/engine_fidelity_multiturn.py:128`) emits
  `f"switch {species}"` on the **non**-force-switch path. That path is dead in
  the curated suite (its only switches are force switches, which correctly use
  the bare form), so it has never fired — but any reuse of that helper on
  ordinary switch boundaries fails immediately. Cost this run: 92 spurious
  `engine_error`s before the harness was corrected.
* Confirms the two sharp edges already in `docs/engine_fidelity_findings.md`
  (force-switch resolution must re-supply the postponed move;
  `Side(last_used_move=...)` takes an index, not a move id).

## 3. Game-level multi-turn transition differential (H6)

### 3.1 Runs

| Run | Mode | Seeds | N | Wall | Throughput | Outcome |
| --- | --- | --- | --- | --- | --- | --- |
| A | strict sleep | 910000–911999 | 2000 | — | — | **killed** by the task supervisor at ~1 h 40 m before it could write its JSON (the harness only serialises at the end — see §3.5) |
| B | approximate sleep | 920000–920499 | 500 | 1268 s | **1419 games/h** (3-way contention) | complete — the widest-coverage dataset, headline below |
| C | strict sleep | 950000–950399 | 400 | 956 s | **1506 games/h** (solo) | complete |
| D | strict sleep | 960000–960059 | 60 | 146 s | 1479 games/h | complete — engine-error attribution only (§3.2.1) |

```bash
# Run B
PYTHONPATH=src .venv/bin/python scripts/engine_transition_differential.py \
  --showdown-root "$SHOWDOWN" --games 500 --seed-start 920000 \
  --approximate-sleep --keep-repro 40 --json transition_approx_500.json

# Run C
PYTHONPATH=src .venv/bin/python scripts/engine_transition_differential.py \
  --showdown-root "$SHOWDOWN" --games 400 --seed-start 950000 \
  --keep-repro 60 --json transition_strict_400.json
```

### 3.2 Run B results — 500 games, approximate sleep (PRE-FIX BASELINE)

| Metric | Value |
| --- | --- |
| games | 500 |
| full-round (both-seats-act) boundaries | 40,765 |
| measured (world constructible **and** pre-state exact) | 39,954 — **98.0 %** |
| transitions matched | 35,471 |
| **transitions diverged** | **4,079** |
| harness engine-errors (not divergences — §3.2.1) | 404 |
| **divergent transitions per game** | **8.16** |
| divergence rate per measured boundary | **10.21 %** |

Per-class, ordered by weight. Classes are **protocol-evidence buckets, not
exclusive attributions**: they say what the step contained, not what is at
fault.

| Class | Count | **Per game** | Per measured boundary | Share | Maps to |
| --- | --- | --- | --- | --- | --- |
| `faint_ply_residual_deferral` | 2131 | **4.262** | 5.33 % | 52.2 % | **D2** |
| `status_support` | 961 | 1.922 | 2.41 % | 23.6 % | D8 (sleep guess) + D2 spill |
| `status_residual` | 357 | 0.714 | 0.89 % | 8.8 % | D2 / residual ordering |
| `damage_band` | 262 | 0.524 | 0.66 % | 6.4 % | D11 (band residue) |
| `spikes_entry_damage` | 212 | **0.424** | 0.53 % | 5.2 % | **D1** |
| `boost_delta_support` | 84 | 0.168 | 0.21 % | 2.1 % | mixed |
| `crit_roll_band` | 50 | 0.100 | 0.13 % | 1.2 % | D11 |
| `faint_boundary` | 22 | 0.044 | 0.06 % | 0.5 % | D2 |

**The two new findings' pre-fix rates, stated plainly:**

* **D1 — Spikes 2/3-layer over-damage: 0.424 divergent transitions per game**
  (212 in 500 games; 0.53 % of measured boundaries).
* **D2 — faint-ply residual deferral: 4.262 divergent transitions per game**
  (2131 in 500 games; 5.33 % of measured boundaries) — the single largest
  class, **52 % of all divergence**.

Skip / fail-closed accounting (Run B):

| Bucket | Count | Note |
| --- | --- | --- |
| `skip:single_seat_boundary` | 4451 | force-switch plies; out of scope by design |
| `skip:world_unsupported:public_effect_blocked` | 250 | Transform (**D3**) |
| `skip:world_unsupported:volatile_unsupported` | 195 | |
| `skip:world_unsupported:self_moveset_mismatch` | 160 | Transform/Mimic desync (**D3**) |
| `world_prestate_mismatch` | 109 | 0.27 % of full rounds (p1_hp 30, p2_hp 42, p1_status 10, p2_status 27) |
| `skip:world_unsupported:self_request_state_unsupported` | 76 | **D7** |
| `skip:world_unsupported:volatile/materialization_blocker` | 18 | PR #871's fix holding |
| `skip:world_unsupported:encore_move_unknown` | 3 | **D9** |

#### 3.2.1 The 404 engine-errors are harness defects, not engine bugs

Run D (60 games, seeds 960000–960059) attributes all 75 of its engine-errors,
every one a `ValueError: Invalid move for sN`:

| Choice that failed | Count | Cause |
| --- | --- | --- |
| `struggle` | 56 (75 %) | the harness passes `struggle` through to `generate_instructions`, but gen3 `MoveChoice::from_string` resolves **only** moves on the active's move list — Struggle is engine-internal, never a submittable choice |
| `unownm` / `unownk` | 19 (25 %) | `EngineWorld.party_species` reports the lettered Unown forme while the built state stores the collapsed `unown`, so the switch string does not resolve |

Both are harness defects and are **excluded** from the divergence rate above.
Production search is unaffected by the second one: it maps engine option
strings → action indices, i.e. it reads ids *from* the engine rather than from
`party_species`.

### 3.2.2 Run C — 400 games, strict sleep (the coverage/fidelity trade)

Strict mode refuses to guess a hidden sleep counter and fails the world closed
instead. It is the cleaner measurement and the poorer sample:

| Metric | Run C (strict, 400 games) | Run B (approx, 500 games) |
| --- | --- | --- |
| full-round boundaries | 32,984 | 40,765 |
| **measured** | 23,409 — **71.0 %** | 39,954 — **98.0 %** |
| `status_unsupported` skips | **9,071** (27.5 % of full rounds) | 0 (approximated) |
| matched | 21,324 | 35,471 |
| diverged | 1,858 | 4,079 |
| divergent transitions / game | **4.65** | 8.16 |
| divergence rate / measured boundary | **7.94 %** | 10.21 % |
| harness engine-errors | 227 | 404 |
| `world_prestate_mismatch` | 52 (0.16 % of full rounds) | 109 (0.27 %) |

Per-class, both runs, normalised **per measured boundary** — this is the
apples-to-apples column:

| Class | Run C (strict) | Run B (approx) | Agreement |
| --- | --- | --- | --- |
| `faint_ply_residual_deferral` (**D2**) | 1262 → **5.39 %** | 2131 → **5.33 %** | **tight** — independent seed sets, 1 % apart |
| `spikes_entry_damage` (**D1**) | 87 → **0.37 %** | 212 → **0.53 %** | same order |
| `status_support` | 240 → 1.03 % | 961 → 2.41 % | 2.3x lower under strict — confirms most of this class was the sleep guess (**D8**), not an engine fault |
| `status_residual` | 117 → 0.50 % | 357 → 0.89 % | |
| `damage_band` | 97 → 0.41 % | 262 → 0.66 % | |
| `boost_delta_support` | 24 → 0.10 % | 84 → 0.21 % | |
| `crit_roll_band` | 16 → 0.07 % | 50 → 0.13 % | |
| `faint_boundary` | 15 → 0.06 % | 22 → 0.06 % | |

Two things this cross-check buys:

1. **D2's rate is real and seed-independent** — 5.39 % vs 5.33 % of measured
   boundaries across two disjoint seed ranges (950000-950399 and
   920000-920499). It is 68 % of all divergence under strict mode.
2. **The `status_*` classes are mostly D8, not engine bugs.** They fall by
   ~2.3x the moment the harness stops guessing sleep counters, which is exactly
   what you would expect if the guess — not the engine — was wrong.

**D8 is the coverage ceiling.** Strict mode can only see 71 % of full-round
boundaries. A "0 divergent transitions over 10,000 games" claim measured in
strict mode is a claim about 71 % of the game; measured in approximate mode it
is contaminated by ~1.4 % of boundaries where the harness guessed. Neither is
sufficient on its own — see §5 item 5.

### 3.3 Throughput

| Configuration | games/h |
| --- | --- |
| single process, solo (Run C 400 games: **1506**; Run D 60 games: 1479; 10-game calibrations: 1735/1925) | **~1500 sustained**, ~1700 on short runs |
| single process, 3-way contention (Run B, 500 games) | **1419** |

The sustained figure is the honest one: short runs over-report because long
games are under-sampled. **Use 1500 games/h for sizing.**

Sizing math and the sharding recipe are in §5.1.

### 3.4 Oracle differential (H7) — encoder axis, not the engine

`scripts/oracle_differential.py --games 300 --seed 940000 --max-steps 200`
— 300 games, 50,498 decisions, 580,730 mon-observations, ~13 min. This audits
the **production observation encoder** against the vendored JS engine's own
`State.serializeBattle`; it is a different axis from the engine ledger and is
recorded for completeness. Exit code 1 = its divergence gate firing, not a
crash.

Clean at rate 0.00000: `active`, `hp_fraction`, `level`, `present`,
`legal_switchable(fainted)`, all five `boost/*`, `field/turn`,
`field/weather_cat`, `field/{self,opp}_{hazards,screens}`,
`sleep_turns(elapsed)`, `toxic_stage`, `status_cat/self`,
`partC/belief_active`, `actual_stat/hp`, `base_stat/hp`.

Divergent:

| Column family | Rate | Attribution |
| --- | --- | --- |
| `status_cat/opp` | 28,065 / 277,742 = **10.1 %** | 100 % `fainted_mon_retains_stale_status` (low-sev): a fainted mon keeps its pre-faint status in the observation — the same faint/status conflation the H6 matcher has to carve out |
| `base_stat/{atk,def,spa,spd,spe}` | 166–181 / 580,730 ≈ 0.03 % | consistent with Transform (**D3**) — a transformed mon reports copied base stats |
| `actual_stat/{atk,def,spa,spd,spe}` | 87 / 205,154 ≈ 0.04 % | same |
| `belief_over_pruned_true_move` | 46 | known candidate over-pruning class (Mew) |

Part B invariant violations: `tox_implies_stage_ge_1` 90,
`partC/true_move_pruned_offscript` 46,
`monotonicity/candidate_set_count_increased` 5, `legal_mask/legal_move_zero_pp` 2.

### 3.5 Harness limitation found the hard way

`engine_transition_differential.py` serialises its report **only at the end**,
so Run A's ~1 h 40 m of work was lost when the process was stopped. Before the
10,000-game acceptance run, add incremental checkpointing (append per-game
counter rows to a JSONL sidecar) — otherwise a single interruption costs the
whole shard.

## 4. Engine-as-environment driver inventory

**Answer: no. No engine-as-environment collection driver exists in this repo,
not even a partial or experimental one.** `engine_selfplay`, `engine_env`,
`EngineEnv` and "engine environment" have **zero** hits across `src/`,
`scripts/`, `tests/` and `rust/`. Every self-play game the repo can produce is
advanced by the Node Showdown `BattleStream`.

### 4.1 What exists (verified against this checkout)

* **One** concrete `PokeZeroEnv`: `src/pokezero/local_showdown.py:276`
  `LocalShowdownEnv`. It hard-requires the built Node simulator —
  `local_showdown.py:1204` raises unless `<showdown_root>/dist/sim/index.js`
  exists, and `:1209` requires the `node` binary on PATH. There are 29
  `LocalShowdownEnv(...)` construction sites across 13 modules
  (`rollout_cli`, `selfplay_cli`, `bootstrap_cli`, `linear_cli`, `neural_cli`,
  `refutation_cli`, `foulplay_bridge`, `golden_corpus`,
  `golden_corpus_scenarios`, `selfplay_protocol_capture`, `hazard_audit`,
  `checkpoint_factors`, `engine_search`) and **no** alternative implementation.
  The only other things named `*Env` in `src/pokezero/` are the two Protocols
  in `env.py` (`PokeZeroEnv:67`, `AsyncPokeZeroEnv:93`).
* `src/pokezero/engine_search.py` `EngineMctsPolicy` is a **policy**, not an
  environment: its `main()` builds a `LocalShowdownEnv` + `RolloutDriver` and
  plugs the engine in as p1's searcher. Showdown still advances every state.
* `src/pokezero/engine_cli.py` exposes exactly one subcommand — `doctor`
  (`engine_cli.py:23`). No game loop.
* `src/pokezero/engine_fidelity_multiturn.py` is the only code that chains
  `apply_instructions` across steps, and it is measurement-only: it selects the
  branch that **matches Showdown**, i.e. it depends on the very thing an env
  would have to replace.
* **k=0 is already supported and is NOT the blocker.** The knob is
  `transition_token_budget` (`src/pokezero/observation.py:147`; default
  `TRANSITION_TOKEN_COUNT = 128` at `:80`), CLI `--transition-token-budget`
  (`rollout_cli.py:180`). Zero is explicitly legal — `observation.py:151-152`:
  *"0 is a valid budget: the transition region exists but is fully masked
  (Markov-state-only ablations). The encoder fill/mask paths handle it."*
  Validation is `0 <= budget <= TRANSITION_TOKEN_COUNT` (`:154`).

### 4.2 What is missing for a k=0 engine-only collection driver

1. **An `EngineEnv` implementing the `PokeZeroEnv` protocol**
   (`src/pokezero/env.py:67` — `reset` / `observe` / `legal_actions` /
   `requested_players` / `step` / `terminal`). Nothing exists.
2. **A gen3 randbats team generator that does not shell out to Showdown.** The
   only path today is `local_showdown.py:399` `generate_scenario_team`, an RPC
   over the Node bridge. `randbat.py:225` `Gen3RandbatSource` enumerates
   candidate *sets* for belief, not legal 6-mon parties. Biggest single gap.
3. **A payload-free `BattleSpec` constructor.** `engine_world.
   battle_spec_from_payload` and `world_battle_spec` both consume
   `local_showdown._public_materialization_payload` output, i.e. they can only
   *mirror* a live Showdown battle. The primitives (`unpack_team`,
   `_build_pokemon_spec`) are reusable; the wiring is not.
4. **PyO3 exports for legal-action enumeration.**
   `src/pokezero/poke_engine_legal_actions.py` probes
   `ENGINE_OPTION_PROVIDER_CANDIDATES = ('get_all_options',
   'root_get_all_options', 'get_root_options')` (`:41-43`) and reports
   `supported=False` when none is present. Runtime check on the freshly built
   0.0.47 wheel: **none of the three is exported** on `poke_engine` or
   `poke_engine.State`. The Rust crate does have them
   (`rust/pokezero-search/src/lib.rs:247`, `model.rs:385`, `leaf.rs:1821`), so
   adding a `pokezero_search` pyfunction is cheaper than patching
   `poke-engine-py`.
5. **A PyO3 export for `battle_is_over()`** — used only from Rust today
   (`tree.rs:517/552/733`, `lib.rs:266`, `model.rs:430/523/728`) — plus
   turn-cap handling to build `TerminalState(winner, turn_count, capped)`.
6. **A seeded chance-node sampler.** `generate_instructions` returns a
   *distribution*; an env must sample one branch under a reproducible seed.
   Nothing in Python does this — every existing consumer either enumerates all
   branches or picks the one that matches Showdown.
7. **Forced-switch / asymmetric request semantics.** `engine_world` fails closed
   on exactly these (`boundary_not_move_request`, `pending_baton_pass`), and
   **D2** means the engine's ply decomposition at a faint does not match
   Showdown's. An env must pick one convention and own it — see the owner
   decision in §5.2.
8. **An engine-native observation producer.** Even at k=0 the Markov region
   (field / self-mons / opponent-mons / action candidates / tendency stats) is
   built from Showdown protocol in `showdown.py`.
   `rust/pokezero-search/src/leaf.rs:1755` `PyLeafEncoder` already encodes
   engine states natively for leaf eval and is the seam to repurpose.
9. **CLI surface.** No `--mode` / `--backend` / `--driver` on `rollout_cli.py`,
   `selfplay_cli.py` or `engine_cli.py`.

Net: items 1, 2, 6 and 8 are real build work; 4 and 5 are small binding
exports; 3 is a refactor of existing code; 7 is blocked on an owner decision.

## 5. Recommended fix order and acceptance criterion

**Acceptance criterion (unchanged): 0 divergent transitions over 10,000
fresh-seed `gen3randombattle` games**, measured by
`scripts/engine_transition_differential.py`, with the measured fraction of
full-round boundaries reported alongside — a run that skips a third of its
boundaries has not earned the number.

Ordered by (incidence x fixability), cheapest decisive win first. **Items 1–3
have since been fixed on `main`** — see the banner at the top of this document;
they are retained here because the ordering rationale is the evidence for why
they were taken first.

1. ~~**D1 — Spikes 2/3-layer damage.**~~ **DONE** (`5696ac1`). A one-line table fix in a new
   `third_party/poke-engine-gen3-spikes-layers.patch`
   (`maxhp * layers / 8` → `[0,3,4,6][layers] * maxhp / 24`). Deterministic, no
   probability surface, on-distribution, and it removes an entire divergence
   class. Gate: a direct layer-damage unit test plus the `spikes_entry_damage`
   class going to zero in H6.
2. ~~**D2 — faint-ply residual deferral.**~~ **DONE** (`3207b63` — option (a) was taken: the engine now defers). Structural and the largest class. Two
   options: (a) teach the engine to stop at the faint and run the residual
   block on the following resolution (matches Showdown exactly, but changes the
   engine's turn contract and every consumer of `generate_instructions`); or
   (b) declare the engine's single-ply convention authoritative and make the
   *harness* compare across the force-switch ply. (a) is required if an
   engine-as-environment driver must be state-identical to Showdown; (b) is
   sufficient if the engine stays a search backend. **This is an owner
   decision, not an implementation detail** — it should be taken before any
   engine-env work starts.
3. ~~**D3 — Transform.**~~ **DONE** (`eb1272e` + `046f58f`). Implement the gen3 volatile (stat/move/type/ability copy,
   PP=5 on copied moves, cleared on switch) in the vendored engine, then land
   PR #872 on top. Removes ~74 % of remaining world-construction failures and is
   a hard prerequisite for a k=0 engine env, since Ditto is in the pool and the
   env cannot fail closed.
4. **D7 — `trapped` flag vs sampled world.** *(still open)* The remaining ~26 %. Verify the
   flag against the built world's `Side::trapped()` conditions instead of
   refusing on sight, and add the missing gen3 trapping sources (Mean Look /
   Block / Spider Web have no engine volatile at all).
5. **D8 — hidden sleep counters.** *(still open)* Not an engine bug; it is why strict-mode
   coverage caps out (§3.2). For the acceptance run, either accept the
   approximation with the divergence measured under both modes (both are
   reported in §3) or restrict the criterion to boundaries with no publicly
   asleep mon. For an engine env this evaporates — the env owns the counter.
6. **D11 — the damage band.** *(still open — and it now gates the acceptance claim)* Before declaring zero divergence, replace the
   ±16 %-of-net-HP matcher with a per-damage-source comparison (the engine's
   instruction list already itemises every `Damage`/`Heal`). Until then "0
   divergent transitions" means "0 outside a 16 % band", which is not the same
   claim.
7. **D9/D10 — Encore duration, Wish heal amount.** *(still open)* Known, bounded, cheap; fold
   in once the above are clear.

### 5.1 Sizing the 10,000-game acceptance run

Measured single-process throughput of `engine_transition_differential.py` on
this machine (18 cores, macOS, one Node bridge per process): **~1700 games/hour**
— sustained **1506 games/h** over Run C's 400 games; 1419 games/h under 3-way
contention (Run B). Short calibration runs report 1618-1925 games/h but
under-sample long games. **Size with 1500.**

* Single process: 10,000 / 1500 = **6.7 hours**. Too slow.
* Each shard costs roughly **two** cores (one Python + one Node), so ~8 shards
  is the practical ceiling on an 18-core box before per-shard throughput
  collapses.

Recommended recipe — **8 shards × 1250 games**, expected **~1.1–1.5 h** wall
clock. Measured contention scaling: 1506 games/h solo, 1419 games/h at 3-way
(−6 %). Extrapolating to 8-way on 18 cores gives roughly 1100–1200 games/h per
shard, so 1250 / 1150 ≈ **1.1 h** plus startup:

```bash
for k in 0 1 2 3 4 5 6 7; do
  PYTHONPATH=src .venv/bin/python scripts/engine_transition_differential.py \
    --showdown-root "$SHOWDOWN" \
    --games 1250 \
    --seed-start $((1000000 + k * 100000)) \
    --keep-repro 100 \
    --json "acceptance_shard_${k}.json" &
done
wait
```

Shard *k* consumes seeds `1000000 + k*100000` .. `+1249`, i.e. blocks
`1000000-1001249`, `1100000-1101249`, … `1700000-1701249`. The 100k stride
(rather than a contiguous 1000000-1009999 span) leaves each shard 98,750 unused
seeds for re-runs after a fix without ever colliding with a prior acceptance
attempt. Conservative alternative: **6 shards × 1667 games**, ~1.3–1.6 h. Either way the
run fits comfortably under 2 h, which is the point.

Aggregate the shards by summing `boundaries_measured`,
`transitions_matched`, `transitions_diverged`, `engine_errors` and the
`divergence_classes` histogram; the criterion is met only when
`transitions_diverged + engine_errors == 0` **and** the aggregate
`measured_fraction_of_full_rounds` is reported (a run that skips a third of its
boundaries has not earned the number — see D8).

### 5.2 Seeds already burned — exclude from the acceptance run

Every seed below has been consumed by measurement on this branch and must not
appear in the acceptance run, or the run is no longer fresh-seed:

| Seed range | Consumer |
| --- | --- |
| 11–18, 21–26 | curated fixture seeds (H1, H2) |
| 770000–770199 | attract differential (H4) |
| 880000–880059 | rapidspin differential (H3) |
| 900000–900009 | H6 calibration / debug runs |
| 910000–911999 | **H6 strict 2000-game sweep** (incl. 911003, the D2 trace) |
| 920000–920499 | H6 approximate-sleep 500-game sweep |
| 930000–930059 | H5 engine-MCTS fallback census |
| 940000–940299 | H7 oracle differential |
| 950000–950399 | H6 strict 400-game sweep (Run C) |
| 960000–960059 | H6 engine-error attribution (Run D) |
| 7000–7099 | historical `engine_search` bench seeds (pre-existing) |

**Seeds burned during fix development must be added to this table.** The fixes
for D1/D2/D3 are in flight on `scott/engine-gen3-divergence-fixes` (PR #874) and
its stacked children `scott/engine-gen3-spikes-residuals`,
`scott/engine-gen3-transform`, `scott/engine-gen3-confusion-duration`. To keep
this simple: **fix development should stay below seed 1,000,000**, which
reserves the entire `>= 1,000,000` space as a pristine acceptance block. If any
fix branch has already used seeds at or above 1,000,000, shift the shard bases
to `2000000 + k*100000` instead.
