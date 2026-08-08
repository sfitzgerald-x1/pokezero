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
| D9 | **Encore duration** | gen3 Showdown rolls 3–6 turns; the engine applied the volatile with `volatile_status_durations.encore` stuck at 0 and never expired it. | not separately rated | `docs/engine_fidelity_findings.md` §"Confirmed deviation 5" | **FIXED** — `third_party/poke-engine-gen3-encore-duration.patch`; hazard ladder over the existing counter, pinned by `rust/pokezero-search/tests/gen3_encore_fidelity.rs` + differential scenarios `encoreduration` / `encoreoutlivesshortest` / `encoredurationslow` / `encoredurationcontrol`. Correction to this row's original rating: Encore does **not** fail closed in world construction — `engine_world` supports it and fails closed only in the `encore_move_unknown` sub-case, so the divergence was reaching search whenever the locked move was derivable. |
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
    --seed-start $((2000000 + k * 100000)) \
    --keep-repro 100 \
    --json "acceptance_shard_${k}.json" &
done
wait
```

Shard *k* consumes seeds `2000000 + k*100000` .. `+1249`, i.e. blocks
`2000000-2001249`, `2100000-2101249`, … `2700000-2701249`. The 100k stride
(rather than a contiguous span) leaves each shard 98,750 unused seeds for
re-runs after a fix without ever colliding with a prior acceptance attempt.
Conservative alternative: **6 shards × 1667 games**, ~1.3–1.6 h. Either way the
run fits comfortably under 2 h, which is the point.

> **Base moved from 1,000,000 to 2,000,000.** The hardening pass (Appendix A)
> itself burned 1200000-1200299, which sits inside what would have been shard 2.
> Reserving `>= 2,000,000` keeps the acceptance block pristine; measurement and
> fix-development runs should stay **below** it.

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
for D1/D2/D3 landed on `scott/engine-gen3-divergence-fixes` (PR #874) and its
stacked children; any seeds those branches consumed belong here too. Appendix A
adds its own (listed at the end of that appendix, including 1200000-1200299).
Rule going forward: **measurement and fix development stay below seed
2,000,000**, reserving `>= 2,000,000` as the pristine acceptance block.

---

# Appendix A — hardening pass (same day, after PRs #874–#878 merged)

Branch `scott/differential-hardening`. Everything above is the pre-fix baseline
at `941c86d` with 5 gen3 patches; this appendix re-measures against `84f60f4`
with **11** gen3 patches, resolves the three rows the engine fixes did not touch
(D9/D10/D11), and closes the harness gaps the baseline run exposed.

Rebuild provenance: `vendor_poke_engine_src.sh` + `setup_poke_engine.sh` both
driven from the shared `third_party/poke-engine-gen3-patches.txt`, then
`maturin build --release`. Gates: **275 engine tests OK**, crate builds clean.

## A.1 D1/D2/D3 confirmed fixed in live games

300 fresh games (seeds 1200000–1200299) against `84f60f4`, support-based gating
on, **1679 games/h**, **0 harness errors**, 22,938 of 23,574 full-round
boundaries measured (**97.30 %**):

| Metric | Pre-fix baseline | Post-fix (this run) |
| --- | --- | --- |
| measured fraction of full rounds | 71.0 % strict / 98.0 % approx | **97.3 %** |
| divergent transitions per game | 4.65 strict / 8.16 approx | **2.14** |
| divergence rate per measured boundary | 7.94 % strict / 10.21 % approx | **2.80 %** |

Per-class, normalised per measured boundary — the apples-to-apples column
against §3.2.2:

| Class | Pre-fix (strict) | Post-fix | Change |
| --- | --- | --- | --- |
| `faint_ply_residual_deferral` (**D2**) | 5.39 % | **0.275 %** | **−95 %** |
| `spikes_entry_damage` (**D1**) | 0.37 % | **0.070 %** | **−81 %** |
| `status_support` | 1.03 % | 1.225 % | — (now measured over 3.4x more sleep boundaries) |
| `status_residual` | 0.50 % | 0.650 % | — |
| `damage_band` | 0.41 % | 0.419 % | flat (D11 band residue) |
| `boost_delta_support` | 0.10 % | 0.083 % | — |
| `crit_roll_band` | 0.07 % | 0.052 % | — |
| `faint_boundary` | 0.06 % | 0.026 % | — |

D1 and D2 are the two classes that moved, by 81 % and 95 %, which is the fix
landing. **They did not go to exactly zero** — the classes are protocol-evidence
buckets, not causal attributions, so any divergence that happens to occur on a
faint ply or a Spikes ply still lands in them. The residue is now the same order
as the other classes rather than 52–68 % of all divergence.

`public_effect_blocked` (Transform) world failures are **gone** from the skip
histogram entirely, confirming the D3 fix; `self_moveset_mismatch` (285) is the
remaining Transform/Mimic-shaped skip.

## A.2 D8 RESOLVED — support-based validation for hidden-counter mechanics

**The old bar was wrong in principle.** Strict mode demanded that the
constructed world reproduce a counter no observer can see. gen3 Showdown rolls
sleep duration privately (`data/mods/gen3/conditions.ts` `slp.onStart`:
`this.effectState.time = this.random(2, 6)`), and the engine does not even store
"turns remaining" — it models wake-up as a hazard conditioned on turns already
slept, `chance_to_wake_up(t) = 1/(1 + MAX_SLEEP_TURNS - t)` with
`MAX_SLEEP_TURNS = 4`
(`third_party/poke-engine-src/src/gen3/generate_instructions.rs:44-71`), which
reproduces the uniform 1–4 turn duration exactly. Requiring counter-state match
threw away 27.5 % of boundaries to measure nothing.

**The new bar.** For hidden-counter mechanics only, build one world per legal
counter assignment and require the realized transition to lie in the **union of
those worlds' branch supports with nonzero probability**.

> **Scope corrected (review of PR #887).** The first version of this section
> overstated what shipped. As merged, only SLEEP's counter was actually swept,
> the recoverable-reason set included the generic `volatile_unsupported`, and the
> retry flipped the whole approximate bundle — which admitted yawn,
> partial-trap and substitute-health approximations whose effects (a sleep
> landing, a chip tick) are **observable**, contradicting the rule this mode
> rests on. Appendix B fixes all three: confusion's counter is now swept for
> real, and the retry widens **only** the mechanic that caused the failure.

| Mechanic | Swept domain | Why hidden |
| --- | --- | --- |
| Sleep | `sleep_turns` 0..`MAX_SLEEP_TURNS` (4), plus `rest_turns` 1..2 (Rest is a separate fixed-duration engine path) | duration rolled privately at `slp.onStart` |
| Confusion | `volatile_status_durations.confusion` 0..`MAX_CONFUSION_TURNS` (4) | snap-out count is private; the engine prices it as a hazard ladder since PR #875 |

Everything else keeps **exact** gating — damage, status application, hazards,
screens, weather, boosts and faints are all publicly observable and their
divergence bar is unchanged.

**Before/after, 150 games on identical seeds (981000–981149):**

| Metric | Before (fail-closed) | After (support-based) |
| --- | --- | --- |
| full-round boundaries | 12,807 | 12,807 |
| **measured** | 9,171 — **71.61 %** | 12,601 — **98.39 %** |
| `status_unsupported` skips | **3,453** | **0** |
| boundaries gated exact | 9,171 | 9,171 |
| boundaries gated support-based | 0 | 3,430 |
| transitions matched | 8,927 | 12,241 |
| transitions diverged | 194 | 290 |

**+26.8 points of coverage**, 3,430 boundaries recovered. The recovered
boundaries are not a free pass: they diverge at 96/3,430 = **2.80 %**, slightly
*above* the 2.12 % rate of the exactly-gated ones, so the weaker bar is still
catching real disagreement rather than waving boundaries through.
`--no-hidden-counter-support` reproduces the old behaviour for A/B.

## A.3 D9 / D10 / D11 verdicts

New instrument: `scripts/gen3_duration_differential.py`. Each scenario reads the
gen3 rule out of the **vendored** simulator first — house rule: *gen3 inherits
gen4, not gen5*.

| Row | Mechanic | Verdict | Rate / evidence | Repro |
| --- | --- | --- | --- | --- |
| **D9** | Encore duration | **CONFIRMED** | Showdown ends Encore after **3, 4, 5 or 6** turns (60/60 samples span the full documented range); the engine's `ENCORE` volatile **never expires** — still present after 12 turns. 100 % divergence on any trajectory that outlives the shortest roll. | `--scenario encore --seeds 60 --seed-start 991000` |
| **D10** | Wish heal amount | **CANNOT-REPRODUCE** — the claim had the generations backwards | Showdown healed exactly **201 = recipient maxhp/2** (403 max) in 6/6 unclamped samples. Engine healed **200 = recipient maxhp/2** (400 max). They agree. | `--scenario wish --seeds 10` |
| **D11** | Sub-band damage bias | **CANNOT-REPRODUCE** | Across 4 matchups (n = 40–57 each) the engine's representative roll sits within **±1 %** of Showdown's sampled mean: +0.50 %, +0.99 %, −0.48 %, −0.77 %. Max |bias| **0.0099**. | `--scenario damage --seeds 60` |

### D9 detail

gen3 Encore is `durationCallback() { return this.random(3, 7); }`
(`data/mods/gen3/moves.ts`) = 3–6 turns. Measured Showdown lock lengths: `{3, 4,
5, 6}` over 60 seeds — the full distribution. The engine applies the volatile
and never removes it; `volatile_status_durations.encore` stays 0. This is a
**real, unfixed divergence** and is the only confirmed engine bug left on this
ledger. It stays largely latent in search only because `engine_world` fails
Encore closed at construction (`encore_move_unknown`), i.e. the engine is
protected from the bug by refusing to express the state at all.

*Measurement trap worth recording:* the first revision of this scenario let p1
attack, every battle ended by step 5, and only the shortest (3-turn) Encores
were ever observed — the measured distribution collapsed to a single value and
looked like a clean rule. p1 must be harmless for the duration to be observable.

### D10 detail — the inherited claim was wrong

`docs/engine_fidelity_findings.md` recorded "poke-engine ignores the `wish`
tuple's amount and heals the RESOLVING ACTIVE's maxhp/2; gen3 heals by the
CASTER's maxhp/2". The first half is true, the second is the **gen5+** rule:

* base `data/moves.ts` `wish.condition.onStart` → `this.effectState.hp = source.maxhp / 2` (caster) — gen5+;
* `data/mods/gen4/moves.ts` `wish.condition.onEnd` → `this.heal(target.baseMaxhp / 2)` (**recipient**) — and **gen3 inherits gen4**.

So the engine's behaviour is correct for gen3. It does store a caster-based
amount (`SetWish SideTwo: 100` for a 200-max caster) and then ignore it, healing
`Heal SideTwo: 200` for the 400-max recipient — inert bookkeeping, right answer.
The differential pins both sims at recipient/2. **D10 is struck from the ledger.**

### D11 detail — and a warning about how it is measured

No systematic sub-band bias exists. But the first revision of this scenario
reported biases of **+12 % to +39 %**, which were entirely an artifact: the
engine branch's post-state HP is a **net** figure (damage minus the end-of-turn
Leftovers heal) while Showdown's `|-damage|` line is **gross**, so every
Leftovers holder manufactured a fake ~maxhp/16 bias. Making every fixture mon
itemless collapsed the bias to ≤1 %.

That is the same failure mode D11 warns about, pointed the other way: a net-HP
comparison can invent bias as easily as it can hide it. The acceptance-run
caveat stands — "0 divergent transitions" still means "0 outside a ±16 % band"
until the matcher compares per-damage-source instead of net HP.

## A.4 Harness hardening

* **JSONL checkpointing** (`--checkpoint PATH`): one record per completed game,
  written and flushed as it finishes, so a supervisor kill loses at most the
  in-flight game. This is the §3.5 fix — Run A lost 1 h 40 m of work.
* **`--resume`**: skips seeds already present in the checkpoint and folds their
  counters into the final report.
* **`--merge-from A.jsonl B.jsonl …`**: aggregates any number of shards into one
  report with the **same schema** the single-process path emits, de-duplicating
  seeds and warning if shards overlap. This is the aggregator for the 8×1250
  acceptance run in §5.1.
* Torn final lines (hard kill mid-write) are discarded with a warning rather
  than failing the load.
* **Harness defects from §3.2.1 are fixed**, taking `engine_errors` to **0**:
  `recharge` now maps to the engine's `"none"` choice (a MUSTRECHARGE seat has
  no submittable move), `struggle` is a counted skip (it is engine-internal,
  never a choice string), and switch targets resolve against the **built
  state's** `pkmn.id` rather than `EngineWorld.party_species` — the two disagree
  for cosmetic Unown formes (`unownb` vs the collapsed `unown`).

## A.5 What is left

| Row | Status after this pass |
| --- | --- |
| D1, D2, D3 | fixed on `main`, confirmed absent in live games |
| D4, D5, D6 | already cannot-reproduce |
| D7 | closed by PR #880 + #884 |
| **D8** | **resolved** — support-based validation, 71.6 % → 98.4 % coverage |
| **D9** | **CONFIRMED, unfixed** — Encore never expires. Engine-lane fix; the bar is a duration counter and expiry, mirroring the CONFUSION work in PR #875. |
| **D10** | **struck** — claim was the gen5 rule; engine is correct for gen3 |
| **D11** | **cannot-reproduce** — no bias beyond ±1 %; the ±16 % band caveat on the acceptance claim still stands |

Only **D9** requires engine work. The acceptance criterion is unchanged: 0
divergent transitions over 10,000 fresh-seed games, now measurable over ~98 % of
full-round boundaries instead of ~71 %.

Seeds burned by this appendix (add to §5.2): 970000–970005, 971000–971002,
980000–980024, 981000–981149, 990000–990059 (wish/damage), 991000–991059
(encore), 1200000–1200299.

---

# Appendix B — strict per-source matcher and residue root-cause (13-patch main)

Branch `scott/differential-strict-matcher`, measured against `a2a081a`
(main + #887) with the engine re-vendored to **13** gen3 patches. **These
numbers supersede Appendix A's**, which were taken on an 11-patch tree.
Gate: 205 engine/env tests OK on the rebuilt wheel.

## B.1 The ±16 % band is gone

The banded matcher compared **net active HP** within ±16 % of the turn's damage.
Appendix A.3 already proved that unsound in one direction — a Leftovers tick
riding on an attack manufactured a 12–39 % fake "bias". The strict matcher
replaces it with a **per-damage-source** comparison:

* **Showdown side** — every `|-damage|`/`|-heal|` line carries its own
  attribution (`[from] psn`, `[from] Sandstorm`, `[from] item: Leftovers`,
  `[from] Leech Seed`, `[from] Spikes`, `[from] Recoil`, …). A bare `-damage`
  is direct move damage; a bare `-heal` is a move heal.
* **Engine side** — the same vocabulary, produced by the shipped
  instruction→event mapper (`pokezero_search.branch_events`, PR #727), which
  renders a branch's instruction list as protocol lines. Comparing
  rendered-vs-real protocol keeps both sides in one vocabulary instead of
  guessing at positions in the instruction list.

**Exact vs roll-scaled.** Every deterministic component — status residuals,
weather chip, Leftovers, Leech Seed, hazards, move heals — must match **to the
HP point**, with no tolerance of any kind.

Roll-scaled components (direct move damage, recoil, drain, confusion self-hit)
are accepted when they are, in order: **equal** to the engine's value, **or** a
member of the enumerated legal roll set (gen3 computes
`floor(base * random(85,100) / 100)` and `poke_engine.calculate_damage` returns
the base, so the set is enumerable), **or within ±9 % of the engine's
representative roll**.

> **That third rung is a band, and the earlier wording here ("membership is
> checked, not proximity") over-claimed.** The legal set is computed from the
> PRE-state with an assumed move order, so it is unreliable whenever the turn
> reorders or a same-turn stat change moves the base — it therefore never
> vetoes, only accepts. What is genuinely gone is the **net-HP** band: every
> remaining tolerance is scoped to one roll-scaled component, and no
> deterministic component gets any tolerance at all. This is the second
> over-claim of this shape (the first was A.2's D8 scope); the standing
> correction is to **describe the implemented predicate, not the intended one**.

`--matcher banded` keeps the old behaviour for continuity.

### Old vs new on the same 150-game seed set (981000–981149)

| Metric | `--matcher banded` | `--matcher strict` |
| --- | --- | --- |
| full-round boundaries | 12,807 | 12,807 |
| measured | 12,601 (98.39 %) | 12,601 (98.39 %) |
| matched | 12,311 | 11,715 |
| **diverged** | **290** | **886** |
| **rate per measured boundary** | **2.30 %** | **7.03 %** |
| diverged per game | 1.93 | 5.91 |

**The band was hiding 596 divergences — 67 % of the real total.** Coverage is
identical, so this is purely the bar tightening.

## B.2 Residue root-cause (300 games, seeds 1310000–1310299)

97.20 % of full-round boundaries measured, 1,889 divergences, 6.30/game,
**7.61 % per measured boundary**. Classes below are from the 120 sampled repros
(6.4 % of divergences, `--keep-repro` capped) — proportions are sample-based,
counts are not the population.

| Class | Share of sample | Verdict | Repro |
| --- | --- | --- | --- |
| `magnitude:psn` (toxic ladder off by 0–(stage−1) HP) | **66/120 (55 %)** | **REAL ENGINE DIVERGENCE → fix lane** (B.3) | seed 1310000 step 143 |
| `magnitude:heal` (Showdown move-heal vs engine Leftovers-only) | 13/120 | **UNRESOLVED** — could not isolate within budget; two live hypotheses in B.4 | seed 1310001 steps 72, 95 |
| `mixed:itemleftovers|movewish` (same amount, different label) | 8/120 | **HARNESS — mapper mis-attribution** (B.5) | seed 1310002 step 35 |
| `missing_in_engine:itemleftovers` / `magnitude:itemleftovers` | 8/120 | residue of the faint-ply residual ordering (D2 edge cases) | seed 1310005 step 15 |
| `magnitude:movewish` | 4/120 | same family as the mapper mis-attribution | seed 1310008 step 93 |
| `missing_in_engine:spikes` | 4/120 | **NEEDS ENGINE CHECK** — Spikes tick absent entirely (distinct from D1's wrong fraction, which is fixed) | seed 1310018 steps 47, 48 |
| `mixed:itemleftovers,psn|psn` | 3/120 | compound of the toxic bug + Leftovers ordering | seed 1310000 step 125 |
| `mixed:heal|itemleftovers` | 3/120 | same family as `magnitude:heal` | seed 1310001 step 72 |
| `missing_in_engine:psn` / `extra_in_engine:psn` | 4/120 | faint-ply residual-ordering residue | seed 1310005 step 1 |
| `missing_in_engine:leechseed` | 2/120 | low incidence, needs check | seed 1310009 step 25 |
| `roll_scaled` | 2/120 | genuine roll disagreement, low incidence | seed 1310003 step 26 |
| `magnitude:brn` | 1/120 | same rounding family as the toxic bug | seed 1310000 step 193 |

Separately counted, **not** divergences: `strict:lossy_render` 490 branches the
mapper self-reports it cannot render (excluded), of which only **2 boundaries**
had *every* branch lossy and became skips. That is the honest ceiling of the
mapper-rendered approach.

**No Encore-class divergence appears in the residue.** D9 is being fixed in the
engine lane; `engine_world` still fails Encore closed at construction
(`encore_move_unknown`, 1 skip in 300 games), so it never reaches the matcher.

## B.3 CONFIRMED: toxic residual rounds the wrong way

The single largest residue class, and a **new** finding the ±16 % band could
never have surfaced — it is a 1–3 HP error.

* **Showdown** (`data/conditions.ts` `tox.onResidual`, no gen3 override):
  `this.damage(this.clampIntRange(pokemon.baseMaxhp / 16, 1) * this.effectState.stage)`.
  `clampIntRange` floors its argument first, so this is
  **floor(maxhp / 16) × stage** — *floor then multiply*.
* **Engine**: measured **floor(maxhp × stage / 16)** — *multiply then floor*.

Direct probe (`toxic_count = n`, engine tick vs both formulas):

| maxhp | n | engine | multiply-then-floor | floor-then-multiply | differs |
| --- | --- | --- | --- | --- | --- |
| 470 | 2 | 88 | 88 | 87 | **yes** |
| 470 | 3 | 117 | 117 | 116 | **yes** |
| 503 | 2 | 94 | 94 | 93 | **yes** |
| 317 | 1 | 39 | 39 | 38 | **yes** |
| 317 | 3 | 79 | 79 | 76 | **yes (3 HP)** |
| 451 | 1–3 | 56 / 84 / 112 | same | same | no (maxhp ≡ 0 mod 16) |

The two agree only when `maxhp % 16 == 0`. Error grows with the ladder, up to
`stage − 1` HP. Toxic is ubiquitous in gen3 randbats, so this is on-distribution
on every stall turn. **Hand to the engine lane**; the same floor-then-multiply
rule should be checked for burn (`magnitude:brn` above is likely the same bug).

## B.4 UNRESOLVED: `magnitude:heal`

Showdown shows a large bare `-heal` (251, 136 HP — Recover/Soft-Boiled/Rest
shape) where the engine branch shows only a Leftovers tick. Two hypotheses, not
separated within this pass:

1. **mapper** — a move heal is rendered without the event the extractor expects,
   so the component is invisible on the engine side;
2. **engine/world** — the heal move genuinely did not execute in any enumerated
   branch (e.g. a sleep-sweep candidate where the mon cannot act).

Both are cheap to settle by dumping the full rendered event list for seed
1310001 step 72 next to the Showdown slice. Flagged rather than guessed.

## B.5 HARNESS: mapper mis-attributes a Leftovers heal as Wish

`observed_only=[('itemleftovers', 18)] engine_only=[('movewish', 18)]` —
**identical amount, different label**, and the amount is maxhp/16 (Leftovers),
not maxhp/2 (Wish). The engine's state transition is right; the mapper's
`[from]` tag is wrong. Fix belongs in `rust/pokezero-search/src/events.rs`, not
the engine. Until then it inflates the strict divergence count by ~10 % of the
sample.

## B.6 Review fixes folded in

| Item | Fix |
| --- | --- |
| **MEDIUM** — D8 scope claim wrong | Confusion's counter is now genuinely swept (`_confusion_counter_variants`, rungs 0..4). The retry widens **only** the failing mechanic: `status_unsupported` → sleep alone; `volatile_unsupported` → confusion **only when confusion is the sole unsupported volatile**, because yawn rides the same `engine_world` flag and yawn's sleep landing is observable. A.2's text is corrected in place. Measured cost: **zero** — `hidden_counter_support:sleep` fires 215×, confusion 0×, and coverage is unchanged at 98.39 %, confirming the review's note that the entire win was always sleep. |
| **LOW** — `_mon()` defaults to Leftovers | The D11 runner now **asserts** every damage fixture is itemless and fails loudly with the net-vs-gross rationale, so the artifact cannot be silently reintroduced. (`_mon`'s default is left alone: the curated one-turn cases deliberately hold Leftovers to exercise residual ordering.) |
| **LOW** — `load_checkpoint` skipped any bad line | Now enforces the documented rule: only the **final** line may be torn; a mid-file parse failure raises instead of silently shrinking the shard's denominator. |
| **Staleness** | These 13-patch numbers are the cited state; Appendix A's 11-patch figures are superseded. |
| *(found while re-running)* D11's verdict flipped with sample size | It compared a SAMPLE mean against a point value with a fixed 1 % threshold, so the same fixtures read `cannot-reproduce` at n=60 (bias 0.0099) and `confirmed` at n=20 (bias 0.017). The tolerance is now **3 standard errors of the mean**, floored at 1 % — stable across N, and the verdict stays `cannot-reproduce` at both. |

## B.7 Bottom line for the acceptance run

The bar is now strict per-source. On current main the residue is
**7.61 % of measured boundaries**, and it is no longer unexplained: one
confirmed engine bug (B.3) accounts for ~55 % of the sample, one mapper
mis-attribution (B.5) ~10 %, faint-ply ordering residue ~10 %, and one
unresolved class (B.4) ~13 %.

**The 8×1250 acceptance run should not be launched yet.** Zero divergence is
unreachable while B.3 is unfixed, and the measurement itself would be
contaminated by B.5. Order: fix B.3 (engine lane) → fix B.5 (mapper) → settle
B.4 → re-measure → then run acceptance from seed 2,000,000.

Seeds burned by this appendix: 981000–981149 (A/B, re-used deliberately),
1300000–1300009, 1310000–1310299.

---

# Method rule — for an upstream field, acceptance proves nothing

**A constructor accepting a field is not evidence the engine honours it. Only
ROUND-TRIP SURVIVAL is: set the field, serialize, re-parse, and confirm it is
still there and still doing something.**

Third recurrence of this shape, hence a standing rule:

| Field | What "accepted" hid |
| --- | --- |
| `wish` amount | stored and then ignored — the engine heals the resolving active's half regardless |
| `TRANSFORM` volatile | accepted by `Side(...)` and stored, with **zero** behavioural references in gen3 |
| Rest-provenance / attempt counts | accepted, but the semantics differed from the field's name |

The trap: `pe.Side(volatile_statuses={"TRANSFORM"})` raises nothing and the value
reads back, so the field looks supported. It is inert. An upstream binding will
happily accept anything its type signature allows, and a value that is stored but
never read is indistinguishable from a value that works — until a differential
disagrees and the cause is hunted somewhere else entirely.

The check that actually discriminates:

1. set the field on a state;
2. `state.to_string()` -> `State.from_string(...)` and confirm it survives;
3. drive a transition that the field must change, and confirm the instruction
   stream differs from the field-absent case.

Step 3 is the one that matters. Steps 1-2 pass for inert fields too.

Companion to the Dex-resolution rule below: both are cases where the obvious
check returns a clean answer to a question nobody asked.

# Method rule — resolve gen3 data through the Dex, never read base data

**Read move and condition data through `Dex.mod('gen3')`. Never read
`data/moves.ts` directly, and never hand-walk the mod chain.**

`scripts/gen3_dex_resolve.py` is the probe:

```
$ python scripts/gen3_dex_resolve.py toxic spikes flail thunderwave
resolved through Dex.mod('gen3') — moves
  toxic          accuracy=85  ...
  spikes         accuracy=True ...   condition callbacks: onEntryHazard, ...
  flail          accuracy=100 ...    callbacks: basePowerCallback
  thunderwave    accuracy=100 ...
```

This rule exists because the inheritance-chain trap has produced **five** wrong
reads in this program, each of which reached a hand-off before being caught:

| Move / condition | The trap |
| --- | --- |
| Spikes | layer fractions live in the **gen4** mod, not base |
| burn | residual fraction changed at **gen6**, so base is wrong for gen3 |
| Flail / Reversal | gen3 has its **own** override; the gen4 ladder is not it |
| Thunder Wave | **gen6** declares a value BELOW gen3 in the chain |
| Rest (PP) | **gen8** declares it; neither gen3 nor base carries the value |
| `stall` (Protect decay) | gen4's `inherit: true` resolves to **gen5's full definition**, not base |

### The fifth direction, and why it is the strongest case for probing

`stall` is the one where two independent expert readers resolved the same
`inherit: true` differently, in the same review cycle:

* **Reading A** (engine lane, then the reviewer independently): gen3 -> gen4
  `{ inherit: true, counterMax: 8 }` -> **base** (`counter = 3`, `counter *= 3`),
  giving a 1/3, 1/9 ladder — and a puzzled note about the gen4 comment saying
  1/8 when the arithmetic says 1/9.
* **Reading B**: gen4's `inherit: true` resolves to **gen5**, which carries a
  *full* definition (`counter = 2`, `counter *= 2`), giving 1/2, 1/4, 1/8.

The measurement decides it. 160 seeds, real sim, gen3:

| attempt | measured | Reading A (1/3ⁿ) | Reading B (1/2ⁿ) |
| --- | --- | --- | --- |
| 1st | 546/546 = 1.000 | 1.000 | 1.000 |
| 2nd | 239/496 = **0.482** | 0.333 | **0.500** |
| 3rd | 40/204 = **0.196** | 0.111 | **0.250** |

The 2nd-attempt figure is ~7σ from Reading A. **Reading B is correct**, and the
engine's existing `0.5 ** n` was right all along — a "fix" derived from Reading A
would have replaced correct behaviour with wrong behaviour.

Note what made this trap resistant to the existing probe: for `stall`, base and
gen5 both define `onStart`, `onRestart` and `onStallMove`, so
`gen3_dex_resolve.py --condition stall` lists the identical callback *names*
either way and the reader must still hand-walk to find whose *body* survives —
the exact step the rule says not to do by hand.

**Proposed resolver extension:** report the surviving callback's source file
(the mod that actually contributed the body), not just its name. That single
addition would have prevented both misreads here, and is the only trap of the
five the current probe cannot already close.

**IMPLEMENTED (cycle six)** as `gen3_dex_resolve.py --sources`. It prints the mod
owning each field and each callback **body**, and it reproduces the documented
answer for every provenance trap unaided:

```
stall  -> inherit: true at gen4;  callback bodies from **gen5**   (the x2 ladder)
flail  -> callback bodies from **gen3**                           (own 48-scale override)
brn    -> callback bodies from **gen6**: onResidual               (the gen6 change)
rest   -> values from **gen8**: pp                                (declared below gen3)
```

One line — `callback bodies from **gen5**` — is what both readers of the fifth
trap were missing.

**A second trap found while building it, worth more than the feature.**
`Dex.mod()` merges inherited data **into the required data objects in place**,
and `require` caches those objects. So any raw-table read performed after a
`Dex.mod()` call *in the same process* reports the merged result: with `Dex`
touched first, gen3 appears to own `stall` outright — `counterMax: 8` plus all
three callbacks — when `data/mods/gen3/conditions.ts` does not mention `stall`
anywhere. That is a **silent** wrong answer that looks authoritative, and it is
the exact shape of the misread it was built to prevent. The scan therefore runs
in a Dex-free process, pinned by a comment at `_SOURCE_JS` so it is not
"simplified" back into one process later.

The failure is always the same shape and always looks like diligence: someone
opens the right file, reads a real value, and reports it — having walked a chain
that was truncated, or not walked one at all. "gen3 inherits gen4, not gen5" is a
useful heuristic and is **not** a substitute for resolution: Flail and Thunder
Wave both satisfy the heuristic and both were still read wrong.

Two corollaries:

* The resolved value is the simulator's, chain fully applied — it is *the*
  answer, not evidence toward one.
* Where the rule lives in a **callback body** (`basePowerCallback`,
  `durationCallback`, `onResidual`), the probe lists which callbacks survive
  resolution. Read the body in source — but only after the probe has told you
  **which file's** version survives, which is exactly the step the four misreads
  skipped.

# Process gap — repro states are captured but not retained

`scripts/replay_residue.py` replays recorded `engine_state` values out of a
report, so the harness *does* capture what replay-first triage needs. What is
missing is retention: **no cycle-five report is committed**, and none is
reachable from the repo. Asked to replay a specific recorded row
(seed 1500010 step 39), the engine lane could not — the ledger carries the prose
finding but not the state, and `find` turns up no artifact.

The consequence is that "replay before you label" silently degrades to
"regenerate, then replay", which is sound only when the build and seed are
pinned. Regeneration on a pinned build + seed is acceptable provenance;
narrating from a previous cycle's prose is not.

**Proposal:** persist repro states by default. The differential already has
`--checkpoint`, which writes one JSONL record per game and is crash-safe; the
cheap fix is to make acceptance runs always pass it and to commit (or archive
alongside the ledger) the resulting checkpoint for the seeds a cycle reports on.
The states are small relative to the report and are the only artifact that makes
a row re-examinable after main advances.

Related, from the same review: **record the repo commit SHA next to every build
fingerprint.** A fingerprint alone stops being verifiable once main moves, and
`--check` does *not* detect a stale wheel after a re-vendor restamp — so the
behavioural probe stays mandatory regardless of what `--check` says.

# Method rule — replay before you label

**Triaging a residue class starts by replaying the recorded boundary through the
engine. The class label is a hypothesis about a symptom; the replay is the
experiment.** This is step one, not the fallback after a diagnosis fails.

The rule was earned three times over. #893 re-attributed a "spikes" class to
Whirlwind from branch-count reasoning; #895 then disproved that same lead by
replay before finding the real bug (Protect blocks phazing) during source
verification. Applying it here immediately disproved two more of my own labels —
`roll_scaled_component` turned out to be an extractor bug hiding the primary
move damage (C.7), and a second instance of the same label turned out to be a
32 % damage disagreement rather than roll noise.

    PYTHONPATH=src python scripts/replay_residue.py --report <report>.json \
        [--seed N] [--step N] [--class <substring>] [--limit K]

It deserializes every recorded candidate `engine_state` (the whole
hidden-counter sweep — those states ARE the union the matcher judged), re-runs
the exact recorded joint action through `generate_instructions` and the
instruction->event mapper, and prints each branch's attributed components beside
the observation's. **A class verdict in this ledger requires a replay, not a
label read.**

# Appendix C — mapper attribution, matcher component semantics, full classification

Branch `scott/mapper-attribution-fixes`, stacked on #890. Measured on the
**15-patch** engine (main + #888 terminal-options, #889 encore-duration).
**Deliberately NOT re-based onto #893's 17 patches**: holding the engine fixed
isolates the effect of the harness changes below. The post-#893 re-measurement
is sequenced after phaze-protect and PP-ordering land.

Build currency was verified rather than assumed (the engine lane's warning that
cargo's cache survives an rsync'd vendored-tree swap): Encore expires at turn 6
in the built wheel, so patch 14 is genuinely present.

## C.1 Mapper: a pending Wish stole the Leftovers tag

`residual_heal_cause` keyed on `wish.0 > 0`, which is the **pending-turn
counter**, not "a wish is landing". The engine emits `DecrementWish` *before*
the Leftovers heal on a non-resolving turn, so the counter is still positive
when the heal renders — and every ordinary Leftovers tick on a side carrying a
wish was tagged `[from] move: Wish`.

The instruction stream **does** distinguish them, by adjacency:

| situation | instructions |
| --- | --- |
| wish resolving | `Heal`, `DecrementWish`, [`Heal` (Leftovers)] |
| pending, not resolving | `DecrementWish`, `Heal` (Leftovers) |
| no wish | `Heal` (Leftovers) |

So the wish heal is exactly the one **immediately followed by `DecrementWish`
for the same side**. `render_residual_instruction` now takes that lookahead.
No instruction-level contract gap — the information was already there.
Regression test covers all three cases plus the other side's `DecrementWish`.

Verified on the reported repros: seeds **1310002 step 35**, **1310005 steps
10 and 49** all resolve.

## C.2 The move-heal class (B.4) was a matcher defect — verdict

**Neither hypothesis was right; it was my component classification.** The class
is **Rest**. Showdown renders it `|-heal|…|253/253 slp|[silent]` — a bare heal
with no `[from]`, which my extractor treated as a deterministic component
requiring exact equality.

But a heal that **caps at max HP restores `maxhp − hp`**, so its magnitude is
set by whatever damage landed earlier in the same turn — it inherits that hit's
roll. Seed 1310001 step 72: Showdown healed **251** from 2 HP, the engine healed
**247** from 6 HP. Same mechanic, different Surf roll. The engine was right.

Fix: any heal that tops the mon out is roll-scaled, whatever its tag (Rest,
Recover, and equally a Leftovers or Wish tick that happens to cap). The tag is
preserved so attribution is still compared; only the magnitude is relaxed, and
only in the capped direction (clipping can only reduce, so the test is an
inequality, not a window). **The class is gone from the residue.**

## C.3 Branch-consistency and capped residuals (#893 exoneration)

Two related corrections, both mine:

* **Check order.** Roll-scaled components are now compared **first**. A branch
  whose roll does not match is the wrong branch, and comparing its deterministic
  components against a different damage history reported the wrong mechanic.
* **Capped lethality.** A residual that *kills* is clipped to the HP that
  remained, so it can be arbitrarily smaller than the uncapped tick
  (seed 1310000 step 193: burn −20 observed where the uncapped tick is −26).
  Bucketed as roll-scaled and compared as an inequality.

The residual case where Showdown's roll left the mon alive-then-killed-by-burn
and the engine's roll did not is **not decidable** by per-component comparison —
the two sims took different stochastic outcomes. It is now its own named class,
`limit:roll_divergent_lethality` (4.2 % of divergences), rather than being
forced into a verdict.

## C.4 Phaze drag targets

Whirlwind/Roar drag a **random** target: the engine fans out over its world's
alive reserve while Showdown drew from the real hidden team, so entry-hazard
arithmetic can land on a different mon with a different max HP. The engine lane
verified these are determinization limits, not engine bugs. Now classified
`limit:world_sample_drag_target` instead of being charged to hazards.

## C.5 Every divergence is named

The old classifier read only step protocol and left ~28 % `unclassified`.
Classification is now driven by the **failing component**, which the strict
matcher always knows; protocol evidence is a fallback and its labels are
prefixed `evidence:` so no reader mistakes evidence for attribution. In
particular `faint_ply_residual_deferral` is gone — it was pointing the residue
table at D2 for boundaries where nothing faints in the move phase.

**60 games (seeds 1330000–1330059), 99.0 % measured, 0 harness errors,
240 divergences over 4,471 boundaries = 5.37 %, `unclassified` = 0:**

| Class | n | Share |
| --- | --- | --- |
| `component_magnitude:psn` (the toxic rounding bug) | 152 | **63.3 %** |
| `roll_scaled_component` | 27 | 11.2 % |
| `component_missing_in_engine:itemleftovers` | 11 | 4.6 % |
| `limit:roll_divergent_lethality` | 10 | 4.2 % |
| `component_mismatch:itemleftovers,psn|psn` | 5 | 2.1 % |
| 20 further named classes, each ≤ 4 | 35 | 14.6 % |

## C.6 Updated projection

`component_magnitude:psn` is the toxic multiply-then-floor bug reported in B.3,
**fixed by #893** — which is not in this build. Removing it and the psn-bearing
compound classes projects roughly **2 % of measured boundaries**, from 7.6 % in
Appendix B. That is a projection, not a measurement; the real number comes from
the post-#893 re-run once phaze-protect and PP-ordering land.

Acceptance remains held. Two classes are documented comparison limits
(`limit:*`) rather than divergences — the acceptance criterion should be stated
against them explicitly, either by driving them to zero with a better matcher or
by excluding them by name.

Seeds burned: 1320000–1320299, 1330000–1330059, plus single-game repro checks at
1310000, 1310002, 1310005.

## C.7 Replay-first found two matcher bugs the labels hid

Both surfaced on the first two rows replayed, and both were mine:

* **The primary move damage was invisible.** `damage_components` used the first
  HP line for a slot to establish its baseline, so that line's delta was
  dropped — hiding the step's main damage whenever the slot had no earlier line,
  i.e. most steps. Seeded from the pre-state now (the pre-state gate proves it
  equal on both sides). Seed 1340001 step 47.
* **Zero-delta components were recorded.** The engine emits `Heal SideTwo: 0`
  for a Rest that cannot heal a full-HP mon, where Showdown emits `|-fail|` and
  no HP line at all. The no-op made the component lists differ in length and
  surfaced as a spurious roll mismatch. Seed 1340000 step 110.

## C.8 Comparator tolerances, and why the census had to be regenerated

Review of #896 caught that `capped_lethal` and `*_to_full` shared one
**one-sided** test, `abs(obs) <= abs(eng) + 1`. They cap in **opposite**
directions:

| class | mechanism | correct test |
| --- | --- | --- |
| `capped_lethal` | residual clipped by remaining HP — can only shrink | `obs <= eng + 1` |
| `*_to_full` | heal restores `maxhp - hp_before`; a bigger preceding roll leaves less HP, so the heal is **larger** | two-sided, bounded by the preceding roll spread |

Shared, the test was inverted for heals: it rejected the motivating Rest case
(251 vs 247) while accepting a heal 24x too small (10 vs 247) — and unbounded
below. `*_to_full` is now bounded symmetrically by `0.18 x` the roll-scaled
damage on that slot this step (observed `d >= 0.85 * base`, so the spread
`0.15 * base <= 0.176 * d`), plus 1 HP of flooring slack.

`tests/test_transition_differential_matcher.py` pins both reviewer cases plus
the extraction rules — 11 tests. The comparator decides every acceptance
verdict, so it is tested directly rather than only through census aggregates: a
lenient comparator produces clean aggregates, which is precisely the failure
mode that looks like success.

**Every number below was regenerated through the fixed comparator**, and it moved
the wrong way on purpose: 1.65 % under the lenient version, **3.01 %** under the
correct one.

## C.9 Census on the 19-patch base (#893/#894/#895 merged)

60 games, seeds 1350000-1350059, strict matcher, support gating.
Build currency verified by probe (Encore expires turn 6), not assumed.

| Metric | Value |
| --- | --- |
| measured | 5,310 / 5,438 = **97.65 %** |
| diverged | 160 = **3.01 %** of measured |
| harness errors | **0** |
| `unclassified` | **0** |

| Class | n | Share |
| --- | --- | --- |
| `roll_scaled_component` | 86 | 53.8 % |
| `limit:roll_divergent_lethality` | 11 | 6.9 % |
| `component_extra_in_engine:itemleftovers` | 9 | 5.6 % |
| `component_mismatch:sandstorm|psn` | 9 | 5.6 % |
| `component_missing_in_engine:psn` | 9 | 5.6 % |
| 11 further named classes | 36 | 22.5 % |

**`roll_scaled_component` is NOT triaged.** Replaying one row (seed 1350001
step 49) shows p1 −95 vs engine −100 (inside the roll spread, fine) but p2
**−116 vs engine −170** — a 32 % damage disagreement, well outside any roll. That
is a substantive candidate, not roll noise, and the class name says nothing
about how many of the 86 are like it. Per the method rule above it needs
per-row replay triage before the acceptance run, and one sample does not license
a verdict on the class.

---

# Appendix D — `roll_scaled_component` triage, and the build-freshness gate

Branch `scott/roll-component-triage`, off #896. 19-patch engine.
Census seeds 1350000-1350059 (60 games, 5,310 measured boundaries).

## D.1 Triage of all 86 rows

`scripts/triage_roll_components.py` replays **every** row rather than sampling:
it re-runs each recorded candidate state + joint action, compares the observed
magnitudes against the engine's own legal roll set (`calculate_damage` gives the
100 % base for non-crit and crit), and buckets what is left by the multipliers a
gen3 damage calculation is built from.

**First result, and the one that matters: 0 of 86 rows had a fully-matching
branch.** Every divergence verdict was correct. What was wrong was the *label*.

| Bucket | n | Verdict | Representative |
| --- | --- | --- | --- |
| branch matched rolls, failed a DETERMINISTIC component | 30 (34.9 %) | **HARNESS — mislabelled**, fixed below | seed 1350005 step 87 |
| no branch matched rolls — real disagreement | 56 (65.1 %) | see D.3 | seed 1350001 step 49 |

## D.2 Two harness fixes, both from the triage

* **Miss ranking.** Roll-first rejection meant a branch that failed on its roll
  reported a roll reason even when another branch cleared the rolls and failed
  on a residual — and the first-listed miss drives classification. The miss is
  now the one from the branch that got **furthest**. This alone reclassified 36
  rows out of the class.
* **Partial-trap naming.** Showdown tags the tick `[from] move: Wrap`; the
  engine carries a generic `PARTIALLYTRAPPED` volatile and its mapper tags it
  `partiallytrapped`. The move identity is not recoverable engine-side and does
  not affect state (every gen3 Wrap-class move ticks maxhp/16), so both sides
  normalize to one canonical source. This was **18 rows, 11 % of all
  divergences, purely on naming.**

Census effect: **160 -> 142 divergences (3.01 % -> 2.67 %** of measured), and
`roll_scaled_component` 86 -> 50.

## D.3 Engine-lane hand-off

### CONFIRMED: gen3 Flail deals no damage

Direct probe, attacker at every HP fraction, defender 300/300, no items or
abilities in play:

| attacker HP | engine Flail damage |
| --- | --- |
| 300/300 | **0** |
| 150/300 | **0** |
| 60/300 | **0** |
| 20/300 | **0** |
| 5/300 | **0** |

gen3 Flail is an HP-proportional move (`basePowerCallback`, 20-200 BP as HP
falls — gen3 inherits the gen4 mod). The engine deals **zero at every fraction**,
so the move is inert.

* expected per Showdown: seed 1350004 step 3, p2 takes −20 from Flail; step 14,
  p2 takes −104
* got per engine: no damage component at all
* suspected mechanism: `basePowerCallback`-class move with no gen3
  implementation (same shape as the Transform gap — present in the move table,
  no behaviour)
* repro rows: seeds 1350004 steps 3, 13, 14

### NEEDS ITS OWN PROBE: Sleep Talk

`sleeptalk` is the single most common move in the structural bucket (11 of 29).
Sleep Talk calls a random move and the engine branches over the call, so a
component-count difference is expected *sometimes*; whether these are
enumeration gaps or real is not established, and I am not asserting it from
co-occurrence. Repro: seed 1350019 step 82, seed 1350030 step 65.

### Residual long tail

21 rows are magnitude disagreements with no clean multiplier signature (ratios
0.29-6.38, almost all singletons). No dominant mechanism; they need individual
replay and are the lowest-value slice.

Structural shapes: observed has a component the engine lacks in **21 of 29**
(`obs1_eng0` 11, `obs2_eng1` 10) — i.e. the engine is *missing damage*, which is
consistent with the Flail finding generalising to other unimplemented moves.

## D.4 Build-freshness gate (review finding 2)

The differential now **refuses to run** unless the installed engine was built
from the checked-out patch set. Two independent checks, because they catch
different halves:

* **Fingerprint (content, exact).** sha256 over the shared patch list plus every
  patch file it names — exactly the inputs both builders read.
  `scripts/setup_poke_engine.sh` and `scripts/vendor_poke_engine_src.sh` stamp
  it into the venv at build time; the harness compares stamp vs HEAD. Catches a
  stale **wheel** against a current tree.
* **Freshness (mtime).** Installed extension modules must be newer than every
  patch file and every vendored `.rs`. Catches the **crate** half, which maturin
  builds outside the stamping scripts, and catches a rebuild that silently
  no-op'd on a cache.

Failure is a hard stop with the exact rebuild sequence, because this class of
bug does not error — it produces a plausible number (4.43 % vs 1.11 % on
identical seeds). `--skip-build-check` exists only for `--merge-from`, where no
engine call is made. Preferred over a behavioural probe because probes only
cover mechanics someone thought to probe; the Encore probe remains as a
belt-and-braces manual check.

## D.5 State of the residue

| | |
| --- | --- |
| measured | 5,310 / 5,438 = 97.65 % |
| diverged | **142 = 2.67 %** of measured |
| harness errors | 0 |
| `unclassified` | 0 |
| wrong verdicts found by triage | **0 of 86** |

Remaining blockers before acceptance: PP-ordering, locked-move PP, and the Flail
fix. Sleep Talk needs a probe before it can be called either way.

---

# Appendix E — Sleep Talk verdict, and the acceptance-run plan

Branch `scott/sleeptalk-probe`, off #897. 19-patch engine.

## E.1 Ground truth, read from the vendored simulator

gen3 overrides Sleep Talk (`data/mods/gen3/moves.ts` `sleeptalk.onHit`). Its
candidate list keeps a slot when `moveid && !flags['nosleeptalk'] &&
!flags['charge']`, then samples **uniformly**. Two gen3-specific details:

* **`charge` (two-turn) moves are excluded** — **17** moves carry the flag in the
  gen3 Dex table, of which **8 are gen3-legal**: Solar Beam, Fly, Dig, **Dive**,
  **Bounce**, Sky Attack, Razor Wind, Skull Bash. (An earlier version of this line
  listed six and omitted Dive and Bounce, both of which are gen3-native.)
* a sampled slot at **0 PP** emits `|cant|<mon>|nopp|<move>` and the turn does
  nothing — gen3 does **not** resample.

poke-engine (`State::get_sleep_talk_choices`, `src/state.rs:1014`) keeps every
slot except Sleep Talk and NONE: no `nosleeptalk` test, no `charge` test, no PP
test.

## E.2 Probe results (`scripts/gen3_sleeptalk_probe.py`)

**Fan-out weights are CORRECT.** With no excluded move in the set the engine
splits exactly uniformly, matching gen3:

| case | gen3 callable | engine called | engine shares |
| --- | --- | --- | --- |
| `sleeptalk, bodyslam, curse, rest` | bodyslam, curse, rest (33.33 % each) | same 3 | 33.33 / 33.33 / 33.33 — **ok** |
| `sleeptalk, bodyslam, solarbeam, rest` | bodyslam, rest (50 % each) | + **solarbeam** | 33.33 each — **MISMATCH** |
| `sleeptalk, solarbeam, fly, rest` | rest (100 %) | + **solarbeam, fly** | 33.33 each — **MISMATCH** |

**Showdown differential confirms the exclusion**: a sleeping Snorlax with
Sleep Talk + Solar Beam + Body Slam + Rest, 60 seeds, called `bodyslam` 62x and
`rest` 58x — **Solar Beam never once**, exactly as `flags['charge']` prescribes.

**But the FLAG ARM is UNREACHABLE on the randbats distribution.** Across **1,682**
gen3 randbats variants, **70** carry Sleep Talk and **0** pair it with a
gen3-excluded move.

> **Scope, added 2026-08-03.** That 0 covers the `charge`/`nosleeptalk` **flag**
> arm and nothing else — it is a SET-COMPOSITION scan.
>
> The flag-arm zero also survives runtime moveset mutation, which this ledger
> records elsewhere (`self_moveset_mismatch` = 285): **0 of 1,682 variants carry
> Mimic or Sketch**, and the 7 Transform variants copy another pool variant, which
> is itself 0-pairing. So no mutation path can create a flag-arm pairing the static
> scan missed.
>
> Three further divergences are STATE conditions no composition scan can see.
> NONE of them is measured here, and only ONE — Encore — has its precondition
> present on this pool:
>
> * **0 PP.** gen3 emits `|cant|MON|nopp|MOVE` and does NOT resample; poke-engine
>   has no PP test. NOT MEASURED — no set scan can see a PP state, and this probe
>   ran no games, so "reachable" here is an argument from the mechanism, not an
>   observation.
> * **Encore.** gen3's `sleeptalk` inherits gen4's `onTryHit`
>   (`!volatiles.choicelock && !volatiles.encore`), so it FAILS outright while
>   Encored. **95 of the 1,682 variants carry Encore — carried by the OPPONENT: 0
>   variants carry both Encore and Sleep Talk**, which is what makes this a live
>   pairing rather than a self-inflicted one. Reachable, unmeasured.
> * **choicelock.** The other half of that same `onTryHit`, and by the section's
>   own set logic it is NOT reachable: 160 variants carry Choice Band (the pool's
>   only Choice item) and **0 of those carry Sleep Talk**. A mon can only be
>   choice-locked by holding the item itself, so this needs an item-transfer path
>   — `Trick`, 5 variants in the pool, **all 5 of which already hold the Choice
>   Band themselves** (2 Furret, 3 Kecleon), so it is a one-move transfer rather
>   than a Knock-Off-then-Trick chain — which nothing here measures. Bundling it
>   with Encore and answering "yes" on Encore's number would be the same
>   conflation this note exists to remove.
>
> Also note the exclusion sets are **gen3-resolved**, not base-table: a mod
> entry's `flags` replaces its parent's wholesale, and gen4/gen5 strip
> `nosleeptalk` from `fly`, `mimic`, `sketch`, `naturepower` and `struggle`. gen3
> has **35** `nosleeptalk` moves in its Dex table, not the base table's 40.
> Precisely: `data/mods/gen5/moves.ts` strips **fly** (gen5 = 39) and
> `data/mods/gen4/moves.ts` strips **mimic, sketch, naturepower, struggle**
> (gen4 = 35 = gen3).
> Reading the base table reports Mimic and Sketch as gen3-excluded when they are
> not (#1056).

## E.3 Verdict

| Question | Answer |
| --- | --- |
| Is the engine's Sleep Talk call fan-out wrong? | **Yes, in source** — it omits the `charge` and `nosleeptalk` exclusions and the 0-PP rule |
| Do the branch weights diverge? | **No** — uniform 1/n is correct whenever the candidate sets agree |
| Is the FLAG arm reachable in gen3 randbats? | **No** — 0 of 1,682 variants |
| Is the 0-PP arm reachable? | **Not measured** — a state condition, invisible to a set scan |
| Is the Encore arm reachable? | **Yes, unmeasured** — 95 of 1,682 variants carry Encore, opponent-side (0 variants carry both) |
| Is the choicelock arm reachable? | **No** by set logic — 0 of the 160 Choice Band variants carry Sleep Talk; needs Trick |
| Does it explain the co-occurring residue rows? | **No** |

So the **FLAG ARM** is a latent engine divergence: real, source-confirmed,
empirically demonstrated, and off-distribution for that arm. It belongs in the
engine lane as low priority (the flag arm would matter for `gen3customgame` or a
pool change), and that arm is **not** an acceptance blocker.

**This does not clear the state arms.** "Off-distribution" and "a pool change
would matter" are set-composition framings; the Encore arm needs neither, since
95 of 1,682 variants already carry Encore on the opposing side. Nothing in this
appendix measures it.

**Re-triage of the co-occurring rows.** 19 divergent rows in the 1350000-1350059
census involve Sleep Talk: 14 `roll_scaled_component`, 5 other. Their shape is
`observed=[('', -78)] engine=[]` — Showdown's call dealt damage and the engine's
branch has none. Since the fan-out and weights are correct and the **flag-arm**
exclusion bug is unreachable, these are not a Sleep Talk defect *on that arm*;
they are the called move's damage failing to match, which is the same "engine
is missing damage"
family as the confirmed variable-BP bug. They should be re-checked after the
variable-BP fixes land rather than tracked as a Sleep Talk item.

**Matcher accounting: no change needed.** Sleep Talk's branching is already
handled correctly — each call is its own branch and the matcher takes the union,
exactly as with hidden counters. The probe found no accounting defect.

## E.4 Review fixes folded in

* **Triage population guard (MEDIUM).** `triage_roll_components.py` quoted
  shares of a class computed over whatever rows it was handed. The first run
  covered 64 of 86 rows — capped by `--repros-per-game` — and produced
  confident percentages over an incomplete population; that got a narrative in
  the write-up instead of a guard. It now reads the class total from the
  census's `divergence_classes` and **refuses**, printing the regeneration
  flags; `--allow-partial` opts into a deliberate sample and relabels the shares
  as sample-scoped. Verified both directions.
* **Stamp write no longer swallowed (LOW).** Both builders called the
  fingerprint writer with `|| true`. A missing or stale stamp is exactly the
  state the gate exists to catch, so the `|| true` is removed — a failed stamp
  now fails the build.

## E.5 Note for the final re-measurement: Flail

The #897 hand-off cited the gen4 ladder. Review corrected it: **gen3 overrides
gen4** with a 48-scale and thresholds `<2, <5, <10, <17, <33`. The
variable-BP lane holds the correction; any re-measurement note referencing
Flail must cite **gen3's own override**, not gen4's. The confirmed finding
itself is unchanged — the engine deals 0 at every HP fraction.

## E.6 The acceptance run

**Blocked on:** PP-ordering, locked-move PP, and the variable-BP family. Once
those merge, the final re-measurement is 300 games strict on the full patch set
with the freshness gate live, producing the acceptance-readiness residue table.

**Bar.** Zero divergent transitions outside the named, adjudicated limit
classes. As of this appendix those are:

| Limit class | Why it is not a divergence |
| --- | --- |
| `limit:roll_divergent_lethality` | Showdown's roll left the mon alive to be killed by a residual and the engine's did not (or vice versa) — two different stochastic outcomes cannot be aligned component-wise |
| `limit:world_sample_drag_target` | Whirlwind/Roar drag a random target; the engine fans out over its own world's reserve, so hazard arithmetic can land on a different mon |

Any class added to that table needs a mechanism, a repro, and adjudication —
never a label alone.

**Run plan.** 8 shards x 1250 games from seed **2,000,000** (the reserved
pristine block; every measurement seed to date is below it):

```bash
for k in 0 1 2 3 4 5 6 7; do
  PYTHONPATH=src .venv/bin/python scripts/engine_transition_differential.py \
    --showdown-root "$SHOWDOWN" --matcher strict \
    --games 1250 --seed-start $((2000000 + k * 100000)) \
    --keep-repro 5000 --repros-per-game 300 \
    --checkpoint "acceptance_shard_${k}.jsonl" \
    --json "acceptance_shard_${k}.json" &
done
wait

PYTHONPATH=src .venv/bin/python scripts/engine_transition_differential.py \
  --merge-from acceptance_shard_*.jsonl --json acceptance_merged.json
```

Checkpointed so a supervisor kill costs one game, not a shard; `--resume` picks
each shard back up. The freshness gate runs at every shard's startup and refuses
a stale engine. `--repros-per-game 300` so the residue population is complete
and the triage guard passes. Expect ~1.1-1.5 h wall clock at 8-way concurrency.

**Read the merged report as:** `transitions_diverged` minus the adjudicated
`limit:*` classes must be **0**, with `measured_fraction_of_full_rounds` and
`unclassified` (which must be 0) reported alongside — a run that skips a third
of its boundaries, or cannot name a class, has not earned the number.

---

# Appendix F — re-measurement prep: four review items, and a corrected family assignment

Branch `scott/remeasurement-prep`, off merged main (19 patches).

## F.1 CORRECTION: the Sleep Talk residue rows are HARNESS, not engine

Appendix E re-triaged 19 Sleep Talk-involving residue rows into the
"engine is missing damage" family, alongside the confirmed variable-BP bug.
**Independently replaying two of them shows that was wrong.**

`seed 1350014 step 55` — Showdown's Sleep Talk called **Seismic Toss** for −78:

```
observed   p1: rolled = move −78
engine     p1: exact  = residual −78     LOSSY=['sleeptalk_called_unidentified']
```

`seed 1350019 step 99` — Showdown's Sleep Talk called **Psychic** for −103:

```
observed   p2: rolled = move −103
engine     p2: exact  = residual −97     LOSSY=['sleeptalk_called_unidentified']
```

**The engine computed the damage correctly in both cases** (−78 exactly; −97 vs
−103 is inside the roll window). What fails is the rendering: the mapper cannot
identify which move Sleep Talk called, so it marks the branch
`sleeptalk_called_unidentified` and attributes the called move's damage to a
generic `residual`. That lands it in the matcher's **exact** bucket instead of
the **roll-scaled** bucket, which can never match a bare `-damage` line — and
the matcher discards lossy branches anyway (671 lossy renders in the census).

This is a **known, documented mapper limitation**, not a new one:
`rust/pokezero-search/src/events.rs:1230` calls it out as "unrecoverable from
the delta (documented insufficiency)". So the honest verdict is:

| | |
| --- | --- |
| Verdict | **HARNESS / mapper** — called-move identity is not recoverable from the instruction delta |
| Not | engine missing damage (the previous, incorrect assignment) |
| Effect on residue | these rows cannot be matched while the branch renders lossy |
| Next step | either teach the mapper to recover the called move (an instruction-level contract change, needs the engine lane) or have the matcher treat a lossy Sleep Talk branch as a support-based case rather than discarding it |

**Process note.** This is the fourth mislabel replay has caught in six PRs, and
the second where *my own* summary of a class was wrong until a row was actually
re-run. The re-triage was load-bearing exactly as the review said. The standing
rule stands and is now cheap to obey: `scripts/replay_residue.py`.

## F.2 What "variant" means (the 1,682 / 70 figures)

Appendix E's **flag-arm** pairing numbers are in **expanded variants**; a re-checker
using the standard set denominator gets different totals and the *same zero*.
All three units, computed from
`Gen3RandbatSource.to_payload()["universes"]`:

| Unit | Total | Carrying Sleep Talk | Paired with a gen3-excluded move |
| --- | --- | --- | --- |
| species | 220 | 40 | **0** |
| source **sets** (`source_set_id`) | **393** | **44** | **0** |
| expanded **variants** (`variants[]`) | **1,682** | **70** | **0** |

A *set* is one generator entry for a species (Showdown's own unit); a *variant*
is one concrete 4-move realisation the belief layer expands that set into, so
variants > sets. Appendix E quoted the variant row; the reviewer's independent
44/393 is the set row. **Same FLAG-ARM zero at every denominator.**

This table says nothing about whether the state arms (0 PP, Encore, choicelock)
FIRE. Their PRECONDITIONS are countable here — that is where the 95 and the 160/0
come from, and the 160/0 is what licenses calling choicelock unreachable — but no
set-composition unit can express a PP count or a volatile, so no denominator
settles whether the arm is exercised. See the scope note in §E.2.

## F.3 Builder note: the stamp step runs last

Both builders now say so inline: the fingerprint write is the **last** step,
after the engine is built and installed, so if it is the only thing that failed
the engine itself is fine and the stamp can be re-run standalone:

```bash
python scripts/engine_build_fingerprint.py --write
```

Without this, removing the `|| true` turns a stamp-write failure into what looks
like a broken engine build.

## F.4 Reports record whether the build gate ran

Same label-the-output rule as `--allow-partial`. Every report now carries:

```json
"build_check": "gated",            "acceptance_eligible": true
"build_check": "NOT-GATED: skipped", "acceptance_eligible": false
```

The flag is stored **per checkpoint record**, not just per report, so a merge
across shards can tell that *any* of them ran ungated — one skipped shard
contaminates the merged report and prints a warning:

```
"build_check": "NOT-GATED: gated,skipped"
"acceptance_eligible": false
WARNING: merged report is NOT-GATED: ... NOT acceptance-eligible.
```

Verified in all three configurations. Pre-field checkpoints read `unknown`,
which is also not acceptance-eligible — absence of proof is not proof.

**The acceptance run must be read with `acceptance_eligible: true`.** That is now
a machine-checkable property of the artifact rather than a claim about how it
was produced.

## F.5 Sleep Talk unknown-callee: support-based matching, not a limit class

F.1 established the engine computes the called move's damage correctly and only
the mapper's LABEL is missing. The information needed to validate therefore
exists, so this is neither a `limit:` class nor an engine contract change — it is
fixed in the matcher with the same support-based principle as hidden counters.

**What changed.** A branch whose ONLY lossy marker is
`sleeptalk_called_unidentified` is no longer discarded. It is kept for the union,
and its unattributed `[from] residual` **damage** is reclassified as roll-scaled
`move_unknown_callee`. The realized outcome is then validated against the union
of the candidate branches' supports — each branch carries a concrete callee's
damage and roll set — on the same strict per-component basis as everything else.
Any other lossy marker still disqualifies a branch: this is scoped to the one
insufficiency whose shape is understood.

**Discard ordering (checked, as asked).** The discard happened *before* the union
could be taken. In the 1350000-1350059 census **every** lossy branch was
Sleep-Talk-only, so `strict:lossy_render` goes **671 -> 0** and 186 branches are
now retained for union purposes. Nothing else was being thrown away.

**Effect.**

| | before | after |
| --- | --- | --- |
| diverged | 142 (2.67 %) | **131 (2.47 %)** |
| lossy branches discarded | 671 | **0** |
| Sleep-Talk-involving divergent rows | 19 | **8** |
| `unclassified` | 0 | 0 |

The 11 rows that resolved are the ones F.1 predicted. The **8 that remain fail
for unrelated reasons** — a Leftovers tick the engine has and Showdown does not,
a capped-lethal roll disagreement, a missing psn component — i.e. they are
ordinary members of the other named classes that happened to contain a Sleep
Talk, not a residue of this gap. That is the outcome that distinguishes a real
fix from a blanket pass.

**Pins** (`tests/test_transition_differential_matcher.py`, 17 tests):

* `seed 1350014 step 55` — Showdown `-78` bare vs engine `-78` unattributed:
  **PASSES**;
* `seed 1350019 step 99` — Showdown `-103` vs engine `-97`, inside the roll
  window: **PASSES**;
* fabricated wrong damage (`-20` and `-200` against `-78`): **FAILS**;
* a missing component against a present one: **FAILS** — reclassification must
  not make absence match presence;
* the reclassification is off by default and only applies to `-damage`, so a
  genuine residual keeps its exact comparison.

Both replayed rows also verified end-to-end through the harness: previously
divergent, now clean.

**Acceptance bar unchanged.** No new limit class, no engine change. These
boundaries now pass or fail on the same strict per-component basis as every
other boundary.

## F.6 Two corrections to F.5 and the freshness gate

### The mtime half of the gate was unsatisfiable

maturin stamps extension modules with the reproducible-build epoch **315561600**
(1980-01-01), and archive-extracting installers preserve it. Such a file's mtime
carries **no** provenance — but the gate compared it against source mtimes and
reported a *freshly built* engine as STALE, permanently. The operator's only
escape was `--skip-build-check`, i.e. the gate trained the exact habit it exists
to prevent. (It never fired locally because `uv pip install --force-reinstall`
rewrites mtimes; it reproduces on any archive-extracting install path.)

Fixed with a three-step ladder, failure direction unchanged:

| artifact timestamp | treated as |
| --- | --- |
| >= 2000-01-01 | real install time — compared as before |
| pre-2000, but the wheel's `dist-info/RECORD` is dated | RECORD's mtime (real **install** time) |
| pre-2000 and RECORD undated | **unknown provenance** — never "old"; freshness rests on the content fingerprint, which is exact and timestamp-independent |

Verified in all three states. A pre-2000 stamp is now evidence of *nothing*
rather than evidence of staleness, and an honest rebuild can satisfy the gate.

**What state 3 does not cover.** The content fingerprint spans the **patch
set**, not the built artifact, so an artifact built from the right patches but
at the wrong `rust/pokezero-search` source commit is invisible in state 3 —
states 1-2 catch that by mtime, and it is unreachable for the acceptance run
because each shard rebuilds from a clean vendor.


### F.5's containment was described more narrowly than it is implemented

F.5 said the reclassification applies to "the called move's damage". **It does
not.** The implemented predicate is: any `-damage` line inside a
Sleep-Talk-flagged branch whose source fell through to the mapper's generic
`residual` tag. Nothing in the rendered stream identifies which line the callee
produced — that is the very reason the branch is flagged.

The two descriptions coincide in practice because every other gen3 residual the
mapper emits is *named* (psn / brn / Sandstorm / Hail / Leech Seed /
partialtrap); the fall-through is reachable only for a residual with no cause
branch, and both candidates are absent from the pool — Nightmare in **0 of 393**
sets, and all 5 Curse users are non-Ghost, so Ghost-Curse never occurs. So the
broader predicate is **latent, not exercised**.

That is a reason to state it accurately, not a reason to leave it unstated —
this is the same recurring lesson as A.2's D8 scope and B.1's roll-window
wording: **describe the implemented predicate, not the intended one.** The
containment boundary is now pinned rather than merely observed:
`test_named_residual_is_NOT_reclassified` asserts a named `psn`/`Sandstorm`
residual inside a flagged branch keeps its exact comparison, and
`test_unattributed_HEAL_is_not_reclassified` asserts the scope is `-damage` only.
19 tests.

### Interim reading on the 20-patch build (variable-BP merged)

Not the final re-measurement — PP-ordering and locked-move PP are still
outstanding — but the gate now passes at 20 patches and Flail is fixed:

| attacker HP | Flail damage, pre-#20 | post-#20 |
| --- | --- | --- |
| 300/300 | 0 | 52 |
| 150/300 | 0 | 99 |
| 60/300 | 0 | 241 |
| 20/300 | 0 | 300 |

Census, seeds 1350000-1350059, `acceptance_eligible: true`:
**128 / 5,310 = 2.41 %** (from 2.47 % at 19 patches), 97.65 % measured, 0 harness
errors, `unclassified` 0.

---

# Appendix G — pre-flight for the final re-measurement

Branch `scott/final-remeasurement`. **Not the final pass** — #904 (locked-move
PP) is still open, so the base is 21 patches, not 22.

## G.1 Base and census

Fresh build per protocol (vendor → touch → wheel → touch → crate → stamp);
gate reports current at **21 patches** (`2567011c59245daf`), report
`acceptance_eligible: true`.

| base | diverged / measured | rate |
| --- | --- | --- |
| 19 patches | 131 / 5,310 | 2.47 % |
| 20 patches (variable-BP) | 128 / 5,310 | 2.41 % |
| **21 patches (PP-ordering)** | **128 / 5,310** | **2.41 %** |

PP-ordering moved nothing on this seed set, as expected: charging PP correctly
does not change any HP component. 97.65 % measured, 0 harness errors,
`unclassified` 0.

## G.2 Replay-verified verdict on the dominant class

`roll_scaled_component` (37 rows, 28.9 %) triaged; its largest bucket is
`structural_component_count` (16 rows, 43 %). Per the method rule the label was
replayed, not read — and it was wrong again.

`seed 1350004 step 66`:

```
|move|p1a: Mew|Soft-Boiled|p1a: Mew
|-heal|p1a: Mew|263/263
|move|p2a: Exeggutor|Solar Beam|p1a: Mew|[from] lockedmove   <- RELEASE turn
|-damage|p1a: Mew|189/263
|-heal|p1a: Mew|205/263|[from] item: Leftovers

observed  p1: rolled = heal_to_full +27, move -74   p2: exact = itemleftovers +18
engine    p1: rolled = heal_to_full +27             p2: exact = itemleftovers +18
raw instructions: Heal SideOne: 27        <- and nothing else
```

Showdown is on the **second** turn of a two-turn move (`[from] lockedmove`); the
engine executed no Solar Beam at all.

**Root cause — a world-construction gap, and a silent one.** `lockedmove` is not
in `_SUPPORTED_VOLATILES` (`engine_world.py:91`) and no allowlist branch adds it,
so a payload reporting it would fail closed. This boundary was *measured*
(`gating=exact`), which means the public materialization **never carried the
charge state at all**. The world is therefore built with the charging mon free to
act, and submitting the move starts a **fresh charge** instead of releasing —
the engine loses a whole turn of damage and neither errors nor falls back.

| | |
| --- | --- |
| Verdict | **WORLD CONSTRUCTION** (`engine_world` / materialization payload), not the matcher |
| Class | silent wrongness — no error, no fallback, a plausible wrong state |
| Repro | seed 1350004 step 66 (`softboiled` vs `solarbeam`) |
| Spec | the two-turn charge state (`|-prepare|`, Showdown's `lockedmove`) is public and must either be expressed on the engine world or fail the boundary closed; today it does neither |
| Related | the same two-turn family the Sleep Talk probe found unhandled in the callee exclusion set (E.2) |

This is a candidate explanation for a meaningful share of the
`structural_component_count` bucket, whose signature is "the engine is missing a
damage component" — but that share is **not** asserted here on one replay. It
needs the same per-row treatment before the final table.

## G.3 What remains before the acceptance run

1. #904 merges → rebuild to 22 patches, re-run 300 games strict.
2. Finish replay-verified verdicts for the remaining named classes.
3. Adjudicate: engine lane, harness, or a named `limit:` class with a mechanism.
4. If clean against the bar — zero divergent transitions outside the adjudicated
   `limit:` classes — proceed to 8x1250 from seed 2,000,000 per F.6/the plan.

---

# Appendix H — replay-verified adjudication of the named residue classes

Branch `scott/residue-adjudication`, 22-patch main. Verdicts below come from
replaying rows, per the method rule — not from reading class names. Census is
the 21-patch run (seeds 1350000-1350059, 128 divergences / 5,310 measured);
the 22-patch re-run is held until the charge-state fix lands so its rows do not
contaminate the structural classes.

## H.1 CONFIRMED: one mapper bug explains five classes (30 rows, 23.4 %)

| Class | n | Replayed repro |
| --- | --- | --- |
| `component_mismatch:sandstorm\|psn` | 10 | seed 1350002 step 14 |
| `component_mismatch:heal,itemleftovers\|leechseed` | 9 | seed 1350006 step 46 |
| `component_mismatch:partialtrap\|sandstorm` | 5 | seed 1350055 step 53 |
| `component_mismatch:leechseed\|psn` | 3 | seed 1350024 step 137 |
| `component_mismatch:sandstorm\|brn` | 3 | — same shape |

**In every replayed case the engine's HP arithmetic is IDENTICAL to Showdown's.
Only the labels differ.** seed 1350002 step 14, raw instructions:

```
Damage SideOne: 18     <- sandstorm    rendered "[from] psn"
Damage SideTwo: 23     <- sandstorm    rendered "[from] Sandstorm"
Heal   SideOne: 18     <- Leftovers    rendered "[from] item: Leftovers"
Heal   SideTwo: 23     <- Leftovers
Damage SideOne: 37     <- poison       rendered "[from] psn"
observed  p1: itemleftovers +18, psn -37, sandstorm -18
engine    p1: itemleftovers +18, psn -37, psn      -18
```

**Root cause.** `residual_damage_cause` / `residual_heal_cause`
(`rust/pokezero-search/src/events.rs`) attribute by inspecting the side's STATE
in a fixed priority order — status, then Leech Seed, then weather, then partial
trap — and return the first match for **every** residual instruction on that
side. A mon with two simultaneous residual sources therefore has *all* its ticks
labelled with the highest-priority one. Each observed pair is exactly that:

* poisoned mon in sand -> the sand tick is tagged `psn` (status checked first);
* seeded + poisoned -> the leech tick is tagged `psn`;
* trapped mon in sand -> the trap tick is tagged `Sandstorm` (weather before trap);
* Leftovers holder whose opponent is seeded -> its Leftovers heal is tagged
  `Leech Seed` (opponent-leechseed checked before Leftovers).

| | |
| --- | --- |
| Verdict | **HARNESS / mapper** — same family as the Wish/Leftovers mis-tag (C.1) |
| Lane | mine |
| Acceptance | **blocking** — 23.4 % of the residue |
| State fidelity | unaffected; the engine's transitions are correct |

**Fix spec.** Attribute residual instructions **positionally**, against the
engine's own end-of-turn order, instead of guessing from state. That order is
explicit in `gen3/generate_instructions.rs::add_end_of_turn_instructions` and
runs per side in `[first_move_side, other]`:

1. weather decrement / dissipation
2. weather chip (Hail, then Sand)
3. residual order 5 — Leftovers / Shed Skin
4. Leech Seed sap
5. status damage (burn / poison / toxic)
6. later-order effects (threshold berries, Rain Dish)

Implementation shape: build a per-side ordered queue of expected residual events
from the pre-residual state, then consume it as `Damage`/`Heal` instructions
arrive. Amount alone is insufficient to disambiguate — sand chip and partial trap
are both maxhp/16, and seed 1350055 step 53 shows exactly that collision — so
position and expected amount must be used together.

**Deliberately not implemented in this pass.** A half-right positional
attributor emits *confident wrong* labels, which is worse than today's honest
mislabel: the current bug is loud (it diverges) whereas a subtly wrong attributor
would be silent. It wants its own change with its own pins, and the window while
the charge-state fix is in flight is the right place for it.

## H.2 NOT the label bug: component present in one sim only

| Class | n | Replayed | Finding |
| --- | --- | --- | --- |
| `component_missing_in_engine:itemleftovers` | 11 | — | |
| `component_missing_in_engine:psn` | 9 | seed 1350002 step 31 | observed p1 has `itemleftovers +16, psn -16, sandstorm -16`; the engine branch has only `itemleftovers +16, sandstorm -16` — a genuinely absent component, not a relabel |
| `component_extra_in_engine:itemleftovers` | 8 | seed 1350007 step 85 | engine heals p2 `+19`; Showdown gives p2 no tick at all |
| `component_magnitude:itemleftovers` | 5 | — | |
| `component_missing_in_engine:heal` / `:leechseed` | 5 / 5 | — | |

These have a different signature from H.1 — component **counts** differ rather
than labels — so the mapper bug does not explain them and they are **not
adjudicated**. Two were replayed to establish that much; the rest need per-row
replay. Some are plausibly the charge-state gap (G.2), whose signature is also
"the engine is missing a component", but that is a hypothesis and this ledger
does not record hypotheses as verdicts.

## H.3 Adjudication status

| Family | n | Verdict | Lane |
| --- | --- | --- | --- |
| residual mis-attribution (H.1) | 30 | **CONFIRMED harness/mapper** | mine, fix specced |
| structural / charge state (G.2) | ~16 | **CONFIRMED world construction** | engine lane, in progress |
| `limit:roll_divergent_lethality` | 10 | adjudicated limit | — |
| `limit:world_sample_drag_target` | 2 | adjudicated limit | — |
| present-in-one-sim-only (H.2) | ~38 | **UNADJUDICATED** | needs per-row replay |
| long-tail ratios | ~21 | **UNADJUDICATED** | singletons, lowest value |

**The acceptance table cannot yet call itself fully adjudicated.** Two named
families are confirmed and lane-assigned; roughly 59 rows across H.2 and the
ratio tail still need replay-verified verdicts. That work, plus the mapper fix
and the charge-state fix, is what stands between here and a meaningful
acceptance run.

---

# Appendix I — positional residual attribution

Branch `scott/positional-residual-attribution`, 22-patch base, gated,
`acceptance_eligible: true`.

## I.1 The fix

`residual_damage_cause` / `residual_heal_cause` guessed a source by testing the
side's STATE in a fixed priority order and returning the first match for **every**
residual on that side (H.1). Replaced with `ResidualPlan`: a per-side ordered
list of expected residual events, built from the pre-residual state in the
engine's own emission order and consumed as instructions arrive.

Order taken from `gen3/generate_instructions.rs::add_end_of_turn_instructions`,
which iterates `[first_move_side, other]` within each phase:

```
weather chip -> future sight -> wish -> order-5 items (Leftovers)
  -> Leech Seed sap -> status damage -> order-10 items -> volatiles (partial trap)
```

**Never amount-based, and that is not a stylistic preference.** The sandstorm
chip and the partial-trap tick are *both* `maxhp/16`: on a mon carrying both they
are numerically identical, so no amount-based attributor can separate them ever.
`sand_and_trap_collide_on_amount_and_only_order_separates_them` pins that exact
collision permanently.

The plan predicts phases with **presence predicates only** — never damage
formulas — because re-deriving the engine's arithmetic is the fragile thing.
And it is used for a side **only when its predicted counts exactly match the HP
instructions that side actually emits**. On any mismatch that side falls back to
the generic `residual` tag, which is *loud* (it diverges) rather than confidently
wrong; `count_mismatch_falls_back_to_the_generic_tag` pins that.

## I.2 A second, separate mapper bug found on the way

With attribution fixed, `component_mismatch:heal,itemleftovers|leechseed`
collapsed to `heal|leechseed` — still 9 rows. A live trace settles it:

```
|-damage|p2a: Golduck|79/262|[from] Leech Seed|[of] p1a: Bellossom
|-heal|p1a: Bellossom|259/293|[silent]          <- the seeder's drain
```

Showdown tags the **victim's damage** with Leech Seed and emits the **seeder's
heal bare**. The mapper was tagging the drain `[from] Leech Seed`. Fixed: an
empty cause now renders `|-heal|...|[silent]`, matching the real protocol.

## I.3 Effect

| build | diverged / 5,310 | rate |
| --- | --- | --- |
| 21 patches, old attributor | 128 | 2.41 % |
| 22 patches, positional attribution | 103 | 1.94 % |
| 22 patches, + silent sap heal | **84** | **1.58 %** |

**44 rows cleared, 34 % of the residue.** Five classes are gone entirely
(`sandstorm|psn`, `partialtrap|sandstorm`, `leechseed|psn`, `sandstorm|brn`,
`heal|leechseed`). 97.65 % measured, 0 harness errors, `unclassified` 0.

Pins: 5 new unit tests (poisoned-in-sand, trapped-in-sand, the amount collision,
Leftovers-vs-seeded, the triple-source side, plus the desync fallback);
17 crate suites green, including the 22nd patch's own locked-move PP tests.

## I.4 Remaining residue (84 rows)

| Class | n | Status |
| --- | --- | --- |
| `roll_scaled_component` | 37 | structural bucket — charge-state gap (G.2), engine lane |
| `component_missing_in_engine:itemleftovers` | 11 | **unadjudicated** |
| `limit:roll_divergent_lethality` | 10 | adjudicated limit |
| `component_extra_in_engine:itemleftovers` | 8 | **unadjudicated** |
| `component_magnitude:itemleftovers` | 5 | **unadjudicated** |
| `component_missing_in_engine:psn` | 5 | **unadjudicated** |
| `limit:world_sample_drag_target` | 2 | adjudicated limit |
| 5 singleton classes | 6 | **unadjudicated** |

The Leftovers-shaped families now dominate the unadjudicated set. They were
*not* cleared by this fix, which means they are not attribution errors — a
component is genuinely present in one sim and not the other. Per the sequencing
note these get their verdicts after the charge-state fix merges, since that gap
has the same "engine is missing a component" signature and would otherwise force
a re-triage.

## I.5 Two hardening items before the acceptance run

### The fingerprint now covers the crate's own sources

It hashed the patch list and patch files only, so a `.so` built before an
`events.rs` edit passed the content check whenever timestamps were in the
provenance-unknown state — the mapper changes, the fingerprint does not. **I hit
this seam myself** while landing the positional attributor: the crate compiled
against a 21-patch vendored tree on a 22-patch checkout, and it surfaced as test
failures rather than as a gate failure.

`compute_fingerprint` now also hashes every `.rs` under
`rust/pokezero-search/src`, keyed by repo-relative path. Verified both ways: a
genuine edit to `events.rs` without a rebuild now **fails**; the same file
touched but unchanged **passes**.

That second case forced a related fix. Touching a source bumps its mtime above
the artifact's, so the mtime half reported STALE while the content fingerprint
matched exactly — the same unsatisfiable false positive the reproducible-epoch
ladder removed. **The exact check now outranks the heuristic**: when the
fingerprint matches, an mtime complaint is printed as a note, not an error. It is
still an error whenever the fingerprint is missing or mismatched, which is
exactly when the heuristic is the only signal left.

The stamp must be written at the **end of a full rebuild** (wheel *and* crate) —
a stamp written after rebuilding only one would claim currency the other has not
earned. The documented rebuild sequence already ends that way.

### The eight-section emission order is pinned as a whole

The per-source tests each pin one pair. **None of them would catch an engine that
REORDERS the end-of-turn sections**: the counts would still reconcile, the plan
would still be "usable", and every tick would be silently mislabelled — the one
failure mode the existing pins miss.

`end_of_turn_section_order_is_pinned_against_the_engine` drives the real
`generate_instructions_from_move_pair` with five sections firing on one side at
once and asserts the rendered sequence:

```
Sandstorm -> item: Leftovers -> Leech Seed -> psn -> partiallytrapped
```

A reorder in `add_end_of_turn_instructions` now fails there instead of silently
relabelling the census. The trace also contains the amount collision live — the
sandstorm chip and the partial-trap tick are both 20 in it.

Census after both items: **84 / 5,310 = 1.58 %**, byte-identical to before —
the hardening is measurement-neutral, as it should be. 17 crate suites green.

---

# Appendix J — FINAL RE-MEASUREMENT and acceptance verdict

Full fix set: 22 vendored patches + charge-state world construction + the
hardened harness. Clean vendor, both wheels rebuilt, stamped; 18 crate suites
green; gate current (`acc9a7852528a1c9`, 22 patches + 8 crate sources).

## J.1 The numbers

300 games, seeds **1500000-1500299**, strict matcher, `--repros-per-game 300`.

| | |
| --- | --- |
| games | 300 |
| full-round boundaries | 23,860 |
| **measured** | **23,334 — 97.80 %** |
| matched | 23,023 |
| **diverged** | **311 — 1.33 % of measured** |
| harness errors | **0** |
| `unclassified` | **0** |
| `build_check` | `gated` |
| `acceptance_eligible` | `true` |
| throughput | 1,419 games/h |
| **divergences in adjudicated `limit:` classes** | **47** |
| **divergences OUTSIDE limit classes** | **264** |

## J.2 ACCEPTANCE VERDICT: NOT MET — the acceptance run was NOT started

The bar is *zero divergent transitions outside the named, adjudicated limit
classes*. There are **264**. The 8x1250 run is therefore **not** launched: it
would spend ~1.5 h producing a number that cannot pass, and the standing
authorization is conditioned on a clean table.

| Class | n | Verdict |
| --- | --- | --- |
| `roll_scaled_component` | 145 | **open** — 89 structural, 56 magnitude (J.3, J.4) |
| `limit:roll_divergent_lethality` | 42 | adjudicated limit |
| `component_missing_in_engine:itemleftovers` | 27 | **open** |
| `component_extra_in_engine:itemleftovers` | 25 | **open** |
| `component_missing_in_engine:psn` | 24 | **open** |
| `component_extra_in_engine:psn` | 9 | **open** |
| `component_extra_in_engine:itemleftovers,psn` | 6 | **open** |
| `limit:world_sample_drag_target` | 5 | adjudicated limit |
| 17 smaller classes | 28 | **open** |

## J.3 NEW, CONFIRMED: Substitute is not expressed in the constructed world

`seed 1500002 step 75` — Seismic Toss into a Substitute:

```
Showdown:  |move|p1a: Hypno|Seismic Toss|p2a: Lugia
           |-end|p2a: Lugia|Substitute          <- the SUB breaks; Lugia untouched
engine:    Damage SideTwo: 88                   <- 88 straight into Lugia
```

The engine has no Substitute, so fixed damage lands on the Pokemon instead of
breaking the sub. Downstream, p2's HP is then wrong for the rest of the turn,
which perturbs its residuals — so this one gap manufactures `missing`/`extra`
Leftovers and psn components too.

**Incidence: 49 of 311 divergent rows (16 %) have a Substitute in the step**,
spread across six classes. Same silent-wrongness shape as the charge-state gap:
no error, no fallback, a plausible wrong state.

| | |
| --- | --- |
| Verdict | **WORLD CONSTRUCTION** — engine lane |
| Repro | seed 1500002 step 75 (`seismictoss` vs `toxic`) |
| Note | the harness already passes `approximate_substitute_health=True`, so the allowlist permits `substitute`; the state is not reaching the world regardless, which is where the investigation should start |

## J.4 NEW: the charge-state fix works, but the release damage is wrong

The fix lands — the engine now executes the release turn instead of starting a
fresh charge. On the row originally filed (G.2, `seed 1350004 step 66`) the
missing component is gone. But the damage disagrees, and the engine's own
branches are internally inconsistent:

| | damage |
| --- | --- |
| Showdown | **−74** |
| engine, 93.75 % branch | **−41** |
| engine, 6.25 % (crit) branch | **−163** |

Gen 3 crits are 2x. 163/41 = **3.98**, so the two engine branches disagree with
*each other* by a factor of two. The inference drawn here — that the non-crit
release was halving base power while the crit was not — **was wrong**, and is
corrected below; `calculate_damage` derives both branches from the same `choice`,
so a base-power error moves both together and cannot separate them.

**The row's actual 4x mechanism (dumped from `charge60.json`, seed 1350004
step 66).** Side one, the defender:

    active_index          = 3            (MEW)
    side_conditions       = 0;0;0;...;0  (all nineteen zero — no screen)
    special_attack_boost  = 3
    special_defense_boost = 2            <- the mechanism
    weather               = NONE

Solar Beam is Special, so the defender's **+2 SpD** halves the non-crit branch
while the crit ignores it (gen3 crits ignore the defender's positive stages) —
2x2 = 4x, from a rule the engine implements **correctly**. Calm Mind is on 31
gen3 randbats species. An earlier draft attributed the 4x to Light Screen, which
also yields 4x but is **unreachable here**: Light Screen and Reflect are 0/220
species in the gen3 randbats pool, and this row came from the randbats
re-measurement. All four sign cases are now pinned in
`rust/pokezero-search/tests/gen3_crit_boost_rules.rs`.

The single real defect was the base power itself — Solar Beam halved in clear
weather (deviation 10) — and Showdown's −74 sits within roll of the unhalved
~82.

Class-shape evidence on the fixed build (same 60-game seed set): the structural
bucket fell 29 -> 15 while the magnitude bucket rose to 22 — rows converted from
"component missing" to "component wrong size", which is exactly what a working
release with wrong damage looks like.

| | |
| --- | --- |
| Verdict | **ENGINE** — damage on the two-turn release; engine lane |
| Repro | seed 1350004 step 66 (`softboiled` vs `solarbeam`) |

## J.5 What is left before acceptance

1. **Substitute world construction** (J.3) — 16 % of rows, engine lane.
2. **Two-turn release damage** (J.4) — engine lane.
3. Re-triage the Leftovers/psn families once 1 lands; a large share are
   downstream of the Substitute HP error rather than independent bugs.
4. The 56 magnitude rows in `roll_scaled_component` still need per-row replay.

The measurement apparatus itself is done: 97.80 % coverage, 0 harness errors,
0 unclassified, gated and acceptance-eligible. **Every remaining divergence is a
simulator-fidelity question, not a measurement question** — which is the state
this program was trying to reach, just not yet at zero.

## J.6 Artifacts

| Artifact | Path |
| --- | --- |
| final 300-game report | `<scratch>/reports/final300.json` |
| per-game checkpoint (resumable) | `<scratch>/reports/final300.jsonl` |
| run log | `<scratch>/reports/final300.log` |
| triage of the dominant class | `<scratch>/reports/tri300.json` |
| charge-fix A/B (60 games, seeds 1350000-1350059) | `<scratch>/reports/charge60.json` |

`<scratch>` =
`/private/tmp/claude-501/-<home>-workspace-agents-pokezero-agent/47b7c392-a7b8-43cf-b071-8a500f9bc9bf/scratchpad`

## J.7 Acceptance trail: the run that was NOT started

A negative decision is part of the acceptance trail. Without this entry a later
reader finds an authorization, no acceptance artifact, and no explanation — and
cannot tell whether the run was skipped, lost, or failed.

| | |
| --- | --- |
| Decision | **The 8x1250 acceptance run was NOT started.** |
| When | immediately after the §J.1 re-measurement, before any shard was launched |
| Authorization | standing, and explicitly conditioned on a clean table |
| Condition | zero divergent transitions outside the named, adjudicated `limit:` classes |
| Observed | **264** outside those classes (311 total, 47 adjudicated) |
| Therefore | the condition was not met, so the authorization did not apply |

**Why not run it anyway.** A ~1.5 h run would have produced a real, correct,
citable artifact showing a number that cannot pass. The failure mode this whole
ledger exists to prevent is a plausible number acquiring the wrong label — and a
"baseline acceptance run" is precisely that shape: an artifact whose filename
says acceptance and whose content is not a pass. Confirmed on review: **the
acceptance artifact must be a pass, full stop; no baseline run.**

**Consequence worth checking.** The reserved block at seed **2,000,000+ remains
entirely unconsumed.** No acceptance shard has ever run, so the next attempt
still has a pristine block, exactly as §5.2 reserved it. Every ACCEPTANCE seed burned to
date remains below 2,000,000.

**Amended 2026-08-04 (C116 M7): the sentinel needed a namespace.** As originally
written this clause said "every seed burned to date remains below 2,000,000" and
that any report citing seeds at or above 2,000,000 implied an unrecorded
acceptance attempt. That became a false positive on every engine-fidelity report,
because the fidelity differential window sits at **19,000,000+** and has been
swept dozens of times. A provenance sentinel that fires on routine work stops
being trusted, which is worse than not having one. The seed space is therefore
partitioned by purpose, and the invariant is restated against the acceptance
namespace only:

| namespace | range | purpose | status |
| --- | --- | --- | --- |
| acceptance, C14 | `2,000,000`–`2,701,249` | the 8x1250 acceptance run | **CONSUMED** — Appendix Z12, an honest FAIL at 3,821 divergent |
| acceptance, C15 registered | `2,800,000`–`3,501,249` | | consumed |
| acceptance, C26 registered | `16,000,000`–`16,701,249`, plus probe band `17,000,000`–`17,000,999` | | burned, never swept |
| acceptance, **C32 — the ACTIVE registration** | `18,000,000`–`18,701,249` | the next acceptance attempt | **reserved, unconsumed** |
| fidelity differential, dev window | `19,000,000`–`19,000,199` | the 200-game window the 208 → 7 era iterated against | consumed continuously |
| fidelity differential, earlier 800-game sweep | `19,500,000`–`19,500,799` | `reports/c73_eight_hundred_game_sweep.json`, 800 games from `run.seed_start` `19,500,000` | consumed |
| fidelity differential, **validation holdout** | `19,100,000`–`19,100,199` | out-of-window check | reserved (C116 Phase 1) — reserved *for* that use, and swept on every fix branch since |
| fidelity differential, **final holdout** — as REGISTERED | `19,200,000`–`19,200,199` | terminal fidelity claim, registered as a single measurement | **partly swept, and not on the registered span.** Decomposed in the next two rows |
| fidelity differential, final holdout — the span C141 **actually swept** | `19,200,060`–`19,200,259` | `reports/artifacts/c141_final_holdout_sweep.json` | **CONSUMED** — 200 games, 16,274 boundaries measured, 2 divergent. Overruns the registered end by **60 seeds** |
| fidelity differential, final holdout — the registered head C141 **did not reach** | `19,200,000`–`19,200,059` | the first 60 seeds of the registered block | **unswept, and contaminated.** No committed artifact covers these 60 seeds; 60 games were run over them before the guard existed |
| fidelity differential, final holdout — C151 **RATIFIED replacement window** | `19,300,000`–`19,300,199` | the terminal one-shot measurement, re-registered on virgin seeds because the `19,200,000` block is burned | **RATIFIED (owner scott, 2026-08-08) and NOT YET SWEPT.** One sweep, ever, and only once the precondition holds: ledger terminal and engine fingerprint declared frozen for the claim. `reports/c151_final_holdout_rereg_prediction.md` is frozen as to window, protocol and preconditions; the trigger has not fired |

**C151 — the owner ratified a replacement window, burned the old block, and deferred the
sweep.** Ratified **2026-08-08** by **scott**, in these words:

> *"Ratified: final holdout re-registered as 19,300,000–19,300,199, one sweep ever, to run
> only after the ledger is terminal and the engine fingerprint is frozen; old block burned
> in the guard; c141 demoted to dev evidence; nonzero-result protocol pre-registered per
> plan."*

C151 adds **one** row to this table, the one above. The three rows that decompose the
`19,200,000` block are #1189's and are untouched — they were deliberately silent on whether
a future sweep is permitted, and the disposition is recorded here rather than by rewriting
them.

**The `19,200,000`–`19,200,259` block is BURNED, and no part of it is salvageable.** Three
independent reasons, and the guard now refuses the whole span:

* `19,200,000`–`19,200,059` is **contaminated** — executed by the pre-guard convenience
  loop, disclosed at the time, JSON deleted unread.
* `19,200,060`–`19,200,199` was swept by C141 on a window the executing agent
  **chose it itself** rather than deferring to the owner. Its own pre-registration says *"chosen by me
  rather than deferred"*, and the disclosure below explicitly left the disposition to the
  owner and recorded *"I have **not** chosen"*. An adversarial audit ruled the self-blessing
  defeats the terminal claim.
* `19,200,200`–`19,200,259` was consumed by that same run, which **overran** its
  registration by 60 seeds.

Enforced, not merely stated: `_reject_burned_final_holdout` in
`scripts/engine_transition_differential.py` refuses that span **unconditionally**, at
execution and at `--merge-from` aggregation.
`--final-holdout-i-mean-it` **does not open it** — a flag that can reopen a burned block is
a flag that will reopen it. Pinned by
`TheBurnedBlockAndTheOwnerRatificationTests` in `tests/test_final_holdout_guard.py`,
including end-to-end through `main()` with the opt-in passed.

**The vanished disclosure is RECOVERED, and it is now committed.** The ledger previously
recorded that `reports/rust-fidelity/final_holdout_contamination_disclosure.md` — the sole
justification for C141 narrowing its window — existed in **no tree in this repository's
history**, and correctly guessed the cause: a dropped `agents/` prefix, since `5a44c04e`'s
commit message gives the path as `agents/reports/rust-fidelity/…` and marks it *"outside
this repo"*. It was found there, on disk, unmodified since it was written, and is committed
verbatim at the path the frozen C141 citation names, so that citation now resolves.
Provenance, so a reader can check rather than trust — **ordered strongest leg first**, because
the durable evidence is in git and the filesystem metadata only corroborates it:

| rank | item | value |
|---|---|---|
| 1 — in-repo, immutable | corroborating commit | `5a44c04e` (#1122), `2026-08-05 22:20:36`. Its message **names this exact path** and marks it *"outside this repo"*, and **reproduces the disclosure's shell loop verbatim** — `for start in 19100000 19000000 19200000`. Independent of the recovered file, and it establishes both that the document existed and what it said |
| 2 — in-repo, immutable | the owed work | the disclosure files the tool-side guard as *"owed work"*; `5a44c04e` **is** that work, 2h17m later |
| 3 — external, corroborating | recovered from | `agents/reports/rust-fidelity/final_holdout_contamination_disclosure.md`, outside this repository |
| 4 — external, corroborating | sha256 | `a749c698ec7ac38d6a9709627836761ad548ad11cc8b1748c1eea83f19ff650e`; the committed copy is `cmp`-identical to the on-disk original |
| 5 — external, weakest | filesystem metadata | birth = mtime = ctime = `2026-08-05 20:03:30`, i.e. never modified after writing. **Local metadata, and forgeable in principle** — it supports the dating, it does not carry it. Rows 1 and 2 carry it |
| 6 — external, context | the probe it describes | `agents/reports/rust-fidelity/a12_candidate_residuals_skipped_on_move_faint.md`, written `2026-08-06` |

**What the recovered document changes, and what it does not.** It converts the 60-seed
contamination from testimony into a dated, self-reported record with a corroborating commit
two hours later — so that span no longer *"rests on testimony alone"*, and the earlier note
in this section saying the path exists nowhere is superseded by this one rather than
deleted. What it does **not** do is make the seeds re-measurable: the disclosure itself says
*"'I didn't look at the number' is mitigation, not absolution"*. It also **strengthens** the
audit's finding, because it lists three dispositions under a heading that reads
*"Disposition, which is the repository owner's call and not mine"* — shift the window,
declare and proceed, or *"retire the range, reserve a fresh window entirely
(`19,300,000+`)"* — and closes *"I have **not** chosen. Until the owner decides, I am
treating all of `19,200,000+` as still reserved and will not touch it again."* C141 then
took the first option without the owner deciding. The owner has now taken the third, at the
seed block the disclosure itself named.

**Quoted completely, including the part that cuts the other way.** Option 1 also carries
*"This is what I would recommend."* — so the agent that later took option 1 was following
its own recorded recommendation, not inventing a window. That is stated because a demotion
note which quotes selectively is a hostage to the next reader, and it does **not** soften
the finding: a recommendation offered inside a section titled *"the repository owner's call
and not mine"*, immediately above *"I have **not** chosen"*, is precisely the deferral C141
overrode. Recommending a disposition and taking it are different acts, and the disclosure is
explicit about which one was the author's to make.

**C141 is DEMOTED, not deleted.** `reports/artifacts/c141_final_holdout_sweep.json`, its
replay, and `reports/c141_final_holdout_prediction.md` are now **dev-window evidence**: 200
then-fresh seeds, a 71-patch engine at `44ee1430708cbb55`, a self-chosen window. They are
ordinary development evidence and **terminal for nothing**. They must never be re-cited as
"the holdout result". They stay committed for three reasons: they are the only witness for
two of the four bands in `REGISTERED_BANDS`, so deleting them turns
`test_every_registered_band_has_a_committed_witness` red; they are the record of what those
seeds were spent on, and a spent band with no artifact reads as a virgin band; and a
superseded measurement that is retained and relabelled is the difference between a program
that corrects itself and one that edits its history.

**Why `19,300,000`, and the collision that had to be resolved.** The owner's reasoning is
that `19,2xx` is now a graveyard namespace and adjacency invites off-by-N archaeology
forever, so the replacement should be visibly distinct from everything touched.
`scripts/engine_transition_differential.py` previously spent the literal `19,300,000` in
prose, as the canonical typo the unbounded floor exists to catch; an illustration that names
the real target reads backwards, so C151 moved the exemplar to **`19,700,000`**, which
**was** absent from every blob in the object database — reachable and unreachable alike —
immediately before this change.

**The past tense is load-bearing, and an earlier draft got it wrong.** That sentence was
first written in the present tense, and **its own commit falsified it**: as committed, six
blobs carry `19,700,000`, and all six are C151's own — the guard comment, this paragraph, the
prediction document and the guard test. It is the stale-denominator defect one turn tighter,
a measurement invalidated by the change that states it, so it is corrected here rather than
absorbed. What the evidence supports is a claim about the database *before* the edit; what
keeps the exemplar honest going forward is not its absence from prose but that **no artifact
ever records a seed there**, which `tests/test_seed_registry_coverage.py` already enforces
for every band outside the registered four.

**`19,300,000`–`19,300,199` is virgin, and the scope of that word is the scan that produced
it.** Three passes, all over **every ref and the reflog**, not over `main`:

1. the shape-agnostic `_seed_intervals` extractor from
   `tests/test_seed_registry_coverage.py` over every `*.json` blob under `reports/` or
   `docs/` reachable from `git rev-list --objects --all --reflog` — **900** blobs, **145**
   reaching fidelity seed space. The union of every fidelity seed ever touched is exactly
   the four bands already in this table, and it does not intersect the window;
2. a boundary-correct whole-number token pass over **every reachable blob**, any path and
   any file type;
3. the same pass over **every object in the database** — `git cat-file
   --batch-all-objects` — unreachable and dangling objects included. The only blobs carrying
   a value inside the window are C151's own, the recovered disclosure (whose option 3 names
   `19,300,000+`), and the pre-C151 revisions of the guard comment that C151 moves. **No
   artifact, no `seeds.min`/`max`, no `run.seed_start`.**

**The object counts are deliberately not quoted here, and that is the lesson of this PR's own
correction applied one more time.** The database grows with every commit — including the
commit that would state the figure, and including the merge that lands it — so a permanent
ledger carrying it would be stale on arrival, exactly as the `19,700,000` sentence above was
falsified by its own commit. For orientation only, re-derived on the merged tree at the head
of this PR and expected to drift: 23,778 objects, 8,766 blobs, 8,529 reachable, 237
unreachable, 42 blobs carrying an in-window value and 0 of them artifacts. **Re-derive the
scan; do not quote the denominator.**

One blob does not parse — a conflict-marker intermediate of
`reports/c102_consumed_choice_double_mutation.json`, reachable only from the reflog. An
unparseable blob makes a "nothing here" claim *easier*, so it was read by hand: the only
seeds in it are `19000038` and `19000113`, both in the dev band.

**Ratified is not swept, and the row above must not be read as permission to run today.**
The sweep's trigger is a precondition on the program state, not a date: **the ledger must be
terminal and the engine fingerprint declared frozen for the claim.** C116 already places
item 13 last. The reasoning is C141's own failure mode generalised — it measured
`44ee1430708cbb55` / 71 patches while patches were still landing weekly, on a program whose
job is landing patches, and `main` now ships `8e912b45544034e6` / 74. A sweep taken today
buys an unbiased measurement of a fingerprint that is superseded within days. Registration
must predate any result by construction; execution must not.

**The band is deliberately absent from `REGISTERED_BANDS`** in
`tests/test_seed_registry_coverage.py`. That tuple's rule is that a band joins it only once
the sweep that fills it is committed, and **ratification is not a witness**.
`TheRatifiedC151WindowIsNotYetSweptTests` pins the unswept state and re-derives the window's
virginity from the live corpus, so the row cannot quietly become a measurement and cannot
quietly stop being true. `OWNER_RATIFIED` in the guard carries the window and the owner's
name, so a future change to either is a diff requiring review — the point being to convert
the blessing from a sentence an agent can walk past into a mechanical check.

**The final-holdout row was false twice over, and is corrected above.** Until this
amendment it read `19,200,000`–`19,200,199` / *"reserved, untouched"*. The row was
written at `785e28e9` (#1071, 2026-08-04 11:03); C141's sweep landed at `aa2f2d40`
(2026-08-07 03:00) and falsified it, and nothing here moved:

* *"Untouched"* was false **at the latest** from `aa2f2d40`, and the record does not
  fix an earlier bound. `reports/artifacts/c141_final_holdout_sweep.json` is committed
  and records `seeds.min` `19,200,060`, `seeds.max` `19,200,259`, 200 distinct seeds,
  **16,274 boundaries measured**, 16,268 matched, 2 divergent,
  `acceptance_eligible: true`. Read the artifact, not this sentence.
  **It may have been false from the moment it was written.** The pre-guard 60-game run
  on `19,200,000`–`19,200,059` is datable in this repository only as *"before the
  `#1122` guard existed"*, and `#1122` is `5a44c04e`, **2026-08-05 22:20 — a day after
  the row**. Nothing narrows it further: there is no C-entry for the run, no committed
  sibling artifact (`distinct == 60` matches only `c26`, `c27` and `c6`, all far below
  the fidelity floor), and no `git log -S` hit. So "the row was true when written" is
  *not* something this record supports, and an earlier draft of this note asserted it
  anyway. What is established is the ordering of the two commits above.
* The recorded span stopped describing anything that happened. C141 chose its window
  to skip the contaminated head, so the sweep starts 60 seeds *inside* the
  registration and ends 60 seeds *past* it: `19,200,000`–`19,200,059` was never
  swept, and `19,200,200`–`19,200,259` was swept without ever having appeared in
  this registry. The choice was pre-registered — `reports/c141_final_holdout_prediction.md`
  names `19,200,060`–`19,200,259` before the run — so this is a bookkeeping failure
  in the table, not an undisclosed sweep.

**On the contaminated 60, what the record can and cannot support.**
`reports/c141_final_holdout_prediction.md` (lines 13–17) discloses that a
convenience shell loop over three `--seed-start` values executed 60 games of
`19,200,000`–`19,200,059` before the `#1122` guard existed, that the JSON was
**deleted unread**, and cites
`reports/rust-fidelity/final_holdout_contamination_disclosure.md` as an external
record. That path exists in **no tree in this repository's history** — checked with
`git rev-list --objects --all --reflog`, whose 22,000-odd lines name **1,487 distinct
paths** across every ref and reflog, zero of them matching `contamination` or
`rust-fidelity`. (The line count is not the path count; most lines are commits and
trees, and most named blobs repeat. The negative holds on either denominator.)
**The path is very likely a dropped prefix rather than a missing file**: `5a44c04e`'s
own commit message gives it as `agents/reports/rust-fidelity/…` and marks it *"outside
this repo"*, so the prediction file's in-repo-looking spelling points at an external
tree. That resolves the *intent* and changes nothing about the *auditability*: from
inside this repository the contamination is attested only by the prose above and by
`tests/test_final_holdout_guard.py`, whose module docstring records the same incident
and whose pins now prevent a repeat. Nothing here says
whether those 60 seeds, or the unswept head generally, may be swept again — that is
an open owner decision and this table does not pre-empt it in either direction.

**Scanning for consumed seed space: there are FOUR artifact shapes, not one.** A
selector keyed on `seeds.min` under `reports/artifacts/` answers for 80 of the 93
committed artifacts that reach fidelity seed space, and it is wrong. (Those two
figures were measured at `8f52ac95`, out of a 375-file corpus; they move with every
committed sweep, so re-derive them rather than quoting this line. The whole-corpus
total is deliberately left out of the table below for the same reason — a permanent
ledger should not carry a number that changes on every artifact commit.) Its highest
seed is exactly `19,200,259`, which is how the claim *"everything at or above
`19,200,260` is virgin"* got asserted here today — and refuted by
`reports/c73_eight_hundred_game_sweep.json`, which is consumed, is named by
`tests/test_final_holdout_guard.py`, sweeps `19,500,000`–`19,500,799`, lives one
directory **up** in `reports/`, and records `run.seed_start` + `run.games` with no
`seeds` object at all. The four shapes actually in the corpus:

| shape | example |
| --- | --- |
| `seeds.min` / `seeds.max` | 80 files under `reports/artifacts/`, 15 directly under `reports/` (16 counting any nested `min`/`max`), 0 under `docs/` |
| `run.seed_start` + `run.games` | `reports/c72_fresh_local_sweep.json`, `reports/c73_eight_hundred_game_sweep.json` |
| `sample.seed_start`, closed by `sample.seed_end` **or** by `sample.games` | `reports/c82_head_era_fresh_sweep.json` carries `seed_end`; `c83` and `c86` carry only `games` |
| `windows.{dev,holdout}.seed_start` + `games` | `reports/artifacts/c147_g33b_gate_reach.json` |

Note the third row: a span whose end must be *computed* from a game count is still a
span, and three of these files never state their last seed at all. A scanner that
reads only explicit endpoints under-reports consumed space.

Scan `reports/` **and** `docs/`, recursively, and do not key on a shape. The
enforced version is `tests/test_seed_registry_coverage.py`, which walks every
committed JSON under both trees, extracts seed intervals structurally rather than by
shape, and asserts in both directions: no committed fidelity seed lies outside a
band recorded above, and every band recorded above has a committed witness. It
deliberately asserts nothing about the *status words* in this table — the validation
holdout row is the counterexample that kills the obvious rule, being both "reserved"
and swept on every fix branch.

**What that pin does NOT cover, so nobody reads it as covering it.** It enforces
*containment* — that no committed seed escapes a registered band — and nothing about
*multiplicity*. A second sweep of `19,200,060`–`19,200,259` would sit inside a
registered band and pass every assertion in the module. The "exactly one measurement"
invariant below is `#1122`'s job, enforced at run time by
`_reject_unguarded_final_holdout` in `scripts/engine_transition_differential.py` and
pinned by `tests/test_final_holdout_guard.py`; this pin is the record-keeping half,
not the enforcement half.

**The invariant, restated against the ACTIVE registration.** If a future reader
finds seeds in a registered acceptance band that Appendix Z12 does not account for
— in particular anything in **`18,000,000`–`18,701,249`** — an acceptance attempt
happened that this ledger does not record. Seeds at `19,000,000`+ are
fidelity-differential seeds and carry no such implication.

**The canonical list is in code, not in this prose.** `PUBLIC_CONSUMED_SEED_RANGES`
in `tests/test_cert_contract_registration.py` is the enforced registry, and
`reports/c32_current_engine_resweep_spec.json`'s `seed_blocks` holds the active
reservation. This table is a reader's map of them and can go stale; those two
cannot, because `test_blocks_are_disjoint_from_publicly_consumed_seeds` fails if a
registration collides. Prefer them over this table on any disagreement.

**A correction to this amendment's own first draft**, recorded because it is the
error the amendment exists to prevent. The first version of this table declared the
`2,000,000`+ band *"pristine — never consumed"* and set the sentinel on
`2,000,000`–`2,799,999`. Both were false. §J.7's "remains unconsumed" was era-true
when written and Appendix Z12 — some 3,600 lines later in this same file — records
the run that consumed `2,000,000`–`2,701,249`. The draft was derived from §5.2, a
pre-C15 section, instead of from the enforced registry, and it would have
un-sentinelled the live C32 reservation while false-positiving on five merged cert
reports. Caught by independent review.

**Two related staleness items left standing, deliberately.** §5.2's *"measurement
and fix development stay below seed 2,000,000"* now contradicts this section's
blessing of `19,000,000`+, and seven restatements of "seed block 2,000,000+ remains
unconsumed" (lines ~2594, 3010, 3234, 3539, 3855, 4548, 4792) predate Z12. They are
era-true where they sit and rewriting history in place is worse than a pointer;
this note is the pointer.

**The new invariant this creates**, which is the one C116 Phase 1 depends on: the
final holdout block — the *registered* one, `19,200,000`–`19,200,199`, not
"`19,200,000`+", which as written also captures `c73`'s consumed
`19,500,000`–`19,500,799` — must appear in **exactly one** measurement in the whole
record. If it appears twice, it stopped being a holdout and the terminal fidelity
claim built on it is void. Iterating against a window and then reporting that window
is the failure mode the entire 208 → 7 era is exposed to (C116 M6); these two
reserved blocks exist to bound it. What the record now shows against that invariant
is set out in the seed-registry table above: one gated 200-game sweep on
`19,200,060`–`19,200,259`, and one pre-guard 60-game run on
`19,200,000`–`19,200,059` whose output was deleted unread.

**What would have changed the decision.** Nothing about the measurement — the
apparatus was gated, `acceptance_eligible: true`, 97.80 % coverage, 0 harness
errors, 0 unclassified. The blocker was entirely on the simulator-fidelity side
(§J.3, §J.4). That distinction is why the hold is recorded as a *result*, not as
a failure to execute.
---

# Appendix K — the Substitute divergence: mechanism found, §J.3 premise retracted

Branch `scott/substitute-diagnosis`. The implementation lane falsified §J.3's
premise before building on it — the world constructor **does** express
substitute — leaving my observation real but its mechanism unexplained. Diagnosed
here.

## K.1 §J.3's stated cause was WRONG

Full payload dump at the exact boundary (`seed 1500002 step 75`):

```
BUILT p1: volatiles=()               substitute_health=0
BUILT p2: volatiles=('substitute',)  substitute_health=66
```

The substitute **is** in the payload and **is** on the built world with health.
§J.3 said "Substitute is not expressed in the constructed world". **That is
retracted.** The observation (Showdown breaks the sub, the engine damages the
mon) stands; the cause I gave for it does not.

Process note: I asserted a cause from a single replay without dumping the
payload that would have falsified it. The replay showed *what* happened; I
narrated *why* without checking. The implementation lane caught it before
building — that stop is what the method rule is for, and it applied to me here.

## K.2 The real mechanism: fixed-damage moves bypass the Substitute

Synthetic probe, defender behind a 66 HP sub:

| attacker move | engine instruction | mon HP | sub HP | |
| --- | --- | --- | --- | --- |
| `seismictoss` | `Damage SideTwo: 88` | 264 -> **176** | 66 (untouched) | **sub bypassed** |
| `superfang` | `Damage SideTwo: 132` | 264 -> **132** | 66 (untouched) | **sub bypassed** |
| `tackle` | `DamageSubstitute SideTwo: 42` | 264 (safe) | 66 -> 24 | correct |

Seismic Toss and Super Fang are handled by the move-id-keyed
`choice_special_effect` path, which writes damage **directly to the Pokemon**
instead of going through the substitute-aware damage routing. Ordinary damaging
moves route correctly.

**This is the same handler family, and the same shape, as the Protect leak the
rapid-spin patch fixed** (`docs/engine_fidelity_findings.md`): `choice_special_
effect` sits outside the normal damage path, so it missed the Protect guard then
and misses Substitute routing now. Fixing one did not generalise to the other.

| | |
| --- | --- |
| Verdict | **ENGINE — damage routing**, patch lane |
| Not | materialization-side, and not the fresh-health pinning |
| Repro | `seed 1500002 step 75`; synthetic probe above reproduces in isolation |
| Reachability | **Seismic Toss is in 17 of 393 sets / 50 of 1,682 variants** — squarely on-distribution. Super Fang is absent from the pool, so it matters only for `gen3customgame`. |

**Share of the residue: 15 of the 49 sub-adjacent rows (31 %), all Seismic
Toss**, 14 of them in `roll_scaled_component`. The other 34 sub-adjacent rows do
**not** involve a fixed-damage move and are still unexplained — they are not
covered by this finding and must not be assumed to be.

## K.3 Separate finding: Night Shade and Dragon Rage are inert

Found while probing, and **not** substitute-related — they emit no instructions
even with no substitute present:

| move | engine damage, no sub, no immunity |
| --- | --- |
| `nightshade` | **0** |
| `dragonrage` | **0** |

Same shape as the confirmed Flail bug: present in the move table, no gen3
behaviour. **Both are absent from the gen3 randbats pool** (0/393 sets), so this
is latent — it matters for `gen3customgame` or a pool change, and is not an
acceptance blocker. Filed rather than prioritised.

## K.4 What this means for the treatment ladder

The pre-approved ladder (fix materialization exactly / express zero-hit subs and
fail closed on hit subs / support-based interval sampling) was scoped to a
materialization or health-pinning cause. **Neither is the cause here**, so the
ladder does not apply to these 15 rows — they need the engine-side damage-routing
fix.

The ladder's premise is still independently sound and worth keeping: the harness
pins substitute health at a fresh `maxhp/4` via `approximate_substitute_health`,
and gen3 emits `-activate` with no amount when a sub is hit, so **post-hit sub
health genuinely is not public**. That remains a real approximation in the
harness — it simply is not what produced this row, where the sub was untouched
since creation and `maxhp/4` was therefore exact.

---

# Appendix L — acceptance cycle on the 24-patch set

Fresh clean vendor, both wheels rebuilt, stamped
(`faba658cf3432251`, 24 patches + 8 crate sources); 21 crate suites green;
freshness ladder live.

## L.1 ACCEPTANCE VERDICT: NOT MET — run not started (second hold)

300 games, seeds **1500000-1500299**, strict, `--repros-per-game 300`.

| | 22-patch (§J) | **24-patch (this)** |
| --- | --- | --- |
| measured | 23,334 / 23,860 = 97.80 % | **23,334 / 23,860 = 97.80 %** |
| matched | 23,023 | 23,036 |
| **diverged** | 311 | **298 — 1.28 %** |
| harness errors | 0 | **0** |
| `unclassified` | 0 | **0** |
| adjudicated `limit:` | 47 | **47** |
| **OUTSIDE limit classes** | 264 | **251** |
| throughput | 1,419 games/h | 1,510 games/h |

Bar is zero outside adjudicated limit classes. **251.** The 8x1250 was **not**
started; seed block 2,000,000+ remains unconsumed (§J.7 invariant holds).

## L.2 Both fixes verified — and the collapse predictions, scored honestly

Direct probes confirm both landed:

* **#915 fixed-damage routing** — Seismic Toss and Super Fang into a 66 HP sub
  now leave the mon at 264/264 and take the sub to 0. Previously they put 88 and
  132 straight into the Pokemon.
* **#914 Solar Beam** — release damage now varies with weather (166 none / 117
  sand / 249 sun) where it was previously a single wrong value.

Scoring the predictions on record:

| Prediction | Outcome |
| --- | --- |
| 15 Seismic Toss rows collapse | **13 of 15 resolved.** Sub-adjacent rows 49 -> 36. |
| Solar Beam drives broad movement in `roll_scaled_component` | **Not testable here — there are ZERO Solar Beam rows in seeds 1500000-1500299**, in either run. Verified separately on seeds 1350000-1350059, where the filed row lives: `seed 1350004 step 66` is **RESOLVED** (83 -> 81 diverged). |
| sub-adjacent cascade rows collapse | partially — the cascade was smaller than the 49-row figure implied; only the 13 fixed-damage rows moved |

**Net −13 divergences, and all 13 are the Seismic Toss rows.** The honest reading
is that both fixes work and neither had the breadth I projected: I sized the
Solar Beam impact from a row in a *different seed set* and let that imply
movement in this one, and I let "49 sub-adjacent" imply a cascade when only the
15 fixed-damage rows were ever attributable.

## L.3 The Leftovers/psn families did not move AT ALL

| Class | 22-patch | 24-patch |
| --- | --- | --- |
| `component_missing_in_engine:itemleftovers` | 27 | **27** |
| `component_extra_in_engine:itemleftovers` | 25 | **25** |
| `component_missing_in_engine:psn` | 24 | **24** |
| `component_extra_in_engine:psn` | 9 | **9** |
| `component_extra_in_engine:itemleftovers,psn` | 6 | **6** |

Identical counts across two independent builds. These 91 rows (36 % of the
residue outside limit classes) are demonstrably independent of everything fixed
so far, and they are the largest unexplained block.

## L.4 A surviving row replayed — and the obvious hypothesis falsified

`seed 1500004 step 73` (`seismictoss` vs `hypnosis`), one of the 78 structural
rows:

```
Showdown: |-status|p1a: Registeel|slp|[from] move: Hypnosis
          |cant|p1a: Registeel|slp          <- Seismic Toss never happens
engine:   Damage SideTwo: 78                 <- it happens anyway
```

The obvious reading is "the engine lets a mon slept this turn still move". **That
is false.** Isolation probe, faster sleeper vs slower attacker:

| sleep move | engine behaviour |
| --- | --- |
| `spore` (100 %) | 1 branch: target slept, **attack blocked** — correct |
| `hypnosis` (60 %) | 2 branches: 60 % slept+blocked, 40 % missed+attacked — correct |

So the engine's sleep gating is right, and the divergence is something about
*this boundary's constructed world* — the replayed branch had no Hypnosis outcome
at all (single 100 % branch, no status instruction), which a 60 %-accurate move
cannot produce from a healthy target. **Mechanism not established; not
attributed.** It needs a payload dump like §K.1, which is the method that worked
last time and which I did not have budget to complete this cycle.

## L.5 Residue table with verdicts

| Class | n | Verdict |
| --- | --- | --- |
| `roll_scaled_component` | 133 | **OPEN** — 78 structural, 55 magnitude. One replayed (L.4), obvious hypothesis falsified, mechanism unknown |
| `limit:roll_divergent_lethality` | 42 | adjudicated limit |
| `component_missing_in_engine:itemleftovers` | 27 | **OPEN** — untouched by 2 builds |
| `component_extra_in_engine:itemleftovers` | 25 | **OPEN** — untouched |
| `component_missing_in_engine:psn` | 24 | **OPEN** — untouched |
| `component_extra_in_engine:psn` | 9 | **OPEN** — untouched |
| `component_extra_in_engine:itemleftovers,psn` | 6 | **OPEN** — untouched |
| `limit:world_sample_drag_target` | 5 | adjudicated limit |
| 16 smaller classes | 27 | **OPEN** |

## L.6 Recommended next step

The Leftovers/psn block (91 rows, stable across builds) is the highest-value
target and has never been replayed. §K showed that a payload dump turns an
unexplained observation into a precise mechanism in one pass; the same treatment
applied to `component_extra_in_engine:itemleftovers` and
`component_missing_in_engine:psn` is what I would do next, before any further
engine work is scoped from class names.

## L.7 Artifacts

| Artifact | Path (under `<scratch>/reports/`) |
| --- | --- |
| 24-patch 300-game report | `cyc300.json` |
| per-game checkpoint | `cyc300.jsonl` |
| run log | `cyc300.log` |
| triage of dominant class | `tri_cyc.json` |
| Solar Beam verification (60 games) | `sb60.json` |
| prior 22-patch run for comparison | `final300.json` |

`<scratch>` =
`/private/tmp/claude-501/-<home>-workspace-agents-pokezero-agent/47b7c392-a7b8-43cf-b071-8a500f9bc9bf/scratchpad`
# Appendix M — payload-dump diagnosis of the 91-row Leftovers/psn block

Diagnosis only, §K method: full payload + built world + instruction dump at each
boundary, mechanism stated from evidence.

## M.1 Headline: the block is NOT one mechanism

The question posed was whether one mechanism spans most of the 91. **It does
not**, and the split is visible before any replay:

| Sub-class | n | gating | step mentions slp/Rest |
| --- | --- | --- | --- |
| `missing:itemleftovers` | 27 | 16 exact / 11 support | **13** |
| `missing:psn` | 24 | 18 exact / 6 support | **17** |
| `extra:itemleftovers` | 25 | 20 exact / 5 support | 4 |
| `extra:psn` | 9 | 9 exact | **0** |
| `extra:itemleftovers,psn` | 6 | 6 exact | **0** |

The **missing** half is sleep-entangled (30 of 51 steps mention slp/Rest); the
**extra** half is almost entirely not (4 of 40). Treating them as one block was
wrong — as suspected in the tasking.

## M.2 CONFIRMED mechanism (missing half): Rest-sleep provenance is lost

`seed 1500004 step 30`, `earthquake` vs `hypnosis`:

```
Showdown: |-status|p1a: Claydol|slp|[from] move: Hypnosis
          |cant|p1a: Claydol|slp          <- Earthquake never happens
engine:   Damage SideTwo: 183             <- it happens; NO sleep branch at all
```

Payload dump — p1's bench:

```
{"active": false, "condition": "121/209 slp", "species": "Dusclops"}
```

and the game's own history, step 16:

```
|move|p1a: Dusclops|Rest|p1a: Dusclops
|-status|p1a: Dusclops|slp|[from] move: Rest
```

**Dusclops is REST-asleep.** Gen 3 Sleep Clause exempts self-inflicted Rest
sleep, so Showdown correctly allows Hypnosis on Claydol.

The engine is **also correct** — isolation probe, benched teammate asleep:

| bench state | engine result |
| --- | --- |
| no sleeper (control) | 2 branches, target sleeps ✓ |
| `sleep_turns=1` (move-induced) | 1 branch, **target does not sleep** ✓ clause fires |
| `rest_turns=2` (Rest-induced) | 2 branches, target sleeps ✓ clause exempts Rest |

**The defect is in world construction.** The payload carries only `slp` with no
provenance, so a Rest-asleep benched mon is encoded with `sleep_turns` instead of
`rest_turns`. The engine's clause then fires when it should not, the sleep move
silently fails, the target acts when Showdown says it cannot, and every HP-driven
residual downstream disagrees — which is what surfaces as `missing:itemleftovers`
and `missing:psn`.

`[from] move: Rest` is a **public protocol line**, so this is publicly derivable
and not an information gap.

| | |
| --- | --- |
| Verdict | **WORLD CONSTRUCTION** — engine/world lane |
| Not | the engine's sleep clause (verified correct in both directions) |
| Repro | `seed 1500004 step 30`; also `seed 1500004 step 73` (the §L.4 survivor — same mechanism, and this **retires that open item**) |
| Spec | carry Rest-vs-move sleep provenance into the materialization payload and set `rest_turns` rather than `sleep_turns` for Rest-asleep mons |
| Coverage | bounded by the 30 of 51 sleep-entangled `missing` rows; **not** asserted for the rest |

## M.3 CHARACTERIZED (extra half): faint-divergent residuals

`seed 1500005 step 67`, `crunch` vs a switch:

```
Showdown: |-crit|p2a: Electabuzz
          |-damage|p2a: Electabuzz|0 fnt   <- crit KOs; NO residuals for a fainted mon
          |faint|p2a: Electabuzz
engine:   Damage SideTwo: 74               <- non-crit; Electabuzz SURVIVES
          Heal SideOne: 17, Damage SideTwo: 15, ToxicCount: 1   <- so it gets residuals
```

Showdown's roll critted and killed; the engine's did not. A fainted mon takes no
end-of-turn residuals, so the survivor's Leftovers and toxic ticks appear in one
sim and not the other — surfacing as `extra:itemleftovers`, `extra:psn` and
`extra:itemleftovers,psn`.

This is the **same phenomenon** already adjudicated as
`limit:roll_divergent_lethality` — two sims taking different stochastic
outcomes — but it lands in a *different class* because when a mon faints its
residuals vanish **entirely** (a component present/absent) rather than appearing
as a clipped `capped_lethal` value, which is the only shape the limit classifier
recognises.

| | |
| --- | --- |
| Verdict | **CHARACTERIZED, one row** — consistent with the existing limit class, not a new engine bug |
| Status | **NOT adjudicated.** One replay. The `extra` half is 40 rows and I am not generalising from a single sample — that is the exact error this ledger has logged three times |
| Next | replay a further sample of `extra:*`; if the pattern holds, the honest fix is to widen `limit:roll_divergent_lethality` to recognise vanished-residual shapes, which is a **classifier** change, not an engine one |

## M.4 Mechanism table

| Sub-class | n | Mechanism | Verdict | Repro |
| --- | --- | --- | --- | --- |
| `missing:itemleftovers` | 27 | Rest-sleep provenance lost -> spurious Sleep Clause block | **WORLD CONSTRUCTION** (confirmed) | seed 1500004 step 30 |
| `missing:psn` | 24 | same | **WORLD CONSTRUCTION** (confirmed, same shape) | seed 1500004 step 30 |
| `extra:itemleftovers` | 25 | faint-divergent residuals | characterized, likely `limit:` | seed 1500005 step 67 |
| `extra:psn` | 9 | same | characterized, likely `limit:` | — |
| `extra:itemleftovers,psn` | 6 | same | characterized, likely `limit:` | — |
| §L.4 survivor (`seed 1500004 step 73`) | 1 | Rest-sleep provenance — **retires the open item** | **WORLD CONSTRUCTION** | seed 1500004 step 73 |

My §L.4 speculation ("the world's target wasn't asleep at build time") was
**wrong** — the target's own status was fine; it was a *benched teammate's* sleep
provenance. Verified against the payload rather than inherited, as instructed.

## M.5 What this changes

* One engine-lane spec: **Rest-sleep provenance in the materialization payload.**
* One likely **classifier** item, not an engine one: vanished-residual faints
  belong with `limit:roll_divergent_lethality`. If that holds across a proper
  sample, ~40 rows move from "unexplained residue" to "adjudicated limit", which
  would materially change the acceptance arithmetic — so it deserves the sample
  before anyone banks it.
* No fixes in this pass, per the §K standard.

---

# Appendix N — the extra-half sample: SPLIT, no widening

Stratified sample of 16 of the 40 extra-half rows (8 `extra:itemleftovers`,
5 `extra:psn`, 3 `extra:itemleftovers,psn`), each replayed and verdicted.

## N.1 Result: the hypothesis fails — 2 of 16

| Verdict | n | share |
| --- | --- | --- |
| **(a)** roll-divergent lethality, vanished residuals | **2** | 12.5 % |
| (b) faint IS reproducible by an engine branch | **9** | 56 % |
| (b) no faint in the step at all | **5** | 31 % |

**The classifier widening is NOT drafted.** §M.3's characterization — built on a
single row — does not survive a sample. Had it been banked, ~40 rows would have
moved into an adjudicated limit class on a mechanism that holds for 2 of them,
which would have lowered the acceptance bar on a false premise. This is the
outcome the sample requirement existed to produce.

## N.2 The three sub-populations

**(b) faint reproducible — 9 rows.** Showdown faints a mon *and* an engine branch
faints the same mon. So this is **not** roll-divergent lethality: the engine can
reach the outcome. Something else in those branches disagrees. Undiagnosed.

**(b) no faint at all — 5 rows, every `extra:psn` row sampled.** Diagnosed, and
it is a new engine bug (N.3).

**(a) genuine roll-divergent lethality — 2 rows.** Real, but a small minority.

## N.3 CONFIRMED ENGINE BUG: gen3 Toxic never misses

`seed 1500027 step 4`:

```
Showdown: |move|p2a: Tentacruel|Toxic|p1a: Corsola|[miss]
          |-miss|p2a: Tentacruel|p1a: Corsola      <- Corsola is NOT poisoned
engine:   every branch carries  p1 exact=psn=-16   <- Toxic landed in all of them
```

Isolation probe:

```
Toxic  -> pct=100.00  ['ChangeStatus SideOne-P0: NONE -> TOXIC', ...]   ONE branch
```

**The engine applies Toxic at 100 %.** Ground truth, read from the vendored
simulator: `data/moves.ts` gives Toxic `accuracy: 90`, and
`data/mods/gen4/moves.ts` overrides it to **85**; gen3 has no override of its own
and therefore inherits gen4's 85 (the same inheritance rule as Wish and Sleep
Talk).

| | |
| --- | --- |
| Verdict | **ENGINE — move accuracy**, patch lane |
| Rule | gen3 Toxic accuracy **85** (base 90 -> gen4 85 -> inherited by gen3) |
| Engine | 100 % — no miss branch generated at all |
| Repro | `seed 1500027 step 4`; isolation probe above |
| Reachability | Toxic is in **152 of 393 sets (39 %)** and **443 of 1,682 variants (26 %)** — one of the most common moves in the pool |
| Residue reach | **54 of 298 divergent rows (18 %)** have a player choosing Toxic. That is an upper bound on what this could explain, not a claim about how many it does. |

Because the miss branch is absent entirely rather than mis-weighted, every
Toxic use in search is priced as a guaranteed poison — a systematic optimism on a
move that appears in a quarter of all sets.

## N.4 Deliverable summary

| Item | Outcome |
| --- | --- |
| Classifier widening | **NOT drafted** — sample split 2/16 |
| Adversarial review | not required; nothing to review |
| New engine spec | **gen3 Toxic accuracy 85 %** (N.3) |
| Still undiagnosed | 9 sampled "faint reproducible" rows; the mechanism is not lethality divergence |
| Retired | §M.3's characterization of the extra half — superseded by this sample |

The Rest-provenance fix (§M.2, 51 rows) remains dispatched to the
world-construction lane and is unaffected by this result.

---

# Appendix O — the 9 faint-reproducible rows: terminal-boundary residual truncation

Diagnosis only. Toxic-guaranteed-poison ruled out **first, per row**, as directed.

## O.1 Toxic rule-out

Four of the nine chose Toxic. The Toxic-never-misses signature is *Showdown shows
`|-miss|` and the engine poisons anyway*. **No row in the sample shows a Toxic
miss** (`showdown-miss=False` on all nine), and where the engine carries an extra
`psn` component it is explained by O.2 rather than by a phantom poison. **Toxic
is not implicated in any of the nine.**

## O.2 CONFIRMED mechanism: the battle ends and Showdown stops the residual block

`seed 1500064 step 126`:

```
|-heal|p2a: Zapdos|90/255 tox|[from] item: Leftovers
|-damage|p2a: Zapdos|0 fnt|[from] psn
|faint|p2a: Zapdos
|win|PokeZero p1          <- battle over; p1's OWN Leftovers never fires
```

`seed 1500081 step 61` is identical in shape. Showdown's last Pokemon faints
during the residual phase, the battle ends **immediately**, and every residual
still queued — notably the winner's own Leftovers — never executes. The engine
runs the residual block to completion and applies them, so the engine has
components Showdown does not.

This is why the rows read as "faint reproducible": the engine **can** faint the
mon, and does. The divergence is not the faint, it is everything Showdown skips
*after* it.

**Census, not a sample** (the property is `|win|` in the step, so it is
decidable for every row):

| | |
| --- | --- |
| of the 9 sampled | **7 terminal** |
| of the whole residue | **32 of 298 (11 %)** |
| of the 40 extra-half rows | **24** (19 `extra:itemleftovers`, 5 `extra:itemleftovers,psn`) |

## O.3 The 2 non-terminal rows are each something else

* `seed 1500084 step 31` — **genuine roll-divergent lethality, other side.**
  Showdown's Lickitung survives at 17/322 (`move=-179`); the engine's roll kills
  it (`capped_lethal=-196`). My batch classifier called it "faint reproducible"
  because it keyed on the faint Showdown *did* have (p2 Delibird, Recoil) rather
  than the one it did not. A classifier limitation, not a new mechanism.
* `seed 1500005 step 67` — component **splitting**: Showdown emits one
  `capped_lethal=-150`; the engine emits `move=-149` plus `capped_lethal=-1`.
  Same total, different decomposition. Undiagnosed.

## O.4 Mechanism table

| Mechanism | n of 9 | Residue-wide | Verdict | Repro |
| --- | --- | --- | --- | --- |
| terminal-boundary residual truncation | **7** | **32 (11 %)** | see O.5 | seed 1500064 step 126 |
| roll-divergent lethality (other side) | 1 | — | existing adjudicated limit | seed 1500084 step 31 |
| component splitting on a lethal hit | 1 | — | **open** | seed 1500005 step 67 |
| Toxic-guaranteed-poison | **0** | — | ruled out | — |

## O.5 Recommendation: fix the engine, do not widen the limit class

Two treatments are available and they are not equivalent.

**Engine fix (recommended).** Stop the residual block when the battle ends, which
is what Showdown does. Small, exactly specified, testable, and it removes the
class outright — 11 % of the residue — **without touching the acceptance bar**.

**Limit widening (not recommended).** One could argue a terminal boundary is
unobservable-by-construction: the game is over, no decision follows, and the
winner's post-final-turn HP never influences play or training. That argument is
*plausible*, which is exactly why I am not banking it. It shrinks the residue by
moving rows into an adjudicated class rather than by making the two simulators
agree, and §N is a fresh reminder of what happens when a plausible
reclassification goes unchallenged.

If the engine lane judges the fix not worth it, the widening should go to
adversarial review on rows the reviewer picks — the §N protocol — rather than
being adopted here.

---

# Appendix P — cycle three (26 patches): NOT MET, third hold

Fresh clean vendor, both wheels rebuilt and stamped (`4f4a174102fadc14`,
26 patches + 8 crate sources); 24 crate suites green; gate current.

## P.1 ACCEPTANCE VERDICT: NOT MET — run not started

300 games, seeds 1500000-1500299, strict.

| | cycle 2 (24p) | **cycle 3 (26p)** |
| --- | --- | --- |
| measured | 23,334 / 23,860 = 97.80 % | **23,335 — 97.80 %** |
| **diverged** | 298 | **306 — 1.31 %** |
| harness errors | 0 | **0** |
| `unclassified` | 0 | **0** |
| adjudicated `limit:` | 47 | **47** |
| **OUTSIDE limit classes** | 251 | **259** |

Seed block 2,000,000+ remains unconsumed.

## P.2 The count went UP — and that is mostly the measurement getting stronger

Row-identity diff, not just class counts:

| | n |
| --- | --- |
| fixed since cycle 2 | **11** |
| newly divergent | **19** |
| net | **+8** |

**All 19 new rows are `gating=exact`.** The Rest-provenance fix made a large
population of sleep boundaries exactly constructible that previously fell back to
the hidden-counter sweep:

| counter | cycle 2 | cycle 3 |
| --- | --- | --- |
| `gating:exact` | 17,161 | **21,258** (+4,097) |
| `gating:support` | 6,173 | **2,077** (−4,096) |

The support sweep validates against the **union** of a counter's legal values, so
it matches whenever *any* value would. Moving ~4,100 boundaries to exact gating
replaces that union with one committed state — a strictly stronger bar. Some of
what it now catches was always divergent and simply unmeasurable.

**That does not make the 19 rows acceptable**, and the ledger does not get to
claim the increase is purely virtuous. P.3 shows a real defect underneath.

## P.3 CONFIRMED: the `restSleepAttempts` -> `rest_turns` conversion is off by one

`seed 1500013 step 72`, payload:

```
p1 Dunsparce  "90/319 slp"   restSleepAttempts: 1   ->  world rest_turns=2
p2 Dunsparce  "175/319 slp"  restSleepAttempts: 2   ->  world rest_turns=1
```

Showdown: **both** mons are `|cant|...|slp`. Neither moves.
Engine: p2 wakes and attacks (`move=-85` on p1).

Engine `rest_turns` semantics, probed directly:

| `rest_turns` | attacker moves this turn |
| --- | --- |
| 1 | **yes** — wakes |
| 2 | no |
| 3 | no |

So `rest_turns=1` means "wakes now". A mon that has made **2** sleep attempts
must still be asleep in gen3 (Rest costs two full turns; the user acts on the
third), and Showdown confirms it directly here. The conversion produced
`rest_turns = 3 - attempts`; it needs `4 - attempts`:

| attempts | current -> engine behaviour | required |
| --- | --- | --- |
| 1 | 2 -> stays asleep ✓ | 3 |
| 2 | **1 -> WAKES ✗** | **2** |
| 3 | 0 | 1 -> wakes ✓ |

| | |
| --- | --- |
| Verdict | **WORLD CONSTRUCTION** — off-by-one in the attempt-count conversion |
| Repro | `seed 1500013 step 72` (payload above; engine probe above) |
| Why it matters now | the same fix moved 4,097 boundaries onto exact gating, so this error went from absorbed-by-the-union to load-bearing |

## P.4 The battle-end residual fix did not move the residue

Terminal-boundary rows (`|win|` in the step) — a census property, decidable per
row:

| | cycle 2 | cycle 3 |
| --- | --- | --- |
| terminal divergent rows | **32** | **32** |

**Unchanged.** §O predicted these would clear. They did not. Either the fix does
not cover the case §O.2 documented (the winner's own Leftovers firing after the
final faint), or it is not reaching this path. Handed back to that lane with the
§O.2 repro (`seed 1500064 step 126`) intact.

## P.5 What worked

| Fix | Predicted | Actual |
| --- | --- | --- |
| Toxic Poison-gate | 8-set share | **`component_extra_in_engine:psn` 9 -> 0**, class extinct ✓ |
| Rest-provenance | ~51 rows | directionally right — +4,097 exact-gated — but P.3 off-by-one |
| battle-end residuals | 32 rows | **0 rows** ✗ (P.4) |
| #879 sub/confusion/perish | some | no measurable movement in those families |

## P.6 Residue

| Class | n | Note |
| --- | --- | --- |
| `roll_scaled_component` | 150 | +17, and 17 of the 19 new rows land here (P.3) |
| `limit:roll_divergent_lethality` | 42 | adjudicated |
| `component_extra_in_engine:itemleftovers` | 29 | 24 terminal (P.4) |
| `component_missing_in_engine:itemleftovers` | 28 | Rest-provenance family |
| `component_missing_in_engine:psn` | 24 | unchanged across three builds |
| `limit:world_sample_drag_target` | 5 | adjudicated |
| 14 smaller classes | 28 | |

## P.7 Artifacts

Under `<scratch>/reports/`: `c3.json`, `c3.jsonl` (resumable), `c3.log`;
`cyc300.json` (cycle 2) for the diff. `<scratch>` =
`/private/tmp/claude-501/-<home>-workspace-agents-pokezero-agent/47b7c392-a7b8-43cf-b071-8a500f9bc9bf/scratchpad`

---

# Appendix Q — AMENDMENT to P.3: the off-by-one diagnosis was wrong

§K precedent: the author retracts in place, the trail stays honest.

## Q.1 RETRACTED

**P.3 claimed the `restSleepAttempts -> rest_turns` conversion was off by one and
prescribed `4 - k`. Both the diagnosis and the prescription are wrong.**

`3 - k` is correct and was already correct. The build lane **refused the
instructed change and proved it at the simulator**: a plain Rest emits exactly
**two** `|cant|` lines and then wakes-and-acts on the third attempt. Both
conventions decrement-then-check, which the vendored source confirms directly —
`data/mods/gen3/conditions.ts`, `slp.onBeforeMove`:

```js
pokemon.statusState.time--;                    // decrement FIRST
if (pokemon.statusState.time <= 0) { pokemon.cureStatus(); return; }   // then check
this.add('cant', pokemon, 'slp');
```

With `time = 3`: attempt 1 -> 2, cant; attempt 2 -> 1, cant; attempt 3 -> 0,
cure and act. Two cants, act on the third. `4 - k` would have held every
Rest sleeper an extra turn.

**Why earlier checks passed:** `k = 0` is the single value where `3 - k` and
`4 - k` agree on behaviour, and that is the case the round-trip and unit checks
exercised.

## Q.2 The real mechanism: gen3's `skippedTime` refund

Verified independently at source rather than inherited — same file,
`slp.onStart` / `onSwitchIn` / `onBeforeMove`:

```js
onStart:     this.effectState.time = this.random(2, 6);
             this.effectState.skippedTime = 0;
onSwitchIn:  this.effectState.time += this.effectState.skippedTime;   // REFUND
             this.effectState.skippedTime = 0;
onBeforeMove: ... this.add('cant', pokemon, 'slp');
              if (move.sleepUsable) { this.effectState.skippedTime++; return; }  // acts anyway
              this.effectState.skippedTime = 0;
              return false;
```

A `sleepUsable` move — **Sleep Talk or Snore** — emits `|cant|` *and then still
acts*, banking the turn in `skippedTime`; switching out and back **refunds** it.
So a Rest + Sleep Talk mon that pivots can emit **up to four** `|cant|` lines for
a two-turn sleep.

**The public cant count therefore does not determine the clock for those mons.**
The conversion was sound; its *input* was not, for that population. Seed
1500013's Dunsparce is exactly such a set — 3 of its 5 randbats variants carry
both Rest and Sleep Talk, and **70 of 1,682 pool variants (4.2 %)** do.

## Q.3 The fix, and its honest trade

PR #927 (in review): **retire the Rest entry once a `sleepUsable` move is
selected**, so those mons **decline** — fail closed — rather than build a wrong
clock.

The trade, stated plainly: the exact-gating population **shrinks** by the
Rest + Sleep Talk share. §P.2 recorded +4,097 boundaries moving onto exact
gating as this cycle's headline; some of those go back to support/declined. That
is the right direction — a declined boundary is visible, a wrongly-built one is
not — but it is a cost, not a free win. Modelling the refund properly (tracking
`skippedTime` from the public stream) is a possible follow-up that would recover
the population.

## Q.4 Cycle-four expectations, adjusted

| Population | Prediction |
| --- | --- |
| plain-Rest rows | **clear** under the already-correct `3 - k` |
| Rest + Sleep Talk rows | move to **declined / support**, not exact — measured coverage may dip |
| net divergence | should fall, but coverage is the number to watch, not just the count |

## Q.5 The error chain, recorded in full

1. My probe showed **what**: `rest_turns=1` wakes this turn, 2 and 3 do not.
2. I narrated **why** — "off-by-one conversion" — without checking it against
   the simulator. The probe result was true; the explanation was invented.
3. That narration was turned into a prescription (`4 - k`) and dispatched as an
   instruction.
4. The build lane refused it and proved the correct behaviour at the sim.

Steps 2 and 3 are both part of the failure: a wrong cause was manufactured, then
promoted to an instruction without an independent check anywhere in between. The
lesson is the one this ledger already carries three times over — **the probe
shows WHAT; the WHY needs its own evidence** — and it now has a fourth entry, in
which the person who wrote the rule broke it.

Worth naming precisely: a refusal-with-evidence from the implementing lane is the
control that caught this, and it is the same control that caught §J.3. Neither
was caught by review of the reasoning; both were caught by someone trying to
build the thing and finding it did not hold.

---

# Appendix R — Cycle four (27 patches): the fourth hold

Run: 300 games, `--matcher strict`, seeds 1,500,000–1,500,299, `build_check =
gated`, `acceptance_eligible = true`, fingerprint `32a9b325db5f2655`. Both wheels
(`poke_engine` and `pokezero_search`) were rebuilt from a wiped vendor tree into
the measuring interpreter's own prefix before the run, per the #930 reviewer's
stale-wheel warning; the #930 behaviour was independently probed in the built
wheel (fast side's whole residual block resolves before the slow side's) rather
than assumed from the patch count.

## R.1 ACCEPTANCE VERDICT: NOT MET — run not started (fourth hold)

193 rows remain outside the adjudicated limit classes. The acceptance criterion is
*0 divergent transitions over 10,000 fresh-seed games*; an 8x1250 run from seed
2,000,000 would have produced a failing artifact, and the standing rule is that
the acceptance artifact must be a pass. The run was not started. **Seed block
2,000,000+ remains entirely unconsumed** — the invariant of §J.7 still holds after
four cycles.

## R.2 Headline numbers, against cycle three

| metric | c3 (26 patches) | c4 (27 patches) |
|---|---|---|
| boundaries measured | 23,335 | 23,335 |
| coverage (measured / full rounds) | 0.978 | **0.978** |
| transitions diverged | 306 | **240** |
| outside limit classes | 259 | **193** |
| in limit classes | 47 | 47 |
| harness errors | 0 | 0 |
| `unclassified` | 0 | 0 |
| gating exact / support | 21,258 / 2,077 | 20,478 / 2,857 |
| games/hour (single process) | 1,474 | 1,508 |

Outside-limit residue fell 25.5% (259 -> 193), the first real drop of the program.

## R.3 Zero regressions — established by row-level diff, not by totals

c3 and c4 ran identical seeds over identical boundaries, so rows are comparable by
`(seed, step)` key. The full transition matrix:

| rows | c3 class | c4 class |
|---|---|---|
| 33 | `roll_scaled_component` | matched |
| 22 | `component_extra_in_engine:itemleftovers` | matched |
| 4 | `component_missing_in_engine:leechseed` | matched |
| 2 | `component_missing_in_engine:heal` | matched |
| 2 | `component_missing_in_engine:itemleftovers` | matched |
| 1 | `component_missing_in_engine:psn` | matched |
| 1 | `component_extra_in_engine:itemleftovers,psn` | matched |
| 1 | `component_missing_in_engine:itemleftovers,movewish,sandstorm` | matched |
| 1 | `roll_scaled_component` | `component_missing_in_engine:heal` |
| 1 | `component_mismatch:sandstorm|brn,itemleftovers` | `component_mismatch:sandstorm|brn` |
| 1 | `component_extra_in_engine:itemleftovers` | `component_mismatch:heal|leechseed` |

**No row moved from matched to diverged.** Both apparently-new classes are
re-labelled survivors of rows that already diverged in c3 — one component of a
two-component miss was fixed, exposing the remainder. There are zero new
divergent boundaries.

## R.4 Prediction scorecard

Registered before measurement (`c4_predictions.json`).

| id | prediction | outcome |
|---|---|---|
| P1 | `extra:itemleftovers` family extinct | **refuted** — 33 -> 9, not 0 |
| P2a | 15.4% slice reduction | **exceeded** — actual 25.5% |
| P2b | reduction concentrated in that family | **refuted** — family gave 24 of 66; the largest single mover was `roll_scaled_component` (-34) |
| P2c | zero new classes | **refuted literally, upheld in substance** — 2 new class labels, 0 new divergent rows (§R.3) |
| P3 | plain-Rest rows clear under `3 - k` | **upheld** |
| P4 | Rest+Sleep-Talk rows move to *declined*; coverage drops | **refuted, favourably** — they moved to *support-based gating*, not declined: 780 boundaries shifted exact -> support and coverage held at 0.978. Retiring the false-exact Rest clock cost **zero** coverage |
| P5 | ordering corrections move rows in unpredicted classes | **upheld** — attributed by the §R.3 row diff, not by narrative |

Four of seven predictions were wrong. Recording them before the run is what makes
that legible.

## R.5 Attribution caveat: cycle four moved TWO variables

Cycle four is not a clean read on #930. Between the c3 and c4 measurements, main
gained both the engine patch (`dffcdda`) **and** three `src/` world-construction
commits on the Rest/Sleep-Talk clock (`46204b9`, `5e7cfa4`, `4cca974`). The
exact -> support gating shift of 780 boundaries is attributable to the latter, not
to residual ordering — a residual-order patch cannot change how a hidden counter
is gated. The 66 fixed rows are therefore a joint effect and are not claimed for
#930 alone.

## R.6 Mechanism decomposition of the 193 survivors

| rows | share | signature |
|---|---|---|
| 120 | 62.2% | not yet attributed |
| 41 | 21.2% | **Pain Split `-sethp` — harness bug (§R.7)** |
| 21 | 10.9% | straddles the lethal threshold (engine branch faints, Showdown's roll does not) |
| 11 | 5.7% | both sides capped lethal |

Of the 120 unattributed, 14 are the Rest full-HP engine bug of §R.8.

## R.7 CONFIRMED HARNESS BUG: the observation side is blind to `|-sethp|`

`damage_components` accepts only `-damage` and `-heal`
(`scripts/engine_transition_differential.py:394`). Showdown expresses Pain Split
as `|-sethp|...|[from] move: Pain Split`, so the HP change is dropped from the
observation entirely and its effect is absorbed into the next attributed delta.

Seed 1500008 step 101, Dusclops (132/209) Pain Split vs Wigglytuff (125/407):

```
|-sethp|p2a: Wigglytuff|128/407|[from] move: Pain Split|[silent]
|-sethp|p1a: Dusclops|128/209|[from] move: Pain Split
|-heal|p2a: Wigglytuff|153/407|[from] item: Leftovers
|-heal|p1a: Dusclops|141/209|[from] item: Leftovers
```

The true Leftovers heals are +13 (128->141) and +25 (128->153). The harness
reported +9 and +28 — the deltas from the *pre-state* HP, with Pain Split's
-4/+3 folded in. The engine emitted exactly +13 and +25. **The engine is correct
and the instrument is wrong.** The signature is unmistakable in the residue: the
class carries impossible values such as `('itemleftovers', -73)` — a negative
Leftovers heal.

All 86 `-sethp` lines across the population are Pain Split; no other gen3 move in
the pool reaches this tag. 43 of the 240 divergent rows contain one, 41 of them
outside the limit classes. This is harness-side and is mine to fix; it requires no
engine change.

## R.8 CONFIRMED ENGINE BUG: gen3 Rest succeeds at full HP

Showdown's `rest` resolves through `Dex.mod('gen3')` with an `onTry` callback that
fails the move on either of two conditions:

```js
if (source.status === 'slp' || source.hasAbility('comatose')) return false;
if (source.hp === source.maxhp) { ... }
```

The engine implements only the first:

```rust
// third_party/poke-engine-src/src/gen3/choice_effects.rs:782
Choices::REST => {
    let active_index = attacking_side.active_index;
    let active_pkmn = attacking_side.get_active();
    if active_pkmn.status != PokemonStatus::SLEEP {
```

There is no `hp == maxhp` guard, so a full-HP Rest puts the user to sleep instead
of failing. Seed 1500004 step 86: Xatu at 246/246 uses Rest. Showdown emits
`|-fail|p2a: Xatu|heal`, leaving Xatu statusless, so the incoming Toxic lands and
poison ticks. The engine emits `ChangeStatus SideTwo-P3: NONE -> SLEEP` +
`SetRestTurns 0 -> 3`, which blocks the Toxic entirely — a *status-level*
divergence, not a damage-roll one, and one that changes the legal action set on
every subsequent turn.

14 of the 240 rows are this bug, and **14 of 14 are the full-HP case with zero
asleep-user cases** — the existing guard cannot catch any of them. All 14 sit in
`component_missing_in_engine:psn`, where the missing poison tick is the visible
symptom of the suppressed status.

This contradicts the working assumption that #930 was the last engine change of
the program.

## R.9 Sample-before-reclassify, applied again

A single replay of `component_missing_in_engine:itemleftovers` (26 rows) showed
roll-divergent lethality, and the tempting move was to widen the limit class. The
required sample of 14 split 3 Pain Split / 8 plain / 3 observed-faint, and the
programmatic threshold test over the full class split it 4 / 6 / 11 / 5 across
four signatures. **No widening drafted.** The one-row narrative would have
mis-adjudicated at least 11 rows into a limit class they do not belong to.

Note also that the naive test was itself wrong in an instructive way: keying
"roll-divergent lethality" off a faint *in the observation* misses the common
case, where the engine's branch faints the target and Showdown's roll leaves it
alive (seed 1500099 step 5: engine crit 290 = exactly lethal, Showdown crit 287,
target survives at 3 HP and takes its Leftovers tick). The threshold is straddled
from either side.

## R.10 Recommended fix order

1. **Pain Split `-sethp`** (harness, mine) — up to 41 of 193 rows, ~21%. Teach
   `damage_components` the tag and attribute it to `movepainsplit`; confirm the
   mapper labels the engine's paired Damage/Heal identically, or the rows will
   re-appear as an attribution mismatch instead.
2. **gen3 Rest full-HP failure** (engine, build lane) — 14 rows, and the only
   *status-level* divergence currently known. Highest severity per row: it
   silently changes the action set for the rest of the battle.
3. **Re-measure**, then attribute the remaining ~106.

Steps 1 and 2 are independent and can run in parallel.

## R.11 Artifacts

- report `reports/c4.json`, checkpoint `reports/c4.jsonl`, log `reports/c4.log`
- pre-registered predictions `reports/c4_predictions.json`
- triage `reports/c4_tri_roll.json` (116/116 rows, population guard satisfied)

---

# Appendix S — The `-sethp` fix, and the engine bug it was hiding

Fix for §R.7, plus the finding that came out from under it.

## S.1 Sim-side facts, established before touching code

- **Pain Split is the only `-sethp` emitter in the pool.** The whole
  Showdown tree contains five `this.add('-sethp', ...)` sites: two in Pain
  Split's `onHit`, and three for Flip Turn inside the `chatbats` joke mod, which
  gen3 randbats cannot reach. There is no gen3/gen4/gen5 override of Pain Split,
  so gen3 inherits the base implementation.
- **It does NOT carry both slots' values on one line.** It emits *two* lines,
  each with only its own slot's `getHealth`: the target's first and `[silent]`,
  then the user's, visible. Both must be consumed — the silent one still moves
  HP.
- **Reachability**: 4 of 220 gen3 randbats species (dusclops, misdreavus,
  swalot, weezing). Reachable, therefore encoded.

## S.2 The fix, both sides of the instrument

Observation side (`damage_components`): admit `-sethp`. Its `[from]` payload
normalizes to `movepainsplit`, which is *not* in `_ROLL_SCALED_SOURCES`, so the
component is compared exactly — correct, because Pain Split is deterministic
(`floor((targetHP + userHP) / 2)`). An untagged `-sethp` is given the source
`sethp` rather than `""`, so a missing tag can never silently fall into the
roll-scaled bucket.

Mapper side (`events.rs`): Pain Split was in the "genuine self-costs" list and
rendered bare, under a comment asserting *"the real protocol renders them bare"*.
The sim says otherwise. Both halves now render as `-sethp` with an identical
`[from] move: Pain Split`, target `[silent]`.

The pairing is the load-bearing part and is pinned by its own test: components
are compared by normalized source, so had the two halves been tagged differently
— or one left bare — the rows would have returned as attribution mismatches
rather than matching.

## S.3 The mapper bug was also a live Track B defect

`fold.rs:1683` keys its Pain Split `self_hp_cost` branch on exactly `-sethp`
plus a `[from]` payload whose `side_condition_identifier` is `painsplit`. The
mapper's bare `-damage` matched neither, so **that branch never fired on the
engine-as-environment path** and Pain Split's self-cost was silently uncharged —
while the same fold, fed real Showdown logs, charged it. The encoder disagreed
with itself depending on which side supplied the protocol. This was not visible
from the differential at all; it fell out of asking what else consumes the lines
before changing them.

## S.4 Result: 30 of 43 rows cleared, and 12 turned out to be an engine bug

| | c4 (pre-fix) | c4b (post-fix) |
|---|---|---|
| diverged | 240 | 212 |
| outside limits | 193 | **167** |
| Pain Split rows diverging | 43 | 13 |
| coverage | 0.978 | 0.978 |
| harness errors | 0 | 0 |

Two rows that previously *matched* now diverge. They are not regressions: they
are false passes the broken instrument was producing, and both are §S.5.

## S.5 CONFIRMED ENGINE BUG: gen3 Pain Split does not clamp to `maxhp`

```rust
// third_party/poke-engine-src/src/gen3/choice_effects.rs:877
let target_hp = (attacking_side.get_active_immutable().hp
    + defending_side.get_active_immutable().hp) / 2;
...
attacking_side.get_active().hp = target_hp;
defending_side.get_active().hp = target_hp;
```

The average is assigned to both actives with no clamp against each mon's own
`maxhp`. Showdown assigns through `Pokemon#sethp` (`sim/pokemon.ts:1656`), which
clamps. Whenever the average exceeds a mon's maximum — routinely, when a full-HP
mon splits with a higher-`maxhp` opponent — the engine leaves `hp > maxhp`.

Seed 1500037 step 7: Weezing 238/238 splits with Groudon 252/252. Average 245.
Showdown clamps Weezing to 238/238 (`-sethp` with no change) and moves Groudon
to 245/252. The engine gives Weezing +7 *above its maximum*.

**12 of the 15 surviving `movepainsplit` rows are this bug** (measured by
checking, for every row, whether Showdown's two post-split HPs differ with one
sitting exactly at its max). The remaining 3 are a separate mechanism and are
recorded as unattributed rather than folded in.

This is worse than a protocol disagreement: `hp > maxhp` is corrupt state that
every later damage, heal and faint check reads.

It was *invisible* until the instrument was repaired — the `-sethp` blindness was
swallowing the entire component. Fixing a measurement bug exposed an engine bug
underneath it, which is an argument for repairing instruments even when the
residue they produce is comfortably attributable to something else.

## S.6 Hand-off

Engine lane, alongside the §R.8 Rest patch: clamp both assignments to the
receiving mon's `maxhp`, mirroring `sethp`. Expected to clear 12 of the 15
remaining `movepainsplit` rows; the other 3 need their own attribution.

---

# Appendix T — Cycle five (29 patches): the fifth hold

Run: 300 games, `--matcher strict`, seeds 1,500,000–1,500,299. Both wheels
rebuilt from a wiped vendor tree into the measuring interpreter's own prefix.
Build gate recorded verbatim, per the sibling agent's stale-wheel incident:

```
engine build is current (29 patches, cafd1ce9cce2b6b0)      [exit 0]
```

All 29 patches applied with **fuzz 0**. Both new patches were probed
behaviourally in the built wheel rather than inferred from the count:

- patch 28 (Rest full-HP): full-HP Rest emits **no instructions**; a damaged
  Rest still sleeps and heals.
- patch 29 (Pain Split clamp): Weezing 238/238 vs Groudon 252/252 emits
  `Damage SideOne: 0` / `Damage SideTwo: 7` — the clamp, matching `sethp`.

## T.1 ACCEPTANCE VERDICT: NOT MET — run not started (fifth hold)

141 rows remain outside the adjudicated limit classes. The acceptance artifact
must be a pass, so the 8x1250 sweep was not started. Seed block 2,000,000+
remains unconsumed; verified mechanically this cycle rather than asserted — the
highest seed consumed by *any* dev or measurement run in the workspace is
**1,500,297**.

## T.2 Numbers

| metric | c4 (27p) | c4b (+harness fix) | c5 (29p) |
|---|---|---|---|
| boundaries measured | 23,335 | 23,335 | 23,335 |
| diverged | 240 | 212 | **186** |
| **outside limits** | 193 | 167 | **141** |
| coverage | 0.978 | 0.978 | 0.978 |
| harness errors | 0 | 0 | 0 |
| gating exact / support | 20,478 / 2,857 | same | same |

## T.3 Both fixes landed exactly as predicted — to the row

| prediction | predicted | measured |
|---|---|---|
| §R.8 Rest full-HP clears its rows | 14 | `missing_in_engine:psn` 23 -> 9 = **14** |
| §S.5 Pain Split clamp clears 12 of 15 | 12 | `movepainsplit` family 15 -> 3 = **12** |

Zero new classes, zero regressions, coverage flat. Two engine patches predicted
in advance at the row level and hitting exactly is the strongest evidence yet
that the instrument's attributions are real rather than curve-fitted to the
residue.

## T.4 What the 141 are

`roll_scaled_component` (82) triaged over its full population:

| rows | bucket |
|---|---|
| 46 | `damage_calc:*` (ratio not a clean roll or type/STAB multiplier) |
| 32 | `structural_component_count` |
| 2 | `legitimate_roll_in_legal_set` |
| 2 | `no_usable_branch` |

`structural_component_count` fell 66 -> 32 with Pain Split gone. The dominant
remaining shape (13 of 32) is *observation has a damage component the engine
does not*, and `protect` appears in 10 of the 32.

The rest of the 141: `missing_in_engine:itemleftovers` 22,
`missing_in_engine:psn` 9, `extra_in_engine:itemleftovers` 6,
`magnitude:movepainsplit` 3, `missing:itemleftovers,sandstorm` 3, and 16 in
singleton or pair classes.

## T.5 CANDIDATE, not a finding: Protect

Seed 1500010 step 39. Spinda (282/282) uses Protect; Showdown emits
`|-fail|p1a: Spinda` and Porygon2's Return lands for 102. The engine produces a
**single branch at 100%** in which Protect succeeds and blocks the move
entirely.

That is the whole of what is established. The obvious story — gen3's
consecutive-use success decay — is *not* recorded here as the cause, because it
has not been tested: the attempt to probe it failed (`SideConditions.protect` is
not writable through the Python binding), and no override was found in the gen3
mod chain to read the rule from. Per the standing rule this ledger has now
broken four times, **the probe shows WHAT; the WHY needs its own evidence**.

Hand-off to the engine lane with the mechanism explicitly open: does gen3
Protect ever produce a failure branch, and is its success probability
conditioned on consecutive use? A binding that exposes the counter, or a
Rust-side unit probe, would settle it.

### T.5.1 RESOLVED (cycle six) — and the row is now credited to #942 by measurement

Both open questions are answered, and the row is closed. The Rust-side probe
T.5 asked for is what settled the mechanism (160 seeds: 2nd attempt 0.482, 3rd
0.196 — the ×2 ladder, §"fifth direction"), and the retained-repro replay is
what closed the row.

**Cause, from the replay rather than from the story.** The pre-state carried
`side_conditions = {"p1": [], "p2": []}`: `k = 0`, so the engine priced Protect
at `0.5**0 = 1.0` and emitted a *single* branch. The observed failure was not
improbable in the engine's model — it was **absent from the branch set**. The
mechanism T.5 declined to name was right, but the reason the row diverged was
the world never seeding the counter, not the decay rule being wrong.

**Before/after, same pinned engine build** (29 patches, `7909290e14e065cd`;
Python at `a3c98a6` before, `84a6712` after — #942 is pure Python, so the engine
binary is byte-identical across the two runs and cannot confound them):

| run | boundaries | measured | matched | diverged |
| --- | --- | --- | --- | --- |
| seed 1500010, pre-#942 | 65 | 64 | 63 | **1** |
| seed 1500010, with #942 | 65 | 64 | 64 | **0** |

Identical boundary counts confirm it is the same game, so the row closed rather
than the trajectory shifting out from under the measurement.

**Whole-census effect, 40 games (seeds 1500000-1500039), same two builds:**
measured 3,580 in both; diverged **18 → 17**; the *only* class that moves is
`roll_scaled_component` 9 → 8. **One row closed, zero regressions** — #942
neither fixed nor broke anything else in that population.

The earlier ~10-protect-row prediction is **not** supported at this seed range:
one protect row existed and one closed. The prediction was extrapolated from a
different census; it should not be carried forward as a pending credit.

**Seam linkage (closing the #942 review carry).** `test_engine_world_stall_counter.py`
asserts the `0.5 ** k` pricing in two linked halves because the shared fixture's
dex has no Protect. Both *one-sided* renames are caught by the halves
themselves; the residual gap was **joint drift** — engine field renamed, half 2
updated to match, half 1 left stale — which no unit assertion inside the file
can see. That gap is closed by this end-to-end replay, and only by it: the
replay drives the real field through parser → `engine_world` → engine branch
probabilities, so joint drift would surface as the row failing to close. Keeping
this row in the retained-repro set is therefore load-bearing for the seam, not
just a historical record.

**Process note:** this replay is the first cash-out of the retention proposal in
§"Process gap" — the row T.5 could not re-examine was recoverable this cycle
only because `--checkpoint` retained its `engine_state`.

## T.6 Carry: shard vintage is now a lineage boundary (#936 review, Finding 1)

The §S.2 mapper fix changes the *encoder's* output on the engine-as-environment
path, not only the differential's verdicts. `fold.rs`'s Pain Split
`self_hp_cost` branch was unreachable from engine protocol before the fix and is
reachable after, so `self_hp_cost` turns non-zero on Pain Split turns and some
Pain Split windows flip constant -> non-constant.

**Rule: engine-env shards carry a vintage, and vintages must not be mixed.**
Any training data generated through the engine-as-environment path *before* the
§S.2 fix is a lineage discontinuity for the four Pain-Split-carrying species —
weezing, misdreavus, swalot, dusclops (4 of 220, so it is a thin but real slice
of the pool). Do not pool pre-fix and post-fix engine-env shards in any future
run; regenerate, or partition and label.

This is the same failure shape as the parity-lineage bug: a silent input change
that leaves both halves individually plausible and only shows up as a trunk that
never learns.

## T.7 Carry: a documented NON-residual (#938)

Patch 29 deliberately does **not** model `sethp`'s fainted no-op — Showdown's
`sethp` returns without effect on a fainted target. Verified unreachable: Pain
Split requires a live target, so the branch cannot be entered in gen3 randbats.

Recorded here so a future reader diffing the engine against `sim/pokemon.ts`
finds the omission already adjudicated rather than re-opening it. This is the
`limit:`-class pattern applied to an engine omission instead of a residue class:
**an unreachable divergence is not a divergence**, provided the reachability
claim is written down where the next person will look.

## T.8 Artifacts

- report `reports/c5.json`, checkpoint `reports/c5.jsonl`, log `reports/c5.log`
- triage `reports/c5_tri.json` (82/82 rows, population guard satisfied)
- prior cycle for comparison: `reports/c4.json`, `reports/c4b.json`

---

# Appendix U — Cycle six: adjudicating `itemleftovers` and `psn`

Branch `scott/residue-triage-leftovers-psn`. Engine 29 patches, fingerprint
`7909290e14e065cd`, `--check` current; Python at `84a6712` (post-#942/#943).
Every verdict below comes from replaying a retained `engine_state`, not from a
class name. Reports are committed under `reports/` — see U.5.

## U.1 The assignment's counts were pre-#908/#930 and are stale

Re-running the **C.9 census verbatim** — 60 games, seeds 1350000-1350059, strict
matcher — on the current build gives the *identical denominator*, so the
comparison is like-for-like rather than a re-scoped population:

| | C.9 (19-patch) | now (29-patch, #908/#930/#942 in) |
| --- | --- | --- |
| measured | 5,310 / 5,438 | **5,310 / 5,438** (identical) |
| diverged | 160 = 3.01 % | **50 = 0.94 %** |
| `roll_scaled_component` | 86 | 21 |
| `component_extra_in_engine:itemleftovers` | 9 | **0** |
| `component_mismatch:sandstorm\|psn` | 9 | **0** |
| `component_missing_in_engine:psn` | 9 | **2** |
| `component_missing_in_engine:itemleftovers` | 11 | 9 |

Two of the three assigned classes are already **closed or nearly closed** by
merged work — chiefly #908's positional attributor, which lives in the Rust
mapper (`rust/pokezero-search/src/events.rs`), not in the differential script.
The brief's "22 missing + 6 extra / 9 psn" figures describe a pre-#908
population and should not be carried forward as open work.

## U.2 The headline: 8 of 12 rows are not residual bugs at all

Pooling both populations (the 1350000 census and the 1500000 40-game sweep)
gives 12 rows across `itemleftovers` / `psn` / `brn`. Classifying each by what
its **highest-probability** branch actually disagrees on:

| top-branch failure mode | rows |
| --- | --- |
| roll-scaled **move damage** | **8** |
| the named residual component | 4 |

In all 8, the named residual is **present and numerically identical** in the
engine's majority branch. The class label is produced by *minority* branches
where the residual is correctly absent — and those branches are exactly the ones
a correct engine must have:

| row | label | majority branch | why the label appears |
| --- | --- | --- | --- |
| s1500034 st6 | `missing:brn` | `brn=-29` **present** | 4.69 %+ lethal branches — mon dies to the move, so no burn tick |
| s1350007 st76 | `missing:psn` | `psn=-17` **present** | 15 % of branches = Toxic **missed** (85 % accuracy) |
| s1500014 st69 | `missing:itemleftovers` | `+18/+14` **present** | 6.25 % = the **crit** branch, lethal in engine |

The real disagreement in these rows is move damage — s1350007 st76 is observed
−66 vs engine −106, s1500014 st69 is −214 vs −116 — which is the
`roll_scaled_component` / damage_calc lane, **explicitly not this lane's scope**
this cycle. C.9 already flagged that family as substantive rather than roll
noise (seed 1350001 step 49, −116 vs −170); these rows are further instances of
it wearing a residual's name.

**Method note.** This is the fourth cycle in which a class name turned out to be
a symptom rather than a mechanism. Had these been "fixed" as residual bugs,
the fix would have targeted components the engine already gets right.

## U.3 The 4 genuine rows, and they are three different things

### U.3.1 NEW SIGNATURE (open): Explosion + a can't-move status

Two rows, one shape, and it is not a residual bug either:

| row | Showdown | engine |
| --- | --- | --- |
| s1350013 st38 | `\|cant\|p2a: Forretress\|frz` — move blocked, upkeep runs, Leftovers `+15` | `Damage SideTwo: 67` + `ToggleSideTwoForceSwitch` — Explosion **resolves**, self-KO |
| s1350022 st34 | `\|cant\|p2a: Nosepass\|slp` — move blocked, upkeep runs, Leftovers both sides | `Damage SideTwo: 215` + `ToggleSideTwoForceSwitch` — Explosion **resolves** |

No branch in either sweep reproduces the observed transition, and **none emits
the Leftovers tick**, which is what produces the `missing:itemleftovers` label.
Both rows pair **Explosion** with a fully-incapacitating status.

Per the standing rule — *the probe shows WHAT, the WHY needs its own evidence* —
the mechanism is recorded as **open**. What is established is the WHAT above;
what is **not** established is whether the cause is the status gate, the
forced-switch toggle suppressing `add_end_of_turn_instructions`, or the
hidden-counter sweep failing to include a stays-incapacitated candidate. The
narrow signature (Explosion + frz/slp) makes this cheaply reproducible for
whoever takes it.

### U.3.2 Boundary artifact, not an engine bug: s1500005 st67

Showdown's slice **ends at the faint with no `|upkeep`** — Crunch crit-KOs
Electabuzz, Static fires, protocol stops. The residual block has not been
deferred *incorrectly*; it has not run **yet**, because a replacement is pending.
The engine, given the same joint action, completes the turn and emits
`Heal SideOne: 17` (+ toxic `-15`). The engine is right about the game; the
*measurement boundary* is what disagrees. Same family as the #876 deferral work.

### U.3.3 Mapper attribution, #908/I.2 lineage: s1500017 st55

90 % branch: observed `heal +31` vs engine `leechseed +31` — **same magnitude,
different label**. This is the I.2 shape (Showdown emits the seeder's drain bare
/ `[silent]`, tagging only the victim's damage) surviving in one direction.
Cited, not re-derived.

## U.4 Adjudication summary

| verdict | rows | lane |
| --- | --- | --- |
| symptom of move-damage divergence; residual is correct | 8 | damage_calc — **not this lane** |
| Explosion + can't-move status | 2 | **OPEN, new signature** |
| measurement-boundary truncation at a faint | 1 | harness — #876 family |
| mapper attribution `heal` vs `leechseed` | 1 | harness — #908/I.2 family |

**No engine fix is proposed by this cycle for `itemleftovers` or `psn`, and that
is the finding.** Both classes are adjudicated: the bulk are mislabelled
damage-calc rows, and the genuine remainder belongs to a different mechanism
(U.3.1) that deserves its own investigation rather than a residual patch.

## U.5 Artifacts — the retention proposal, discharged

The §"Process gap" entry proposed persisting repro states because a cycle-five
row could not be re-examined. That proposal is **executed here**, not just
restated: these reports carry the `engine_state` for every divergent row and are
replayable with `scripts/replay_residue.py --report <file>`.

- `reports/c6_census1350.json` — 60 games, seeds 1350000-1350059 (U.1, U.2)
- `reports/c6_1500_pre942.json` — 40 games, seeds 1500000-1500039, **pre-#942**
- `reports/c6_1500010_post942.json` — seed 1500010, **post-#942** (T.5.1 pair)

The last two are the before/after that closed T.5, and are the first artifacts
in this repo that make a specific row re-examinable after main advances.

**Caveat that travels with them:** the reports pin the engine fingerprint but a
fingerprint alone is not a build identity once main moves — the repo SHA is
recorded above for exactly that reason.

---

# Appendix V — Cycle seven (30 patches) and the damage_calc triage brief

Commit `dfb4d10285c003e3f324ea1e1dcb06b296ff8fd3`, fingerprint
`814b2bd28d3983813b972ba3fd0af7fcc46871085fdea6f0e654c767d076b577`, 30 patches,
fuzz 0, 28 crate suites green. Seeds 1,500,000–1,500,299.

## U.1 Build verified BEHAVIOURALLY, per this ledger's own carry

`--check` passed, but §T's carry is that it *cannot* detect a stale wheel after a
re-vendor restamp — the stamp is computed from source content, so it matches
whether or not the wheel was rebuilt. Both probes were therefore run against the
built wheel and are the actual gate:

| probe | expected | measured |
|---|---|---|
| patch 29 regression: Pain Split clamp | `SideOne: 0`, `SideTwo: 7` | matched |
| patch 30 / #942: Protect stall ladder | k=1 prices at 50% | k=0 -> single 100% branch; **k=1 -> 50.0%**; k=2 -> 25.0%; k=3 -> 12.5% |

The ladder is exactly `0.5^k`, and k=0 correctly produces no failure branch.

## U.2 ACCEPTANCE: still not authorized to fire (sixth hold)

Outside-limit residue 141 -> **127**. The gate is *non-damage_calc residue fully
attributed*; it is not. Of the 127, 46 are `damage_calc` and **81 are not** —
22 structural, 22 `missing:itemleftovers`, 6 `missing:psn`, and 31 across small
classes. The sweep was not started. Seed block 2,000,000+ remains unconsumed.

## U.3 Expectations, scored honestly

- **Protect credit was 14 rows GROSS, not ~1.** The itemisation credits -16 rows
  to Protect; two rows moved the OTHER way, so the NET drop is -14. The figure
  quoted here and in #949 is the gross credit, and the two clauses must be read
  together — an itemisation that reports only its gross side will over-credit
  whatever it is itemising. The measured drop is still larger than the retired
  10-row prediction *and* than the ~1-row revision. `roll_scaled` -10,
  `missing:psn` -3, `limit:roll_divergent_lethality` -2, `extra:itemleftovers,sandstorm` -1.
- **The predicted leftovers/psn RELABEL did not happen.**
  `missing_in_engine:itemleftovers` is unchanged at 22. #946 adjudicated those 8
  rows as really being `damage_calc`, but the classifier was not changed, so the
  labels did not move. The adjudication currently lives in the ledger, not in the
  instrument. Either the classifier should be taught the distinction or the
  ledger should say plainly that this class is known-mislabelled — leaving it
  implicit is how a wrong label survives four more cycles.

## U.4 DAMAGE_CALC TRIAGE BRIEF

46 `damage_calc` rows yield 62 damaged-slot findings.

**The partition that decides reality.** gen3 rolls 85–100% in 16 steps, so two
honest implementations routinely disagree on one roll. A finding is real only
when Showdown's value is reachable by **no** engine roll:

| findings | verdict |
|---|---|
| 10 | reachable — a roll disagreement the `unexplained_ratio_*` label over-reported |
| 52 | unreachable — real |

The `unexplained_ratio_*` labels are **not** this test: their window is
0.92–1.09, narrower than gen3's true roll spread, so they over-report. Cluster
on legal-set membership; use the ratio only to *describe* a cluster once real.

**A caveat about my own tool, before anyone uses its numbers.** The brief pairs
observed against engine components by sorted magnitude, which mispairs when a
slot has more than one (a 2-point residual gets paired against a 136-point move
hit, yielding a nonsense ratio of 0.015). 15 of the 54 multi-component findings
are excluded for this reason. **39 unambiguous single-component real findings**
remain, and only those are clustered below.

### Clusters

| n | ratio band | reading |
|---|---|---|
| 11 | 0.85–0.92 | ~~best lead~~ **RETRACTED — instrument artifact, see W.3** |
| 10 | 0.50–0.70 | a ~1/2 or ~2/3 factor — candidate: type-effectiveness or resist rounding |
| 7 | < 0.50 | scattered extremes |
| 10 | > 1.00 | scattered extremes |
| 1 | 0.70–0.85 | — |

### Factors, ruled in and out

- **Screens: ruled OUT, cleanly — 0 of 39.** Consistent with the standing note
  that Reflect/Light Screen are unreachable in the gen3 randbats pool.
- Defender stat stages: present in only 4 of 39.
- Weather: present in 11 of 39 — worth a controlled look, not yet a cluster.
- **No dominant move**: the 39 spread across 20+ moves, which argues against a
  per-move data error and for a shared formula factor.
- **4 findings are Sleep-Talk-called** and must be separated before any fit: the
  move label is the *caller*, not the move that dealt the damage, and these
  travel the known unknown-callee union path.

### Recommended entry point for the fix lane — RETRACTED, see W.3

~~Start with the 0.90–0.92 cluster.~~ The band is this filter's own artifact and
is not a lead. The damage lane has been redirected to the 0.50–0.70 group. Take the tightest exemplars, compute
gen3 base damage by hand for both implementations, and find the factor. Do not
start from the extremes — they are a mixture, and at least some are pairing
artifacts of the multi-component slots excluded above.

## U.5 Retention

Committed under #946's process fix: `reports/c7_summary.json` (counters and class
census, **repros stripped** — 170 rows excluded) and
`reports/c7_damage_calc_brief.json` (39 adjudicated row extracts, protocols
stripped). No checkpoints or full-run dumps in tree.

---

# Appendix W — Two instrument fixes (classifier adjudication, source pairing)

Both are instrument changes, not engine changes. Neither alters the residue
count; V.1 was explicitly constrained so that it cannot.

## V.1 The #946 adjudication, made mechanical

`branch_misses` is in branch order, so `misses[0]` may be a MINORITY branch.
s1500014 st69 has three: the 6.25% branch reports a missing `itemleftovers`, and
the 75.00% + 18.75% branches report `observed=[('', -214)] engine=[('', -116)]`
— a damage disagreement of nearly 2x. The classifier read the 6.25% branch and
filed the row under Leftovers. It held that label for four cycles.

The rule now: when the first miss names only an adjudicable residual
(`itemleftovers`, `psn`, `brn`, `sandstorm`, `tox`) and the branch carrying the
largest probability mass complains only about a roll-scaled component, classify
from the majority. Deliberately narrow — reordering every row by probability
would re-classify rows nobody has adjudicated.

**Effect, stated in advance and then measured** (replayed over the c7 rows):

| | before | after |
|---|---|---|
| rows relabelled | — | **14** |
| `roll_scaled_component` | 72 | 86 |
| `component_missing_in_engine:itemleftovers` | 22 | 11 |
| `component_missing_in_engine:psn` | 6 | 4 |
| `component_missing_in_engine:brn` | 2 | 1 |
| **total diverged** | 170 | **170** |
| **outside limits** | 127 | **127** |
| limit classes | 43 | 43 |

14 of the 28 known-mislabels relabel; the other 14 do not meet the predicate and
keep their labels honestly rather than being swept along.

### The guard that matters more than the rule

The first implementation moved **15 additional rows into
`limit:roll_divergent_lethality`** and dropped outside-limits from 127 to 112.
That is an instrument change silently handing the acceptance gate a 15-row
credit. #946 adjudicated those rows as `damage_calc`, not as a comparison limit.

The override therefore refuses to fire when the majority miss carries
`capped_lethal`. **A relabel must never reduce the residue.** If those rows
belong in a limit class, that is a separate decision on its own evidence.

## V.2 Source pairing replaces sorted-magnitude pairing

The triage paired observed against engine components by sorting both sides by
bare magnitude and zipping. In any slot with more than one component that pairs
unlike things — a 2-point residual against a 136-point move hit, reported as a
ratio of 0.015 describing nothing. 15 findings were excluded from the cycle-seven
brief for this reason: dark data feeding nothing.

Pairing is now keyed on source, with magnitude used only as a tiebreak within a
source. Components with no counterpart are dropped rather than paired: an
unmatched count is a structural difference, already reported as
`structural_component_count`, and must not be manufactured into a ratio.

### Answer to the question asked: they join existing clusters

**9 previously-dark findings became usable. None forms a new cluster.**

| ratio band | old (single-component only) | new (all, source-paired) | of which previously dark |
|---|---|---|---|
| < 0.50 | 7 | 5 | 0 |
| 0.50–0.70 | 10 | 12 | 2 |
| 0.70–0.85 | 1 | 0 | 0 |
| 0.85–1.00 | 11 | 14 | 2 |
| > 1.00 | 10 | 14 | 5 |

Real findings go 52 -> 45, because unmatched components are no longer
manufactured into ratios. The nonsense extremes are gone: no 0.015, no 25.0.

The 0.85–1.00 band — the damage lane's recommended entry point — gains 2 and
still holds the tight 0.90–0.92 sub-cluster. **The brief is an additive update:
the 39-row structure the lane is working from is unchanged in character, and no
conclusion it has drawn is invalidated.**

A secondary benefit: source pairing surfaces residual-source findings that
magnitude pairing hid entirely (`drain`, `movewish_to_full`,
`itemleftovers_to_full` now appear beside the move finding on the same row).

## W.3 CORRECTION: the 0.90–0.92 "cluster" was my own filter's artifact

> **SUPERSEDED BY §Y — THIS SECTION'S VERDICT IS WRONG.** The band is 11 real
> findings and a genuine lead; the artifact story was a cardinality-not-membership
> error. Only the struck Shadow Ball anchor survives. Kept unedited below because
> the retraction and its reversal are both part of the trail.

§V.4 recommended the 0.90–0.92 band as the damage lane's entry point. **That
guidance was wrong and is withdrawn.** So is the closing line of §W.2 as
originally written ("keeps its tight 0.90–0.92 sub-cluster, so nothing they've
concluded is invalidated") — the sub-cluster is not a finding, and that sentence
should not be read as reassurance.

`_classify_ratio`'s low edge is `0.92`, which treats the engine's damage value as
the **mean** roll. If the engine instead reports top-of-range, the honest window
is `[0.85, 1.00]`-shaped and everything between 0.85 and 0.92 is *the filter's
own floor*, not a signal. Findings pile up against a threshold because the
threshold is there.

Measured on the source-paired set: **11 real findings sit in [0.85, 0.92)**.
Remove them and the 0.92–1.00 range holds **3** findings (0.925, 0.926, 0.958).
Three scattered points are not a cluster and were never a lead.

The **Shadow Ball 116/107 anchor is struck**: it appears in no committed extract
and I could not source it. It entered §V.4 as a reported figure and I repeated
it as if it were evidence — the same "narrate the WHY" failure this ledger has
now recorded five times, in the appendix that was supposed to hand the next lane
clean structure.

The damage lane is redirected to the **0.50–0.70 group** (12 findings).

## W.4 The partition is now machine-checkable

The reachability partition was prose-only, so a reader had to re-derive which
rows were artifact from the ratios — and the §V.4 mismatch survived exactly
because nobody could check it cheaply. Every finding in
`reports/c7_damage_calc_brief.json` now carries a **`reachable` boolean**
(legal-roll-set membership); 54 findings, **0 null**. Partition on that field,
not on the ratio.

## W.5 Window semantics: deferred deliberately, not guessed

The strict matcher already answers this correctly for roll-scaled components —
**legal-roll-set membership**, which needs no constant and cannot develop a floor
artifact. The brief should follow it, and the `reachable` field is that answer.

If ratio constants are kept anywhere, they must wait on the damage lane's
source-cited determination of the engine's damage-value semantics (max vs mean,
with line citation). **No constant is chosen here.** Guessing one is how the
0.92 floor manufactured an eleven-row cluster in the first place.

## W.6 Inventory: the known-mislabels the override does NOT move

Listed, deliberately **not** adjudicated. 30 rows carry the known-mislabel
classes; 14 relabel under W.1; **16 do not**:

| rows | majority-miss type |
|---|---|
| 11 | roll-scaled carrying `capped_lethal` — the override refuses by design (W.1's guard) |
| 5 | attributed-components: the majority branch **also** blames the residual |

| seed / step | source | majority | type |
|---|---|---|---|
| s1500012 st24 | itemleftovers | 79.1% | capped_lethal |
| s1500050 st33 | itemleftovers | 52.7% | capped_lethal |
| s1500054 st125 | itemleftovers | 33.3% | attributed-components |
| s1500074 st57 | itemleftovers | 70.3% | capped_lethal |
| s1500105 st111 | itemleftovers | 79.1% | capped_lethal |
| s1500112 st40 | itemleftovers | 31.2% | attributed-components |
| s1500168 st97 | psn | 79.7% | capped_lethal |
| s1500188 st33 | itemleftovers | 75.0% | capped_lethal |
| s1500219 st62 | itemleftovers | 93.8% | capped_lethal |
| s1500242 st56 | psn | 79.7% | capped_lethal |
| s1500242 st60 | itemleftovers | 39.5% | capped_lethal |
| s1500243 st79 | psn | 85.0% | attributed-components |
| s1500251 st56 | itemleftovers | 44.0% | capped_lethal |
| s1500255 st55 | itemleftovers | 93.8% | capped_lethal |
| s1500287 st76 | brn | 100.0% | attributed-components |
| s1500294 st110 | psn | 100.0% | attributed-components |

What the inventory decides for cycle eight: the **5 attributed-components rows**
are labelled correctly — their majority branch really does blame the residual, so
they are not mislabels at all and #946's set was over-broad by that much. The
**11 `capped_lethal` rows** are the real open question, and it is the one W.1's
guard deliberately declines to answer: they may be `damage_calc`, or genuinely
`limit:roll_divergent_lethality`, and deciding costs 11 rows off the residue
either way. That is a decision to take on its own evidence, not as a side effect
of a relabel.

Note the count correction: I reported "14 unrelabelled" in the #950 summary. The
classes hold **30** rows, 14 relabel and **16** do not.

---

# Appendix X — Evidence standard for the 11 `capped_lethal` rows

Design only. **Not to be executed until the damage lane reports** — their formula
findings may adjudicate several of these for free, and a row explained by a
confirmed arithmetic defect needs no separate branch walk.

## X.1 The question, stated so it can be answered wrongly

For each row, exactly one of:

- **genuinely `limit:roll_divergent_lethality`** — Showdown's observed outcome is
  reachable under *some* legal roll and ordering that the engine's branch set
  already prices. The two sims took different draws from the same distribution;
  no per-component comparison can align them, and the residue is a limit of the
  comparison.
- **`damage_calc`** — Showdown's outcome is reachable under *no* legal roll or
  ordering the engine prices. The engine's damage arithmetic differs.

The distinction is worth **11 rows off the residue either way**, which is why it
gets a standard rather than a judgement call.

## X.2 The evidence: per-row branch enumeration

Modelled on the #946 row-walk. For each row, produce a table over the engine's
**full** branch set — not the majority branch, not a sample:

| column | meaning |
|---|---|
| branch pct | the engine's own probability for this branch |
| damage instructions | every `Damage`/`Heal` with side and amount |
| roll ladder | all 16 legal rolls for each damaging move in the branch, crit **and** non-crit |
| ordering variants | any residual/speed-tie ordering the engine prices distinctly |
| post-state HP vector | both actives' HP after the branch, and the faint set |

The comparison target is the **observed post-state HP vector, the faint set, AND
the multiset of non-residual damage components**.

The original draft stopped at the HP vector and faint set, justified by the
component lists differing in length. **That justification holds only for the
residual and faint components** — the ones `capped_lethal` actually makes
incomparable. It does not extend to ordinary move damage, which is directly
comparable on both sides. Discarding it lets an engine branch dealing `100 + 20`
match a Showdown outcome of `110 + 10` because both land on the same post-state,
adjudicating a genuine `damage_calc` defect as a limit. Non-residual damage
components stay in the target; only residual/faint components are exempted.

## X.3 The decision rule — CORRECTED, the draft had a counting hole

**The hole.** The draft's test was an unbounded existential: *some* (branch, roll
assignment, ordering) reproducing a roughly two-integer target. The search space
is branches x 16 rolls per damaging move x crit/no-crit x orderings — on the
order of a thousand combinations. Against a target that small, a match is close
to guaranteed **by counting alone**, so a motivated walker could send all 11 rows
to `limit` without ever writing something false. A standard that cannot fail is
not a standard.

Three constraints close it.

### X.3.1 Report the mass, and gate the limit verdict on it

A reproducing assignment is not evidence on its own; its **probability under the
engine's own distribution** is. Every walk reports that mass, computed as the
branch percentage times the probability of the specific roll assignment (each
roll index is 1/16).

`limit` requires **mass >= 1%**. A 0.02% corner of the distribution is a
coincidence the search space handed you, not two sims drawing differently from
the same distribution.

The floor is deliberately above the two-damaging-move floor: one move at
`1/16 = 6.25%` clears it comfortably, two independent rolls at `(1/16)^2 ~ 0.39%`
do not. Rows in that gap are **not** thereby `damage_calc` — they are reachable —
so they take the fourth exit below rather than being forced either way.

### X.3.2 Roll consistency across the branch

One roll index per move, applied everywhere that move appears in the branch.
Per-component free choice is what makes the space combinatorial in the first
place, and it is not a thing the engine can do: a move rolls once.

### X.3.3 The four exits

| condition | verdict | residue effect |
|---|---|---|
| reachable (X.2 target, X.3.2 consistency) **and** mass >= 1% | `limit:roll_divergent_lethality` | -1 row |
| not reachable under any consistent assignment | `damage_calc`, carrying a quantified gap (closest achievable target and its distance) | none |
| reachable but mass < 1% | **`limit_not_established`** — row keeps its current label | none |
| branch set not recoverable / world unbuildable | `cannot_enumerate` — row keeps its current label | none |

Three of the four exits leave the residue untouched. That asymmetry is
intentional: only the verdict that *reduces* the residue has to clear a bar.

### X.3.4 When `cannot_enumerate` may be claimed

A licensing test, not a list of examples — examples let a walker reason by
resemblance. `cannot_enumerate` is available **only** when a named, checkable
precondition fails:

- `generate_instructions` raises, or returns no branch, on the recorded state; or
- no candidate `engine_state` deserializes; or
- the branch set is non-finite under X.3.2 (no move admits a determinate roll
  index); or
- the recorded row lacks a field the walk requires (no `engine_states`, no
  `pre_features`).

Each claim names which precondition failed and quotes the failure. "The walk was
inconclusive" is not a licence.

## X.4 Pre-registration and controls

Standing practice, applied here specifically:

1. **Predict the split before walking, and score it.** Record the expected
   limit / damage_calc / limit_not_established / cannot_enumerate counts in the
   run artifact *before* the first row, and report **predicted vs actual in the
   same artifact** at the end. A prediction that is recorded but never scored is
   decoration; the cycle-four scorecard was useful precisely because four of its
   seven predictions were marked wrong.
2. **Sample before generalising.** Walk 4 of the 11 first. If they do not agree
   on a mechanism, walk all 11 individually and do not extrapolate — the
   `component_missing_in_engine:itemleftovers` sample split 3/8/3 and refuted its
   own author's hypothesis.
3. **Replay before labelling.** The row-walk *is* the replay; no row is labelled
   from its recorded miss string alone.
4. **Refusal is licensed.** If the evidence does not decide a row, say so and
   leave it. Eleven rows correctly labelled beats eleven rows adjudicated.

## X.5 Ordering of work

1. Wait for the damage lane's report.
2. Cross off any of the 11 that a confirmed formula defect already explains —
   those are `damage_calc` on the lane's evidence, no walk needed. Record which,
   and on what finding.
3. Walk only the remainder, under X.2–X.4.

## X.6 The two elevens are different sets

There are now two unrelated groups of eleven, and conflating them would
double-count:

- **the 11 `capped_lethal` rows** — this appendix's subject, drawn from the
  known-mislabel classes (§W.6);
- **the 11 damage findings in ratio [0.85, 0.92)** — the damage lane's lead
  (§W.3 as amended in §Y).

Different objects entirely: rows versus findings, different selection criteria,
and membership was checked rather than assumed — the two sets do not overlap.
When X.5 step 2 crosses off rows the damage lane has already explained, it
crosses off members of the *first* group only, and the cross-off must be recorded
by seed/step so no row is counted in both places.

## X.7 What this must not do

The residue may not fall as a side effect of this walk. Rows move to
`limit:roll_divergent_lethality` only with a per-row reachability demonstration
attached — the same bar §W.1's guard enforces mechanically. If the walk cannot
produce that demonstration, the row stays where it is and the residue stays where
it is.

---

# Appendix Y — §W.3 amended: the artifact story was wrong, and how it died

## Y.1 Final state

**The 0.900–0.917 band is 11 REAL findings and a genuine lead.** §W.3's
"instrument artifact" verdict is withdrawn. What survives from §W.3 is only the
struck Shadow Ball anchor — still unsourced, still struck.

Measured on the committed brief, which now carries the field that settles it:

| findings in ratio [0.85, 0.92) | 11 |
|---|---|
| `reachable = false` (real) | **11** |
| `reachable = true` (over-reported) | **0** |

The `reachable = true` findings — 9 of them — sit at ratios 0.926, 0.930, 0.930,
0.964, 0.986, 1.062, 1.068, 1.087, 1.169. **Every one is outside the band. The
overlap is zero.**

The 11 real findings, tight at 0.900–0.917 across eleven *different* moves
(hiddenpowerground, aerialace, surf, silverwind, fireblast, doubleedge, return,
psychic, sleeptalk, thunderbolt, hiddenpowerice), which is the signature of a
shared formula factor rather than a per-move data error.

## Y.2 The arc, recorded in full

1. §V.4 offered the band as the damage lane's entry point.
2. The #949 review judged it an artifact of `_classify_ratio`'s 0.92 floor: the
   filter's low edge would pack over-reported rows against it.
3. **I agreed and amended §W.3 to match** — and went further than asked, adding
   the reasoning about mean-vs-top-of-range roll semantics.
4. The `reachable` field (§W.4) was added for an unrelated reason: to stop the
   partition being prose-only.
5. Checked against that field, the artifact story is simply false. The two sets
   are disjoint.

The mechanism of the error is worth naming exactly. Both of us observed *about
eleven* rows in the band and *about eleven* over-reported rows in the population,
and concluded they were the same rows. **They were never the same rows.** The
count matched; the membership did not.

Nothing about the 0.92 floor reasoning was wrong in itself — a filter can pack
rows against its threshold. It just did not happen here, and the arithmetic
coincidence was mistaken for the evidence.

## Y.3 METHOD RULE: check membership, not cardinality

> **Before asserting that two sets are the same set, check membership. Equal
> cardinality is not identity.**

This generalizes the rule already in this ledger about `git ls-files` counts, and
it now has its canonical case: two independent readers, one of them the author of
the data, agreed on a false identification for no reason other than that both
groups numbered about eleven.

The test is cheap whenever the sets are materialized — here it was one
comprehension over a committed JSON file. It is expensive only when the sets live
in prose, which is the argument for §W.4's machine-checkable field and, more
generally, for emitting the *predicate* alongside the count. **A count is a
summary of a set; do not reason about the set from the summary when the set
itself is at hand.**

## Y.4 Standing consequence

The retraction survived one review cycle and was corrected only because a
falsifiable field existed to check it against. Where a claim partitions rows,
the partition ships as data, not as a description of data — this is the fourth
finding in this ledger that a reviewer could only catch because the underlying
predicate was recoverable.

# Appendix Z — damage_calc fix lane: three engine patches, and what the 45 real findings actually were

Fix-lane worktree, branch `scott/damage-calc-fix`, based on the cycle-seven
ledger commit plus main (#950/#951 instrument fixes). Baseline build verified
behaviourally at fingerprint `814b2bd28d3983813b972ba3fd0af7fcc46871085fdea6f0e654c767d076b577`
(30 patches; Pain Split clamp `SideOne: 0 / SideTwo: 7`, Protect ladder
`0.5^k` — both matched §U.1's expectations). Every row below was regenerated
from its seed with `--games 1 --keep-repro 5000` and replayed through
`scripts/replay_residue.py` before any cause was named.

## Z.0 The engine damage-value semantics, source-cited (decides the matcher window)

- `calculate_damage(state, side, choice, DamageRolls::Max)` returns the 100%
  roll: `damage.floor()` with no roll multiplier
  (`third_party/poke-engine-src/src/gen3/damage_calc.rs`, `DamageRolls::Max`
  arm; at the cycle-seven fingerprint, lines 326–329).
- Both gen3 call sites pass `Max`: `gen3/generate_instructions.rs:2442`
  (branch generation) and `:4521` (`calculate_damage_rolls`, the path under
  the Python `poke_engine.calculate_damage` binding the matcher's legal-roll
  set is built from).
- But the damage the engine's branches APPLY is the truncated **average**:
  `gen3/generate_instructions.rs:2495` `let avg_damage_dealt =
  (max_damage_dealt as f32 * 0.925) as i16;` (and `:2541-2543` for the
  crit/kill-branch variants).

So an extract's `engine` component = `floor(0.925 × max)` while the legal set
is anchored on `max`. A legal observation therefore sits at
observed/engine ∈ [0.85/0.925, 1.00/0.925] = **[0.919, 1.081]** of the branch
value, and the brief's 0.90–0.917 cluster is the band just *below* the legal
floor — observations 0.833–0.848 of the engine's max, i.e. bottom rolls of a
sim max that is 1–2 points LOWER than the engine's. The "~8–10% high" reading
was the average-roll disguise on a ~1% max-roll inflation.

## Z.1 The 0.462–0.674 band (14 findings): five causes, none of them type effectiveness

The second-priority cluster ("candidate type-effectiveness/resist rounding")
dissolved completely under replay:

| rows | cause | verdict |
|---|---|---|
| 1500025/7, 1500025/98, 1500164/67(p1), 1500030/82, 1500228/70, 1500294/22 | **Guts wake-turn phantom boost** (Hariyama/Machamp asleep, `-curestatus slp [msg]` then attack; engine boosted 1.5x, sim attacks unboosted) | ENGINE DEFECT — patch 31, **all 6 cleared** |
| 1500222/17 | **Trace-copied Intimidate activates** (Porygon2 traces Intimidate; sim gen3 fires no 'Start' on setAbility; engine dropped Mightyena's Attack and undershot the return fire) | ENGINE DEFECT — patch 32, **cleared** |
| 1500204/83, 1500191/20, 1500074/12, 1500074/32 | **Kecleon Color Change world-drift**: Showdown's Kecleon had type-changed on an earlier turn (`-start … typechange`), the reconstructed engine state still had it NORMAL — resist appears (defender side) or STAB disappears (attacker side) | NOT an engine bug: the boundary state builder never applies typechange. Instrument lane. |
| 1500206/45, 1500040/70 | **Heal-cap structural artifact**: engine max EXACTLY equals sim max (13=13, 16=16); the divergence is the engine world evolving on the average roll so its Leftovers heal capped `_to_full` while the observed roll's heal did not — component shapes differ, magnitudes agree | Matcher artifact. Instrument lane. |
| 1500051/124 | **Same-turn Encore redirection**: Illumise Encores first; sim redirects Wailord's chosen Surf to its last-used Ice Beam (no STAB), engine executes Surf | Engine modeling gap, out of damage-calc scope; documented |

Method note: the triage's best-branch pairing had matched several of these
against the CRIT branch (6.25%) because the non-crit branch differed
*structurally* (the heal-cap artifact above), which is where the "half"
ratios came from. The label "type-effectiveness rounding" would have been a
fit to an artifact.

## Z.2 Patch 31 — `poke-engine-gen3-guts-facade-wake-turn.patch`

`before_move` runs `ability_modify_attack_being_used`/`modify_choice`
**before** `generate_instructions_from_existing_status_conditions` branches
sleep/freeze (`gen3/generate_instructions.rs`), but a sleeping or frozen
attacker's move only executes on the branch that cured the status. Showdown
cures in `slp.onBeforeMove` (`data/mods/gen3/conditions.ts`) and evaluates
Guts at damage time via `onModifyAtk` (`data/abilities.ts:1725-1731`) — so
wake-turn attacks are unboosted. Upstream keyed Guts (and Facade's doubling,
`gen3/choice_effects.rs`) off the pre-turn status. Sleep Talk-called moves
execute while still asleep and keep the boost — preserved via
`choice.sleep_talk_move`.

Sim ground truth (gen3 Custom Game fixtures, `pokezero.showdown_fixture`):
Spored Guts Machamp L80 (atk 237) Rock Slide into Smeargle L90 (def 95),
wake-turn damages over seeds 1–8 = {116,119,122,125,126,127,129} — the roll
set of unboosted max **129**, never of boosted max 192 (min legal 163).
Facade wake-turn over seeds 1–10 = {108..120} = un-doubled max **120**.
Thunder-Waved control = {165,167,176,190} ⊂ boosted max-192 set (Guts stays
for statuses that persist). Pins: `tests/test_engine_guts_trace_fidelity.py`
(3 divergence pins FAIL on a 31-patch-less build, 3 controls pass both ways).

## Z.3 Patch 32 — `poke-engine-gen3-trace-no-activation.patch`

`sim/pokemon.ts setAbility()` fires the copied ability's `Start` event only
when `this.battle.gen > 3`; gen3's trace override
(`data/mods/gen3/abilities.ts:186-196`) calls plain `setAbility`. So a gen3
traced Intimidate/Drizzle/Drought/Sand Stream copies silently. Upstream's
`ability_on_switch_in` ran the switch-in activation match on the copied
ability by design ("tracing intimidate will activate intimidate" — gen4+
behaviour). Fix: a `_ if ability_gained_by_trace => {}` arm short-circuits
the activation match; the passive status/volatile cures and the Forecast
recompute (onUpdate/onImmunity semantics) still apply. Sim probe (seed 42):
Porygon2 switch-in vs active Intimidate Mightyena emits the `-ability …
Trace` line and **no** `-unboost` — matching the repro row's recorded
protocol.

## Z.4 Patch 33 — `poke-engine-gen3-damage-stepwise-truncation.patch` (the 0.90–0.917 factor)

Formula derivation, sim side (all vendored):
- base: `tr(tr(tr(tr(2·L/5+2)·BP·A)/D)/50)` — `sim/battle-actions.ts` getDamage
  (line 1722 at the vendored checkout).
- gen3 modifier order, each step integer-floored:
  `data/mods/gen3/scripts.ts modifyDamage` (lines 33–118): burn(×0.5) →
  screens → spread(0.5, doubles only) → weather → physical-min-1 → **+2** →
  crit(×2) → STAB via `battle.modify` (`tr((v·tr(1.5·4096)+2047)/4096)` =
  `floor(v·1.5)`) → type effectiveness as TWO separate steps (`×2` exact,
  resist `Math.floor(v/2)`) → **roll last**:
  `battle.ts randomizer` (line 2407) `tr(tr(v·(100−random(16)))/100)`.

Engine defect: `common_pkmn_damage_calc` multiplied type × weather × STAB ×
burn × volatile in f32 and floored ONCE at the end. The STAB half-point on an
odd (base+2) survives the ×2 type step: engine max = 3·(B+2) where the sim
gets `floor(1.5·(B+2))·2` — one point lower whenever B+2 is odd.

Row-level proofs (states replayed from the recorded extracts):
- 1500229/3 — Jirachi L73 (spa 188) STAB Psychic vs Vileplume (2× vs Poison):
  B+2 = 53; engine max 3·53 = **159**, sim `floor(79.5)=79 → ×2 =` **158**;
  observed 134 = `floor(158·0.85)` — the sim's exact bottom roll, one below
  the engine's legal floor `floor(159·0.85)=135`.
- 1500201/16 — Light Ball Pikachu L87 STAB Thunderbolt vs Milotic (2× vs
  Water): B+2 = 81; engine 3·81 = **243**, sim `floor(121.5)=121 → 242`;
  observed 205 = `floor(242·0.85)`, engine legal floor 206.

Fix: `common_pkmn_damage_calc` rewritten as the sim's integer pipeline
(burn and weather floored before the +2, physical-min-1, +2, crit ×2 inside,
STAB floored, two type steps floored; screens and the rare volatile
modifiers keep their trailing float position — screens are pool-unreachable
per §U.4). Guts' burn-compensation double is dropped in favour of the
pipeline's Guts-guarded burn halving. Post-build probes reproduce the sim
exactly on both proof rows (159→**158**, 243→**242**).

**Known residual (mechanism 2, documented not fixed):** the sim applies
Choice Band / Light Ball / Guts to the ATTACK STAT with a `modify()` floor
*before* the base formula's `/D` and `/50` truncations (and floors boosted
stats); the engine multiplies `base_power` in float. Exact ×2 mods (Light
Ball) are equivalent; ×1.5 mods on odd stats are not — e.g. 1500024/9
(Ursaring −1 atk, Guts poisoned, STAB Return: sim atk' `floor(257·2/3)=171 →
guts 256` gives max 118; engine 120). Fixing this requires moving those
modifiers from BP to the stat (choice-architecture change) — the right shape
for a follow-up patch, not a rider on this one.

## Z.5 Build chain and gates

| build | patches | fingerprint | gates |
|---|---|---|---|
| baseline | 30 | `814b2bd2…` | §U.1 probes matched |
| +guts/trace | 32 | `ebc7cb07…` | pins 6/6; probes: wake 44 (was 66), trace no-Boost, Pain Split 0/7, ladder 0.5^k |
| +stepwise | 33 | `887a722d…` | pins 12/12; engine tree 17/17; all `rust/pokezero-search` suites green; proof rows 158/242 |

Unpatched-pin evidence: a 30-patch wheel built into a throwaway venv fails
exactly the three divergence pins (`test_guts_wake_turn_rockslide_is_unboosted`,
`test_facade_wake_turn_not_doubled`, `test_traced_intimidate_does_not_activate`)
and passes the three controls.

## Z.6 Row clearance (per-seed reruns, same seeds/games as the cycle-seven run)

Cleared by patches 31/32 (verified before patch 33 landed, then re-verified
under 33): all six Guts rows, the Trace row — and 1500164/67 cleared whole,
including its p2 `sleeptalk`-labelled finding, which was the same boundary's
pairing echo. Controls (Kecleon rows, heal-cap rows, Encore row) unchanged —
no new classes appeared on any rerun seed.

Full sweep over all 32 brief seeds at 33 patches: of the 40 boundaries
carrying at least one real (unreachable) finding, **19 cleared, 21 remain**.

- Patch 33 additionally cleared both proof rows (1500229/3, 1500201/16), the
  0.907-band rows 1500051/95 and 1500115/36, and EIGHT of the >1.0 scattered
  extremes (1500028/66, 1500028/173, 1500059/9, 1500063/3, 1500063/73,
  1500107/20, 1500121/32, 1500127/57) — the extremes largely explained
  themselves once the truncation order was right, as §U.4 hoped.
- Remaining 21, decomposed: 4 Kecleon world-drift (instrument: state builder
  must apply `-start typechange`), 2 heal-cap structural (matcher), 1
  same-turn Encore (engine gap, documented), 1 Sleep-Talk-called extreme
  (1500174/43, the known callee-union path), ~8 mechanism-2 candidates
  (Choice Band / boosted-stat flooring; 1500024/9 proven by arithmetic,
  1500221/13 and 1500126/13 both Choice Band holders), and 5 unreplayed
  extremes (1500180/35, 1500028/44, 1500051/117, 1500121/67, 1500207/27).
- One prior pin updated: `test_struggle_is_physical_burn_halves` asserted the
  engine's OLD trailing burn halving (`healthy // 2`); the sim halves before
  the +2 and the crit x2, and the pin now asserts that stepwise relation.

---

# Appendix Z2 — Cycle eight (33 patches): the seventh hold

Commit `57ea9a62e7a935448eaab459618985375270339e`, fingerprint
`887a722dd2d6cd9b16c7e9736e07f0f5e7f591b17e38a8b9a7a593f31bc6659d`, 33 patches,
28 crate suites green. Seeds 1,500,000–1,500,299.

## Z2.1 Result, against predictions registered BEFORE the run

| | c7 | c8 | predicted |
|---|---|---|---|
| diverged | 170 | 151 | — |
| **outside limits** | 127 | **108** | 105 (interval 95–115) |
| coverage | 0.978 | 0.978 | unchanged ✓ |
| engine errors | 0 | 0 | 0 ✓ |
| new classes | — | 0 | 0 ✓ |

- **P1 (relabels) — exact.** `missing:itemleftovers` 22 -> 11, `psn` 6 -> 4,
  `brn` 2 -> 1: the 14 known-mislabels moved mechanically, as designed.
- **P2 (stepwise/Guts/Trace) — inside interval.** 19 rows cleared.
  `roll_scaled_component` went 72 -> 67 while *gaining* the 14 relabels, so the
  damage clearance is 72 + 14 - 67 = **19 rows** — matching Z.6's measured 19/40
  on the brief seeds almost exactly.

## Z2.2 ACCEPTANCE: not authorized to fire (seventh hold)

108 rows outside limits. The gate is *non-damage_calc, non-documented-follow-up
residue fully attributed*, and the decomposition below is not yet done. The sweep
was not started; seed block 2,000,000+ remains unconsumed after eight cycles.

## Z2.3 Window constant corrected

`_ROLL_LOW` 0.92 -> **0.919**. The engine reports `trunc(0.925 * max)`, so
against Showdown's 85–100% ladder the honest low edge is `0.85 / 0.925 = 0.919`.
The old 0.92 was invented, and an invented threshold is exactly what made an
eleven-row band look like a filter artifact in §W.3 — the reasoning was sound,
the constant was a guess. Roll-set membership remains preferred wherever
available: it needs no constant and cannot grow a floor artifact.

## Z2.4 Probe status — two adopted, two NOT RUN

| probe | status |
|---|---|
| Pain Split clamp | PASS |
| Protect k=1 = 50% | PASS |
| STAB-odd non-crit max 278 | **NOT RUN** |
| burned-Struggle crit 58 | **NOT RUN** |

The two stepwise discriminators could not be run: the battery is not vendored,
the constants appear in no test in this checkout, and `generate_instructions.rs`
does not have the cited content at line 2495 here. **I did not reconstruct the
fixtures from the expected values** — a probe built backwards from its own answer
verifies nothing, and asserting PASS on a fixture I invented would be the §Y
failure with the stakes reversed. Adopting them as standing probes requires the
fixtures themselves; until then the build evidence is fingerprint + 28 green
suites + the two probes that did run.

## Z2.5 Carried forward, not done

Remaining for cycle nine, all explicitly open:

1. **The 11-row `capped_lethal` cross-off** under §X as amended — cross off rows
   patch 31/33 formula fixes already explain (citing the Z finding), then walk
   the remainder with the mass-gated four-exit standard, split pre-registered.
2. **Residue decomposition** of the 108 into the expected shape: mechanism-2
   (~8, documented follow-up, needs a choice-architecture change and is not this
   lane's), Kecleon typechange world-drift (4), heal-cap matcher artifact (2),
   Encore gap (1), small classes.
3. **The Kecleon question specifically**: whether the boundary builder's failure
   to apply `-start typechange` is mine or `engine_world`'s — unassessed.

---

# Appendix Z3 — Cycle nine: the 11-row cross-off, the 108 decomposed, and the Kecleon answer

Commit `c496b1beb8d95dcc8e197b5eddef7b5b56f113ea` (main after #956), fingerprint
`887a722dd2d6cd9b16c7e9736e07f0f5e7f591b17e38a8b9a7a593f31bc6659d` — byte-identical
to the c8 build identity, 33 patches. Fresh worktree, fresh venv, wheel and crate
rebuilt from vendor. Predictions registered in `reports/c9_predictions.json`
BEFORE the regeneration ran; scored below.

## Z3.1 The two NOT-RUN discriminators are now standing probes

`scripts/engine_behavioral_probes.py` is committed: Pain Split clamp, Protect
ladder k=0..4 (with the k>=4 hold at 1/8), the STAB-odd stepwise discriminator
(278 vs old-float 279), and the burned-Struggle discriminator (crit 58 vs old
56). Every fixture and expected constant is TRANSCRIBED from the #955 battery
agent's artifacts (its probe_battery.py fixtures and the sim-anchored
struggle_probe.log RELATION line, quoted verbatim in the module docstring) —
nothing re-derived. References are content-addressed (symbols:
`avg_damage_dealt`, `DamageRolls::Max`, `common_pkmn_damage_calc`; no line
numbers — the "line 2495" citation went stale mid-review in #956 when the #955
rewrite moved the region). Run against this cycle's fresh build: **9/9 PASS**
(fingerprint printed from the venv stamp alongside). Z2.4's refusal is
discharged: future cycles run one command after every rebuild.

## Z3.2 Regeneration with row retention — P1 exact

Same 300 seeds, strict matcher, `--keep-repro 5000`: diverged 151, outside
limits 108, coverage 0.978, engine errors 0, `divergence_classes` **identical
to `reports/c8_summary.json` key for key**. Row identities now exist
(`reports/c9_summary.json`; c8 had committed only class counts, and the 19
cleared rows were never reconstructed from class arithmetic — per the #956
review note, they didn't need to be).

## Z3.3 The 11-row `capped_lethal` cross-off (Appendix X as amended)

**Step 1 — cross-off against Appendix Z: zero rows, as pre-registered (P2).**
All 16 W.6 rows persist in the regeneration with unchanged classes (verified by
seed/step, not by class arithmetic). Patches 31/33 were already in this build;
a row they explained would not have survived it. All 11 proceeded to the walk.

**Step 2 — the walk.** `scripts/capped_lethal_walk.py` (committed) implements
X.2–X.4: target = post-state HP vector + faint set + multiset of non-residual,
non-capped damage components; one roll index per move instance; mass = sum of
reproducing (branch, assignment) probabilities; four exits, 1% floor. Capped
residual ticks are reconstructed from the engine's own residual formulas,
transcribed from the vendored source (`gen3/generate_instructions.rs` status/
item/leech blocks and the Perish Song `damage_amount = active_pkmn.hp` arm) —
never inferred from the branch value.

**The walker itself was wrong three times before it was right**, and each
defect was caught by hand-replaying a verdict against
`scripts/replay_residue.py` before believing it: (1) faint-capped residuals
were compared at their capped branch value, sending reachable rows to
damage_calc; (2) a status tick that kills in-branch but not under the
substituted roll was discarded even when nothing later in the residual order
could fire (the un-faint is licensed by the state: no partial trap, no Perish,
no pending Future Sight — checked, not assumed); (3) recoil/drain were held
fixed instead of re-scaling with the substituted roll (engine:
`trunc(dealt * fraction)`). The sample-of-4 was walked first (P3's four named
rows), disagreed on mechanism, and all 11 were walked individually per X.4.2.

| seed/step | exit | mass | note |
|---|---|---|---|
| s1500050 st33 | **limit** | 6.59% | tbolt m=63, two rolls hit -56; tick caps to kill |
| s1500168 st97 | **limit** | 4.98% | Return roll 88% = -259 exact; Blissey survives at 3 as observed |
| s1500219 st62 | **limit** | 5.86% | tbolt roll 100% = -28; tox tick + Perish kill reconstruct exactly |
| s1500242 st56 | **limit** | 4.98% | HP Fire 87% = -71; capped tick -38 |
| s1500255 st55 | **limit** | 5.86% | DE roll 100% = -146, recoil re-scales to -48, tick kills at 28 |
| s1500012 st24 | limit_not_established | 0.343% | reachable ONLY as two independent exact rolls (X.3.1's anticipated gap) |
| s1500105 st111 | limit_not_established | 0.343% | same two-roll conjunction shape |
| s1500242 st60 | limit_not_established | 0.659% | corrected DOWN from 1.32% when calculate_damage pinned m=54 over the inversion's {53,54} |
| s1500074 st57 | damage_calc (mechanism: U.3.1 signature) | 0 | Explosion + `\|cant\|par`: engine's 25% blocked branch emits NO residuals |
| s1500188 st33 | damage_calc (mechanism: U.3.1 signature) | 0 | same, Swalot par + Explosion; blocked branch drops the whole EOT block |
| s1500251 st56 | damage_calc (1-pt gap) | 0 | obs Surf -116 not in m=130's roll set ({115,117} adjacent); no mechanism-2 marker (Leftovers/Water Veil, no boosts); 1-point max family, mechanism OPEN |

Predicted split 7/2/1/1 (limit/lne/dc/ce); actual **5/3/3/0**. Scored: wrong on
every exact count, right that the majority walks to limit and that reachability
dominates (8/11 reachable). The two damage_calc-with-mechanism rows are the
**Explosion + incapacitating-status signature extended to `par`** — and
s1500188's blocked branch drops residuals with NO Explosion resolving in it,
which narrows U.3.1's open WHY: the suppression rides the blocked-Explosion
branch itself, not the explosion resolving.

**Residue effect: 108 -> 103.** Five rows move to `limit:roll_divergent_
lethality`, each with the per-row demonstration attached
(`reports/c9_capped_lethal_walk.json`: branch tables, roll ladders, masses,
reproducing assignments). Three keep their labels at sub-floor mass; three are
damage_calc with mechanism or quantified gap. The classifier is deliberately
NOT taught any of this — these are ledger adjudications with evidence, exactly
the shape W.1's guard demands.

## Z3.4 The 108, fully decomposed (P4)

Every outside-limits row classified, row-by-row, in
`reports/c9_decomposition.json` (per-row family + evidence basis; replay-first
for everything not already documented). By lane:

| lane | rows |
|---|---|
| matcher/instrument families (documented) | 52 |
| named engine gaps | 20 |
| damage_calc lane (incl. documented follow-up) | 13 |
| comparison-limit shapes (adjudicated candidates, labels kept) | 11 |
| world-construction drift (instrument, boundary builder) | 7 |
| adjudicated `limit` this cycle (Z3.3) | 5 |

Against the pre-registered expected shape:

- **Kecleon typechange world-drift: 4 — exact** (1500204/83, 1500191/20,
  1500074/12, 1500074/32).
- **Explosion + can't-move status: 2 — exact** (1500074/57, 1500188/33; the
  signature's third status variant, `par`).
- **mechanism-2: 6 in the interval [6,10]** — 4 marker-confirmed (Guts+psn+
  boost 1500024/9; Choice Band 1500221/13, 1500126/13, 1500267/77) + 2
  candidates (fresh Intimidate -1 atk 1500028/44; defender -1 spd 1500076/96).
  Documented follow-up, not this lane's.
- **heal-cap structural: predicted 2, actual 9.** The two Z.1 rows are present
  and confirmed; the FAMILY (engine world evolves on the average roll, cap
  state diverges, component shapes differ) is population-wide once you look:
  7 more rows replay to the same mechanism. The 2 was a brief-seed count, not
  a population count — scored against the prediction.
- **Encore: predicted 1, actual 11.** Same scoring error, larger: same-turn
  Encore redirection (sim redirects the chosen move to last-used; engine
  executes the chosen move) explains 3 of Z.6's 5 "unreplayed extremes"
  (1500121/67 ratio 7.12, 1500051/117, 1500180/35), the documented 1500051/124,
  and 7 structural rows (mechanical marker: `-start Encore` + executed move !=
  chosen move; spot-verified by replay on 1500232/54). The single biggest
  named engine gap in the residue.
- **The 5 "unreplayed extremes" are now all replayed**: 3 Encore-redirect,
  1 mechanism-2 candidate (1500028/44), 1 NEW recoil-rounding candidate
  (1500207/27: obs recoil 25 = sim `round(75/3)` vs engine `trunc(0.33*75)` =
  24 — one point, every Double-Edge max roll).

New named candidates out of the remainder triage (WHAT recorded per row,
candidate-not-finding):

1. **Fixed-damage path skips post-damage hooks** (3 rows + 1 Counter row):
   Seismic Toss KOs without firing Rough Skin (1500103/76, 1500274/15) or
   Flame Body (1500287/76), and Counter after Seismic Toss returns nothing
   (1500192/91). Grep-supported: `apply_fixed_damage` and the fixed-damage
   arms in `gen3/choice_effects.rs` never register `damage_dealt` and run no
   contact-ability hook. Appendix K lineage (fixed damage bypassing
   Substitute was the same path).
2. **Water Absorb fires on Rain Dance** (1500124/58): engine emits
   `Heal SideTwo: 72` (= maxhp/4) immediately after `ChangeWeather RAIN` with
   a Water Absorb active on that side. One row, 100% branch, instruction-level
   evidence.
3. **Counter/Mirror Coat semantics** (1500155/22, 1500264/40 + the fixed-damage
   row above): inverse direction too — Showdown's Mirror Coat retaliated with
   nothing where the engine dealt 2x59, with the engine's own
   `ChangeDamageDealtMoveCatagory Physical -> Special` in the branch. WHY open.
4. **World-construction drift, two new members of the Kecleon ownership class**:
   Trace-copied ability never materializes (1500248/77+78: world has
   `ability=TRACE` + FLASHFIRE volatile; engine damages straight through the
   traced Flash Fire immunity) and a stale toxic stage across a Rest cure
   (1500243/79: engine ticks stage 5 where Showdown ticks a fresh stage 1).
5. **Compound-class roll-divergent-lethality shapes** (7 rows, incl. 1500054/125
   and 1500112/40 from W.6's attributed set): same faint-divergent shape the
   walk adjudicated, wearing compound class names the #950 override cannot
   touch. Recorded as candidates for a future cross-off under the same
   standard; labels kept, residue NOT reduced for them.

## Z3.5 The Kecleon question — answered, with the drop points cited

**The gap is engine_world's (production world construction), not the
differential's.** The differential has no boundary builder of its own: it calls
the production constructor verbatim
(`scripts/engine_transition_differential.py:1408-1450` — `world_battle_spec`
with `_public_effect_signals`), and `world_battle_spec`'s signature
(`src/pokezero/engine_world.py:762-790`) has **no channel for a live type
override at all**, so the harness could not pass one even if it derived it.

The event IS parsed, and dies between the parser and the world:

- consumed: `src/pokezero/showdown.py:1679-1690`
  (`_update_live_type_override` stores `type:<T>` per slot; `typechange` is
  deliberately NOT in `TRACKED_VOLATILES` at :2595, so it never enters
  `replay.volatiles`), and the OBSERVATION path applies it at :1994-1995 via
  `_apply_live_type_override` (:1928). The encoder sees the retype; the world
  does not.
- dropped (payload): `src/pokezero/local_showdown.py:2080`
  `_public_materialization_payload` — the per-side dict (~:2101-2119) carries
  boosts/volatiles/toxicStage/stallCounter/sideConditions but no
  `live_type_override` field.
- dropped (constructor): `src/pokezero/engine_world.py:1316` builds every mon
  with `types=info.types` (dex base types). The only live-retype arms are
  `_apply_transform` (:619, donor types + TYPECHANGE volatile at :649) and
  `_apply_forecast_types` (:731-757, Castform, re-derived from public weather
  precisely because the payload carries no type event).

The fix shape is a two-point production change (payload field + a consuming
arm mirroring `_apply_transform`), and it reaches the LIVE search path —
`engine_search` builds worlds through the same constructor (:648). That is not
a small contained boundary-builder change, so per this cycle's brief it is
**reported, not made**. The same lane owns the two new drift candidates above
(traced ability, toxic stage): all three are protocol-public state the payload
never carries.

## Z3.6 ACCEPTANCE (eighth hold — and the gate question answered)

The sweep was NOT started; seed block 2,000,000+ remains unconsumed after nine
cycles. But the gate condition — *non-damage_calc, non-documented-follow-up
residue fully attributed* — is now MET on the evidence above: all 108 rows
carry a named family with per-row evidence; the damage_calc lane holds 13
(6 mechanism-2 documented follow-up, 7 one-point/rounding candidates with the
observed value placed against the exact legal roll set); everything else is an
adjudicated limit, a documented instrument family, a named engine gap with
row-level WHAT (and WHY where established), or world-construction drift with
the owning lane identified. The honest register of what "attributed" means per
family is in `reports/c9_decomposition.json` — three families carry WHAT-level
candidates rather than proven WHYs (counter/mirror-coat semantics, confusion
fan shape, battle-end attribution tie), disclosed as such. Launching the sweep
is a separate decision with its own execution requirements (per-shard wheel
rebuild, registry seeds, `--checkpoint` retention) and is not taken here.

## Z3.7 Artifacts

- `scripts/engine_behavioral_probes.py` — standing rebuild gate (Z3.1)
- `scripts/capped_lethal_walk.py` — the X.2-X.4 walker
- `reports/c9_predictions.json` — pre-registered, scored in Z3.2-Z3.4
- `reports/c9_summary.json` — regeneration aggregates (repros stripped, size policy)
- `reports/c9_capped_lethal_walk.json` — per-row walk evidence for the 11
- `reports/c9_decomposition.json` — all 108 rows, family + basis

---

# Appendix Z4 — Cycle ten: the Encore family was never an engine gap

Branch `scott/engine-gen3-encore-redirect`. Engine **unchanged**: 33 patches, fingerprint
`887a722dd2d6cd9b16c7e9736e07f0f5e7f591b17e38a8b9a7a593f31bc6659d`, identical to the c9
decomposition's. **No vendored patch.** The fix is parser + `engine_world` — public-repo
Python, normal review.

## Z4.1 RELABEL: `encore_redirect_gap` (11) -> `boundary_builder_last_used_move_absent`

The c9 brief recorded the WHAT from a single row: *the engine models Encore duration and
failencore but not the same-turn redirect.* The first half is right. **The second half is
wrong, and the engine has implemented the redirect all along** — `generate_instructions.rs`,
immediately after `apply_instructions`, mirroring Showdown's
`encore.condition.onOverrideAction`. It was found while looking for somewhere to *insert* it.

**Replay evidence — 4 rows, all drawn from the 6 that c9 classified by mechanical marker
rather than replay** (the inference-based subset, chosen deliberately as the place the
classification could break):

| row | chose | Showdown executed | engine executed |
| --- | --- | --- | --- |
| 1500099/49 | Drill Peck | **Protect** (encored; then failed) | Drill Peck, 146 |
| 1500123/38 | Will-O-Wisp | **Fire Blast**, 134 | Will-O-Wisp (burn) |
| 1500136/27 | HP Flying | **Toxic** (missed) | HP Flying, 127 |
| 1500161/44 | Rest | **Earthquake**, 35 | Rest (failed at full HP) |

4/4 are same-turn redirection, so the FAMILY was correctly identified. Encore resolved first
in all four. But every one shows `SetLastUsedMove <target>: None -> …` and **no
`ApplyVolatileStatus ENCORE` anywhere** — the Encore never applied at all.

**Engine probe pair, same state twice, nothing else changed:**

```
last_used_move = Move(0) [seismictoss]      last_used_move = None
  ApplyVolatileStatus SideTwo: ENCORE         SetLastUsedMove SideOne: None -> Move(M0)
  Damage SideOne: 100   <- REDIRECTED         SetLastUsedMove SideTwo: None -> Move(M1)
  ChangeVolatileStatusDuration ENCORE: 1      <- no Encore; the chosen move ran
```

The redirect fires the moment it is given a last move. With `None` it cannot: Encore's own
`onStart` guard — Showdown's `if (!move) return false`, implemented faithfully as
`LastUsedMove::None => true` in `move_has_no_effect` — fails the Encore outright, so
duration, the move-slot lock and the redirect all never happen.

**Root cause.** `engine_world.py` set `last_used_move` **only inside `if "encore" in
volatiles`** — i.e. only for a mon *already* encored. A mon being encored *this turn* reached
the engine as `None`. The engine was correct at every step; the world never told it.

The gap is therefore strictly **larger** than the label said (the whole Encore, not just the
redirect) and **smaller** in blast radius (a missing seed, not an engine-behaviour patch).

## Z4.2 Consumer survey before seeding

Seeding a field unconditionally can unlock behaviours beyond the one being fixed, so every
engine read of `last_used_move` was enumerated first:

| consumer | what it does | reachable in gen3 randbats? |
| --- | --- | --- |
| Encore — option filter (`state.rs`) | restricts selectable moves to the locked one | **yes** |
| Encore — `onStart` failure guard | fails Encore on None/Switch/failencore/0-PP | **yes** |
| Encore — same-turn redirect | the `onOverrideAction` mirror | **yes** |
| Encore — PP-exhaustion end | ends Encore when the locked move hits 0 PP | **yes** |
| **Fake Out** (`choice_effects.rs`) | **strips ALL effects once the user has moved** | **no — not in the pool** |

Spite, Grudge and Mirror Move do **not** read `last_used_move` in gen3 (GRUDGE exists only as
a volatile enum entry; MIRRORMOVE appears only in the failencore list).

Fake Out is the one genuine non-Encore behaviour change: because `last_used_move` was
previously almost always `None`, the engine let Fake Out flinch **unconditionally**. It is
pool-unreachable in randbats but reachable in `gen3customgame`, which the fixture harness
uses, so it is pinned rather than left to be discovered.

## Z4.3 Parser semantics, transcribed from the patch that already bound the engine

`poke-engine-gen3-lastmove-semantics.patch` moved the ENGINE's record point to match
`Pokemon.moveUsed()`. The parser is written against that same truth table, because a world
that disagrees with the engine about what "last move" means is worse than one that omits it —
it would lock Encore onto the *wrong* move rather than onto nothing:

* a move that **misses, fails, or is blocked by Protect still counts as used** — `moveUsed`
  precedes `useMove`, and Showdown emits the `|move|` line either way;
* every immobilizer (par/slp/frz/flinch/confusion-self-hit/attract) returns false from
  `onBeforeMove`, so **no `|move|` line is emitted at all** — Showdown emits `|cant|`, and the
  parser records nothing. The match is by construction, not by enumeration;
* **Sleep Talk's CALLER records; its CALLEE must not.** The callee runs through `useMove`,
  which never touches `lastMove`. Publicly the callee's line carries `[from]`, which is the
  discriminator. The engine-side patch called the inversion out explicitly as the naive
  mistake; the parser pins the same row;
* switch-out clears to a **`switch` sentinel, not to unknown**. `Pokemon.clearVolatile()`
  nulls `lastMove`, and Encore correctly fails against a fresh switch-in — that is a positive
  fact the engine has a distinct variant for. Collapsing it into `None` would relabel
  knowledge as ignorance.

**PP.** Sim-probed rather than assumed: with the redirect firing, `seismictoss` went 32 -> 30
across two turns while the chosen `rest` stayed at **16, untouched**. PP is charged to the
**encored** move. Pinned, because a PP mis-charge is exactly the sort of secondary divergence
that resurfaces two cycles later as a phantom PP row with no visible link to Encore.

**Unresolvable last moves stay `None` on purpose.** If the observed move is absent from the
constructed moveset (an unrevealed slot on a sampled world), the seed is left `None` rather
than guessed — reproducing today's behaviour for that side instead of inventing a lock.

## Z4.4 An information-boundary assertion that needed narrowing

`test_public_materialization_samples_deferred_baton_pass_action_without_private_request`
asserted that `"harden"` appears **nowhere** in the public payload, as a proxy for "nothing
was copied from p2's private request". Adding `lastUsedMove` broke it — and the proxy, not
the change, was at fault: p2 **visibly used Harden on turn 1**, verified by reading the
protocol rather than by argument (`|move|p2a: Ditto|Harden|p2a: Ditto`). Its appearance is
public fact. The turn-2 Harden that Baton Pass interrupted is the one that must not leak, and
does not — it never reached a `|move|` line. The assertion now checks the real invariant on
p2's subtree (no moveset, no deferred action, and Protect — never publicly used — absent).

## Z4.5 Carries from the #957 verification

* **The relabel above is the one #957's merge note announced as incoming.** It does not
  weaken the c9 gate: the rows stay attributed, the family assignment improves, and the
  story strengthens — engine already correct plus a missing seed is a smaller blast radius
  than an engine-behaviour patch.
* **Erratum 1.** The committed c9 walk artifact was serialized by a slightly earlier walker
  iteration than the committed script. Direction of error: conservative.
* **Erratum 2.** 1500242/60's committed branch table lists **7 of 15** branches while
  claiming `branches_truncated: 0`; the 8 omitted are zero-reproducing crit arms. Direction
  of error: conservative (the omitted arms could not have reproduced the observation).

## Z4.6 Differential: predicted 11, cleared 12 — and the 12th is the interesting one

300 games, seeds 1500000-1500299, strict matcher, same fingerprint. Prediction was recorded
before the run (`reports/c10_encore_prediction.md`).

| | predicted | actual |
| --- | --- | --- |
| the 11 `encore_redirect_gap` rows | 11 clear | **11/11 clear** |
| other families | **zero change** | **one more row cleared** |
| outside-limits population | 108 -> 97 | 108 -> **96** |

The extra row is **seed 1500285 step 14**, which c9 filed under
`measurement_boundary_residual_truncation`. Replayed at the base build before concluding
anything:

```
|move|p2a: Smeargle|Encore|p1a: Octillery      chose: thunderwave
|-start|p1a: Octillery|Encore
|move|p1a: Octillery|Hidden Power|p2a: Smeargle    <- the ENCORED move
|-crit| |-damage|p2a: Smeargle|0 fnt  |faint|      <- KO ends the turn
engine (base): ChangeStatus PARALYZE + Heal SideOne 17 + Heal SideTwo 15
```

It is the same mechanism as the other 11. It was classified by its **downstream symptom**
rather than its cause: the redirect's crit-KO ends the turn before residuals, so the engine's
surviving Leftovers looked exactly like the boundary-truncation signature (#876/U.3.2), and
the row was filed there. The truncation reading was a true description of what the row looked
like and a false one of why.

So the family was **under-counted by one**, and the correct size is **12**. The prediction
missing by +1 in this direction is the healthy direction — it means the fix reached a row the
census had not attributed to it, not that it perturbed something unrelated.

**The `zero change elsewhere` half of the prediction held exactly.** Nothing outside the
Encore mechanism moved: the 5 adjudicated `limit:roll_divergent_lethality` rows and the 3
`limit_not_established_keeps_label` rows are all still present and unchanged, as are the
remaining 96. The 39 rows present now but absent from the c9 108 are precisely the 39
`limit:*` rows that the outside-limits population excludes by definition — not regressions.

**Method note.** The 12th row was found by diffing the c9 row identities against the new run
rather than by comparing counts, and diagnosed by reverting the change (via a saved diff, not
a stash) to replay the row at the base build. A count-level comparison would have shown
"12 cleared, expected 11" and invited a hand-wave about noise; the identity diff named the
row, and the replay named the cause.

## Z4.7 Three corrections from the #958 verification

### (i) Full-frame clearance was 16, not 12

Z4.6 counted only the **outside-limits** frame (108 -> 96). The full frame cleared **16**:
the 12 named there plus four more that sat in the **limit bucket**, and so were invisible to
a population defined as outside-limits:

| row | where it sat |
| --- | --- |
| 1500000/11 | limit bucket |
| 1500138/88 | limit bucket |
| 1500294/71 | limit bucket |
| 1500297/14 | limit bucket |

Verified: none of the four is among the c9 decomposition's 108, and all four are absent from
the post-fix run. The Encore mechanism therefore accounted for **16** rows across the whole
frame, against a documented family of 11.

Standing lesson, and it is the same shape as the 12th row: **a population definition is a
lens, not a census.** "Outside-limits" was the right frame for adjudicating the 108, and the
wrong frame for measuring a fix's reach. Report clearance against the full frame and the
sub-population separately, because a fix does not know which bucket a row was filed in.

### (ii) Erratum 2 was FALSE — and it was my transcription, not the artifact

Z4.5 recorded that 1500242/60's committed branch table "lists 7 of 15 branches while claiming
`branches_truncated: 0`". **The artifact says no such thing.** Read directly out of
`reports/c9_capped_lethal_walk.json`, row index 8, candidate 0:

```
branches:            <list len 7>
branches_truncated:  8            <- 7 + 8 = 15, correct and self-consistent
```

The `0` came from the **#957 merge note**, and I carried it into the ledger without opening
the file it described. There was no artifact defect. Erratum 2 is withdrawn; the only real
erratum from that verification is erratum 1 (the walk artifact serialized by a slightly
earlier walker iteration).

**The transcription-chain lesson, which is worth more than the erratum was.** I have a
standing rule about replaying before narrating and a standing rule about verifying provenance
at source, and I applied neither here, because the claim arrived in a *review note* rather
than in a tool result — and review prose reads like conclusion, not evidence. It is neither:
**a merge note is upstream data.** It is written by someone reading the same artifacts, at
speed, and it can be wrong in exactly the ways an artifact cannot (an artifact at least
disagrees with itself loudly). The rule generalizes past this ledger:

> Verify against the artifact, not the note. Prose that summarizes evidence is not evidence,
> whoever wrote it — including a reviewer, including me.

This is the fourth entry in this ledger where a **partial truth positioned where a reader
looks for the whole one** cost something: the class label (U.2), the unenforced comment and
the half-true "latches with the schema" note (encoder vocab), the function name, and now a
merge note. The failure mode is identical every time and the defence is always the same one:
open the thing being described.

### (iii) The Fake Out pin is an ENGINE-BEHAVIOR pin

Z4.2 lists Fake Out under the consumer survey, which is right, but the pin itself was
described alongside the parser pins as though it were of a kind with them. It is not, and the
distinction matters for what a future failure would mean:

* **6 parser pins** — the seeding **discriminators**. They fix what the world is allowed to
  claim it knows: executed-vs-immobilized, caller-vs-callee, switch-vs-unknown. A failure
  there means the world is describing history wrongly.
* **1 engine-behavior pin** (Fake Out) — fixes what the ENGINE does once correctly seeded.
  Nothing about the parser changes if it breaks; it would mean the engine's Fake Out handling
  moved underneath a now-populated field.

Same file, different failure meaning. Worth keeping legible because the seeding pins are
stable-by-construction while the behavior pin tracks an engine the vendored patch stack keeps
editing.

---

# Appendix Z5 — Patch 34: self-destruct is gated on the move actually executing

Branch `scott/engine-gen3-explosion-blocked-residuals`. **Vendored patch 34**
(`poke-engine-gen3-explosion-selfdestruct-gate.patch`), fixture-refresh still last at 35.
Fingerprint `5b29e611468d3baa930984d5b8557280835e72f1ce38d8dc3c6b183e15c344dc`; the
unpatched comparison build is `9ecfacadc938c0da`.

## Z5.1 The documented WHY was the symptom again

The brief carried this as *"when the engine's branch has the move BLOCKED (frz/slp/par cant),
it drops the WHOLE end-of-turn residual block"*. **It does not, and it never did.** One
control settles it — an ordinary blocked move, same paralyzed mon, same Leftovers:

```
TACKLE,    p2 paralyzed, 25% blocked branch:  ['Heal SideTwo: 19']      <- residuals fine
EXPLOSION, p2 paralyzed, 25% blocked branch:  ['Damage SideTwo: 89']    <- user is DEAD
```

The engine applied Explosion's self-faint in `choice_before_move`, which runs at
`generate_instructions.rs:2245` — **before** the immobilizers are rolled at `:2258`. So the
blocked branch killed its own user, and residuals then correctly did not run, because the mon
was dead. The missing Leftovers tick was real; its cause was one step upstream of where the
family name pointed.

Showdown, verified at source: `selfdestruct` lives in `useMoveInner`
(`sim/battle-actions.ts:501`, guarded `gen !== 4`), and `runMove` calls `useMove` only after
the `BeforeMove` gate returns true. **An immobilized Pokemon never explodes.**

The fix relocates the faint to immediately after the status-condition gate — the blocked
branches were already pushed to `final_instructions` by that call and never reach it. Position
within the surviving branch is unchanged: still before damage (gen3 faints the user first),
still firing through Protect (Showdown's faint precedes `tryMoveHit`), DAMP guard preserved,
and Sleep Talk's callee still reaches it because the gate is skipped for it.

**This is the third consecutive assignment whose documented WHY named a downstream symptom**
(encore-redirect -> missing world seed; boundary-truncation -> Encore misfiling; residual-drop
-> pre-gate self-KO). The common shape: a residual/HP-component signature is the most VISIBLE
part of a divergence and the least diagnostic, because every upstream cause that changes who
is alive at end-of-turn produces the same missing tick. **Read the branch that is wrong, not
the component that is missing.**

## Z5.2 Pins: 2 fail unpatched, 4 pass both ways

| pin | unpatched | patched |
| --- | --- | --- |
| `the_blocked_branch_no_longer_kills_its_own_user` | **FAIL** | pass |
| `the_blocked_branch_keeps_its_end_of_turn_residuals` | **FAIL** | pass |
| `an_ordinary_blocked_move_keeps_its_residuals` | pass | pass |
| `the_firing_branch_still_faints_the_user_and_still_skips_residuals` | pass | pass |
| `damp_still_prevents_the_faint` | pass | pass |
| `a_healthy_user_still_explodes` | pass | pass |

The third row is the load-bearing control and the reason it is written at all: it encodes the
refutation, so a future reader who takes the family name at face value and re-adds residuals
to blocked branches will find the premise already disproved in the test file rather than
rediscovering it from a census.

## Z5.3 Differential: predicted 2, cleared 4 — the family was under-counted again

300 games, seeds 1500000-1500299, strict matcher. Prediction recorded before the run
(`reports/c10_explosion_prediction.md`). Post-#958 baseline: 96 outside-limits.

| | predicted | actual |
| --- | --- | --- |
| named rows (1500074/57, 1500188/33) | clear | **both clear** |
| other families | zero change | **two more cleared** |
| outside-limits | 96 -> 94 | 96 -> **92** |
| newly divergent | 0 | **0** |

The two extra rows were checked on the **unpatched** build rather than assumed, per the
prediction's own instruction. Both are the same signature:

| row | c9 family | what it actually is |
| --- | --- | --- |
| 1500188/57 | `matcher_accounting_best_branch` | `p2: explosion` + `\|cant\|p2a: Swalot\|par` |
| 1500286/38 | `matcher_overreport_legal_roll` | `p2: explosion` + `\|cant\|p2a: Regirock\|par` |

So the family is **4**, not 2, and the two strays were filed by downstream symptom — a
branch-accounting mismatch — rather than by cause, exactly as 1500285/14 was filed as
boundary truncation in the Encore family.

**Both of the last two fixes have under-counted their own family in the same direction, for
the same reason.** A mechanism that changes *which branch is right* shows up in whatever
class the matcher lands in once the right branch is missing — residual truncation, branch
accounting, legal-roll overreport. Those class names describe the matcher's experience, not
the engine's error. **Predicting from family labels therefore systematically under-predicts;
predict from the mechanism's signature instead** (here: any boundary pairing a self-destruct
move with an incapacitating status), and identity-diff the full frame to catch the rest.

The first count-level comparison here was *also* misleading in a second way worth recording:
the default report retains only 25 repros of 130+ divergences, so an identity diff computed
from `repros` silently compared two truncated samples and reported "0 cleared, 0 new". The
re-run with `--repros-per-game 40` is what produced the table above. **A diff over a truncated
set is not a diff.**

## Z5.4 Artifacts

- `reports/c10_explosion_prediction.md` — prediction, pre-registered
- patch: `third_party/poke-engine-gen3-explosion-selfdestruct-gate.patch` (34 of 35;
  fixture-refresh last)
- pins: `rust/pokezero-search/tests/gen3_selfdestruct_gate.rs`

---

# Appendix Z6 — Live typechange reaches the world; and retention becomes verifiable

Branch `scott/engine-world-kecleon-typechange`. **No vendored patch** — parser payload +
`engine_world` only. Engine 34 patches, fingerprint `5b29e611468d3baa…`, unchanged by this
change.

## Z6.1 The parser knew for months; only the observation path listened

`live_type_override` has been produced since the v3 observation work
(`showdown.py::_update_live_type_override`), and exactly one consumer existed:
`_apply_live_type_override`, on the **observation** path. The world was still assembled from
base Pokédex types, so a Kecleon whose Color Change had retyped it arrived at the engine as
plain Normal.

This is the same shape as the `last_used_move` gap (Z4) — a publicly-observed fact the parser
already derived and the world silently dropped — and it is now the second instance. Worth
naming as a class: **a parser field with exactly one consumer is a latent world gap.** The
obs path and the world path read the same protocol for different purposes, and a field added
for one of them does not reach the other by default.

**Wrong in both directions from one field**, which is why the two replayed rows look unrelated
until you notice they share a cause:

| row | direction | Showdown | engine |
| --- | --- | --- | --- |
| 1500074/12 | Kecleon **attacks** (Return) | 27 — no Normal STAB | 43 (43 / 1.5 = 28.7) |
| 1500191/20 | Kecleon **is hit** (HP Ice) | 16, `-resisted` | 34, neutral |

Both fall out of one `types` field, so one seeding fixes both.

## Z6.2 Precedence, and why the observed arm goes last

Three arms can retype the active mon. Applied **transform -> forecast -> typechange**:

| arm | source | kind |
| --- | --- | --- |
| `_apply_transform` | the donor's types | derived from a rule |
| `_apply_forecast_types` | public weather | derived from a rule |
| `_apply_live_typechange` | an observed `typechange` line | **OBSERVED** |

**Observation beats derivation.** The first two reconstruct what the types *should* be from a
rule; the third is the sim stating what they *are*. Showdown reaches the same answer by a
different route: Color Change's `onAfterMoveSecondary` calls `setType(type)`
(`data/abilities.ts:554-562`), and since every arm mutates `pokemon.types` in event order,
whichever fired most recently wins. Applying the observation last reproduces that **without
modelling the ordering** — which matters, because the ordering is the part most likely to be
got wrong later.

`setType` REPLACES rather than appends, so a retyped mon is mono-type even if it was
dual-typed. Pinned, because "add a type" is the plausible misreading and it would leave the
old type resisting things it no longer resists.

**Only the `type:` form is consumed.** `forme:` (Castform Forecast) is deliberately left to
`_apply_forecast_types`, which already derives the same answer from the same public weather.
Consuming it here too would give Castform **two writers that must agree** — precisely the
shape that let the encoder-vocabulary bug live for months. Forecast is `onUpdate`, so the
derived arm cannot lag the observation. Pinned as a deliberate non-consumption so a future
reader does not "complete" the handler and reintroduce the second writer.

## Z6.3 Retention provenance: the payload now records it (carry from the #959 verification)

The #959 clearance numbers had no committed artifact, and the committed
`c10_encore_differential.json` showed 25 repros with no record of which flags produced them.
Fixed mechanically rather than by assertion — and building the fix surfaced *why* the original
diff went wrong:

**`--repros-per-game` and `--keep-repro` are different knobs, and only one of them is what a
diff depends on.** `run_game` takes `--repros-per-game` (what each game retains, hence what
lands in the checkpoint); `build_report` takes `--keep-repro` (what the aggregated report
carries). A run invoked with `--repros-per-game 40` and no `--keep-repro` still writes a
report truncated to the `--keep-repro` default. That is exactly what happened in Z5.3: the
identity diff silently compared two 25-row samples of 130+ divergences and reported "0
cleared, 0 new" from a real 4-row change.

`build_report` now emits:

```json
"repro_retention": {
  "repros_per_game": 40, "keep_repro": 500,
  "repros_retained": 131, "transitions_diverged": 131, "repros_complete": true
}
```

`repros_complete` is the field a future identity diff should check before trusting
`report["repros"]`. Known limit, recorded rather than papered over: on the `--merge-from`
path `repros_per_game` is `null`, because the per-game flag is not recoverable from checkpoint
records — `repros_complete` remains valid there and is the load-bearing field.

`reports/c10_explosion_differential.json` is committed with this PR, rebuilt from #959's
retained checkpoint so it carries the full 131-row set. It independently reproduces that PR's
headline: **92 outside-limits**.

## Z6.4 Differential: predicted >= 4, cleared exactly 4 — and the refinement that matters

300 games, seeds 1500000-1500299, strict matcher, 34 patches. Prediction pre-registered
(`reports/c10_kecleon_prediction.md`).

| | predicted | actual |
| --- | --- | --- |
| named rows | 4 clear (floor) | **4/4** |
| more, by mechanism signature | expected | **none** |
| outside-limits | 92 -> 88 or better | 92 -> **88** |
| newly divergent | 0 | **0** |

**Both retention blocks report `repros_complete: true`** (127 of 127 and 131 of 131), so this
identity diff is over the FULL divergent set on both sides and that is checkable from the
committed artifacts rather than asserted here. It is the first diff in this program for which
that is true.

**I predicted more than 4 and was wrong, and the reason refines the Z5.3 lesson rather than
contradicting it.** The Encore and Explosion mechanisms both changed **branch structure** —
who is alive, which move executed — so once the right branch was missing the matcher filed
those rows wherever its accounting broke (`measurement_boundary_residual_truncation`,
`matcher_accounting_best_branch`, `matcher_overreport_legal_roll`), scattering the family.
This mechanism changes only a **magnitude inside an otherwise-correct branch**: the right
branch is present, its damage number is wrong. Nothing gets re-filed, so the family stayed
exactly where c9 put it — all four in `roll_scaled_component`.

So the sharpened rule:

> **Structural mechanisms scatter across matcher classes and under-count; magnitude
> mechanisms do not.** Before predicting from a family label, ask whether the mechanism
> changes which branch is right or only what a number inside it says.

That is a cheap test and it would have called both of the previous two cycles correctly, and
this one too.

## Z6.5 Artifacts

- `reports/c10_kecleon_prediction.md` — pre-registered
- `reports/c10_kecleon_differential.json` — 127 rows, `repros_complete: true`
- `reports/c10_explosion_differential.json` — #959's missing artifact, 131 rows, reproduces 92
- pins: `tests/test_engine_world_live_typechange.py` (9)
---

# Appendix Z7 — Cycle eleven: mechanism-2 stat-modifier flooring (patches 35-36), and what the family label got right and wrong

Fix-lane worktree, branch `scott/engine-gen3-stat-modifier-flooring`. The work was
authored against main at #958 (33 patches, fingerprint `887a722dd2d6cd9b...`, probes
9/9), where all six `mechanism2_stat_modifier_flooring` rows were regenerated from seed
and replayed through `scripts/replay_residue.py` before any design; #959 (patch 34,
explosion-selfdestruct-gate) and #961 (Kecleon world seeding) merged mid-review, so the
branch carries a merge of main, the two patches renumbered to slots 35-36
(fixture-refresh still last), and the binding measurement below re-run against the
88-row post-#961 baseline. The pre-merge run (96-frame, fingerprint `a987f3db...`,
outside-limits 96 -> 86) measured the identical ten-row clearance and is retained in
branch history; every number in Z7.5 is from the post-merge tree.

## Z7.1 Derivation: where gen3 Showdown puts the stat modifiers (all vendored-source citations)

- `getDamage` (sim/battle-actions.ts:1589-1726) computes
  `attack = attacker.calculateStat(attackStat, atkBoosts)` — boost stages floor AT THE
  STAT (`Pokemon.calculateStat`, sim/pokemon.ts:561-595: `Math.floor(stat * table[boost])`
  up, `Math.floor(stat / table[-boost])` down) — then fires the
  `ModifyAtk`/`ModifySpA`/`ModifyDef`/`ModifySpD` events (:1712-1713) BEFORE the base
  formula `tr(tr(tr(tr(2L/5+2) * BP * A) / D) / 50)` (:1722).
- Every effective-gen3 handler `chainModify`s: Guts 1.5 / Huge+Pure Power 2 (prio 5,
  data/abilities.ts), Choice Band 1.5 (prio 1, data/items.ts), Thick Club 2 (gen4 mod),
  Light Ball 2 — **SpA-only in gen3** (data/mods/gen3/items.ts:187 defines `onModifySpA`
  and kills the inherited hooks), Soul Dew 1.5 SpA/SpD (gen6 mod), Marvel Scale 1.5 Def
  (prio 6), Metal Powder **x2 Def, untransformed Ditto** (base data/items.ts, unmodified
  through the chain), and the gen3 1.1x type items — **stat-side in gen3**
  (data/mods/gen3/items.ts re-hooks every one of them from onBasePower to
  onModifyAtk/onModifySpA at prio 1; Sea Incense is 1.05). Hustle alone modifies the stat
  DIRECTLY (`return this.modify(atk, 1.5)` — its own floor before the chain's).
- Event tail: `chainModify` accumulates `(prev*next + 2048) >> 12` at 4096 scale and
  runEvent applies ONE `modify(relayVar, modifier)` = `tr((tr(v*tr(m*4096)) + 2047)/4096)`
  — round-half-down (sim/battle.ts:2318-2359, runEvent:929-933).
- Explosion/Self-Destruct: gen<=4 halves the MODIFIED defense stat,
  `clampIntRange(Math.floor(defense/2), 1)` (battle-actions.ts:1715-1717) — not BP.
- gen3 Plus/Minus check `getAllActive()` (data/mods/gen3/abilities.ts:115-135) — the foe
  counts, so a Minun facing a Plusle is boosted x1.5 in singles.
- **Burn is NOT a stat modifier in gen3**: it is the FIRST damage-side step of
  `modifyDamage` (data/mods/gen3/scripts.ts:41-43), halving physical damage pre-+2 and
  skipped entirely for Guts. Patch 33's placement was correct; no reconciliation needed.
- Stays BP-side per the same reading (NOT touched): the pinch abilities (gen4 mod
  onBasePower), Thick Fat (gen4 mod onSourceBasePower 0.5), Facade (basePowerCallback),
  screens/spread/weather (damage-side, gen3 scripts.ts modifyDamage).

Every stat-side rule above was probed live (gen3customgame fixtures,
`pokezero.showdown_fixture`, seeds 1-10 each) and agrees with the in-house sim-exact
oracle `pokezero.gen3_damage` (itself live-sim cross-checked in tests/test_gen3_damage.py).

## Z7.2 The architectural difference, and patch 34

The engine applied ALL of these as f32 multipliers on `choice.base_power`
(gen3/items.rs `item_modify_attack_being_used` / `item_modify_attack_against`,
gen3/abilities.rs, gen3/choice_effects.rs Explosion x2). Attaching a x1.5 to BP instead
of the stat is the same product but a DIFFERENT truncation locus: the sim floors
`modify(stat, 1.5)` before `*BP/D` and `/50`; the engine carried the half-point through
both. Proof-row arithmetic (recorded states, replayed this cycle):

- 1500221/13 — Choice Band Pidgeot L87 atk 189, Aerial Ace vs Unown def 153: sim
  `floor(189*1.5)=283 -> max 121`; engine `BP 60*1.5 -> max 123`. Observed 102 = the
  sim's exact min roll, below the engine's legal floor 104.
- 1500024/9 — Guts+psn Ursaring L81 atk 257 at -1, Return vs Slowbro def 228: sim
  `floor(floor(257/1.5)*1.5)=256 -> max 118`; engine 120. Observed 101 below the
  engine's legal floor 102.

`poke-engine-gen3-stat-modifier-flooring.patch` (slot 35) adds the fixed-point
stat-modifier chains to gen3/damage_calc.rs (`gen3_offensive_stat_modifiers` /
`gen3_defensive_stat_modifiers`, applied to the normal AND crit stat variants inside
`get_attacking_and_defending_stats`) and deletes the relocated BP arms. Guts now reads
the status at damage time — the engine's wake/thaw branch has already cured it and a
Sleep Talk callee still carries SLEEP, so patch 31's wake-turn special-casing is
subsumed, its pins still pass. Folded in, each sim-cited and pinned:

- **Light Ball physical doubling removed** (gen3 is SpA-only; the engine doubled Quick
  Attack too), **Thick Club special doubling removed**, **Dragon Scale's 1.1 dropped**
  (a bare evolution item in the sim; upstream's boost was invented).
- **Metal Powder corrected wholesale**: x2 Def, untransformed Ditto, physical only —
  upstream had /1.5 any-category behind a guard that tested the ATTACKER for being
  Ditto, so it also never fired. Pool-unreachable (randbats Ditto gets Leftovers);
  pinned like Fake Out rather than left to be rediscovered.
- **Explosion/Self-Destruct def-halving relocated** from BP x2 to
  `max(floor(def'/2), 1)` after the defense chain (odd defenses shift by a point;
  live-probed vs Skarmory: observed max 129 is exactly the halved-def prediction).
- **gen3 Plus/Minus added** (x1.5 SpA vs the opposing partner ability; live-probed
  Minun vs Plusle: observed {78..90} = the boosted roll set of max 90, control vs
  Slowbro unboosted). Both species are in the randbats pool, so the pair is reachable.
- **CONFIRMED LIVE BUG, fixed: burned Guts was 3x — and Z.4's record of it is WRONG.**
  Z.4 states patch 33 dropped the upstream Guts burn-compensation double ("Guts'
  burn-compensation double is dropped in favour of the pipeline's Guts-guarded burn
  halving"). It did not: patch 33 only touched damage_calc.rs, the double survived in
  abilities.rs, and
  once patch 33's Guts-guarded burn halving stopped halving, a burned Guts attacker
  multiplied BP by 1.5*2 with nothing compensating (probe: burned Guts Machamp Rock
  Slide dealt the 226-hp cap where paralyzed dealt 177). No prior residue row carried
  the signature (burned Guts users are rare); the stat pipeline makes burn just another
  Guts status (live probe: burned == paralyzed max 172 on the probe stats).

## Z7.3 The family label: 4/6 right, 2/6 wrong — and the wrong ones are two different things

Replay-first decomposition of the six `mechanism2_stat_modifier_flooring` rows:

| row | verdict | mechanism |
|---|---|---|
| 1500024/9, 1500126/13, 1500221/13, 1500267/77 | label RIGHT | stat-modifier flooring (Guts / Choice Band), arithmetic above |
| 1500076/96 | label WRONG — real engine defect, different mechanism | **type-effectiveness netting** (below) |
| 1500028/44 | label WRONG — not an engine defect | heal-cap structural matcher artifact: engine max = sim max = **19** at the recorded state (Vigoroth -1 atk Shadow Ball vs Mightyena — the -1 costs nothing, 183*2/3 is exact); the divergence is the engine world's avg-roll Leftovers heal capping `_to_full` while the observed roll's heal did not. Instrument lane, predicted NOT to clear, did not clear. |

1500076/96 (HP Grass into Shuckle, -1 SpD): the boost flooring is identical on both
sides (floor(506/1.5)=337 both); the point is lost at the TYPE step. **This is a defect
in the SHIPPED patch 33** (merged in the damage_calc lane, Appendix Z.4): its rewrite of
common_pkmn_damage_calc mis-transcribed the sim's type block. Found by this lane's
replay-first decomposition, fixed by patch 36; cycle history should read patch 33 as
correct on the modifier ORDER and truncation points but wrong on type-step NETTING. The sim NETS the
type exponent first (`typeMod = runEffectiveness` sums +1/-1 across the defender's
types) and applies only the net (data/mods/gen3/scripts.ts:88-104); patch 33's
transcription applied the two types as independent steps, so a net-neutral (0.5, 2)
pair with the resist FIRST in the type tuple floors an odd value: engine
`floor(29/2)*2 = 28` where the sim leaves 29. Observed 29 = the sim's exact 100% roll.
`poke-engine-gen3-type-effectiveness-netting.patch` (slot 36) nets the exponent in
common_pkmn_damage_calc; only (2x, 0.5x) pairs change, and only when the resisted type
leads the tuple and the pre-type value is odd.

## Z7.4 Pins, build chain and gates

`tests/test_engine_stat_modifier_fidelity.py`: 14 divergence pins + 4 controls, every
divergence pin's sim value transcribed from live probe runs (roll sets in the
docstrings; discriminating stats found by exact two-pipeline search where the loci
coincide on natural stats). On the 33-patch wheel (throwaway venv, built from the
pristine 33-patch tree): exactly the 14 divergence pins FAIL — each with the predicted
unpatched branch value — and the 4 controls pass. On the 35-patch build: 18/18.

| build | patches | fingerprint | gates |
|---|---|---|---|
| authoring baseline (#958 main) | 33 | `887a722d...` | probes 9/9; six family rows reproduce |
| pre-merge (+stat-flooring +netting) | 35 | `a987f3db...` | probes 9/9 (no expectation moved); pins 18/18; engine tree 17/17; pokezero-search 264/264; first differential (96-frame): 96 -> 86 |
| **post-merge (binding)** — main@#961 + slots 35-36 | 36 | `bdb6ad30f2722540c7b8e4fe1c63dde96627f890f1001c95d2677240a32eebb5` | fuzz=0 through both builders in the merged order (explosion-selfdestruct-gate applies BEFORE these two); probes 9/9 (no expectation moved by the netting patch); pins 18/18 + #961's typechange pins green; engine tree 17/17; pokezero-search 270/270 (incl. #959's gate tests) |

The fixture-refresh patch stays LAST and needed no extension on any of the three
builds: no upstream expectation moved.

## Z7.5 Prediction and the identity diff (registered BEFORE each run)

Per the standing rule (two prior fixes under-counted by trusting family labels), the
prediction was made by MECHANISM SIGNATURE: all 96 then-surviving outside-limits rows
regenerated at the 33-patch authoring baseline (96/96 reproduced) and marker-scanned on
the row payloads (`reports/c11_statfloor_prediction.json`). The pre-merge run scored it
on the 96-frame (5/5 arithmetic-verified rows cleared, 4/7 marker candidates cleared,
1500028/44 stayed as predicted, one extra netting clearance — 1500072/48, Silver Wind
into Metagross, missed by the scan because Silver Wind was absent from its move-type
table, named by identity diff and diagnosed by arithmetic: sim 85 vs engine 84).

After #959/#961 merged, the prediction was RESTATED against the new baseline before the
binding run (same artifact, `postmerge_restatement`): the same ten rows clear, 88 -> 78,
zero interaction with the eight rows #959/#961 cleared, nine remaining walk rows stay,
limit census unchanged. Binding run: 300 games, seeds 1500000-1500299, strict,
`--repros-per-game 40 --keep-repro 500`, fingerprint `bdb6ad30...`
(`reports/c11_statfloor_differential.json`). **Both sides of the identity diff carry
`repro_retention.repros_complete: true`** (baseline `reports/c10_kecleon_differential.json`
127/127; this run 117/117), so the diff is over the full divergent sets.

| | predicted | actual |
|---|---|---|
| the ten rows (5 verified + 4 scored candidates + 1500072/48) | all clear | **10/10 cleared** |
| 1500028/44 (heal-cap matcher artifact) | stays | **stayed** |
| #959/#961's eight cleared rows | stay absent (zero interaction) | **all absent** |
| remaining marker candidates (1500054/125, 1500174/43, 1500253/70) | stay, labels kept | **stayed** |
| capped-lethal walk (9 of 11 remain post-#959) | 9 present | **9/9 present** |
| limit:* population | 39, unchanged | **39, same classes** |
| new rows / class changes | 0 / 0 | **0 / 0** |
| outside-limits population | 88 -> 78 | **88 -> 78**; diverged 127 -> 117 on identical boundary counts (23335 measured) |

## Z7.6 Honest coverage statement

- The three surviving marker candidates (1500054/125, 1500174/43, 1500253/70) keep
  their labels: a marker on a boundary move does not make the modifier's fixed-point
  floor the row's divergence (the shift only exists when the fraction survives /50, and
  the row's divergence may sit elsewhere). They remain where c9 filed them.
- Of the eleven c9 capped-lethal walk rows, #959 already cleared two (1500074/57,
  1500188/33); nine remain on the post-#961 baseline. Two of those nine carry markers
  from this lane's scan (1500012/24 Choice Band, 1500242/60 netting); both remain
  divergent and keep their adjudications, but their recorded branch VALUES may have
  shifted by a point — the walk evidence for those two is stale-in-detail until the
  walker is rerun. The other seven are a clean zero-change control.
- Documented BP-side one-point residue this lane deliberately did NOT touch (the sim
  floors the BasePower event's result to an integer; the engine carries f32): pinch
  abilities (x1.5 on odd BP), Thick Fat (0.5 on odd BP), Facade, the weakened-condition
  halving. Same shape as mechanism 2 but a different event locus
  (`damage_calc_one_point_family` candidates; two of its rows cleared here via netting
  and Choice Band, the rest keep the label).
- Flash Fire's volatile is ModifyDamagePhase1 in gen3 (chains WITH screens, pre-+2);
  the engine still applies it as a trailing float multiplier after the type steps.
  Reachable in principle, no attributed row; left documented.
- Deep Sea Tooth/Scale (Clamperl) do not exist in the engine and Clamperl is not in the
  randbats species pool; not added. Beat Up's per-ally SpA override (gen3 moves.ts
  condition) is not modelled. Choice Specs (not a gen3 item) kept its arm, moved
  stat-side for uniformity.

## Z7.7 Artifacts

- `third_party/poke-engine-gen3-stat-modifier-flooring.patch` (34) and
  `third_party/poke-engine-gen3-type-effectiveness-netting.patch` (35), fixture-refresh
  still last
- `tests/test_engine_stat_modifier_fidelity.py` — 14 divergence pins + 4 controls
- `reports/c11_statfloor_prediction.json` — pre-registered on the 96-frame, restated
  pre-run on the 88-frame (`postmerge_restatement`), both scored in Z7.5
- `reports/c11_statfloor_differential.json` — the binding post-merge run
  (`repros_complete: true`, 117/117)

---

# Appendix Z8 — Cycle twelve: the walk relabels applied (78 -> 73), the 73 decomposed, and the final-wave backlog

Commit `d5389b1` (main at #962), fingerprint
`bdb6ad30f2722540c7b8e4fe1c63dde96627f890f1001c95d2677240a32eebb5`, 36 patches.
Fresh worktree/venv; standing probes **9/9** before any row was read. The 300-game
verification was NOT re-run: the binding measurement is #962's identity-exact
differential (`reports/c11_statfloor_differential.json`, `repros_complete: true`,
117/117), cited per the assignment. Predictions were committed SEPARATELY AND
FIRST (`reports/c12_predictions.json`, its own commit, per the #961 carry), derived
exclusively from committed artifacts — the c11 row data was not opened until the
prediction commit existed.

## Z8.1 Predictions, scored

| | predicted | actual |
|---|---|---|
| P1: the 78 = c9's 108 minus the 30 fix-lane rows, at IDENTITY level | exact set | **exact** — zero predicted-but-absent, zero present-but-unpredicted |
| P2: the 9 surviving walk rows present, classes unchanged | 9/9 | **9/9** |
| P3: re-walk at 36 patches reproduces c9 verdicts | 5 limit / 3 lne / 1 dc | **5/3/1 — and every MASS reproduced to the third decimal, including both Z7.6 marker rows** |
| P4: family table of the 73 | table | **holds by construction + content check** (below) |
| P5: clearance signatures for the backlog | — | future-scored |

Content integrity, checked not assumed: all 78 surviving rows are **byte-identical
in class and branch-miss content** between `c10_kecleon_differential.json` (88-frame
baseline) and the binding c11 run — #962 touched nothing it did not claim. The c9
family attributions therefore carry, and Z7.6's stale-in-detail caveat on
1500012/24 and 1500242/60 is **discharged by re-derivation** rather than argued:
`reports/c12_walk_rederivation.json` re-walks all nine rows against the c11 report
at this build. Same exits, same masses.

## Z8.2 RELABELS APPLIED: five rows move to `limit:roll_divergent_lethality` (78 -> 73)

The X-standard's sanctioned manual path, exercised for the first time. Each row
moves with TWO demonstrations attached — the c9 walk (33 patches) and this cycle's
re-derivation (36 patches), identical verdict and mass:

| row | mass | reproducing shape |
|---|---|---|
| 1500050/33 | 6.59% | tbolt m=63, two rolls hit -56; capped tick kills |
| 1500168/97 | 4.98% | Return roll 88% = -259; Blissey survives at 3 as observed |
| 1500219/62 | 5.86% | tbolt roll 100% = -28; tox tick + Perish kill reconstruct |
| 1500242/56 | 4.98% | HP Fire 87% = -71; capped tick -38 |
| 1500255/55 | 5.86% | DE roll 100% = -146, recoil re-scales, tick kills at 28 |

The three `limit_not_established` rows (1500012/24, 1500105/111, 1500242/60:
0.34/0.34/0.66%) STAY outside limits — reachable but sub-floor, exactly as X.3.3
prices them. 1500251/56 stays `damage_calc` (obs 116 still outside m=130's roll
set at 36 patches). The classifier is still deliberately not taught any of this;
the adjudicated census lives in `reports/c12_decomposition.json`.

**Adjudicated outside-limits residue: 73.**

## Z8.3 The 73, decomposed (identities per family in `reports/c12_decomposition.json`)

| lane | rows | families |
|---|---|---|
| matcher/instrument | **49** | heal-cap structural 10 (incl. 1500028/44 per Z7.3), accounting 9, boundary truncation 6, overreport window 6, overreport legal 3, I.2 attribution 3, Pain Split roll-inheritance 3, leech-cap 1, battle-end tie 1, Sleep Talk union 7 |
| limit-shape candidates | **11** | compound RDL shapes 7, limit_not_established 3, confusion fan 1 |
| engine follow-ups | **10** | fixed-damage post-hooks 4 (incl. Counter member), Counter/MC semantics 2, one-point family 2, recoil rounding 1, WaterAbsorb-on-RainDance 1 |
| world-construction drift | **3** | traced-ability 2, toxic-stage staleness 1 |

## Z8.4 The final-wave fix backlog

Full row identities and clearance-signature predictions in
`reports/c12_decomposition.json` `fix_backlog`; summary:

**Engine-side** — E1 BP-side one-point locus (2 rows; honesty: NEITHER carries a
pinch/ThickFat/Facade marker — the documented BP-side loci currently have ZERO
attributed rows, these two are 1-point gaps with mechanism OPEN filed as
candidates); E2 recoil rounding (1: trunc(0.33·d) vs round(d/3)); E3 Water Absorb
on Rain Dance (1); E4 fixed-damage post-hooks (4: no `damage_dealt` registration,
no contact hooks — structural, WILL scatter, predict by signature not by family);
E5 Counter/Mirror Coat beyond E4 (2, WHY open — mechanism work before any patch);
E6 Flash Fire Phase1 position (0 rows, documented).

**World/parser-side** — W1 traced-ability materialization (2; third member of the
Z6.1 "parser-visible fact with no world consumer" class); W2 toxic-stage
staleness across a Rest cure (1).

**Instrument-side** — I1 heal-cap shape relaxation (10), I2 matcher accounting +
legal-set availability (18), I3 roll-inherited exact components (4), I4 mapper
attribution (4), I5 boundary-truncation handling (6), I6 Sleep Talk callee union
(7): 49 rows that need matcher/mapper work or a formal limit-class adjudication —
they are measurement artifacts, not engine errors, and fixing the engine will
never clear them.

**Adjudication, not fixes** — A1 the 7 compound RDL shapes (walkable under the X
standard next), A2 the 3 sub-floor rows (terminal under the current standard),
A3 confusion fan (1).

## Z8.5 SWEEP-GATE STATEMENT on the 73

Attribution completeness: **every one of the 73 rows carries a named family, a
row-level basis, and a lane owner** — the gate condition (non-damage_calc,
non-documented-follow-up residue fully attributed) REMAINS MET, now against the
73. WHAT-level candidates are again foregrounded rather than buried: E1's two
rows and E5's two rows have open mechanisms; A1–A3's eleven are adjudication
candidates, not attributions of error.

For **"all currently known divergences fixed"** to hold, exactly this must land:
E1–E5 (10 rows, engine), W1–W2 (3 rows, world/parser), and for the 49
instrument rows either matcher/mapper fixes or a reviewed limit-class
adjudication per family (I1–I6) — plus the A1 walks if the residue definition
is to price those 7 rows honestly rather than leave them as shapes. E6 is
documented with zero attributed rows and does not block. Until then the sweep
gate stands on attribution, and the eighth-hold discipline continues: nothing
here launches the sweep.

## Z8.6 Artifacts

- `reports/c12_predictions.json` — pre-registered, SEPARATE FIRST COMMIT
- `reports/c12_walk_rederivation.json` — 9-row re-walk at 36 patches
- `reports/c12_decomposition.json` — relabels applied, the 73 with identities, the backlog

---

# Appendix Z9 — Traced ability and toxic staleness: the class closes its own backlog

Branch `scott/engine-world-trace-toxic-seeding`. **No vendored patch** — parser +
`engine_world` only. Engine 36 patches, fingerprint `bdb6ad30f2722540`, unchanged.
Prediction pre-registered and committed BEFORE implementation (`b65bcd4`).

## Z9.1 W1 — the traced ability, and a comment that was true of the wrong set

`|-ability|<mon>|<Ability>|[from] ability: Trace` publicly replaces the holder's ability, and
the belief engine has recorded it on every `-ability` line for as long as it has existed. The
world kept rebuilding from the **sampled set**, so it handed the engine `TRACE` and played the
mon without the copied ability at all.

Replay, 1500248/77: Flareon's Flamethrower into Porygon2 is absorbed
(`|-start|p1a: Porygon2|ability: Flash Fire`) and Showdown deals **no damage**. Every engine
branch deals **-111**.

This falsifies a claim written into `engine_world._SUPPORTED_VOLATILES`, which admitted
`flashfire` as a public-seedable volatile on the grounds that it is boost-only and so

> "never wrong, at worst incomplete if a sampled world lacked the ability, **which cannot
> happen for the mono-ability Gen 3 randbats carriers** nor for the request-known self side"

True of **native** Flash Fire carriers, and silently wrong about **acquired** ones: a Trace
user that copied Flash Fire is exactly a world lacking the ability. The comment reasoned over
the species list and the mechanism reasons over the battle. Fifth entry in this ledger's
running tally of a partial truth positioned where a reader looks for the whole one — and the
first where the partial truth was a *reachability* argument rather than a *provenance* one.

Fix seeds the **ability field only**. gen3 does not fire the copied ability's Start event on
acquisition (#962, patch 32), so nothing simulates an on-switch-in activation; the engine's
own hooks handle the ability when it is used. Pinned as a shape assertion, because "seed the
ability AND its activation" is the plausible over-implementation and in gen3 it is wrong.

### Z9.1.1 The first version of this fix shipped a regression, and the differential caught it

Worth recording in full, because the mistake is more instructive than the fix.

The first implementation read `belief.revealed_ability` — the field the belief engine already
populates on every `-ability` line. It cleared all three target rows, passed 13 pins, and the
300-game differential then reported **2 newly divergent rows** (1500009 steps 34 and 68),
against a prediction of zero.

Both were **Spikes damage the engine no longer applied**. Instrumenting the stamp showed
Gardevoir being handed `levitate`, `shellarmor` and `waterveil` — abilities belonging to the
OPPONENT's team. Gardevoir has Trace, so across the battle it had traced each of them in turn,
and `revealed_ability` is **persistent**: it holds the last ability the mon ever traced, not
the one it holds now. `levitate` grants Spikes immunity, so the seeding silently made a mon
immune to hazards it should have taken.

**A traced ability is transient and I used a persistent field for it** — precisely the error
class Z9.2 fixes on the other half of this same PR, where a toxic ramp survived a status it no
longer belonged to. Two instances in one change, one caught by a census and one introduced by
me while fixing the first.

The corrected version tracks `traced_ability` in the parser, set only on the
`[from] ability: Trace` discriminator and **cleared on switch-out** beside the
`live_type_override` clear that already lives there. After the correction all three target
seeds AND the regression seed return zero divergences.

Two durable notes:

1. **The persistent/transient distinction is the thing to check when seeding a world from a
   parser field**, not whether the field exists. Both halves of this PR are the same bug in
   opposite directions.
2. **The regression was invisible to every check except the full identity diff.** Target rows
   cleared, pins passed, targeted suites passed. Only "newly divergent: 2" against a
   pre-registered "0" caught it — which is the entire argument for pre-registering that
   number and for diffing identities rather than counts.

## Z9.2 W2 — the stale half was the PARSER, not the world

The backlog filed this as "world carried toxic_count=4". The world is innocent:
`_materialization_toxic_stage` is a pure `max(0, stage - 1)` of the parser's value — the
documented one-residual-ahead boundary idiom — so it can only report what it is given.

`_update_toxic_stage` reset the ramp on `-curestatus` and `-cureteam` **only**. But
`Pokemon.setStatus` replaces `statusState` wholesale (`sim/pokemon.ts:1733`:
`this.statusState = this.battle.initEffectState(...)`), the toxic counter lives in
`statusState.stage`, and Showdown emits **no cure line for the status it displaced**. So Rest
putting a badly-poisoned mon to sleep left the ramp standing at a stage that no longer
existed, and a later re-tox in the same stint was priced from it: a stage-5 tick of **-75**
where Showdown ticked a fresh stage-1 **-15**.

The world half is pinned **as unchanged**, deliberately: the offset was not the bug, and the
symptom points straight at it. Without that pin the next reader chasing a wrong toxic tick has
an inviting one-character "fix" available.

## Z9.3 The one-consumer audit — and the discriminator that makes it work

The parser-field-with-one-consumer heuristic predicted both previous finds, so it was run
across every `ShowdownReplayState` field. **Raw counts are misleading**: most single-consumer
fields resolve to `local_showdown.py`, which IS the world payload builder — one consumer there
means the field *does* reach the world. That is health, not a gap.

The real discriminator is **a single consumer that is not the world path**, or none at all.
Under that lens the heuristic reproduces its own track record: before this PR
`live_type_override`'s only consumer was `deep_line_audit.py` (the observation lane), and
`last_used_move` did not exist.

**Open candidates, listed not fixed.** Four counters are consumed exclusively inside
`showdown.py`'s own observation encoder and have **no world consumer**:

| field | why it is a candidate |
| --- | --- |
| `confusion_elapsed` | engine models confusion duration |
| `encore_elapsed` | engine models Encore duration (patch 29's ladder) |
| `wrap_trap_elapsed` | engine models partial-trap duration |
| `meanlook_trap` | engine models move-trapping |

What makes them more than a naming coincidence: `engine_world` carries
`approximate_hidden_duration_volatiles` and `approximate_partial_trap_turns` flags — **the
world APPROXIMATES exactly the durations the parser already derives from public protocol.**
That is the `last_used_move` shape precisely: a derived public fact on one side, an
approximation on the other, and no wire between them.

Stated as candidates, not findings: each needs its own replay before anyone claims a
divergence, and "the parser has a number the engine also has" is not yet evidence the two
disagree. `weather_upkeeps` and `stall_move_pending` also show zero external consumers but are
transient bookkeeping for other fields rather than world state.

## Z9.4 Differential: 4 cleared, 0 new — after the first attempt scored 4 cleared and 2 new

300 games, seeds 1500000-1500299, 36 patches, fingerprint unchanged. Baseline
`reports/c11_statfloor_differential.json` (78 outside-limits). Both sides
`repros_complete: true` (113/113 and 117/117), so the identity diff is over full populations.

| | predicted | first attempt | corrected |
| --- | --- | --- | --- |
| W1 rows (1500248/77, /78) | >= 2 | 2 | **2** |
| W2 rows (1500243/79) | exactly 1 | 1 | **2** |
| newly divergent | **0** | **2** | **0** |
| outside-limits | 78 -> 75 or better | 78 -> 75 | 78 -> **74** |

Frame bridge: 78 and 74 are the DIFFERENTIAL-OBSERVED counts. On the adjudicated frame
(Z8's 73, after #965's five ledger-level limit relabels, which do not overlap this batch's
four clears), the residue is now **69** — verified at row level by the #967 execution gate
(74 observed = 69 adjudicated + the 5 relabeled rows still counted by the classifier).

**The extra W2 row is 1500294/110** — Suicune woken from Rest-sleep and freshly Toxic'd,
ticking stage 1 (`-16` on a 270 HP mon). Same signature as the named row, same class
(`component_missing_in_engine:psn`); the backlog had simply attributed one instance of it and
not the other.

**This is a partial miss on my own Z6.4 rule and the distinction matters.** The rule says
magnitude mechanisms "clear in place, no scatter", and W2 was correctly typed as magnitude —
the extra row did NOT scatter into a matcher-accounting class, it sat in the same class as the
named row. What the rule does not do, and was never claimed to do, is predict how completely a
*decomposition* attributed a signature. Scatter and under-attribution are different failures:

* **scatter** (structural mechanisms) — rows land in classes that describe the matcher's
  confusion, so the family is spread across labels;
* **under-attribution** (any mechanism) — rows sit in the right class and simply were not
  all traced to the same cause.

The Z6.4 rule addresses the first. For the second, the defence is what caught it here:
identity-diff the full frame and attribute every extra, rather than checking the count.

## Z9.5 Artifacts

- `reports/c12_trace_toxic_prediction.md` — pre-registered in its own commit, before any code
- `reports/c12_trace_toxic_differential.json` — 113 rows, `repros_complete: true`
- pins: `tests/test_world_trace_and_toxic_seeding.py` (17), including
  `test_a_stale_trace_never_leaks_into_a_later_switch_in` for the self-inflicted regression
# Appendix Z10 — Batch E (patches 37-41): five mechanisms, two family labels inverted, and the recoil exactness trade

Fix-lane worktree, branch `scott/engine-gen3-damage-residue-wave`, based on main at #964
(36 patches, fingerprint `bdb6ad30...`, probes 9/9); #965/#966 merged mid-work
(docs/reports only — no engine or harness code), so the c12 decomposition's 73-frame and
this appendix's letter follow Z8. All ten backlog rows (E1-E5,
`reports/c12_decomposition.json` fix_backlog) regenerated from seed and replayed at the
base build before any design; every row reproduced.

## Z10.1 E1 — the family label was wrong for BOTH rows, two different ways

The brief's caution ("the documented BP-side loci have zero attributed rows") was
correct: neither row is the pinch/ThickFat/Facade locus.

- **1500123/79 is Flash Fire POSITION** (the E6 locus, previously zero attributed rows).
  Ninetales carried the FLASHFIRE volatile; gen3 hooks it on **ModifyDamagePhase1**
  (data/mods/gen4/abilities.ts flashfire condition, inherited by gen3): fixed-point
  floor BEFORE the +2/crit/STAB/type steps, chained with screens. The engine applied it
  as a trailing float after the type steps. Row arithmetic: t4 81 -> Phase1
  modify(81, 1.5) = 121 -> +2 -> STAB 184 -> Water resist 92 = sim max; trailing float
  gives 93; observed 78 = floor(92*0.85), the sim's exact minimum roll. **Patch 41**
  (`poke-engine-gen3-flashfire-phase1.patch`) moves the 1.5x into the stepwise pipeline
  between burn and weather; screens keep the trailing caller position
  (pool-unreachable per §U.4).
- **1500251/56 is NOT an engine defect.** At the recorded state the engine's Surf max is
  135 and the observed 116 is inside its legal roll set (`calculate_damage` verified
  live). The c12 walk basis's "obs 116 not in m=130 roll set" was a mis-derivation: 130
  was inferred from the branch value 120, which is the KILL-SPLIT non-lethal average,
  not floor(0.925*max). The actual divergence is the engine world evolving on that
  branch average so the toxic tick's HP cap diverges from the observed roll's chain —
  the I1 cap-state shape. Instrument lane; predicted to stay, stayed.

## Z10.2 E2/E3 — recoil exactness and weather-move targeting (patches 38, 37)

- **Recoil** (row 1500207/27): the sim computes
  `clampIntRange(floor(damageDealt * recoil[0]/recoil[1]), 1)` with Double-Edge at
  [1, 3] (gen4 mod, inherited; gen3 scripts.ts carries the same calcRecoilDamage). The
  engine truncated an f32 product (`0.33 * dealt`), one low whenever the dealt damage
  is a multiple of 3 at or above 30 (floor(75/3) = 25 vs trunc(24.75) = 24 — the row's
  exact shape), and dropped sub-1 recoil the sim clamps to 1. **Patch 38** maps the
  stored fraction back to the exact rational at the single application site. Live
  anchor: seed-4 of the probe run dealt 60 and recoiled 20 where the f32 path gives 19.
- **Weather targeting** (row 1500124/58): `ability_modify_attack_against` HAS a
  targeting guard whose own comment names Rain Dance — but `Choice::default()` gives
  the four weather moves `target: Opponent`, so they sailed past it and Water Absorb
  healed maxhp/4 off an opposing Rain Dance (Flash Fire would eat an opposing Sunny Day
  identically). **Patch 37** sets `target: User` on RAINDANCE/SUNNYDAY/HAIL/SANDSTORM
  in the MOVES table — the data the guard always assumed. Live probes: Rain Dance vs
  Water Absorb emits weather and nothing else; Surf into Water Absorb still absorbs.

## Z10.3 E4 — the fixed-damage path skipped the ENTIRE post-damage suite (patch 39)

Bigger than the family said. The fixed/level/fraction-damage arms applied their damage
directly inside `choice_special_effect`; `calculate_damage` returns the (0, 0)
placeholder for zero-BP moves, and `check_move_hit_or_miss` **zeroes percent_hit on
exactly that placeholder** — so the move never reached `run_move` at all: no
`set_damage_dealt` (Counter after Seismic Toss returned nothing, row 1500192/91), no
defender contact abilities (Rough Skin rows 1500103/76 and 1500274/15, Flame Body row
1500287/76), no secondaries, no Destiny Bond, no Endure — and **Super Fang's 90%
accuracy was never rolled** (its damage had already been applied before the miss
check). Super Fang's amount was also `hp - hp/2` (a ceiling, one high on odd hp; zero
at 1 hp) where the sim is `clampIntRange(floor(hp/2), 1)`.

**Night Shade dealt no damage at all** (Appendix K.3's documented inertness,
re-confirmed by the #966 depth-tactics probe: an EMPTY instruction list): it had no
`choice_special_effect` arm AND the zeroed-percent path would have eaten it anyway. Why
no differential ever flagged it: a sets.json scan shows Night Shade is in **zero gen3
randbats movepools**, and zero of the 244 diverged-row payloads across the c10/c11
full-retention artifacts mention it — unreachable in the measured format, reachable in
scenario/customgame contexts, so it is pinned like Fake Out rather than left to be
rediscovered. Sim probes: deals level (88 observed at L88), and answers `-immune`
against a Normal-type (Ghost-typed damage, gen3 chart).

**Patch 39**: `gen3_fixed_damage_amount` computes the sim's damageCallback value
(Seismic Toss / Night Shade level with their immunity probes, Super Fang
floor-with-min-1, Endeavor's difference, Counter / Mirror Coat 2x damage_dealt) and the
move flows through the ordinary damage path — substitute, endure, damage_dealt, contact
hooks, secondaries, Destiny Bond, accuracy — with no roll spread and no crit branch
(fixed damage neither rolls nor crits). The direct-apply arms are gone. One crate-test
expectation legitimately moved (`gen3_fixed_damage_fidelity`: Super Fang now carries
its real 10% miss branch); the sim rolls it too — 2 misses in 12 probe seeds.

## Z10.4 E5 — gen3 Counter/Mirror Coat treat Hidden Power as PHYSICAL (patch 40)

WHY closed from the sim's own clauses (data/mods/gen3/moves.ts:163-183, 407-427):
Counter's onDamage takes `category === 'Physical' || effect.id === 'hiddenpower'`;
Mirror Coat's takes `category === 'Special' && effect.id !== 'hiddenpower'`. The engine
recorded damage_dealt with Hidden Power's type-derived category, so a special-typed HP
was Mirror Coat-bounceable (rows 1500155/22, 1500264/40 — the sim's Mirror Coat did
nothing while the engine retaliated 2x) and invisible to Counter (the inverse defect,
live-probed: sim Counter returns exactly 2x against HP Grass). **Patch 40** records
Hidden Power as Physical in damage_dealt.

## Z10.5 Pins, build chain, gates

`tests/test_engine_fixed_damage_and_hooks_fidelity.py`: 10 divergence pins + 4
controls, sim values transcribed from live gen3customgame probe runs. On a 36-patch
wheel (throwaway venv, pristine tree): exactly the 10 divergence pins FAIL, each with
the predicted unpatched value (Night Shade's empty branch among them); 14/14 on the
41-patch build.

| build | patches | fingerprint | gates |
|---|---|---|---|
| baseline (main@#964) | 36 | `bdb6ad30...` | probes 9/9; all ten backlog rows reproduce |
| +batch E (37-41) | 41 | `3204c777dec347aa1df930cd509af7634fc1c3d66cddd3ab2c8fc16e91db80ce` | fuzz=0 through both builders; probes 9/9 (no expectation moved); pins 14/14 + statfloor 18/18 + guts/trace green; engine tree 17/17; pokezero-search 270/270 (one expectation updated, above) |

## Z10.6 Prediction and the identity diff (registered FIRST, separate commit)

`reports/c13_batch_e_prediction.json`, registered at the binding fingerprint BEFORE the
run; every marker row (a payload scan of all 117 c11 rows) was single-seed-resolved
before registration. Run: 300 games, seeds 1500000-1500299, strict,
`--repros-per-game 40 --keep-repro 500`
(`reports/c13_batch_e_differential.json`). **Both sides `repros_complete: true`**
(117/117 baseline, 107/107 post).

| | predicted | actual |
|---|---|---|
| will clear (12 named, single-seed verified) | 12 | **12/12** |
| stays (11 named, incl. 1500251/56 and both walk marker rows) | 11 | **11/11** |
| unpredicted clearances | 0 | **0** |
| class changes | 0 | **0** |
| capped-lethal walk rows (9 on the baseline) | 9 present | **9/9 present** |
| new rows | 0 | **2 — MISSED, diagnosed below** |
| population | 117 -> 105 | 117 -> **107** |

On the c12 73-frame: 11 of the 12 clears are outside-limits rows there (the 12th,
1500196/9, is `limit:`-labelled), so the decomposed residue moves **73 -> 62**
(adjudicated frame; observed outside-limits = adjudicated + the 5 #965-relabeled rows
the classifier still counts, per the Z9 bridge). **Cross-differential staleness note
(post-#967 merge):** this appendix's prediction listed 1500248/78 as a stay — TRUE at
this lane's pre-#967 build, where the world still lacked the traced ability, and STALE
on merged main, where #967's seeding clears it (with 1500248/77, 1500243/79 and
1500294/110). Both parents measured from the same c11 baseline with zero clearance
overlap; the merged-tree integration re-baseline below (Z10.8) states both frames.

**The two new rows** (1500129/46, 1500200/87) are the prediction's miss, and they are
the same instrument shape, surfaced by the recoil change: both are Double-Edge
boundaries at a faint edge (Swellow's toxic faint behind a Sludge Bomb; Kangaskhan at
3 hp dying to sandstorm) where the engine world evolves on the branch-average roll and
the recoil now derives EXACTLY from that average (floor(248/3) = 82) while the observed
chain derives from the observed roll (floor(250/3) = 83). The observed recoil is inside
the engine's per-roll legal set — the engine's recoil arithmetic is sim-exact, verified
by pins — but the downstream capped-lethal remainders differ by the point that the old
f32 truncation happened to align. They are I1 cap-state candidates (the family the c12
brief already owns), not engine defects; the honest cost of making recoil exact is that
this artifact can now express through recoil as it always could through heals and
toxic ticks.

## Z10.7 Coverage statement and artifacts

- The documented BP-side one-point loci (pinch abilities, Thick Fat, Facade,
  weakened-condition halving) now have **zero attributed rows and remain unfixed** —
  E1's two rows both decomposed elsewhere (Z10.1).
- 1500182/60 (Sleep-Talk-called Seismic Toss) kept its label: the marker is present but
  the divergence is not the post-hook suite; verified unchanged single-seed.
- Dragon Rage / SonicBoom remain unimplemented as fixed-damage arms (no MOVES wiring
  attempted; not in any c-series row; Night Shade was included because K.3 documented
  it and the helper covers it naturally).
- Artifacts: `reports/c13_batch_e_prediction.json` (pre-registered, first commit),
  `reports/c13_batch_e_differential.json` (binding run, 107/107),
  `tests/test_engine_fixed_damage_and_hooks_fidelity.py`, patches
  `poke-engine-gen3-{weather-move-targeting,recoil-rounding,fixed-damage-pipeline,counter-hiddenpower-category,flashfire-phase1}.patch` (37-41, fixture-refresh still last).

## Z10.8 Cycle-13 integration re-baseline (merged tree: batch E + #967)

Fresh 300-game strict run on the merged tree (engine fingerprint `3204c777...`
unchanged — #967 is Python-only; probes 9/9), pre-registered in
`reports/c13_rebaseline_prediction.json` (separate commit, before the run) and
identity-diffed against BOTH parents, all three artifacts `repros_complete: true`
(c13 batch E 107/107, c12 trace-toxic 113/113, this run 103/103).

| | predicted | actual |
|---|---|---|
| diverged | 103 (= 117 - 12 - 4 + 2) | **103** |
| identity | batch E's 107 minus #967's four clears, exactly | **exact match** |
| vs batch-E parent | -4 (#967's rows), nothing else | **exactly those 4, 0 new** |
| vs trace-toxic parent | -12 (batch E's rows), +2 (the I1 recoil rows) | **exactly those** |
| outside-limits | 65 observed = 58 adjudicated + 5 relabeled + 2 new-I1 | **65 / 58 / 38 limit** |
| class changes | 0 | **0** |
| capped-lethal walk | 9/9 present | **9/9** |

Zero clearance overlap between the parents held (the near-miss 1500248/78 is #967's
clear and this lane's stale stay, per the Z10.6 bridge). The merged-main residue
frame for the next cycle: **103 diverged = 58 adjudicated outside-limits + 5
relabeled + 2 I1-candidate new rows + 38 limit.**

---

# Appendix Z11 — Cycle fourteen: the A1 walks, the last engine row dissolves, and the instrument adjudication

Commit `05c4624` (main at #968), fingerprint `3204c777…`, 41 patches, probes 9/9.
Walked against the c13 re-baseline (`repros_complete: true`, 103/103). Predictions
pre-registered in `reports/c14_predictions.json` (separate first commit).

## Z11.1 The A1 walks: predicted 5/2/0/0, actual 2/1/0/4 — and both misses taught the walker something

| row | exit | mass |
|---|---|---|
| 1500200/47 | **limit** (RELABELED) | 5.86% |
| 1500217/112 | **limit** (RELABELED) | 11.72% |
| 1500055/60 | limit_not_established | 0.687% |
| 1500054/125, 1500108/67, 1500112/40, 1500182/25 | cannot_enumerate | — |

The four refusals share one structural cause, verified per row: a faint-capped
damage value on a boundary where `calculate_damage` prices the wrong move or
defender (Sleep Talk callee, or damage into a mid-turn switch-in), so no move
admits a determinate roll index — X.3.4's licence, quoted per row in
`reports/c14_walks.json`. The walker previously INFERRED a base there; the
first sample-of-4 produced two damage_calc verdicts that hand-replay exposed as
base artifacts, and the walker now refuses instead (predicted-vs-actual scored
honestly: the split prediction was wrong).

**Kill-split correction (Z10.1), propagated to every verdict.** The engine's
kill-split non-lethal arm applies `compare_health_with_damage_multiples`' 
conditional average, not `trunc(0.925·max)`. The walker now derives bases
through BOTH identities against `calculate_damage`'s maxes and uses the
engine's float-increment roll ladder. All 7 prior relabels re-verified under
the corrected derivation — identical verdicts and masses (1500050/33's recorded
base corrects 63 -> 66, mass unchanged at 6.59%). **1500251/56's c9 damage_calc
verdict is formally corrected**: true max 135 (kill-split identity, matching
Z10.1's live verification), observed 116 reachable at 0.215% mass ->
`limit_not_established`, I1 cap-state shape, instrument lane. The last
engine-side attributed row dissolves.

## Z11.2 The two Z10.8 I1-candidates: walked per the registered conditional

Replay showed both are roll-divergent capped-tick shapes on clean boundaries
(not heal-cap shapes) — P2's "formalize without a walk" was wrong, its
conditional fired. 1500129/46: reachable, 0.343% (two-roll conjunction), keeps
its label. 1500200/87: reachable at **4.40%** — above the floor, recorded as
candidate evidence with the demonstration attached, **NOT relabeled** (not a
member of the sanctioned adjudication set; pre-registered as such).

## Z11.3 The adjudicated frame after this cycle

65 observed outside-limits = **7 relabeled** (5 from c9 + 2 this cycle, every
demonstration re-verified at 41 patches) + **58 residue**, all attributed:

| lane | rows |
|---|---|
| instrument families (I1–I6) | 46 |
| limit-shape candidates (incl. both I1-walk rows and 1500251/56) | 12 |
| engine-side | **0** |
| world-side | **0** |

## Z11.4 Instrument adjudication (the #965-endorsed disjunction, exercised)

**Instrument families are adjudicated rather than fixed because the divergence
is in the comparison, not the engine.** Per family: I1 heal-cap/cap-state
shapes (11 incl. 1500028/44 and 1500251/56), I2 matcher accounting +
legal-set availability (17), I3 roll-inherited exact components (3), I4 mapper
attribution (4), I5 measurement-boundary truncation (6), I6 Sleep Talk callee
union (7) — 46 rows plus the 12 limit-shape candidates whose divergence is a
sampling comparison against a stochastic branch set. Each family's mechanism
is documented with row identities in `reports/c12_decomposition.json` (bases
carried, content byte-identity verified through four fix boundaries) and the
per-row walk evidence in `reports/c9_capped_lethal_walk.json` /
`reports/c12_walk_rederivation.json` / `reports/c14_walks.json`. For the
certification claim these families are the documented comparison-limit
classes: the engine is not wrong on them, and no engine change can clear them.

## Z11.5 GATE: held

Every row attributed; engine-side known divergences all fixed (#967/#968,
patches 34–41) or formally zero-row (E6 was fixed as patch 41 with its first
attributed row; no engine-side family retains an attributed row). The
certification sweep is authorized under its standing requirements.

---

# Appendix Z12 — The certification sweep: 10,000 games, and an honest FAIL

Run per the standing requirements: 8 x 1250 games, seeds `2000000 + k*100000`
(validated against the full protected-band registry — 26 bands — with this
sweep's own reservation filed BEFORE launch; the registry and reservations live
outside this repository); per-shard wheel REBUILD into eight separate venvs + in-shard
behavioral probes — all 8 gates 9/9 at `3204c777` (41 patches); `--checkpoint`
retention on every shard (which is what saved the run: the shards died once
mid-flight and resumed from checkpoint with zero loss, 1081-1149 games each at
the restart); class-rate table pre-registered in its own commit before launch
(`reports/c14_sweep_prediction.json`).

## Z12.1 Aggregate, against the pre-registration

| | predicted | observed |
|---|---|---|
| diverged | 3433, 95% [2832, 4161] | **3821 — inside the interval** |
| engine errors | 0 | **0** |
| coverage (measured fraction) | reported alongside | **0.9773** |
| retention | complete | **3821/3821, all 8 shards `repros_complete`** |

## Z12.2 VERDICT: **FAIL** under the pre-registered zero-unattributed criterion

3479 of 3821 rows attribute mechanically to documented families (readout
instrument validated 103/103 on the c13 population before any fresh row was
read — a validation that, per Z12.6, was structurally blind to absorb shapes):
1552 documented limit classes, 524 I1, 505 LS shapes, 240 I3, 215 I6, 165 I5,
166 I4, 112 I2. **342 rows do not attribute** (as amended by the #969 review,
Z12.6) — and per the pre-registration they are reported, not absorbed:

**Four NEW named mechanisms (183 rows as amended, samples replayed per family):**

| rows | mechanism |
|---|---|
| 56 | **recharge-turn residual gap** — on a `\|cant\|<mon>\|recharge` boundary (choice `none`), the engine's branch emits NO end-of-turn residuals; Showdown runs the block (Leftovers, psn tick observed-only at 100%) |
| 48 | **Truant loaf-phase drift** — Slaking boundaries where the engine's branch loafs when the sim attacked (engine=[] vs observed damage) or vice versa: the world's truant phase is mis-seeded |
| 45 | **absorb-ability heal fires through Protect** (26 uncapped + 19 capped `_to_full`, per Z12.6) — Showdown blocks the move entirely; the engine emits the Water/Volt Absorb quarter heal anyway (the #944 Protect-gate class, one dispatcher arm wider) |
| 34 | absorb-ability variants (15 incl. Sleep-Talk-callee misses + 19 heals applied on MISSED moves, per Z12.6) |

**161 rows PENDING replay-first triage** (next cycle's brief, sub-shapes
counted mechanically): magnitude pairs in the majority miss (74), engine-only
damage in the majority (30), accuracy-miss boundaries (7), other (50).

## Z12.3 Why 300-game cycles never saw these

The four new families sum to ~1.4% of divergences ≈ 0.014/game; a 300-game
census expects ~1-4 rows TOTAL across them, and drew ~0 by chance. The
certification sweep did exactly what a 33x scale-up is for: it priced the tail.
The residue program's method holds — every one of the 3517 attributed rows
lands in a family this ledger derived at 300-game scale, the pre-registered
total interval CONTAINED the observation, and the four new families arrived
with signatures crisp enough to name from samples. The gate that matters
stays honest: **zero UNEXPLAINED is the criterion, and 304 rows are currently
unexplained.** No certification is claimed.

## Z12.4 Artifacts

- `reports/c14_sweep_prediction.json` — pre-registered before launch (separate commit)
- `reports/c14_cert_sweep_readout.json` — full readout: per-class observed-vs-
  predicted with Wilson intervals, per-row attribution, the 304 with failure
  families
- `scripts/cert_sweep_readout.py` — the attribution instrument (validated
  103/103 pre-sweep)
- shard checkpoints/reports retained off-tree (scratchpad `cert/`), seeds
  2000000-2701249 consumed and filed in the registry

## Z12.5 Triage of the 161 (replay-first, bucket-sampled, pre-registered)

Predictions in `reports/c14_triage_predictions.json` (own commit, first); 4-6 rows
replayed per bucket before generalizing; full per-row table in
`reports/c14_cert_sweep_readout.json` (`triage_161`). Scored: the Slakoth
hypothesis was WRONG (zero Slakoth rows); "most fold into documented families or
the new mechanisms" held at 101/161.

| rows | verdict |
|---|---|
| 58 | LS structural-arm echo — observed shape lives in a sibling arm; majority complains on count (documented comparison family, sweep-scale variant) |
| 33 | LS crit-arm pairing echo — observed crit outcome paired against the non-crit majority (incl. crit-KO-ends-turn) |
| 6 + 2 | I4 attribution ties at multi-residual boundaries; I2 window accounting |
| 2 | absorb-family echoes (join the fix lane's population) |
| **28** | **CANDIDATE (WHAT): unresolved majority-magnitude gaps** |
| **12** | **CANDIDATE (WHAT): recoil basis when the hit broke a Substitute** |
| **11** | **CANDIDATE (WHAT): incapacitated-arm pricing (observed `|cant|` frz / fresh-slp not the engine majority)** |
| **9** | **CANDIDATE (WHAT): same-turn boost/status boundaries with ratio 0.70-0.96** |

**Updated ledger arithmetic for the 304**: 99 fold into documented comparison
families; 145 sit in the four NEW named mechanisms (recharge 56, Truant 48,
absorb-through-Protect 26, absorb variants 15 — fix lanes running); **60 remain
WHAT-level candidates in four named shapes, WHY open — these are the honest
unexplained count now**, and at least recoil-vs-Substitute looks engine-side,
i.e. the 161 DO add candidate fix items beyond the three running lanes.

**Re-sweep readiness**: a fresh certification sweep on unburned blocks needs
(1) the recharge/absorb/Truant fixes landed with identity-diff verification;
(2) WHY-level adjudication of the four candidate shapes (60 rows, replay
evidence retained in the shard checkpoints); (3) the readout instrument taught
this cycle's sweep-scale signatures so its mechanical coverage starts where
this one ended (96.9% raw, ~99% after triage). Seeds 2000000-2701249 are
consumed and registry-filed; the next sweep draws fresh blocks.

## Z12.6 Amendments from the #969 review

**Instrument correction (material):** the I1 `_to_full` signature fired BEFORE
any absorb check, hiding **38 rows** of the sweep's new absorb mechanisms inside
documented families — 19 absorb-through-Protect where the heal happened to CAP
(e.g. 2000377/9) and 19 absorb-on-MISSED-move (e.g. 2000014/84: the engine heals
in both accuracy arms while the sim rolled the miss). The absorb-shape exclusion
now precedes the I1 rule in `scripts/cert_sweep_readout.py`; deterministic
re-run: **UNATTRIBUTED 304 -> 342** (I1 559 -> 524, LS 481 -> 479, I5 166 ->
165), absorb-through-Protect **45** (26 uncapped + 19 capped), absorb variants
**34** (15 + 19 missed-move). **The 103/103 validation was structurally blind to
this signature: c13 contains zero absorb rows** — a validation set can only
validate the shapes it contains.

**Z12.3 denominator correction:** 143 (now 179) new-mechanism rows are 3.7%
(4.7%) **of divergences** but ~1.4% (1.8%) **of games**. And "drew ~0 by
chance" understates what the census could see: the measured per-game rate is
~1 in 60, i.e. a 300-game census expected ~5 such rows near-unclustered across
four mechanisms (P per-mechanism ~1.3-1.7%) — thin classes it could plausibly
miss or file singly, not a fluke of zero.

**Durability:** shard reports, checkpoints, probe logs and the readout are
archived at `~/workspace/agents/pokezero-agent/cert-sweep-20260730/`; their
sha256s are committed here so the evidence is pinned even off-tree:

```
a24719a0e9ce76d6fc630d93ffad2d945d03c43961c1b1947e2cc4d84a2044ac  cert_readout_v2.json
b4a0d2c2a182e693554b3fbe2b241126d367178319cd6d9f690e28e8d524dd59  cert_shard_0.json
6819192a8a520923d0e2bba21f3e000ed63f4d7fed4a23a47af20b2d5bcf1e89  cert_shard_0.jsonl
3259aa315252f505002b5570343f74b2f305ed00f30e1437ad19d081590e2c3a  cert_shard_1.json
4ff386cfe852d1978776ae00281e5d69c9c5f4635138d57b897efc254d6c81a1  cert_shard_1.jsonl
8003fbc80d86c7d07b2fbb9f39c521093d047ce98b2cb4c2c9497d0df4722d7b  cert_shard_2.json
e93269f8aae14fbed41c51c1715e9889b1c522fadd448c590dea410d4cae15e4  cert_shard_2.jsonl
442417c845c9046d7b7e4855dcdc4cba9d20f17a3acf76a447cddfa61882786c  cert_shard_3.json
638d460e51cd04ae6559a4d28c290587e070a444d832bcf43c3813268f6e870c  cert_shard_3.jsonl
486b635854c62e6eecf0eae54c98dcf53f49940404f14b96c9d2ab5994985d0d  cert_shard_4.json
ac7ff66f252df9f775c16e32fa6f32c3e3a66b13459547e1b2349f5de4752be9  cert_shard_4.jsonl
5a9e87fd694da86a7fa36a2c8db0f05b424485328769b1729aee7fca28073a95  cert_shard_5.json
ea207da8ad2ec84b5028fa0ae26ed6f45a37345909fa97c9435cddb4d9a7910a  cert_shard_5.jsonl
962ac6efd52cd4b689154ac083678ae8ba6e28c96465c1b963c4453cbcb953c4  cert_shard_6.json
502562b27ba284e9a9ee4fe9eb1094fafc4405ee31ccb4ec507d4aeb460c8099  cert_shard_6.jsonl
5c0235d98fbc91f72338e51001a4b82506899c46eae3cbdfdfe9d86c5e9e462d  cert_shard_7.json
7b3e3523ad3bb415c35d6bfa15ca2e88e36a56ab11d13b826314dd37ff4a23ce  cert_shard_7.jsonl
045afcbe4606a4a28e580bdb046228953a947335db38743ed2d03f7d53a67cee  shard_0_probes.log
045afcbe4606a4a28e580bdb046228953a947335db38743ed2d03f7d53a67cee  shard_1_probes.log
045afcbe4606a4a28e580bdb046228953a947335db38743ed2d03f7d53a67cee  shard_2_probes.log
045afcbe4606a4a28e580bdb046228953a947335db38743ed2d03f7d53a67cee  shard_3_probes.log
045afcbe4606a4a28e580bdb046228953a947335db38743ed2d03f7d53a67cee  shard_4_probes.log
045afcbe4606a4a28e580bdb046228953a947335db38743ed2d03f7d53a67cee  shard_5_probes.log
045afcbe4606a4a28e580bdb046228953a947335db38743ed2d03f7d53a67cee  shard_6_probes.log
045afcbe4606a4a28e580bdb046228953a947335db38743ed2d03f7d53a67cee  shard_7_probes.log
```

**Ordering standard (process):** the c14 walk RESULTS were committed one commit
before the walker CODE that produced them (caught in review). Standing order
going forward: instrument change first, artifacts it produces second — the same
separation the prediction-first rule already enforces on the other side.

---

# Appendix Z13 — Truant loaf phase: derived, probed, and 40 of 45

Branch `scott/engine-world-truant-phase`. Parser + `engine_world` only; engine unchanged
(41 patches, `3204c777dec347aa`). Prediction pre-registered in its own commit before any code.

## Z13.1 gen3 owns Truant, and the existing rule was a proxy for the wrong thing

`data/mods/gen3/abilities.ts` replaces base's volatile machinery (`onStart: undefined`) with a
free-running boolean:

```js
onSwitchIn(p) { p.truantTurn = this.turn !== 0; }
onResidualOrder: 27
onResidual(p) { p.truantTurn = !p.truantTurn; }   // EVERY turn end, unconditionally
```

`engine_search._truant_loaf_slots` used **"moved last round -> loafs now"**. That is a proxy
for the bit, not the bit: the first turn a holder is stopped by something OTHER than Truant
(sleep, paralysis, flinch, freeze, recharge, a switch) the two disagree, and **the parity
stays inverted for the rest of the stint**. One mechanism, tens of rows.

## Z13.2 Probe over derivation — and the probe corrected the derivation once

The composed derivation for a TRACED Truant (patch 32: a copied ability's Start event does not
fire, so `onSwitchIn` never runs for the tracer; `sim/pokemon.ts` leaves `truantTurn` false on
entry; `onResidual` flips regardless) predicts the tracer LOAFS on its first move turn — the
opposite of native. Three `gen3customgame` probes:

| scenario | first move turn | matches derivation? |
| --- | --- | --- |
| traced at **turn 0** (lead) | **ACTS** | no — derivation said loaf |
| traced **mid-battle** (turn 2) | **LOAFS** | yes |
| switch out, re-enter, **re-trace** | **LOAFS** again (parity resets) | yes |

The lead exception is the same missing **end-of-turn-0 residual** that broke the native lead:
there is no residual to flip before turn 1. Both cases collapse to one rule once that is
accounted for. The derivation was right about the mechanism and wrong about one boundary
condition, which is exactly what the probe-over-derivation rule exists to catch.

## Z13.3 What shipped, and what a flip-count model cannot do

* **Native** holders (`slakoth`/`slaking`, both mono-ability): seeded at switch-in with
  `turn != 0`, flipped per `|turn|` **from turn 2** (no end-of-turn-0 residual).
* **Traced** holders: **no derived seed.** Seeding `false` at acquisition and counting `|turn|`
  flips reproduces the probe in the common case and misses when the acquisition switch-in is a
  mid-turn replacement **after a faint** — there is no `|turn|` boundary between acquisition
  and the next move, so the count is short by one and the parity inverts. Measured: it cost a
  new divergence (2200291/41) while fixing three. Removing it made the same seed 3 -> 0 clean.
* **Anchors** establish and correct the phase from what the sim publishes:
  `|cant|...|ability: Truant` means loafing this turn, a holder's own `|move|` means acting.
  Exact, and needs no flip accounting.

**The general lesson: a derived counter and an observed fact are not interchangeable.** The
derivation is worth having — it explains the family — but where the protocol states the answer
outright, anchor on it. The four earlier formulations of this fix all fought because they were
deriving a quantity the sim was already publishing.

## Z13.4 CORRECTED — 54 cleared, 0 new, after the replacement guard

**The first version of this appendix understated its deviation, and the understatement was a
SCOPE artifact.** It reported 40 cleared / 2 new from a re-read of the 44 seeds my own
signature filter had selected. A reviewer re-read 116 seeds and found the position both
stronger and dirtier than claimed: **53 cleared** (including **9 rows #969 had already
attributed to documented families** — the under-attribution this program keeps rediscovering,
predicted in Z6.4 and corroborated again here) and **at least 5 newly divergent by identity**:

| newly divergent (pre-guard) | how it was missed |
| --- | --- |
| 2100487/48, 2300743/79 | disclosed by me |
| 2000583/38, 2101037/44, 2300534/47 | **invisible to a 44-seed scope** — their baseline rows belonged to OTHER families, so the seeds were never in my list |

All five share the shape I had already named as the open hole: `|cant| ability: Truant` with
the engine attacking. Base-code controls reproduce their baselines exactly, so they were
PR-caused, not pre-existing.

**The standing lesson is about the instrument, not the count.** A re-read scoped to the seeds
a fix was *aimed* at can measure CLEARANCE but **cannot bound NEW rows**, because a fix can
only create divergence in games it touches, and it touches games the aim never selected. What
I reported as "2 new" was a rate with an unmeasured tail, not a closed list — and I described
it as though it were closed. **Clearance and regression need different scopes: the aimed set
for one, everything the mechanism can reach for the other.**

### The guard, and the corrected numbers

The open sub-case was probed rather than argued: a holder entering as a **post-residual faint
replacement** (between `|upkeep|` and the next `|turn|`) LOAFS on its first move turn, where a
holder switched in as the turn's ACTION acts. Same seed value, opposite outcome; the only
difference is which side of the residual it entered on. Without the guard the `|turn|` flip
double-counts a residual the mon was never present for, and the parity inverts for the stint.

The trigger is parser-visible, which is why the guard is cheap: `|upkeep|` opens a window,
`|turn|` closes it, and a holder switching in inside it skips exactly one flip.

Re-read at the reviewer's **116-seed scope**, after the guard:

| | |
| --- | --- |
| baseline rows in scope | 161 |
| **CLEARED** | **54** |
| still divergent | 107 (other families in the same games) |
| **newly divergent** | **0** |
| **net** | **+54** |

All five of the reviewer's new rows are gone; for the three carrying base controls the
post-guard row set matches the baseline **exactly** (`[23,67]`, `[49]`, `[45]`), so the guard
neither introduced rows nor cost clears.

**Residual honesty:** 0 new at 116 seeds is still not a proof of 0 new. That scope is far
wider than the aimed set and includes the seeds that caught the original miss, but
previously-clean Slaking games remain unmeasured until the next full sweep. This is a much
better bound than before, not a closed one.

## Z13.5 Family correction, forwarded

Seed 2400315's rows are **not** loaf parity. They are `|cant|`-BOUNDARY residual drops
(`component_missing_in_engine:sandstorm` on both `|cant| Truant` and `|cant| recharge` lines):
engine and sim agree on who loafed and disagree on whether the end-of-turn block runs. The
recharge-residual-gap family is therefore **broader than `|cant| recharge`** — it is any
`|cant|` boundary. My signature filter mis-bucketed them, which also explains part of the
44-vs-48 gap.

---

# Appendix Z14 — WHY-adjudication of two sweep shapes (and a refused ownership)

Branch `scott/c15-why-adjudication`. Engine 41 patches, fingerprint `3204c777dec347aa`,
unchanged — this round is adjudication, not repair.

**Scope honesty up front: two of the four assigned shapes are adjudicated here.**
`CAND_unresolved_magnitude` (28 rows) and `CAND_same_turn_stat_event_gap` (9 rows) were NOT
sampled and carry no verdict from me. They are listed so the next reader knows the boundary of
this appendix rather than inferring coverage from its title.

## Z14.1 CAND_recoil_vs_substitute_basis (12 rows) — CORRECTED: world knowledge-limit

**The sim-chain half of this entry stands; the engine half was wrong and is withdrawn.**

### What holds

gen3 has no `substitute` override, so the condition resolves UP to **gen4's**
(`data/mods/gen4/moves.ts:1283`), which clamps `damage` to the sub's remaining HP, applies
recoil *inside* that handler from the clamped value, and returns `HIT_SUBSTITUTE === 0` — so
`move.totalDamage` is falsy and the outer recoil block (`gen3/scripts.ts:460`) never
double-applies. gen3 also owns `calcRecoilDamage` and **floors** where base rounds.

### What was FALSE

I wrote that "the engine computes recoil from the FULL damage". **It does not.** The engine's
substitute path sets `damage_dealt = min(calculated, substitute_health)` and recoil consumes
the clamped value — **semantically identical to gen4's handler**. The engine path is
**verified sim-exact**.

### The real mechanism, and how I got it wrong

`engine_world.py` seeds the sub with a **documented upper-bound approximation**:

```python
substitute_health = 0
if "substitute" in volatiles:
    # Public info does not carry the sub's remaining HP; a fresh sub costs
    # maxhp/4, so that is the documented upper-bound approximation.
    substitute_health = party[active_index].maxhp // 4
```

There is **no depletion tracking**. So the engine clamps correctly — to a sub that the world
believes is always full. Venusaur: `262 // 4 = 65`, and `floor(65/3) = 21` — **exactly** the
engine's observed recoil. The engine was reading its own seeded 65, not a full-damage figure.

My −21/−19 arithmetic **back-fitted**: I inferred "full damage ≈ 63" from `floor(x/3) = 21`
when **63, 64 and 65 all floor to 21**, and picked the value that fit my hypothesis instead of
reading the retained `DamageSubstitute` instruction, which states the number outright. The
artifact was in hand and I reasoned past it.

### Re-laned, and the prediction replaced

**Lane: world / knowledge limit**, not engine damage.

* An **engine patch clears ZERO rows** — there is nothing wrong with the engine path.
* **Depletion tracking** in the world clears the **inferable subset**: sub-breaking hits whose
  absorbed amounts can be reconstructed from public damage lines.
* The remainder is a **genuine limit shape**: Showdown does not publish how much a Substitute
  absorbed on a non-breaking hit, so the world cannot always know the remaining HP. Part of
  this family is not fixable without hidden information, and should be adjudicated as a limit
  rather than carried as an open defect.

## Z14.2 CAND_incapacitated_arm_pricing (11 rows) — CORRECTED: split, and one refusal withdrawn

The blanket refusal was **half right and is half withdrawn**. Three rows sampled; two
mechanisms, two lanes.

### Freeze rows — engine lane, mechanism corrected

I wrote that the engine "has no 80/20 freeze arm". **It has one; it did not fire.** The thaw
exclusion at `gen3/generate_instructions.rs:1580` reads:

```rust
&& ![Choices::HIDDENPOWER, Choices::WEATHERBALL].contains(&choice.move_id)
```

— only the **generic** `HIDDENPOWER`. Espeon's `HIDDENPOWERFIRE70` is a distinct enum variant,
so the engine treated it as a thawing fire hit, thawed the frozen mon, and let it act. gen3
says Hidden Power must not thaw.

The same file already enumerates **all 33 HP variants** thirty lines earlier (line 1374, for
`MoveCategory`), so the fix class is exactly the enumerate-every-variant pattern this ledger
established for the counter/mirrorcoat Hidden Power work — a known precedent, not a new design.

Confirmed on a second row: **2100471/18**, Chimecho's Hidden Power into a frozen Sableye, same
absent freeze arm, engine's Sableye acting for 185.

### The fresh-sleep row — refusal WITHDRAWN, world lane (mine)

**2000281/99 was my lane and I refused it wrongly.** The engine does not "apply the sleep and
then let the mon move" — it emits **zero instructions on every branch**, because its
rest-aware sleep clause saw a **benched Entei seeded with `rest_turns = 0`**. The world's
rest-provenance machinery (`restSleepAttempts` → `3 − k`) failed to mark a **BENCHED** public
Rest sleeper, so the clause could not engage.

That is a world-construction gap — precisely the lane I declined. **Sub-shape (b)
("same-turn status not gating the second mover") is withdrawn**: no sampled row exhibits it,
and it should be re-established from a row that does before anyone builds on it.

### Sizing the split

Of three sampled rows: **2 engine** (HP-variant thaw), **1 world** (benched Rest provenance).
A 3-row sample cannot size an 11-row family with confidence; it establishes that the family is
**mixed**, which is the part that matters for routing. The remaining 8 need the same
treatment before either lane commits to a clearance number.

## Z14.3 The rule, and the appendix that instantiated it

The original text of this section proposed: *a WHAT names where a divergence is VISIBLE; the
lane follows from the WHY, and the two are routinely different.* The rule survives. What did
not survive is my application of it — **this appendix was itself an instance**, and the split
falls exactly along one line:

| WHY derived from | outcome |
| --- | --- |
| **sim source, read directly** (gen4 substitute chain, `HIT_SUBSTITUTE`, gen3 `calcRecoilDamage`) | **held** |
| **arithmetic, without reading the retained branch instructions** (engine recoil basis, "no freeze arm", "applies the sleep then moves") | **all three refuted** |

I have a standing rule for this — replay before narrating, read the artifact rather than the
note — and I applied it to the SIM and skipped it on the ENGINE. The retained repros carried
`DamageSubstitute`, the branch dumps, and the instruction lists that state each of the three
answers outright. Every failed claim was an inference over a number when the number's
provenance was one field away.

**Sharpened rule: derivation is licensed for a source you are READING and never for a
component whose output you are only INFERRING.** Reading Showdown's chain to predict engine
behaviour is half a derivation; the other half is opening the engine's own recorded output.
The recoil row is the cleanest illustration — `floor(x/3) = 21` admits 63, 64 and 65, and the
artifact said 65.
---

# Appendix Z15 — Certification instrument correction before the second sweep

Review of the re-sweep preparation found two attribution rules that were too broad to support
the zero-unattributed gate. Both corrections are intentionally conservative: a row returns to a
named WHAT-level candidate unless its documented comparison basis is present in the same row.

* **I4 mapper ties:** an equal-magnitude, different-label tie now attributes only when it is in
  the majority arm. A minority-arm tie cannot explain an unexplained majority-arm mismatch.
* **LS structural-arm echoes:** a component-count mismatch now attributes only when a same-side
  sibling engine arm exactly carries the full observed nonempty component multiset. The former
  `s2000561/67` citation was stale: its sibling arms do not carry the observed hit. Unsupported
  count mismatches remain named-unattributed instead of being treated as branch-set accounting.

The associated re-sweep specification also records the corrected #972 map without treating the
adjudication as completed implementation: recoil-versus-Substitute requires the public
depletion/knowledge-limit world lane; the incapacitated-arm family requires the engine typed-HP
thaw exclusion, world bench-Rest provenance, and adjudication of the eight rows not sampled by
#972. Every remaining WHAT-level candidate pool must close to a fix, documented follow-up, or
proven comparison limit before launch.

The immutable c14 archive was hash-verified (17/17) and regenerated twice under the amended
instrument with byte-identical output. The c13 regression population remains **103/103 PASS**.
The honest c14 accounting is **324/3821 unattributed** (was 277): I4 drops 176 -> 166,
structural echoes drop 41 -> 0 because no archived row satisfies the new sibling proof, and six
minority-only I4 rows fall back to the generic pool. This changes generic fallback rows 54 ->
60 and named coverage **98.59% -> 98.43%**. The zero-unattributed certification gate is
unchanged; the returned rows are the required WHY-adjudication work, not a pass condition.

---

# Appendix Z16 — Engine patches 42-45: rebuild and identity-diff evidence

Four engine corrections identified by the certification sweep and independent review were
repaired behind separate prediction commits:

| patch | mechanism |
| --- | --- |
| 42 | A voluntary switch beside a recharge `cant` incorrectly skipped the full end-of-turn residual block. |
| 43 | Water/Volt Absorb conversion erased Protect and accuracy, so heals fired through Protect and on missed moves. |
| 44 | Typed Hidden Power variants bypassed the Gen 3 no-thaw exclusion and unconditionally thawed a frozen target. |
| 45 | Protect and full-HP absorb outcomes with identical state deltas collapsed distinct public histories; the Rust end-of-turn mirror also omitted the engine's force-switch condition. |

The patch-44 retained-population evidence remains the completed identity-diff instrument.
Both engine consumers then rebuilt all **45 patches with fuzz=0** at fingerprint
`0fd05522647f5af2670bd32630a5d994111d2758fef5f15b5e693bcd4fda3a10`;
the fixture refresh remains last. The builders clean their temporary trees, so `.orig` artifacts
are not retained. Behavioral probes passed 9/9, focused Python fidelity tests passed 29/29,
the public-invariant test passed 1/1, and the Rust release suite passed 271/271.

## Z16.1 Full retained-population re-read through patch 44

All 3,821 retained sweep rows were re-read through both the 43-patch ablation and the patch-44
build:

| build | matched | still divergent |
| --- | ---: | ---: |
| 43 patches | 153 | 3,668 |
| 44 patches | 158 | 3,663 |

Patch 44 changed exactly five identities: its four pre-registered pure rows plus the one
pre-registered mixed candidate (`2100295/88`). No other row changed verdict or class.

Patches 42-43 cleared every registered pure row (56/56 recharge, 75/75 absorb) and eleven of
the fifteen registered mixed candidates. The prediction's corrected absorb scan was still too
narrow: **eleven additional Protect/miss rows cleared**, and `2701065/24` retained a poison
divergence but changed class after the spurious absorb arm disappeared. Those misses are
reported rather than pocketed; the exact identities live in
`reports/c15_engine_patch_verification.json`.

Patch 45 closes composed edge cases found in review. A 64-seed live Showdown probe produced
57 Protect/ability activations and 7 misses in both Hydro Pump scenarios. The corrected native
mapper now preserves the exact 80/20 public-history split for Hydro Pump into Protect and into
a full-HP Water Absorb target, with zero lossy markers. Its detailed evidence is in
`reports/c15_engine_review_results.json`. The final fresh sweep, rather than the patch-44
archive, is the regression instrument for this correction.

## Z16.2 Fresh census regression bound

A strict 300-game census over seeds 1,500,000-1,500,299 on patch 44 measured 23,335
boundaries, retained all 103 divergences, and produced zero engine errors. Against the c13
re-baseline, the divergence identity set, class mapping, counters, and measured-boundary count
were identical; the canonical identity/class hash is
`f7f1c580146100d6b11531cd06fa158af1d7e2b10852a1285f35fe1d4f1b9d60`.

This closes the patches' targeted regression requirement, not the certification program. The
remaining retained rows still require their documented limit/follow-up dispositions, and the
binding gate remains a fresh 10,000-game re-sweep with zero unattributed rows.

---

# Appendix Z17 — C25 Toxic residual-stage recovery hardening

The prior parser repair correctly changed the replay residual formula to
`max(1, floor(maxhp / 16)) * stage`, matching Gen 3 Showdown and the Rust
engine's next-residual `toxic_count + 1` convention. Recovery review found
three provenance defects around that otherwise-correct formula: rounded `/100`
conditions were reverse-engineered into a hidden stage, a benched cure line
could clear the active counter, and a fainted active retained its counter until
a replacement arrived.

The replay now records whether each counter is known from public protocol and
whether each side's HP stream is exact, percentage-form, or unknown. A legacy
snapshot with active `tox` but no provenance, a rounded residual without a
public reset, or a condition-only residual cannot seed a world; materialization
fails closed rather than asserting `toxic_count = 0`. A switch/drag reset plus
the first percentage-form residual proves stage 1, while an exact 100-HP Pokemon
still uses its real six-HP Toxic unit. At ordinary request boundaries the
handoff subtracts one; at post-upkeep/pre-turn forced-switch boundaries the
just-applied stage passes through unchanged. Internal parser value 16 preserves
an already-saturated Showdown stage 15 across the ordinary-boundary subtraction,
while observation encoding remains capped at 15. Controls cover the exact-100
scenario, the 316-HP real capture with Leftovers-before-Toxic ordering,
repeated ticks, switch/drag, Baton Pass, Rest/status replacement, Natural Cure,
failed/reapplied Toxic, faint, and resume. This was the original parser/world
construction disposition. Independent review later found the separate engine
stage-cap defect documented in Z17.1; the final disposition includes both fixes.

## Z17.1 Independent-review amendment: saturation and production rendering

The original repair incorrectly assumed the engine capped `toxic_count + 1`.
It did not: a parser raw saturation sentinel of 16 became engine counter 15,
then produced an illegal stage-16 residual. The engine now caps the residual
stage at 15 and stores at most pre-tick counter 14. The construction bridge
therefore maps ordinary raw 16 and post-upkeep stage 15 to 14, while rejecting
unrepresentable values. A two-residual 640-HP engine advance proves both ticks
are 600 HP (15/16), with no stage-16 tick.

The production event renderer intentionally omits public cure lines for some
status operations. Rather than fabricate protocol, it carries a private ordered
active-status transition into leaf metadata. Render-to-evolve coverage proves
that Rest, Refresh, and Heal Bell clear Toxic stage/provenance without changing
their public output. Clean switch and drag entries also clear stage and active
provenance before deriving a possible Toxic re-entry.

## Z17.2 Post-upkeep poisoned replacement: the only public stage-zero proof

Gen 3 Showdown's `tox.onSwitchIn` sets `statusState.stage = 0`; its residual
handler increments that value before applying the first 1/16 tick. The normal
world bridge therefore correctly represents this first pending tick as engine
`toxic_count = 0`. Most active-Toxic zero snapshots remain ambiguous and fail
closed, but one public chronology is exact: a same-seat public active `|faint|`
is followed by `|upkeep|`, then that seat's non-Baton-Pass `|switch|` is a
faint replacement whose condition still says `tox`. That Pokemon missed the
preceding residual, so an ordinary request after the next `|turn|` may
materialize zero. The prior bare-post-upkeep rule was a proof forgery and is
not accepted.

The snapshot-carried faint latch is side-local, consumed by its replacement,
and cleared on next-turn truncation, malformed/duplicate replacement,
incompatible same-seat transition, or scenario reuse. Its resulting proof
lasts only until its first Toxic residual. Switch/drag replacement, active
status application or replacement, cure, faint, and the first Toxic residual
all clear or replace it; a legacy snapshot without the field stays fail-closed.
A synthetic post-upkeep `|drag|`
is deliberately rejected: Gen 3 resolves phazing in the move action before the
residual action emits `|upkeep|`. The proof is construction-only, so V2, V2.1,
and V2.2 observation identities remain byte-identical.

## Standing rule: arm contents come from replay, never from `branch_misses`

Any claim about what an engine arm *contains* must be derived with
`scripts/replay_residue.py`, which prints both slots' full component sets for
every branch. `branch_misses` answers "why did this branch fail", not "what did
this branch do": `engine_transition_differential.py:1893` loops
`for slot in ("p1","p2")` and breaks on the first failing slot, so a branch
contributes exactly one reason for one slot — and even that is the roll-scaled
subset (`:1934-1937`), or a set difference (`:1944-1948`), and is truncated at
`:1969` and `:2220`.

Reading it as a component set manufactures arms that are not there. It fails
*consistently* rather than randomly, so internal cross-checks do not catch it;
C92 was retracted in full and C84 and C89 were corrected for this. Two
classifier paths still compute on it and are flagged in
`reports/c94_method_retraction.json`: `scripts/family_bucket_audit.py:164-185`
(decisive, published an adjudication) and
`scripts/cert_sweep_readout.py:117-149` (currently fires no rows).

## Standing rule: never report divergence count without the skip and matched counters

`transitions_diverged` alone is not a safe metric. A change can lower it by
moving boundaries **out of evaluation** rather than by fixing them, and the
repro artifact will not show it — the dropped rows simply vanish from the repro
set with no new classes and no churn, which reads as a clean improvement.

Measured instance, the closed PR #1037 (`scott/i5-double-faint-replacement`):

| counter | baseline | patch | delta |
|---|---|---|---|
| `boundaries_measured` | 15224 | 15224 | 0 |
| `transition:matched` | 15184 | 15147 | **−37** |
| `transition:diverged` | 39 | 37 | −2 |
| `strict:lossy_render` | 11 | 50 | **+39** |
| `skip:strict_all_branches_lossy` | 1 | 40 | **+39** |

Both sides reconcile to 15224.

Baseline is `sweep_base.json` (`a723ea2e`, fingerprint `d9cab2b1…`) and test is
`sweep_i5.json` (fingerprint `ff749be8…`). The two trees differ only in `docs/`
and `reports/` plus the renderer patch itself, so this pair is single-variable
under rule 3 below.

Separately, `sweep_exact.json` (`cbcb6d27`) has the **same fingerprint** as
`sweep_base` and **byte-identical counters**, while running a *different*
matcher — pre- and post-#1032. That is a direct measurement that #1032 was
counter- and class-neutral, and it is also a demonstration of rule 3's blind
spot: `engine_build_fingerprint` does not hash `scripts/`, so the fingerprint
alone would not have told us if those two had disagreed.

The mechanism generalises to any renderer change: a predicate that makes
`segment()` return `None` sends that **branch** to `segmentation_failed`
(`rust/pokezero-search/src/events.rs:808-821`), which is not in
`_TELEMETRY_ONLY_LOSSY_MARKERS` (`scripts/engine_transition_differential.py:354-370`,
`:1872`).

There are then **two** outcomes, and only the first removes the boundary:

* **Every** branch lossy → `usable_branches == 0` → the boundary is skipped
  (`:1964-1968`, `:2190`) and leaves the denominator. This is what #1037 did.
* **Some** branches lossy → the boundary is still evaluated, on a rump branch
  set, which can turn a matched boundary into a *divergent* one. This is live on
  main today: `strict:lossy_render` is 11 against
  `skip:strict_all_branches_lossy` of 1, and row `19000093/51` is evaluated as
  divergent with 11 branches and ~10% of its mass surviving.

So a renderer change **can** buy a lower divergence count by rendering less —
not *always*, since skipping a matched boundary does not lower `diverged` at all
(#1037 skipped 39 and moved `diverged` by only −2), and partial lossiness can
raise it.

Rules:

1. Report `transition:matched`, `skip:*` and `transition:diverged` together.
   A fidelity claim requires `matched` to go **up** or hold — or, if it falls,
   a per-row account of which rows left `matched` and why. A fix that removes a
   *spurious* match correctly lowers it.
2. Check that `transition:matched + transition:diverged + engine_error +
   skip:strict_all_branches_lossy == boundaries_measured` on both sides. Summing
   all `skip:*` does **not** reconcile (it is 2,322 here); that one `skip:*`
   counter is the one in the identity, because it is the only one that fires
   *after* `boundaries_measured` has incremented.

   **C144 correction:** this rule shipped with `engine_error` omitted (three
   terms). It is FOUR-term. `engine_error` is also counted after
   `boundaries_measured` increments and also takes the boundary out of both
   `transition:*` tallies, so a run with a pyo3 panic in the matcher violates the
   three-term form exactly as it violates the two-term one. It has been 0 on
   every committed artifact, so the omission was never exercised — which is the
   same reason the two-term form survived as long as it did.
   `verdict_partition_failures()` in `scripts/engine_transition_differential.py`
   is now the mechanized form of this rule and
   `scripts/cert_sweep_readout.py` gates on it per shard. See
   `reports/c144_boundary_identity_correction.md`.

   **C142 addition:** the identity is now FIVE-term. The second outcome above —
   *some* branches lossy — is no longer adjudicated at all: it exits as
   `skip:rump_branch_set`, which like the other two is counted after
   `boundaries_measured` increments and takes the boundary out of both
   `transition:*` tallies. `verdict_partition_failures()` carries it, so the
   mechanized form of this rule is the thing to extend when a new post-measure
   exit is added — not the prose. `transition:diverged ==
   strict:diverged_on_full_branch_set` is the accompanying invariant: every
   reported divergence rests on 100 % of its enumerated mass. See
   `reports/c142_rump_branch_adjudication.md`.
3. Baseline and test must differ **only** by the patch under test, and you must
   be able to **show** that from the artifacts. Record `engine_fingerprint`
   *and* a hash of the counter-computing harness, then exhibit the
   baseline↔test diff.

   `source_commit` does not pin the code: `sweep_ph.json` and `sweep_ph2.json`
   share one and have different fingerprints and different counters, and any
   committed patch changes the field anyway.

   But `engine_fingerprint` alone is **not sufficient either**, and the gap is
   exactly the one this section is about. `build_inputs()`
   (`build_inputs()` in `scripts/engine_build_fingerprint.py`) hashes the patch stack,
   `rust/pokezero-search/src/**` and the Cargo inputs — **not `scripts/`**. So a
   change to `engine_transition_differential.py`, the file that computes these
   very counters, leaves the fingerprint untouched. `sweep_exact` and
   `sweep_base` are that case: identical fingerprint, different matcher.

   "Assert a clean tree" is *not* the rule, because it is unauditable —
   `_checkpoint_provenance()` in `engine_transition_differential.py` records only
   `source_commit`, `engine_fingerprint` and `image_commit`, so no artifact says
   whether the tree was dirty — and enforcing it literally would reject four of
   the five sweeps this section rests on, whose patches were deliberately
   uncommitted at measurement time.

   **Owed:** add `harness_sha256` and `worktree_dirty` to `_checkpoint_provenance()`. Until
   then rules 1 and 2 are checkable from the JSON and rule 3 is not.
4. The renderer keeps a replica of the engine's `end_of_turn_triggered`
   (`events.rs:406`, original at `gen3/generate_instructions.rs:4646` — engine line numbers drift, `third_party/poke-engine-src` is gitignored). Changing
   one without the other desynchronises them and shows up as mass
   `segmentation_failed`, not as a divergence.
