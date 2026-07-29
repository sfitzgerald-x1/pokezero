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

This rule exists because the inheritance-chain trap has produced **four** wrong
reads in this program, each of which reached a hand-off before being caught:

| Move / condition | The trap |
| --- | --- |
| Spikes | layer fractions live in the **gen4** mod, not base |
| burn | residual fraction changed at **gen6**, so base is wrong for gen3 |
| Flail / Reversal | gen3 has its **own** override; the gen4 ladder is not it |
| Thunder Wave | **gen6** declares a value BELOW gen3 in the chain |

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

* **`charge` (two-turn) moves are excluded** — Solar Beam, Fly, Dig, Sky Attack,
  Razor Wind, Skull Bash;
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

**But it is UNREACHABLE on the randbats distribution.** Across **1,682**
gen3 randbats variants, **70** carry Sleep Talk and **0** pair it with a
gen3-excluded move.

## E.3 Verdict

| Question | Answer |
| --- | --- |
| Is the engine's Sleep Talk call fan-out wrong? | **Yes, in source** — it omits the `charge` and `nosleeptalk` exclusions and the 0-PP rule |
| Do the branch weights diverge? | **No** — uniform 1/n is correct whenever the candidate sets agree |
| Is it reachable in gen3 randbats? | **No** — 0 of 1,682 variants |
| Does it explain the co-occurring residue rows? | **No** |

So this is a **latent** engine divergence: real, source-confirmed, empirically
demonstrated, and off-distribution. It belongs in the engine lane as low
priority (it would matter for `gen3customgame` or a pool change), and it is
**not** an acceptance blocker.

**Re-triage of the co-occurring rows.** 19 divergent rows in the 1350000-1350059
census involve Sleep Talk: 14 `roll_scaled_component`, 5 other. Their shape is
`observed=[('', -78)] engine=[]` — Showdown's call dealt damage and the engine's
branch has none. Since the fan-out and weights are correct and the exclusion bug
is unreachable, these are **not** a Sleep Talk defect; they are the called
move's damage failing to match, which is the same "engine is missing damage"
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

Appendix E's reachability numbers are in **expanded variants**; a re-checker
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
44/393 is the set row. **Same conclusion at every denominator: unreachable.**

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
`/private/tmp/claude-501/-Users-scott-workspace-agents-pokezero-agent/47b7c392-a7b8-43cf-b071-8a500f9bc9bf/scratchpad`

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
still has a pristine block, exactly as §5.2 reserved it. Every seed burned to
date remains below 2,000,000. If a future reader finds seeds at or above
2,000,000 in any report, an acceptance attempt happened that this ledger does
not record.

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
`/private/tmp/claude-501/-Users-scott-workspace-agents-pokezero-agent/47b7c392-a7b8-43cf-b071-8a500f9bc9bf/scratchpad`
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
