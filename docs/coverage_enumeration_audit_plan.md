# Coverage-enumeration encoder audit plan (deterministic species × ability × move sweep)

**Audience:** an executing agent (Codex) running against the pokezero checkout.
**Status:** plan for a follow-on audit that complements `docs/deep_line_audit_plan.md`.
**Relationship to the deep-line plan:** the deep-line plan chases **depth** (multi-turn,
multi-mechanic state-accumulation lines). This plan chases **breadth** — it guarantees that
**every** catalog atom (each species, each reachable ability, each move, and each item) is exercised through
the production observation encoder at least twice, deterministically, rather than relying on the
random generator to sample it. It directly closes the deep-line plan's stated blind spot
("Won't catch: bugs gated on rare *species/moves* absent from the sample") and the report's own
Remaining-Work note that "a future exhaustive component sweep would need generated fixtures or
direct source-derived starts for every catalog member."

## 0. Objective

Bypass `Teams.generate('gen3randombattle')`. Instead, **draft** the gen3 randbats catalog with a
"choose-and-remove" pass so that every species is visited **twice** across deterministic games —
the first visit using the species' **first reachable ability**, the second visit its **second
reachable ability** (when it has one). While drafting, **census every move** in the catalog; any
move not covered by the draft is force-included in an engineered follow-up game on a **valid
carrier**. Encode every constructed game and assert every public column against the omniscient
Showdown oracle. This is a **read-only audit**: find and flag; each confirmed bug becomes a
separate checkpoint-compatible fix PR that fast-follows the runs.

## 1. Why enumerate instead of sample

Random battles cover the *head* of the distribution fast and the *tail* never. Rare species,
second abilities, and low-roll moves may not appear at all in a bounded random sample, so a
species-, ability-, or move-specific mis-encoding (a wrong base-stat/type for a cosmetic forme, a
second-ability mis-attribution, a move-token vocab gap, an HP-type mis-encode) can sit latent
forever. Every encoder bug found so far was a *single-event* mis-encoding discovered by luck; the
one class incidence-driven sampling structurally under-covers is **the long tail of atoms**.
"Choose-and-remove" drafting turns "hope the sampler hits it" into "visit it by construction."

## 2. The universe — grounded numbers and exact access

Source of truth: **`{SHOWDOWN_ROOT}/data/random-battles/gen3/sets.json`** (read from the Showdown
checkout — it is **not** bundled in the engine repo). `SHOWDOWN_ROOT` =
`pokezero.local_showdown.DEFAULT_SHOWDOWN_ROOT` (env-overridable via `POKEZERO_SHOWDOWN_ROOT`).

| Quantity | Count | How to enumerate |
| --- | --- | --- |
| Species | **220** | `randbat_vocab.gen3_randbat_entities(root)["species"]` (sorted 220-tuple of ids) |
| Sets | 393 | `sets.json[species]["sets"]` |
| Distinct moves | **125** (112 normal + 13 `hiddenpower<type>`) | `gen3_randbat_entities(root)["moves"]` |
| Randbat abilities | **71** (**≤2 per species**) | union of `sets[*]["abilities"]` |
| Items | 13 | union of the variant item sets |
| Exact variants | **1,748** | complete `(species, role, level, four moves, ability, item)` source tuples |

`sets.json` shape: `{species_id: {"level": int, "sets": [{"role","movepool":[move_id…],
"abilities":[display_name…],"preferredTypes":[…]}]}}`. Species-level candidate/variant universe
is also available pre-parsed via `randbat.load_gen3_randbat_source_cached(root).universe_for(id)`.

### 2.1 Ability-ordering caveat (must read before defining "first/second ability")

`sets.json` `abilities` lists are **alphabetically sorted, not Pokédex slot order** (verified:
11 of 17 multi-ability sets disagree with slot order, e.g. `omastar` randbat
`["Shell Armor","Swift Swim"]` vs slots `["Swift Swim","Shell Armor"]`). True Pokédex slot-0/slot-1
ordering is **not available in the Python layer** (`dex.py`'s `SpeciesInfo` has no `abilities`
field); it exists only via node `Dex.forGen(3).species.get(id).abilities`.

Decision for this plan: **target the randbat-*reachable* ability set** — the abilities the
generator can actually roll — because that is exactly what training data contains. Define, per
species:

```
reachable_abilities(sp) = sorted(set(a for s in sets.json[sp]["sets"] for a in s["abilities"]))
# length 1 or 2 (gen3 has ≤2 abilities/species)
```

"First ability" = `reachable_abilities[0]`; "second ability" = `reachable_abilities[1]` when it
exists. The order is deterministic (alphabetical); ordering *semantics* do not matter for
coverage — only that **both** reachable abilities are visited. Note that for **55 species the
randbat ability set is a strict subset** of the Pokédex abilities (e.g. `raticate` → `["Guts"]`
only), so those species are effectively single-ability *in randbats* and only get one ability
across their two visits. Do **not** construct a Pokédex ability the generator never rolls — that
would be encoding a state the model never sees in training. (Optional stretch lane §7.3 covers the
defensive case via node slot data; it is out of the default scope.)

### 2.2 Move universe caveats

- The 125-move census is the **movepool union**. Every movepool move has ≥1 legal carrier
  (§2.3), so the census is 100% reachable by construction.
- `struggle`, `recharge`, and bare `hiddenpower` are **vocab-only** (in the encoder vocabulary but
  in **no** species movepool). They cannot be "missing from a movepool" and cannot be placed via
  team construction; they are exercised mechanically in a dedicated mini-lane (§7.2).
- Happiness aliases (`return<n>`/`frustration<n>`) collapse to base `move:return`/`move:frustration`;
  cover the base tokens.

### 2.3 Move → valid carrier (for the gap-fill)

No helper exists; build the inverse from movepools:

```
inv = defaultdict(set)
for sp, info in sets.json.items():
    for s in info["sets"]:
        for m in s["movepool"]:
            inv[m].add(sp)
# inv[move] = every species that can legally carry `move` in randbats
```

Singleton-carrier moves (only one legal carrier — the tight gap-fill constraints) include
`volttackle→pikachu`, `flail→dodrio`, `bonemerang→marowak`, `lovelykiss→jynx`,
`spiderweb→ariados`, `charm→togetic`, `razorleaf→sunflora`, `meanlook`/`perishsong→misdreavus`,
`mudshot→kingler`. For "can species X carry move M at all," **raw movepool membership is the
correct and sufficient test**; `_valid_gen3_move_combo` (full 4-move-set legality) only matters if
you want a *generator-legal* 4-move set, which Custom Game does **not** require.

## 3. The draft — every species visited twice, both reachable abilities

### 3.1 Construction primitives (all already exist — reuse verbatim)

- **Set object:** `pokezero.showdown_fixture.FixturePokemon(species, moves, ability=…, item=…,
  level=…, nature=…, gender=…, evs=…, ivs=…)` — the **ability is an explicit field** (packed as a
  display name), the **moveset is an explicit field**. Custom Game does not enforce randbat
  legality, so any legal ability/moveset for the species is honored.
- **Team packing:** `pack_team(Sequence[FixturePokemon]) -> str` (accepts a 1-tuple → **1v1 is
  valid**).
- **Start override:** `pokezero.env.BattleStartOverride(player_teams={"p1": packed, "p2": packed},
  format_id="gen3customgame")` — the only gen3 format that accepts arbitrary curated teams with no
  set-gen and no Team Preview.
- **Driver:** `LocalShowdownEnv.reset_with_start_override(seed=…, start_override=…)`.
- Convenience: `golden_corpus_scenarios._mon(...)` and `_scenario_override(spec)` are the existing
  `ScenarioSpec → BattleStartOverride` bridge; `_audit_scenario` (in `scripts/deep_line_audit.py`)
  is a near-drop-in deterministic driver template.

### 3.2 Why 1v1 is the coverage unit

A single 1v1 game encodes **both** mons from **both** perspectives at the turn-1 decision
boundary: mon on p1 is encoded as **self** in `env.observe("p1")` (exact item/ability/stat token
surface, per #767) **and** as **opponent** in `env.observe("p2")` (belief/candidate token surface),
simultaneously. So one game covers the self *and* opponent token surfaces for both drafted mons —
no seat-swapping needed. All **static/identity columns** (species, type, base-stats, ability,
moves, item, level) are fully populated at turn 1, which is exactly what the atom sweep targets.
Dynamic columns (status, boosts, toxic stage) are the deep-line plan's job, not this one.

### 3.3 The two passes (choose-and-remove)

```
species = gen3_randbat_entities(root)["species"]          # 220
for pass_idx, ability_slot in ((A, 0), (B, 1)):
    pool = list(species)                                  # fresh pool each pass
    while pool:
        x = pool.pop(); y = pool.pop() if pool else pool_wrap(x)   # choose-and-remove, pair
        for mon in (x, y):
            ab = reachable_abilities(mon)[min(ability_slot, len-1)] # slot-1 → slot-0 for single-ability
            moves = draft_moveset(mon, pass_idx)          # §4
        run_1v1(x, y, seed=deterministic(pass_idx, x, y)) # encode + assert both perspectives
```

- **Pass A** drafts every species with its **first** reachable ability; **Pass B** with its
  **second** (single-ability species repeat their only ability — flag them
  `ability_coverage=complete_after_A`, but still run the second visit for the extra move/state
  coverage the second draft provides).
- Odd pool tail (220 is even, but be robust): if a pass ends with one unpaired species, pair it
  against any already-drafted species (a re-use, not a coverage gap).
- Result: **every (species, reachable-ability) pair encoded ≥ once**, every species encoded on
  both seats, ~220 games total.

## 4. The move census and moveset assignment

Goal: exercise all 125 movepool moves and all 13 source items through the encoder. Each mon has **2 visits × 4 move slots
= 8 move-slots**; the catalog has 220 mons × 8 = 1 760 slots for 125 moves — ample capacity, so
the draft can cover nearly all 125 and gap-fill is a small safety net.

**Draft moveset assignment (`draft_moveset`) — deterministic global greedy set-cover:**
iterate species in the draft order; for each of a mon's 2 visits, choose a valid source variant
for the pass's target ability which maximizes globally-**uncovered** moves, then uncovered items,
with a stable variant-id tiebreak. This preserves an actual randbat variant while deterministically
maximizing draft-time move/item coverage. Track both
`move_first_covered_by[move] = game_id` and `item_first_covered_by[item] = game_id`.

**Coverage ledger:** after the draft, compute covered and missing sets for both moves and items.
Report a first-covering game for every atom. Items are a hard target, not a reporting-only
statistic: both `missing_moves` and `missing_items` must be empty in a completed audit.

## 5. The gap-fill — the "second draft" for missing moves

For each remaining uncovered move or item (deterministic order):

1. Pick a valid **source variant** which covers the most remaining moves/items, with a stable
   variant-id tiebreak. A missing item must select a variant carrying that exact source item.
2. Run that valid variant in a 1v1 game (opponent = any valid source variant), encode, assert,
   and mark every move/item on both fixture mons covered.

Loop until both missing sets are empty. Because every movepool move and source item has at least
one source-variant carrier, 100% coverage is achievable; if either atom kind remains uncovered,
that is itself a finding (a broken carrier relation). Log every engineered game and its target
atoms — **no silent coverage caps** (state exactly what, if anything, could not be covered and why).

## 6. What each boundary asserts (reuse the merged oracle)

Reuse `scripts/oracle_differential.py` (merged as #765, on `origin/main`) — it is
**format-agnostic** (reads `env.snapshot().bridge_snapshot["battle"]`), so it works unchanged on
`gen3customgame`. Plug the same asserter into the coverage-driver by swapping **only** the reset
(`reset_with_start_override`) and the action selection (turn-1 only, or the §7.1 move-use script):

- `audit_side` — per mon, both sides: `hp_fraction`, `active`/`present`/`fainted`, `level`,
  species `base_stat/{hp,atk,def,spa,spd,spe}` (the **atom sweep's core** — catches dex/forme/
  base-stat errors for rare species: Deoxys formes, cosmetic Unown, Castform), self `actual_stat/*`,
  `boost/*`, `toxic_stage`, `sleep_turns`, `status` categorical.
- `audit_field` — turn, hazards (spikes), screens, weather.
- `audit_legal_mask` — the 9-action projection vs alive-bench × PP>0 (Struggle exception).
- `audit_belief_partc` — opponent candidate-universe: true moveset stays in-universe, candidate
  set monotone non-increasing (#757 over-pruning guard).
- `run_invariants` — bounds + consistency (`hp∈[0,1]`, `boost∈[−1,1]`, `tox⇒stage≥1`,
  `fainted⇒hp0`, `alive⇒hp>0 ∧ level>0`).

**Ability + move token assertions (add to the driver):**

- **Ability:** the self token encodes the drafted ability (self-known ability, per #767); the
  opponent ability-candidate bucket **contains** the true ability and narrows correctly. The
  second-ability pass is what makes a *second-ability mis-attribution* visible.
- **Move tokens:** each drafted move encodes to the **correct vocab row** (catches move-vocab
  gaps, HP-type mis-encode per #756/#758, Return/Frustration aliasing). Because every atom is
  **reachable** (reachable abilities + movepool moves + `sets.json` levels), the encoder should
  **never** emit an OOV/placeholder — if it does for a reachable atom, that is a finding.

Optionally also plug `deep_line_audit`'s richer per-boundary units (`audit_live_decision`,
`_compare`, `AuditFinding`/`DeepLineAuditReport`) for the self-known-fact, transform-identity, and
snapshot-roundtrip lanes. Note: `scripts/deep_line_audit.py` + `src/pokezero/deep_line_audit.py`
live on `origin/scott/deep-line-audit` (not merged) — retrieve with
`git show origin/scott/deep-line-audit:scripts/deep_line_audit.py`.

## 7. Extensions (bounded, opt-in)

### 7.1 Move-use scripting (exercise the reveal → belief-narrowing path)
Static move-slot tokens are covered at turn 1, but the **move-reveal on the opponent** (which
narrows the opponent's candidate set and is where #756-class bugs live) is dynamic. Optionally
script each drafted mon to **use each of its moves once** via `ScriptedPreferencePolicy` (per-turn
preference lists; consumes no RNG). This exercises PP decrement, move reveal, and opponent-belief
narrowing for every atom — a strict superset of turn-1 static coverage.

### 7.2 Universal-move mini-lane (`struggle`, `recharge`, `hiddenpower`)
These three vocab-only tokens can't be placed via movepool. Exercise them mechanically:
`struggle` — a mon whose moves are all PP-depleted (or a single-move set driven to 0 PP);
`recharge` — a set including a recharge move if any carrier exists (else document as
mechanically-only reachable); bare `hiddenpower` — confirm whether any HP move ever encodes to the
bare token vs a typed `hiddenpower<type>` row. Assert each surfaces the correct vocab row.

### 7.3 Defensive Pokédex-ability stretch lane (out of default scope)
If desired, additionally cover the Pokédex abilities the generator never rolls (the 55 restricted
species) using node slot data — purely a *defensive* encoder check for states the model does not
see in training. Keep separate from the reachable-set coverage report so the two are not conflated.

### 7.4 Exact-variant lane (optional, stronger tuple guarantee)

The atom sweep proves every individual source fact reaches the encoder, but it
does not prove every exact **combination** of role, moves, ability, and item.
Run the opt-in `--exact-variants` mode when that stronger guarantee is needed:

```sh
POKEZERO_SHOWDOWN_ROOT="$POKEZERO_SHOWDOWN_ROOT" \
POKEZERO_AUDIT_IMAGE_DIGEST="$IMAGE_DIGEST" \
  uv run python scripts/coverage_enumeration_audit.py \
    --exact-variants \
    --observation-schema v3 \
    --json /tmp/pokezero-exact-variant-audit.json \
    --coverage-json /tmp/pokezero-exact-variant-ledger.json
```

This mode bypasses greedy drafting and pairs every source variant exactly once
into deterministic 1v1 fixtures. At the current source hash, 1,748 variants
produce 874 fixtures. Because each fixture is audited from both seats, every
variant is checked through its self-known encodable facts (including level) and
as the opponent's true surviving belief candidate. `role` is a source-selection
constraint rather than a direct observation feature, but remains part of the
candidate `variant_id` identity. The ledger adds `variants`, per-variant first
coverage, and an explicit `uncovered.variants` set; completion requires it to
be empty alongside the atom sets. It remains static tuple coverage, not an
exhaustive move-use or multi-turn interaction sweep.

### 7.5 Bounded exact-variant depth run (planned, do not launch before fixes land)

After all encoder and belief fixes under test have landed in one fresh image,
run a bounded dynamic extension over the same exact-variant universe. This is
not a replacement for random deep-line games: it is a deterministic way to
exercise post-action request, belief-pruning, PP, action-token, and
perspective surfaces for every source tuple.

**Preflight contract:** rebuild Showdown, record the source hash, verify the
image includes every intended fix, and reserve a new output directory. Do not
reuse the 874-fixture static audit evidence as proof of this dynamic run. The
dynamic run must pin **observation schema v3** and stamp that schema in every
summary, ledger, and failure artifact: its no-op checks are invalid under the
current checkpoint-free v2.2 default. If the audit CLI cannot explicitly select
v3, it is a preflight failure, not permission to fall back to the default.

Use eight independent shard workers, each with a distinct summary/ledger path
and the shared failure directory below:

```sh
for shard in 0 1 2 3 4 5 6 7; do
  POKEZERO_SHOWDOWN_ROOT="$POKEZERO_SHOWDOWN_ROOT" \
    uv run python scripts/coverage_enumeration_audit.py \
      --exact-variants \
      --observation-schema v3 \
      --depth-rounds 8 \
      --shard "$shard/8" \
      --no-universal-lane \
      --failure-dir "$OUT/failures" \
      --json "$OUT/audit-$shard.json" \
      --coverage-json "$OUT/ledger-$shard.json" &
done
wait
```

`--depth-rounds` is deliberately bounded: it scripts legal moves in source
order, then after every action re-runs the live differential oracle,
perspective check, action-token identity check, and true-variant-survival
check. It also records the post-action protocol-tag census in the shard audit
JSON. That census is evidence for the v3 silent-noop sweep's adjudication
table, not a replacement for its exhaustive static reachability/handler diff:
a deterministic 1v1 path can miss event subtypes and six-mon interactions. A
terminal game simply stops early; it is reported as bounded depth, not
misrepresented as a complete move-use proof.

The runner writes no protocol trace for a successful fixture. If a fixture has
a finding or execution exception, it writes exactly one reproducible JSON file
under `$OUT/failures/`, containing the source tuple, seed, executed moves,
the full protocol trace, terminal state, and only that fixture's findings.
Workers continue after a failed fixture in depth mode, so one bad variant does
not discard coverage of the remaining source universe. Inspect the failure
directory first; aggregate summaries are for counts and ledger completion.

After all workers finish, merge the eight coverage ledgers and require an empty
`uncovered.variants` set:

```sh
uv run python scripts/merge_coverage_ledgers.py \
  --input "$OUT"/ledger-{0,1,2,3,4,5,6,7}.json \
  --output "$OUT/ledger-merged.json"
```

Any nonzero worker exit, nonempty failure directory, or nonempty merged
uncovered set is a failed audit and must be triaged before treating the run as
clean. In exact-variant mode, the merged ledger additionally requires every
planned fixture to complete; atom coverage alone cannot hide a dropped tuple.

### 7.6 Curated party and silent-noop interaction lane (planned, separate from tuple depth)

The exact-variant depth lane is intentionally 1v1: it exhaustively validates
every source tuple, but cannot create a party lifecycle. Do **not** inflate it
into arbitrary 3v3 permutations. Reuse the existing scripted interaction
registry in `golden_corpus_scenarios.interaction_registry_specs()` as the
party-mode lane, with its encoded assertions and live protocol evidence
recorded in `docs/validated_interactions.md` and
`docs/protocol_coverage_matrix.md`.

That registry already covers the high-value party mechanics: a 3v2
Spikes-stack-and-switch sequence, Intimidate/Trace on entry, drag reset,
Baton Pass transfer, and the existing Trick/Knock-Off, berry, Wish, RestTalk,
and recharge scenarios. Re-running those fixtures is a local, minutes-scale
regression lane; it is not part of the 874-fixture parallel job and does not
change the tuple ledger.

Before calling the interaction lane complete, add targeted scenarios or
assertions for the remaining silent-noop-prone boundaries:

| Scenario / assertion | Required shape | Required observation contract |
| --- | --- | --- |
| Natural Cure switch lifecycle | Status a Natural Cure mon, voluntarily switch it out, then inspect both public views and the re-entry state. | Status and toxic-stage state clear; any sleep-clause holder derived from the cured mon clears too. |
| Second-sleep / Yawn block | Establish one non-Rest opponent sleeper, then attempt a second sleep and a pending Yawn resolution under the clause. | The action is retained as a failed action rather than conflated with no action; v3 fail and sleep-clause fields remain side-relative. |
| Toxic into an already-statused target | Apply Toxic, then attempt Toxic again while the target still has a nonvolatile status. | The second action emits/records the failure path and cannot look identical to a skipped turn. |
| Protect lifetime | Use Protect on one turn and inspect the following decision boundary. | Its same-turn evidence is retained in the transition surface, with no stale permanent volatile on the next turn. |

The v3 silent-noop sweep remains the authority for reachability, handler
coverage, and accepted-loss decisions. This lane supplies live, party-shaped
repros for its adjudicated rows; it must not use absence from the finite
scenario set as evidence that a protocol event is unreachable.

The interaction lane must also run under an explicit v3 spec, rather than the
checkpoint-free default. Its manifest records the scenario ids, source hash,
observation schema, and the silent-noop verdict-table revision it exercised.
That provenance lets a later schema change rerun only the affected scenarios
without misrepresenting older v2.2 evidence as coverage of v3 fields.

## 8. Orchestration & budget

Deterministic and embarrassingly parallel across games (~220 draft games + a small gap-fill set,
or 874 exact-variant games, each 1–few boundaries). Shard across workers; merge the per-shard `DeepLineAuditReport`/`Acc`
accumulators and the coverage ledgers. CLI shape mirrors the existing harnesses: `--showdown-root`,
`--json PATH` (findings; exit 1 iff any real-bug signature), `--coverage-json PATH` (the ledger),
`--pass {A,B,both}`, `--gap-fill`, `--exact-variants`, `--observation-schema v3`,
`--use-moves` (§7.1), `--shard i/N`.

## 9. Deliverable

1. The **coverage-driver** (`scripts/coverage_enumeration_audit.py` + a `src/pokezero/` module):
   the enumeration/draft/census/gap-fill logic emitting `BattleStartOverride`s and consuming the
   reused asserter. This is the **only** new code.
2. A **coverage report artifact** (JSON): per-(species×ability) status (target 100% of reachable
   pairs), per-move and per-item first-covering game, the uncovered sets (target ∅ for both
   movepool moves and source items), and the universal-move mini-lane result.
3. A **ranked flagged-inconsistency list** — each finding with the triggering atom (species,
   ability, move), the divergent column(s), oracle-vs-encoder values, a minimal repro,
   TRAINING-AFFECTING yes/no + incidence, and a classification.
4. Regression stubs for confirmed flags, and the committable coverage harness as a permanent gate.

## 10. Honest limits

**Catches:** species/ability/move-specific *single-visit* mis-encodings — dex/forme/base-stat
errors on rare species, ability-attribution errors (including second abilities), move-vocab gaps,
HP-type mis-encodes, per-column public-surface errors across the **entire** catalog, and any OOV
on a reachable atom.
**Won't catch (by design — that is the deep-line plan's job):** multi-turn state-accumulation
bugs, mechanic-chain interactions, sequence-dependent mis-encodings. This plan is **breadth**
(every atom once); the deep-line plan is **depth** (sequences). They compose — run both.
**Custom-game caveat:** `gen3customgame` does not enforce randbat legality, so the driver *could*
construct sets the generator never rolls. Constrain every atom to the **reachable** universe
(reachable abilities, movepool moves, `sets.json` levels/items) so every encoded atom is one that
can appear in training data; flag any construction that steps outside it. `Oracle.species_of_mon`
applies `canonical_gen3_randbat_species_id` — verify custom species canonicalize as expected
(Deoxys formes distinct; Unown formes collapse to `unown`).
**The ultimate validator** remains behavior analysis on the retrained checkpoints — back-trace any
anomaly to its encoder cause.

## 11. Guardrails

- Read-only on production encoder code; flag, don't fix. Each confirmed bug → a **separate**
  checkpoint-compatible fix PR (value-only, no observation-schema change) that fast-follows the
  runs.
- Any finding touching the transition/fold surface must re-check Rust-fold parity
  (`validate_corpus_v2.py --backend rust`) and regen the golden corpus if the fold products move —
  the #758/#767 lesson (the sample corpus does not catch it when it lacks the triggering fixtures).
- Commit as the user; no AI co-author trailer.
