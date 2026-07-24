# Endgame Scenario Studio Plan

**Audience:** an implementation agent working from current `main`.
**Status:** implemented locally; the first ten source-composed materialization fixtures are
committed, while forced-line proof remains deferred.
**Primary outcome:** a local website for constructing, validating, saving, loading, and
evaluating Gen 3 randbats endgame scenarios.

## 0. Objective

Build a small local "scenario studio" that lets an author:

1. Generate source-valid Gen 3 randbats teams.
2. Build each side explicitly by adding, removing, and ordering selected randbats Pokemon.
3. Set each Pokemon's current HP.
4. Set each move's remaining PP.
5. Set major status, active volatile effects, weather, hazards, and screens.
6. Select the active Pokemon and side to move.
7. Record the expected winning action or line.
8. Validate the scenario through Pokemon Showdown.
9. Save and load stable, reviewable JSON files.
10. Inspect how a checkpoint ranks the legal root actions.

The first use is a curated endgame set that asks whether a policy can identify a guaranteed
line to victory. This is an authoring and evaluation tool, not a replacement battle simulator.
Pokemon Showdown remains the legality and battle-state oracle. The Rust engine can later prove
bounded forced lines after a scenario has been materialized and parity-checked.

## 1. Recommended Product Cut

The easiest trustworthy implementation is:

- A loopback-only Python HTTP server using the standard library.
- A vanilla HTML/CSS/JavaScript frontend served by that process.
- A side-roster builder backed by `Gen3RandbatSource`, with species and exact-set selection.
- Pokemon Showdown's `Teams.generate("gen3randombattle")` as an optional full-team shortcut.
- `FixturePokemon` and `pack_team` for packed-team construction.
- A dedicated, allowlisted battle-bridge command for applying current HP, PP, active-slot,
  major-status, volatile, weather, and side-condition overrides to a `gen3customgame` battle.
- Versioned JSON files under `scenarios/endgame/` as the canonical persistence format.

Do not begin with React, a database, user accounts, cloud hosting, arbitrary simulator JSON
editing, or a general-purpose battle editor. None is needed for a single-user local tool, and
each would make validation harder.

## 2. What "Valid" Means

An editor cannot honestly call every arbitrary HP/PP combination a reachable random battle.
Expose three separate validity levels in the UI and saved file:

| Badge | Meaning |
|---|---|
| `set-valid` | Every Pokemon is an exact set from the pinned Gen 3 randbats source. |
| `state-consistent` | Team, active slot, HP, PP, and battle invariants pass local and Showdown validation. |
| `replay-proven` | A deterministic action prefix reaches the same state from a generated battle. |

The MVP must provide the first two. Replay proof is optional and should not block authoring.
Reports must never describe a merely state-consistent scenario as naturally reachable.

Two team-construction modes are useful:

| Mode | Contract |
|---|---|
| `generated` | The untouched six-Pokemon party came from Showdown's randbats team generator. |
| `source-composed` | The author selected one to six source-valid Pokemon; composition may not be generator-reachable. |

Both modes are first-class creation workflows. An author can start with an empty side and add
specific species, or generate a full party as a shortcut. Replacing, removing, or adding a
Pokemon on a generated party automatically changes that side to `source-composed`. Merely
marking generated party members fainted retains `generated` provenance. This preserves honest
provenance while allowing one-on-one, two-on-two, and other late-game positions.

## 3. Existing Building Blocks

Reuse these contracts rather than creating parallel formats:

- `src/pokezero/randbat.py`
  - `Gen3RandbatSource` provides the source-hashed legal variant universe.
  - Move and species metadata provide display and stat inputs.
- `src/pokezero/showdown_fixture.py`
  - `FixturePokemon`, `pack_pokemon`, and `pack_team` produce Showdown-compatible packed teams.
- `src/pokezero/env.py`
  - `BattleStartOverride` starts complete packed teams in `gen3customgame`.
- `src/pokezero/local_showdown.py`
  - `LocalShowdownEnv` supplies requests, legal actions, observations, snapshots, and stepping.
- `scripts/battle_bridge.mjs`
  - The bridge already serializes and restores Showdown battle state.
- `src/pokezero/golden_corpus_scenarios.py`
  - `ScenarioSpec` and scripted preferences demonstrate deterministic curated fixtures.
- The production checkpoint adapter and policy-probe code
  - Reuse checkpoint loading and action-label reporting for root action inspection.

Packed teams encode species, moves, item, ability, level, nature, gender, EVs, and IVs. They do
not encode in-progress HP or PP. Treat current HP/PP materialization as a first-class seam, not
as an extension to the packed-team string.

## 4. Scope

### MVP

- Gen 3 singles only.
- Local machine only, bound to `127.0.0.1`.
- Generate complete random-battle teams.
- Select one to six specific Pokemon independently for each side.
- Choose an exact source-valid randbats set for every selected species.
- Add, remove, replace, and reorder Pokemon on either side.
- Browse exact source variants.
- Set active slots, current HP, and remaining PP.
- Set exact major status counters, reconstructable active volatiles, Gen 3 weather, Spikes,
  Reflect, Light Screen, Safeguard, and Mist.
- Mark unavailable Pokemon by setting HP to zero.
- Perfect-information endgame scenarios.
- Record expected root actions and an optional principal variation.
- Validate, save, load, duplicate, import, and export JSON.
- Show the root legal-action labels.
- Run one checkpoint at the root and display action probabilities/ranks.

### Follow-on

- Replay-proven scenario authoring.
- Explicit public-knowledge and hidden-world editing.
- Batch checkpoint evaluation.
- A bounded forced-win verifier using Rust apply/reverse search.
- Scenario tags, suites, and aggregate scorecards.
- State import from captured real games.

### Non-goals

- Editing arbitrary Showdown internals.
- Serving the website on a network interface.
- Proving every arbitrary state is reachable.
- Training from scenarios.
- Replacing Showdown legality checks with Python approximations.
- Letting the editor silently reveal hidden information to a production policy.
- Blocking the basic editor on a formal game-theoretic solver.

## 5. Canonical Scenario Contract

Add a typed Python domain model and a stable JSON representation. A representative file:

```json
{
  "schema_version": "endgame-scenario-v1",
  "scenario_id": "blaze-recoil-001",
  "title": "Preserve the Blaze cleaner",
  "description": "Find the forced two-ply line without losing the endgame to recoil.",
  "tags": ["threshold", "recoil", "two-ply"],
  "format_id": "gen3customgame",
  "source_format_id": "gen3randombattle",
  "seed": 1701,
  "provenance": {
    "randbat_source_hash": "required",
    "replay_proven": false
  },
  "knowledge_mode": "fully_revealed",
  "perspective": "p1",
  "side_to_move": "p1",
  "teams": {
    "p1": {
      "construction_mode": "source-composed",
      "generated_team_seed": null,
      "active_slot": 0,
      "pokemon": [
        {
          "slot": 0,
          "variant_id": "source-backed-id",
          "species": "Blaziken",
          "level": 77,
          "ability": "Blaze",
          "item": "Leftovers",
          "moves": [
            {"id": "fireblast", "pp": 3, "max_pp": 8},
            {"id": "skyuppercut", "pp": 9, "max_pp": 24},
            {"id": "rockslide", "pp": 7, "max_pp": 16},
            {"id": "swordsdance", "pp": 20, "max_pp": 32}
          ],
          "current_hp": 61,
          "max_hp": 281
        }
      ]
    },
    "p2": {
      "construction_mode": "source-composed",
      "generated_team_seed": null,
      "active_slot": 0,
      "pokemon": [
        {
          "slot": 0,
          "variant_id": "source-backed-id-p2",
          "species": "Snorlax",
          "level": 73,
          "ability": "Immunity",
          "item": "Leftovers",
          "moves": [
            {"id": "bodyslam", "pp": 4, "max_pp": 24},
            {"id": "earthquake", "pp": 6, "max_pp": 16},
            {"id": "rest", "pp": 1, "max_pp": 16},
            {"id": "sleeptalk", "pp": 8, "max_pp": 16}
          ],
          "current_hp": 94,
          "max_hp": 391
        }
      ]
    }
  },
  "objective": {
    "kind": "forced_win",
    "expected_root_actions": ["move fireblast"],
    "principal_variation": [],
    "max_plies": 6,
    "verification": {
      "status": "unverified",
      "engine": null,
      "artifact": null
    }
  },
  "author_notes": ""
}
```

The implementation may normalize field names during the first schema PR, but the following
properties are required:

- Files store stable IDs and source provenance, not display text alone.
- Construction provenance is recorded independently for p1 and p2.
- Current HP and PP are exact integers.
- `max_hp` and `max_pp` are validator-derived and checked on load.
- Variant fields remain present for human review, but `variant_id` is authoritative.
- JSON key order and indentation are deterministic for clean diffs.
- Derived fields are either recomputed or verified; stale derived values fail closed.
- Future schema versions require an explicit migration function.

## 6. Public Information And History

A true engine state is not automatically a valid model observation. The policy also consumes
public reveals, belief state, and transition history. The MVP therefore uses one narrow,
explicit contract:

- `knowledge_mode` is `fully_revealed`.
- The evaluated side is given all species, moves, items, and abilities selected by the scenario.
- The scenario is labeled synthetic unless it has a replay proof.
- Empty or synthetic history is recorded in evaluation output.
- Synthetic scenarios may compare actions and checkpoint versions, but reports must not treat
  them as an in-distribution strength benchmark without a replay or capture provenance.

Do not smuggle omniscient state through production observation code. Build a dedicated scenario
observation adapter that creates the declared fully revealed belief state and uses the normal
encoder. Add a parity test proving that self-side HP/PP, opponent public HP, reveals, active
slots, and legal actions match the materialized battle.

The follow-on hidden-information mode must store true world and public knowledge separately.
It is not a boolean "hide opponent" switch.

## 7. Battle-State Materialization

This is the first implementation task because it decides whether the rest of the website has a
sound execution path.

Add a dedicated command to `scripts/battle_bridge.mjs` that:

1. Requires an existing `gen3customgame` battle created from complete packed teams.
2. Serializes the battle with the existing Showdown state API.
3. Applies only allowlisted fields:
   - active party index;
   - current HP;
   - remaining PP for existing move slots;
   - major status and its exact sleep/toxic counter;
   - reconstructable active volatile effects and their required counters;
   - weather and its duration/permanence;
   - Spikes layers and timed side conditions.
4. Rejects unknown paths, extra fields, invalid slots, invalid move IDs, and invalid values.
5. Restores the patched state through Showdown's deserializer.
6. Regenerates a clean actionable request boundary for both players.
7. Returns a normalized state summary, requests, legal actions, and a validation token.

The command must not accept raw serialized battle JSON from the browser. The browser sends the
scenario contract; Python validates it and sends a narrow patch to the bridge.

### Spike acceptance

One deterministic two-on-two fixture must prove:

- The requested active slots appear in Showdown requests.
- Exact HP appears correctly in self requests and public protocol.
- Remaining PP appears correctly in the owning player's move slots.
- The legal action mask is correct.
- Snapshot, restore, and a second snapshot preserve authored state exactly.
- One legal turn can be stepped after materialization.
- An out-of-range HP or PP value fails before mutating the live battle.

If a clean request boundary cannot be regenerated without bypassing Showdown invariants, stop
this task and record the failure. The fallback is a replay-prefix scenario format. Do not switch
to Rust-only state construction for checkpoint evaluation because the model observation would
then disagree with the battle world.

## 8. Local Web Architecture

Create a small package, for example:

```text
src/pokezero/scenario_studio/
  __init__.py
  cli.py
  domain.py
  catalog.py
  materialize.py
  storage.py
  server.py
  static/
    index.html
    app.js
    styles.css
```

Expose a command such as:

```bash
pokezero-scenarios \
  --showdown-root /path/to/pokemon-showdown \
  --scenario-dir scenarios/endgame
```

Defaults:

- Bind to `127.0.0.1`.
- Choose an available local port or accept `--port`.
- Open no external network listener.
- Print the local URL and source hash.
- Never require a Node or frontend build step.

Use `ThreadingHTTPServer` or an equally small standard-library server. Keep request handling and
domain logic separate so the domain and materializer can be tested without a browser.

### API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/catalog` | Species, exact variants, source hash, move metadata, and generated-team capability. |
| `POST` | `/api/teams/generate` | Generate a full source-valid p1 or p2 randbats team from a seed. |
| `POST` | `/api/validate` | Validate and materialize an unsaved scenario without writing it. |
| `GET` | `/api/scenarios` | List saved scenario metadata. |
| `GET` | `/api/scenarios/{slug}` | Load one scenario. |
| `PUT` | `/api/scenarios/{slug}` | Validate and atomically save one scenario. |
| `DELETE` | `/api/scenarios/{slug}` | Optional after save/load is stable. |
| `POST` | `/api/evaluate-root` | Run one checkpoint at the validated root state. |

Return structured validation errors with a JSON pointer and user-facing message. Do not return
Python tracebacks or filesystem paths to the browser.

## 9. Website Workflow

### Layout

- Scenario metadata and validity badges at the top.
- Side-by-side p1 and p2 team panels.
- Objective and expected-line editor below the teams.
- Validation and action-preview panel on the right or below on small screens.
- Save, Load, Import, Export, and Validate actions in a persistent toolbar.

### Team editing

- Each side starts with `Add Pokemon` and optional `Generate team` actions.
- `Add Pokemon` opens a searchable species picker sourced from `Gen3RandbatSource`.
- Selecting a species shows only its exact legal randbats variants.
- Selecting a variant adds that Pokemon, its legal moves, ability, item, and level to the side.
- Each Pokemon can be replaced, removed, and reordered independently.
- `Generate team` creates all six sets through Showdown from a displayed seed.
- Editing the membership of a generated party automatically labels it `source-composed`.
- A side may contain one to six selected Pokemon, independent of the other side's count.
- Each Pokemon card shows level, item, ability, moves, max HP, current HP, and PP.
- HP has both a number input and slider.
- PP is an integer input bounded by Showdown-derived max PP.
- Exactly one living Pokemon is selected active per side.
- Setting HP to zero marks a Pokemon unavailable/fainted.
- Moves, item, ability, level, EVs, and IVs are locked to the exact variant in the MVP.
- Choosing another source variant is the supported way to change a set.

### Persistence

- `Save` is disabled until local and Showdown validation pass.
- `Load` shows title, tags, source hash, validity badges, and last modification time.
- `Import` validates a selected JSON file but does not save until confirmed.
- `Export` downloads the canonical normalized JSON.
- A browser-local draft may protect unsaved edits, but it is never canonical storage.

### Evaluation preview

- Show root legal actions using existing action-label helpers.
- Accept a checkpoint path and display each action's logit/probability/rank.
- Highlight expected root actions without forcing the policy to select them.
- Stamp checkpoint identity, observation schema, transition budget, source hash, and synthetic
  history status into the result.

## 10. Validation Rules

Local validation must complete before starting Showdown:

- Schema version is supported.
- Scenario IDs and save slugs match a strict safe pattern.
- Both sides contain one to six Pokemon.
- Every `variant_id` resolves in the pinned source hash.
- Species, level, ability, item, and moves match the resolved variant.
- No duplicate canonical species exist on one side.
- Slots are unique and ordered.
- Exactly one living Pokemon is active on each side at an actionable root.
- Each side has at least one Pokemon with HP greater than zero.
- `0 <= current_hp <= max_hp`.
- `0 <= pp <= max_pp` for every move.
- Move slots and PP entries have the same length and IDs.
- All-zero PP is accepted only if Showdown returns the intended Struggle action.
- `perspective` and `side_to_move` are valid player IDs.
- Each side's `generated` mode preserves that side's complete generated party and seed provenance.
- A source-hash mismatch requires explicit revalidation or migration; never silently relatch.

Showdown validation must then prove:

- Packed teams start successfully in `gen3customgame`.
- The allowlisted state patch succeeds.
- The returned request boundary matches the scenario.
- Legal actions are non-empty for the requested side.
- A snapshot/restore round trip preserves the normalized state.
- The normal PokeZero observation can be constructed under the declared knowledge mode.

File storage must:

- Resolve all paths under the configured scenario directory.
- Reject traversal, absolute paths, symlinks escaping the directory, and ambiguous slugs.
- Write to a temporary file, flush, and atomically rename.
- Preserve an existing valid file if validation or writing fails.

## 11. What Counts As A Guaranteed Line

State authoring and forced-win proof are separate.

For deterministic branches, a forced win means:

- At least one root action wins against every legal opponent reply.
- The claim holds through every legal continuation within the declared search depth.
- Terminal victory is reached or a separately defined solved tablebase condition is met.

For random branches, "guaranteed" means every enumerated chance outcome wins. If only a
probability threshold is met, label the scenario `probabilistic`, store the measured lower bound,
and do not call it guaranteed.

The MVP stores:

- expected root action or actions;
- optional principal variation;
- proof status: `unverified`, `manual`, `engine_exhaustive`, or `probabilistic`;
- maximum proof depth;
- optional proof artifact path.

A later Rust verifier should consume the canonical scenario, materialize the same root state,
enumerate legal actions/replies/chance outcomes with apply/reverse, and emit a machine-readable
proof artifact. It must first pass Showdown parity on the scenario root and any mechanics used by
the line.

## 12. Implementation Queue

Each numbered task is one reviewable PR unless the diff remains trivially small. The executing
agent should complete them in order and keep this status table current.

| Task | Deliverable | Status |
|---|---|---|
| S0 | Domain contract and typed battle-state materialization | Complete |
| S1 | Catalog, validation, and atomic JSON persistence | Complete |
| S2 | Local server and team editor website | Complete |
| S3 | Save/load/import/export and browser workflow tests | Complete |
| S4 | Root checkpoint evaluation and report contract | Complete (torch-backed checkpoint smoke remains environment-dependent) |
| S5 | Seed endgame suite and batch scorecard | Complete for the initial 10-fixture materialization suite; tactical proof is deferred |
| S6 | Optional replay proof and Rust forced-line verifier | Deferred |

### S0: Contract and materialization

Deliver:

- Typed scenario v1 domain objects.
- Canonical JSON round trip.
- Exact source-variant resolver.
- Dedicated bridge materialization command.
- Deterministic battle-state fixture satisfying the materialization acceptance criteria.

Stop condition:

- Do not start the website until the HP/PP fixture can be stepped through Showdown.

### S1: Catalog, validator, and storage

Deliver:

- Catalog service over `Gen3RandbatSource`.
- Full-team generation adapter over Showdown's randbats generator.
- Local and Showdown-backed validators.
- Atomic filesystem repository.
- Source mismatch and path traversal tests.

Stop condition:

- A command-line test can generate, validate, save, reload, and revalidate one scenario with
  byte-stable canonical JSON.

### S2: Local website

Deliver:

- Loopback server and static assets.
- Independent side-roster builders with add, replace, remove, and reorder controls.
- Searchable species picker and exact legal-set picker.
- Optional generated-team shortcut with automatic provenance downgrade after membership edits.
- Active, HP, and PP controls.
- Objective editor, validity badges, and validation errors.
- Responsive desktop and narrow-window layout.

Stop condition:

- A user can select four specific Pokemon, arrange a two-on-two endgame, and validate it without
  editing JSON.

### S3: Persistence workflow

Deliver:

- Save and load controls.
- Import and export.
- Unsaved-change warning and optional local draft.
- Duplicate-name conflict handling.
- Browser/API integration tests.

Stop condition:

- A saved scenario survives server restart and loads identically.

### S4: Root policy evaluation

Deliver:

- Checkpoint picker/path input.
- Root legal-action probabilities and ranks.
- Expected-action highlighting.
- Machine-readable evaluation result with full provenance.

Stop condition:

- A deterministic test checkpoint produces stable root ranks on a fixture scenario.

### S5: Initial scenario suite

Create 10 to 20 manually reviewed scenarios spanning distinct tactical patterns:

- priority and speed ordering;
- sacrifice into a guaranteed cleaner;
- recoil and low-HP ability thresholds;
- finite PP and forced Struggle;
- recharge and two-turn commitments;
- trapping and forced switches;
- weather or hazard chip;
- recovery denial;
- status timing;
- preserving the only winning move.

Do not count cosmetic variants of the same decision pattern as breadth. Each scenario needs a
human-readable explanation of why the expected line wins and a regression test that it still
materializes.

### S6: Optional proof layer

Only start after the editor and root evaluator are useful. Add replay-proven prefixes and/or the
Rust exhaustive verifier. This task must not cause a schema replacement; extend the existing
objective verification fields.

## 13. Verification Matrix

| Surface | Required evidence |
|---|---|
| Scenario domain | JSON round trip, stable formatting, version rejection, migration hook. |
| Randbat legality | Every selected set resolves to the pinned source; generated teams retain seeds. |
| Battle state | HP/PP, status counters, volatile counters, weather, hazards, screens, invalid values, snapshot/restore. |
| Active slot | Living active requirement, switch legality, force-switch boundary rejection. |
| Showdown | Start, patch, request, legal mask, one step, terminal path. |
| Observation | Declared knowledge mode, HP/PP/reveal parity, legal-action parity. |
| Storage | Atomic overwrite, failed-write preservation, traversal and symlink rejection. |
| API | Valid and invalid requests, structured errors, no traceback leakage. |
| Browser | Generate, edit, validate, save, reload, import, export. |
| Checkpoint | Stable labels/ranks and complete provenance. |

Run the narrow tests for each task first, then the full Python suite before merge. If the bridge
changes, run its existing snapshot/search tests plus the new materialization fixture.

## 14. Agent Operating Loop

For each task:

1. Pull current `main` and read this document plus the referenced source contracts.
2. Select the first `Not started` or `In progress` task.
3. Create a `scott/` feature branch from current `main`.
4. Write a one-sentence task contract in the PR description before expanding scope.
5. Implement only that task and its required tests.
6. Run the narrow verification, then the broader affected suite.
7. Update this task table and any decision notes.
8. Open a PR with Summary, Changes Introduced, Risk Assessment, and Verification Evidence.
9. Run the required independent adversarial review.
10. Fix substantive findings, merge under the active goal rules, pull `main`, and continue.

Do not stop merely because a follow-on task exists. Stop only on a documented stop condition,
an unsafe/destructive action, a real permission failure, or an architecture decision that
cannot be answered from the repository. Long-running validation belongs in persistent jobs; do
not hold a foreground agent session open by polling.

## 15. Completion Criteria

The core goal is complete when:

- A local command starts the studio.
- Both sides can be built from independently selected Pokemon and exact legal randbats sets.
- Either side can optionally be populated by the full-team generator.
- The author can select active Pokemon and edit exact HP and PP.
- Invalid sets and inconsistent states fail closed with useful errors.
- A valid scenario materializes and steps in Showdown.
- Save/load/import/export preserve canonical JSON and provenance.
- A checkpoint's root legal-action ranking can be inspected.
- At least 10 distinct endgame scenarios are committed and materialize in CI.
- The UI and reports distinguish set validity, state consistency, and replay proof.

The forced-line solver is a separate completion gate. Until it lands, scenario claims are
manual or unverified and must be labeled accordingly.

## 16. Decisions To Preserve

- Pokemon Showdown is the source of legality and state-transition truth.
- The website edits a narrow domain contract, never raw simulator state.
- Direct side composition and generated teams are both supported; provenance distinguishes them.
- HP/PP must round-trip through a real actionable Showdown request before UI work proceeds.
- The first scenario set is perfect-information and synthetic-history by explicit design.
- Formal proof is valuable but does not block scenario authoring.
- Saved JSON is portable, diffable, source-hashed, and the canonical artifact.
