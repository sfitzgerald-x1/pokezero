# Golden encoder corpus (track B) — schema, provenance, and gaps

Companion to `docs/test_time_search_plan_v3.md` ("The golden corpus"). The
corpus is the bit-exactness reference for the engine-side v2.2 encoder: track
B is done when every stored observation tensor is reproduced bit-for-bit from
the stored position context. Generator, reader, and verifier live in
`src/pokezero/golden_corpus.py`; tests in `tests/test_golden_corpus.py`.

## Corpus v1 (generated 2026-07-18)

- Location: `corpus/golden-v1/` (gitignored — the corpus is a local/build
  artifact, never committed; regenerate with the command below).
- Command:
  `python -m pokezero.golden_corpus --showdown-root <built-showdown> \
   --games 10 --seed-start 1000 --out corpus/golden-v1 --belief-set-source on`
- Contents: 10 games (seeds 1000–1009), 1028 decision rows
  (p1: 512, p2: 516; 72 rows are solo-request/force-switch boundaries;
  0 games hit the round cap).
- Hashes:
  - `manifest.json` SHA-256: `a5367aa026f92ce40a55a1d6b55eb0e86acf14fd95ec6e8d4fb2e837fccf29da`
  - `rows.jsonl` SHA-256: `b67933a91ba40c29e6a53ac2f1d152516db5b4fcb80aff7d99deeb3bca9eae00`
  - `arrays.npz` SHA-256: `7c8b7f4caa7fd551ad4388020d35861ae01b561c93c5e962bd9c9cdb86283e4f`
- Sizes: rows.jsonl ~31 MB, arrays.npz ~2.2 MB.
- Belief candidate-set source: **enabled** (matches the belief-run v2.2
  production line), source hash `e648ed6a6fbf7c78`.
- Observation contract: schema `pokezero.observation.v2.2`, 151 tokens,
  51 categorical columns, 155 numeric columns, 9 actions, default feature
  masks (stats on, exact-state on, transition budget 128, tier2 residuals on,
  tier2 investment off). All stamped in the corpus header.
- Determinism: `rows.jsonl` is byte-identical across regenerations with the
  same Showdown build + belief source (verified twice for seed 1000).
  `arrays.npz` is content-identical but not byte-identical across runs (zip
  member timestamps); the per-row `arrays_sha256` values are the stable
  identity.
- A committed 5-row sample (first 5 decisions of seed 1000) lives at
  `tests/data/golden_corpus_sample/` and is verified by
  `GoldenCorpusCommittedSampleTest` on every test run — the permanent
  regression net for schema and shapes.

## Schema `pokezero.golden-encoder-corpus.v1`

Directory layout: `rows.jsonl` + `arrays.npz` + `manifest.json`.

`rows.jsonl` records (canonical JSON: sorted keys, ASCII, `allow_nan=False`):

1. `header` — schema id + schema SHA-256, generator config, observation
   contract (schema/shape census/feature masks), belief-source provenance.
2. `game` (one per battle) — `battle_seed`, `battle_id`, `format_id`,
   `policy_ids`, `true_teams` (below), `terminal`
   (winner/turn_count/capped), `decision_row_count`.
3. `decision` (one per decision point per requested seat) —
   - identifiers: `battle_seed`, `battle_id`, `format_id`, `player_id`,
     `decision_round_index`, `requested_players`;
   - `observation`: schema version, perspective (seat/slot mapping),
     `array_row_index` into `arrays.npz`, `arrays_sha256`, inline
     `legal_action_mask`;
   - `observation_metadata`: the production observation's metadata dict
     **verbatim** — includes the `belief_view` overlay (both seats' revealed
     facts, candidate sets, uncertainty), `self_team`/`opponent_team`
     request-known views, action candidates, side conditions, etc.;
   - `public_materialization`: the
     `local_showdown._public_materialization_payload(state)` dict
     **verbatim** — the public/player-known branch-point payload direct
     search uses to construct sampled worlds (track A's input surface);
   - the acted decision: `chosen_action_index`, `chosen_policy_id`,
     `chosen_action_probability`;
   - `row_sha256`: SHA-256 of the canonical row JSON (excluding itself);
     binds the JSON context to the arrays via `arrays_sha256`.

`arrays.npz` — the golden arrays stacked over decision rows, in canonical
little-endian dtypes:

| field | dtype | per-row shape |
|---|---|---|
| `categorical_ids` | `<i4` | (151, 51) |
| `numeric_features` | `<f8` | (151, 155) |
| `token_type_ids` | `<i2` | (151,) |
| `attention_mask` | `\|b1` | (151,) |
| `legal_action_mask` | `\|b1` | (9,) |

`arrays_sha256` = SHA-256 over the row's five slices concatenated in the
table's order as C-order bytes. `manifest.json` carries schema id/hash, game
and decision counts, array shapes/dtypes, and SHA-256 + byte size of both
files. `verify_golden_corpus` re-checks everything: file hashes, per-row
JSON hashes, per-row array hashes, ordering, counts, and the inline legal
mask against the npz copy.

### Precision note (what "as the model consumes it" means here)

`numeric_features` is stored at full encoder precision (Python floats =
IEEE float64, exactly what `observation_from_player_state` emits). Downstream
consumers quantize: the live inference path casts to float32 at tensor build;
training caches store float16. Bit-exactness for track B is defined at the
encoder output (float64) — every downstream quantization is derivable from
it, never the reverse. A candidate encoder that matches these bytes matches
every consumer.

## Capturable vs. gapped

**Closed (previously assumed gapped): generator-internal EVs/IVs.** Player
requests do not carry EVs/IVs, but the corpus does not rely on requests for
ground truth: the generator takes one oracle `LocalShowdownEnv.snapshot()` at
the opening request boundary (before any decision) and stores each side's
generator `PokemonSet` verbatim — species, moves, ability, item, level,
**evs, ivs** (including Hidden Power IV spreads), plus randbats `role` and
`hpType`. The snapshot is corpus-only ground truth; no policy ever sees it.
`true_teams.<seat>.packed` is a `showdown_fixture.pack_team` string built
from those sets, suitable for `BattleStartOverride`-style replay.

Remaining gaps in the true-team record (documented, not fabricated —
exemption candidates for any comparison that consumes packed teams):

- **Nature** — gen3 randbats generator sets carry no `nature`; the packed
  string leaves the nature field empty (Showdown's neutral default applies).
- **Gender** — the generator set's `gender` is unset (`""`); the battle rolls
  a concrete gender at start. The packed string pins the battle-actual gender
  (taken from the snapshot's battle state, also stored per mon as
  `battle_gender`), while the verbatim `set` payload preserves the
  generator's own unset value.
- **Happiness / shiny / pokeball** — generator defaults; `shiny` is present
  in the verbatim set, happiness and pokeball are Showdown defaults and are
  not separately recorded.

Per-seat **request-known** team views (what a real player/agent could know)
remain available independently of the oracle record: each decision row's
`observation_metadata.self_team` carries the seat's request-derived team with
computed battle stats (incl. max HP), exact move ids, ability, and item.

**Encoder-comparison exemption list: empty.** No field of the golden arrays
is exempted. Per the plan's exemption rule, any legitimately unequal field
found during track B development must be added here with a justification —
never handled by a global loosening of the comparison.
