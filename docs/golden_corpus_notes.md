# Golden encoder corpus (track B) — schema, provenance, and gaps

Companion to `docs/test_time_search_plan_v3.md` ("The golden corpus"). The
corpus is the bit-exactness reference for the engine-side v2.2 encoder: track
B is done when the BOUNDARY cells are reproduced bit-for-bit from the per-row
surface (met, PR #710) and the HISTORY cells via the fold-state advance check
row-pair by row-pair (schema v2, below). Generator, reader, and verifier live
in `src/pokezero/golden_corpus.py`; the schema-v2 fold surface in
`src/pokezero/golden_corpus_fold.py`; tests in `tests/test_golden_corpus.py`,
`tests/test_golden_corpus_fold.py`.

## Corpus v1 (generated 2026-07-18, superseded by v2 below)

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
  regression net for schema and shapes. (Now regenerated as v2, below.)

## Corpus v2 (generated 2026-07-18) — the per-row fold surface

Schema v2 implements the plan's schema-v2 decision
(`test_time_search_plan_v3.md`, "Schema v2"): v1 rows cannot validate
history-derived cells (transition tokens 23–150, tendency aggregates), so v2
records, per decision row, the **incremental fold state** at the boundary
(`pokezero.transitions_fold.FoldState.to_payload()`, production-default tail
limits 512/128 — the corpus exercises production state, never shrunken knobs),
the **inter-decision event slice** (public protocol lines since the previous
SAME-SEAT decision boundary, `|t:|` wall-clock lines filtered), the **Tier-2
annotation overlay** applied at the boundary (the live trackers' per-index
conclusions — `apply_annotations`' runtime input), and the **boundary
products** (`FoldProducts`, generation-time asserted equal to the production
encoder state's surfaces before anything is written). The golden v1 columns
are UNCHANGED — the regenerated random battery's `arrays.npz` is byte-identical
to corpus v1 (`7c8b7f4c…`, same games, same golden arrays).

Generated artifacts (both gitignored; regenerate with the commands below):

- `corpus/golden-v2` — the random battery: 10 games (seeds 1000–1009),
  **1028 decision rows / 1028 fold rows**, belief candidate-set source ON.
  `python -m pokezero.golden_corpus --showdown-root <built-showdown> \
   --games 10 --seed-start 1000 --out corpus/golden-v2 --belief-set-source on`
- `corpus/golden-v2-scenarios` — the full curated scenario suite
  (Truant/Transform/Encore/recharge/Baton Pass/Wish/sand+Shedinja/RestTalk/
  screens/toxic): 10 games, **290 decision rows / 290 fold rows** — these
  deterministically exercise the OOV safety net (screens/Safeguard are outside
  the closed randbats vocab) and the risky fold components.
  `python -m pokezero.golden_corpus_scenarios --showdown-root <built-showdown> \
   --out corpus/golden-v2-scenarios --belief-set-source on`

Review notes (PR #720): the scenario suite records **0/290 rows with an
annotation overlay** — its scripted games produce no assessable Tier-2
strikes (production-confirmed; the generation binding would fail otherwise),
so overlay/annotation coverage comes from the random battery (905/1028 rows)
and the `test_transitions_fold` differential (535 annotated boundaries), not
the scenarios. Separately, overlays store ABSOLUTE token indices and the
reference `advance` requires them within the last `action_tail_limit=512`
actions; gen3randombattle maxes out around index reach 177 (~3x headroom),
and exceeding the wall in a longer format is a LOUD generation/validation
error ("annotation index outside identifiable range"), never silent
divergence.

### Size engineering (the fold payloads are tail-dominated)

Late-game fold payloads are ~226 KB (mean fold record ~125 KB, max ~383 KB over
1028 rows). The chosen lever is **whole-file gzip of the sidecar**
(`fold.jsonl.gz`, `mtime=0` so bytes are deterministic; stdlib-only — no zstd
dependency in the shared venv). Per-row compression was rejected (base64
overhead, loses the enormous key-repetition redundancy), and every-Nth-state
storage was not needed at these sizes (and would have indirected the row-pair
contract).

| corpus | fold rows | naive (uncompressed JSONL) | `fold.jsonl.gz` | ratio |
|---|---|---|---|---|
| golden-v2 | 1028 | 128,537,481 B (~122.6 MiB) | 5,069,549 B (~4.8 MiB) | 25.4x |
| golden-v2-scenarios | 290 | 11,881,630 B (~11.3 MiB) | 375,861 B (~0.36 MiB) | 31.6x |

(The golden files are unchanged: `rows.jsonl` ~32.5 MB / ~2.7 MB,
`arrays.npz` ~2.3 MB / ~0.6 MB.) The naive total is recorded per corpus in
`manifest.json` (`files."fold.jsonl.gz".uncompressed_bytes`).

### Row-pair validation contract (the Rust advance() gate)

`scripts/validate_corpus_v2.py` — per (game, seat) chain: row 0 must be
reproduced from `FoldState.initial()` advanced over its slice (+ overlay);
every consecutive same-seat pair must satisfy
`load(row_n.fold_state).advance(row_{n+1}.event_slice)` +
`apply_annotations(row_{n+1}.annotation_overlay)` == row_{n+1}'s recorded
fold state AND products, **canonical-JSON byte-exact**, each pair validated
independently from the recorded row-n state — exactly the transition a
search-time advance performs. The backend seam
(`pokezero.golden_corpus_fold.FoldBackend`: `start` / `load` / `step` over
payload dicts) mirrors `validate_rust_encoder.py`'s: the Rust `advance()`
port registers as one more backend and inherits the whole harness.

Results (2026-07-18, `--backend python-reference --verify-corpus`):

- golden-v2: 10 games x 2 seats = 20 chains, **1028/1028 boundaries validated**
  (20 chain starts + 1008 row pairs), state + products byte-exact, 2.1s.
- golden-v2-scenarios: 20 chains, **290/290 boundaries validated**
  (20 starts + 270 pairs), byte-exact, 0.2s.

Encoder-gate regression (unchanged golden surface, proven at scale):
`validate_rust_encoder.py --backend compare-backends` exits 0 with **ALL
EXACT** over all 1028 golden-v2 rows AND all 290 scenario rows (rust vs
python-reference, every array byte-for-byte).

### Rust backend (the advance() port, validated 2026-07-18)

`pokezero_search.FoldState` (rust/pokezero-search `src/fold.rs`) is the Rust
port of `pokezero.transitions_fold.FoldState` — event-line fold, windows,
turn-merge (incl. lead pass, cold pairs, Pursuit continuation, RestTalk/Baton
Pass collapse), tendency counters with the opportunity dedupe scalar, mon
counters, last-2-turn maps, pursuit ring buffer, Tier-2 annotation overlay +
pinned reductions (investment clamp included), production tail bounds, and the
full `pokezero.fold-state.v1` payload codec. It returns NATIVE Python objects
from `to_payload()` / `products_payload()` so the harness's canonical
`json.dumps` defines the compared bytes (float canonicalization never happens
in Rust). Backend adapter: `scripts/golden_fold_backends.py` (`rust`, plus
`compare-backends` — rust and python-reference side by side with JSON-path
divergence locators, the port's debugging loop).

    PYTHONPATH=src python scripts/validate_corpus_v2.py \
        --corpus corpus/golden-v2 --corpus corpus/golden-v2-scenarios \
        --backend rust --verify-corpus

Results (2026-07-18, this machine): golden-v2 **1028/1028 boundaries
byte-exact** (20 chain starts + 1008 row pairs, state + products, 2.1s);
golden-v2-scenarios **290/290 byte-exact** (0.2s); `compare-backends` prints
zero divergences over both corpora. `RustCommittedSampleFoldChainTest`
(`tests/test_golden_corpus_fold.py`) row-pair-validates the committed 5-row
sample through the rust backend on every test run (skip-if-absent on the
wheel, like the encoder gate). Rebuild the wheel with
`scripts/build_search_crate_model.sh` (keeps the `model` feature) or plain
`maturin build --release` from rust/pokezero-search.

**Input contract (matters most for the upcoming instruction→event mapping):**
the fold advance assumes well-formed Showdown protocol with ASCII-integer
numeric fields (HP `155/307`, `|turn|N`, `|-hitcount|...|N`). Inside that
domain rust and python-reference are byte-exact (corpus-proven; the PR #724
adversarial review's ~42k-slice differential fuzz found zero
protocol-plausible divergences). Outside it there is a documented
both-succeed-different class on malformed numeric literals: Python's
`float()`/`int()` accept underscore separators and non-ASCII unicode digits
(`float("1_0") == 10.0`, `int("٣") == 3`) where Rust's `str::parse` rejects
them — affecting the HP-fraction, `|turn|`, and `-hitcount` parses. Such
literals are unreachable from engine-emitted protocol; whoever builds the
instruction→event mapping must synthesize plain ASCII integers or the two
implementations may silently disagree. Literal `nan`/`inf` HP fields (both
languages parse them) follow Python's clamp exactly — NaN/+inf → 1.0,
-inf → 0.0, kept finite for native product consumption — guarded by
`RustFoldNanHpClampTest` and the crate's `fold::tests`.

Perf (`scripts/bench_fold_advance.py`, golden-v2's 1028 real boundary cases,
mean slice 9.6 lines, best of 5 passes): rust `clone_state()+advance_in_place`
**9.8µs/boundary (~102k boundaries/s)** vs the Python reference's pure
`advance()` at 91.7µs (~11k/s) — **9.3x** on the per-chance-outcome unit the
search-tree contract multiplies. Materializing the boundary products as
Python objects (`products_payload()`, +107µs) is deliberately excluded from
that headline: it is a harness/boundary-crossing artifact (the ~20k-object
payload graph), not part of the in-search path, where the in-crate encoder
will consume fold products natively.

### Determinism

Two back-to-back single-game regenerations (seed 1000) produce byte-identical
`rows.jsonl`, `arrays.npz`, `manifest.json`, AND `fold.jsonl.gz` (gzip written
with `mtime=0`; canonical JSON throughout). The committed 5-row sample now
carries its fold sidecar; `CommittedSampleFoldChainTest`
(`tests/test_golden_corpus_fold.py`) row-pair-validates it on every test run
with no Showdown checkout — the permanent regression net for
`FoldState.advance` against recorded production state.

## Schema `pokezero.golden-encoder-corpus.v2`

Directory layout: `rows.jsonl` + `arrays.npz` + `fold.jsonl.gz` (schema v2)
+ `manifest.json`. The v1 record shapes below are carried unchanged (records
now stamp the v2 schema id); the fold sidecar is optional at the writer level
(synthetic/array-only corpora may omit it) and always present in the
reference corpus.

`fold.jsonl.gz` (gzip JSONL, `mtime=0`): one `fold_header` record, then one
`fold` record per decision row in corpus row order. Fold record fields:
identifiers (`battle_seed`/`battle_id`/`format_id`/`player_id`/
`decision_round_index`), `chain_index` (per game+seat, from 0),
`array_row_index` + `row_sha256` (links to the golden decision row),
`event_slice`, `annotation_overlay` (`{token_index: [residual,
residual_valid, cb_bit, investment]}`), `fold_state` (+`fold_state_sha256`),
`products` (+`products_sha256`). `verify_golden_corpus` checks file hashes,
per-record payload hashes, row links, and chain contiguity; the sidecar
streams via `pokezero.golden_corpus_fold.iter_fold_records` and is
deliberately not loaded by `load_golden_corpus`.

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

## Encoder input contract (added after independent review — READ FIRST)

The corpus stores SEVERAL candidate input surfaces. Only one reproduces the
golden observation:

- **The golden observation is a function of the PUBLIC/BELIEF surface
  only**: `public_materialization` (the payload) + the acting seat's
  `observation_metadata` (including `belief_view` and the request-known
  `self_team`). An encoder that reproduces the golden arrays consumes
  exactly this surface — nothing else.
- **`true_teams` is ORACLE-ONLY.** It exists for world determinization
  (constructing engine states / `BattleStartOverride` replays where a
  concrete world is legitimately required, e.g. leaf simulation). It
  strictly over-informs relative to the golden observation — at an early
  decision the acting seat's opponent view is EMPTY while `true_teams`
  holds the full private set. It must NEVER be an input to the
  golden-observation reproduction, and never a production-model feature.
  If a candidate encoder only matches the golden arrays when given
  `true_teams`, that encoder is leaking private truth: the bit-exactness
  gate has caught a bug, not a formatting issue.
- Corollary for the engine-side leaf encoder (track B's consumer): at
  search leaves the engine state IS a determinized concrete world by
  design; the golden pairing in this corpus validates the PUBLIC-surface
  encoding path. The determinized-leaf encoding asymmetry is a design
  decision documented in the v3 plan, not something this corpus resolves.

## Determinism note (softened per review)

Two back-to-back regenerations produced byte-identical `arrays.npz` on
numpy 2.5.0, but per-row `arrays_sha256` (not npz file bytes) remains the
stable identity guarantee across numpy/zip versions.
