# C157 — Stage 1 preflight for the encode optimization plan

Scope: verification of `docs/encode_optimization_plan_20260810.md` §1 against source, and one
finding that changes what Stage 1 can implement. **No code change.** Nothing here alters the
plan's sequencing: §2 still gates a landed encode change on the axis study.

Tree: `main` at `230269db`, engine rebuilt against it (8/8 steps, crate fingerprint
`028a4c52a4ad9fe78145a3f74c7823ef333083bef9b0ffb23ec35f911a998374`).

## 1. §1 checked — clone claim exact, lookup count plausible but unconfirmed

**`leaf_row_inputs` deep-clones the root per leaf.** Exact:
`rust/pokezero-search/src/leaf.rs:1131` is `let mut row = self.root.clone()`, a
`serde_json::Value` tree clone to mutate a handful of fields.

**"~55 string-keyed `get()` lookups" — plausible, NOT confirmed.** I could not verify the number
to the precision the plan states it, and the counting is easy to get wrong. Lookups go through the
local helper `fn get<'a>(value: &'a Value, key: &str)` at `encoder.rs:34`.

Encoder-wide, measured with a receiver-agnostic pattern:

| receiver | helper-style `get(recv, "k")` sites |
|---|---|
| `md` | 34 |
| `entry` | 19 |
| `layout_value` | 13 |
| `constants` | 12 |
| `candidate` | 6 |
| `masks` | 5 |
| others / method-style `.get("k")` | remainder of **115 helper + 6 method = 121 total**, 100 distinct keys |

The `md` receiver — the per-leaf row metadata, which is what §1 is about — accounts for **34
static sites**, distributed:

| function | `get(md, …)` | inside a `for` | calls per leaf |
|---|---|---|---|
| `encode_field_token` | 19 | 2 | 1 |
| `encode_pokemon_tokens` | 9 | 1 | 2 (self + opponent) |
| `encode_row_value` | 3 | 0 | 1 |
| `write_history_cells` | 2 | 0 | 1 |
| `request_moves` | 1 | 0 | 1 |

So the `md` dynamic count per leaf is roughly **40**, with only 3 loop-resident sites. Reaching
~55 requires including `entry`/`candidate` (25 more static sites, loop residency unmeasured here).
**The plan's ~55 is consistent with that but is not independently confirmed by this preflight.**

`layout_value` / `constants` / `masks` (30 sites) look load-time rather than per-leaf — that is an
inference from the receiver names, **not measured**, and it matters, because those are precisely
the ones §3's "at table load, resolve once" would target for free.

**Counting hazard, recorded because it bit this preflight twice.** Earlier passes of this analysis
produced "43 sites / 36 keys" and "28 gets × mons × 2 teams". Both were regex artifacts — the
first restricted the receiver set, the second read static sites in a function body as if they were
all inside its loop when `encode_pokemon_tokens` is already hoisted (one loop-resident `md` get).
Any re-derivation should separate helper-style from method-style, name the receiver, and check
loop residency rather than body membership.

## 2. FINDING — Stage 1's stated mechanism collides with the §4 gate

§3 specifies: *"At table load, resolve the ~55 string paths to positional indices once; read
positionally per leaf,"* described as *"mechanical, no data shape change."* Those two are in
tension:

1. `rust/pokezero-search/Cargo.toml:34` enables `serde_json` with `features = ["float_roundtrip"]`
   only. **`preserve_order` is not enabled**, so `serde_json::Map` is a `BTreeMap` — there is no
   positional index into the row to resolve *to*.
2. Enabling `preserve_order` swaps the map to `IndexMap`, changing key iteration from sorted to
   insertion order.
3. That is byte-affecting: `leaf.rs:2568` does `serde_json::to_string(&row)` in
   `leaf_inputs_json`. Key order reaches serialized output. The same Cargo.toml already documents
   that `float_roundtrip` is *required* for bit-exactness, so this crate's serialization is known
   to be byte-sensitive.

A literal Stage 1 would therefore be a data-shape change arriving at gates 1 and 4 — which the
plan expected to be a formality for this stage.

**The intent survives without the risk.** Read `md` into a typed struct **once per leaf** and
access fields positionally thereafter. Same win (one map traversal instead of ~40 md lookups), no
map-type change, no serialization change, and it is a precursor to Stage 2's typed Row rather
than work thrown away when Stage 2 lands.

## 3. What this does not do

- **Does not confirm the ~55.** See §1 — `md` alone is ~40/leaf; the rest depends on
  `entry`/`candidate` loop residency, unmeasured here.
- **Does not run the profile.** §2 assigns that to the axis study
  (`docs/mcts_axis_cost_strength_study_20260809.md`), which is not in this repo. The static counts
  above size the *lookup* work only; they say nothing about its share of decision wall, and the
  plan's own §2 is explicit that 41.5% is "a ceiling claim until then, not a promise."
- **Does not establish the §5 baseline.** That is the study's cells, on the current build.
- **Does not touch Rust**, so it does not trip the c155 source-bytes citation tax.

## 4. Recommended amendments before Stage 1 is written

1. Restate Stage 1 as "one typed read of `md` per leaf, fields accessed positionally" rather than
   "resolve string paths to positional indices," and note explicitly that `preserve_order` is out
   of scope because it is byte-affecting.
2. Keep the §4 gate set unchanged. The revised mechanism should pass gates 1 and 4 trivially,
   and if it does not, that is a real divergence and §6 applies.
