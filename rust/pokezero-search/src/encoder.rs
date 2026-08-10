//! Native schema-bound observation encoder (track B of the engine-swap plan).
//!
//! Port of `pokezero.showdown.observation_from_player_state` for the golden
//! corpus's sanctioned per-row input surface (`observation_metadata` +
//! `public_materialization`; docs/golden_corpus_notes.md "Encoder input
//! contract"). Validated bit-exactly against the golden corpus through
//! `scripts/validate_rust_encoder.py --backend rust`.
//!
//! Every table (vocabulary row mapping, layout column indices, dex facts)
//! is loaded from the JSON artifact produced by
//! `scripts/export_encoder_tables.py` — nothing is hand-transcribed.
//!
//! The boundary-only entry point reproduces the sanctioned per-row surface.
//! `NativeEncoder.encode_with_fold` additionally consumes the incremental
//! public-event fold and reproduces the complete history surface.

use std::collections::HashMap;

use blake2::digest::{Update, VariableOutput};
use blake2::Blake2bVar;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};
use serde_json::Value;

// ---------------------------------------------------------------------------
// Small JSON access helpers (the metadata payload is deeply dynamic)
// ---------------------------------------------------------------------------

fn err(msg: impl Into<String>) -> PyErr {
    PyValueError::new_err(msg.into())
}

fn get<'a>(value: &'a Value, key: &str) -> &'a Value {
    value.get(key).unwrap_or(&Value::Null)
}

fn as_str<'a>(value: &'a Value) -> Option<&'a str> {
    value.as_str()
}

fn str_or_empty(value: &Value) -> String {
    value.as_str().unwrap_or("").to_string()
}

fn as_bool(value: &Value) -> bool {
    match value {
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().map(|v| v != 0.0).unwrap_or(false),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
        Value::Null => false,
    }
}

fn as_i64(value: &Value) -> i64 {
    match value {
        Value::Number(n) => n
            .as_i64()
            .or_else(|| n.as_f64().map(|f| f as i64))
            .unwrap_or(0),
        Value::Bool(b) => *b as i64,
        _ => 0,
    }
}

fn as_f64(value: &Value) -> f64 {
    match value {
        Value::Number(n) => n.as_f64().unwrap_or(0.0),
        Value::Bool(b) => {
            if *b {
                1.0
            } else {
                0.0
            }
        }
        _ => 0.0,
    }
}

fn empty_array() -> &'static Vec<Value> {
    static EMPTY: Vec<Value> = Vec::new();
    &EMPTY
}

fn as_array<'a>(value: &'a Value) -> &'a Vec<Value> {
    value.as_array().unwrap_or_else(|| empty_array())
}

// ---------------------------------------------------------------------------
// String normalization (must mirror the Python encoders exactly; all tokens
// in the closed gen3 universe are ASCII)
// ---------------------------------------------------------------------------

/// `category_vocab.normalize_category_value`: strip + lowercase.
fn normalize_category(value: &str) -> String {
    value.trim().to_lowercase()
}

/// `showdown._normalize_identifier` / `dex.normalize_id`: lowercase, drop
/// everything outside `[a-z0-9]`.
fn normalize_identifier(value: &str) -> String {
    value
        .to_lowercase()
        .chars()
        .filter(|c| c.is_ascii_lowercase() || c.is_ascii_digit())
        .collect()
}

/// Port of `randbat.canonical_gen3_randbat_species_id`: collapse a cosmetic Unown forme onto the
/// base species id that the dex and the randbat source actually carry.
///
/// Only genuine Unown cosmetic suffixes collapse -- the 26 letters plus `exclamation` and
/// `question`. Real distinct dex formes (Deoxys-Attack/Defense/Speed, Castform's weather formes,
/// Nidoran-F/M) resolve on the direct lookup and must never reach here, which is why this matches
/// the suffix set exactly rather than stripping any `-suffix`.
fn canonical_gen3_randbat_species_id(value: &str) -> String {
    let normalized = normalize_identifier(value);
    if let Some(suffix) = normalized.strip_prefix("unown") {
        let cosmetic = suffix.len() == 1
            && suffix.chars().next().is_some_and(|c| c.is_ascii_lowercase())
            || suffix == "exclamation"
            || suffix == "question";
        if cosmetic {
            return "unown".to_string();
        }
    }
    normalized
}

// ---------------------------------------------------------------------------
// Tables (vocab + layout + dex) from scripts/export_encoder_tables.py
// ---------------------------------------------------------------------------

struct Layout {
    schema_version: String,
    token_count: usize,
    categorical_width: usize,
    numeric_width: usize,
    action_count: usize,
    move_action_count: usize,
    cat: HashMap<String, usize>,
    num: HashMap<String, usize>,
    offsets: HashMap<String, usize>,
    belief_ability_buckets: usize,
    belief_item_buckets: usize,
    belief_move_buckets: usize,
    volatile_buckets: usize,
    actual_stat_divisor: f64,
    stat_count_divisor: f64,
    matchup_count_divisor: f64,
    timed_condition_duration: i64,
    hazard_conditions: Vec<String>,
    screen_conditions: Vec<String>,
    trap_abilities: Vec<String>,
    boost_stat_slots: Vec<(String, usize)>,
    base_stat_slots: Vec<(String, usize)>,
    actual_stat_slots: Vec<(String, usize)>,
    timed_condition_slots: Vec<(String, usize, usize)>,
    weather_reveal_order: Vec<String>,
    stats_block: bool,
    exact_state: bool,
    transition_token_budget: usize,
    tier2_residuals: bool,
    tier2_investment: bool,
    feature_pack_last_move: bool,
}

impl Layout {
    /// Grouped-layout lineage: v3 and everything after it. Gates every V3 public signal, which
    /// v4 also carries. Mirrors the Python encoder's `schema_v3` flag, which is likewise
    /// `schema_v4 || == v3` rather than an equality test.
    fn is_v3(&self) -> bool {
        self.schema_version == "pokezero.observation.v3" || self.is_v4()
    }

    /// V4: the k0 feature pack, and NO transition region at all. Turn-merged and grouped-layout
    /// are separate axes — v4 is grouped-layout but not turn-merged, because it has no history
    /// rows for a turn-merged surface to live in.
    fn is_v4(&self) -> bool {
        self.schema_version == "pokezero.observation.v4"
    }

    fn cat_col(&self, name: &str) -> PyResult<usize> {
        self.cat
            .get(name)
            .copied()
            .ok_or_else(|| err(format!("layout missing categorical column {name}")))
    }

    fn num_col(&self, name: &str) -> PyResult<usize> {
        self.num
            .get(name)
            .copied()
            .ok_or_else(|| err(format!("layout missing numeric column {name}")))
    }

    fn num_col_opt(&self, name: &str) -> Option<usize> {
        self.num.get(name).copied()
    }

    fn offset(&self, name: &str) -> PyResult<usize> {
        self.offsets
            .get(name)
            .copied()
            .ok_or_else(|| err(format!("layout missing token offset {name}")))
    }
}

struct SpeciesEntry {
    types: Vec<String>,
    base_stats: HashMap<String, i64>,
}

struct MoveEntry {
    id: String,
    move_type: String,
    gen3_category: String,
    base_power: i64,
    accuracy: f64,
    priority: i64,
    effect_label: String,
    effect_chance: i64,
    self_hp_cost: f64,
    max_pp: i64,
}

pub struct Tables {
    vocab_index: HashMap<String, i32>,
    oov_buckets: u64,
    oov_offset: u64,
    layout: Layout,
    species: HashMap<String, SpeciesEntry>,
    moves: HashMap<String, MoveEntry>,
}

fn slot_pairs(value: &Value) -> Vec<(String, usize)> {
    as_array(value)
        .iter()
        .filter_map(|pair| {
            let items = as_array(pair);
            if items.len() == 2 {
                Some((str_or_empty(&items[0]), as_i64(&items[1]) as usize))
            } else {
                None
            }
        })
        .collect()
}

fn string_list(value: &Value) -> Vec<String> {
    as_array(value).iter().map(str_or_empty).collect()
}

impl Tables {
    pub fn from_json(tables_json: &str) -> PyResult<Self> {
        let root: Value =
            serde_json::from_str(tables_json).map_err(|e| err(format!("tables JSON: {e}")))?;
        if get(&root, "schema_version").as_str() != Some("pokezero.encoder-tables.v1") {
            return Err(err("unsupported encoder tables schema"));
        }

        let vocab = get(&root, "vocab");
        let mut vocab_index = HashMap::new();
        if let Some(index) = get(vocab, "index").as_object() {
            for (token, row) in index {
                vocab_index.insert(token.clone(), as_i64(row) as i32);
            }
        }
        if vocab_index.is_empty() {
            return Err(err("tables vocab index is empty"));
        }

        let layout_value = get(&root, "layout");
        let mut cat = HashMap::new();
        if let Some(map) = get(layout_value, "categorical_columns").as_object() {
            for (name, column) in map {
                cat.insert(name.clone(), as_i64(column) as usize);
            }
        }
        let mut num = HashMap::new();
        if let Some(map) = get(layout_value, "numeric_columns").as_object() {
            for (name, column) in map {
                num.insert(name.clone(), as_i64(column) as usize);
            }
        }
        let mut offsets = HashMap::new();
        if let Some(map) = get(layout_value, "token_offsets").as_object() {
            for (name, offset) in map {
                offsets.insert(name.clone(), as_i64(offset) as usize);
            }
        }
        let constants = get(layout_value, "constants");
        let buckets = get(layout_value, "belief_buckets");
        let masks = get(layout_value, "default_feature_masks");
        let schema_version = str_or_empty(get(layout_value, "schema_version"));
        let tables_schema_is_v4 = schema_version == "pokezero.observation.v4";
        if !matches!(
            schema_version.as_str(),
            "pokezero.observation.v2.2" | "pokezero.observation.v3" | "pokezero.observation.v4"
        ) {
            return Err(err(format!(
                "unsupported observation layout schema {schema_version:?}"
            )));
        }
        let layout = Layout {
            schema_version,
            token_count: as_i64(get(layout_value, "token_count")) as usize,
            categorical_width: as_i64(get(layout_value, "categorical_feature_count")) as usize,
            numeric_width: as_i64(get(layout_value, "numeric_feature_count")) as usize,
            action_count: as_i64(get(layout_value, "action_count")) as usize,
            move_action_count: as_i64(get(layout_value, "move_action_count")) as usize,
            cat,
            num,
            offsets,
            belief_ability_buckets: as_i64(get(buckets, "ability")) as usize,
            belief_item_buckets: as_i64(get(buckets, "item")) as usize,
            belief_move_buckets: as_i64(get(buckets, "move")) as usize,
            volatile_buckets: as_i64(get(layout_value, "volatile_bucket_count")) as usize,
            actual_stat_divisor: as_f64(get(constants, "actual_stat_divisor")),
            stat_count_divisor: as_f64(get(constants, "stat_count_divisor")),
            // /8, not the tendency block's /64: a single (their mon x our mon) cell is visited
            // a handful of times per game. Exported so the two encoders cannot disagree.
            // No fallback ON PURPOSE: this constant is exported precisely so the two encoders
            // cannot disagree about it, and a silent default would reintroduce the drift it
            // exists to prevent. Absent or non-positive is a malformed table.
            matchup_count_divisor: {
                let value = as_f64(get(constants, "matchup_count_divisor"));
                if value > 0.0 {
                    value
                } else if tables_schema_is_v4 {
                    return Err(err(
                        "v4 encoder tables are missing or have a non-positive \
                         constants.matchup_count_divisor (it divides a count, so zero \
                         and negatives are malformed, not merely absent)",
                    ));
                } else {
                    // Pre-v4 tables never carry it and never read it.
                    f64::NAN
                }
            },
            timed_condition_duration: as_i64(get(constants, "timed_condition_duration")),
            hazard_conditions: string_list(get(constants, "hazard_conditions")),
            screen_conditions: string_list(get(constants, "screen_conditions")),
            trap_abilities: string_list(get(constants, "trap_abilities")),
            boost_stat_slots: slot_pairs(get(constants, "boost_stat_slots")),
            base_stat_slots: slot_pairs(get(constants, "base_stat_slots")),
            actual_stat_slots: slot_pairs(get(constants, "actual_stat_slots")),
            timed_condition_slots: as_array(get(constants, "timed_condition_slots"))
                .iter()
                .filter_map(|triple| {
                    let items = as_array(triple);
                    if items.len() == 3 {
                        Some((
                            str_or_empty(&items[0]),
                            as_i64(&items[1]) as usize,
                            as_i64(&items[2]) as usize,
                        ))
                    } else {
                        None
                    }
                })
                .collect(),
            weather_reveal_order: string_list(get(constants, "weather_reveal_order")),
            stats_block: as_bool(get(masks, "stats_block")),
            exact_state: as_bool(get(masks, "exact_state")),
            transition_token_budget: as_i64(get(masks, "transition_token_budget")).max(0) as usize,
            tier2_residuals: as_bool(get(masks, "tier2_residuals")),
            tier2_investment: as_bool(get(masks, "tier2_investment")),
            // v4 pack A2 ablation. Absent in pre-v4 tables, where the column does not exist,
            // so default TRUE (pack-whole) rather than false — a missing key must not silently
            // blank a column every v4 checkpoint expects.
            feature_pack_last_move: match masks.get("feature_pack_last_move") {
                Some(value) => as_bool(value),
                None => true,
            },
        };
        if layout.token_count == 0 || layout.categorical_width == 0 || layout.numeric_width == 0 {
            return Err(err("tables layout census is incomplete"));
        }

        let dex = get(&root, "dex");
        let mut species = HashMap::new();
        if let Some(map) = get(dex, "species").as_object() {
            for (key, entry) in map {
                let mut base_stats = HashMap::new();
                if let Some(stats) = get(entry, "base_stats").as_object() {
                    for (stat, v) in stats {
                        base_stats.insert(stat.clone(), as_i64(v));
                    }
                }
                species.insert(
                    key.clone(),
                    SpeciesEntry {
                        types: string_list(get(entry, "types")),
                        base_stats,
                    },
                );
            }
        }
        let mut moves = HashMap::new();
        if let Some(map) = get(dex, "moves").as_object() {
            for (key, entry) in map {
                moves.insert(
                    key.clone(),
                    MoveEntry {
                        id: key.clone(),
                        move_type: str_or_empty(get(entry, "type")),
                        gen3_category: str_or_empty(get(entry, "gen3_category")),
                        base_power: as_i64(get(entry, "base_power")),
                        accuracy: as_f64(get(entry, "accuracy")),
                        priority: as_i64(get(entry, "priority")),
                        effect_label: str_or_empty(get(entry, "effect_label")),
                        effect_chance: as_i64(get(entry, "effect_chance")),
                        self_hp_cost: as_f64(get(entry, "self_hp_cost")),
                        max_pp: as_i64(get(entry, "max_pp")),
                    },
                );
            }
        }
        if species.is_empty() || moves.is_empty() {
            return Err(err("tables dex is empty"));
        }

        Ok(Tables {
            vocab_index,
            oov_buckets: as_i64(get(vocab, "oov_buckets")).max(1) as u64,
            oov_offset: as_i64(get(vocab, "oov_offset")) as u64,
            layout,
            species,
            moves,
        })
    }

    pub(crate) fn layout_action_count(&self) -> usize {
        self.layout.action_count
    }

    pub(crate) fn layout_move_action_count(&self) -> usize {
        self.layout.move_action_count
    }

    pub(crate) fn layout_timed_condition_duration(&self) -> i64 {
        self.layout.timed_condition_duration
    }

    pub(crate) fn move_max_pp(&self, id: &str) -> Option<i64> {
        self.move_info(id).map(|info| info.max_pp)
    }

    /// `CategoryVocabulary.encode`: pad 0 for empty, direct row lookup, else
    /// the deterministic blake2b-8 OOV bucket.
    fn vocab_encode(&self, value: &str) -> i32 {
        // Empty cells short-circuit BEFORE `normalize_category`, which is
        // `value.trim().to_lowercase()` and therefore allocates a String on every call.
        // `Grid::finish` maps EVERY cell of a token_count x categorical_width grid through
        // here (23 x 41 = 943 per leaf at v4) and most cells are never populated, so the
        // allocation was being paid for the empty majority.
        //
        // Byte-identical by construction, not merely by test: `normalize_category("")` is
        // `""`, which the emptiness check below already turns into 0. Whitespace-only input is
        // unaffected -- it is not `is_empty()`, so it still takes the normalize path and still
        // returns 0 via that check.
        if value.is_empty() {
            return 0;
        }
        let normalized = normalize_category(value);
        if normalized.is_empty() {
            return 0;
        }
        if let Some(row) = self.vocab_index.get(&normalized) {
            return *row;
        }
        let mut hasher = Blake2bVar::new(8).expect("blake2b-8");
        hasher.update(normalized.as_bytes());
        let mut digest = [0u8; 8];
        hasher
            .finalize_variable(&mut digest)
            .expect("blake2b-8 output");
        let bucket = u64::from_be_bytes(digest) % self.oov_buckets;
        (self.oov_offset + bucket) as i32
    }

    fn species_info(&self, name: &str) -> Option<&SpeciesEntry> {
        let key = normalize_identifier(name);
        if key.is_empty() {
            return None;
        }
        self.species.get(&key)
    }

    /// `species_info` with a cosmetic-forme fallback to the base species.
    ///
    /// The twin of Python's `showdown._species_info_base_fallback`. gen3 randbats emit Unown as
    /// lettered cosmetic formes (`Unown-C`, `Unown-Z`, `Unown-Exclamation`, ...) which are NOT
    /// separate Pokedex entries, so the direct lookup misses and the mon encoded with blank types
    /// and zero base stats while Python emitted real values. Measured at 514 diverging cells over
    /// one 120-game replay.
    ///
    /// Deliberately NOT applied at every `species_info` call site. Python routes exactly four
    /// ENCODE PATHS through its fallback -- formechange types, type categories, base stats and
    /// expected stats -- which is **five call sites**, because `_encode_expected_stats` looks up
    /// twice (`battle_info` and `hp_info`). The native side mirrors that one-for-one: five
    /// `species_info_base_fallback` calls, three bare `species_info` calls. Counting paths and
    /// counting call sites give different numbers, and an earlier description of this change said
    /// "4 of 7 call sites" by conflating them.
    ///
    /// The three left bare pair with Python's three bare ones: transformed expected-HP base
    /// species, transformed original base HP, and the acting mon's user types. Python has a
    /// FOURTH bare call (`_is_grounded_for_spikes`) with no native counterpart at all -- spikes
    /// layers arrive precomputed on the token -- so the two site sets are 8 and 9, not equal.
    ///
    /// Parity means matching Python, including where Python is arguably inconsistent: it uses the
    /// fallback for `hp_info` in the expected-stats path but NOT in the transform path's twin.
    /// Widening this further would be a NEW divergence, not a fix.
    fn species_info_base_fallback(&self, name: &str) -> Option<&SpeciesEntry> {
        if let Some(info) = self.species_info(name) {
            return Some(info);
        }
        let canonical = canonical_gen3_randbat_species_id(name);
        if canonical.is_empty() {
            return None;
        }
        // Only retry when the collapse actually changed something, mirroring Python's
        // `if canonical and canonical != species`.
        if canonical == normalize_identifier(name) {
            return None;
        }
        self.species.get(&canonical)
    }

    fn move_info(&self, name: &str) -> Option<&MoveEntry> {
        let key = normalize_identifier(name);
        if key.is_empty() {
            return None;
        }
        self.moves.get(&key)
    }
}

// ---------------------------------------------------------------------------
// Row-input views
// ---------------------------------------------------------------------------

/// `showdown._condition_features`.
struct ConditionFeatures {
    hp_fraction: Option<f64>,
    status: String,
    fainted: bool,
}

fn condition_features(condition: Option<&str>) -> ConditionFeatures {
    let text = condition.unwrap_or("");
    let parts: Vec<&str> = text.split_whitespace().collect();
    let mut hp_fraction = None;
    if let Some(first) = parts.first() {
        if let Some((numerator, denominator)) = first.split_once('/') {
            if let (Ok(n), Ok(d)) = (numerator.parse::<f64>(), denominator.parse::<f64>()) {
                if d != 0.0 {
                    hp_fraction = Some((n / d).clamp(0.0, 1.0));
                }
            }
        } else if *first == "0" {
            hp_fraction = Some(0.0);
        }
    }
    let fainted = parts.iter().skip(1).any(|p| *p == "fnt");
    let status = parts
        .iter()
        .skip(1)
        .find(|p| **p != "fnt")
        .map(|p| p.to_string())
        .unwrap_or_else(|| "none".to_string());
    ConditionFeatures {
        hp_fraction,
        status,
        fainted,
    }
}

/// `showdown._level_from_details`.
///
/// A details string with NO `L` token means level 100, not "unknown". Showdown omits the token
/// when -- and only when -- the level is exactly 100 (`sim/pokemon.ts::getUpdatedDetails`:
/// ``name + (level === 100 ? '' : `, L${level}`)``). None is returned only when there is no
/// details string at all, which is the one case that carries no level information.
///
/// None therefore means "no level information", with one unreachable exception: an `L` token
/// too large for i64 fails to parse and also yields None, where Python's unbounded `int()`
/// would not. A level token would need 19 digits to get there.
///
/// This returned None for a level-100 mon until 2026-08-03. Of the three callers, the one in
/// `encode_expected_stats` treated None as "skip the whole block", so every L100 opponent
/// encoded with ELEVEN zeroed
/// numeric cells -- ten from that block plus NUMERIC_LEVEL, which `:1433` skips separately --
/// where Python wrote real values. Nine gen3 randbats species are L100 (Beautifly, Ditto, Ledian, Luvdisc, Magcargo, Nosepass,
/// Shedinja, Spinda, Unown), so it was not rare. Worse than merely missing: an all-zero stat
/// block is the DELIBERATE sentinel `_encode_transformed_expected_stats` writes for an
/// unidentifiable Transform target, so the native leaf was feeding the model a false positive
/// for a signal it had only ever seen mean something else.
fn level_from_details(details: Option<&str>) -> Option<i64> {
    let details = details?;
    if details.is_empty() {
        // Python's `if not details` treats "" the same as None. Kept distinct from the
        // no-L-token case below, which is a real level-100 mon.
        return None;
    }
    for part in details.split(',') {
        let token = part.trim();
        if let Some(rest) = token.strip_prefix('L') {
            if !rest.is_empty() && rest.chars().all(|c| c.is_ascii_digit()) {
                return rest.parse::<i64>().ok();
            }
        }
    }
    Some(100)
}

/// `showdown._gen3_stat` (integer arithmetic; all operands non-negative).
fn gen3_stat(base: i64, level: i64, ev: i64, iv: i64, hp: bool) -> i64 {
    let core = ((2 * base + iv + ev / 4) * level) / 100;
    if hp {
        core + level + 10
    } else {
        core + 5
    }
}

/// `gen3_damage.HIDDEN_POWER_IVS`, transcribed from the vendored `data/typechart.ts` `HPivs`
/// tables. Unlisted stats stay 31. Dark carries no overrides (all-31 IS its spread), which is
/// why the table returns an empty slice rather than None for it -- "no overrides" and "not a
/// Hidden Power set" take DIFFERENT branches in the Atk-zeroing rule below.
fn hidden_power_ivs(hp_type: &str) -> Option<&'static [(&'static str, i64)]> {
    Some(match hp_type {
        "bug" => &[("atk", 30), ("def", 30), ("spd", 30)],
        "dark" => &[],
        "dragon" => &[("atk", 30)],
        "electric" => &[("spa", 30)],
        "fighting" => &[("def", 30), ("spa", 30), ("spd", 30), ("spe", 30)],
        "fire" => &[("atk", 30), ("spa", 30), ("spe", 30)],
        "flying" => &[("hp", 30), ("atk", 30), ("def", 30), ("spa", 30), ("spd", 30)],
        "ghost" => &[("def", 30), ("spd", 30)],
        "grass" => &[("atk", 30), ("spa", 30)],
        "ground" => &[("spa", 30), ("spd", 30)],
        "ice" => &[("atk", 30), ("def", 30)],
        "poison" => &[("def", 30), ("spa", 30), ("spd", 30)],
        "psychic" => &[("atk", 30), ("spe", 30)],
        "rock" => &[("def", 30), ("spd", 30), ("spe", 30)],
        "steel" => &[("spd", 30)],
        "water" => &[("atk", 30), ("def", 30), ("spa", 30)],
        _ => return None,
    })
}

/// `gen3_damage.hidden_power_type`: the type suffix of the set's Hidden Power move, if any.
fn hidden_power_type_of(moves: &[String]) -> Option<&str> {
    // LAST match wins, matching the generator (`teams.ts` loops every move assigning `hpType`
    // with no `break`) and the Python twin `gen3_damage.hidden_power_type`. `find_map` took the
    // first, which diverges from both when a set carries two Hidden Powers -- unreachable in the
    // current pool, but the two implementations must not disagree on the rule.
    moves
        .iter()
        .rev()
        .find_map(|m| m.strip_prefix("hiddenpower").filter(|rest| !rest.is_empty()))
}

/// The generator's legal spread set. Any value outside it means the generator drifted or this
/// core was mis-called, and the Python twin RAISES rather than emit a plausible-but-wrong stat.
const LEGAL_HP_EVS: [i64; 5] = [85, 81, 77, 73, 69];
const LEGAL_ATK_EVS: [i64; 2] = [85, 0];
const LEGAL_HP_IVS: [i64; 2] = [31, 30];
// Def/SpA/SpD/Spe IVs are 31, or 30 where the Hidden Power type's `HPivs` entry lowers them.
// Twin of `showdown._LEGAL_NON_HP_IVS`.
const LEGAL_NON_HP_IVS: [i64; 2] = [31, 30];

/// One candidate variant's generator-exact stats. Twin of `gen3_damage.RandbatsSpread.stats`.
///
/// All six, deliberately: an earlier shape returned only `(hp, atk)`, which is why the Python
/// fix for the Hidden Power IV override could land without the native side following.
#[derive(Clone, Copy, Debug)]
struct RandbatsSpread {
    hp: i64,
    atk: i64,
    def: i64,
    spa: i64,
    spd: i64,
    spe: i64,
}

impl RandbatsSpread {
    fn non_hp(&self, stat: &str) -> i64 {
        match stat {
            "def" => self.def,
            "spa" => self.spa,
            "spd" => self.spd,
            "spe" => self.spe,
            _ => 0,
        }
    }
}

/// Native twin of `gen3_damage.randbats_spread_details`, returning all six stats the encoder
/// bands, not just `(hp, atk)`. Mirrors `data/random-battles/gen3/teams.ts` -- 85 EVs / 31 IVs / neutral
/// everywhere, then Hidden Power IV overrides, the first HP-trim loop, Atk zeroing, and the
/// second HP-trim pass.
///
/// Exists because at V4 the Python encoder stopped approximating this and started asking the
/// generator's own spread core (`showdown._variant_spread_stats`), while this crate kept the old
/// approximation -- so the two encoders disagreed on `NUMERIC_EXPECTED_{HP,ATK}_LOW` for the same
/// state. A model trained on Python-encoded rows and searched with a native leaf would read a
/// different stat band at the leaf than it was trained on. The approximation is wrong in two
/// specific ways, both measured: the trimmed-HP bound jumped straight to ev=0 (a full 85-EV
/// strip) where the generator removes 4 at a time and stops at the first value satisfying its
/// modular condition, and the zeroed-Atk bound hardcoded iv=0, missing the Hidden Power
/// `ivs.atk - 28` carry-through.
///
/// `Ok(None)` = not derivable, which the caller must treat as "abandon the whole band" rather
/// than "skip this candidate": an unevaluable candidate could BE the true variant, so excluding
/// it from a min/max would report a bound no real variant has.
fn randbats_spread_stats(
    base_stats: &HashMap<String, i64>,
    hp_base: i64,
    atk_base: i64,
    level: i64,
    moves: &[String],
    item: &str,
    has_physical_attack: bool,
) -> PyResult<Option<RandbatsSpread>> {
    let mut hp_ev = 85i64;
    let mut atk_ev = 85i64;
    let mut hp_iv = 31i64;
    let mut atk_iv = 31i64;
    // Def/SpA/SpD/Spe carry their own IVs, because the Hidden Power override lowers them too.
    // Keeping them here rather than at the call site is what stops this function being a
    // partial view of the generator's spread -- the shape that let the Python side get fixed
    // while the native side kept the old flat iv=31 (the "Rust spread fork" defect).
    let mut non_hp_ivs: [(&str, i64); 4] =
        [("def", 31), ("spa", 31), ("spd", 31), ("spe", 31)];
    let hp_type = hidden_power_type_of(moves).map(|t| t.to_string());
    if let Some(hp_type) = hp_type.as_deref() {
        // `HIDDEN_POWER_IVS.get(hp_type, {})` -- an UNRECOGNIZED type takes the same branch as
        // `dark`: no IV overrides, but `hp_type is not None` still holds, so Atk zeroing below
        // uses 31-28=3 rather than 0.
        //
        // Refusing here instead would arguably be safer in the abstract, and an earlier draft
        // did. It is the wrong call for this file: refusing makes the native encoder disagree
        // with Python on an input Python accepts, which is the exact train/serve divergence
        // this port exists to remove -- and it would be a divergence the parity test cannot
        // see, since the corpus never carries an unknown type. All 16 gen3 types are in the
        // table, so nothing reaches this branch today. If the fallback is wrong it is wrong in
        // gen3_damage.py, and that is where it should be fixed and then ported.
        let overrides = hidden_power_ivs(hp_type).unwrap_or(&[]);
        for (stat, value) in overrides {
            match *stat {
                "hp" => hp_iv = *value,
                "atk" => atk_iv = *value,
                other => {
                    for slot in non_hp_ivs.iter_mut() {
                        if slot.0 == other {
                            slot.1 = *value;
                        }
                    }
                }
            }
        }
    }
    let hp_value = |hp_ev: i64, hp_iv: i64| gen3_stat(hp_base, level, hp_ev, hp_iv, true);

    let has = |name: &str| moves.iter().any(|m| m == name);
    let has_substitute = has("substitute");
    let flail_reversal = has("flail") || has("reversal");
    let pinch_item = matches!(item, "salacberry" | "petayaberry" | "liechiberry");

    // First HP-trim loop (teams.ts "Prepare optimal HP"). The trailing `else break` is load
    // bearing: a set with none of these shapes leaves 85 EVs untouched.
    while hp_ev > 1 {
        let hp = hp_value(hp_ev, hp_iv);
        if has_substitute && flail_reversal {
            if hp % 4 > 0 {
                break;
            }
        } else if has_substitute && pinch_item {
            if hp % 4 == 0 {
                break;
            }
        } else if has("bellydrum") {
            if hp % 2 > 0 {
                break;
            }
        } else {
            break;
        }
        hp_ev -= 4;
    }

    // Minimize confusion damage: no physical attacks and no Transform -> zero Atk.
    if !has_physical_attack && !has("transform") {
        atk_ev = 0;
        // `(ivs.atk || 31) - 28` in the generator: a Hidden Power set keeps its overridden Atk
        // IV and drops 28 from it (31 -> 3, or 30 -> 2), it does NOT fall to 0.
        atk_iv = if hp_type.is_some() {
            (if atk_iv == 0 { 31 } else { atk_iv }) - 28
        } else {
            0
        };
    }

    // Second HP-trim pass.
    let mut hp = hp_value(hp_ev, hp_iv);
    if has_substitute && (has("endeavor") || flail_reversal) {
        if hp % 4 == 0 {
            hp_ev -= 4;
        }
    } else if has_substitute && pinch_item {
        while hp % 4 > 0 {
            hp_ev -= 4;
            hp = hp_value(hp_ev, hp_iv);
        }
    }

    if !LEGAL_HP_EVS.contains(&hp_ev)
        || !LEGAL_ATK_EVS.contains(&atk_ev)
        || !LEGAL_HP_IVS.contains(&hp_iv)
    {
        return Err(err(&format!(
            "randbats spread outside the generator's legal set (hp_ev={hp_ev}, atk_ev={atk_ev}, \
             hp_iv={hp_iv}); the generator has drifted or the spread core was mis-called -- \
             refusing to emit a plausible-but-wrong stat"
        )));
    }
    for (stat, iv) in non_hp_ivs.iter() {
        if !LEGAL_NON_HP_IVS.contains(iv) {
            return Err(err(&format!(
                "randbats spread outside the generator's legal set ({stat}_iv={iv}); the \
                 generator has drifted or the spread core was mis-called -- refusing to emit a \
                 plausible-but-wrong stat"
            )));
        }
    }
    // Shedinja (the only base-1-HP species): the engine pins max HP to 1.
    let hp_stat = if hp_base == 1 {
        1
    } else {
        hp_value(hp_ev, hp_iv)
    };
    let non_hp = |stat: &str| {
        let iv = non_hp_ivs
            .iter()
            .find(|(name, _)| *name == stat)
            .map(|(_, iv)| *iv)
            .unwrap_or(31);
        gen3_stat(
            base_stats.get(stat).copied().unwrap_or(0),
            level,
            85,
            iv,
            false,
        )
    };
    Ok(Some(RandbatsSpread {
        hp: hp_stat,
        atk: gen3_stat(atk_base, level, atk_ev, atk_iv, false),
        def: non_hp("def"),
        spa: non_hp("spa"),
        spd: non_hp("spd"),
        spe: non_hp("spe"),
    }))
}

// Sorted-by-normalized-key dedupe keeping the first-seen original string:
// `showdown._compact_belief_values`.
fn compact_belief_values(values: &[String], limit: Option<usize>) -> Vec<String> {
    let mut by_key: Vec<(String, String)> = Vec::new();
    for raw in values {
        let value = raw.trim();
        if value.is_empty() {
            continue;
        }
        let key = normalize_identifier(value);
        if key.is_empty() || by_key.iter().any(|(k, _)| *k == key) {
            continue;
        }
        by_key.push((key, value.to_string()));
    }
    by_key.sort_by(|a, b| a.0.cmp(&b.0));
    let mut compact: Vec<String> = by_key.into_iter().map(|(_, v)| v).collect();
    if let Some(limit) = limit {
        compact.truncate(limit);
    }
    compact
}

/// `showdown._known_or_possible_values`.
fn known_or_possible(known: Option<&str>, possible: &[String]) -> Vec<String> {
    match known {
        Some(value) if !value.is_empty() => vec![value.to_string()],
        _ => compact_belief_values(possible, None),
    }
}

/// `showdown._prioritized_belief_moves`.
fn prioritized_belief_moves(revealed: &[String], possible: &[String], limit: usize) -> Vec<String> {
    let mut values: Vec<String> = revealed.to_vec();
    let mut seen: Vec<String> = revealed
        .iter()
        .map(|m| normalize_identifier(m))
        .filter(|k| !k.is_empty())
        .collect();
    seen.sort();
    seen.dedup();
    for candidate in possible {
        if seen.len() >= limit {
            break;
        }
        let key = normalize_identifier(candidate);
        if key.is_empty() || seen.contains(&key) {
            continue;
        }
        values.push(candidate.clone());
        seen.push(key);
    }
    values
}

fn string_vec(value: &Value) -> Vec<String> {
    as_array(value).iter().map(str_or_empty).collect()
}

// ---------------------------------------------------------------------------
// The encoder
// ---------------------------------------------------------------------------

pub struct EncodedArrays {
    pub categorical: Vec<i32>,
    pub numeric: Vec<f64>,
    pub token_types: Vec<i16>,
    pub attention: Vec<u8>,
    pub legal: Vec<u8>,
}

struct Grid<'t> {
    tables: &'t Tables,
    categorical: Vec<String>,
    numeric: Vec<f64>,
    cat_width: usize,
    num_width: usize,
}

impl<'t> Grid<'t> {
    fn new(tables: &'t Tables) -> Self {
        let layout = &tables.layout;
        Grid {
            tables,
            categorical: vec![String::new(); layout.token_count * layout.categorical_width],
            numeric: vec![0.0; layout.token_count * layout.numeric_width],
            cat_width: layout.categorical_width,
            num_width: layout.numeric_width,
        }
    }

    fn set_cat(&mut self, token: usize, column: usize, value: impl Into<String>) {
        if column < self.cat_width {
            self.categorical[token * self.cat_width + column] = value.into();
        }
    }

    fn set_num(&mut self, token: usize, column: usize, value: f64) {
        if column < self.num_width {
            self.numeric[token * self.num_width + column] = value;
        }
    }

    fn finish(self) -> (Vec<i32>, Vec<f64>) {
        let categorical = self
            .categorical
            .iter()
            .map(|value| self.tables.vocab_encode(value))
            .collect();
        (categorical, self.numeric)
    }
}

struct MonToken<'a> {
    entry: &'a Value,
}

impl<'a> MonToken<'a> {
    fn species(&self) -> String {
        str_or_empty(get(self.entry, "species"))
    }
    fn condition(&self) -> Option<&str> {
        as_str(get(self.entry, "condition"))
    }
    fn active(&self) -> bool {
        as_bool(get(self.entry, "active"))
    }
    fn details(&self) -> Option<&str> {
        as_str(get(self.entry, "details"))
    }
    fn live_type_source(&self) -> Option<&str> {
        as_str(get(self.entry, "live_type_source")).filter(|value| !value.is_empty())
    }
    /// The mon's own request-side move ids (`ShowdownPokemon.moves`) — the fallback source for
    /// resolving generic Hidden Power's typed variant ("hiddenpowerice", ...); see
    /// `self_move_mechanics_id`.
    fn moves(&self) -> Vec<String> {
        as_array(get(self.entry, "moves"))
            .iter()
            .map(str_or_empty)
            .collect()
    }
    fn ability(&self) -> Option<&str> {
        as_str(get(self.entry, "ability"))
    }
    /// The request-known CURRENT-held item (`ShowdownPokemon.item`) — the self-side source for the
    /// self token's item bucket / revealed-item flag (`showdown._encode_pokemon_tokens`, the
    /// `role == "self"` branch). Empty once the request shows the mon holding nothing (Knock Off /
    /// Trick / consumed berry), which is how not-currently-held surfaces for free.
    fn item(&self) -> Option<&str> {
        as_str(get(self.entry, "item"))
    }
    fn stats(&self) -> Option<&Value> {
        let stats = get(self.entry, "stats");
        if stats.is_object() {
            Some(stats)
        } else {
            None
        }
    }
    fn stat(&self, key: &str) -> Option<i64> {
        self.stats().map(|stats| as_i64(get(stats, key)))
    }
}

pub fn encode_row(tables: &Tables, row_json: &str) -> PyResult<EncodedArrays> {
    let row: Value = serde_json::from_str(row_json).map_err(|e| err(format!("row JSON: {e}")))?;
    encode_row_value(tables, &row, None)
}

fn transition_row_count(layout: &Layout) -> PyResult<usize> {
    Ok(layout.token_count - layout.offset("transition")?)
}

/// Encode one row-inputs value, optionally consuming fold PRODUCTS natively
/// (in-crate; no Python payload crossing) for the history-derived cells:
/// turn-merged transition rows 23 through the schema-bound final row,
/// the stats-token tendency counters,
/// the per-opponent-mon tendency triple, the pinned Tier-2 conclusions, and
/// the transition extent of the attention mask.
pub(crate) fn encode_row_value(
    tables: &Tables,
    row: &Value,
    products: Option<&crate::fold::ProductsData>,
) -> PyResult<EncodedArrays> {
    let md = get(row, "observation_metadata");
    let pm = get(row, "public_materialization");
    if !md.is_object() || !pm.is_object() {
        return Err(err(
            "row inputs must carry observation_metadata and public_materialization objects",
        ));
    }
    let layout = &tables.layout;
    let row_schema = str_or_empty(get(row, "observation_schema_version"));
    if row_schema != layout.schema_version {
        return Err(err(format!(
            "row observation schema {row_schema:?} does not match encoder-table layout {:?}",
            layout.schema_version
        )));
    }

    let self_team = as_array(get(md, "self_team"));
    let opponent_team = as_array(get(md, "opponent_team"));
    let self_mons: Vec<MonToken> = self_team.iter().map(|entry| MonToken { entry }).collect();
    let opponent_mons: Vec<MonToken> = opponent_team
        .iter()
        .map(|entry| MonToken { entry })
        .collect();

    // --- legal action mask (sanctioned source: metadata action candidates). ---
    let mut legal = vec![0u8; layout.action_count];
    let candidates = as_array(get(md, "action_candidates"));
    for candidate in candidates {
        let index = as_i64(get(candidate, "action_index"));
        if index >= 0 && (index as usize) < layout.action_count && as_bool(get(candidate, "legal"))
        {
            legal[index as usize] = 1;
        }
    }

    // --- token type ids (constant per spec). ---
    let field_offset = layout.offset("field")?;
    let self_offset = layout.offset("self_pokemon")?;
    let opponent_offset = layout.offset("opponent_pokemon")?;
    let action_offset = layout.offset("action_candidates")?;
    let stats_offset = layout.offset("stats")?;
    let transition_offset = layout.offset("transition")?;
    let mut token_types = vec![0i16; layout.token_count];
    for index in 0..layout.token_count {
        token_types[index] = if index == field_offset {
            0
        } else if index >= self_offset && index < opponent_offset {
            1
        } else if index >= opponent_offset && index < action_offset {
            2
        } else if index >= action_offset && index < stats_offset {
            3
        } else if index >= stats_offset && index < transition_offset {
            5
        } else {
            6
        };
    }

    // --- attention mask. Transition extent: without fold products the stored
    // per-row surface has no event stream, so the merged-token count is not
    // derivable — rows stay masked (the documented stored-surface ceiling,
    // matching the Python reference backend fed the same inputs). With fold
    // products the extent is the filled turn-merged row count, exactly the
    // production `_attention_mask` computation. ---
    let mut attention = vec![0u8; layout.token_count];
    attention[field_offset] = 1;
    for slot in 0..(opponent_offset - self_offset) {
        attention[self_offset + slot] = (slot < self_mons.len()) as u8;
    }
    for slot in 0..(action_offset - opponent_offset) {
        attention[opponent_offset + slot] = (slot < opponent_mons.len()) as u8;
    }
    for index in action_offset..stats_offset {
        attention[index] = 1;
    }
    for index in stats_offset..transition_offset {
        attention[index] = layout.stats_block as u8;
    }
    if let Some(products) = products.filter(|_| !layout.is_v4()) {
        let transition_count = layout.token_count - transition_offset;
        let filled = products
            .turn_merged_tokens
            .len()
            .min(layout.transition_token_budget)
            .min(transition_count);
        for index in 0..transition_count {
            attention[transition_offset + index] = (index < filled) as u8;
        }
    }

    let mut grid = Grid::new(tables);

    encode_field_token(tables, &mut grid, md, pm, field_offset)?;
    encode_pokemon_tokens(
        tables,
        &mut grid,
        &self_mons,
        self_offset,
        opponent_offset - self_offset,
        Role::SelfTeam,
        md,
    )?;
    encode_pokemon_tokens(
        tables,
        &mut grid,
        &opponent_mons,
        opponent_offset,
        action_offset - opponent_offset,
        Role::Opponent,
        md,
    )?;
    encode_action_tokens(tables, &mut grid, md, pm, &self_mons, action_offset, &legal)?;
    // Stats token: role + presence. The tendency counters are history-derived:
    // zero without fold products (the stored-surface ceiling), real when the
    // fold state is supplied.
    if layout.stats_block {
        grid.set_cat(stats_offset, layout.cat_col("CATEGORY_ROLE")?, "stats");
        grid.set_num(stats_offset, layout.num_col("NUMERIC_PRESENT")?, 1.0);
    }
    if let Some(products) = products {
        // v4 has NO transition region, but the products still carry the tendency counters and
        // the pinned Tier-2 conclusions, which are current-state surfaces on the mon tokens.
        // write_history_cells owns both, and skips the transition rows itself at v4.
        write_history_cells(tables, &mut grid, products, md, &self_mons, &opponent_mons)?;
    }

    let (categorical, numeric) = grid.finish();
    Ok(EncodedArrays {
        categorical,
        numeric,
        token_types,
        attention,
        legal,
    })
}

// ---------------------------------------------------------------------------
// Field token
// ---------------------------------------------------------------------------

fn side_condition_features(counts: &Value, layout: &Layout) -> (f64, f64) {
    let hazards: i64 = layout
        .hazard_conditions
        .iter()
        .map(|name| as_i64(get(counts, name)))
        .sum();
    let screens = layout
        .screen_conditions
        .iter()
        .filter(|name| as_bool(get(counts, name)))
        .count() as f64;
    ((hazards as f64 / 3.0).min(1.0), (screens / 2.0).min(1.0))
}

/// `showdown._timed_condition_turns`, reconstructed from the materialization
/// payload (set turns + active counts + the boundary turn).
fn timed_condition_turns(pm: &Value, slot: &str, layout: &Layout) -> HashMap<String, i64> {
    let side = get(get(pm, "sides"), slot);
    let set_turns = get(side, "sideConditionSetTurns");
    let counts = get(side, "sideConditions");
    let turn = as_i64(get(pm, "turn"));
    let mut remaining = HashMap::new();
    if let Some(map) = set_turns.as_object() {
        for (condition, set_turn) in map {
            if !as_bool(get(counts, condition)) {
                continue;
            }
            let left = (layout.timed_condition_duration - (turn - as_i64(set_turn))).max(0);
            remaining.insert(condition.clone(), left);
        }
    }
    remaining
}

fn encode_field_token(
    tables: &Tables,
    grid: &mut Grid,
    md: &Value,
    pm: &Value,
    token: usize,
) -> PyResult<()> {
    let layout = &tables.layout;
    let request_kind = str_or_empty(get(md, "request_kind"));
    grid.set_cat(
        token,
        layout.cat_col("CATEGORY_PRIMARY")?,
        format!("request_kind:{request_kind}"),
    );
    grid.set_cat(token, layout.cat_col("CATEGORY_ROLE")?, "field");
    grid.set_num(token, layout.num_col("NUMERIC_PRESENT")?, 1.0);
    let weather = str_or_empty(get(md, "weather"));
    if !weather.is_empty() {
        grid.set_cat(
            token,
            layout.cat_col("CATEGORY_SECONDARY")?,
            format!("weather:{weather}"),
        );
    }
    let (self_haz, self_scr) =
        side_condition_features(get(md, "self_side_condition_counts"), layout);
    let (opp_haz, opp_scr) =
        side_condition_features(get(md, "opponent_side_condition_counts"), layout);
    grid.set_num(token, layout.num_col("NUMERIC_SELF_HAZARDS")?, self_haz);
    grid.set_num(token, layout.num_col("NUMERIC_OPP_HAZARDS")?, opp_haz);
    if let Some(column) = layout.num_col_opt("NUMERIC_SELF_SCREENS") {
        grid.set_num(token, column, self_scr);
    }
    if let Some(column) = layout.num_col_opt("NUMERIC_OPP_SCREENS") {
        grid.set_num(token, column, opp_scr);
    }
    let turn_number = as_i64(get(md, "turn_number"));
    if turn_number != 0 {
        grid.set_num(
            token,
            layout.num_col("NUMERIC_TURN_COUNT")?,
            (turn_number as f64 / 1000.0).min(1.0),
        );
    }
    let self_future = as_i64(get(md, "self_future_sight_turns"));
    if self_future != 0 {
        if let Some(column) = layout.num_col_opt("NUMERIC_SELF_FUTURE_SIGHT") {
            grid.set_num(token, column, (self_future as f64 / 2.0).min(1.0));
        }
    }
    let opp_future = as_i64(get(md, "opponent_future_sight_turns"));
    if opp_future != 0 {
        if let Some(column) = layout.num_col_opt("NUMERIC_OPP_FUTURE_SIGHT") {
            grid.set_num(token, column, (opp_future as f64 / 2.0).min(1.0));
        }
    }
    if layout.is_v3() {
        if as_bool(get(md, "self_sleep_clause_blocks")) {
            grid.set_num(
                token,
                layout.num_col("NUMERIC_SLEEP_CLAUSE_BLOCKS_SELF")?,
                1.0,
            );
        }
        if as_bool(get(md, "opponent_sleep_clause_blocks")) {
            grid.set_num(
                token,
                layout.num_col("NUMERIC_SLEEP_CLAUSE_BLOCKS_OPP")?,
                1.0,
            );
        }
        let self_wish_turns = as_i64(get(md, "self_wish_turns"));
        if self_wish_turns != 0 {
            grid.set_num(
                token,
                layout.num_col("NUMERIC_SELF_WISH_TURNS")?,
                (self_wish_turns as f64 / 2.0).min(1.0),
            );
        }
        let opponent_wish_turns = as_i64(get(md, "opponent_wish_turns"));
        if opponent_wish_turns != 0 {
            grid.set_num(
                token,
                layout.num_col("NUMERIC_OPP_WISH_TURNS")?,
                (opponent_wish_turns as f64 / 2.0).min(1.0),
            );
        }
    }
    if layout.is_v4() {
        // Part B credit block. `showdown.field_credit_values` derives these ONCE in Python and
        // publishes the settled numbers on the observation metadata, so this side reads them
        // rather than re-implementing the gen3 grounding rule — a re-derivation is exactly what
        // drifts silently between two languages.
        for (key, column) in [
            ("self_hazard_credit", "NUMERIC_SELF_HAZARD_CREDIT"),
            ("opponent_hazard_credit", "NUMERIC_OPP_HAZARD_CREDIT"),
            ("self_hazard_expected", "NUMERIC_SELF_HAZARD_EXPECTED"),
            ("opponent_hazard_expected", "NUMERIC_OPP_HAZARD_EXPECTED"),
            ("self_items_removed_credit", "NUMERIC_SELF_ITEMS_REMOVED_CREDIT"),
            ("opponent_items_removed_credit", "NUMERIC_OPP_ITEMS_REMOVED_CREDIT"),
        ] {
            let value = as_f64(get(md, key));
            if value != 0.0 {
                grid.set_num(token, layout.num_col(column)?, value.min(1.0));
            }
        }
    }
    if !layout.exact_state {
        return Ok(());
    }
    // Exact-state layer (`_encode_field_exact_state`).
    if as_bool(get(md, "self_sleep_clause_used")) {
        grid.set_num(token, layout.num_col("NUMERIC_SELF_SLEEP_CLAUSE")?, 1.0);
    }
    if as_bool(get(md, "opponent_sleep_clause_used")) {
        grid.set_num(token, layout.num_col("NUMERIC_OPP_SLEEP_CLAUSE")?, 1.0);
    }
    if !weather.is_empty() {
        let weather_turns = as_i64(get(md, "weather_turns_remaining"));
        grid.set_num(
            token,
            layout.num_col("NUMERIC_WEATHER_TURNS")?,
            (weather_turns as f64 / layout.timed_condition_duration as f64).min(1.0),
        );
        if as_bool(get(md, "weather_permanent")) {
            grid.set_num(token, layout.num_col("NUMERIC_WEATHER_PERMANENT")?, 1.0);
        }
    }
    let self_slot = str_or_empty(get(md, "showdown_slot"));
    let opponent_slot = str_or_empty(get(md, "opponent_showdown_slot"));
    let self_timed = timed_condition_turns(pm, &self_slot, layout);
    let opp_timed = timed_condition_turns(pm, &opponent_slot, layout);
    for (condition, self_col, opp_col) in &layout.timed_condition_slots {
        let self_turns = self_timed.get(condition).copied().unwrap_or(0);
        if self_turns != 0 {
            grid.set_num(
                token,
                *self_col,
                (self_turns as f64 / layout.timed_condition_duration as f64).min(1.0),
            );
        }
        let opp_turns = opp_timed.get(condition).copied().unwrap_or(0);
        if opp_turns != 0 {
            grid.set_num(
                token,
                *opp_col,
                (opp_turns as f64 / layout.timed_condition_duration as f64).min(1.0),
            );
        }
    }
    if as_bool(get(md, "self_wish_pending")) {
        grid.set_num(token, layout.num_col("NUMERIC_SELF_WISH_PENDING")?, 1.0);
    }
    if as_bool(get(md, "opponent_wish_pending")) {
        grid.set_num(token, layout.num_col("NUMERIC_OPP_WISH_PENDING")?, 1.0);
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Pokemon tokens (self team + opponent team with belief overlay)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq)]
enum Role {
    SelfTeam,
    Opponent,
}

fn gender_from_details(details: Option<&str>) -> Option<&str> {
    details.and_then(|value| {
        value
            .split(',')
            .map(str::trim)
            .find(|part| matches!(*part, "M" | "F"))
    })
}

fn live_type_slots(tables: &Tables, source: &str) -> Option<(String, Option<String>)> {
    let (kind, payload) = source.split_once(':')?;
    let payload = payload.trim();
    if payload.is_empty() {
        return None;
    }
    if kind == "type" {
        let mut types = payload
            .split('/')
            .map(str::trim)
            .filter(|value| !value.is_empty());
        let first = types.next()?;
        return Some((first.to_string(), types.next().map(str::to_string)));
    }
    if kind != "forme" {
        return None;
    }
    // Base-forme fallback: Python's twin (`showdown.py:5972`) goes through
    // `_species_info_base_fallback` here, so a cosmetic Unown retype must resolve.
    if let Some(info) = tables.species_info_base_fallback(payload) {
        if let Some(first) = info.types.first() {
            return Some((first.clone(), info.types.get(1).cloned()));
        }
    }
    match normalize_identifier(payload).as_str() {
        "castformsunny" => Some(("Fire".to_string(), None)),
        "castformrainy" => Some(("Water".to_string(), None)),
        "castformsnowy" => Some(("Ice".to_string(), None)),
        "castform" => Some(("Normal".to_string(), None)),
        _ => None,
    }
}

struct BeliefEntry<'a> {
    entry: &'a Value,
}

impl<'a> BeliefEntry<'a> {
    fn condition(&self) -> Option<&str> {
        as_str(get(self.entry, "condition"))
    }
    fn status(&self) -> Option<&str> {
        as_str(get(self.entry, "status"))
    }
    fn revealed_moves(&self) -> Vec<String> {
        string_vec(get(self.entry, "revealed_moves"))
    }
    fn revealed_ability(&self) -> Option<&str> {
        as_str(get(self.entry, "revealed_ability")).filter(|s| !s.is_empty())
    }
    fn revealed_item(&self) -> Option<&str> {
        as_str(get(self.entry, "revealed_item")).filter(|s| !s.is_empty())
    }
    fn possible(&self, key: &str) -> Vec<String> {
        string_vec(get(self.entry, key))
    }
    fn uncertainty(&self) -> f64 {
        as_f64(get(self.entry, "uncertainty"))
    }
    fn candidate_set_count(&self) -> i64 {
        as_i64(get(self.entry, "candidate_set_count"))
    }
    fn transformed(&self) -> bool {
        as_bool(get(self.entry, "transformed"))
    }
    fn transform_species(&self) -> Option<&str> {
        as_str(get(self.entry, "transform_species")).filter(|s| !s.is_empty())
    }
    fn sleep_turns(&self) -> i64 {
        as_i64(get(self.entry, "sleep_turns"))
    }
    fn rest_sleep(&self) -> bool {
        as_bool(get(self.entry, "rest_sleep"))
    }
    fn turns_active(&self) -> i64 {
        as_i64(get(self.entry, "turns_active"))
    }
    fn move_uses(&self) -> HashMap<String, i64> {
        let mut uses = HashMap::new();
        for pair in as_array(get(self.entry, "move_uses")) {
            let items = as_array(pair);
            if items.len() == 2 {
                uses.insert(str_or_empty(&items[0]), as_i64(&items[1]));
            }
        }
        uses
    }
    fn ruled_out_abilities(&self) -> Vec<String> {
        string_vec(get(self.entry, "ruled_out_abilities"))
    }
    fn candidate_variants(&self) -> &Vec<Value> {
        as_array(get(self.entry, "candidate_variants"))
    }
}

fn belief_by_species<'a>(overlay_side: &'a Value) -> HashMap<String, BeliefEntry<'a>> {
    let mut map = HashMap::new();
    for entry in as_array(overlay_side) {
        let species = normalize_identifier(&str_or_empty(get(entry, "species")));
        map.insert(species, BeliefEntry { entry });
    }
    map
}

/// `showdown._certain_opponent_ability`.
fn certain_opponent_ability(exact: &BeliefEntry) -> Option<String> {
    if let Some(revealed) = exact.revealed_ability() {
        return Some(revealed.to_string());
    }
    let ruled_out: Vec<String> = exact
        .ruled_out_abilities()
        .iter()
        .map(|a| normalize_identifier(a))
        .collect();
    let live: Vec<String> = exact
        .possible("possible_abilities")
        .into_iter()
        .filter(|ability| !ruled_out.contains(&normalize_identifier(ability)))
        .collect();
    if live.len() == 1 {
        Some(live[0].clone())
    } else {
        None
    }
}

/// `showdown._opponent_rest_wake_known`.
fn opponent_rest_wake_known(exact: &BeliefEntry) -> bool {
    if exact.revealed_ability().is_some() {
        return true;
    }
    let ruled_out: Vec<String> = exact
        .ruled_out_abilities()
        .iter()
        .map(|a| normalize_identifier(a))
        .collect();
    let candidates: Vec<String> = exact
        .possible("possible_abilities")
        .iter()
        .map(|a| normalize_identifier(a))
        .filter(|a| !ruled_out.contains(a))
        .collect();
    if candidates.is_empty() {
        return false;
    }
    !candidates.iter().any(|a| a == "earlybird")
}

fn encode_species_type_categories(
    tables: &Tables,
    grid: &mut Grid,
    token: usize,
    species: &str,
) -> PyResult<()> {
    let layout = &tables.layout;
    // Base-forme fallback: twin of `_encode_species_type_categories` (`showdown.py:5985`).
    // Without it every `unown*` but the base left CATEGORY_TYPE_1/2 unset.
    if let Some(info) = tables.species_info_base_fallback(species) {
        if let Some(first) = info.types.first() {
            grid.set_cat(
                token,
                layout.cat_col("CATEGORY_TYPE_1")?,
                format!("type:{first}"),
            );
        }
        if let Some(second) = info.types.get(1) {
            grid.set_cat(
                token,
                layout.cat_col("CATEGORY_TYPE_2")?,
                format!("type:{second}"),
            );
        }
    }
    Ok(())
}

fn encode_pokemon_stats(
    tables: &Tables,
    grid: &mut Grid,
    token: usize,
    species: &str,
    details: Option<&str>,
) -> PyResult<()> {
    let layout = &tables.layout;
    if let Some(level) = level_from_details(details) {
        grid.set_num(
            token,
            layout.num_col("NUMERIC_LEVEL")?,
            (level as f64 / 100.0).min(1.0),
        );
    }
    // Base-forme fallback: twin of the base-stat block at `showdown.py:6045`. Without it a
    // cosmetic Unown zeroed all six NUMERIC_BASE_* columns.
    if let Some(info) = tables.species_info_base_fallback(species) {
        for (stat, column) in &layout.base_stat_slots {
            let value = info.base_stats.get(stat).copied().unwrap_or(0);
            if value != 0 {
                grid.set_num(token, *column, (value as f64 / 200.0).min(1.0));
            }
        }
    }
    Ok(())
}

fn encode_actual_stats(
    tables: &Tables,
    grid: &mut Grid,
    token: usize,
    mon: &MonToken,
) -> PyResult<()> {
    let layout = &tables.layout;
    if mon.stats().is_none() {
        return Ok(());
    }
    for (stat, column) in &layout.actual_stat_slots {
        let value = mon.stat(stat).unwrap_or(0);
        if value != 0 {
            grid.set_num(
                token,
                *column,
                (value as f64 / layout.actual_stat_divisor).min(1.0),
            );
        }
    }
    Ok(())
}

fn encode_active_boosts(grid: &mut Grid, token: usize, boosts: &Value, layout: &Layout) {
    for (stat, column) in &layout.boost_stat_slots {
        let stage = as_i64(get(boosts, stat));
        if stage != 0 {
            grid.set_num(token, *column, (stage as f64 / 6.0).clamp(-1.0, 1.0));
        }
    }
}

fn encode_active_volatiles(
    tables: &Tables,
    grid: &mut Grid,
    token: usize,
    volatiles: &[String],
) -> PyResult<()> {
    let layout = &tables.layout;
    let mut sorted: Vec<String> = volatiles.to_vec();
    sorted.sort();
    sorted.dedup();
    let offset = layout.cat_col("CATEGORY_VOLATILE_OFFSET")?;
    for (index, name) in sorted.iter().take(layout.volatile_buckets).enumerate() {
        grid.set_cat(
            token,
            offset + index,
            format!("volatile:{}", normalize_identifier(name)),
        );
    }
    Ok(())
}

fn encode_belief_fact(
    tables: &Tables,
    grid: &mut Grid,
    token: usize,
    kind: &str,
    values: &[String],
) -> PyResult<()> {
    let layout = &tables.layout;
    let (offset, buckets) = match kind {
        "possible_ability" => (
            layout.cat_col("CATEGORY_BELIEF_ABILITY_OFFSET")?,
            layout.belief_ability_buckets,
        ),
        "possible_item" => (
            layout.cat_col("CATEGORY_BELIEF_ITEM_OFFSET")?,
            layout.belief_item_buckets,
        ),
        "possible_move" => (
            layout.cat_col("CATEGORY_BELIEF_MOVE_OFFSET")?,
            layout.belief_move_buckets,
        ),
        _ => return Err(err(format!("unsupported belief fact kind {kind}"))),
    };
    for (index, value) in compact_belief_values(values, Some(buckets))
        .iter()
        .enumerate()
    {
        grid.set_cat(
            token,
            offset + index,
            format!("belief:{kind}:{}", normalize_identifier(value)),
        );
    }
    Ok(())
}

/// `showdown._encode_expected_stats` (non-transformed branch) +
/// `_encode_transformed_expected_stats`.
#[allow(clippy::too_many_arguments)]
fn encode_expected_stats(
    tables: &Tables,
    grid: &mut Grid,
    token: usize,
    base_species: &str,
    battle_species: &str,
    details: Option<&str>,
    belief: Option<&BeliefEntry>,
    transformed: bool,
    transform_target: Option<&MonToken>,
) -> PyResult<()> {
    let layout = &tables.layout;
    let divisor = layout.actual_stat_divisor;
    if transformed {
        let target_stats = transform_target.and_then(|target| target.stats());
        let Some(stats) = target_stats else {
            return Ok(());
        };
        let all_present = ["atk", "def", "spa", "spd", "spe"]
            .iter()
            .all(|key| stats.get(*key).is_some());
        if !all_present {
            return Ok(());
        }
        for (stat, column_name) in [
            ("def", "NUMERIC_EXPECTED_DEF"),
            ("spa", "NUMERIC_EXPECTED_SPA"),
            ("spd", "NUMERIC_EXPECTED_SPD"),
            ("spe", "NUMERIC_EXPECTED_SPE"),
        ] {
            grid.set_num(
                token,
                layout.num_col(column_name)?,
                (as_i64(get(stats, stat)) as f64 / divisor).min(1.0),
            );
        }
        let atk_value = (as_i64(get(stats, "atk")) as f64 / divisor).min(1.0);
        for column_name in [
            "NUMERIC_EXPECTED_ATK",
            "NUMERIC_EXPECTED_ATK_LOW",
            "NUMERIC_EXPECTED_ATK_HIGH",
        ] {
            grid.set_num(token, layout.num_col(column_name)?, atk_value);
        }
        let level = level_from_details(details);
        let hp_base = tables
            .species_info(base_species)
            .and_then(|info| info.base_stats.get("hp").copied())
            .unwrap_or(0);
        if let Some(level) = level {
            if hp_base != 0 {
                let hp_value = (gen3_stat(hp_base, level, 85, 31, true) as f64 / divisor).min(1.0);
                for column_name in [
                    "NUMERIC_EXPECTED_HP",
                    "NUMERIC_EXPECTED_HP_LOW",
                    "NUMERIC_EXPECTED_HP_HIGH",
                ] {
                    grid.set_num(token, layout.num_col(column_name)?, hp_value);
                }
            }
        }
        return Ok(());
    }

    // `showdown.py:6778-6780` does this fix in TWO halves and both are load bearing:
    // `_level_from_details` returns 100 for a token-less details string, AND this caller
    // coerces a None level to 100 anyway -- "belt-and-suspenders ... rather than silently
    // zeroing this otherwise-deterministic block". Porting only the first half left
    // `details` of None or "" still zeroing all ten expected-stat columns, which is the very
    // sentinel collision this change exists to remove, just on a neighbouring input shape.
    //
    // The other two callers of level_from_details do NOT coerce, and must not: the one in
    // `encode_pokemon_stats` mirrors `if level is not None` (an absent level writes no
    // NUMERIC_LEVEL) and the one in the transformed-stats path mirrors `if level is None ...
    // return`. Only this one, matching `_encode_expected_stats`.
    let level = level_from_details(details).unwrap_or(100);
    // The scope limit that stood here -- `Tables::species_info` being a bare lookup with no twin of
    // Python's `_species_info_base_fallback`, so every `unown*` but the base returned early and
    // left all ten NUMERIC_EXPECTED_* columns at zero -- is CLOSED: both lookups below now go
    // through `species_info_base_fallback`.
    let Some(battle_info) = tables.species_info_base_fallback(battle_species) else {
        return Ok(());
    };
    let Some(hp_info) = tables.species_info_base_fallback(base_species) else {
        return Ok(());
    };
    let atk_base = battle_info.base_stats.get("atk").copied().unwrap_or(0);
    let hp_base = hp_info.base_stats.get("hp").copied().unwrap_or(0);
    let variants = belief
        .map(|b| b.candidate_variants())
        .filter(|v| !v.is_empty());
    // Generator-exact spreads for every candidate, computed once and shared by the
    // Def/SpA/SpD/Spe block below and the HP/Atk bands -- the twin of the Python pre-pass. These
    // four are NOT spread-invariant: the generator overwrites IVs from the carried Hidden Power
    // type's `HPivs` entry, lowering one or more of them to 30 on 716 of the 1682 real candidate
    // variants (42.6%).
    // Emitting the flat iv=31 value here while Python asked the spread core is precisely the
    // "Rust spread fork" defect -- the native encoder keeping an approximation the Python side
    // had already dropped.
    let mut exact_variant_spreads: Vec<RandbatsSpread> = Vec::new();
    if layout.is_v4() {
        if let Some(variants) = variants {
            for variant in variants {
                // `_variant_spread_stats` returns None when `moves` is not a list, and the
                // caller treats that as unevaluable. `as_array` flattens missing/null/scalar to
                // an EMPTY Vec, which would instead be evaluated as a moveless set -- a real
                // spread for a variant that does not exist. Check the raw value.
                if !get(variant, "moves").is_array() {
                    exact_variant_spreads.clear();
                    break;
                }
                let moves: Vec<String> = as_array(get(variant, "moves"))
                    .iter()
                    .map(|m| normalize_identifier(&str_or_empty(m)))
                    .collect();
                let item = normalize_identifier(&str_or_empty(get(variant, "item")));
                let has_physical = moves.iter().any(|move_id| {
                    tables
                        .move_info(move_id)
                        .map(|info| info.gen3_category == "Physical" && info.base_power > 0)
                        .unwrap_or(false)
                });
                match randbats_spread_stats(
                    &battle_info.base_stats,
                    // The BATTLE species' HP base, matching the Python twin, which hands the
                    // spread core `battle_info.base_stats` whole while taking the BASELINE's hp
                    // from `hp_info`. The two differ only on a forme change, and the transformed
                    // path returns before reaching here.
                    battle_info.base_stats.get("hp").copied().unwrap_or(0),
                    atk_base,
                    level,
                    &moves,
                    &item,
                    has_physical,
                )? {
                    Some(spread) => exact_variant_spreads.push(spread),
                    None => {
                        // Unevaluable candidate: abandon the WHOLE set rather than skip this one.
                        // A max over a strict subset can fall BELOW the true value if the skipped
                        // candidate was the true variant, which is unsound in the one direction
                        // this column claims to be safe in.
                        exact_variant_spreads.clear();
                        break;
                    }
                }
            }
        }
    }
    for (stat, column_name) in [
        ("def", "NUMERIC_EXPECTED_DEF"),
        ("spa", "NUMERIC_EXPECTED_SPA"),
        ("spd", "NUMERIC_EXPECTED_SPD"),
        ("spe", "NUMERIC_EXPECTED_SPE"),
    ] {
        let value = battle_info.base_stats.get(stat).copied().unwrap_or(0);
        if value != 0 {
            let emitted = if exact_variant_spreads.is_empty() {
                gen3_stat(value, level, 85, 31, false)
            } else {
                exact_variant_spreads
                    .iter()
                    .map(|spread| spread.non_hp(stat))
                    .max()
                    .unwrap_or_else(|| gen3_stat(value, level, 85, 31, false))
            };
            grid.set_num(
                token,
                layout.num_col(column_name)?,
                (emitted as f64 / divisor).min(1.0),
            );
        }
    }
    if atk_base == 0 || hp_base == 0 {
        return Ok(());
    }
    let atk_baseline = gen3_stat(atk_base, level, 85, 31, false);
    let hp_baseline = gen3_stat(hp_base, level, 85, 31, true);
    let (mut atk_low, mut atk_high) = (atk_baseline, atk_baseline);
    let (mut hp_low, mut hp_high) = (hp_baseline, hp_baseline);
    if let Some(variants) = variants {
        let mut atk_values: Vec<i64> = Vec::new();
        let mut hp_values: Vec<i64> = Vec::new();
        let pinch_berries = ["liechiberry", "petayaberry", "salacberry"];
        for (index, variant) in variants.iter().enumerate() {
            let moves: Vec<String> = as_array(get(variant, "moves"))
                .iter()
                .map(|m| normalize_identifier(&str_or_empty(m)))
                .collect();
            let item = normalize_identifier(&str_or_empty(get(variant, "item")));
            let has_physical = moves.iter().any(|move_id| {
                tables
                    .move_info(move_id)
                    .map(|info| info.gen3_category == "Physical" && info.base_power > 0)
                    .unwrap_or(false)
            });
            if layout.is_v4() {
                // Already computed by the pre-pass above, which the Def/SpA/SpD/Spe block needs
                // as well; an empty list there means some candidate was unevaluable, which lands
                // on the identical fallback either way. Substituting the baseline for a missing
                // candidate would report a bound partly derived from a value no real variant
                // has, and the model reads that as confidently as a true one; low == high ==
                // baseline is an honest "unknown", a fabricated range is not.
                let Some(spread) = exact_variant_spreads.get(index) else {
                    atk_values.clear();
                    hp_values.clear();
                    break;
                };
                atk_values.push(spread.atk);
                hp_values.push(spread.hp);
            } else {
                atk_values.push(if has_physical {
                    atk_baseline
                } else {
                    gen3_stat(atk_base, level, 0, 0, false)
                });
                let has = |name: &str| moves.iter().any(|m| m == name);
                let hp_trimmed = has("bellydrum")
                    || (has("substitute")
                        && (has("flail")
                            || has("reversal")
                            || pinch_berries.contains(&item.as_str())));
                hp_values.push(if hp_trimmed {
                    gen3_stat(hp_base, level, 0, 31, true)
                } else {
                    hp_baseline
                });
            }
        }
        // `unwrap_or(baseline)` is the empty-vector case, which is now reachable two ways: no
        // variants at all, and the v4 abandon-the-band break above. Both mean the same thing --
        // collapse to the baseline.
        atk_low = *atk_values.iter().min().unwrap_or(&atk_baseline);
        atk_high = *atk_values.iter().max().unwrap_or(&atk_baseline);
        hp_low = *hp_values.iter().min().unwrap_or(&hp_baseline);
        hp_high = *hp_values.iter().max().unwrap_or(&hp_baseline);
    }
    for (column_name, value) in [
        ("NUMERIC_EXPECTED_HP", hp_baseline),
        ("NUMERIC_EXPECTED_HP_LOW", hp_low),
        ("NUMERIC_EXPECTED_HP_HIGH", hp_high),
        ("NUMERIC_EXPECTED_ATK", atk_baseline),
        ("NUMERIC_EXPECTED_ATK_LOW", atk_low),
        ("NUMERIC_EXPECTED_ATK_HIGH", atk_high),
    ] {
        grid.set_num(
            token,
            layout.num_col(column_name)?,
            (value as f64 / divisor).min(1.0),
        );
    }
    Ok(())
}

fn encode_pokemon_tokens(
    tables: &Tables,
    grid: &mut Grid,
    mons: &[MonToken],
    offset: usize,
    limit: usize,
    role: Role,
    md: &Value,
) -> PyResult<()> {
    let layout = &tables.layout;
    let overlay = get(md, "belief_view");
    let (boosts, volatiles, toxic_stage) = match role {
        Role::SelfTeam => (
            get(md, "self_active_boosts"),
            string_vec(get(md, "self_active_volatiles")),
            as_i64(get(md, "self_toxic_stage")),
        ),
        Role::Opponent => (
            get(md, "opponent_active_boosts"),
            string_vec(get(md, "opponent_active_volatiles")),
            as_i64(get(md, "opponent_toxic_stage")),
        ),
    };
    // Belief maps: the opponent side is BOTH the belief-fact source and the
    // exact-state ledger; the self side only the exact-state ledger.
    let self_exact = belief_by_species(get(overlay, "self_pokemon"));
    let opponent_beliefs = belief_by_species(get(overlay, "opponent_pokemon"));
    // Transform copy targets (opponent side): our own team by normalized species.
    let self_team = as_array(get(md, "self_team"));
    let transform_targets: HashMap<String, MonToken> = self_team
        .iter()
        .map(|entry| {
            (
                normalize_identifier(&str_or_empty(get(entry, "species"))),
                MonToken { entry },
            )
        })
        .collect();

    let role_label = match role {
        Role::SelfTeam => "pokemon:self",
        Role::Opponent => "pokemon:opponent",
    };
    for (slot, candidate) in mons.iter().take(limit).enumerate() {
        let token = offset + slot;
        let species = candidate.species();
        let species_key = normalize_identifier(&species);
        let belief = match role {
            Role::SelfTeam => None,
            Role::Opponent => opponent_beliefs.get(&species_key),
        };
        let exact = match role {
            Role::SelfTeam => self_exact.get(&species_key),
            Role::Opponent => opponent_beliefs.get(&species_key),
        };

        // `_condition_features(belief.condition if belief is not None else candidate.condition)`
        let condition = condition_features(match belief {
            Some(b) => b.condition(),
            None => candidate.condition(),
        });
        let revealed_moves = belief.map(|b| b.revealed_moves()).unwrap_or_default();
        // #767: our own mons carry no set-source belief (they are fully known by design), so the
        // belief-derived reveals are empty and the self-token item/ability buckets +
        // NUMERIC_REVEALED_ITEM/ABILITY would encode NOTHING — the policy could not condition on
        // its OWN current item or ability. Populate them straight from the request-known candidate
        // fields (exactly how self stats/details already flow, direct from the request row, not
        // through the belief engine), as zero-uncertainty singletons (NUMERIC_UNCERTAINTY is
        // already 0.0 for self below). CURRENT-held semantics come for free: candidate.item is
        // empty once the request shows the mon holding nothing (Knock Off / Trick / consumed berry
        // / White Herb), so a stripped mon encodes not-currently-held (revealed_item -> None ->
        // NUMERIC_REVEALED_ITEM 0.0, empty bucket) while its still-known ability stays encoded.
        // Mirrors showdown._encode_pokemon_tokens' `if role == "self"` branch; nothing not
        // request-known is exposed and the opponent path is untouched.
        let (revealed_ability, revealed_item, possible_abilities, possible_items) = match role {
            Role::SelfTeam => {
                let revealed_ability = candidate.ability().filter(|s| !s.is_empty());
                let revealed_item = candidate.item().filter(|s| !s.is_empty());
                let possible_abilities = revealed_ability
                    .map(|a| vec![a.to_string()])
                    .unwrap_or_default();
                let possible_items = revealed_item
                    .map(|i| vec![i.to_string()])
                    .unwrap_or_default();
                (
                    revealed_ability,
                    revealed_item,
                    possible_abilities,
                    possible_items,
                )
            }
            Role::Opponent => {
                let revealed_ability = belief.and_then(|b| b.revealed_ability());
                let revealed_item = belief.and_then(|b| b.revealed_item());
                let possible_abilities = belief
                    .map(|b| b.possible("possible_abilities"))
                    .unwrap_or_default();
                let possible_items = belief
                    .map(|b| b.possible("possible_items"))
                    .unwrap_or_default();
                (
                    revealed_ability,
                    revealed_item,
                    possible_abilities,
                    possible_items,
                )
            }
        };
        let possible_moves = belief
            .map(|b| b.possible("possible_moves"))
            .unwrap_or_default();
        let ability_values = known_or_possible(revealed_ability, &possible_abilities);
        let item_values = known_or_possible(revealed_item, &possible_items);
        let candidate_set_count = belief.map(|b| b.candidate_set_count()).unwrap_or(0);
        let uncertainty = match role {
            Role::SelfTeam => 0.0,
            Role::Opponent => belief.map(|b| b.uncertainty()).unwrap_or(1.0),
        };
        // #766: a transformed mon (Ditto) fights as its target — encode species/types/base stats
        // from the copied identity so the model sees the effective battler, not Ditto's base
        // 48-across. The Transform flag lives in whichever per-mon ledger tracks this side's exact
        // state: the OPPONENT passes its set-source belief (carrying the flag) as `belief`, but the
        // SELF side passes only the exact belief (its set-source belief is None by design), so for
        // our own transformed Ditto `belief` is None and the copied identity would never surface
        // (self token stuck on ditto/Normal/48-across). Fall back to the exact belief when the
        // set-source belief lacks the flag. For the opponent both maps resolve to the same entry,
        // so this is a no-op there; a non-transformed self mon is likewise unchanged. Mirrors
        // showdown._encode_pokemon_tokens' transform_belief resolution.
        let transform_belief = match belief {
            Some(b) if b.transformed() => Some(b),
            _ => exact,
        };
        let transformed = transform_belief
            .map(|b| b.transformed() && b.transform_species().is_some())
            .unwrap_or(false);
        let enc_species = if transformed {
            transform_belief
                .and_then(|b| b.transform_species())
                .unwrap_or(species.as_str())
                .to_string()
        } else {
            species.clone()
        };

        grid.set_cat(
            token,
            layout.cat_col("CATEGORY_PRIMARY")?,
            format!("species:{enc_species}"),
        );
        encode_species_type_categories(tables, grid, token, &enc_species)?;
        if let Some(source) = candidate.live_type_source() {
            if let Some((type1, type2)) = live_type_slots(tables, source) {
                grid.set_cat(
                    token,
                    layout.cat_col("CATEGORY_TYPE_1")?,
                    format!("type:{type1}"),
                );
                grid.set_cat(
                    token,
                    layout.cat_col("CATEGORY_TYPE_2")?,
                    type2
                        .map(|value| format!("type:{value}"))
                        .unwrap_or_default(),
                );
            }
        }
        encode_pokemon_stats(tables, grid, token, &enc_species, candidate.details())?;
        if transformed {
            let original_hp = tables
                .species_info(&species)
                .and_then(|info| info.base_stats.get("hp").copied())
                .unwrap_or(0);
            if original_hp != 0 {
                grid.set_num(
                    token,
                    layout.num_col("NUMERIC_BASE_HP")?,
                    (original_hp as f64 / 200.0).min(1.0),
                );
            }
        }
        encode_actual_stats(tables, grid, token, candidate)?;
        if layout.is_v3() {
            match gender_from_details(candidate.details()) {
                Some("M") => grid.set_num(token, layout.num_col("NUMERIC_GENDER_MALE")?, 1.0),
                Some("F") => grid.set_num(token, layout.num_col("NUMERIC_GENDER_FEMALE")?, 1.0),
                _ => {}
            }
        }
        if candidate.active() {
            encode_active_boosts(grid, token, boosts, layout);
            // v4 pack A1: forced recharge joins the bag from the parser tracker on the
            // metadata. Injected here, not upstream, so v3's bag is untouched — the label has
            // no v3 vocabulary row and would hash into the OOV band there.
            let mut bag = volatiles.clone();
            if layout.is_v4() {
                let prefix = if role == Role::SelfTeam { "self" } else { "opponent" };
                if as_bool(get(md, &format!("{prefix}_must_recharge"))) {
                    bag.push("mustrecharge".to_string());
                }
            }
            encode_active_volatiles(tables, grid, token, &bag)?;
            if toxic_stage != 0 {
                grid.set_num(
                    token,
                    layout.num_col("NUMERIC_TOXIC_STAGE")?,
                    (toxic_stage as f64 / 15.0).min(1.0),
                );
            }
            if layout.is_v3() {
                let prefix = if role == Role::SelfTeam {
                    "self"
                } else {
                    "opponent"
                };
                for (suffix, column, divisor) in [
                    ("stall_counter", "NUMERIC_STALL_COUNTER", 8.0),
                    ("confusion_elapsed", "NUMERIC_CONFUSION_TURNS", 5.0),
                    ("encore_elapsed", "NUMERIC_ENCORE_TURNS", 6.0),
                    ("wrap_trap_elapsed", "NUMERIC_WRAP_TRAP_TURNS", 5.0),
                ] {
                    let value = as_i64(get(md, &format!("{prefix}_{suffix}")));
                    if value != 0 {
                        grid.set_num(
                            token,
                            layout.num_col(column)?,
                            (value as f64 / divisor).min(1.0),
                        );
                    }
                }
                if as_bool(get(md, &format!("{prefix}_meanlook_trap"))) {
                    grid.set_num(token, layout.num_col("NUMERIC_MEANLOOK_TRAP")?, 1.0);
                }
            }
            if layout.is_v4() {
                let prefix = if role == Role::SelfTeam { "self" } else { "opponent" };
                for (key, column) in [
                    ("truant_loaf", "NUMERIC_TRUANT_LOAF"),
                    ("choice_locked", "NUMERIC_CHOICE_LOCKED"),
                    ("item_swapped", "NUMERIC_ITEM_SWAPPED"),
                ] {
                    if as_bool(get(md, &format!("{prefix}_{key}"))) {
                        grid.set_num(token, layout.num_col(column)?, 1.0);
                    }
                }
                for (key, column) in [
                    ("last_damage_dealt", "NUMERIC_LAST_DAMAGE_DEALT"),
                    ("last_damage_taken", "NUMERIC_LAST_DAMAGE_TAKEN"),
                ] {
                    let value = as_f64(get(md, &format!("{prefix}_{key}")));
                    if value > 0.0 {
                        grid.set_num(token, layout.num_col(column)?, value.min(1.0));
                    }
                }
                // A2: three arrival/identity states. The switch and Baton-Pass sentinels are
                // positive facts, not padding — see the Python write site.
                let last_move = str_or_empty(get(md, &format!("{prefix}_last_used_move")));
                if !last_move.is_empty() && layout.feature_pack_last_move {
                    let label = if normalize_identifier(&last_move) == "switch" {
                        if as_bool(get(md, &format!("{prefix}_arrived_by_baton_pass"))) {
                            "lastmove:batonpass".to_string()
                        } else {
                            "lastmove:switch".to_string()
                        }
                    } else {
                        format!("move:{}", normalize_identifier(&last_move))
                    };
                    grid.set_cat(token, layout.cat_col("CATEGORY_LAST_USED_MOVE")?, label);
                }
                // A4: the CURRENT Trace copy, cleared on switch-out. Never the belief's
                // persistent revealed-ability channel, which holds the last-ever-traced one.
                let traced = str_or_empty(get(md, &format!("{prefix}_traced_ability")));
                if !traced.is_empty() {
                    grid.set_cat(
                        token,
                        layout.cat_col("CATEGORY_TRACED_ABILITY")?,
                        format!("ability:{}", normalize_identifier(&traced)),
                    );
                }
            }
        }
        // The matchup-conditional pair, on ALL SIX opponent tokens (not just the active one):
        // it answers "which of their mons has been willing to face what I have out", which is
        // a question about their bench too. Written in the always-run path because its source
        // is the observation METADATA — already conditioned on our current active by
        // normalize — not FoldProducts; behind the products gate it silently read zero on the
        // boundary-surface encode while Python wrote real values. Shares the tendency block's
        // mask: same channel, conditioned.
        if layout.is_v4() && role == Role::Opponent && layout.stats_block {
            let matchup_key = normalize_identifier(&candidate.species());
            let evidence = get(md, "opponent_matchup_switch_evidence");
            if let Some(cell) = evidence.get(&matchup_key).and_then(|v| v.as_array()) {
                for (index, column) in [
                    "NUMERIC_MON_SWITCHED_VS_ACTIVE",
                    "NUMERIC_MON_STAYED_VS_ACTIVE",
                ]
                .iter()
                .enumerate()
                {
                    let count = cell.get(index).map(as_i64).unwrap_or(0);
                    if count != 0 {
                        grid.set_num(
                            token,
                            layout.num_col(column)?,
                            (count as f64 / layout.matchup_count_divisor).min(1.0),
                        );
                    }
                }
            }
        }
        let status = belief
            .and_then(|b| b.status())
            .map(|s| s.to_string())
            .unwrap_or_else(|| condition.status.clone());
        grid.set_cat(
            token,
            layout.cat_col("CATEGORY_SECONDARY")?,
            format!("status:{status}"),
        );
        grid.set_cat(token, layout.cat_col("CATEGORY_ROLE")?, role_label);
        encode_belief_fact(tables, grid, token, "possible_ability", &ability_values)?;
        encode_belief_fact(tables, grid, token, "possible_item", &item_values)?;
        let bucket_moves = compact_belief_values(
            &prioritized_belief_moves(&revealed_moves, &possible_moves, layout.belief_move_buckets),
            Some(layout.belief_move_buckets),
        );
        encode_belief_fact(tables, grid, token, "possible_move", &bucket_moves)?;
        grid.set_num(
            token,
            layout.num_col("NUMERIC_HP_FRACTION")?,
            condition.hp_fraction.unwrap_or(0.0),
        );
        grid.set_num(
            token,
            layout.num_col("NUMERIC_ACTIVE")?,
            if candidate.active() { 1.0 } else { 0.0 },
        );
        grid.set_num(
            token,
            layout.num_col("NUMERIC_LEGAL")?,
            if condition.fainted { 0.0 } else { 1.0 },
        );
        grid.set_num(token, layout.num_col("NUMERIC_PRESENT")?, 1.0);
        grid.set_num(
            token,
            layout.num_col("NUMERIC_REVEALED_MOVE_COUNT")?,
            revealed_moves.len() as f64,
        );
        grid.set_num(
            token,
            layout.num_col("NUMERIC_CANDIDATE_SET_COUNT")?,
            candidate_set_count as f64,
        );
        grid.set_num(token, layout.num_col("NUMERIC_UNCERTAINTY")?, uncertainty);
        grid.set_num(
            token,
            layout.num_col("NUMERIC_POSSIBLE_ABILITY_COUNT")?,
            ability_values.len() as f64,
        );
        grid.set_num(
            token,
            layout.num_col("NUMERIC_POSSIBLE_ITEM_COUNT")?,
            item_values.len() as f64,
        );
        grid.set_num(
            token,
            layout.num_col("NUMERIC_POSSIBLE_MOVE_COUNT")?,
            possible_moves.len() as f64,
        );
        grid.set_num(
            token,
            layout.num_col("NUMERIC_REVEALED_ABILITY")?,
            if revealed_ability.is_some() { 1.0 } else { 0.0 },
        );
        grid.set_num(
            token,
            layout.num_col("NUMERIC_REVEALED_ITEM")?,
            if revealed_item.is_some() { 1.0 } else { 0.0 },
        );
        if layout.exact_state {
            // `_encode_mon_exact_state`.
            if let Some(exact) = exact {
                if status == "slp" {
                    grid.set_num(
                        token,
                        layout.num_col("NUMERIC_SLEEP_TURNS")?,
                        (exact.sleep_turns() as f64 / 5.0).min(1.0),
                    );
                    if exact.rest_sleep() {
                        grid.set_num(token, layout.num_col("NUMERIC_REST_SLEEP")?, 1.0);
                        let wake_known = match role {
                            Role::SelfTeam => true,
                            Role::Opponent => opponent_rest_wake_known(exact),
                        };
                        if wake_known {
                            grid.set_num(token, layout.num_col("NUMERIC_WAKE_KNOWN")?, 1.0);
                        }
                    }
                }
                if candidate.active() && exact.turns_active() != 0 {
                    grid.set_num(
                        token,
                        layout.num_col("NUMERIC_TURNS_ACTIVE")?,
                        (exact.turns_active() as f64 / layout.stat_count_divisor).min(1.0),
                    );
                }
            }
            let trap_ability = match role {
                Role::SelfTeam => candidate.ability().map(|a| a.to_string()),
                Role::Opponent => exact.and_then(certain_opponent_ability),
            };
            if let Some(ability) = trap_ability {
                let key = normalize_identifier(&ability);
                if !key.is_empty()
                    && layout.trap_abilities.iter().any(|t| *t == key)
                    && !condition.fainted
                    && !candidate.active()
                {
                    grid.set_num(token, layout.num_col("NUMERIC_TRAPPER_ALIVE")?, 1.0);
                }
            }
            // Substitute HP fraction (v2.1+): active mon with a live sub.
            let has_sub = volatiles
                .iter()
                .any(|name| normalize_identifier(name) == "substitute");
            if candidate.active() && has_sub {
                let max_hp = candidate.stat("hp").unwrap_or(0);
                let fraction = if max_hp > 0 {
                    (max_hp / 4) as f64 / max_hp as f64
                } else {
                    0.25
                };
                grid.set_num(token, layout.num_col("NUMERIC_SUB_HP_FRACTION")?, fraction);
            }
            if role == Role::Opponent {
                // `_encode_opponent_move_pp_fractions` (with validity bits).
                if let Some(exact) = exact {
                    let revealed_keys: Vec<String> = exact
                        .revealed_moves()
                        .iter()
                        .map(|m| normalize_identifier(m))
                        .filter(|k| !k.is_empty())
                        .collect();
                    if !revealed_keys.is_empty() {
                        let uses = exact.move_uses();
                        let pp_offset = layout.num_col("NUMERIC_OPP_MOVE_PP_OFFSET")?;
                        let valid_offset = layout.num_col("NUMERIC_OPP_MOVE_PP_VALID_OFFSET")?;
                        for (index, bucket_move) in bucket_moves
                            .iter()
                            .take(layout.belief_move_buckets)
                            .enumerate()
                        {
                            let key = normalize_identifier(bucket_move);
                            if !revealed_keys.contains(&key) {
                                continue;
                            }
                            grid.set_num(token, valid_offset + index, 1.0);
                            let max_pp =
                                tables.move_info(&key).map(|info| info.max_pp).unwrap_or(0);
                            if max_pp <= 0 {
                                continue;
                            }
                            let remaining = (max_pp - uses.get(&key).copied().unwrap_or(0)).max(0);
                            grid.set_num(
                                token,
                                pp_offset + index,
                                remaining as f64 / max_pp as f64,
                            );
                        }
                    }
                }
                let transform_target = if transformed {
                    transform_targets.get(&normalize_identifier(&enc_species))
                } else {
                    None
                };
                encode_expected_stats(
                    tables,
                    grid,
                    token,
                    &species,
                    &enc_species,
                    candidate.details(),
                    exact,
                    transformed,
                    transform_target,
                )?;
            }
        }
        // Tendency triple + pinned Tier-2 conclusions: history-derived,
        // not reconstructable from the stored per-row surface — left zero.
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Action-candidate tokens
// ---------------------------------------------------------------------------

/// Merge `pm.selfActiveMoves` (pp/maxpp/disabled) with the metadata action
/// candidates (every request slot, incl. PP-less pseudo-moves like recharge)
/// — the mirror of `golden_encoder_backends._synthesized_request`.
struct RequestMove {
    name: String,
    /// The request's display move name (`move` field, e.g. "Hidden Power Fighting 70"), the
    /// authoritative source for resolving generic Hidden Power's typed variant. The golden-corpus
    /// materialization omits it (only id/pp/maxpp/disabled), so it is usually None and resolution
    /// falls back to the mon's own typed move id; see `self_move_mechanics_id`.
    display: Option<String>,
    disabled: bool,
    pp_fraction: Option<f64>,
}

fn request_moves(md: &Value, pm: &Value) -> Vec<RequestMove> {
    let payload_moves = as_array(get(pm, "selfActiveMoves"));
    let mut move_candidates: Vec<&Value> = as_array(get(md, "action_candidates"))
        .iter()
        .filter(|c| get(c, "kind").as_str() == Some("move"))
        .collect();
    move_candidates.sort_by_key(|c| as_i64(get(c, "action_index")));
    let mut moves = Vec::new();
    let mut cursor = 0usize;
    for candidate in move_candidates {
        let slot = as_i64(get(candidate, "move_slot"));
        let move_name = str_or_empty(get(candidate, "move_name"));
        if move_name == format!("slot:{slot}") {
            break; // absent request slot — request move lists are dense prefixes
        }
        let move_id = str_or_empty(get(candidate, "move_id"));
        let payload = payload_moves
            .get(cursor)
            .filter(|entry| get(entry, "id").as_str() == Some(move_id.as_str()));
        match payload {
            Some(entry) => {
                cursor += 1;
                let pp = get(entry, "pp");
                let maxpp = get(entry, "maxpp");
                let pp_fraction = if pp.is_number() && maxpp.is_number() && as_f64(maxpp) != 0.0 {
                    Some((as_f64(pp) / as_f64(maxpp)).clamp(0.0, 1.0))
                } else {
                    None
                };
                moves.push(RequestMove {
                    name: str_or_empty(get(entry, "id")),
                    display: as_str(get(entry, "move")).map(str::to_string),
                    disabled: as_bool(get(entry, "disabled")),
                    pp_fraction,
                });
            }
            None => moves.push(RequestMove {
                name: move_name,
                display: None,
                disabled: as_bool(get(candidate, "disabled")),
                pp_fraction: None,
            }),
        }
    }
    moves
}

/// `dex.resolve_move_base_power`.
///
/// Return/Frustration scale with happiness, which gen3 randbats leaves at the engine default 255
/// (-> 102 / 1). Their static dex base power is a 0 placeholder, so this resolves the constant at
/// encode time, mirroring `dex._HAPPINESS_BASE_POWER` (checked FIRST, before the HP-fraction guard,
/// exactly as Python does). This is deliberately NOT baked into the exported table: the raw
/// `MoveEntry.base_power` field is also read as the static dex value by the tier2 `is_physical`
/// heuristic (`info.base_power > 0`, mirroring showdown._is_physical_attack), which must stay 0 for
/// Return to keep byte-parity with Python. See scripts/export_encoder_tables.py for the split.
fn resolve_move_base_power(info: &MoveEntry, user_hp_fraction: Option<f64>) -> i64 {
    match info.id.as_str() {
        "return" => return 102,
        "frustration" => return 1,
        _ => {}
    }
    let Some(fraction) = user_hp_fraction else {
        return info.base_power;
    };
    let fraction = fraction.clamp(0.0, 1.0);
    if info.id == "reversal" || info.id == "flail" {
        let scaled = (48.0 * fraction) as i64;
        return if scaled <= 1 {
            200
        } else if scaled <= 4 {
            150
        } else if scaled <= 9 {
            100
        } else if scaled <= 16 {
            80
        } else if scaled <= 32 {
            40
        } else {
            20
        };
    }
    if info.id == "eruption" || info.id == "waterspout" {
        return ((150.0 * fraction) as i64).max(1);
    }
    info.base_power
}

/// `dex.resolve_move_effect` (Curse type-dependence).
fn resolve_move_effect(info: &MoveEntry, user_types: &[String]) -> (String, i64, f64) {
    if info.id == "curse" {
        if user_types.iter().any(|t| t.to_lowercase() == "ghost") {
            return ("curse".to_string(), 100, 0.5);
        }
        return ("curse_setup".to_string(), 100, 0.0);
    }
    (
        info.effect_label.clone(),
        info.effect_chance,
        info.self_hp_cost,
    )
}

/// `showdown._HIDDEN_POWER_TYPES`: the 16 gen3 Hidden Power types.
const HIDDEN_POWER_TYPES: [&str; 16] = [
    "bug", "dark", "dragon", "electric", "fighting", "fire", "flying", "ghost", "grass", "ground",
    "ice", "poison", "psychic", "rock", "steel", "water",
];

/// `showdown._hidden_power_variant_from_name`: typed Hidden Power id from a request's display name.
/// "Hidden Power Fighting 70" -> "hiddenpowerfighting". None if no recognizable HP type is present.
fn hidden_power_variant_from_name(display_name: Option<&str>) -> Option<String> {
    let lowered = display_name?.to_lowercase();
    // Mirror re.findall(r"[a-z]+", ...): scan maximal runs of ascii lowercase letters.
    for token in lowered.split(|c: char| !c.is_ascii_lowercase()) {
        if !token.is_empty() && HIDDEN_POWER_TYPES.contains(&token) {
            return Some(format!("hiddenpower{token}"));
        }
    }
    None
}

/// `showdown._self_move_mechanics_id`: the move id to look up for SELF action-token MECHANICS
/// (type / base power / damage class).
///
/// Hidden Power's request keys `id` to the generic family ("hiddenpower"), whose dex entry is a
/// 0-power Normal placeholder. The real typed identity is self-observable: authoritatively from the
/// display `move` field ("Hidden Power Fighting 70"), and, as a fallback, from the mon's own typed
/// move id in the request side list ("hiddenpowerice", which Showdown derives from its IVs). Resolve
/// the typed variant for the mechanics lookup ONLY; the action token's move IDENTITY
/// (CATEGORY_PRIMARY = `move:hiddenpower`) stays generic and checkpoint-stable. Every non-Hidden
/// Power move passes straight through.
fn self_move_mechanics_id(entry: &RequestMove, own_move_ids: &[String]) -> String {
    if normalize_identifier(&entry.name) != "hiddenpower" {
        return entry.name.clone();
    }
    if let Some(typed) = hidden_power_variant_from_name(entry.display.as_deref()) {
        return typed;
    }
    for candidate in own_move_ids {
        let normalized = normalize_identifier(candidate);
        if normalized.starts_with("hiddenpower") && normalized.len() > "hiddenpower".len() {
            return normalized;
        }
    }
    entry.name.clone()
}

fn encode_move_mechanics(
    tables: &Tables,
    grid: &mut Grid,
    token: usize,
    move_name: &str,
    user_types: &[String],
    user_hp_fraction: Option<f64>,
) -> PyResult<()> {
    let layout = &tables.layout;
    let Some(info) = tables.move_info(move_name) else {
        return Ok(());
    };
    let base_power = resolve_move_base_power(info, user_hp_fraction);
    // Struggle is TYPELESS from Generation II onward: neutral vs every type (it
    // HITS Ghosts) and grants no STAB. The Showdown dex still records Struggle as
    // Normal-type, so emit the enumerated typeless token `type:???` directly, in
    // byte-parity with the Python reference encoder (_encode_move_mechanics) and
    // mirroring gen3 Curse (dex type already "???"). Matches the engine fix
    // (third_party/poke-engine-gen3-struggle-typeless.patch).
    let move_type_token = if info.id == "struggle" {
        "???"
    } else {
        info.move_type.as_str()
    };
    grid.set_cat(
        token,
        layout.cat_col("CATEGORY_TYPE_1")?,
        format!("type:{move_type_token}"),
    );
    grid.set_cat(
        token,
        layout.cat_col("CATEGORY_MOVE_CATEGORY")?,
        format!("move_category:{}", info.gen3_category),
    );
    grid.set_cat(
        token,
        layout.cat_col("CATEGORY_MOVE_PRIORITY")?,
        format!("move_priority:{}", info.priority),
    );
    grid.set_num(
        token,
        layout.num_col("NUMERIC_BASE_POWER")?,
        (base_power as f64 / 200.0).min(1.0),
    );
    grid.set_num(
        token,
        layout.num_col("NUMERIC_PRIORITY")?,
        (info.priority as f64 / 5.0).clamp(-1.0, 1.0),
    );
    grid.set_num(
        token,
        layout.num_col("NUMERIC_ACCURACY")?,
        if info.accuracy != 0.0 {
            info.accuracy / 100.0
        } else {
            1.0
        },
    );
    let (effect_label, effect_chance, self_hp_cost) = resolve_move_effect(info, user_types);
    if !effect_label.is_empty() {
        grid.set_cat(
            token,
            layout.cat_col("CATEGORY_MOVE_EFFECT")?,
            format!("move_effect:{effect_label}"),
        );
    }
    grid.set_num(
        token,
        layout.num_col("NUMERIC_EFFECT_CHANCE")?,
        (effect_chance as f64 / 100.0).min(1.0),
    );
    grid.set_num(
        token,
        layout.num_col("NUMERIC_SELF_HP_COST")?,
        self_hp_cost.clamp(0.0, 1.0),
    );
    Ok(())
}

fn encode_action_tokens(
    tables: &Tables,
    grid: &mut Grid,
    md: &Value,
    pm: &Value,
    self_mons: &[MonToken],
    action_offset: usize,
    legal: &[u8],
) -> PyResult<()> {
    let layout = &tables.layout;
    let moves = request_moves(md, pm);
    let active = self_mons.iter().find(|mon| mon.active());
    let user_types: Vec<String> = active
        .and_then(|mon| tables.species_info(&mon.species()))
        .map(|info| info.types.clone())
        .unwrap_or_default();
    let user_hp_fraction = active.and_then(|mon| condition_features(mon.condition()).hp_fraction);
    // The acting mon's own typed move ids ("hiddenpowerice", ...) — the request-side fallback for
    // resolving generic Hidden Power's real type/base power (see self_move_mechanics_id).
    let own_move_ids: Vec<String> = active.map(|mon| mon.moves()).unwrap_or_default();

    for move_index in 0..layout.move_action_count {
        let token = action_offset + move_index;
        let entry = moves.get(move_index);
        let move_name = entry
            .map(|m| m.name.clone())
            .unwrap_or_else(|| format!("slot:{}", move_index + 1));
        let disabled = entry.map(|m| m.disabled).unwrap_or(true);
        grid.set_cat(
            token,
            layout.cat_col("CATEGORY_PRIMARY")?,
            format!("move:{move_name}"),
        );
        grid.set_cat(token, layout.cat_col("CATEGORY_SECONDARY")?, "action:move");
        grid.set_cat(token, layout.cat_col("CATEGORY_ROLE")?, "action");
        grid.set_cat(
            token,
            layout.cat_col("CATEGORY_SLOT")?,
            format!("move_slot:{}", move_index + 1),
        );
        if let Some(entry) = entry {
            // Identity (CATEGORY_PRIMARY above) stays generic; only the MECHANICS lookup resolves
            // Hidden Power's typed variant so its true type / base power / damage class land on the
            // acting mon's decision surface.
            let mechanics_name = self_move_mechanics_id(entry, &own_move_ids);
            encode_move_mechanics(
                tables,
                grid,
                token,
                &mechanics_name,
                &user_types,
                user_hp_fraction,
            )?;
            grid.set_num(
                token,
                layout.num_col("NUMERIC_MOVE_PP_FRACTION")?,
                entry.pp_fraction.unwrap_or(1.0),
            );
        }
        grid.set_num(
            token,
            layout.num_col("NUMERIC_LEGAL")?,
            if legal.get(move_index).copied().unwrap_or(0) != 0 {
                1.0
            } else {
                0.0
            },
        );
        grid.set_num(
            token,
            layout.num_col("NUMERIC_PRESENT")?,
            if entry.is_some() { 1.0 } else { 0.0 },
        );
        grid.set_num(
            token,
            layout.num_col("NUMERIC_ACTIVE")?,
            if disabled { 0.0 } else { 1.0 },
        );
    }

    // Switch candidates: dense decode over non-active team members in order.
    let active_team_index = self_mons.iter().position(|mon| mon.active());
    let switch_targets: Vec<usize> = match active_team_index {
        Some(active_index) if self_mons.len() >= 2 => (0..self_mons.len())
            .filter(|index| *index != active_index)
            .collect(),
        _ => Vec::new(),
    };
    let switch_count = layout.action_count - layout.move_action_count;
    for switch_slot in 0..switch_count {
        let action_index = layout.move_action_count + switch_slot;
        let token = action_offset + action_index;
        let mon = switch_targets
            .get(switch_slot)
            .and_then(|team_index| self_mons.get(*team_index));
        let condition = condition_features(mon.and_then(|m| m.condition()));
        let species = mon
            .map(|m| m.species())
            .unwrap_or_else(|| format!("slot:{}", switch_slot + 1));
        grid.set_cat(
            token,
            layout.cat_col("CATEGORY_PRIMARY")?,
            format!("species:{species}"),
        );
        if let Some(mon) = mon {
            encode_species_type_categories(tables, grid, token, &mon.species())?;
            encode_pokemon_stats(tables, grid, token, &mon.species(), mon.details())?;
            encode_actual_stats(tables, grid, token, mon)?;
        }
        grid.set_cat(
            token,
            layout.cat_col("CATEGORY_SECONDARY")?,
            "action:switch",
        );
        grid.set_cat(token, layout.cat_col("CATEGORY_ROLE")?, "action");
        grid.set_cat(
            token,
            layout.cat_col("CATEGORY_SLOT")?,
            format!("switch_slot:{}", switch_slot + 1),
        );
        grid.set_num(
            token,
            layout.num_col("NUMERIC_HP_FRACTION")?,
            condition.hp_fraction.unwrap_or(0.0),
        );
        grid.set_num(
            token,
            layout.num_col("NUMERIC_ACTIVE")?,
            if mon.map(|m| m.active()).unwrap_or(false) {
                1.0
            } else {
                0.0
            },
        );
        grid.set_num(
            token,
            layout.num_col("NUMERIC_LEGAL")?,
            if legal.get(action_index).copied().unwrap_or(0) != 0 {
                1.0
            } else {
                0.0
            },
        );
        grid.set_num(
            token,
            layout.num_col("NUMERIC_PRESENT")?,
            if mon.is_some() { 1.0 } else { 0.0 },
        );
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// History-derived cells from fold PRODUCTS (native in-crate consumption)
// ---------------------------------------------------------------------------

use crate::fold::{Kind as FoldKind, ProductsData, Status as FoldStatus, SubBlock};

fn fold_kind_str(kind: Option<FoldKind>) -> &'static str {
    kind.map(FoldKind::as_str).unwrap_or("")
}

/// `showdown._tm_first_action_label`.
fn tm_first_action_label(kind: Option<FoldKind>, action: &str) -> String {
    match kind {
        Some(FoldKind::Move) => format!("move:{action}"),
        Some(FoldKind::Switch) => format!("species:{action}"),
        _ => format!("cant:{action}"),
    }
}

/// `showdown._tm_second_action_label`.
fn tm_second_action_label(kind: Option<FoldKind>, action: &str) -> String {
    match kind {
        Some(FoldKind::Move) => format!("tt2_move:{action}"),
        Some(FoldKind::Switch) => format!("tt2_species:{action}"),
        _ => format!("tt2_cant:{action}"),
    }
}

fn opt_str_nonempty(value: &Option<String>) -> Option<&str> {
    value.as_deref().filter(|s| !s.is_empty())
}

/// Every history-derived observation cell, written from the fold products —
/// the native mirror of `_encode_turn_merged_transition_tokens` (rows 23
/// through the schema-bound final row), `_encode_stats_token` (token 22),
/// `_encode_mon_tendency` +
/// the pinned Tier-2 conclusions (opponent-team tokens).
fn write_history_cells(
    tables: &Tables,
    grid: &mut Grid,
    products: &ProductsData,
    md: &Value,
    self_mons: &[MonToken],
    opponent_mons: &[MonToken],
) -> PyResult<()> {
    let layout = &tables.layout;
    let self_slot = str_or_empty(get(md, "showdown_slot"));
    let turn_number = as_i64(get(md, "turn_number"));
    if !layout.is_v4() {
        // v4 has no transition region. The rest of this function still runs: the stats-token
        // counters and the per-opponent-mon tendency / pinned Tier-2 conclusions are
        // CURRENT-STATE surfaces on real tokens, and they survive the region trim.
        write_turn_merged_rows(tables, grid, products, &self_slot, turn_number)?;
    }
    if layout.stats_block {
        write_stats_token(tables, grid, products)?;
    }
    write_opponent_mon_history(tables, grid, products, self_mons, opponent_mons)?;
    Ok(())
}

/// `showdown._encode_turn_merged_transition_tokens` from fold products.
fn write_turn_merged_rows(
    tables: &Tables,
    grid: &mut Grid,
    products: &ProductsData,
    self_slot: &str,
    turn_number: i64,
) -> PyResult<()> {
    let layout = &tables.layout;
    let transition_offset = layout.offset("transition")?;
    let transition_count = transition_row_count(layout)?;
    let budget = layout.transition_token_budget.min(transition_count);
    let tokens = &products.turn_merged_tokens;
    let start = tokens.len().saturating_sub(budget);
    let stat_divisor = layout.stat_count_divisor;

    for (index, token) in tokens[start..].iter().enumerate() {
        let row = transition_offset + index;
        let first = &token.first;
        let actor_role = if crate::fold::side_str(first.actor_slot) == self_slot {
            "self"
        } else {
            "opponent"
        };
        grid.set_cat(
            row,
            layout.cat_col("CATEGORY_PRIMARY")?,
            format!("species:{}", first.actor_species),
        );
        grid.set_cat(
            row,
            layout.cat_col("CATEGORY_SECONDARY")?,
            tm_first_action_label(first.kind, &first.action),
        );
        grid.set_cat(
            row,
            layout.cat_col("CATEGORY_ROLE")?,
            format!("transition:{actor_role}"),
        );
        grid.set_cat(
            row,
            layout.cat_col("CATEGORY_SLOT")?,
            format!("tt_phase:{}", token.phase.as_str()),
        );
        grid.set_cat(
            row,
            layout.cat_col("CATEGORY_TM_FIRST_KIND")?,
            format!("tt_kind:{}", fold_kind_str(first.kind)),
        );
        if first.kind == Some(FoldKind::Move) {
            grid.set_cat(
                row,
                layout.cat_col("CATEGORY_TYPE_1")?,
                format!("tt_outcome:{}", first.damage_outcome.as_str()),
            );
            grid.set_cat(
                row,
                layout.cat_col("CATEGORY_TYPE_2")?,
                format!("tt_effectiveness:{}", first.effectiveness.as_str()),
            );
            grid.set_cat(
                row,
                layout.cat_col("CATEGORY_MOVE_CATEGORY")?,
                format!("tt_side_effect:{}", first.side_effect.as_str()),
            );
            if let Some(defender) = opt_str_nonempty(&first.defender_species) {
                grid.set_cat(
                    row,
                    layout.cat_col("CATEGORY_MOVE_PRIORITY")?,
                    format!("species:{defender}"),
                );
            }
        }
        if let Some(weather) = opt_str_nonempty(&token.weather) {
            grid.set_cat(
                row,
                layout.cat_col("CATEGORY_MOVE_EFFECT")?,
                format!("weather:{weather}"),
            );
        }
        if let Some(cant) = opt_str_nonempty(&first.cant_reason) {
            grid.set_cat(
                row,
                layout.cat_col("CATEGORY_TM_FIRST_CANT")?,
                format!("cant:{cant}"),
            );
        }
        if let Some(bp) = opt_str_nonempty(&first.baton_pass_species) {
            grid.set_cat(
                row,
                layout.cat_col("CATEGORY_TM_FIRST_BP")?,
                format!("species:{bp}"),
            );
        }
        grid.set_num(row, layout.num_col("NUMERIC_PRESENT")?, 1.0);
        write_sub_block_numerics(tables, grid, row, first, SubBlockColumns::FIRST)?;
        if token.own_spikes_layers != 0 {
            grid.set_num(
                row,
                layout.num_col("NUMERIC_TT_OWN_SPIKES")?,
                (token.own_spikes_layers as f64 / 3.0).min(1.0),
            );
        }
        if token.opp_spikes_layers != 0 {
            grid.set_num(
                row,
                layout.num_col("NUMERIC_TT_OPP_SPIKES")?,
                (token.opp_spikes_layers as f64 / 3.0).min(1.0),
            );
        }
        grid.set_num(
            row,
            layout.num_col("NUMERIC_TT_ABS_TURN")?,
            (token.turn as f64 / 1000.0).min(1.0),
        );
        let turns_ago = (turn_number - token.turn).max(0);
        grid.set_num(
            row,
            layout.num_col("NUMERIC_TT_TURNS_AGO")?,
            (turns_ago as f64 / stat_divisor).min(1.0),
        );

        let second = &token.second;
        if second.status != FoldStatus::Action {
            // NEGATED vs ABSENT: categorical status + the consumed mon's
            // identity when the fold knows it; all TM2 numerics stay 0.0.
            grid.set_cat(
                row,
                layout.cat_col("CATEGORY_TM_SECOND_KIND")?,
                format!("tt2_status:{}", second.status.as_str()),
            );
            if !second.actor_species.is_empty() {
                grid.set_cat(
                    row,
                    layout.cat_col("CATEGORY_TM_SECOND_SPECIES")?,
                    format!("tt2_species:{}", second.actor_species),
                );
            }
            continue;
        }
        grid.set_cat(
            row,
            layout.cat_col("CATEGORY_TM_SECOND_KIND")?,
            format!("tt2_kind:{}", fold_kind_str(second.kind)),
        );
        grid.set_cat(
            row,
            layout.cat_col("CATEGORY_TM_SECOND_SPECIES")?,
            format!("tt2_species:{}", second.actor_species),
        );
        grid.set_cat(
            row,
            layout.cat_col("CATEGORY_TM_SECOND_ACTION")?,
            tm_second_action_label(second.kind, &second.action),
        );
        if second.kind == Some(FoldKind::Move) {
            grid.set_cat(
                row,
                layout.cat_col("CATEGORY_TM_SECOND_OUTCOME")?,
                format!("tt2_outcome:{}", second.damage_outcome.as_str()),
            );
            grid.set_cat(
                row,
                layout.cat_col("CATEGORY_TM_SECOND_EFFECTIVENESS")?,
                format!("tt2_effectiveness:{}", second.effectiveness.as_str()),
            );
            grid.set_cat(
                row,
                layout.cat_col("CATEGORY_TM_SECOND_SIDE_EFFECT")?,
                format!("tt2_side_effect:{}", second.side_effect.as_str()),
            );
            if let Some(defender) = opt_str_nonempty(&second.defender_species) {
                grid.set_cat(
                    row,
                    layout.cat_col("CATEGORY_TM_SECOND_DEFENDER")?,
                    format!("tt2_species:{defender}"),
                );
            }
        }
        if let Some(cant) = opt_str_nonempty(&second.cant_reason) {
            grid.set_cat(
                row,
                layout.cat_col("CATEGORY_TM_SECOND_CANT")?,
                format!("tt2_cant:{cant}"),
            );
        }
        if let Some(bp) = opt_str_nonempty(&second.baton_pass_species) {
            grid.set_cat(
                row,
                layout.cat_col("CATEGORY_TM_SECOND_BP")?,
                format!("tt2_species:{bp}"),
            );
        }
        grid.set_num(row, layout.num_col("NUMERIC_TM2_PRESENT")?, 1.0);
        write_sub_block_numerics(tables, grid, row, second, SubBlockColumns::SECOND)?;
    }
    Ok(())
}

/// Numeric column names for the first vs second sub-block of a turn-merged row.
struct SubBlockColumns {
    damage_fraction: &'static str,
    n_hits: &'static str,
    called: &'static str,
    transformed: &'static str,
    crit: &'static str,
    miss: &'static str,
    ko: &'static str,
    pursuit_intercept: &'static str,
    residual: &'static str,
    residual_valid: &'static str,
    cb_bit: &'static str,
    investment: &'static str,
    self_hp_cost: &'static str,
    fail: &'static str,
}

impl SubBlockColumns {
    const FIRST: SubBlockColumns = SubBlockColumns {
        damage_fraction: "NUMERIC_TT_DAMAGE_FRACTION",
        n_hits: "NUMERIC_TT_N_HITS",
        called: "NUMERIC_TT_CALLED",
        transformed: "NUMERIC_TT_TRANSFORMED",
        crit: "NUMERIC_TT_CRIT",
        miss: "NUMERIC_TT_MISS",
        ko: "NUMERIC_TT_KO",
        pursuit_intercept: "NUMERIC_TT_PURSUIT_INTERCEPT",
        residual: "NUMERIC_TT_RESIDUAL",
        residual_valid: "NUMERIC_TT_RESIDUAL_VALID",
        cb_bit: "NUMERIC_TT_CB_BIT",
        investment: "NUMERIC_TT_INVESTMENT_BIT",
        self_hp_cost: "NUMERIC_TT_SELF_HP_COST",
        fail: "NUMERIC_TT_FAIL",
    };
    const SECOND: SubBlockColumns = SubBlockColumns {
        damage_fraction: "NUMERIC_TM2_DAMAGE_FRACTION",
        n_hits: "NUMERIC_TM2_N_HITS",
        called: "NUMERIC_TM2_CALLED",
        transformed: "NUMERIC_TM2_TRANSFORMED",
        crit: "NUMERIC_TM2_CRIT",
        miss: "NUMERIC_TM2_MISS",
        ko: "NUMERIC_TM2_KO",
        pursuit_intercept: "NUMERIC_TM2_PURSUIT_INTERCEPT",
        residual: "NUMERIC_TM2_RESIDUAL",
        residual_valid: "NUMERIC_TM2_RESIDUAL_VALID",
        cb_bit: "NUMERIC_TM2_CB_BIT",
        investment: "NUMERIC_TM2_INVESTMENT",
        self_hp_cost: "NUMERIC_TM2_SELF_HP_COST",
        fail: "NUMERIC_TM2_FAIL",
    };
}

/// The shared numeric sub-block writes of the turn-merged encoder (identical
/// structure for the first and second sub-blocks, differing only in columns).
fn write_sub_block_numerics(
    tables: &Tables,
    grid: &mut Grid,
    row: usize,
    sub: &SubBlock,
    columns: SubBlockColumns,
) -> PyResult<()> {
    let layout = &tables.layout;
    // The fold keeps confusion self-damage separate from move damage for
    // every schema. Reconstruct only the frozen V2.2 aggregate; V3 consumes
    // the corrected semantic value and additionally exposes the presence bit.
    let damage_fraction = if !layout.is_v3() && sub.confusion_selfhit {
        sub.damage_fraction + sub.confusion_selfhit_fraction
    } else {
        sub.damage_fraction
    };
    if damage_fraction != 0.0 {
        grid.set_num(
            row,
            layout.num_col(columns.damage_fraction)?,
            damage_fraction.min(1.0),
        );
    }
    if sub.kind == Some(FoldKind::Move) {
        grid.set_num(
            row,
            layout.num_col(columns.n_hits)?,
            (sub.n_hits as f64 / 5.0).min(1.0),
        );
    }
    for (column, flag) in [
        (columns.called, sub.called),
        (columns.transformed, sub.transformed),
        (columns.crit, sub.crit),
        (columns.miss, sub.miss),
        (columns.ko, sub.ko),
        (columns.pursuit_intercept, sub.pursuit_intercept),
    ] {
        if flag {
            grid.set_num(row, layout.num_col(column)?, 1.0);
        }
    }
    if layout.tier2_residuals && sub.residual_valid {
        if let Some(residual) = sub.residual {
            grid.set_num(
                row,
                layout.num_col(columns.residual)?,
                residual.clamp(-1.0, 1.0),
            );
            grid.set_num(row, layout.num_col(columns.residual_valid)?, 1.0);
        }
    }
    if layout.tier2_residuals && sub.cb_bit {
        grid.set_num(row, layout.num_col(columns.cb_bit)?, 1.0);
    }
    if layout.tier2_residuals && layout.tier2_investment && sub.investment != 0.0 {
        grid.set_num(
            row,
            layout.num_col(columns.investment)?,
            sub.investment.clamp(-1.0, 1.0),
        );
    }
    if sub.self_hp_cost != 0.0 {
        grid.set_num(
            row,
            layout.num_col(columns.self_hp_cost)?,
            sub.self_hp_cost.min(1.0),
        );
    }
    if layout.is_v3() {
        if sub.fail {
            grid.set_num(row, layout.num_col(columns.fail)?, 1.0);
        }
        if sub.confusion_selfhit {
            grid.set_num(row, layout.num_col("NUMERIC_TT_CONFUSION_SELFHIT")?, 1.0);
        }
    }
    Ok(())
}

/// `showdown._encode_stats_token` from fold products (counts /64 + the
/// opponent weather-reveal pairs).
fn write_stats_token(tables: &Tables, grid: &mut Grid, products: &ProductsData) -> PyResult<()> {
    let layout = &tables.layout;
    let stats_offset = layout.offset("stats")?;
    let stats = &products.tendency_stats;
    for (column, count) in [
        ("NUMERIC_STAT_OPP_SWITCH_COUNT", stats.opponent_switch_count),
        (
            "NUMERIC_STAT_OPP_DECISION_OPPORTUNITIES",
            stats.opponent_decision_opportunities,
        ),
        (
            "NUMERIC_STAT_BLOCKED_ON_OUR_ATTACK",
            stats.blocked_on_our_attack_count,
        ),
        (
            "NUMERIC_STAT_PURSUIT_INTERCEPT_PREDICT",
            stats.pursuit_intercept_predict_count,
        ),
        ("NUMERIC_STAT_MY_SWITCH_TURNS", stats.my_switch_turn_count),
    ] {
        if count != 0 {
            grid.set_num(
                stats_offset,
                layout.num_col(column)?,
                (count as f64 / layout.stat_count_divisor).min(1.0),
            );
        }
    }
    let reveal_offset = layout.num_col("NUMERIC_STAT_WEATHER_REVEAL_OFFSET")?;
    for (index, weather) in layout.weather_reveal_order.iter().enumerate() {
        let Some((_, from_ability)) = stats
            .opponent_weather_reveals
            .iter()
            .find(|(id, _)| id == weather)
        else {
            continue;
        };
        grid.set_num(stats_offset, reveal_offset + 2 * index, 1.0);
        if *from_ability {
            grid.set_num(stats_offset, reveal_offset + 2 * index + 1, 1.0);
        }
    }
    Ok(())
}

/// Per-opponent-mon history columns: the tendency triple
/// (`_encode_mon_tendency`, gated on the stats block like production) and the
/// pinned Tier-2 conclusions (CB + investment, mask-gated like production).
fn write_opponent_mon_history(
    tables: &Tables,
    grid: &mut Grid,
    products: &ProductsData,
    self_mons: &[MonToken],
    opponent_mons: &[MonToken],
) -> PyResult<()> {
    let layout = &tables.layout;
    let opponent_offset = layout.offset("opponent_pokemon")?;
    let action_offset = layout.offset("action_candidates")?;
    let limit = action_offset - opponent_offset;
    let cb_pinned: Vec<String> = products
        .cb_pinned_species
        .iter()
        .map(|s| normalize_identifier(s))
        .collect();
    // Hoisted for the same reason `cb_pinned` is: both are loop-invariant, and the matchup table
    // is (their mon x our mon), so a per-token `.find()` that normalized as it went would
    // allocate two Strings per candidate cell per token -- up to ~36 cells against the tendency
    // lookup's <=6, inside the MCTS leaf-pricing closure that ROW_WRITE_NANOS times.
    //
    // Gated on the SAME condition as the consuming block below, not just hoisted. The fold
    // populates `opponent_mon_matchups` at every schema, so an ungated hoist made v2.2 and v3 --
    // where the consumer never runs and the cost used to be exactly zero -- pay up to ~72 String
    // allocations per encode for a value that is discarded. Three lineages are training against
    // those layouts right now.
    let matchup_write = layout.is_v4() && layout.stats_block;
    let our_active = matchup_write
        .then(|| {
            self_mons
                .iter()
                .find(|mon| mon.active())
                .map(|mon| normalize_identifier(&mon.species()))
                .filter(|species| !species.is_empty())
        })
        .flatten();
    let matchup_keys: Vec<(String, String)> = if matchup_write {
        products
            .tendency_stats
            .opponent_mon_matchups
            .iter()
            .map(|cell| {
                (
                    normalize_identifier(&cell.species),
                    normalize_identifier(&cell.opposing_species),
                )
            })
            .collect()
    } else {
        Vec::new()
    };
    // `.rev()` on the zip below pairs cell i with key i ONLY while the lengths match. They do by
    // construction (`.map()` over the same Vec), but a later `filter_map` here would silently
    // misalign the pairing under `.rev()` and write wrong values without panicking.
    debug_assert!(
        !matchup_write
            || matchup_keys.len() == products.tendency_stats.opponent_mon_matchups.len(),
        "matchup_keys must stay 1:1 with opponent_mon_matchups for the zip+rev lookup"
    );
    for (slot, mon) in opponent_mons.iter().take(limit).enumerate() {
        let token = opponent_offset + slot;
        let species_key = normalize_identifier(&mon.species());
        if layout.stats_block {
            // Production keys a dict by normalized species (later entries win),
            // hence the reverse find.
            if let Some(tendency) = products
                .tendency_stats
                .opponent_mon_tendencies
                .iter()
                .rev()
                .find(|t| normalize_identifier(&t.species) == species_key)
            {
                for (column, count) in [
                    (
                        "NUMERIC_MON_SWITCHED_BEFORE_ATTACK",
                        tendency.switched_out_before_attacking,
                    ),
                    (
                        "NUMERIC_MON_STAYED_AND_ATTACKED",
                        tendency.stayed_and_attacked,
                    ),
                    ("NUMERIC_MON_TURNS_ACTIVE_TOTAL", tendency.turns_active),
                ] {
                    if count != 0 {
                        grid.set_num(
                            token,
                            layout.num_col(column)?,
                            (count as f64 / layout.stat_count_divisor).min(1.0),
                        );
                    }
                }
            }
            // The matchup-conditional pair, written from the FOLD rather than the metadata.
            //
            // `encode_pokemon_tokens` already wrote these two columns from
            // `md["opponent_matchup_switch_evidence"]`, which is correct on the root/boundary
            // encode (no products) but FROZEN at the leaf: leaf.rs rebuilds the team rows from
            // live engine state and never rebuilds that fold-derived key, so a search line saw
            // the root's counts against the root's active. That mattered more than ordinary
            // staleness because this column's divisor is 8, not 64 -- one event moves it 12.5%
            // against the tendency triple's 1.6% -- so the 8x more sensitive column sat still
            // while the insensitive one advanced live, an off-distribution (matchup, tendency)
            // pair the model never trained on. Measured movement: 0.191 cell changes per ply,
            // 55.9% of 4-ply windows and 79.8% of 8-ply windows containing at least one.
            //
            // Runs AFTER the metadata write (write_history_cells is called after
            // encode_pokemon_tokens) and writes unconditionally, zero included, so it fully
            // overrides the stale pair instead of leaving it standing when the live cell is
            // empty. Mirrors `_matchup_switch_evidence` (showdown.py:3959): select the column
            // of the (their mon x our mon) table whose `opposing_species` is OUR CURRENT
            // active, absent active or absent cell meaning an honest (0, 0).
            if matchup_write {
                // Later entries win, as in the tendency lookup above: production keys a dict by
                // normalized species and Python's builder assigns in iteration order.
                let cell = our_active.as_ref().and_then(|ours| {
                    products
                        .tendency_stats
                        .opponent_mon_matchups
                        .iter()
                        .zip(matchup_keys.iter())
                        .rev()
                        .find(|(_, (species, opposing))| {
                            *species == species_key && opposing == ours
                        })
                        .map(|(cell, _)| cell)
                });
                for (column, count) in [
                    (
                        "NUMERIC_MON_SWITCHED_VS_ACTIVE",
                        cell.map(|c| c.switched_out_before_attacking).unwrap_or(0),
                    ),
                    (
                        "NUMERIC_MON_STAYED_VS_ACTIVE",
                        cell.map(|c| c.stayed_and_attacked).unwrap_or(0),
                    ),
                ] {
                    grid.set_num(
                        token,
                        layout.num_col(column)?,
                        (count as f64 / layout.matchup_count_divisor).min(1.0),
                    );
                }
            }
        }
        // RETIRED AT V4, both of them: the Choice Band and investment conclusions narrow the
        // belief candidate set there instead of being projected onto these columns, so v4
        // layouts do not carry either (the Python twin gates both on `not schema_v4`).
        // Resolved with num_col_opt rather than num_col: under v4 the columns are legitimately
        // absent, and a hard "layout missing numeric column" at leaf-encode time would be
        // wrong. Extraction keeps running — `products.cb_pinned_species` and
        // `products.investment_pinned` are still populated, they simply have no column to land
        // in.
        if layout.tier2_residuals && cb_pinned.iter().any(|s| *s == species_key) {
            if let Some(column) = layout.num_col_opt("NUMERIC_TIER2_CB_PINNED") {
                grid.set_num(token, column, 1.0);
            }
        }
        if layout.tier2_residuals && layout.tier2_investment {
            if let Some(column) = layout.num_col_opt("NUMERIC_TIER2_INVESTMENT_PINNED") {
                if let Some((_, code)) = products
                    .investment_pinned
                    .iter()
                    .find(|(species, _)| normalize_identifier(species) == species_key)
                {
                    if *code != 0.0 {
                        grid.set_num(token, column, code.clamp(-1.0, 1.0));
                    }
                }
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// PyO3 surface
// ---------------------------------------------------------------------------

fn bytes_from_i32(py: Python<'_>, values: &[i32]) -> Py<PyBytes> {
    let mut buffer = Vec::with_capacity(values.len() * 4);
    for value in values {
        buffer.extend_from_slice(&value.to_le_bytes());
    }
    PyBytes::new(py, &buffer).into()
}

fn bytes_from_f64(py: Python<'_>, values: &[f64]) -> Py<PyBytes> {
    let mut buffer = Vec::with_capacity(values.len() * 8);
    for value in values {
        buffer.extend_from_slice(&value.to_le_bytes());
    }
    PyBytes::new(py, &buffer).into()
}

fn bytes_from_i16(py: Python<'_>, values: &[i16]) -> Py<PyBytes> {
    let mut buffer = Vec::with_capacity(values.len() * 2);
    for value in values {
        buffer.extend_from_slice(&value.to_le_bytes());
    }
    PyBytes::new(py, &buffer).into()
}

pub(crate) fn encoded_to_dict(py: Python<'_>, encoded: &EncodedArrays) -> PyResult<Py<PyDict>> {
    let out = PyDict::new(py);
    out.set_item("categorical_ids", bytes_from_i32(py, &encoded.categorical))?;
    out.set_item("numeric_features", bytes_from_f64(py, &encoded.numeric))?;
    out.set_item("token_type_ids", bytes_from_i16(py, &encoded.token_types))?;
    out.set_item("attention_mask", PyBytes::new(py, &encoded.attention))?;
    out.set_item("legal_action_mask", PyBytes::new(py, &encoded.legal))?;
    Ok(out.into())
}

/// Encode one golden-corpus decision row. `row_inputs_json` is the sanctioned
/// input surface (see `scripts/golden_encoder_backends.row_inputs_from_decision_row`);
/// `tables_json` is the artifact from `scripts/export_encoder_tables.py`.
/// Returns `{name: little-endian bytes}` for the five observation arrays in
/// the corpus's canonical dtypes (i4 / f8 / i2 / b1 / b1).
#[pyfunction]
pub fn encode_decision(
    py: Python<'_>,
    row_inputs_json: &str,
    tables_json: &str,
) -> PyResult<Py<PyDict>> {
    let tables = Tables::from_json(tables_json)?;
    let encoded = encode_row(&tables, row_inputs_json)?;
    encoded_to_dict(py, &encoded)
}

/// Persistent encoder handle: parses the tables artifact ONCE and serves
/// per-row encodes, with or without a fold state. `encode_with_fold` consumes
/// the fold's products natively in-crate (no `products_payload` Python
/// crossing) — transition rows 23 through the schema-bound final row,
/// tendency/stats counters, pinned
/// Tier-2 conclusions, and the transition attention extent become real.
#[pyclass(name = "NativeEncoder", module = "pokezero_search")]
pub struct NativeEncoder {
    tables: Tables,
}

#[pymethods]
impl NativeEncoder {
    #[new]
    fn new(tables_json: &str) -> PyResult<Self> {
        Ok(NativeEncoder {
            tables: Tables::from_json(tables_json)?,
        })
    }

    /// Boundary-surface encode (identical to `encode_decision`).
    fn encode(&self, py: Python<'_>, row_inputs_json: &str) -> PyResult<Py<PyDict>> {
        let encoded = encode_row(&self.tables, row_inputs_json)?;
        encoded_to_dict(py, &encoded)
    }

    /// Full-surface encode: boundary cells from the row inputs + history cells
    /// from `fold`'s products (native consumption).
    fn encode_with_fold(
        &self,
        py: Python<'_>,
        row_inputs_json: &str,
        fold: &crate::fold::PyFoldState,
    ) -> PyResult<Py<PyDict>> {
        let row: Value =
            serde_json::from_str(row_inputs_json).map_err(|e| err(format!("row JSON: {e}")))?;
        let products = fold.inner().products();
        let encoded = encode_row_value(&self.tables, &row, Some(&products))?;
        encoded_to_dict(py, &encoded)
    }
}

#[cfg(test)]
mod spread_tests {
    use super::{randbats_spread_stats, RandbatsSpread};
    use std::collections::HashMap;

    /// Base stats for the four non-HP columns. The HP/Atk assertions in this module do not read
    /// them, so a map carrying only what each case needs keeps the existing expectations intact.
    fn bases(def: i64, spa: i64, spd: i64, spe: i64) -> HashMap<String, i64> {
        [("def", def), ("spa", spa), ("spd", spd), ("spe", spe)]
            .into_iter()
            .map(|(k, v)| (k.to_string(), v))
            .collect()
    }

    /// Every expected value here was produced by the PYTHON source of truth --
    /// `gen3_damage.randbats_spread_details` -- not by re-deriving the rules by hand. That is the
    /// whole point: this port exists because the crate had its own approximation of the
    /// generator, and a test written from the same reading that produced the port would
    /// reproduce the same misreading.
    fn spread(
        hp_base: i64,
        atk_base: i64,
        level: i64,
        moves: &[&str],
        item: &str,
        has_physical: bool,
    ) -> Option<(i64, i64)> {
        let moves: Vec<String> = moves.iter().map(|m| m.to_string()).collect();
        randbats_spread_stats(
            &bases(100, 100, 100, 100),
            hp_base,
            atk_base,
            level,
            &moves,
            item,
            has_physical,
        )
        .unwrap()
        .map(|s: RandbatsSpread| (s.hp, s.atk))
    }

    #[test]
    fn plain_set_is_the_untrimmed_baseline() {
        assert_eq!(
            spread(95, 125, 78, &["earthquake", "rockslide"], "", true),
            Some((276, 240))
        );
    }

    #[test]
    fn belly_drum_trims_hp_by_steps_not_to_zero_evs() {
        // hp_ev 85 -> 77: TWO steps of the generator's loop. The approximation this port
        // replaced jumped straight to ev=0, which reports 320 -- 9 HP low, and low by a
        // DIFFERENT amount for every species, so it does not even fail consistently.
        assert_eq!(
            spread(160, 110, 68, &["bellydrum", "bodyslam", "rest", "sleeptalk"], "leftovers", true),
            Some((329, 189))
        );
    }

    #[test]
    fn substitute_pinch_berry_trims_through_the_second_pass() {
        assert_eq!(
            spread(40, 100, 84, &["substitute", "flail", "endure"], "salacberry", true),
            Some((203, 216))
        );
    }

    #[test]
    fn substitute_endeavor_trims_one_step_in_the_second_pass() {
        assert_eq!(
            spread(40, 100, 84, &["substitute", "endeavor", "protect"], "leftovers", true),
            Some((204, 216))
        );
    }

    #[test]
    fn atk_zeroing_without_hidden_power_uses_iv_zero() {
        assert_eq!(
            spread(95, 75, 84, &["surf", "psychic", "thunderwave", "rest"], "leftovers", false),
            Some((297, 131))
        );
    }

    #[test]
    fn atk_zeroing_with_hidden_power_keeps_iv_minus_28() {
        // ivs.atk = 30 (Hidden Power Ice override) - 28 = 2, NOT 0. The old approximation
        // hardcoded iv=0 here and was wrong on 43% of Atk-zeroed variants, every one an HP set.
        assert_eq!(
            spread(60, 65, 80, &["hiddenpowerice", "thunderbolt", "icebeam", "substitute"], "leftovers", false),
            Some((227, 110))
        );
    }

    #[test]
    fn hidden_power_flying_lowers_the_hp_iv_too() {
        // The only HP type whose override touches `hp`, so it is the only one that moves the
        // HP stat off the all-31 value: 227 -> 226.
        assert_eq!(
            spread(60, 65, 80, &["hiddenpowerflying", "thunderbolt", "icebeam", "rest"], "leftovers", false),
            Some((226, 110))
        );
    }

    #[test]
    fn shedinja_hp_is_pinned_to_one() {
        assert_eq!(
            spread(1, 90, 84, &["shadowball", "silverwind", "protect"], "lumberry", true),
            Some((1, 199))
        );
    }

    #[test]
    fn transform_suppresses_atk_zeroing() {
        // The generator zeroes Atk only when the set has no physical attack AND no Transform.
        // Ditto-with-Transform therefore keeps the 85/31 Atk, and dropping the `transform`
        // clause would silently halve it.
        let with = spread(96, 60, 76, &["transform"], "leftovers", false);
        let without = spread(96, 60, 76, &["splash"], "leftovers", false);
        assert_ne!(with, without);
        assert_eq!(with, spread(96, 60, 76, &["transform"], "leftovers", true));
    }

    #[test]
    fn an_unknown_hidden_power_type_follows_python_and_takes_the_dark_branch() {
        // Python's `HIDDEN_POWER_IVS.get(hp_type, {})` gives an unknown type no overrides while
        // leaving `hp_type is not None` true, so Atk zeroing uses 31-28=3. Byte-identical to
        // `dark`, whose overrides are legitimately empty. Pinned because an earlier draft
        // refused here instead, and the parity test could not see the difference: the corpus
        // carries no unknown type, so the divergence would have shipped silently.
        assert_eq!(
            spread(60, 65, 80, &["hiddenpowerfairy"], "leftovers", false),
            spread(60, 65, 80, &["hiddenpowerdark"], "leftovers", false),
        );
        assert_eq!(
            spread(60, 65, 80, &["hiddenpowerfairy"], "leftovers", false),
            Some((227, 111)),
        );
        // ...and the generic, untyped move carries no type at all, so Atk falls to IV 0 (109),
        // not 3 (111). The two branches must stay distinguishable.
        assert_eq!(spread(60, 65, 80, &["hiddenpower"], "leftovers", false), Some((227, 109)));
    }

    #[test]
    fn an_illegal_spread_is_refused_rather_than_emitted() {
        // Survived a mutation sweep: replacing the whole legality check with `if false` left
        // every other test green, because no gen3 variant reaches an illegal spread AT ITS OWN
        // POOL LEVEL. Belly Drum at L1 does -- the trim loop runs all the way down to hp_ev 1,
        // far past the generator's floor of 69 -- so the guard is reachable and now pinned.
        // Verified against the Python core, which reaches the same illegal hp_ev=1 and whose
        // `_variant_spread_stats` raises on it.
        let moves: Vec<String> = ["bellydrum", "bodyslam"].iter().map(|m| m.to_string()).collect();
        let refused = randbats_spread_stats(&bases(100, 100, 100, 100), 160, 110, 1, &moves, "leftovers", true);
        assert!(refused.is_err(), "an illegal spread was emitted: {refused:?}");
        // ...and the same set at its real level is fine, so the guard is not simply always-on.
        assert!(randbats_spread_stats(&bases(100, 100, 100, 100), 160, 110, 68, &moves, "leftovers", true)
            .unwrap()
            .is_some());
    }
}

#[cfg(test)]
mod level_tests {
    use super::level_from_details;

    /// Values cross-checked against `showdown._level_from_details`, which is the source of truth
    /// and documents the rule from the vendored `sim/pokemon.ts::getUpdatedDetails`.
    #[test]
    fn a_details_string_without_an_l_token_is_level_100() {
        // The regression: these returned None, and the caller reads None as "skip the block".
        assert_eq!(level_from_details(Some("Shedinja, M")), Some(100));
        assert_eq!(level_from_details(Some("Ditto")), Some(100));
        assert_eq!(level_from_details(Some("Unown, F")), Some(100));
    }

    #[test]
    fn an_explicit_level_still_wins() {
        assert_eq!(level_from_details(Some("Slowbro, L84, M")), Some(84));
        assert_eq!(level_from_details(Some("Skarmory, L79, F")), Some(79));
    }

    #[test]
    fn only_a_missing_details_string_is_unknown() {
        // Python's `if not details` covers both. Everything else has a level, even if implicit.
        assert_eq!(level_from_details(None), None);
        assert_eq!(level_from_details(Some("")), None);
    }

    #[test]
    fn a_bare_l_or_a_non_numeric_l_token_is_not_a_level() {
        // "L" alone and "Lv84" are not level tokens; the mon is still level 100, not unknown.
        assert_eq!(level_from_details(Some("Ditto, L")), Some(100));
        assert_eq!(level_from_details(Some("Ditto, Lv84")), Some(100));
        // ...but a gender token starting with L must not be mistaken for one either.
        assert_eq!(level_from_details(Some("Ludicolo, M")), Some(100));
    }
}
