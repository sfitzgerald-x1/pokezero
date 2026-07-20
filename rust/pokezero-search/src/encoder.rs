//! Native v2.2 observation encoder (track B of the engine-swap plan).
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
//! IMPLEMENTED (bit-exact target): token_type_ids, legal_action_mask,
//! attention_mask (to the stored-surface ceiling: the transition extent is
//! not derivable per-row), and the categorical + numeric surfaces of tokens
//! 0..=22 (field, self team, opponent team + belief overlay, action
//! candidates, stats token).
//!
//! NOT IMPLEMENTED (documented 0%): the turn-merged transition tokens
//! (23..=150) and every history-derived column (tendency triples, pinned
//! Tier-2 conclusions, stats-token counters) — the corpus rows do not store
//! the public event stream they are extracted from, so the Python reference
//! itself cannot reproduce them from the same surface (track B phase 1
//! finding). Those cells stay zero, matching the reference backend.

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

// ---------------------------------------------------------------------------
// Tables (vocab + layout + dex) from scripts/export_encoder_tables.py
// ---------------------------------------------------------------------------

struct Layout {
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
}

impl Layout {
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
        let layout = Layout {
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
        hasher.finalize_variable(&mut digest).expect("blake2b-8 output");
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
fn level_from_details(details: Option<&str>) -> Option<i64> {
    let details = details?;
    for part in details.split(',') {
        let token = part.trim();
        if let Some(rest) = token.strip_prefix('L') {
            if !rest.is_empty() && rest.chars().all(|c| c.is_ascii_digit()) {
                return rest.parse::<i64>().ok();
            }
        }
    }
    None
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
/// turn-merged transition rows 23..=150, the stats-token tendency counters,
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

    let self_team = as_array(get(md, "self_team"));
    let opponent_team = as_array(get(md, "opponent_team"));
    let self_mons: Vec<MonToken> = self_team.iter().map(|entry| MonToken { entry }).collect();
    let opponent_mons: Vec<MonToken> = opponent_team.iter().map(|entry| MonToken { entry }).collect();

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
    if let Some(products) = products {
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
    encode_action_tokens(
        tables,
        &mut grid,
        md,
        pm,
        &self_mons,
        action_offset,
        &legal,
    )?;
    // Stats token: role + presence. The tendency counters are history-derived:
    // zero without fold products (the stored-surface ceiling), real when the
    // fold state is supplied.
    if layout.stats_block {
        grid.set_cat(stats_offset, layout.cat_col("CATEGORY_ROLE")?, "stats");
        grid.set_num(stats_offset, layout.num_col("NUMERIC_PRESENT")?, 1.0);
    }
    if let Some(products) = products {
        write_history_cells(tables, &mut grid, products, md, &opponent_mons)?;
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
    (
        (hazards as f64 / 3.0).min(1.0),
        (screens / 2.0).min(1.0),
    )
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
    grid.set_num(token, layout.num_col("NUMERIC_SELF_SCREENS")?, self_scr);
    grid.set_num(token, layout.num_col("NUMERIC_OPP_SCREENS")?, opp_scr);
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
        grid.set_num(
            token,
            layout.num_col("NUMERIC_SELF_FUTURE_SIGHT")?,
            (self_future as f64 / 2.0).min(1.0),
        );
    }
    let opp_future = as_i64(get(md, "opponent_future_sight_turns"));
    if opp_future != 0 {
        grid.set_num(
            token,
            layout.num_col("NUMERIC_OPP_FUTURE_SIGHT")?,
            (opp_future as f64 / 2.0).min(1.0),
        );
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
    if let Some(info) = tables.species_info(species) {
        if let Some(first) = info.types.first() {
            grid.set_cat(token, layout.cat_col("CATEGORY_TYPE_1")?, format!("type:{first}"));
        }
        if let Some(second) = info.types.get(1) {
            grid.set_cat(token, layout.cat_col("CATEGORY_TYPE_2")?, format!("type:{second}"));
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
    if let Some(info) = tables.species_info(species) {
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
            grid.set_num(token, *column, (value as f64 / layout.actual_stat_divisor).min(1.0));
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
    for (index, value) in compact_belief_values(values, Some(buckets)).iter().enumerate() {
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
                let hp_value =
                    (gen3_stat(hp_base, level, 85, 31, true) as f64 / divisor).min(1.0);
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

    let Some(level) = level_from_details(details) else {
        return Ok(());
    };
    let Some(battle_info) = tables.species_info(battle_species) else {
        return Ok(());
    };
    let Some(hp_info) = tables.species_info(base_species) else {
        return Ok(());
    };
    for (stat, column_name) in [
        ("def", "NUMERIC_EXPECTED_DEF"),
        ("spa", "NUMERIC_EXPECTED_SPA"),
        ("spd", "NUMERIC_EXPECTED_SPD"),
        ("spe", "NUMERIC_EXPECTED_SPE"),
    ] {
        let value = battle_info.base_stats.get(stat).copied().unwrap_or(0);
        if value != 0 {
            grid.set_num(
                token,
                layout.num_col(column_name)?,
                (gen3_stat(value, level, 85, 31, false) as f64 / divisor).min(1.0),
            );
        }
    }
    let atk_base = battle_info.base_stats.get("atk").copied().unwrap_or(0);
    let hp_base = hp_info.base_stats.get("hp").copied().unwrap_or(0);
    if atk_base == 0 || hp_base == 0 {
        return Ok(());
    }
    let atk_baseline = gen3_stat(atk_base, level, 85, 31, false);
    let hp_baseline = gen3_stat(hp_base, level, 85, 31, true);
    let (mut atk_low, mut atk_high) = (atk_baseline, atk_baseline);
    let (mut hp_low, mut hp_high) = (hp_baseline, hp_baseline);
    let variants = belief.map(|b| b.candidate_variants()).filter(|v| !v.is_empty());
    if let Some(variants) = variants {
        let mut atk_values: Vec<i64> = Vec::new();
        let mut hp_values: Vec<i64> = Vec::new();
        let pinch_berries = ["liechiberry", "petayaberry", "salacberry"];
        for variant in variants {
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
            atk_values.push(if has_physical {
                atk_baseline
            } else {
                gen3_stat(atk_base, level, 0, 0, false)
            });
            let has = |name: &str| moves.iter().any(|m| m == name);
            let hp_trimmed = has("bellydrum")
                || (has("substitute")
                    && (has("flail") || has("reversal") || pinch_berries.contains(&item.as_str())));
            hp_values.push(if hp_trimmed {
                gen3_stat(hp_base, level, 0, 31, true)
            } else {
                hp_baseline
            });
        }
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
        grid.set_num(token, layout.num_col(column_name)?, (value as f64 / divisor).min(1.0));
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
        let revealed_ability = belief.and_then(|b| b.revealed_ability());
        let revealed_item = belief.and_then(|b| b.revealed_item());
        let possible_abilities = belief
            .map(|b| b.possible("possible_abilities"))
            .unwrap_or_default();
        let possible_items = belief
            .map(|b| b.possible("possible_items"))
            .unwrap_or_default();
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
        let transformed = belief
            .map(|b| b.transformed() && b.transform_species().is_some())
            .unwrap_or(false);
        let enc_species = if transformed {
            belief
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
        if candidate.active() {
            encode_active_boosts(grid, token, boosts, layout);
            encode_active_volatiles(tables, grid, token, &volatiles)?;
            if toxic_stage != 0 {
                grid.set_num(
                    token,
                    layout.num_col("NUMERIC_TOXIC_STAGE")?,
                    (toxic_stage as f64 / 15.0).min(1.0),
                );
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
                            let max_pp = tables
                                .move_info(&key)
                                .map(|info| info.max_pp)
                                .unwrap_or(0);
                            if max_pp <= 0 {
                                continue;
                            }
                            let remaining =
                                (max_pp - uses.get(&key).copied().unwrap_or(0)).max(0);
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
        let payload = payload_moves.get(cursor).filter(|entry| {
            get(entry, "id").as_str() == Some(move_id.as_str())
        });
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
    grid.set_cat(
        token,
        layout.cat_col("CATEGORY_TYPE_1")?,
        format!("type:{}", info.move_type),
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
    let user_hp_fraction =
        active.and_then(|mon| condition_features(mon.condition()).hp_fraction);
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
        grid.set_cat(token, layout.cat_col("CATEGORY_SECONDARY")?, "action:switch");
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
/// the native mirror of `_encode_turn_merged_transition_tokens` (rows
/// 23..=150), `_encode_stats_token` (token 22), `_encode_mon_tendency` +
/// the pinned Tier-2 conclusions (opponent-team tokens).
fn write_history_cells(
    tables: &Tables,
    grid: &mut Grid,
    products: &ProductsData,
    md: &Value,
    opponent_mons: &[MonToken],
) -> PyResult<()> {
    let layout = &tables.layout;
    let self_slot = str_or_empty(get(md, "showdown_slot"));
    let turn_number = as_i64(get(md, "turn_number"));
    write_turn_merged_rows(tables, grid, products, &self_slot, turn_number)?;
    if layout.stats_block {
        write_stats_token(tables, grid, products)?;
    }
    write_opponent_mon_history(tables, grid, products, opponent_mons)?;
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
            grid.set_cat(row, layout.cat_col("CATEGORY_TM_FIRST_CANT")?, format!("cant:{cant}"));
        }
        if let Some(bp) = opt_str_nonempty(&first.baton_pass_species) {
            grid.set_cat(row, layout.cat_col("CATEGORY_TM_FIRST_BP")?, format!("species:{bp}"));
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
            grid.set_cat(row, layout.cat_col("CATEGORY_TM_SECOND_CANT")?, format!("tt2_cant:{cant}"));
        }
        if let Some(bp) = opt_str_nonempty(&second.baton_pass_species) {
            grid.set_cat(row, layout.cat_col("CATEGORY_TM_SECOND_BP")?, format!("tt2_species:{bp}"));
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
    if sub.damage_fraction != 0.0 {
        grid.set_num(
            row,
            layout.num_col(columns.damage_fraction)?,
            sub.damage_fraction.min(1.0),
        );
    }
    if sub.kind == Some(FoldKind::Move) {
        grid.set_num(row, layout.num_col(columns.n_hits)?, (sub.n_hits as f64 / 5.0).min(1.0));
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
            grid.set_num(row, layout.num_col(columns.residual)?, residual.clamp(-1.0, 1.0));
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
                    ("NUMERIC_MON_STAYED_AND_ATTACKED", tendency.stayed_and_attacked),
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
        }
        if layout.tier2_residuals && cb_pinned.iter().any(|s| *s == species_key) {
            grid.set_num(token, layout.num_col("NUMERIC_TIER2_CB_PINNED")?, 1.0);
        }
        if layout.tier2_residuals && layout.tier2_investment {
            if let Some((_, code)) = products
                .investment_pinned
                .iter()
                .find(|(species, _)| normalize_identifier(species) == species_key)
            {
                if *code != 0.0 {
                    grid.set_num(
                        token,
                        layout.num_col("NUMERIC_TIER2_INVESTMENT_PINNED")?,
                        code.clamp(-1.0, 1.0),
                    );
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
/// crossing) — transition rows 23..=150, tendency/stats counters, pinned
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
