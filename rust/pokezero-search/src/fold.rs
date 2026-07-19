//! Incremental fold-state advance (track B): Rust port of
//! `pokezero.transitions_fold.FoldState`.
//!
//! The unit of correctness is the fold-state ADVANCE
//! (`advance(fold_state, events) -> (fold_state', products')`), validated
//! row-pair by row-pair against the golden corpus v2 fold sidecar through
//! `scripts/validate_corpus_v2.py --backend rust` — canonical-JSON byte
//! equality on both the serialized state and the boundary products.
//!
//! Ported from the PYTHON REFERENCE (`src/pokezero/transitions_fold.py`,
//! prefix-closure proven per docs/fold_closure_probe.md), NOT from the batch
//! fold in `transitions.py` — where the reference delegates to
//! `turn_merged._merge_turn` / production helpers, the exact observable
//! behavior is ported and the corpus is the oracle for every case.
//!
//! Float discipline: all arithmetic is plain IEEE f64 in the same operation
//! order as the Python reference; payloads are returned as NATIVE Python
//! objects (dict/list/float/int/bool/str/None) so the validation harness's
//! own `json.dumps` canonicalization defines the bytes — this crate never
//! emits JSON text for the fold surfaces.

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use pyo3::exceptions::{PyAssertionError, PyKeyError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

pub const DEFAULT_MERGED_TAIL_LIMIT: usize = 128;
pub const DEFAULT_ACTION_TAIL_LIMIT: usize = 512;

const FOLD_STATE_SCHEMA: &str = "pokezero.fold-state.v1";
const FOLD_PRODUCTS_SCHEMA: &str = "pokezero.fold-products.v1";

// ---------------------------------------------------------------------------
// Enumerations (string vocabularies of transitions.py / turn_merged.py)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind {
    Move,
    Switch,
    Cant,
}

impl Kind {
    fn as_str(self) -> &'static str {
        match self {
            Kind::Move => "move",
            Kind::Switch => "switch",
            Kind::Cant => "cant",
        }
    }
    fn parse(value: &str) -> PyResult<Kind> {
        match value {
            "move" => Ok(Kind::Move),
            "switch" => Ok(Kind::Switch),
            "cant" => Ok(Kind::Cant),
            other => Err(PyValueError::new_err(format!("unknown token kind {other:?}"))),
        }
    }
    // Sub-blocks default to kind "" (NEGATED/PENDING/ABSENT).
    fn parse_opt(value: &str) -> PyResult<Option<Kind>> {
        if value.is_empty() {
            Ok(None)
        } else {
            Kind::parse(value).map(Some)
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Outcome {
    Absorbed,
    Immune,
    Blocked,
    BrokeSub,
    HitSub,
    Endured,
    Normal,
}

impl Outcome {
    fn rank(self) -> u8 {
        match self {
            Outcome::Absorbed => 0,
            Outcome::Immune => 1,
            Outcome::Blocked => 2,
            Outcome::BrokeSub => 3,
            Outcome::HitSub => 4,
            Outcome::Endured => 5,
            Outcome::Normal => 9,
        }
    }
    fn as_str(self) -> &'static str {
        match self {
            Outcome::Absorbed => "absorbed",
            Outcome::Immune => "immune",
            Outcome::Blocked => "blocked",
            Outcome::BrokeSub => "broke-sub",
            Outcome::HitSub => "hit-sub",
            Outcome::Endured => "endured",
            Outcome::Normal => "normal",
        }
    }
    fn parse(value: &str) -> PyResult<Outcome> {
        Ok(match value {
            "absorbed" => Outcome::Absorbed,
            "immune" => Outcome::Immune,
            "blocked" => Outcome::Blocked,
            "broke-sub" => Outcome::BrokeSub,
            "hit-sub" => Outcome::HitSub,
            "endured" => Outcome::Endured,
            "normal" => Outcome::Normal,
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown damage outcome {other:?}"
                )))
            }
        })
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum SideEffect {
    Charging,
    Drain,
    HazardSet,
    HazardClear,
    WeatherSet,
    StatusInflicted,
    Heal,
    Boost,
    None_,
}

impl SideEffect {
    fn rank(self) -> u8 {
        match self {
            SideEffect::Charging => 0,
            SideEffect::Drain => 1,
            SideEffect::HazardSet => 2,
            SideEffect::HazardClear => 3,
            SideEffect::WeatherSet => 4,
            SideEffect::StatusInflicted => 5,
            SideEffect::Heal => 6,
            SideEffect::Boost => 7,
            SideEffect::None_ => 99,
        }
    }
    fn as_str(self) -> &'static str {
        match self {
            SideEffect::Charging => "charging",
            SideEffect::Drain => "drain",
            SideEffect::HazardSet => "hazard-set",
            SideEffect::HazardClear => "hazard-clear",
            SideEffect::WeatherSet => "weather-set",
            SideEffect::StatusInflicted => "status-inflicted",
            SideEffect::Heal => "heal",
            SideEffect::Boost => "boost",
            SideEffect::None_ => "none",
        }
    }
    fn parse(value: &str) -> PyResult<SideEffect> {
        Ok(match value {
            "charging" => SideEffect::Charging,
            "drain" => SideEffect::Drain,
            "hazard-set" => SideEffect::HazardSet,
            "hazard-clear" => SideEffect::HazardClear,
            "weather-set" => SideEffect::WeatherSet,
            "status-inflicted" => SideEffect::StatusInflicted,
            "heal" => SideEffect::Heal,
            "boost" => SideEffect::Boost,
            "none" => SideEffect::None_,
            other => {
                return Err(PyValueError::new_err(format!("unknown side effect {other:?}")))
            }
        })
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Effectiveness {
    Neutral,
    Super,
    Resisted,
    Immune,
}

impl Effectiveness {
    fn as_str(self) -> &'static str {
        match self {
            Effectiveness::Neutral => "neutral",
            Effectiveness::Super => "super",
            Effectiveness::Resisted => "resisted",
            Effectiveness::Immune => "immune",
        }
    }
    fn parse(value: &str) -> PyResult<Effectiveness> {
        Ok(match value {
            "neutral" => Effectiveness::Neutral,
            "super" => Effectiveness::Super,
            "resisted" => Effectiveness::Resisted,
            "immune" => Effectiveness::Immune,
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown effectiveness {other:?}"
                )))
            }
        })
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum SwitchReason {
    Lead,
    Voluntary,
    Replacement,
    BatonPass,
}

impl SwitchReason {
    fn as_str(self) -> &'static str {
        match self {
            SwitchReason::Lead => "lead",
            SwitchReason::Voluntary => "voluntary",
            SwitchReason::Replacement => "replacement",
            SwitchReason::BatonPass => "baton-pass",
        }
    }
    fn parse(value: &str) -> PyResult<SwitchReason> {
        Ok(match value {
            "lead" => SwitchReason::Lead,
            "voluntary" => SwitchReason::Voluntary,
            "replacement" => SwitchReason::Replacement,
            "baton-pass" => SwitchReason::BatonPass,
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown switch reason {other:?}"
                )))
            }
        })
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Status {
    Action,
    Negated,
    Pending,
    Absent,
}

impl Status {
    fn as_str(self) -> &'static str {
        match self {
            Status::Action => "action",
            Status::Negated => "negated",
            Status::Pending => "pending",
            Status::Absent => "absent",
        }
    }
    fn parse(value: &str) -> PyResult<Status> {
        Ok(match value {
            "action" => Status::Action,
            "negated" => Status::Negated,
            "pending" => Status::Pending,
            "absent" => Status::Absent,
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown sub-block status {other:?}"
                )))
            }
        })
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Phase {
    Turn,
    Lead,
    Replacement,
    Extra,
}

impl Phase {
    fn as_str(self) -> &'static str {
        match self {
            Phase::Turn => "turn",
            Phase::Lead => "lead",
            Phase::Replacement => "replacement",
            Phase::Extra => "extra",
        }
    }
    fn parse(value: &str) -> PyResult<Phase> {
        Ok(match value {
            "turn" => Phase::Turn,
            "lead" => Phase::Lead,
            "replacement" => Phase::Replacement,
            "extra" => Phase::Extra,
            other => return Err(PyValueError::new_err(format!("unknown phase {other:?}"))),
        })
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
enum RepPos {
    First,
    Second,
}

impl RepPos {
    fn as_str(self) -> &'static str {
        match self {
            RepPos::First => "first",
            RepPos::Second => "second",
        }
    }
    fn parse(value: &str) -> PyResult<RepPos> {
        Ok(match value {
            "first" => RepPos::First,
            "second" => RepPos::Second,
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown rep position {other:?}"
                )))
            }
        })
    }
}

// ---------------------------------------------------------------------------
// Sides ("p1"/"p2" as 0/1 internally; strings at every payload boundary)
// ---------------------------------------------------------------------------

fn side_str(side: u8) -> &'static str {
    if side == 0 {
        "p1"
    } else {
        "p2"
    }
}

fn other(side: u8) -> u8 {
    1 - side
}

fn parse_side(value: &str) -> PyResult<u8> {
    match value {
        "p1" => Ok(0),
        "p2" => Ok(1),
        other => Err(PyValueError::new_err(format!(
            "perspective_slot must be 'p1' or 'p2', got {other:?}."
        ))),
    }
}

// ---------------------------------------------------------------------------
// String helpers (ports of showdown.py / belief.py private helpers)
// ---------------------------------------------------------------------------

/// `showdown._normalize_identifier`: lowercase, keep only [a-z0-9].
fn normalize_identifier(value: &str) -> String {
    value
        .to_lowercase()
        .chars()
        .filter(|c| c.is_ascii_lowercase() || c.is_ascii_digit())
        .collect()
}

/// `showdown._slot_from_ident`: `re.match(r"^(p[12])", ident.strip())`.
fn slot_from_ident(ident: &str) -> Option<u8> {
    let bytes = ident.trim().as_bytes();
    if bytes.len() >= 2 && bytes[0] == b'p' {
        match bytes[1] {
            b'1' => Some(0),
            b'2' => Some(1),
            _ => None,
        }
    } else {
        None
    }
}

/// `showdown._species_from_ident`: `ident.split(":", 1)[-1].strip() or "unknown"`.
fn species_from_ident(ident: &str) -> String {
    let after = match ident.find(':') {
        Some(i) => &ident[i + 1..],
        None => ident,
    };
    let trimmed = after.trim();
    if trimmed.is_empty() {
        "unknown".to_string()
    } else {
        trimmed.to_string()
    }
}

/// `showdown._species_from_details`: `details.split(",", 1)[0].strip()`.
fn species_from_details(details: &str) -> String {
    details.split(',').next().unwrap_or("").trim().to_string()
}

/// `showdown._side_condition_identifier`.
fn side_condition_identifier(raw: &str) -> String {
    let mut condition = raw.trim();
    if let Some(colon) = condition.find(':') {
        let prefix = condition[..colon].trim().to_lowercase();
        if prefix == "move" || prefix == "ability" || prefix == "item" {
            condition = condition[colon + 1..].trim();
        }
    }
    normalize_identifier(condition)
}

/// `showdown._condition_features(...).hp_fraction` (the only field the fold reads).
/// Mirrors Python: `partition("/")` on the first whitespace token; float division
/// with ZeroDivisionError -> None; bare "0" -> 0.0; clamp to [0, 1].
fn condition_hp_fraction(condition: Option<&str>) -> Option<f64> {
    let text = condition.unwrap_or("");
    let head = text.split_whitespace().next()?;
    if let Some(slash) = head.find('/') {
        let numerator: f64 = head[..slash].trim().parse().ok()?;
        let denominator: f64 = head[slash + 1..].trim().parse().ok()?;
        if denominator == 0.0 {
            return None; // Python raises ZeroDivisionError -> None
        }
        let value = numerator / denominator;
        Some(if value < 0.0 {
            0.0
        } else if value > 1.0 {
            1.0
        } else {
            value
        })
    } else if head == "0" {
        Some(0.0)
    } else {
        None
    }
}

/// `transitions._from_tag_payload`: regex `\[from\]\s*([^|\[\]]*)`, stripped, empty -> None.
fn from_tag_payload(raw_line: &str) -> Option<String> {
    let idx = raw_line.find("[from]")?;
    let rest = &raw_line[idx + 6..];
    let rest = rest.trim_start_matches(|c: char| c.is_whitespace());
    let end = rest
        .find(|c| c == '|' || c == '[' || c == ']')
        .unwrap_or(rest.len());
    let payload = rest[..end].trim();
    if payload.is_empty() {
        None
    } else {
        Some(payload.to_string())
    }
}

/// `transitions._of_tag_slot`: regex `\[of\]\s*(p[12])` (first matching occurrence).
fn of_tag_slot(raw_line: &str) -> Option<u8> {
    let mut haystack = raw_line;
    while let Some(pos) = haystack.find("[of]") {
        let after = &haystack[pos + 4..];
        let trimmed = after.trim_start_matches(|c: char| c.is_whitespace());
        let bytes = trimmed.as_bytes();
        if bytes.len() >= 2 && bytes[0] == b'p' {
            match bytes[1] {
                b'1' => return Some(0),
                b'2' => return Some(1),
                _ => {}
            }
        }
        haystack = after;
    }
    None
}

/// `showdown._line_mentions_baton_pass`: any "baton pass" in parts[4:] (lowercased).
fn line_mentions_baton_pass(parts: &[&str]) -> bool {
    parts
        .iter()
        .skip(4)
        .any(|part| part.to_lowercase().contains("baton pass"))
}

/// `belief._called_move_source`: normalized caller id after the first `[from]`, or None.
fn called_move_source(raw_line: &str) -> Option<String> {
    let marker = raw_line.find("[from]")?;
    let tag = &raw_line[marker + 6..];
    let tag = tag.split('|').next().unwrap_or("");
    let mut tag = tag.trim();
    if tag.to_lowercase().starts_with("move:") {
        tag = tag[5..].trim();
    }
    Some(normalize_identifier(tag))
}

fn is_caller_move(id: &str) -> bool {
    matches!(
        id,
        "metronome" | "mirrormove" | "sleeptalk" | "assist" | "naturepower" | "copycat"
    )
}

fn is_pursuit_scan_boundary(event_type: &str) -> bool {
    matches!(
        event_type,
        "move" | "switch" | "drag" | "replace" | "cant" | "turn" | "upkeep"
    )
}

fn is_cant_no_choice_reason(action: &str) -> bool {
    action == "recharge"
}

fn is_self_cost_from_tag(normalized: &str) -> bool {
    matches!(normalized, "recoil" | "strugglerecoil")
}

fn is_self_faint_cost_move(action: &str) -> bool {
    matches!(action, "explosion" | "selfdestruct" | "memento")
}

fn is_absorb_ability(id: &str) -> bool {
    matches!(id, "voltabsorb" | "waterabsorb" | "flashfire")
}

/// `transitions._is_absorb_signature`.
fn is_absorb_signature(from_payload: Option<&str>) -> bool {
    let payload = match from_payload {
        Some(p) => p,
        None => return false,
    };
    if !payload.to_lowercase().starts_with("ability:") {
        return false;
    }
    let after = match payload.find(':') {
        Some(i) => &payload[i + 1..],
        None => return false,
    };
    is_absorb_ability(&normalize_identifier(after))
}

/// `transitions._is_absorb_start`: `|-start|...|ability: Flash Fire` form.
fn is_absorb_start(event_type: &str, parts: &[&str]) -> bool {
    if event_type != "-start" || parts.len() < 4 {
        return false;
    }
    is_absorb_ability(&side_condition_identifier(parts[3]))
}

/// `showdown._update_side_conditions` (spikes cap 3, others 1; -sideend removes the key).
fn update_side_conditions(parts: &[&str], counts: &mut [BTreeMap<String, i64>; 2]) {
    let event_type = *parts.get(1).unwrap_or(&"");
    if (event_type != "-sidestart" && event_type != "-sideend") || parts.len() < 4 {
        return;
    }
    let slot = match slot_from_ident(parts[2]) {
        Some(s) => s,
        None => return,
    };
    let condition = side_condition_identifier(parts[3]);
    if condition.is_empty() {
        return;
    }
    if event_type == "-sidestart" {
        let max_layers = if condition == "spikes" { 3 } else { 1 };
        let entry = counts[slot as usize].entry(condition).or_insert(0);
        *entry = (*entry + 1).min(max_layers);
    } else {
        counts[slot as usize].remove(&condition);
    }
}

/// `showdown._update_weather` ('none'/absent clears it).
fn update_weather(parts: &[&str], weather: &mut Option<String>) {
    if *parts.get(1).unwrap_or(&"") != "-weather" {
        return;
    }
    let raw = parts.get(2).map(|s| s.trim()).unwrap_or("");
    let identifier = normalize_identifier(raw);
    *weather = if identifier.is_empty() || identifier == "none" {
        None
    } else {
        Some(identifier)
    };
}

// ---------------------------------------------------------------------------
// Core data (mirrors transitions._Window / TransitionToken / turn_merged.*)
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct Window {
    event_index: i64,
    turn: i64,
    side: u8,
    species: String,
    kind: Kind,
    action: String,
    defender_side: Option<u8>,
    defender_species: Option<String>,
    called: bool,
    transformed: bool,
    own_spikes_layers: i64,
    opp_spikes_layers: i64,
    weather: Option<String>,
    damage_fraction: f64,
    self_hp_cost: f64,
    outcome: Outcome,
    crit: bool,
    miss: bool,
    ko: bool,
    pursuit_intercept: bool,
    n_hits: i64,
    effectiveness: Effectiveness,
    side_effect: SideEffect,
    defender_hit_by_move: bool,
    voluntary_switch: bool,
    locked_continuation: bool,
    switch_reason: Option<SwitchReason>,
    other_side_pending_replacement: bool,
}

impl Window {
    #[allow(clippy::too_many_arguments)]
    fn new(
        event_index: i64,
        turn: i64,
        side: u8,
        species: String,
        kind: Kind,
        action: String,
        defender_side: Option<u8>,
        defender_species: Option<String>,
        called: bool,
        transformed: bool,
        own_spikes_layers: i64,
        opp_spikes_layers: i64,
        weather: Option<String>,
    ) -> Window {
        Window {
            event_index,
            turn,
            side,
            species,
            kind,
            action,
            defender_side,
            defender_species,
            called,
            transformed,
            own_spikes_layers,
            opp_spikes_layers,
            weather,
            damage_fraction: 0.0,
            self_hp_cost: 0.0,
            outcome: Outcome::Normal,
            crit: false,
            miss: false,
            ko: false,
            pursuit_intercept: false,
            n_hits: 1,
            effectiveness: Effectiveness::Neutral,
            side_effect: SideEffect::None_,
            defender_hit_by_move: false,
            voluntary_switch: false,
            locked_continuation: false,
            switch_reason: None,
            other_side_pending_replacement: false,
        }
    }

    fn upgrade_outcome(&mut self, outcome: Outcome) {
        if outcome.rank() < self.outcome.rank() {
            self.outcome = outcome;
        }
    }

    fn upgrade_side_effect(&mut self, category: SideEffect) {
        if category.rank() < self.side_effect.rank() {
            self.side_effect = category;
        }
    }
}

#[derive(Clone, Debug)]
struct TransitionToken {
    turn: i64,
    actor_slot: u8,
    actor_species: String,
    kind: Kind,
    action: String,
    called: bool,
    transformed: bool,
    damage_fraction: f64,
    damage_outcome: Outcome,
    crit: bool,
    miss: bool,
    ko: bool,
    pursuit_intercept: bool,
    n_hits: i64,
    effectiveness: Effectiveness,
    side_effect: SideEffect,
    self_hp_cost: f64,
    own_spikes_layers: i64,
    opp_spikes_layers: i64,
    weather: Option<String>,
    defender_species: Option<String>,
    residual: Option<f64>,
    residual_valid: bool,
    cb_bit: bool,
    investment: f64,
}

/// `transitions_fold._token_from_window`.
fn token_from_window(window: &Window) -> TransitionToken {
    TransitionToken {
        turn: window.turn,
        actor_slot: window.side,
        actor_species: window.species.clone(),
        kind: window.kind,
        action: window.action.clone(),
        called: window.called,
        transformed: window.transformed,
        damage_fraction: window.damage_fraction,
        damage_outcome: window.outcome,
        crit: window.crit,
        miss: window.miss,
        ko: window.ko,
        pursuit_intercept: window.pursuit_intercept,
        n_hits: window.n_hits,
        effectiveness: window.effectiveness,
        side_effect: window.side_effect,
        self_hp_cost: window.self_hp_cost,
        own_spikes_layers: window.own_spikes_layers,
        opp_spikes_layers: window.opp_spikes_layers,
        weather: window.weather.clone(),
        defender_species: if window.kind == Kind::Move {
            window.defender_species.clone()
        } else {
            None
        },
        residual: None,
        residual_valid: false,
        cb_bit: false,
        investment: 0.0,
    }
}

#[derive(Clone, Debug)]
struct SubBlock {
    status: Status,
    actor_slot: u8,
    actor_species: String,
    kind: Option<Kind>, // None serializes as "" (NEGATED/PENDING/ABSENT defaults)
    action: String,
    called: bool,
    transformed: bool,
    damage_fraction: f64,
    self_hp_cost: f64,
    damage_outcome: Outcome,
    crit: bool,
    miss: bool,
    ko: bool,
    pursuit_intercept: bool,
    n_hits: i64,
    effectiveness: Effectiveness,
    side_effect: SideEffect,
    defender_species: Option<String>,
    cant_reason: Option<String>,
    baton_pass_species: Option<String>,
    residual: Option<f64>,
    residual_valid: bool,
    cb_bit: bool,
    investment: f64,
}

impl SubBlock {
    fn defaults(status: Status, actor_slot: u8) -> SubBlock {
        SubBlock {
            status,
            actor_slot,
            actor_species: String::new(),
            kind: None,
            action: String::new(),
            called: false,
            transformed: false,
            damage_fraction: 0.0,
            self_hp_cost: 0.0,
            damage_outcome: Outcome::Normal,
            crit: false,
            miss: false,
            ko: false,
            pursuit_intercept: false,
            n_hits: 1,
            effectiveness: Effectiveness::Neutral,
            side_effect: SideEffect::None_,
            defender_species: None,
            cant_reason: None,
            baton_pass_species: None,
            residual: None,
            residual_valid: false,
            cb_bit: false,
            investment: 0.0,
        }
    }
}

#[derive(Clone, Debug)]
struct MergedToken {
    turn: i64,
    phase: Phase,
    first: SubBlock,
    second: SubBlock,
    own_spikes_layers: i64,
    opp_spikes_layers: i64,
    weather: Option<String>,
}

#[derive(Clone, Debug, Default)]
struct StayRecord {
    species: String,
    moved: bool,
}

#[derive(Clone, Debug, Default)]
struct MonCounters {
    switched_out_before_attacking: i64,
    stayed_and_attacked: i64,
    turns_active: i64,
}

type AnnotationValues = (Option<f64>, bool, bool, f64);

// ---------------------------------------------------------------------------
// Merge helpers (ports of turn_merged._merge_turn and friends)
// ---------------------------------------------------------------------------

/// `turn_merged._action_sub_block` (+ the `_chain_sub_block` collapse fields).
fn action_sub_block(
    window: &Window,
    cant_reason: Option<String>,
    baton_pass_species: Option<String>,
) -> SubBlock {
    SubBlock {
        status: Status::Action,
        actor_slot: window.side,
        actor_species: window.species.clone(),
        kind: Some(window.kind),
        action: window.action.clone(),
        called: window.called,
        transformed: window.transformed,
        damage_fraction: window.damage_fraction,
        self_hp_cost: window.self_hp_cost,
        damage_outcome: window.outcome,
        crit: window.crit,
        miss: window.miss,
        ko: window.ko,
        pursuit_intercept: window.pursuit_intercept,
        n_hits: window.n_hits,
        effectiveness: window.effectiveness,
        side_effect: window.side_effect,
        defender_species: window.defender_species.clone(),
        cant_reason,
        baton_pass_species,
        residual: None,
        residual_valid: false,
        cb_bit: false,
        investment: 0.0,
    }
}

/// `turn_merged._single_phase`.
fn single_phase(window: &Window, phase: Phase) -> MergedToken {
    MergedToken {
        turn: window.turn,
        phase,
        first: action_sub_block(window, None, None),
        second: SubBlock::defaults(Status::Absent, other(window.side)),
        own_spikes_layers: window.own_spikes_layers,
        opp_spikes_layers: window.opp_spikes_layers,
        weather: window.weather.clone(),
    }
}

/// `turn_merged._missing_sub_block`.
fn missing_sub_block(
    side: u8,
    turn: i64,
    turn_start_occupants: &BTreeMap<i64, [Option<String>; 2]>,
    status: Status,
) -> SubBlock {
    let species = turn_start_occupants
        .get(&turn)
        .and_then(|occupants| occupants[side as usize].clone())
        .unwrap_or_default();
    let mut sub = SubBlock::defaults(status, side);
    sub.actor_species = species;
    sub
}

/// `turn_merged._is_protocol_constant`.
fn is_protocol_constant(window: &Window) -> bool {
    window.damage_fraction == 0.0
        && window.self_hp_cost == 0.0
        && window.outcome == Outcome::Normal
        && !window.crit
        && !window.miss
        && !window.ko
        && !window.pursuit_intercept
        && window.n_hits == 1
        && window.effectiveness == Effectiveness::Neutral
        && window.side_effect == SideEffect::None_
        && !window.called
}

struct Chain<'a> {
    side: u8,
    start_index: i64,
    representative: &'a Window,
    cant_reason: Option<String>,
    baton_pass_species: Option<String>,
}

fn chain_sub_block(chain: &Chain<'_>) -> SubBlock {
    action_sub_block(
        chain.representative,
        chain.cant_reason.clone(),
        chain.baton_pass_species.clone(),
    )
}

/// `turn_merged._reduce_side_chain`.
fn reduce_side_chain<'a>(side: u8, seq: Vec<&'a Window>) -> (Chain<'a>, Vec<&'a Window>) {
    let start_index = seq[0].event_index;
    let mut cant_reason: Option<String> = None;
    let mut rest: &[&Window] = &seq[..];
    if rest.len() >= 2
        && rest[0].kind == Kind::Cant
        && rest[1].kind == Kind::Move
        && rest[1].action == "sleeptalk"
        && is_protocol_constant(rest[0])
    {
        if rest.len() >= 3 && rest[2].kind == Kind::Move && rest[2].called {
            if is_protocol_constant(rest[1]) {
                cant_reason = Some(rest[0].action.clone());
                rest = &rest[2..];
            }
        } else {
            cant_reason = Some(rest[0].action.clone());
            rest = &rest[1..];
        }
    }
    let representative = rest[0];
    let mut rest = &rest[1..];
    let mut baton_pass_species: Option<String> = None;
    if !rest.is_empty()
        && representative.kind == Kind::Move
        && rest[0].kind == Kind::Switch
        && rest[0].switch_reason == Some(SwitchReason::BatonPass)
    {
        baton_pass_species = Some(rest[0].action.clone());
        rest = &rest[1..];
    }
    (
        Chain {
            side,
            start_index,
            representative,
            cant_reason,
            baton_pass_species,
        },
        rest.to_vec(),
    )
}

/// `turn_merged._merge_turn`.
fn merge_turn(
    turn: i64,
    turn_windows: &[&Window],
    turn_start_occupants: &BTreeMap<i64, [Option<String>; 2]>,
    consumption_confirmed: bool,
    out: &mut Vec<MergedToken>,
) {
    let mut declared: Vec<&Window> = Vec::new();
    let mut replacements: Vec<&Window> = Vec::new();
    for (position, window) in turn_windows.iter().enumerate() {
        if window.switch_reason == Some(SwitchReason::Replacement) {
            // Pursuit KO-intercept continuation: the previously chosen switch
            // completes in the same breath — a declared action, not a replacement.
            let previous = if position > 0 {
                Some(turn_windows[position - 1])
            } else {
                None
            };
            let continuation = previous.map_or(false, |p| {
                p.kind == Kind::Move && p.pursuit_intercept && p.ko && p.side != window.side
            });
            if continuation {
                declared.push(window);
            } else {
                replacements.push(window);
            }
        } else {
            declared.push(window);
        }
    }

    let mut chains: Vec<Chain<'_>> = Vec::new();
    let mut extras: Vec<&Window> = Vec::new();
    for side in [0u8, 1u8] {
        let side_windows: Vec<&Window> = declared
            .iter()
            .copied()
            .filter(|window| window.side == side)
            .collect();
        if side_windows.is_empty() {
            continue;
        }
        let (chain, side_extras) = reduce_side_chain(side, side_windows);
        chains.push(chain);
        extras.extend(side_extras);
    }
    chains.sort_by_key(|chain| chain.start_index);

    let mut phases: Vec<(i64, MergedToken)> = Vec::new();
    if !chains.is_empty() {
        let first_chain = &chains[0];
        let second = if chains.len() > 1 {
            chain_sub_block(&chains[1])
        } else {
            missing_sub_block(
                other(first_chain.side),
                turn,
                turn_start_occupants,
                if consumption_confirmed {
                    Status::Negated
                } else {
                    Status::Pending
                },
            )
        };
        let anchor = first_chain.representative;
        phases.push((
            first_chain.start_index,
            MergedToken {
                turn,
                phase: Phase::Turn,
                first: chain_sub_block(first_chain),
                second,
                own_spikes_layers: anchor.own_spikes_layers,
                opp_spikes_layers: anchor.opp_spikes_layers,
                weather: anchor.weather.clone(),
            },
        ));
    }
    for extra in extras {
        phases.push((extra.event_index, single_phase(extra, Phase::Extra)));
    }

    // Replacement phases: cold pairs merge, sequential faints stay single.
    let mut position = 0usize;
    while position < replacements.len() {
        let window = replacements[position];
        let partner = replacements.get(position + 1).copied();
        let cold_pair = window.other_side_pending_replacement
            && partner.map_or(false, |p| p.side == other(window.side));
        if cold_pair {
            let partner = partner.unwrap();
            phases.push((
                window.event_index,
                MergedToken {
                    turn,
                    phase: Phase::Replacement,
                    first: action_sub_block(window, None, None),
                    second: action_sub_block(partner, None, None),
                    own_spikes_layers: window.own_spikes_layers,
                    opp_spikes_layers: window.opp_spikes_layers,
                    weather: window.weather.clone(),
                },
            ));
            position += 2;
        } else {
            phases.push((window.event_index, single_phase(window, Phase::Replacement)));
            position += 1;
        }
    }

    phases.sort_by_key(|(index, _)| *index); // stable, like Python list.sort
    out.extend(phases.into_iter().map(|(_, token)| token));
}

/// `transitions_fold._merge_window_run`.
fn merge_window_run(
    windows: &[&Window],
    lead_done_in: bool,
    turn_start_occupants: &BTreeMap<i64, [Option<String>; 2]>,
    completed_turns: &BTreeSet<i64>,
    fainted_turns: &BTreeSet<i64>,
) -> (Vec<MergedToken>, bool) {
    let mut tokens: Vec<MergedToken> = Vec::new();
    let mut index = 0usize;
    let mut lead_done = lead_done_in;
    if !lead_done {
        let mut leads: Vec<&Window> = Vec::new();
        while index < windows.len() && windows[index].switch_reason == Some(SwitchReason::Lead) {
            leads.push(windows[index]);
            index += 1;
        }
        if !leads.is_empty() {
            let first = leads[0];
            let second = leads.get(1).copied();
            tokens.push(MergedToken {
                turn: first.turn,
                phase: Phase::Lead,
                first: action_sub_block(first, None, None),
                second: match second {
                    Some(window) => action_sub_block(window, None, None),
                    None => SubBlock::defaults(Status::Absent, other(first.side)),
                },
                own_spikes_layers: first.own_spikes_layers,
                opp_spikes_layers: first.opp_spikes_layers,
                weather: first.weather.clone(),
            });
            for extra in leads.iter().skip(2) {
                // unreachable in singles; bijection safety valve
                tokens.push(single_phase(extra, Phase::Extra));
            }
        }
        if !windows.is_empty() {
            // The batch lead pass runs exactly once, over the stream-initial run.
            lead_done = true;
        }
    }

    while index < windows.len() {
        let turn = windows[index].turn;
        let start = index;
        while index < windows.len() && windows[index].turn == turn {
            index += 1;
        }
        merge_turn(
            turn,
            &windows[start..index],
            turn_start_occupants,
            completed_turns.contains(&turn) || fainted_turns.contains(&turn),
            &mut tokens,
        );
    }
    (tokens, lead_done)
}

/// `transitions_fold._representative_offset`.
fn representative_offset(sub: &SubBlock) -> i64 {
    let mut offset = 0;
    if sub.cant_reason.is_some() {
        offset += 1; // the cant token precedes the representative
        if sub.called {
            offset += 1; // so does the synthesized Sleep Talk click
        }
    }
    offset
}

/// `turn_merged._expansion_length`.
fn expansion_length(sub: &SubBlock) -> i64 {
    let mut length = 1;
    if sub.cant_reason.is_some() {
        length += 1;
        if sub.called {
            length += 1;
        }
    }
    if sub.baton_pass_species.is_some() {
        length += 1;
    }
    length
}

// ---------------------------------------------------------------------------
// The fold state
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct FoldStateInner {
    perspective_slot: u8,
    merged_tail_limit: usize,
    action_tail_limit: usize,

    // --- fold core (probe #1) ---
    event_index: i64,
    side_condition_counts: [BTreeMap<String, i64>; 2],
    weather: Option<String>,
    turn_number: i64,
    hp_fraction: [Option<f64>; 2],
    occupant: [Option<StayRecord>; 2],
    transformed: [bool; 2],
    pending_baton_pass: [bool; 2],
    pending_faint_replacement: [bool; 2],
    lead_seen: [bool; 2],
    pending_charge: [Option<String>; 2],
    current_window: Option<Window>,
    pursuit_buffer: Vec<String>,

    // --- merge staging (probe #8) ---
    lead_done: bool,
    pending_windows: Vec<Window>,
    merged_done: VecDeque<(MergedToken, Option<i64>, Option<i64>)>,
    merged_total: i64,
    expansion_cursor: i64,

    // --- per-action token tail ---
    action_tail: VecDeque<TransitionToken>,
    action_total: i64,

    // --- tendency running state (probe #4/#5/#6) ---
    opponent_switch_count: i64,
    opponent_decision_opportunities: i64,
    last_opponent_opportunity_turn: Option<i64>,
    blocked_on_our_attack_count: i64,
    pursuit_intercept_predict_count: i64,
    my_switch_turn_count: i64,
    mon_counters: BTreeMap<(u8, String), MonCounters>,
    weather_reveals: BTreeMap<(u8, String), bool>,

    // --- bounded recent-turn slices (probe #7) ---
    turn_start_occupants: BTreeMap<i64, [Option<String>; 2]>,
    completed_turns: BTreeSet<i64>,
    fainted_turns: BTreeSet<i64>,

    // --- Tier-2 annotation overlay (probe #9) ---
    annotations: BTreeMap<i64, AnnotationValues>,
    rep_index_map: BTreeMap<i64, (i64, RepPos)>,
    cb_pinned: BTreeSet<String>,
    investment_pinned_state: BTreeMap<String, (i64, f64)>,
}

impl FoldStateInner {
    fn initial(perspective_slot: u8, merged_tail_limit: usize, action_tail_limit: usize) -> Self {
        FoldStateInner {
            perspective_slot,
            merged_tail_limit,
            action_tail_limit,
            event_index: 0,
            side_condition_counts: [BTreeMap::new(), BTreeMap::new()],
            weather: None,
            turn_number: 0,
            hp_fraction: [None, None],
            occupant: [None, None],
            transformed: [false, false],
            pending_baton_pass: [false, false],
            pending_faint_replacement: [false, false],
            lead_seen: [false, false],
            pending_charge: [None, None],
            current_window: None,
            pursuit_buffer: Vec::new(),
            lead_done: false,
            pending_windows: Vec::new(),
            merged_done: VecDeque::new(),
            merged_total: 0,
            expansion_cursor: 0,
            action_tail: VecDeque::new(),
            action_total: 0,
            opponent_switch_count: 0,
            opponent_decision_opportunities: 0,
            last_opponent_opportunity_turn: None,
            blocked_on_our_attack_count: 0,
            pursuit_intercept_predict_count: 0,
            my_switch_turn_count: 0,
            mon_counters: BTreeMap::new(),
            weather_reveals: BTreeMap::new(),
            turn_start_occupants: BTreeMap::new(),
            completed_turns: BTreeSet::new(),
            fainted_turns: BTreeSet::new(),
            annotations: BTreeMap::new(),
            rep_index_map: BTreeMap::new(),
            cb_pinned: BTreeSet::new(),
            investment_pinned_state: BTreeMap::new(),
        }
    }

    fn opponent_slot(&self) -> u8 {
        other(self.perspective_slot)
    }

    // ------------------------------------------------------------------ advance

    fn advance_in_place(&mut self, raw_lines: &[String]) -> PyResult<()> {
        for raw_line in raw_lines {
            let parts: Vec<&str> = raw_line.split('|').collect();
            let event_type = *parts.get(1).unwrap_or(&"");
            if event_type == "t:" {
                continue; // wall-clock line: never battle state (schema-v2 filter)
            }
            self.process_line(raw_line, &parts, event_type)?;
            self.event_index += 1;
            // Pursuit ring buffer maintenance AFTER processing: boundary-type lines
            // clear it; everything else — blank separators included — joins the scan set.
            if is_pursuit_scan_boundary(event_type) {
                self.pursuit_buffer.clear();
            } else {
                self.pursuit_buffer.push(raw_line.clone());
            }
        }
        Ok(())
    }

    fn context_trio(&self) -> (i64, i64, Option<String>) {
        let own = *self.side_condition_counts[self.perspective_slot as usize]
            .get("spikes")
            .unwrap_or(&0);
        let opp = *self.side_condition_counts[self.opponent_slot() as usize]
            .get("spikes")
            .unwrap_or(&0);
        (own, opp, self.weather.clone())
    }

    fn counters_for(&mut self, side: u8, species: &str) -> &mut MonCounters {
        self.mon_counters
            .entry((side, species.to_string()))
            .or_default()
    }

    fn open_window(&mut self, window: Window) {
        self.close_window();
        self.current_window = Some(window);
    }

    fn close_window(&mut self) {
        if let Some(window) = self.current_window.take() {
            let token = token_from_window(&window);
            self.action_tail.push_back(token.clone());
            while self.action_tail.len() > self.action_tail_limit {
                self.action_tail.pop_front();
            }
            self.action_total += 1;
            self.accumulate_tendency(&window, &token);
            self.pending_windows.push(window);
        }
    }

    /// One line of the reference `_process_line`, against carried state.
    fn process_line(&mut self, raw_line: &str, parts: &[&str], event_type: &str) -> PyResult<()> {
        if event_type.is_empty() || event_type == "upkeep" {
            self.close_window();
            if event_type == "upkeep" {
                self.completed_turns.insert(self.turn_number);
            }
            return Ok(());
        }

        if event_type == "turn" {
            self.close_window();
            self.completed_turns.insert(self.turn_number);
            // The pending group is complete: |turn|N+1 guarantees no more turn-N
            // windows AND freezes the NEGATED gate's inputs for turn N (probe #7/#8).
            self.flush_pending()?;
            if let Some(turn) = parts.get(2).and_then(|p| p.trim().parse::<i64>().ok()) {
                self.turn_number = turn;
            }
            for side in [0u8, 1u8] {
                if let Some(stay) = self.occupant[side as usize].clone() {
                    self.counters_for(side, &stay.species).turns_active += 1;
                }
            }
            let occupants = [
                self.occupant[0].as_ref().map(|stay| stay.species.clone()),
                self.occupant[1].as_ref().map(|stay| stay.species.clone()),
            ];
            self.turn_start_occupants.insert(self.turn_number, occupants);
            self.prune_turn_maps();
            return Ok(());
        }

        if event_type == "win" {
            self.close_window();
            self.completed_turns.insert(self.turn_number);
            self.flush_pending()?;
            return Ok(());
        }

        if event_type == "move" && parts.len() >= 4 {
            let side = match slot_from_ident(parts[2]) {
                Some(side) => side,
                None => return Ok(()),
            };
            let species = match &self.occupant[side as usize] {
                Some(stay) => stay.species.clone(),
                None => species_from_ident(parts[2]),
            };
            let called_source = called_move_source(raw_line);
            let called = called_source.as_deref().map_or(false, is_caller_move);
            let move_id = normalize_identifier(parts[3]);
            let locked = called_source.as_deref() == Some("lockedmove")
                || self.pending_charge[side as usize].as_deref() == Some(move_id.as_str());
            self.pending_charge[side as usize] = None;
            let mut stayed_species: Option<String> = None;
            if let Some(stay) = self.occupant[side as usize].as_mut() {
                if !stay.moved {
                    stay.moved = true;
                    stayed_species = Some(stay.species.clone());
                }
            }
            if let Some(stayed) = stayed_species {
                self.counters_for(side, &stayed).stayed_and_attacked += 1;
            }
            self.pending_baton_pass[side as usize] = move_id == "batonpass";
            let defender = parts
                .get(4)
                .and_then(|part| slot_from_ident(part))
                .unwrap_or_else(|| other(side));
            let defender_species = self.occupant[defender as usize]
                .as_ref()
                .map(|stay| stay.species.clone());
            let (own, opp, current_weather) = self.context_trio();
            let mut window = Window::new(
                self.event_index,
                self.turn_number,
                side,
                species,
                Kind::Move,
                move_id.clone(),
                Some(defender),
                defender_species,
                called,
                self.transformed[side as usize],
                own,
                opp,
                current_weather,
            );
            window.locked_continuation = locked;
            // Pursuit intercept, resolved at open time from the ring buffer (probe #3).
            if move_id == "pursuit" {
                for buffered in self.pursuit_buffer.iter().rev() {
                    let buffered_parts: Vec<&str> = buffered.split('|').collect();
                    if buffered_parts.len() >= 4
                        && *buffered_parts.get(1).unwrap_or(&"") == "-activate"
                        && slot_from_ident(buffered_parts[2]) == Some(defender)
                        && side_condition_identifier(buffered_parts[3]) == "pursuit"
                    {
                        window.pursuit_intercept = true;
                        break;
                    }
                }
            }
            self.open_window(window);
            return Ok(());
        }

        if (event_type == "switch" || event_type == "drag" || event_type == "replace")
            && parts.len() >= 4
        {
            let side = match slot_from_ident(parts[2]) {
                Some(side) => side,
                None => return Ok(()),
            };
            let is_lead = !self.lead_seen[side as usize];
            self.lead_seen[side as usize] = true;
            let is_faint_replacement = self.pending_faint_replacement[side as usize];
            let other_pending =
                is_faint_replacement && self.pending_faint_replacement[other(side) as usize];
            self.pending_faint_replacement[side as usize] = false;
            let is_baton_pass =
                self.pending_baton_pass[side as usize] || line_mentions_baton_pass(parts);
            self.pending_baton_pass[side as usize] = false;
            self.pending_charge[side as usize] = None;
            let voluntary =
                event_type == "switch" && !is_lead && !is_faint_replacement && !is_baton_pass;
            let switch_reason = if is_lead {
                SwitchReason::Lead
            } else if is_faint_replacement {
                SwitchReason::Replacement
            } else if is_baton_pass {
                SwitchReason::BatonPass
            } else {
                SwitchReason::Voluntary
            };
            let mut switched_out_species: Option<String> = None;
            if let Some(previous) = &self.occupant[side as usize] {
                if voluntary && !previous.moved {
                    switched_out_species = Some(previous.species.clone());
                }
            }
            if let Some(prev_species) = switched_out_species {
                self.counters_for(side, &prev_species)
                    .switched_out_before_attacking += 1;
            }
            let species = {
                let from_details = species_from_details(parts[3]);
                if from_details.is_empty() {
                    species_from_ident(parts[2])
                } else {
                    from_details
                }
            };
            self.occupant[side as usize] = Some(StayRecord {
                species: species.clone(),
                moved: false,
            });
            self.transformed[side as usize] = false;
            if let Some(fraction) = condition_hp_fraction(parts.get(4).copied()) {
                self.hp_fraction[side as usize] = Some(fraction);
            }
            if event_type == "drag" || event_type == "replace" {
                self.close_window();
                return Ok(());
            }
            let (own, opp, current_weather) = self.context_trio();
            let mut window = Window::new(
                self.event_index,
                self.turn_number,
                side,
                species.clone(),
                Kind::Switch,
                species,
                None,
                None,
                false,
                false,
                own,
                opp,
                current_weather,
            );
            window.voluntary_switch = voluntary;
            window.switch_reason = Some(switch_reason);
            window.other_side_pending_replacement = other_pending;
            self.open_window(window);
            return Ok(());
        }

        if event_type == "cant" && parts.len() >= 4 {
            let side = match slot_from_ident(parts[2]) {
                Some(side) => side,
                None => return Ok(()),
            };
            let species = match &self.occupant[side as usize] {
                Some(stay) => stay.species.clone(),
                None => species_from_ident(parts[2]),
            };
            self.pending_charge[side as usize] = None;
            let (own, opp, current_weather) = self.context_trio();
            let window = Window::new(
                self.event_index,
                self.turn_number,
                side,
                species,
                Kind::Cant,
                side_condition_identifier(parts[3]),
                None,
                None,
                false,
                self.transformed[side as usize],
                own,
                opp,
                current_weather,
            );
            self.open_window(window);
            return Ok(());
        }

        // --- Non-action lines: window accumulation, then global state updates. ---
        let target: Option<u8> = parts.get(2).and_then(|part| slot_from_ident(part));
        let from_payload = from_tag_payload(raw_line);

        if event_type == "-transform" {
            if let Some(target) = target {
                self.transformed[target as usize] = true;
            }
        }

        if event_type == "-damage" && target.is_some() && parts.len() >= 4 {
            let target = target.unwrap();
            let new_fraction = condition_hp_fraction(Some(parts[3]));
            let previous_fraction = self.hp_fraction[target as usize].unwrap_or(1.0);
            if let Some(current) = self.current_window.as_mut() {
                if Some(target) == current.defender_side {
                    if from_payload.is_none() {
                        if current.kind == Kind::Move {
                            if let Some(new_fraction) = new_fraction {
                                let delta = previous_fraction - new_fraction;
                                if delta > 0.0 {
                                    current.damage_fraction += delta;
                                }
                                current.defender_hit_by_move = true;
                            }
                        }
                    } else {
                        current.defender_hit_by_move = false;
                    }
                }
                if target == current.side && current.kind == Kind::Move {
                    if let Some(new_fraction) = new_fraction {
                        let normalized_from = from_payload
                            .as_deref()
                            .map(side_condition_identifier);
                        let counts_as_cost = from_payload.is_none()
                            || normalized_from
                                .as_deref()
                                .map_or(false, is_self_cost_from_tag);
                        if counts_as_cost {
                            let cost_delta = previous_fraction - new_fraction;
                            if cost_delta > 0.0 {
                                current.self_hp_cost += cost_delta;
                            }
                        }
                    }
                }
            }
            if let Some(new_fraction) = new_fraction {
                self.hp_fraction[target as usize] = Some(new_fraction);
            }
        } else if (event_type == "-heal" || event_type == "-sethp")
            && target.is_some()
            && parts.len() >= 4
        {
            let target = target.unwrap();
            let condition_fraction = condition_hp_fraction(Some(parts[3]));
            let previous_fraction = self.hp_fraction[target as usize].unwrap_or(1.0);
            if let Some(current) = self.current_window.as_mut() {
                if event_type == "-sethp"
                    && target == current.side
                    && current.kind == Kind::Move
                    && condition_fraction.is_some()
                    && from_payload
                        .as_deref()
                        .map_or(false, |payload| side_condition_identifier(payload) == "painsplit")
                {
                    let sethp_delta = previous_fraction - condition_fraction.unwrap();
                    if sethp_delta > 0.0 {
                        current.self_hp_cost += sethp_delta;
                    }
                }
            }
            if let Some(fraction) = condition_fraction {
                self.hp_fraction[target as usize] = Some(fraction);
            }
            let is_silent = raw_line.contains("[silent]");
            if let Some(current) = self.current_window.as_mut() {
                if event_type == "-heal" && target == current.side && !is_silent {
                    if from_payload
                        .as_deref()
                        .map_or(false, |payload| normalize_identifier(payload) == "drain")
                    {
                        current.upgrade_side_effect(SideEffect::Drain);
                    } else if from_payload.is_none() {
                        current.upgrade_side_effect(SideEffect::Heal);
                    }
                }
            }
        } else if event_type == "faint" && target.is_some() {
            let target = target.unwrap();
            let remaining = self.hp_fraction[target as usize].unwrap_or(1.0);
            if let Some(current) = self.current_window.as_mut() {
                if target == current.side
                    && current.kind == Kind::Move
                    && is_self_faint_cost_move(&current.action)
                {
                    current.self_hp_cost += remaining;
                }
            }
            self.hp_fraction[target as usize] = Some(0.0);
            self.pending_faint_replacement[target as usize] = true;
            self.fainted_turns.insert(self.turn_number);
            if let Some(current) = self.current_window.as_mut() {
                if Some(target) == current.defender_side && current.defender_hit_by_move {
                    current.ko = true;
                }
            }
        } else if event_type == "-status" && target.is_some() {
            let target = target.unwrap();
            if let Some(current) = self.current_window.as_mut() {
                if target != current.side {
                    current.upgrade_side_effect(SideEffect::StatusInflicted);
                }
            }
        } else if event_type == "-boost" || event_type == "-unboost" || event_type == "-setboost" {
            if from_payload.is_none() {
                if let Some(current) = self.current_window.as_mut() {
                    current.upgrade_side_effect(SideEffect::Boost);
                }
            }
        } else if event_type == "-sidestart" {
            if let Some(current) = self.current_window.as_mut() {
                current.upgrade_side_effect(SideEffect::HazardSet);
            }
        } else if event_type == "-sideend" {
            if let Some(current) = self.current_window.as_mut() {
                current.upgrade_side_effect(SideEffect::HazardClear);
            }
        } else if event_type == "-weather" && parts.len() >= 3 {
            let identifier = normalize_identifier(parts[2]);
            let is_upkeep = raw_line.contains("[upkeep]");
            if !identifier.is_empty() && identifier != "none" && !is_upkeep {
                let current_side = self.current_window.as_ref().map(|window| window.side);
                if let Some(current) = self.current_window.as_mut() {
                    current.upgrade_side_effect(SideEffect::WeatherSet);
                }
                let from_ability = from_payload
                    .as_deref()
                    .map_or(false, |payload| payload.to_lowercase().starts_with("ability:"));
                let setter = of_tag_slot(raw_line).or(current_side);
                if let Some(setter) = setter {
                    let key = (setter, identifier);
                    let entry = self.weather_reveals.entry(key).or_insert(false);
                    *entry = *entry || from_ability;
                }
            }
        } else if event_type == "-prepare" {
            if let Some(current) = self.current_window.as_mut() {
                current.upgrade_side_effect(SideEffect::Charging);
            }
            let prepare_side = parts.get(2).and_then(|part| slot_from_ident(part));
            if let Some(prepare_side) = prepare_side {
                if parts.len() >= 4 {
                    self.pending_charge[prepare_side as usize] =
                        Some(normalize_identifier(parts[3]));
                }
            }
        } else if event_type == "-crit" {
            if let Some(current) = self.current_window.as_mut() {
                if target == current.defender_side {
                    current.crit = true;
                }
            }
        } else if event_type == "-miss" {
            let miss_side = parts.get(2).and_then(|part| slot_from_ident(part));
            if let Some(current) = self.current_window.as_mut() {
                if miss_side == Some(current.side) {
                    current.miss = true;
                }
            }
        } else if event_type == "-supereffective" {
            if let Some(current) = self.current_window.as_mut() {
                if target == current.defender_side {
                    current.effectiveness = Effectiveness::Super;
                }
            }
        } else if event_type == "-resisted" {
            if let Some(current) = self.current_window.as_mut() {
                if target == current.defender_side {
                    current.effectiveness = Effectiveness::Resisted;
                }
            }
        } else if event_type == "-immune" {
            let absorb = is_absorb_signature(from_payload.as_deref());
            if let Some(current) = self.current_window.as_mut() {
                if target == current.defender_side {
                    current.effectiveness = Effectiveness::Immune;
                    if absorb {
                        current.upgrade_outcome(Outcome::Absorbed);
                    } else {
                        current.upgrade_outcome(Outcome::Immune);
                    }
                }
            }
        } else if event_type == "-hitcount" && parts.len() >= 4 {
            let parsed = parts[3].trim().parse::<i64>().ok();
            if let Some(current) = self.current_window.as_mut() {
                if let Some(value) = parsed {
                    current.n_hits = value.max(1);
                }
            }
        } else if event_type == "-activate" && parts.len() >= 4 {
            let identifier = side_condition_identifier(parts[3]);
            if let Some(current) = self.current_window.as_mut() {
                if target == current.defender_side {
                    if identifier == "protect" || identifier == "detect" {
                        current.upgrade_outcome(Outcome::Blocked);
                    } else if identifier == "substitute" {
                        current.upgrade_outcome(Outcome::HitSub);
                    } else if identifier == "endure" {
                        current.upgrade_outcome(Outcome::Endured);
                    }
                }
            }
        } else if event_type == "-end" && parts.len() >= 4 {
            let is_substitute = side_condition_identifier(parts[3]) == "substitute";
            if let Some(current) = self.current_window.as_mut() {
                if target == current.defender_side && is_substitute {
                    current.upgrade_outcome(Outcome::BrokeSub);
                }
            }
        }

        if (event_type == "-heal" || event_type == "-start")
            && (is_absorb_signature(from_payload.as_deref())
                || is_absorb_start(event_type, parts))
        {
            if let Some(current) = self.current_window.as_mut() {
                if target == current.defender_side {
                    current.upgrade_outcome(Outcome::Absorbed);
                }
            }
        }

        update_side_conditions(parts, &mut self.side_condition_counts);
        update_weather(parts, &mut self.weather);
        Ok(())
    }

    // ------------------------------------------------------------------ tendencies

    /// One (token, window) pair of `_tendency_stats_from_fold`'s loop, at close.
    fn accumulate_tendency(&mut self, window: &Window, token: &TransitionToken) {
        let opponent = self.opponent_slot();
        let voluntary_switch = token.kind == Kind::Switch && window.voluntary_switch;
        let is_decision = (token.kind == Kind::Move
            && !token.called
            && !window.locked_continuation)
            || voluntary_switch
            || (token.kind == Kind::Cant && !is_cant_no_choice_reason(&token.action));
        if is_decision
            && token.actor_slot == opponent
            && self.last_opponent_opportunity_turn != Some(token.turn)
        {
            self.opponent_decision_opportunities += 1;
            self.last_opponent_opportunity_turn = Some(token.turn);
        }
        if token.actor_slot == opponent {
            if voluntary_switch {
                self.opponent_switch_count += 1;
            }
            if token.kind == Kind::Move && token.pursuit_intercept {
                self.pursuit_intercept_predict_count += 1;
            }
        } else {
            if voluntary_switch {
                self.my_switch_turn_count += 1;
            }
            if token.kind == Kind::Move && token.damage_outcome == Outcome::Blocked {
                self.blocked_on_our_attack_count += 1;
            }
        }
    }

    // ------------------------------------------------------------------ merge staging

    fn flush_pending(&mut self) -> PyResult<()> {
        if self.pending_windows.is_empty() {
            return Ok(());
        }
        let windows = std::mem::take(&mut self.pending_windows);
        let window_refs: Vec<&Window> = windows.iter().collect();
        let (merged, lead_done) = merge_window_run(
            &window_refs,
            self.lead_done,
            &self.turn_start_occupants,
            &self.completed_turns,
            &self.fainted_turns,
        );
        self.lead_done = lead_done;
        for token in merged {
            let mut first_rep: Option<i64> = None;
            let mut second_rep: Option<i64> = None;
            for (position, sub) in [(RepPos::First, &token.first), (RepPos::Second, &token.second)]
            {
                if sub.status != Status::Action {
                    continue;
                }
                let rep = self.expansion_cursor + representative_offset(sub);
                match position {
                    RepPos::First => first_rep = Some(rep),
                    RepPos::Second => second_rep = Some(rep),
                }
                self.rep_index_map.insert(rep, (self.merged_total, position));
                self.expansion_cursor += expansion_length(sub);
            }
            self.merged_done.push_back((token, first_rep, second_rep));
            self.merged_total += 1;
        }
        while self.merged_done.len() > self.merged_tail_limit {
            self.merged_done.pop_front();
        }
        // The flatten bijection: the flushed merged tokens' expansions cover exactly
        // the flushed windows' per-action tokens.
        if self.expansion_cursor != self.action_total {
            return Err(PyAssertionError::new_err(format!(
                "fold-state invariant violated: merged flatten coverage ({}) != emitted \
                 per-action tokens ({}).",
                self.expansion_cursor, self.action_total
            )));
        }
        self.prune_rep_index_map();
        self.prune_annotations();
        Ok(())
    }

    fn prune_turn_maps(&mut self) {
        let keep_from = self.turn_number - 1;
        self.turn_start_occupants = self
            .turn_start_occupants
            .split_off(&keep_from);
        self.completed_turns = self.completed_turns.split_off(&keep_from);
        self.fainted_turns = self.fainted_turns.split_off(&keep_from);
    }

    fn prune_rep_index_map(&mut self) {
        let oldest_seq = self.merged_total - self.merged_done.len() as i64;
        self.rep_index_map.retain(|_, (seq, _)| *seq >= oldest_seq);
    }

    fn prune_annotations(&mut self) {
        let tail_start = self.action_total - self.action_tail.len() as i64;
        let rep_index_map = &self.rep_index_map;
        self.annotations
            .retain(|index, _| *index >= tail_start || rep_index_map.contains_key(index));
    }

    // ------------------------------------------------------------------ annotations

    /// `FoldState.apply_annotations_in_place` (overlay entries pre-sorted by index).
    fn apply_annotations_in_place(
        &mut self,
        overlay: &[(i64, AnnotationValues)],
    ) -> PyResult<()> {
        for (index, values) in overlay {
            let index = *index;
            let values = values.clone();
            if let Some(existing) = self.annotations.get(&index) {
                if *existing != values {
                    return Err(PyValueError::new_err(format!(
                        "annotation for token index {index} changed after application \
                         ({existing:?} -> {values:?}); tracker conclusions are per-index \
                         immutable."
                    )));
                }
                continue;
            }
            let token = self.token_identity(index)?;
            self.annotations.insert(index, values.clone());
            let (_residual, _residual_valid, cb_bit, investment) = values;
            if cb_bit && token.kind == Kind::Move && token.actor_slot == self.opponent_slot() {
                self.cb_pinned
                    .insert(normalize_identifier(&token.actor_species));
            }
            let has_defender = token
                .defender_species
                .as_deref()
                .map_or(false, |species| !species.is_empty());
            if investment != 0.0
                && token.kind == Kind::Move
                && token.actor_slot == self.perspective_slot
                && has_defender
            {
                let key = normalize_identifier(token.defender_species.as_deref().unwrap());
                let previous = self.investment_pinned_state.get(&key).copied();
                let should_write = match previous {
                    None => true,
                    Some((previous_index, _)) => index >= previous_index,
                };
                if should_write {
                    // Same clamp as the production pinned reduction.
                    let clamped = investment.min(1.0).max(-1.0);
                    self.investment_pinned_state.insert(key, (index, clamped));
                }
            }
        }
        self.prune_annotations();
        Ok(())
    }

    /// `FoldState._token_identity` (owned copy of the identity fields).
    fn token_identity(&self, index: i64) -> PyResult<TransitionToken> {
        let tail_start = self.action_total - self.action_tail.len() as i64;
        if tail_start <= index && index < self.action_total {
            return Ok(self.action_tail[(index - tail_start) as usize].clone());
        }
        if index == self.action_total {
            if let Some(window) = &self.current_window {
                return Ok(token_from_window(window));
            }
        }
        Err(PyValueError::new_err(format!(
            "annotation index {index} is outside the identifiable range \
             [{tail_start}, {}] — apply annotations per boundary (or raise \
             action_tail_limit).",
            self.action_total
        )))
    }

    fn annotated_token(&self, index: i64, token: &TransitionToken) -> TransitionToken {
        match self.annotations.get(&index) {
            None => token.clone(),
            Some((residual, residual_valid, cb_bit, investment)) => {
                let mut annotated = token.clone();
                annotated.residual = *residual;
                annotated.residual_valid = *residual_valid;
                annotated.cb_bit = *cb_bit;
                annotated.investment = *investment;
                annotated
            }
        }
    }

    fn annotated_merged(
        &self,
        token: &MergedToken,
        first_rep: Option<i64>,
        second_rep: Option<i64>,
    ) -> MergedToken {
        let mut out = token.clone();
        for (sub, rep) in [(&mut out.first, first_rep), (&mut out.second, second_rep)] {
            let rep = match rep {
                Some(rep) => rep,
                None => continue,
            };
            let (residual, residual_valid, cb_bit, investment) =
                match self.annotations.get(&rep) {
                    Some(values) => values.clone(),
                    None => continue,
                };
            if residual == sub.residual
                && residual_valid == sub.residual_valid
                && cb_bit == sub.cb_bit
                && investment == sub.investment
            {
                continue;
            }
            sub.residual = residual;
            sub.residual_valid = residual_valid;
            sub.cb_bit = cb_bit;
            sub.investment = investment;
        }
        out
    }

    /// `FoldState._annotate_with_cursor` (virtual merged tokens).
    fn annotate_with_cursor(&self, token: &MergedToken, cursor: i64) -> (MergedToken, i64) {
        let mut cursor = cursor;
        let mut first_rep: Option<i64> = None;
        let mut second_rep: Option<i64> = None;
        for (position, sub) in [(RepPos::First, &token.first), (RepPos::Second, &token.second)] {
            if sub.status != Status::Action {
                continue;
            }
            let rep = cursor + representative_offset(sub);
            match position {
                RepPos::First => first_rep = Some(rep),
                RepPos::Second => second_rep = Some(rep),
            }
            cursor += expansion_length(sub);
        }
        (self.annotated_merged(token, first_rep, second_rep), cursor)
    }

    // ------------------------------------------------------------------ products

    fn products(&self) -> ProductsData {
        let mut virtual_windows: Vec<&Window> = self.pending_windows.iter().collect();
        let mut virtual_token: Option<TransitionToken> = None;
        if let Some(window) = &self.current_window {
            virtual_windows.push(window);
            virtual_token = Some(token_from_window(window));
        }

        // Per-action tail (annotated).
        let tail_start = self.action_total - self.action_tail.len() as i64;
        let mut action_tokens: Vec<TransitionToken> = self
            .action_tail
            .iter()
            .enumerate()
            .map(|(offset, token)| self.annotated_token(tail_start + offset as i64, token))
            .collect();
        let mut total_actions = self.action_total;
        if let Some(token) = &virtual_token {
            action_tokens.push(self.annotated_token(self.action_total, token));
            total_actions += 1;
        }
        if action_tokens.len() > self.action_tail_limit {
            let cut = action_tokens.len() - self.action_tail_limit;
            action_tokens.drain(..cut);
        }

        // Merged stream: finalized tail + virtual merge of the open run.
        let mut merged_tokens: Vec<MergedToken> = self
            .merged_done
            .iter()
            .map(|(token, first_rep, second_rep)| {
                self.annotated_merged(token, *first_rep, *second_rep)
            })
            .collect();
        let (virtual_merged, _) = merge_window_run(
            &virtual_windows,
            self.lead_done,
            &self.turn_start_occupants,
            &self.completed_turns,
            &self.fainted_turns,
        );
        let virtual_count = virtual_merged.len() as i64;
        let mut cursor = self.expansion_cursor;
        for token in &virtual_merged {
            let (annotated, next_cursor) = self.annotate_with_cursor(token, cursor);
            cursor = next_cursor;
            merged_tokens.push(annotated);
        }
        let merged_total = self.merged_total + virtual_count;
        if merged_tokens.len() > self.merged_tail_limit {
            let cut = merged_tokens.len() - self.merged_tail_limit;
            merged_tokens.drain(..cut);
        }

        ProductsData {
            transition_tokens: action_tokens,
            transition_token_total: total_actions,
            turn_merged_tokens: merged_tokens,
            turn_merged_total: merged_total,
            tendency_stats: self.tendency_stats(self.current_window.as_ref()),
            cb_pinned_species: self.cb_pinned.iter().cloned().collect(),
            investment_pinned: self
                .investment_pinned_state
                .iter()
                .map(|(species, (_, code))| (species.clone(), *code))
                .collect(),
        }
    }

    fn tendency_stats(&self, virtual_window: Option<&Window>) -> TendencyStatsData {
        let opponent = self.opponent_slot();
        let mut switches = self.opponent_switch_count;
        let mut opportunities = self.opponent_decision_opportunities;
        let mut blocked = self.blocked_on_our_attack_count;
        let mut pursuit = self.pursuit_intercept_predict_count;
        let mut my_switches = self.my_switch_turn_count;
        if let Some(window) = virtual_window {
            let token = token_from_window(window);
            let voluntary_switch = token.kind == Kind::Switch && window.voluntary_switch;
            let is_decision = (token.kind == Kind::Move
                && !token.called
                && !window.locked_continuation)
                || voluntary_switch
                || (token.kind == Kind::Cant && !is_cant_no_choice_reason(&token.action));
            if is_decision
                && token.actor_slot == opponent
                && self.last_opponent_opportunity_turn != Some(token.turn)
            {
                opportunities += 1;
            }
            if token.actor_slot == opponent {
                if voluntary_switch {
                    switches += 1;
                }
                if token.kind == Kind::Move && token.pursuit_intercept {
                    pursuit += 1;
                }
            } else {
                if voluntary_switch {
                    my_switches += 1;
                }
                if token.kind == Kind::Move && token.damage_outcome == Outcome::Blocked {
                    blocked += 1;
                }
            }
        }

        let mon_tendencies: Vec<MonTendencyData> = self
            .mon_counters
            .iter()
            .filter(|((side, _), _)| *side == opponent)
            .map(|((side, species), counters)| MonTendencyData {
                slot: *side,
                species: species.clone(),
                switched_out_before_attacking: counters.switched_out_before_attacking,
                stayed_and_attacked: counters.stayed_and_attacked,
                turns_active: counters.turns_active,
            })
            .collect();
        // The consumer reduction: {weather: OR(from_ability)} for the opponent side,
        // sorted by weather id (BTreeMap iteration order).
        let weather_reveals: Vec<(String, bool)> = self
            .weather_reveals
            .iter()
            .filter(|((side, _), _)| *side == opponent)
            .map(|((_, weather), from_ability)| (weather.clone(), *from_ability))
            .collect();

        TendencyStatsData {
            perspective_slot: self.perspective_slot,
            opponent_slot: opponent,
            opponent_switch_count: switches,
            opponent_decision_opportunities: opportunities,
            opponent_mon_tendencies: mon_tendencies,
            opponent_weather_reveals: weather_reveals,
            blocked_on_our_attack_count: blocked,
            pursuit_intercept_predict_count: pursuit,
            my_switch_turn_count: my_switches,
        }
    }
}

struct MonTendencyData {
    slot: u8,
    species: String,
    switched_out_before_attacking: i64,
    stayed_and_attacked: i64,
    turns_active: i64,
}

struct TendencyStatsData {
    perspective_slot: u8,
    opponent_slot: u8,
    opponent_switch_count: i64,
    opponent_decision_opportunities: i64,
    opponent_mon_tendencies: Vec<MonTendencyData>,
    opponent_weather_reveals: Vec<(String, bool)>,
    blocked_on_our_attack_count: i64,
    pursuit_intercept_predict_count: i64,
    my_switch_turn_count: i64,
}

struct ProductsData {
    transition_tokens: Vec<TransitionToken>,
    transition_token_total: i64,
    turn_merged_tokens: Vec<MergedToken>,
    turn_merged_total: i64,
    tendency_stats: TendencyStatsData,
    cb_pinned_species: Vec<String>,
    investment_pinned: Vec<(String, f64)>,
}

// ---------------------------------------------------------------------------
// Python payload conversion (native objects; the harness canonicalizes)
// ---------------------------------------------------------------------------

fn py_get<'py>(payload: &Bound<'py, PyAny>, key: &str) -> PyResult<Bound<'py, PyAny>> {
    payload
        .get_item(key)
        .map_err(|_| PyKeyError::new_err(format!("fold payload is missing key {key:?}")))
}

fn py_opt_str(value: &Bound<'_, PyAny>) -> PyResult<Option<String>> {
    if value.is_none() {
        Ok(None)
    } else {
        Ok(Some(value.extract::<String>()?))
    }
}

fn py_opt_f64(value: &Bound<'_, PyAny>) -> PyResult<Option<f64>> {
    if value.is_none() {
        Ok(None)
    } else {
        Ok(Some(value.extract::<f64>()?))
    }
}

fn py_opt_i64(value: &Bound<'_, PyAny>) -> PyResult<Option<i64>> {
    if value.is_none() {
        Ok(None)
    } else {
        Ok(Some(value.extract::<i64>()?))
    }
}

fn window_to_py<'py>(py: Python<'py>, window: &Window) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("event_index", window.event_index)?;
    out.set_item("turn", window.turn)?;
    out.set_item("side", side_str(window.side))?;
    out.set_item("species", &window.species)?;
    out.set_item("kind", window.kind.as_str())?;
    out.set_item("action", &window.action)?;
    out.set_item("defender_side", window.defender_side.map(side_str))?;
    out.set_item("defender_species", window.defender_species.as_deref())?;
    out.set_item("called", window.called)?;
    out.set_item("transformed", window.transformed)?;
    out.set_item("own_spikes_layers", window.own_spikes_layers)?;
    out.set_item("opp_spikes_layers", window.opp_spikes_layers)?;
    out.set_item("weather", window.weather.as_deref())?;
    out.set_item("damage_fraction", window.damage_fraction)?;
    out.set_item("self_hp_cost", window.self_hp_cost)?;
    out.set_item("outcome", window.outcome.as_str())?;
    out.set_item("crit", window.crit)?;
    out.set_item("miss", window.miss)?;
    out.set_item("ko", window.ko)?;
    out.set_item("pursuit_intercept", window.pursuit_intercept)?;
    out.set_item("n_hits", window.n_hits)?;
    out.set_item("effectiveness", window.effectiveness.as_str())?;
    out.set_item("side_effect", window.side_effect.as_str())?;
    out.set_item("defender_hit_by_move", window.defender_hit_by_move)?;
    out.set_item("voluntary_switch", window.voluntary_switch)?;
    out.set_item("locked_continuation", window.locked_continuation)?;
    out.set_item(
        "switch_reason",
        window.switch_reason.map(SwitchReason::as_str),
    )?;
    out.set_item(
        "other_side_pending_replacement",
        window.other_side_pending_replacement,
    )?;
    Ok(out)
}

fn window_from_py(payload: &Bound<'_, PyAny>) -> PyResult<Window> {
    let defender_side = match py_opt_str(&py_get(payload, "defender_side")?)? {
        Some(side) => Some(parse_side(&side)?),
        None => None,
    };
    let switch_reason = match py_opt_str(&py_get(payload, "switch_reason")?)? {
        Some(reason) => Some(SwitchReason::parse(&reason)?),
        None => None,
    };
    Ok(Window {
        event_index: py_get(payload, "event_index")?.extract::<i64>()?,
        turn: py_get(payload, "turn")?.extract::<i64>()?,
        side: parse_side(&py_get(payload, "side")?.extract::<String>()?)?,
        species: py_get(payload, "species")?.extract::<String>()?,
        kind: Kind::parse(&py_get(payload, "kind")?.extract::<String>()?)?,
        action: py_get(payload, "action")?.extract::<String>()?,
        defender_side,
        defender_species: py_opt_str(&py_get(payload, "defender_species")?)?,
        called: py_get(payload, "called")?.is_truthy()?,
        transformed: py_get(payload, "transformed")?.is_truthy()?,
        own_spikes_layers: py_get(payload, "own_spikes_layers")?.extract::<i64>()?,
        opp_spikes_layers: py_get(payload, "opp_spikes_layers")?.extract::<i64>()?,
        weather: py_opt_str(&py_get(payload, "weather")?)?,
        damage_fraction: py_get(payload, "damage_fraction")?.extract::<f64>()?,
        self_hp_cost: py_get(payload, "self_hp_cost")?.extract::<f64>()?,
        outcome: Outcome::parse(&py_get(payload, "outcome")?.extract::<String>()?)?,
        crit: py_get(payload, "crit")?.is_truthy()?,
        miss: py_get(payload, "miss")?.is_truthy()?,
        ko: py_get(payload, "ko")?.is_truthy()?,
        pursuit_intercept: py_get(payload, "pursuit_intercept")?.is_truthy()?,
        n_hits: py_get(payload, "n_hits")?.extract::<i64>()?,
        effectiveness: Effectiveness::parse(
            &py_get(payload, "effectiveness")?.extract::<String>()?,
        )?,
        side_effect: SideEffect::parse(&py_get(payload, "side_effect")?.extract::<String>()?)?,
        defender_hit_by_move: py_get(payload, "defender_hit_by_move")?.is_truthy()?,
        voluntary_switch: py_get(payload, "voluntary_switch")?.is_truthy()?,
        locked_continuation: py_get(payload, "locked_continuation")?.is_truthy()?,
        switch_reason,
        other_side_pending_replacement: py_get(payload, "other_side_pending_replacement")?
            .is_truthy()?,
    })
}

fn transition_token_to_py<'py>(
    py: Python<'py>,
    token: &TransitionToken,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("turn", token.turn)?;
    out.set_item("actor_slot", side_str(token.actor_slot))?;
    out.set_item("actor_species", &token.actor_species)?;
    out.set_item("kind", token.kind.as_str())?;
    out.set_item("action", &token.action)?;
    out.set_item("called", token.called)?;
    out.set_item("transformed", token.transformed)?;
    out.set_item("damage_fraction", token.damage_fraction)?;
    out.set_item("damage_outcome", token.damage_outcome.as_str())?;
    out.set_item("crit", token.crit)?;
    out.set_item("miss", token.miss)?;
    out.set_item("ko", token.ko)?;
    out.set_item("pursuit_intercept", token.pursuit_intercept)?;
    out.set_item("n_hits", token.n_hits)?;
    out.set_item("effectiveness", token.effectiveness.as_str())?;
    out.set_item("side_effect", token.side_effect.as_str())?;
    out.set_item("self_hp_cost", token.self_hp_cost)?;
    out.set_item("own_spikes_layers", token.own_spikes_layers)?;
    out.set_item("opp_spikes_layers", token.opp_spikes_layers)?;
    out.set_item("weather", token.weather.as_deref())?;
    out.set_item("defender_species", token.defender_species.as_deref())?;
    out.set_item("residual", token.residual)?;
    out.set_item("residual_valid", token.residual_valid)?;
    out.set_item("cb_bit", token.cb_bit)?;
    out.set_item("investment", token.investment)?;
    Ok(out)
}

fn transition_token_from_py(payload: &Bound<'_, PyAny>) -> PyResult<TransitionToken> {
    Ok(TransitionToken {
        turn: py_get(payload, "turn")?.extract::<i64>()?,
        actor_slot: parse_side(&py_get(payload, "actor_slot")?.extract::<String>()?)?,
        actor_species: py_get(payload, "actor_species")?.extract::<String>()?,
        kind: Kind::parse(&py_get(payload, "kind")?.extract::<String>()?)?,
        action: py_get(payload, "action")?.extract::<String>()?,
        called: py_get(payload, "called")?.is_truthy()?,
        transformed: py_get(payload, "transformed")?.is_truthy()?,
        damage_fraction: py_get(payload, "damage_fraction")?.extract::<f64>()?,
        damage_outcome: Outcome::parse(&py_get(payload, "damage_outcome")?.extract::<String>()?)?,
        crit: py_get(payload, "crit")?.is_truthy()?,
        miss: py_get(payload, "miss")?.is_truthy()?,
        ko: py_get(payload, "ko")?.is_truthy()?,
        pursuit_intercept: py_get(payload, "pursuit_intercept")?.is_truthy()?,
        n_hits: py_get(payload, "n_hits")?.extract::<i64>()?,
        effectiveness: Effectiveness::parse(
            &py_get(payload, "effectiveness")?.extract::<String>()?,
        )?,
        side_effect: SideEffect::parse(&py_get(payload, "side_effect")?.extract::<String>()?)?,
        self_hp_cost: py_get(payload, "self_hp_cost")?.extract::<f64>()?,
        own_spikes_layers: py_get(payload, "own_spikes_layers")?.extract::<i64>()?,
        opp_spikes_layers: py_get(payload, "opp_spikes_layers")?.extract::<i64>()?,
        weather: py_opt_str(&py_get(payload, "weather")?)?,
        defender_species: py_opt_str(&py_get(payload, "defender_species")?)?,
        residual: py_opt_f64(&py_get(payload, "residual")?)?,
        residual_valid: py_get(payload, "residual_valid")?.is_truthy()?,
        cb_bit: py_get(payload, "cb_bit")?.is_truthy()?,
        investment: py_get(payload, "investment")?.extract::<f64>()?,
    })
}

fn sub_block_to_py<'py>(py: Python<'py>, sub: &SubBlock) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("status", sub.status.as_str())?;
    out.set_item("actor_slot", side_str(sub.actor_slot))?;
    out.set_item("actor_species", &sub.actor_species)?;
    out.set_item("kind", sub.kind.map(Kind::as_str).unwrap_or(""))?;
    out.set_item("action", &sub.action)?;
    out.set_item("called", sub.called)?;
    out.set_item("transformed", sub.transformed)?;
    out.set_item("damage_fraction", sub.damage_fraction)?;
    out.set_item("self_hp_cost", sub.self_hp_cost)?;
    out.set_item("damage_outcome", sub.damage_outcome.as_str())?;
    out.set_item("crit", sub.crit)?;
    out.set_item("miss", sub.miss)?;
    out.set_item("ko", sub.ko)?;
    out.set_item("pursuit_intercept", sub.pursuit_intercept)?;
    out.set_item("n_hits", sub.n_hits)?;
    out.set_item("effectiveness", sub.effectiveness.as_str())?;
    out.set_item("side_effect", sub.side_effect.as_str())?;
    out.set_item("defender_species", sub.defender_species.as_deref())?;
    out.set_item("cant_reason", sub.cant_reason.as_deref())?;
    out.set_item("baton_pass_species", sub.baton_pass_species.as_deref())?;
    out.set_item("residual", sub.residual)?;
    out.set_item("residual_valid", sub.residual_valid)?;
    out.set_item("cb_bit", sub.cb_bit)?;
    out.set_item("investment", sub.investment)?;
    Ok(out)
}

fn sub_block_from_py(payload: &Bound<'_, PyAny>) -> PyResult<SubBlock> {
    Ok(SubBlock {
        status: Status::parse(&py_get(payload, "status")?.extract::<String>()?)?,
        actor_slot: parse_side(&py_get(payload, "actor_slot")?.extract::<String>()?)?,
        actor_species: py_get(payload, "actor_species")?.extract::<String>()?,
        kind: Kind::parse_opt(&py_get(payload, "kind")?.extract::<String>()?)?,
        action: py_get(payload, "action")?.extract::<String>()?,
        called: py_get(payload, "called")?.is_truthy()?,
        transformed: py_get(payload, "transformed")?.is_truthy()?,
        damage_fraction: py_get(payload, "damage_fraction")?.extract::<f64>()?,
        self_hp_cost: py_get(payload, "self_hp_cost")?.extract::<f64>()?,
        damage_outcome: Outcome::parse(&py_get(payload, "damage_outcome")?.extract::<String>()?)?,
        crit: py_get(payload, "crit")?.is_truthy()?,
        miss: py_get(payload, "miss")?.is_truthy()?,
        ko: py_get(payload, "ko")?.is_truthy()?,
        pursuit_intercept: py_get(payload, "pursuit_intercept")?.is_truthy()?,
        n_hits: py_get(payload, "n_hits")?.extract::<i64>()?,
        effectiveness: Effectiveness::parse(
            &py_get(payload, "effectiveness")?.extract::<String>()?,
        )?,
        side_effect: SideEffect::parse(&py_get(payload, "side_effect")?.extract::<String>()?)?,
        defender_species: py_opt_str(&py_get(payload, "defender_species")?)?,
        cant_reason: py_opt_str(&py_get(payload, "cant_reason")?)?,
        baton_pass_species: py_opt_str(&py_get(payload, "baton_pass_species")?)?,
        residual: py_opt_f64(&py_get(payload, "residual")?)?,
        residual_valid: py_get(payload, "residual_valid")?.is_truthy()?,
        cb_bit: py_get(payload, "cb_bit")?.is_truthy()?,
        investment: py_get(payload, "investment")?.extract::<f64>()?,
    })
}

fn merged_token_to_py<'py>(py: Python<'py>, token: &MergedToken) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("turn", token.turn)?;
    out.set_item("phase", token.phase.as_str())?;
    out.set_item("first", sub_block_to_py(py, &token.first)?)?;
    out.set_item("second", sub_block_to_py(py, &token.second)?)?;
    out.set_item("own_spikes_layers", token.own_spikes_layers)?;
    out.set_item("opp_spikes_layers", token.opp_spikes_layers)?;
    out.set_item("weather", token.weather.as_deref())?;
    Ok(out)
}

fn merged_token_from_py(payload: &Bound<'_, PyAny>) -> PyResult<MergedToken> {
    Ok(MergedToken {
        turn: py_get(payload, "turn")?.extract::<i64>()?,
        phase: Phase::parse(&py_get(payload, "phase")?.extract::<String>()?)?,
        first: sub_block_from_py(&py_get(payload, "first")?)?,
        second: sub_block_from_py(&py_get(payload, "second")?)?,
        own_spikes_layers: py_get(payload, "own_spikes_layers")?.extract::<i64>()?,
        opp_spikes_layers: py_get(payload, "opp_spikes_layers")?.extract::<i64>()?,
        weather: py_opt_str(&py_get(payload, "weather")?)?,
    })
}

impl FoldStateInner {
    /// `FoldState.to_payload` — the schema-v1 fold-state export, as native
    /// Python objects (key insertion order is irrelevant: the harness's
    /// canonical JSON sorts keys).
    fn to_payload<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let out = PyDict::new(py);
        out.set_item("schema", FOLD_STATE_SCHEMA)?;
        out.set_item("perspective_slot", side_str(self.perspective_slot))?;
        out.set_item("merged_tail_limit", self.merged_tail_limit as i64)?;
        out.set_item("action_tail_limit", self.action_tail_limit as i64)?;
        out.set_item("event_index", self.event_index)?;

        let side_conditions = PyDict::new(py);
        for side in [0u8, 1u8] {
            let counts = PyDict::new(py);
            for (condition, count) in &self.side_condition_counts[side as usize] {
                counts.set_item(condition, *count)?;
            }
            side_conditions.set_item(side_str(side), counts)?;
        }
        out.set_item("side_condition_counts", side_conditions)?;

        out.set_item("weather", self.weather.as_deref())?;
        out.set_item("turn_number", self.turn_number)?;

        let hp = PyDict::new(py);
        for side in [0u8, 1u8] {
            if let Some(fraction) = self.hp_fraction[side as usize] {
                hp.set_item(side_str(side), fraction)?;
            }
        }
        out.set_item("hp_fraction", hp)?;

        let occupant = PyDict::new(py);
        for side in [0u8, 1u8] {
            if let Some(stay) = &self.occupant[side as usize] {
                let entry = PyDict::new(py);
                entry.set_item("species", &stay.species)?;
                entry.set_item("moved", stay.moved)?;
                occupant.set_item(side_str(side), entry)?;
            }
        }
        out.set_item("occupant", occupant)?;

        let bool_pair = |values: &[bool; 2]| -> PyResult<Bound<'py, PyDict>> {
            let pair = PyDict::new(py);
            pair.set_item("p1", values[0])?;
            pair.set_item("p2", values[1])?;
            Ok(pair)
        };
        out.set_item("transformed", bool_pair(&self.transformed)?)?;
        out.set_item("pending_baton_pass", bool_pair(&self.pending_baton_pass)?)?;
        out.set_item(
            "pending_faint_replacement",
            bool_pair(&self.pending_faint_replacement)?,
        )?;
        out.set_item("lead_seen", bool_pair(&self.lead_seen)?)?;

        let charge = PyDict::new(py);
        charge.set_item("p1", self.pending_charge[0].as_deref())?;
        charge.set_item("p2", self.pending_charge[1].as_deref())?;
        out.set_item("pending_charge", charge)?;

        out.set_item(
            "current_window",
            match &self.current_window {
                Some(window) => window_to_py(py, window)?.into_any(),
                None => py.None().into_bound(py),
            },
        )?;
        out.set_item("pursuit_buffer", PyList::new(py, &self.pursuit_buffer)?)?;
        out.set_item("lead_done", self.lead_done)?;

        let pending = PyList::empty(py);
        for window in &self.pending_windows {
            pending.append(window_to_py(py, window)?)?;
        }
        out.set_item("pending_windows", pending)?;

        let merged_done = PyList::empty(py);
        for (token, first_rep, second_rep) in &self.merged_done {
            let entry = PyDict::new(py);
            entry.set_item("token", merged_token_to_py(py, token)?)?;
            entry.set_item("first_rep", *first_rep)?;
            entry.set_item("second_rep", *second_rep)?;
            merged_done.append(entry)?;
        }
        out.set_item("merged_done", merged_done)?;
        out.set_item("merged_total", self.merged_total)?;
        out.set_item("expansion_cursor", self.expansion_cursor)?;

        let action_tail = PyList::empty(py);
        for token in &self.action_tail {
            action_tail.append(transition_token_to_py(py, token)?)?;
        }
        out.set_item("action_tail", action_tail)?;
        out.set_item("action_total", self.action_total)?;

        out.set_item("opponent_switch_count", self.opponent_switch_count)?;
        out.set_item(
            "opponent_decision_opportunities",
            self.opponent_decision_opportunities,
        )?;
        out.set_item(
            "last_opponent_opportunity_turn",
            self.last_opponent_opportunity_turn,
        )?;
        out.set_item("blocked_on_our_attack_count", self.blocked_on_our_attack_count)?;
        out.set_item(
            "pursuit_intercept_predict_count",
            self.pursuit_intercept_predict_count,
        )?;
        out.set_item("my_switch_turn_count", self.my_switch_turn_count)?;

        let mon_counters = PyDict::new(py);
        for ((side, species), counters) in &self.mon_counters {
            let values = PyList::new(
                py,
                [
                    counters.switched_out_before_attacking,
                    counters.stayed_and_attacked,
                    counters.turns_active,
                ],
            )?;
            mon_counters.set_item(format!("{}|{}", side_str(*side), species), values)?;
        }
        out.set_item("mon_counters", mon_counters)?;

        let weather_reveals = PyDict::new(py);
        for ((side, weather), from_ability) in &self.weather_reveals {
            weather_reveals.set_item(format!("{}|{}", side_str(*side), weather), *from_ability)?;
        }
        out.set_item("weather_reveals", weather_reveals)?;

        let occupants_map = PyDict::new(py);
        for (turn, occupants) in &self.turn_start_occupants {
            let entry = PyDict::new(py);
            for side in [0u8, 1u8] {
                if let Some(species) = &occupants[side as usize] {
                    entry.set_item(side_str(side), species)?;
                }
            }
            occupants_map.set_item(turn.to_string(), entry)?;
        }
        out.set_item("turn_start_occupants", occupants_map)?;

        out.set_item(
            "completed_turns",
            PyList::new(py, self.completed_turns.iter().copied())?,
        )?;
        out.set_item(
            "fainted_turns",
            PyList::new(py, self.fainted_turns.iter().copied())?,
        )?;

        let annotations = PyDict::new(py);
        for (index, (residual, residual_valid, cb_bit, investment)) in &self.annotations {
            let values = PyList::empty(py);
            values.append(*residual)?;
            values.append(*residual_valid)?;
            values.append(*cb_bit)?;
            values.append(*investment)?;
            annotations.set_item(index.to_string(), values)?;
        }
        out.set_item("annotations", annotations)?;

        let rep_index_map = PyDict::new(py);
        for (index, (seq, position)) in &self.rep_index_map {
            let entry = PyList::empty(py);
            entry.append(*seq)?;
            entry.append(position.as_str())?;
            rep_index_map.set_item(index.to_string(), entry)?;
        }
        out.set_item("rep_index_map", rep_index_map)?;

        out.set_item("cb_pinned", PyList::new(py, self.cb_pinned.iter())?)?;

        let investment_pinned = PyDict::new(py);
        for (species, (index, code)) in &self.investment_pinned_state {
            let entry = PyList::empty(py);
            entry.append(*index)?;
            entry.append(*code)?;
            investment_pinned.set_item(species, entry)?;
        }
        out.set_item("investment_pinned", investment_pinned)?;
        Ok(out)
    }

    /// `FoldState.from_payload`.
    fn from_payload(payload: &Bound<'_, PyAny>) -> PyResult<FoldStateInner> {
        let schema = py_get(payload, "schema")?;
        if !schema
            .extract::<String>()
            .map(|s| s == FOLD_STATE_SCHEMA)
            .unwrap_or(false)
        {
            return Err(PyValueError::new_err(format!(
                "unsupported fold-state payload schema: {:?}.",
                schema.to_string()
            )));
        }
        let perspective = parse_side(&py_get(payload, "perspective_slot")?.extract::<String>()?)?;
        let mut state = FoldStateInner::initial(
            perspective,
            py_get(payload, "merged_tail_limit")?.extract::<i64>()? as usize,
            py_get(payload, "action_tail_limit")?.extract::<i64>()? as usize,
        );
        state.event_index = py_get(payload, "event_index")?.extract::<i64>()?;

        let side_conditions = py_get(payload, "side_condition_counts")?;
        let side_conditions = side_conditions.downcast::<PyDict>()?;
        for (side_key, counts) in side_conditions.iter() {
            let side = parse_side(&side_key.extract::<String>()?)?;
            let counts = counts.downcast::<PyDict>()?;
            for (name, count) in counts.iter() {
                state.side_condition_counts[side as usize]
                    .insert(name.extract::<String>()?, count.extract::<i64>()?);
            }
        }

        state.weather = py_opt_str(&py_get(payload, "weather")?)?;
        state.turn_number = py_get(payload, "turn_number")?.extract::<i64>()?;

        let hp = py_get(payload, "hp_fraction")?;
        for (side_key, value) in hp.downcast::<PyDict>()?.iter() {
            let side = parse_side(&side_key.extract::<String>()?)?;
            state.hp_fraction[side as usize] = Some(value.extract::<f64>()?);
        }

        let occupant = py_get(payload, "occupant")?;
        for (side_key, entry) in occupant.downcast::<PyDict>()?.iter() {
            let side = parse_side(&side_key.extract::<String>()?)?;
            state.occupant[side as usize] = Some(StayRecord {
                species: py_get(&entry, "species")?.extract::<String>()?,
                moved: py_get(&entry, "moved")?.is_truthy()?,
            });
        }

        let load_bool_pair = |key: &str| -> PyResult<[bool; 2]> {
            let pair = py_get(payload, key)?;
            let mut values = [false, false];
            for (side_key, value) in pair.downcast::<PyDict>()?.iter() {
                let side = parse_side(&side_key.extract::<String>()?)?;
                values[side as usize] = value.is_truthy()?;
            }
            Ok(values)
        };
        state.transformed = load_bool_pair("transformed")?;
        state.pending_baton_pass = load_bool_pair("pending_baton_pass")?;
        state.pending_faint_replacement = load_bool_pair("pending_faint_replacement")?;
        state.lead_seen = load_bool_pair("lead_seen")?;

        let charge = py_get(payload, "pending_charge")?;
        for (side_key, value) in charge.downcast::<PyDict>()?.iter() {
            let side = parse_side(&side_key.extract::<String>()?)?;
            state.pending_charge[side as usize] = py_opt_str(&value)?;
        }

        let current_window = py_get(payload, "current_window")?;
        state.current_window = if current_window.is_none() {
            None
        } else {
            Some(window_from_py(&current_window)?)
        };

        state.pursuit_buffer = py_get(payload, "pursuit_buffer")?.extract::<Vec<String>>()?;
        state.lead_done = py_get(payload, "lead_done")?.is_truthy()?;

        for entry in py_get(payload, "pending_windows")?.try_iter()? {
            state.pending_windows.push(window_from_py(&entry?)?);
        }

        for entry in py_get(payload, "merged_done")?.try_iter()? {
            let entry = entry?;
            state.merged_done.push_back((
                merged_token_from_py(&py_get(&entry, "token")?)?,
                py_opt_i64(&py_get(&entry, "first_rep")?)?,
                py_opt_i64(&py_get(&entry, "second_rep")?)?,
            ));
        }
        state.merged_total = py_get(payload, "merged_total")?.extract::<i64>()?;
        state.expansion_cursor = py_get(payload, "expansion_cursor")?.extract::<i64>()?;

        for entry in py_get(payload, "action_tail")?.try_iter()? {
            state.action_tail.push_back(transition_token_from_py(&entry?)?);
        }
        state.action_total = py_get(payload, "action_total")?.extract::<i64>()?;

        state.opponent_switch_count =
            py_get(payload, "opponent_switch_count")?.extract::<i64>()?;
        state.opponent_decision_opportunities =
            py_get(payload, "opponent_decision_opportunities")?.extract::<i64>()?;
        state.last_opponent_opportunity_turn =
            py_opt_i64(&py_get(payload, "last_opponent_opportunity_turn")?)?;
        state.blocked_on_our_attack_count =
            py_get(payload, "blocked_on_our_attack_count")?.extract::<i64>()?;
        state.pursuit_intercept_predict_count =
            py_get(payload, "pursuit_intercept_predict_count")?.extract::<i64>()?;
        state.my_switch_turn_count = py_get(payload, "my_switch_turn_count")?.extract::<i64>()?;

        let mon_counters = py_get(payload, "mon_counters")?;
        for (key, values) in mon_counters.downcast::<PyDict>()?.iter() {
            let key = key.extract::<String>()?;
            let (side, species) = key.split_once('|').ok_or_else(|| {
                PyValueError::new_err(format!("malformed mon_counters key {key:?}"))
            })?;
            let values = values.extract::<Vec<i64>>()?;
            if values.len() != 3 {
                return Err(PyValueError::new_err("mon_counters entry must have 3 ints"));
            }
            state.mon_counters.insert(
                (parse_side(side)?, species.to_string()),
                MonCounters {
                    switched_out_before_attacking: values[0],
                    stayed_and_attacked: values[1],
                    turns_active: values[2],
                },
            );
        }

        let weather_reveals = py_get(payload, "weather_reveals")?;
        for (key, from_ability) in weather_reveals.downcast::<PyDict>()?.iter() {
            let key = key.extract::<String>()?;
            let (side, weather) = key.split_once('|').ok_or_else(|| {
                PyValueError::new_err(format!("malformed weather_reveals key {key:?}"))
            })?;
            state
                .weather_reveals
                .insert((parse_side(side)?, weather.to_string()), from_ability.is_truthy()?);
        }

        let occupants_map = py_get(payload, "turn_start_occupants")?;
        for (turn_key, occupants) in occupants_map.downcast::<PyDict>()?.iter() {
            let turn = turn_key.extract::<String>()?.trim().parse::<i64>().map_err(|_| {
                PyValueError::new_err("malformed turn_start_occupants turn key")
            })?;
            let mut entry: [Option<String>; 2] = [None, None];
            for (side_key, species) in occupants.downcast::<PyDict>()?.iter() {
                let side = parse_side(&side_key.extract::<String>()?)?;
                entry[side as usize] = Some(species.extract::<String>()?);
            }
            state.turn_start_occupants.insert(turn, entry);
        }

        for turn in py_get(payload, "completed_turns")?.extract::<Vec<i64>>()? {
            state.completed_turns.insert(turn);
        }
        for turn in py_get(payload, "fainted_turns")?.extract::<Vec<i64>>()? {
            state.fainted_turns.insert(turn);
        }

        let annotations = py_get(payload, "annotations")?;
        for (index_key, values) in annotations.downcast::<PyDict>()?.iter() {
            let index = annotation_index_from_key(&index_key)?;
            state
                .annotations
                .insert(index, annotation_values_from_py(&values)?);
        }

        let rep_index_map = py_get(payload, "rep_index_map")?;
        for (index_key, entry) in rep_index_map.downcast::<PyDict>()?.iter() {
            let index = annotation_index_from_key(&index_key)?;
            let seq = entry.get_item(0)?.extract::<i64>()?;
            let position = RepPos::parse(&entry.get_item(1)?.extract::<String>()?)?;
            state.rep_index_map.insert(index, (seq, position));
        }

        for species in py_get(payload, "cb_pinned")?.extract::<Vec<String>>()? {
            state.cb_pinned.insert(species);
        }

        let investment = py_get(payload, "investment_pinned")?;
        for (species, entry) in investment.downcast::<PyDict>()?.iter() {
            let index = entry.get_item(0)?.extract::<i64>()?;
            let code = entry.get_item(1)?.extract::<f64>()?;
            state
                .investment_pinned_state
                .insert(species.extract::<String>()?, (index, code));
        }
        Ok(state)
    }

    /// `golden_corpus_fold.fold_products_to_payload` — native Python objects.
    fn products_payload<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let products = self.products();
        let out = PyDict::new(py);
        out.set_item("schema", FOLD_PRODUCTS_SCHEMA)?;

        let transition_tokens = PyList::empty(py);
        for token in &products.transition_tokens {
            transition_tokens.append(transition_token_to_py(py, token)?)?;
        }
        out.set_item("transition_tokens", transition_tokens)?;
        out.set_item("transition_token_total", products.transition_token_total)?;

        let merged_tokens = PyList::empty(py);
        for token in &products.turn_merged_tokens {
            merged_tokens.append(merged_token_to_py(py, token)?)?;
        }
        out.set_item("turn_merged_tokens", merged_tokens)?;
        out.set_item("turn_merged_total", products.turn_merged_total)?;

        let stats = &products.tendency_stats;
        let tendency = PyDict::new(py);
        tendency.set_item("perspective_slot", side_str(stats.perspective_slot))?;
        tendency.set_item("opponent_slot", side_str(stats.opponent_slot))?;
        tendency.set_item("opponent_switch_count", stats.opponent_switch_count)?;
        tendency.set_item(
            "opponent_decision_opportunities",
            stats.opponent_decision_opportunities,
        )?;
        tendency.set_item(
            "blocked_on_our_attack_count",
            stats.blocked_on_our_attack_count,
        )?;
        tendency.set_item(
            "pursuit_intercept_predict_count",
            stats.pursuit_intercept_predict_count,
        )?;
        tendency.set_item("my_switch_turn_count", stats.my_switch_turn_count)?;
        let mon_tendencies = PyList::empty(py);
        for entry in &stats.opponent_mon_tendencies {
            let mon = PyDict::new(py);
            mon.set_item("slot", side_str(entry.slot))?;
            mon.set_item("species", &entry.species)?;
            mon.set_item(
                "switched_out_before_attacking",
                entry.switched_out_before_attacking,
            )?;
            mon.set_item("stayed_and_attacked", entry.stayed_and_attacked)?;
            mon.set_item("turns_active", entry.turns_active)?;
            mon_tendencies.append(mon)?;
        }
        tendency.set_item("opponent_mon_tendencies", mon_tendencies)?;
        let weather_reveals = PyList::empty(py);
        for (weather, from_ability) in &stats.opponent_weather_reveals {
            let reveal = PyDict::new(py);
            reveal.set_item("weather", weather)?;
            reveal.set_item("from_ability", *from_ability)?;
            weather_reveals.append(reveal)?;
        }
        tendency.set_item("opponent_weather_reveals", weather_reveals)?;
        out.set_item("tendency_stats", tendency)?;

        out.set_item(
            "cb_pinned_species",
            PyList::new(py, products.cb_pinned_species.iter())?,
        )?;
        let investment_pinned = PyDict::new(py);
        for (species, code) in &products.investment_pinned {
            investment_pinned.set_item(species, *code)?;
        }
        out.set_item("investment_pinned", investment_pinned)?;
        Ok(out)
    }
}

fn annotation_index_from_key(key: &Bound<'_, PyAny>) -> PyResult<i64> {
    if let Ok(index) = key.extract::<i64>() {
        return Ok(index);
    }
    key.extract::<String>()?
        .trim()
        .parse::<i64>()
        .map_err(|_| PyValueError::new_err("annotation index keys must be integers"))
}

/// Python-`float`/`bool` canonicalization of one overlay entry, mirroring
/// `transitions_fold.FoldState.apply_annotations_in_place`.
fn annotation_values_from_py(values: &Bound<'_, PyAny>) -> PyResult<AnnotationValues> {
    let residual = values.get_item(0)?;
    let residual = if residual.is_none() {
        None
    } else {
        Some(residual.extract::<f64>()?)
    };
    Ok((
        residual,
        values.get_item(1)?.is_truthy()?,
        values.get_item(2)?.is_truthy()?,
        values.get_item(3)?.extract::<f64>()?,
    ))
}

// ---------------------------------------------------------------------------
// PyO3 surface
// ---------------------------------------------------------------------------

/// Rust `FoldState`: the incremental fold-state advance, payload-compatible
/// with `pokezero.transitions_fold.FoldState` (schema `pokezero.fold-state.v1`).
#[pyclass(name = "FoldState", module = "pokezero_search")]
pub struct PyFoldState {
    inner: FoldStateInner,
}

#[pymethods]
impl PyFoldState {
    /// `FoldState.initial(perspective_slot=..., ...)`.
    #[staticmethod]
    #[pyo3(signature = (perspective_slot, merged_tail_limit = 128, action_tail_limit = 512))]
    fn initial(
        perspective_slot: &str,
        merged_tail_limit: usize,
        action_tail_limit: usize,
    ) -> PyResult<Self> {
        Ok(PyFoldState {
            inner: FoldStateInner::initial(
                parse_side(perspective_slot)?,
                merged_tail_limit,
                action_tail_limit,
            ),
        })
    }

    /// `FoldState.from_payload(payload)`.
    #[staticmethod]
    fn from_payload(payload: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(PyFoldState {
            inner: FoldStateInner::from_payload(payload)?,
        })
    }

    /// In-place advance over an inter-decision raw event slice (`|t:|` filtered).
    fn advance_in_place(&mut self, raw_lines: Vec<String>) -> PyResult<()> {
        self.inner.advance_in_place(&raw_lines)
    }

    /// In-place `apply_annotations` (overlay: {index: [residual, valid, cb, investment]},
    /// keys int or str; applied in ascending index order like the reference).
    fn apply_annotations_in_place(&mut self, overlay: &Bound<'_, PyAny>) -> PyResult<()> {
        let overlay = overlay.downcast::<PyDict>().map_err(|_| {
            PyValueError::new_err("annotation overlay must be a dict of index -> 4 values")
        })?;
        let mut entries: Vec<(i64, AnnotationValues)> = Vec::with_capacity(overlay.len());
        for (key, values) in overlay.iter() {
            entries.push((
                annotation_index_from_key(&key)?,
                annotation_values_from_py(&values)?,
            ));
        }
        entries.sort_by_key(|(index, _)| *index);
        self.inner.apply_annotations_in_place(&entries)
    }

    /// `FoldState.to_payload()` as native Python objects.
    fn to_payload<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        self.inner.to_payload(py)
    }

    /// `golden_corpus_fold.fold_products_to_payload(state.products())` as native objects.
    fn products_payload<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        self.inner.products_payload(py)
    }

    /// Deep copy (each search chance-child advances its OWN fold state).
    fn clone_state(&self) -> Self {
        PyFoldState {
            inner: self.inner.clone(),
        }
    }

    #[getter]
    fn perspective_slot(&self) -> &'static str {
        side_str(self.inner.perspective_slot)
    }

    #[getter]
    fn action_total(&self) -> i64 {
        self.inner.action_total
    }

    #[getter]
    fn event_index(&self) -> i64 {
        self.inner.event_index
    }
}
