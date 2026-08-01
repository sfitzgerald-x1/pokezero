//! Leaf observation construction (engine-swap capstone: real per-outcome
//! model observations at search leaves).
//!
//! A leaf observation is the ROOT observation EVOLVED per branch
//! (owner-decided architecture, docs/leaf_observation_column_map.md):
//!
//! - FOLD-DERIVED columns (transition rows, tendency/stats counters, pinned
//!   Tier-2 conclusions, transition attention extent) come from the branch's
//!   advanced `FoldState` — shared root prefix + appended synthesized tokens,
//!   NO freezing (encoder.rs `write_history_cells`).
//! - ENGINE-STATE-DERIVED columns (HP / status / boosts / actives /
//!   volatiles / weather / side conditions / action legality / PP) are
//!   recomputed from the ENGINE post-state of the branch by rewriting the
//!   root row-inputs JSON in place (`leaf_row_inputs`) and re-encoding.
//! - WORLD-CONSTANT columns (belief facts: possible abilities/items/moves,
//!   uncertainty, candidate variants, revealed flags — the sampled world's
//!   epistemic surface) stay byte-identical to the root: they are epistemic,
//!   not history, and are legitimately root-frozen per world.
//!
//! Beyond the engine state, several ledger surfaces are LINE-driven
//! ([`LeafMeta`], evolved over the branch's synthesized protocol lines and
//! chained per branch like the fold): toxic stages, active stints
//! (turns_active), per-mon sleep counts, the self-team display order
//! (Showdown switch-swap semantics), and the fresh-active choice-lock reset.
//! Snapshot delta families (opponent `move_uses`, sleep-clause holders)
//! evolve from a root engine snapshot. Both reduce to root values at zero
//! branches. See docs/leaf_observation_column_map.md for the full contract.
//!
//! Gates: `scripts/leaf_root_parity.py` (depth-0 byte-parity vs golden) and
//! `scripts/leaf_vs_reality.py` (one-branch differential vs the NEXT golden
//! row — the gate that exercises everything above).

use std::collections::HashMap;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// Encode sub-phase counters (nanoseconds). Global + relaxed: the leaf-pricing
/// closure has no per-search context to thread, and these are diagnostics, not
/// control flow.
pub(crate) static ROW_INPUT_NANOS: AtomicU64 = AtomicU64::new(0);
pub(crate) static PRODUCTS_NANOS: AtomicU64 = AtomicU64::new(0);
pub(crate) static ROW_WRITE_NANOS: AtomicU64 = AtomicU64::new(0);

/// Drain the encode sub-phase counters, returning (row_inputs, products, write)
/// in seconds and resetting them for the next search.
pub(crate) fn drain_encode_subphases() -> (f64, f64, f64) {
    (
        ROW_INPUT_NANOS.swap(0, Ordering::Relaxed) as f64 / 1e9,
        PRODUCTS_NANOS.swap(0, Ordering::Relaxed) as f64 / 1e9,
        ROW_WRITE_NANOS.swap(0, Ordering::Relaxed) as f64 / 1e9,
    )
}
use pyo3::types::PyDict;
use serde_json::{json, Map, Value};

use poke_engine::engine::state::{MoveChoice, PokemonVolatileStatus, Weather};
use poke_engine::state::{PokemonStatus, PokemonType, Side, State};

use crate::encoder::{encode_row_value, encoded_to_dict, EncodedArrays, Tables};
use crate::events::ActiveStatusTransition;
use crate::fold::{FoldStateInner, PyFoldState};
use crate::parse_state;

fn err(msg: impl Into<String>) -> PyErr {
    PyValueError::new_err(msg.into())
}

/// `showdown._normalize_identifier`.
fn normalize_identifier(value: &str) -> String {
    value
        .to_lowercase()
        .chars()
        .filter(|c| c.is_ascii_lowercase() || c.is_ascii_digit())
        .collect()
}

// ---------------------------------------------------------------------------
// Engine-value mappings (gen3 domain)
// ---------------------------------------------------------------------------

/// Engine status -> protocol status code (parser/ledger vocabulary).
fn status_code(status: PokemonStatus) -> Option<&'static str> {
    match status {
        PokemonStatus::NONE => None,
        PokemonStatus::BURN => Some("brn"),
        PokemonStatus::FREEZE => Some("frz"),
        PokemonStatus::PARALYZE => Some("par"),
        PokemonStatus::POISON => Some("psn"),
        PokemonStatus::SLEEP => Some("slp"),
        PokemonStatus::TOXIC => Some("tox"),
        _ => None,
    }
}

/// Engine weather -> parser weather id.
fn weather_id(weather: Weather) -> Option<&'static str> {
    match weather {
        Weather::NONE => None,
        Weather::SUN => Some("sunnyday"),
        Weather::RAIN => Some("raindance"),
        Weather::SAND => Some("sandstorm"),
        Weather::HAIL => Some("hail"),
    }
}

/// Engine volatiles -> the parser's TRACKED_VOLATILES ids (gen3-reachable
/// subset). Engine-only mechanics volatiles (PROTECT, LOCKEDMOVE,
/// MUSTRECHARGE, TRUANT, FLINCH, ...) have no tracked counterpart and are
/// deliberately dropped — the parser never records them either. CURSE is
/// handled separately (Ghost gate): the gen3 engine applies the base Curse
/// choice (self boosts + USER volatile) with no Ghost/non-Ghost split, so a
/// non-Ghost curser carries a spurious engine CURSE volatile the real
/// protocol never starts (review F5; the Ghost-curse TARGET placement
/// remains an engine-model deviation, documented in the column map).
const VOLATILE_MAP: &[(PokemonVolatileStatus, &str)] = &[
    (PokemonVolatileStatus::CONFUSION, "confusion"),
    (PokemonVolatileStatus::LEECHSEED, "leechseed"),
    (PokemonVolatileStatus::SUBSTITUTE, "substitute"),
    (PokemonVolatileStatus::TAUNT, "taunt"),
    (PokemonVolatileStatus::ENCORE, "encore"),
    (PokemonVolatileStatus::DISABLE, "disable"),
    (PokemonVolatileStatus::TORMENT, "torment"),
    (PokemonVolatileStatus::ATTRACT, "attract"),
    (PokemonVolatileStatus::NIGHTMARE, "nightmare"),
    (PokemonVolatileStatus::INGRAIN, "ingrain"),
    (PokemonVolatileStatus::FORESIGHT, "foresight"),
    (PokemonVolatileStatus::DESTINYBOND, "destinybond"),
    (PokemonVolatileStatus::GRUDGE, "grudge"),
    (PokemonVolatileStatus::FOCUSENERGY, "focusenergy"),
    (PokemonVolatileStatus::CHARGE, "charge"),
    (PokemonVolatileStatus::YAWN, "yawn"),
    (PokemonVolatileStatus::STOCKPILE, "stockpile"),
    (PokemonVolatileStatus::BIDE, "bide"),
    (PokemonVolatileStatus::UPROAR, "uproar"),
    (PokemonVolatileStatus::IMPRISON, "imprison"),
    (PokemonVolatileStatus::MAGICCOAT, "magiccoat"),
    (PokemonVolatileStatus::SNATCH, "snatch"),
    (PokemonVolatileStatus::DEFENSECURL, "defensecurl"),
    (PokemonVolatileStatus::MINIMIZE, "minimize"),
    (PokemonVolatileStatus::RAGE, "rage"),
    (PokemonVolatileStatus::PARTIALLYTRAPPED, "partiallytrapped"),
    (PokemonVolatileStatus::FLASHFIRE, "flashfire"),
    // Perish counts: PERISH4 = song declared, counts not yet announced
    // (parser id "perishsong"); PERISH3..1 = the announced countdown.
    (PokemonVolatileStatus::PERISH4, "perishsong"),
    (PokemonVolatileStatus::PERISH3, "perish3"),
    (PokemonVolatileStatus::PERISH2, "perish2"),
    (PokemonVolatileStatus::PERISH1, "perish1"),
];

fn tracked_volatiles(side: &Side, other: &Side) -> Vec<String> {
    use poke_engine::state::PokemonType;
    let mut out: Vec<String> = VOLATILE_MAP
        .iter()
        .filter(|(vs, _)| side.volatile_statuses.contains(vs))
        .map(|(_, id)| (*id).to_string())
        .collect();
    // CURSE placement (review F5 + the live-game protocol probe, 2026-07-19):
    // the real protocol starts Curse on the cursed TARGET
    // (`|-start|p2a: Blissey|Curse|[of] p1a: Gengar`), while the gen3 engine
    // applies the base Curse choice's USER volatile with no Ghost split. So a
    // Ghost-typed OPPOSING active carrying the engine CURSE volatile means
    // THIS side's active is the cursed one; a non-Ghost active carrying it is
    // the spurious stats-up artifact and is dropped. (The volatile's engine
    // lifetime still follows the CURSER's switch-outs — documented
    // engine-model deviation.)
    let is_ghost = |s: &Side| {
        let types = s.get_active_immutable().types;
        types.0 == PokemonType::GHOST || types.1 == PokemonType::GHOST
    };
    if other
        .volatile_statuses
        .contains(&PokemonVolatileStatus::CURSE)
        && is_ghost(other)
    {
        out.push("curse".to_string());
    }
    out
}

/// Engine move id (Choices debug name, lowercased) -> showdown request id.
/// Hidden Power is typed+BP on the engine side but plain "hiddenpower" in
/// requests/candidates (the `_move_specs` matching rule, inverted).
fn showdown_move_id(engine_id: &str) -> String {
    if engine_id.starts_with("hiddenpower") {
        "hiddenpower".to_string()
    } else {
        engine_id.to_string()
    }
}

fn condition_string(hp: i16, maxhp: i16, status: PokemonStatus) -> String {
    if hp <= 0 {
        return "0 fnt".to_string();
    }
    match status_code(status) {
        Some(code) => format!("{hp}/{maxhp} {code}"),
        None => format!("{hp}/{maxhp}"),
    }
}

fn side_ref_for(state: &State, side_is_p1: bool) -> &Side {
    if side_is_p1 {
        &state.side_one
    } else {
        &state.side_two
    }
}

fn active_index_usize(side: &Side) -> usize {
    side.active_index.serialize().parse::<usize>().unwrap_or(0)
}

// ---------------------------------------------------------------------------
// Root engine snapshot (delta families)
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct MonSnapshot {
    /// PP per move slot, keyed by showdown move id (delta base for
    /// `move_uses` / PP fractions).
    pp: Vec<(String, i8)>,
    hp: i16,
    status: Option<&'static str>,
    types: (PokemonType, PokemonType),
}

impl Default for MonSnapshot {
    fn default() -> Self {
        Self {
            pp: Vec::new(),
            hp: 0,
            status: None,
            types: (PokemonType::TYPELESS, PokemonType::TYPELESS),
        }
    }
}

fn changed_live_type_source(
    root: Option<(PokemonType, PokemonType)>,
    current: (PokemonType, PokemonType),
) -> Option<String> {
    if root == Some(current) {
        return None;
    }
    let mut payload = current.0.to_string();
    if current.1 != PokemonType::TYPELESS {
        payload.push('/');
        payload.push_str(&current.1.to_string());
    }
    Some(format!("type:{payload}"))
}

#[derive(Clone, Debug, Default)]
struct SideSnapshot {
    /// Party-index-aligned snapshots.
    mons: Vec<MonSnapshot>,
    /// Root count of non-Rest sleepers (sleep-clause delta base: the world
    /// constructor cannot distinguish Rest sleep publicly, so the engine
    /// predicate alone over-counts at the root).
    nonrest_sleepers: usize,
}

fn nonrest_sleepers(side: &Side) -> usize {
    side.pokemon
        .into_iter()
        .filter(|p| p.status == PokemonStatus::SLEEP && p.hp > 0 && p.rest_turns == 0)
        .count()
}

fn snapshot_side(side: &Side) -> SideSnapshot {
    let mut mons = Vec::new();
    for p in side.pokemon.into_iter() {
        let mut pp = Vec::new();
        for mv in p.moves.into_iter() {
            let engine_id = format!("{:?}", mv.id).to_lowercase();
            if engine_id == "none" {
                continue;
            }
            pp.push((showdown_move_id(&engine_id), mv.pp));
        }
        mons.push(MonSnapshot {
            pp,
            hp: p.hp,
            status: status_code(p.status),
            types: p.types,
        });
    }
    SideSnapshot {
        mons,
        nonrest_sleepers: nonrest_sleepers(side),
    }
}

/// Line-driven side metadata (review F1 + F4): the parser's toxic stage and
/// the belief ledger's active-stint counter are functions of the PROTOCOL
/// LINES, not of engine state — the engine ticks its toxic counter on every
/// end-of-turn run while the parser escalates only on `|turn|` lines (a
/// faint-pending ply ticks the engine but never the parser: toxic_stall
/// repro), and the stint counter advances per `|turn|` and resets on the
/// side's switch lines. So both evolve by replaying the parser's own rules
/// over the branch's synthesized lines, chained exactly like the fold.
/// Indexed by engine side (0 = p1 / side one).
#[derive(Clone, Debug, Default)]
pub(crate) struct LeafMeta {
    pub(crate) toxic: [i64; 2],
    /// Whether the active mon is publicly known to retain badly poisoned
    /// status. The event renderer omits an unchanged status from ordinary HP
    /// lines, so residual recovery carries this fact across those lines.
    pub(crate) active_toxic: [bool; 2],
    /// Exact HP condition last announced for each active side. This is only
    /// used to recover a Toxic multiplier from the next exact residual.
    pub(crate) active_hp: [Option<(i64, i64)>; 2],
    /// A switch/drag proves the incoming Toxic mon's simulator counter reset
    /// to zero. A first `/100` Toxic residual can therefore prove stage one;
    /// no other rounded residual is safe to invert.
    pub(crate) toxic_reentry_pending: [bool; 2],
    pub(crate) stint: [i64; 2],
    /// Per-mon sleep bookkeeping keyed by (engine side, species key):
    /// (started, cant_count). `started` marks a `|-status|..|slp` seen in the
    /// branch (the ledger's counter restarts at 0 there); `cant_count` is the
    /// observed `|cant ..|slp` turns — the ledger's sleep_turns unit
    /// (belief.py: "observed |cant …|slp turns since the status landed").
    /// Keyed per mon (not per side) so a sleeper that faints or switches out
    /// after its cants still carries them.
    pub(crate) sleep: HashMap<(usize, String), (bool, i64)>,
    /// The side's active switched in during the branch and has not used a
    /// move since (choice locks reset on switch; the world seeds benched
    /// mons with stale per-stint disabled bits and `use_last_used_move` is
    /// off in constructed worlds, so this is line-tracked).
    pub(crate) fresh_active: [bool; 2],
    /// PP charged per (engine side, species key, showdown move id), replayed
    /// over the branch's `|move|` lines with the PARSER's charging rules
    /// (belief.py `_charge_move_use` :796 + the `|move|` ingestion exemptions
    /// :431-445): called moves (`[from] Sleep Talk` class) and locked
    /// continuations (`[from]lockedmove`) charge nothing, Struggle has no PP,
    /// Pressure on the OPPOSING active doubles the charge for foe-targeted
    /// moves. The engine only emits `DecrementPP` below 10 PP
    /// (gen3/generate_instructions.rs "only decrement pp if the move is at 10
    /// or less" optimization), so engine PP alone is root-frozen above 10 —
    /// review F3; this replay is the fix.
    pub(crate) move_charges: HashMap<(usize, String, String), i64>,
    /// Current active species key per engine side (Pressure resolution at
    /// `|move|` time; switches/drags update it from the DETAILS field).
    pub(crate) active: [String; 2],
    /// `|turn|` lines seen during the branch (the parser's turn_number
    /// advance — set-turn arithmetic for in-branch side conditions).
    pub(crate) turns_seen: i64,
    /// In-branch `|-sidestart|`/`|-sideend|` replay for the TIMED conditions
    /// (reflect/lightscreen/safeguard/mist — showdown.py
    /// `_update_timed_side_conditions` :909): `Some(turn delta at set)` per
    /// (engine side, condition id), `None` when the condition ended (the
    /// parser pops the set-turn entry).
    pub(crate) side_condition_sets: HashMap<(usize, String), Option<i64>>,
}

/// Static line-replay context: per-side facts of the sampled world the
/// replay rules need (currently: which species carry Pressure — the parser's
/// PP double-charge reads the opposing active's ability).
#[derive(Clone, Debug, Default)]
pub(crate) struct LeafMetaCtx {
    /// Normalized species keys per engine side whose sampled ability is
    /// Pressure (gen 3 announces Pressure on entry, so world truth and the
    /// parser's revealed-ability rule coincide for on-field mons).
    pub(crate) pressure: [Vec<String>; 2],
    /// HP Percentage Mod applies to these sides' rendered protocol lines.
    /// `/100` damage is rounded, except for the first residual after a
    /// switch/drag reset, which is provably Toxic stage one.
    pub(crate) hp_percent: [bool; 2],
}

/// Construction-only root proof for the one active-Toxic zero that is public:
/// a same-seat fainted mon was replaced after upkeep. Each seat must carry the
/// complete exact-boolean Python attestation; this lives in `ctx_json`, not the
/// observation metadata or any frozen model schema.
fn root_toxic_zero_after_upkeep(ctx: &Value) -> [bool; 2] {
    let proof = ctx
        .get("toxic_stage_zero_after_upkeep")
        .and_then(Value::as_object);
    ["p1", "p2"].map(|side| {
        let attestation = proof
            .and_then(|proof| proof.get(side))
            .and_then(Value::as_object);
        attestation
            .and_then(|attestation| {
                (attestation.get("proof").and_then(Value::as_bool) == Some(true)
                    && attestation.get("pending").and_then(Value::as_bool) == Some(false)
                    && attestation.get("invalid").and_then(Value::as_bool) == Some(false)
                    && attestation
                        .get("post_upkeep_window")
                        .and_then(Value::as_bool)
                        .is_some())
                .then_some(true)
            })
            .unwrap_or(false)
    })
}

fn seed_root_toxic_reentry_pending(meta: &mut LeafMeta, proof: [bool; 2]) {
    for side in 0..2 {
        // The proof is meaningful only for the active toxic zero itself. A
        // malformed/stale context must fail closed instead of leaking into a
        // later residual, cure, switch, or faint.
        meta.toxic_reentry_pending[side] =
            proof[side] && meta.toxic[side] == 0 && meta.active_toxic[side];
    }
}

/// The parser's timed side conditions (showdown.py `_TIMED_SIDE_CONDITIONS`).
const TIMED_SIDE_CONDITIONS: [&str; 4] = ["reflect", "lightscreen", "safeguard", "mist"];

/// Caller moves whose called `|move|` line charges no PP of its own
/// (belief.py `_CALLER_MOVES`; the caller's own line was already charged).
const CALLER_MOVES: [&str; 6] = [
    "metronome",
    "mirrormove",
    "sleeptalk",
    "assist",
    "naturepower",
    "copycat",
];

/// `showdown._side_condition_identifier`: strip a `move:`/`ability:`/`item:`
/// source prefix, then normalize.
fn side_condition_id(raw: &str) -> String {
    let condition = raw.trim();
    let stripped = match condition.split_once(':') {
        Some((prefix, rest))
            if matches!(
                prefix.trim().to_lowercase().as_str(),
                "move" | "ability" | "item"
            ) =>
        {
            rest.trim()
        }
        _ => condition,
    };
    normalize_identifier(stripped)
}

/// `belief._called_move_source`: the normalized `[from]` tag of a |move|
/// line (both the spaced and unspaced forms; `move:` prefix stripped), or
/// None when the line carries no `[from]`.
fn called_move_source(line: &str) -> Option<String> {
    let marker = line.find("[from]")?;
    let tag = line[marker + "[from]".len()..]
        .split('|')
        .next()
        .unwrap_or("")
        .trim();
    let tag = match tag.to_lowercase().strip_prefix("move:") {
        Some(_) => tag.split_once(':').map(|(_, rest)| rest.trim()).unwrap_or(tag),
        None => tag,
    };
    Some(normalize_identifier(tag))
}

/// Parse a protocol HP condition's exact numerator and denominator.
/// Percentage-mode lines use the same shape (`n/100`); callers must consult
/// [`LeafMetaCtx::hp_percent`] before treating that pair as exact HP.
fn condition_hp(raw: &str) -> Option<(i64, i64)> {
    let hp = raw.split_whitespace().next()?;
    let (current, maximum) = hp.split_once('/')?;
    let current = current.parse::<i64>().ok()?;
    let maximum = maximum.parse::<i64>().ok()?;
    (current >= 0 && maximum > 0).then_some((current, maximum))
}

/// The explicit major status on a protocol condition, if the line carries one.
/// Bare HP updates deliberately return `None`: they do not change status.
fn condition_status(raw: &str) -> Option<String> {
    raw.split_whitespace()
        .skip(1)
        .find(|field| *field != "fnt")
        .map(normalize_identifier)
}

fn condition_is_fainted(raw: &str) -> bool {
    raw.split_whitespace().any(|field| field == "fnt")
}

fn clear_toxic_meta(meta: &mut LeafMeta, side: usize) {
    meta.toxic[side] = 0;
    meta.active_toxic[side] = false;
    meta.toxic_reentry_pending[side] = false;
}

/// Re-seed a Toxic stage from a residual damage line before its HP condition
/// replaces the prior value. Exact HP recovers any integral Gen 3 multiplier;
/// rounded `/100` HP recovers only the first tick after a public re-entry.
fn reseed_toxic_from_residual(meta: &mut LeafMeta, rest: &str, ctx: &LeafMetaCtx) {
    let Some(side) = line_slot(rest) else { return };
    let fields: Vec<&str> = rest.split('|').collect();
    let condition = fields.get(1).copied().unwrap_or("");
    let is_toxic_residual = fields.iter().skip(2).any(|field| field.trim() == "[from] psn");
    if !is_toxic_residual || !meta.active_toxic[side] {
        return;
    }
    let Some((current_hp, maximum_hp)) = condition_hp(condition) else {
        meta.toxic_reentry_pending[side] = false;
        return;
    };
    if current_hp <= 0 {
        clear_toxic_meta(meta, side);
        return;
    }
    if ctx.hp_percent[side] {
        if meta.toxic_reentry_pending[side] {
            meta.toxic[side] = 1;
        }
        meta.toxic_reentry_pending[side] = false;
        return;
    }
    let Some((previous_hp, previous_maximum)) = meta.active_hp[side] else {
        meta.toxic_reentry_pending[side] = false;
        return;
    };
    meta.toxic_reentry_pending[side] = false;
    if maximum_hp != previous_maximum || previous_hp <= current_hp {
        return;
    }
    let damage = previous_hp - current_hp;
    let unit = (maximum_hp / 16).max(1);
    if damage % unit != 0 {
        return;
    }
    let stage = damage / unit;
    if (1..=15).contains(&stage) {
        meta.toxic[side] = stage;
    }
}

/// Carry the public active HP/status surface across protocol lines. The
/// renderer writes only HP on ordinary damage/heal lines, so no status token
/// means "unchanged", not "cured".
fn update_active_condition(meta: &mut LeafMeta, side: usize, condition: &str) {
    if let Some(hp) = condition_hp(condition) {
        meta.active_hp[side] = Some(hp);
    }
    if condition_is_fainted(condition) {
        clear_toxic_meta(meta, side);
    } else if let Some(status) = condition_status(condition) {
        meta.active_toxic[side] = status == "tox";
        if status != "tox" {
            meta.toxic[side] = 0;
            meta.toxic_reentry_pending[side] = false;
        }
    }
}

/// Engine side index from a SIDE ident ("p1: name" / "p1a: name").
fn side_ident_slot(rest: &str) -> Option<usize> {
    if rest.starts_with("p1") {
        Some(0)
    } else if rest.starts_with("p2") {
        Some(1)
    } else {
        None
    }
}

fn line_slot(line_after_prefix: &str) -> Option<usize> {
    // "...|p1a: Name|..." — the ident is the first field.
    if line_after_prefix.starts_with("p1a: ") {
        Some(0)
    } else if line_after_prefix.starts_with("p2a: ") {
        Some(1)
    } else {
        None
    }
}

/// The species key from a line's leading ident field ("p1a: Dewgong|..." ->
/// "dewgong"). Synthesized idents are species-based (local domain).
fn ident_species_key(line_after_prefix: &str) -> String {
    let ident = line_after_prefix.split('|').next().unwrap_or("");
    let name = ident.splitn(2, ": ").nth(1).unwrap_or("");
    normalize_identifier(name)
}

/// Normalized species key of a team/belief JSON entry.
fn species_key(obj: &Map<String, Value>) -> String {
    normalize_identifier(obj.get("species").and_then(Value::as_str).unwrap_or(""))
}

/// Replay the parser's toxic/stint rules over synthesized lines
/// (`showdown._ReplayParser._feed_line`: `|-status|..|tox` sets stage 1,
/// status replacement, `|-curestatus|`, `|-cureteam|`, and `|faint|` clear;
/// the side's `|switch|`/`|drag|` resets stage AND stint; `|turn|` escalates
/// every nonzero stage through the internal saturation sentinel 16 and
/// advances every stint. Exact residuals recover their multiplier, while
/// rounded residuals recover only a switch/drag-proven stage one. This also
/// replays the PP-charge rules (see [`LeafMeta`]) and the
/// timed side-condition set-turn replay. `ctx` carries the sampled world's
/// static per-side facts (Pressure holders and HP representation).
pub(crate) fn evolve_leaf_meta(meta: &LeafMeta, lines: &[String], ctx: &LeafMetaCtx) -> LeafMeta {
    evolve_leaf_meta_with_status_transitions(meta, lines, ctx, &[])
}

pub(crate) fn evolve_leaf_meta_with_status_transitions(
    meta: &LeafMeta,
    lines: &[String],
    ctx: &LeafMetaCtx,
    transitions: &[ActiveStatusTransition],
) -> LeafMeta {
    let mut out = meta.clone();
    for line_offset in 0..=lines.len() {
        for transition in transitions.iter().filter(|transition| transition.line_offset == line_offset) {
            if transition.new_status == PokemonStatus::TOXIC {
                out.toxic[transition.side] = 1;
                out.active_toxic[transition.side] = true;
                out.toxic_reentry_pending[transition.side] = false;
            } else {
                clear_toxic_meta(&mut out, transition.side);
            }
        }
        if line_offset == lines.len() {
            break;
        }
        let line = &lines[line_offset];
        if line.starts_with("|turn|") {
            // A root stage-zero proof has exactly one residual opportunity.
            // If the first branch marker arrives without that Toxic residual,
            // do not let a later rounded residual manufacture its stage.
            out.toxic_reentry_pending = [false, false];
            for side in 0..2 {
                if out.toxic[side] > 0 {
                    out.toxic[side] = (out.toxic[side] + 1).min(16);
                }
                out.stint[side] += 1;
            }
            out.turns_seen += 1;
            continue;
        }
        if line == "|upkeep" {
            // `|upkeep` follows the residual block. A live proof here missed
            // its first Toxic tick and is terminal for this leaf branch.
            out.toxic_reentry_pending = [false, false];
            continue;
        }
        if let Some(rest) = line.strip_prefix("|move|") {
            if let Some(side) = line_slot(rest) {
                out.fresh_active[side] = false;
                // PP-charge replay (belief.py |move| ingestion :411-445):
                // called moves and locked continuations charge nothing;
                // Struggle has no PP; otherwise charge 1, doubled when the
                // move is foe-targeted and the OPPOSING active has Pressure.
                let fields: Vec<&str> = rest.split('|').collect();
                let move_id = showdown_move_id(&normalize_identifier(
                    fields.get(1).copied().unwrap_or(""),
                ));
                let called = called_move_source(line).is_some_and(|source| {
                    source == "lockedmove" || CALLER_MOVES.contains(&source.as_str())
                });
                if !called && !move_id.is_empty() && move_id != "struggle" {
                    let target_slot = fields
                        .get(2)
                        .copied()
                        .and_then(line_slot_of_ident);
                    let foe_targeted = target_slot.is_some_and(|t| t != side);
                    let opposing = 1 - side;
                    let charge = if foe_targeted
                        && ctx.pressure[opposing]
                            .iter()
                            .any(|key| *key == out.active[opposing])
                    {
                        2
                    } else {
                        1
                    };
                    *out
                        .move_charges
                        .entry((side, ident_species_key(rest), move_id))
                        .or_insert(0) += charge;
                }
            }
            continue;
        }
        if let Some(rest) = line.strip_prefix("|cant|") {
            if let Some(side) = line_slot(rest) {
                if rest.split('|').nth(1).map(str::trim) == Some("slp") {
                    let key = ident_species_key(rest);
                    out.sleep.entry((side, key)).or_insert((false, 0)).1 += 1;
                }
            }
            continue;
        }
        if let Some(rest) = line.strip_prefix("|-damage|") {
            reseed_toxic_from_residual(&mut out, rest, ctx);
            if let Some(side) = line_slot(rest) {
                if let Some(condition) = rest.split('|').nth(1) {
                    update_active_condition(&mut out, side, condition);
                }
            }
            continue;
        }
        if let Some(rest) = line.strip_prefix("|-heal|") {
            if let Some(side) = line_slot(rest) {
                if let Some(condition) = rest.split('|').nth(1) {
                    update_active_condition(&mut out, side, condition);
                }
            }
            continue;
        }
        if let Some(rest) = line.strip_prefix("|faint|") {
            if let Some(side) = line_slot(rest) {
                clear_toxic_meta(&mut out, side);
                out.active_hp[side] = None;
            }
            continue;
        }
        if let Some(rest) = line.strip_prefix("|-cureteam|") {
            if let Some(side) = side_ident_slot(rest) {
                clear_toxic_meta(&mut out, side);
            }
            continue;
        }
        // Timed side-condition set turns (showdown.py
        // `_update_timed_side_conditions` :909-922): |-sidestart| records the
        // parser's CURRENT turn, |-sideend| pops the entry. Side idents on
        // these lines are side-shaped ("p1: name"), not position-shaped.
        for (prefix, is_start) in [("|-sidestart|", true), ("|-sideend|", false)] {
            let Some(rest) = line.strip_prefix(prefix) else { continue };
            let Some(side) = side_ident_slot(rest) else { break };
            let condition = side_condition_id(rest.split('|').nth(1).unwrap_or(""));
            if TIMED_SIDE_CONDITIONS.contains(&condition.as_str()) {
                out.side_condition_sets.insert(
                    (side, condition),
                    if is_start { Some(out.turns_seen) } else { None },
                );
            }
            break;
        }
        for (prefix, is_status, is_cure) in [
            ("|switch|", false, false),
            ("|drag|", false, false),
            ("|-status|", true, false),
            ("|-curestatus|", false, true),
        ] {
            let Some(rest) = line.strip_prefix(prefix) else { continue };
            let Some(side) = line_slot(rest) else { break };
            if is_status {
                match rest.split('|').nth(1).map(|s| normalize_identifier(s.trim())) {
                    Some(status) if status == "tox" => {
                        out.toxic[side] = 1;
                        out.active_toxic[side] = true;
                        out.toxic_reentry_pending[side] = false;
                    }
                    Some(status) if status == "slp" => {
                        clear_toxic_meta(&mut out, side);
                        let key = ident_species_key(rest);
                        out.sleep.insert((side, key), (true, 0));
                    }
                    Some(_) => clear_toxic_meta(&mut out, side),
                    None => {}
                }
            } else if is_cure {
                clear_toxic_meta(&mut out, side);
            } else {
                clear_toxic_meta(&mut out, side);
                out.stint[side] = 0;
                out.fresh_active[side] = true;
                // Active tracking (Pressure resolution): species from the
                // DETAILS field, exactly like `evolve_self_order`.
                let details = rest.split('|').nth(1).unwrap_or("");
                let species = normalize_identifier(details.split(',').next().unwrap_or(""));
                if !species.is_empty() {
                    out.active[side] = species;
                }
                let condition = rest.split('|').nth(2).unwrap_or("");
                update_active_condition(&mut out, side, condition);
                out.toxic_reentry_pending[side] = out.active_toxic[side];
            }
            break;
        }
    }
    out
}

/// `line_slot` over a bare ident field (no leading event prefix).
fn line_slot_of_ident(ident: &str) -> Option<usize> {
    let ident = ident.trim();
    if ident.starts_with("p1a: ") {
        Some(0)
    } else if ident.starts_with("p2a: ") {
        Some(1)
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// The leaf context
// ---------------------------------------------------------------------------

pub(crate) struct LeafContext {
    pub(crate) tables: Arc<Tables>,
    root: Value,
    /// True when the acting seat is p1 (engine side one).
    self_is_p1: bool,
    /// Normalized species keys per engine side, engine party order.
    species_keys: [Vec<String>; 2],
    root_snapshot: [SideSnapshot; 2],
    /// Root self-team display order as normalized species keys (Showdown
    /// request order, active first). Switches during a branch SWAP the
    /// incoming mon with slot 0 — the exact `switchIn` semantics
    /// (sim/battle-actions.ts) — via `evolve_self_order`.
    root_self_order: Vec<String>,
    /// Root line-driven metadata (toxic stages + active stints).
    root_meta: LeafMeta,
    /// Static replay context (Pressure holders per side, from the sampled
    /// world's engine abilities).
    meta_ctx: LeafMetaCtx,
    /// Root engine active party index per side (self action-token PP base
    /// selection: the ROOT ACTIVE's engine PP is request-seeded and exact;
    /// benched mons' cached request-history PP is stale by their last
    /// stint's final action, so their base is the belief ledger below).
    root_active_party: [usize; 2],
    /// Caller-supplied opponent root request order (empty when absent).
    root_opponent_request_order: Vec<String>,
    /// The SELF side's belief-ledger PP charges per (species key, move id)
    /// (`belief_view.self_pokemon[*].move_uses` — the parser's public
    /// charging count, exact where the cached request-history PP is stale).
    self_ledger_uses: HashMap<(String, String), i64>,
    /// Root battle turn + recorded weather (parser-formula weather ticking).
    root_turn: i64,
    root_weather: Option<String>,
    root_weather_remaining: i64,
}

impl LeafContext {
    pub(crate) fn new(
        tables_json: &str,
        root_inputs_json: &str,
        ctx_json: &str,
        root_state: &State,
    ) -> PyResult<Self> {
        Self::with_tables(Arc::new(Tables::from_json(tables_json)?), root_inputs_json, ctx_json, root_state)
    }

    /// Re-root over ALREADY-PARSED tables.
    ///
    /// The tables artifact is ~475 KB of dex/vocab JSON and parsing it costs
    /// ~3 ms — two orders of magnitude more than an encode. A caller that
    /// re-roots often (an environment refreshing its public-information
    /// surface as the opponent reveals mons) must not pay that again, so the
    /// parsed tables are shared rather than rebuilt.
    pub(crate) fn with_tables(
        tables: Arc<Tables>,
        root_inputs_json: &str,
        ctx_json: &str,
        root_state: &State,
    ) -> PyResult<Self> {
        let root: Value = serde_json::from_str(root_inputs_json)
            .map_err(|e| err(format!("root inputs JSON: {e}")))?;
        let md = root
            .get("observation_metadata")
            .ok_or_else(|| err("root inputs missing observation_metadata"))?;
        let self_slot = md
            .get("showdown_slot")
            .and_then(Value::as_str)
            .ok_or_else(|| err("root metadata missing showdown_slot"))?;
        let self_is_p1 = match self_slot {
            "p1" => true,
            "p2" => false,
            other => return Err(err(format!("unsupported showdown_slot {other:?}"))),
        };
        let ctx: Value =
            serde_json::from_str(ctx_json).map_err(|e| err(format!("ctx JSON: {e}")))?;
        let mut species_keys: [Vec<String>; 2] = [Vec::new(), Vec::new()];
        for (key, out) in [("p1", 0usize), ("p2", 1usize)] {
            let list = ctx
                .get(key)
                .and_then(Value::as_array)
                .ok_or_else(|| err(format!("ctx JSON missing {key} species array")))?;
            for entry in list {
                let name = entry
                    .as_str()
                    .ok_or_else(|| err("ctx species entries must be strings"))?;
                species_keys[out].push(normalize_identifier(name));
            }
        }
        let mut meta_ctx = LeafMetaCtx::default();
        if let Some(hp_percent) = ctx.get("hp_percent").and_then(Value::as_array) {
            for side in hp_percent {
                match side.as_str() {
                    Some("p1") => meta_ctx.hp_percent[0] = true,
                    Some("p2") => meta_ctx.hp_percent[1] = true,
                    other => return Err(err(format!("ctx JSON has bad hp_percent side {other:?}"))),
                }
            }
        }
        let root_toxic_zero_proof = root_toxic_zero_after_upkeep(&ctx);
        let root_self_order: Vec<String> = md
            .get("self_team")
            .and_then(Value::as_array)
            .map(|team| {
                team.iter()
                    .map(|entry| {
                        normalize_identifier(
                            entry.get("species").and_then(Value::as_str).unwrap_or(""),
                        )
                    })
                    .collect()
            })
            .unwrap_or_default();
        // Root line-driven metadata: toxic stages from the recorded ledger
        // fields, stints from the active belief entries' turns_active.
        let mut root_meta = LeafMeta::default();
        let (self_engine, opp_engine) = if self_is_p1 { (0, 1) } else { (1, 0) };
        root_meta.toxic[self_engine] = md
            .get("self_toxic_stage")
            .and_then(Value::as_i64)
            .unwrap_or(0);
        root_meta.toxic[opp_engine] = md
            .get("opponent_toxic_stage")
            .and_then(Value::as_i64)
            .unwrap_or(0);
        let belief = md.get("belief_view");
        for (key, engine_side) in [
            ("self_pokemon", self_engine),
            ("opponent_pokemon", opp_engine),
        ] {
            // The root ACTIVE entry's stint: by ledger active flag, falling
            // back to the engine root active's species.
            let engine_active = if engine_side == 0 {
                active_index_usize(&root_state.side_one)
            } else {
                active_index_usize(&root_state.side_two)
            };
            let active_key = species_keys[engine_side]
                .get(engine_active)
                .cloned()
                .unwrap_or_default();
            let stint = belief
                .and_then(|b| b.get(key))
                .and_then(Value::as_array)
                .and_then(|entries| {
                    entries
                        .iter()
                        .find(|entry| {
                            entry.get("active").and_then(Value::as_bool).unwrap_or(false)
                        })
                        .or_else(|| {
                            entries.iter().find(|entry| {
                                normalize_identifier(
                                    entry.get("species").and_then(Value::as_str).unwrap_or(""),
                                ) == active_key
                            })
                        })
                })
                .and_then(|entry| entry.get("turns_active").and_then(Value::as_i64))
                .unwrap_or(0);
            root_meta.stint[engine_side] = stint;
        }
        // Root actives + Pressure holders (PP-charge replay context). The
        // engine state carries the sampled world's abilities; gen 3 announces
        // Pressure on entry, so world truth matches the parser's
        // revealed-ability double-charge rule for on-field mons.
        for (engine_side, side) in [(0usize, &root_state.side_one), (1, &root_state.side_two)] {
            let active = active_index_usize(side);
            root_meta.active[engine_side] = species_keys[engine_side]
                .get(active)
                .cloned()
                .unwrap_or_default();
            let active = side.get_active_immutable();
            root_meta.active_toxic[engine_side] = active.status == PokemonStatus::TOXIC;
            root_meta.active_hp[engine_side] = (active.maxhp > 0)
                .then_some((active.hp as i64, active.maxhp as i64));
        }
        seed_root_toxic_reentry_pending(&mut root_meta, root_toxic_zero_proof);
        for (engine_side, side) in [(0usize, &root_state.side_one), (1, &root_state.side_two)] {
            for (party, p) in side.pokemon.into_iter().enumerate() {
                if p.ability == poke_engine::engine::abilities::Abilities::PRESSURE {
                    if let Some(key) = species_keys[engine_side].get(party) {
                        meta_ctx.pressure[engine_side].push(key.clone());
                    }
                }
            }
        }
        let root_active_party = [
            active_index_usize(&root_state.side_one),
            active_index_usize(&root_state.side_two),
        ];
        // SELF belief-ledger PP charges (see the field doc). The self side of
        // the ledger is player-known state, not epistemic belief facts.
        let mut self_ledger_uses: HashMap<(String, String), i64> = HashMap::new();
        if let Some(entries) = belief
            .and_then(|b| b.get("self_pokemon"))
            .and_then(Value::as_array)
        {
            for entry in entries {
                let Some(obj) = entry.as_object() else { continue };
                let mon_key = species_key(obj);
                let Some(uses) = obj.get("move_uses").and_then(Value::as_array) else {
                    continue;
                };
                for pair in uses {
                    let Some(items) = pair.as_array() else { continue };
                    if items.len() != 2 {
                        continue;
                    }
                    let Some(move_id) = items[0].as_str().map(normalize_identifier) else {
                        continue;
                    };
                    let charged = items[1].as_i64().unwrap_or(0);
                    self_ledger_uses.insert((mon_key.clone(), move_id), charged);
                }
            }
        }
        let root_turn = ctx.get("turn").and_then(Value::as_i64).unwrap_or(0);
        // The opponent's ROOT REQUEST ORDER, supplied by the caller when it can
        // compute it. This is the channel attempts 2 and 3 lacked: the crate
        // never receives pre-root protocol lines, so it cannot replay the
        // opponent's switch history itself, and every in-crate approximation
        // has been wrong beyond one switch. Python already maintains exactly
        // this permutation (determinization's `current_order`), so it is passed
        // in rather than guessed.
        let root_opponent_request_order: Vec<String> = ctx
            .get("opponent_request_order")
            .and_then(Value::as_array)
            .map(|entries| {
                entries
                    .iter()
                    .filter_map(Value::as_str)
                    .map(normalize_identifier)
                    .collect()
            })
            .unwrap_or_default();
        let root_weather = md
            .get("weather")
            .and_then(Value::as_str)
            .filter(|w| !w.is_empty())
            .map(|w| w.to_string());
        let root_weather_remaining = md
            .get("weather_turns_remaining")
            .and_then(Value::as_i64)
            .unwrap_or(0);
        Ok(LeafContext {
            tables,
            root,
            self_is_p1,
            species_keys,
            root_snapshot: [
                snapshot_side(&root_state.side_one),
                snapshot_side(&root_state.side_two),
            ],
            root_self_order,
            root_meta,
            meta_ctx,
            root_active_party,
            root_opponent_request_order,
            self_ledger_uses,
            root_turn,
            root_weather,
            root_weather_remaining,
        })
    }

    pub(crate) fn root_self_order(&self) -> &[String] {
        &self.root_self_order
    }

    pub(crate) fn root_meta(&self) -> &LeafMeta {
        &self.root_meta
    }

    pub(crate) fn meta_ctx(&self) -> &LeafMetaCtx {
        &self.meta_ctx
    }

    pub(crate) fn self_prefix(&self) -> &'static str {
        if self.self_is_p1 {
            "p1"
        } else {
            "p2"
        }
    }

    fn engine_side_index(&self, slot_is_self: bool) -> usize {
        match (self.self_is_p1, slot_is_self) {
            (true, true) | (false, false) => 0,
            _ => 1,
        }
    }

    /// Engine party index for a display-species key on one engine side.
    fn party_index(&self, engine_side: usize, species_key: &str) -> Option<usize> {
        self.species_keys[engine_side]
            .iter()
            .position(|key| key == species_key)
    }

    /// Rewrite the root row inputs into this LEAF state's view. `turn` is the
    /// leaf's battle turn (root turn + completed simulated turns);
    /// `self_order` the evolved self-team display order (None = root order);
    /// `meta` the evolved line-driven metadata (None = root values).
    ///
    /// `engine_authoritative` says the engine's option surface at this state
    /// IS the request — which is true exactly when the state was reached by
    /// the engine's own transitions rather than reconstructed. It does two
    /// things:
    ///
    /// 1. Reads options from `root_get_all_options` (the DECISION-POINT
    ///    surface, which honors `force_trapped` and the slow-uturn /
    ///    Baton-Pass mid-turn shape) instead of the interior
    ///    `get_all_options`.
    /// 2. Drops the fresh-switch-in move widening. That widening exists
    ///    because a world CONSTRUCTED from a materialization payload seeds
    ///    benched mons with their last stint's stale disabled bits, so the
    ///    engine under-reports a fresh switch-in's moves; a natively-evolved
    ///    state has no such staleness, and the widening would instead
    ///    over-report — marking legal the moves a Baton-Pass-committed seat,
    ///    an Encored seat, or a PP-exhausted slot cannot actually pick.
    ///
    /// Search leaves are interior nodes over constructed worlds and must stay
    /// on `false` — that is also what the root-parity and prior-mapping gates
    /// are calibrated against. An engine-as-environment driver passes `true`.
    pub(crate) fn leaf_row_inputs(
        &self,
        state: &State,
        turn: i64,
        self_order: Option<&[String]>,
        meta: Option<&LeafMeta>,
        engine_authoritative: bool,
    ) -> PyResult<Value> {
        let meta = meta.unwrap_or(&self.root_meta);
        let mut row = self.root.clone();

        // Split borrows: rewrite metadata first, then the materialization.
        let self_is_p1 = self.self_is_p1;
        let self_side = side_ref_for(state, self_is_p1);
        let opp_side = side_ref_for(state, !self_is_p1);
        let self_engine = self.engine_side_index(true);
        let opp_engine = self.engine_side_index(false);
        let self_force_switch = self_side.force_switch
            || self_side.get_active_immutable().hp <= 0;

        let md = row
            .get_mut("observation_metadata")
            .and_then(Value::as_object_mut)
            .ok_or_else(|| err("row inputs missing observation_metadata object"))?;

        // Self-team display order (review F2): golden observations order the
        // self team ACTIVE-FIRST per the request; switches during a branch
        // SWAP the incoming mon with slot 0 (Showdown `switchIn` semantics).
        // Reorder BEFORE any per-mon rewrite so active flags, switch tokens,
        // and mask indices all land on the golden positions.
        if let Some(order) = self_order {
            if let Some(team) = md.get_mut("self_team").and_then(Value::as_array_mut) {
                let mut remaining: Vec<Value> = std::mem::take(team);
                let mut arranged: Vec<Value> = Vec::with_capacity(remaining.len());
                for key in order {
                    if let Some(pos) = remaining.iter().position(|entry| {
                        normalize_identifier(
                            entry.get("species").and_then(Value::as_str).unwrap_or(""),
                        ) == *key
                    }) {
                        arranged.push(remaining.remove(pos));
                    }
                }
                arranged.append(&mut remaining); // defensive: keep unmatched
                *team = arranged;
            }
        }

        // Root ledger values for the delta families (read before overwrite).
        let root_sleep_clause = [
            md.get("self_sleep_clause_used")
                .and_then(Value::as_bool)
                .unwrap_or(false),
            md.get("opponent_sleep_clause_used")
                .and_then(Value::as_bool)
                .unwrap_or(false),
        ];

        // --- field-level scalars ---
        md.insert("turn_number".into(), json!(turn));
        md.insert(
            "request_kind".into(),
            json!(if self_force_switch { "force_switch" } else { "move" }),
        );
        match weather_id(state.weather.weather_type) {
            Some(id) => {
                md.insert("weather".into(), json!(id));
                let permanent = state.weather.turns_remaining < 0;
                md.insert("weather_permanent".into(), json!(permanent));
                // Weather ticking is TURN-driven for the parser
                // (remaining = duration − (turn − set turn)) while the
                // engine decrements per end-of-turn run — a faint-pending
                // ply ticks the engine but not the parser. Same weather as
                // the root: root remaining − completed simulated turns;
                // weather set in-branch: the engine counter (set-ply
                // granularity, documented).
                let remaining = if permanent {
                    self.tables.layout_timed_condition_duration()
                } else if self.root_weather.as_deref() == Some(id) {
                    (self.root_weather_remaining - (turn - self.root_turn).max(0)).max(0)
                } else {
                    state.weather.turns_remaining as i64
                };
                md.insert("weather_turns_remaining".into(), json!(remaining));
            }
            None => {
                md.insert("weather".into(), Value::Null);
                md.insert("weather_permanent".into(), json!(false));
                md.insert("weather_turns_remaining".into(), json!(0));
            }
        }
        md.insert(
            "self_future_sight_turns".into(),
            json!(self_side.future_sight.0 as i64),
        );
        md.insert(
            "opponent_future_sight_turns".into(),
            json!(opp_side.future_sight.0 as i64),
        );
        md.insert("self_wish_pending".into(), json!(self_side.wish.0 != 0));
        md.insert("opponent_wish_pending".into(), json!(opp_side.wish.0 != 0));
        // Sleep clause: the flag marks the side that INFLICTED sleep (the
        // ledger's sleep_clause_holders; golden-proven: Sleep Powder on the
        // opponent raises the USER's flag) — so a side's clause goes up when
        // its OPPONENT gains a new non-Rest sleeper. Toxic stage:
        // line-driven metadata (the parser escalates on |turn| lines only —
        // review F1). The line replay clears every public Toxic-ending event;
        // this state guard also protects rows whose engine state proves the
        // active fainted or has a non-Toxic replacement status.
        for (index, (key_sc, key_tox, side, other, engine_side)) in [
            (
                "self_sleep_clause_used",
                "self_toxic_stage",
                self_side,
                opp_side,
                self_engine,
            ),
            (
                "opponent_sleep_clause_used",
                "opponent_toxic_stage",
                opp_side,
                self_side,
                opp_engine,
            ),
        ]
        .into_iter()
        .enumerate()
        {
            let other_engine = 1 - engine_side;
            let leaf_sleepers = nonrest_sleepers(other);
            let root_sleepers = self.root_snapshot[other_engine].nonrest_sleepers;
            // Clause ENGAGES when the opponent gains a non-Rest sleeper and
            // RELEASES when the sleeper leaves play (faint/wake) — the
            // ledger's holder semantics (leaf-vs-reality double-KO repro).
            let clause = if leaf_sleepers > root_sleepers {
                true
            } else if leaf_sleepers < root_sleepers {
                false
            } else {
                root_sleep_clause[index]
            };
            md.insert(key_sc.into(), json!(clause));
            let active = side.get_active_immutable();
            let mut stage = meta.toxic[engine_side];
            if stage > 0 && (active.hp <= 0 || active.status != PokemonStatus::TOXIC) {
                stage = 0;
            }
            md.insert(key_tox.into(), json!(stage));
        }
        md.insert(
            "self_side_condition_counts".into(),
            side_condition_counts(self_side),
        );
        md.insert(
            "opponent_side_condition_counts".into(),
            side_condition_counts(opp_side),
        );
        md.insert("self_active_boosts".into(), boosts_value(self_side));
        md.insert("opponent_active_boosts".into(), boosts_value(opp_side));
        md.insert(
            "self_active_volatiles".into(),
            json!(tracked_volatiles(self_side, opp_side)),
        );
        md.insert(
            "opponent_active_volatiles".into(),
            json!(tracked_volatiles(opp_side, self_side)),
        );

        // --- team conditions + active flags ---
        for (key, engine_side, side) in [
            ("self_team", self_engine, self_side),
            ("opponent_team", opp_engine, opp_side),
        ] {
            let active_party = active_index_usize(side);
            let mons: Vec<_> = side.pokemon.into_iter().collect();
            if let Some(team) = md.get_mut(key).and_then(Value::as_array_mut) {
                for entry in team.iter_mut() {
                    let Some(obj) = entry.as_object_mut() else { continue };
                    let species = obj
                        .get("species")
                        .and_then(Value::as_str)
                        .unwrap_or_default();
                    let Some(party) = self.party_index(engine_side, &normalize_identifier(species))
                    else {
                        continue;
                    };
                    let Some(p) = mons.get(party) else { continue };
                    // Evolve-on-change: the recorded root condition string is
                    // authoritative (parser surface) until the engine actually
                    // moves this mon's hp/status during a branch.
                    let snapshot = self.root_snapshot[engine_side].mons.get(party);
                    let changed = snapshot
                        .map(|s| s.hp != p.hp || s.status != status_code(p.status))
                        .unwrap_or(true);
                    if changed {
                        let condition = condition_string(p.hp, p.maxhp, p.status);
                        obj.insert("condition".into(), json!(condition));
                        if p.hp <= 0 {
                            // A fainted mon's request condition is "0 fnt" —
                            // no max HP to derive the actual-HP entry from
                            // (`_max_hp_from_condition`); the five request
                            // stats remain. Drop only the hp key.
                            if let Some(stats) =
                                obj.get_mut("stats").and_then(Value::as_object_mut)
                            {
                                stats.remove("hp");
                            }
                        }
                    }
                    if let Some(source) =
                        changed_live_type_source(snapshot.map(|value| value.types), p.types)
                    {
                        obj.insert("live_type_source".into(), json!(source));
                    }
                    obj.insert("active".into(), json!(party == active_party));
                }
            }
        }

        // --- belief-ledger evolution (exact-state fields only; belief FACTS
        //     are world-constants and stay untouched) ---
        // Self-team display names (for synthesizing a fresh SELF ledger
        // entry when a first-time-active mon has none — the self side is
        // fully known, so ledger membership growth is NOT epistemic).
        let self_display: Vec<(String, String)> = md
            .get("self_team")
            .and_then(Value::as_array)
            .map(|team| {
                team.iter()
                    .filter_map(|entry| {
                        let species = entry.get("species").and_then(Value::as_str)?;
                        Some((normalize_identifier(species), species.to_string()))
                    })
                    .collect()
            })
            .unwrap_or_default();
        if let Some(belief) = md.get_mut("belief_view").and_then(Value::as_object_mut) {
            for (key, engine_side, side, is_self) in [
                ("self_pokemon", self_engine, self_side, true),
                ("opponent_pokemon", opp_engine, opp_side, false),
            ] {
                let mons: Vec<_> = side.pokemon.into_iter().collect();
                let active_party = active_index_usize(side);
                let Some(list) = belief.get_mut(key).and_then(Value::as_array_mut) else {
                    continue;
                };
                let mut active_covered = false;
                for entry in list.iter_mut() {
                    let Some(obj) = entry.as_object_mut() else { continue };
                    let species = obj
                        .get("species")
                        .and_then(Value::as_str)
                        .unwrap_or_default();
                    let Some(party) = self.party_index(engine_side, &normalize_identifier(species))
                    else {
                        continue;
                    };
                    let Some(p) = mons.get(party) else { continue };
                    let snapshot = self.root_snapshot[engine_side]
                        .mons
                        .get(party)
                        .cloned()
                        .unwrap_or_default();
                    obj.insert("active".into(), json!(party == active_party));
                    // Evolve-on-change: the root LEDGER values (condition,
                    // status, sleep bookkeeping) are authoritative until the
                    // engine moves this mon's hp/status during a branch — the
                    // ledger legitimately holds conventions the payload-built
                    // engine world cannot see (fainted mons keep their last
                    // status; rest bookkeeping survives approximate seeding;
                    // recorded ledger/payload skews stay as recorded).
                    let engine_status = status_code(p.status);
                    let changed =
                        snapshot.hp != p.hp || snapshot.status != engine_status;
                    if changed {
                        let condition = condition_string(p.hp, p.maxhp, p.status);
                        obj.insert("condition".into(), json!(condition));
                        // A mon fainting during the branch keeps its real
                        // engine-side status (ledger convention: last status
                        // is retained on faint).
                        if p.hp > 0 || engine_status.is_some() {
                            obj.insert(
                                "status".into(),
                                match engine_status {
                                    Some(code) => json!(code),
                                    None => Value::Null,
                                },
                            );
                        }
                        let root_rest = obj
                            .get("rest_sleep")
                            .and_then(Value::as_bool)
                            .unwrap_or(false);
                        let rest_now = p.rest_turns > 0
                            || (root_rest && p.status == PokemonStatus::SLEEP);
                        obj.insert("rest_sleep".into(), json!(rest_now));
                    }
                    // Sleep counting is LINE-driven and PER-MON (belief.py:
                    // "observed |cant …|slp turns since the status landed"):
                    // a fresh |-status|slp restarts the count at 0, each
                    // |cant ..|slp adds one — even when the sleeper later
                    // faints or switches out. Root sleepers keep their
                    // ledger base.
                    if let Some((started, count)) =
                        meta.sleep.get(&(engine_side, species_key(obj)))
                    {
                        let base = if *started {
                            0
                        } else {
                            obj.get("sleep_turns").and_then(Value::as_i64).unwrap_or(0)
                        };
                        obj.insert("sleep_turns".into(), json!(base + count));
                    }
                    // Turns-active (review F4): the ledger counter is a
                    // per-stint count — reset on the side's switch lines,
                    // +1 per |turn| line while active — replayed over the
                    // synthesized lines by the line-driven metadata
                    // (`evolve_leaf_meta`), exactly the parser's rules.
                    if party == active_party {
                        obj.insert("turns_active".into(), json!(meta.stint[engine_side]));
                        active_covered = true;
                    }
                    if !is_self {
                        rewrite_move_uses(obj, engine_side, meta);
                    }
                }
                if is_self && !active_covered {
                    // First-time-active self mon (e.g. the replacement after
                    // a faint): production's ledger grows an entry for it —
                    // synthesize the exact-state fields the encoder reads.
                    if let (Some(p), Some(key_str)) = (
                        mons.get(active_party),
                        self.species_keys[engine_side].get(active_party),
                    ) {
                        let display = self_display
                            .iter()
                            .find(|(k, _)| k == key_str)
                            .map(|(_, d)| d.clone())
                            .unwrap_or_else(|| key_str.clone());
                        let sleep_turns = meta
                            .sleep
                            .get(&(engine_side, key_str.clone()))
                            .map(|(_, count)| *count)
                            .unwrap_or(0);
                        list.push(json!({
                            "species": display,
                            "condition": condition_string(p.hp, p.maxhp, p.status),
                            "status": status_code(p.status),
                            "active": true,
                            "turns_active": meta.stint[engine_side],
                            "sleep_turns": sleep_turns,
                            "rest_sleep": p.rest_turns > 0,
                        }));
                    }
                }
            }
        }

        // --- action candidates + legal mask (engine option surface) ---
        let (s1_options, s2_options) = if engine_authoritative {
            state.root_get_all_options()
        } else {
            state.get_all_options()
        };
        let self_options = if self_is_p1 { &s1_options } else { &s2_options };
        let self_team_order = md
            .get("self_team")
            .and_then(Value::as_array)
            .map(|team| {
                team.iter()
                    .map(|entry| {
                        (
                            normalize_identifier(
                                entry.get("species").and_then(Value::as_str).unwrap_or(""),
                            ),
                            entry.get("active").and_then(Value::as_bool).unwrap_or(false),
                        )
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let (candidates, payload_moves) = self.action_surface(
            self_side,
            self_engine,
            self_options,
            &self_team_order,
            self_force_switch,
            meta,
            engine_authoritative,
        )?;
        md.insert("action_candidates".into(), candidates);

        // --- public materialization scalars the encoder reads ---
        let pm = row
            .get_mut("public_materialization")
            .and_then(Value::as_object_mut)
            .ok_or_else(|| err("row inputs missing public_materialization object"))?;
        pm.insert("turn".into(), json!(turn));
        pm.insert("selfActiveMoves".into(), payload_moves);
        // Timed side-condition set turns: ROOT-set conditions keep their
        // recorded set turn (remaining = leaf turn − set turn keeps ticking
        // correctly through simulated turns); conditions SET or ENDED during
        // the branch replay the parser's own bookkeeping
        // (`_update_timed_side_conditions`): a |-sidestart| records the turn
        // it happened on (root turn + completed |turn| lines at that point,
        // tracked by the line-driven metadata), a |-sideend| pops the entry.
        let (self_slot_key, opp_slot_key) = if self_is_p1 { ("p1", "p2") } else { ("p2", "p1") };
        if let Some(sides) = pm.get_mut("sides").and_then(Value::as_object_mut) {
            for (slot, side, engine_side) in [
                (self_slot_key, self_side, self_engine),
                (opp_slot_key, opp_side, opp_engine),
            ] {
                if let Some(side_obj) = sides.get_mut(slot).and_then(Value::as_object_mut) {
                    side_obj.insert("sideConditions".into(), side_condition_counts(side));
                    let branch_sets: Vec<(&String, &Option<i64>)> = meta
                        .side_condition_sets
                        .iter()
                        .filter(|((s, _), _)| *s == engine_side)
                        .map(|((_, condition), delta)| (condition, delta))
                        .collect();
                    if !branch_sets.is_empty() {
                        if !side_obj.contains_key("sideConditionSetTurns") {
                            side_obj.insert("sideConditionSetTurns".into(), json!({}));
                        }
                        if let Some(set_turns) = side_obj
                            .get_mut("sideConditionSetTurns")
                            .and_then(Value::as_object_mut)
                        {
                            for (condition, delta) in branch_sets {
                                match delta {
                                    Some(delta) => {
                                        set_turns.insert(
                                            condition.clone(),
                                            json!(self.root_turn + delta),
                                        );
                                    }
                                    None => {
                                        set_turns.remove(condition);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok(row)
    }

    /// Rebuild `action_candidates` + `pm.selfActiveMoves` from the engine's
    /// own option surface at this state (the leaf has no Showdown request;
    /// the engine IS the request authority on the search path).
    #[allow(clippy::too_many_arguments)]
    fn action_surface(
        &self,
        self_side: &Side,
        self_engine: usize,
        self_options: &[MoveChoice],
        self_team_order: &[(String, bool)],
        force_switch_shape: bool,
        meta: &LeafMeta,
        engine_authoritative: bool,
    ) -> PyResult<(Value, Value)> {
        let action_count = self.tables.layout_action_count();
        let move_action_count = self.tables.layout_move_action_count();
        // See `leaf_row_inputs`: the widening below is a correction for
        // CONSTRUCTED worlds only.
        let fresh_switch_in = meta.fresh_active[self_engine] && !engine_authoritative;

        // Engine move surface of the active mon, engine slot order. A
        // recharging active (MUSTRECHARGE volatile) presents the production
        // request shape instead: a single PP-less "recharge" pseudo-move,
        // forced (no switching) — the engine's own option surface is a bare
        // `None` which carries no request shape.
        let active = self_side.get_active_immutable();
        let recharging = self_side
            .volatile_statuses
            .contains(&PokemonVolatileStatus::MUSTRECHARGE);
        // A fresh switch-in cannot be move-restricted: choice locks reset on
        // switch, but the world constructor seeds benched mons with their
        // LAST STINT's cached disabled bits (the payload caches per-mon move
        // state) and the engine never re-enables them on a branch switch —
        // present the request semantics instead (leaf-vs-reality repro:
        // Choice-Band Nidoking fresh switch-in shows all four moves legal).
        //
        // PP is LINE-REPLAYED, not engine-read (review F3): the engine only
        // emits DecrementPP below 10 PP, so engine PP is root-frozen above
        // 10. The request's own deduction rules are replayed over the
        // branch's |move| lines (`LeafMeta::move_charges` — the same rules
        // the production request follows: called moves charge the caller,
        // locked continuations and Struggle charge nothing, opposing
        // Pressure doubles foe-targeted charges) against the ROOT snapshot's
        // exact per-mon PP (self PP is request-seeded at world construction).
        let active_party = active_index_usize(self_side);
        let active_key = self.species_keys[self_engine]
            .get(active_party)
            .cloned()
            .unwrap_or_default();
        let root_pp = self.root_snapshot[self_engine].mons.get(active_party);
        let mut engine_moves: Vec<(String, bool, i64)> = Vec::new(); // (showdown id, disabled, pp)
        if !recharging {
            for mv in active.moves.into_iter() {
                let engine_id = format!("{:?}", mv.id).to_lowercase();
                if engine_id == "none" {
                    continue;
                }
                let disabled = if fresh_switch_in { false } else { mv.disabled };
                let sd_id = showdown_move_id(&engine_id);
                let charged = meta
                    .move_charges
                    .get(&(self_engine, active_key.clone(), sd_id.clone()))
                    .copied()
                    .unwrap_or(0);
                // Base PP: the ROOT ACTIVE's engine PP is request-seeded and
                // exact. A mon ACTIVE AT THE LEAF but benched at the root
                // carries request-history-cached PP that is stale by its
                // last stint's FINAL action (requests are pre-action), so
                // its base is the belief ledger's public charging count
                // (maxpp − ledger uses — the same |move|-line rules the
                // request follows).
                // SELF SIDE ONLY. `self_ledger_uses` is keyed by the self
                // team's species keys (belief_view.self_pokemon), so an
                // opponent-side lookup would MISS rather than fail, yielding
                // `max - 0` = full PP and silently overriding the (correct)
                // root-snapshot base with a too-generous one. The opponent has
                // no public charging ledger, so it must fall through to
                // root_base -- see `opponent_action_map`.
                let ledger_side_is_self = self_engine == self.engine_side_index(true);
                let ledger_base = if ledger_side_is_self
                    && active_party != self.root_active_party[self_engine]
                {
                    self.tables.move_max_pp(&sd_id).filter(|max| *max > 0).map(|max| {
                        let uses = self
                            .self_ledger_uses
                            .get(&(active_key.clone(), sd_id.clone()))
                            .copied()
                            .unwrap_or(0);
                        (max - uses).max(0)
                    })
                } else {
                    None
                };
                let root_base = root_pp.and_then(|snapshot| {
                    snapshot
                        .pp
                        .iter()
                        .find(|(id, _)| *id == sd_id)
                        .map(|(_, pp)| *pp as i64)
                });
                let pp = match ledger_base.or(root_base) {
                    Some(base) => (base - charged).max(0),
                    None => (mv.pp as i64).max(0),
                };
                engine_moves.push((sd_id, disabled, pp));
            }
        }

        // Legal move indices + legal switch species from the option surface.
        let mut legal_moves: Vec<usize> = Vec::new();
        let mut legal_switch_keys: Vec<String> = Vec::new();
        for option in self_options {
            match option {
                MoveChoice::Move(index) => {
                    legal_moves.push(index.serialize().parse::<usize>().unwrap_or(0))
                }
                MoveChoice::Switch(index) => {
                    let party = index.serialize().parse::<usize>().unwrap_or(0);
                    // `self_engine`, NOT engine_side_index(true). This function
                    // is seat-generic and is called for the opponent too;
                    // hardcoding the self side here resolved the OPPONENT's
                    // switch options against the SELF team's species keys,
                    // which then failed to match `self_team_order` below --
                    // silently mapping every opponent switch to None (node
                    // falls back to uniform, so the cell measures nothing), or
                    // worse, matching a same-species mon on the other team and
                    // binding the arm to the wrong physical Pokemon.
                    if let Some(key) = self.species_keys[self_engine].get(party) {
                        legal_switch_keys.push(key.clone());
                    }
                }
                MoveChoice::None => {}
            }
        }

        let mut candidates: Vec<Value> = Vec::new();
        let mut payload_moves: Vec<Value> = Vec::new();
        if recharging && !force_switch_shape {
            // Production recharge request: one legal, PP-less "recharge"
            // move in slot 1; no other moves; switching disallowed.
            candidates.push(json!({
                "action_index": 0,
                "kind": "move",
                "legal": active.hp > 0,
                "move_slot": 1,
                "move_id": "recharge",
                "move_name": "recharge",
                "disabled": false,
            }));
            payload_moves.push(json!({"id": "recharge", "disabled": false}));
            for slot in 1..move_action_count {
                candidates.push(json!({
                    "action_index": slot,
                    "kind": "move",
                    "legal": false,
                    "move_slot": slot + 1,
                    "move_id": format!("slot{}", slot + 1),
                    "move_name": format!("slot:{}", slot + 1),
                    "disabled": true,
                }));
            }
            for switch_slot in 0..(action_count - move_action_count) {
                candidates.push(json!({
                    "action_index": move_action_count + switch_slot,
                    "kind": "switch",
                    "legal": false,
                    "switch_slot": switch_slot + 1,
                    "team_index": Value::Null,
                }));
            }
            return Ok((Value::Array(candidates), Value::Array(payload_moves)));
        }
        let moves_present = !force_switch_shape;
        for slot in 0..move_action_count {
            let entry = if moves_present { engine_moves.get(slot) } else { None };
            match entry {
                Some((move_id, disabled, pp)) => {
                    let legal = if fresh_switch_in {
                        *pp > 0 && active.hp > 0
                    } else {
                        legal_moves.contains(&slot) && active.hp > 0
                    };
                    candidates.push(json!({
                        "action_index": slot,
                        "kind": "move",
                        "legal": legal,
                        "move_slot": slot + 1,
                        "move_id": normalize_identifier(move_id),
                        "move_name": move_id,
                        "disabled": *disabled,
                    }));
                    let max_pp = self
                        .tables
                        .move_max_pp(move_id)
                        .filter(|max| *max > 0);
                    match max_pp {
                        Some(max) => payload_moves.push(json!({
                            "id": move_id,
                            "pp": *pp,
                            "maxpp": max,
                            "disabled": *disabled,
                        })),
                        None => payload_moves.push(json!({
                            "id": move_id,
                            "disabled": *disabled,
                        })),
                    }
                }
                None => {
                    candidates.push(json!({
                        "action_index": slot,
                        "kind": "move",
                        "legal": false,
                        "move_slot": slot + 1,
                        "move_id": format!("slot{}", slot + 1),
                        "move_name": format!("slot:{}", slot + 1),
                        "disabled": true,
                    }));
                }
            }
        }

        // Switch candidates: canonical map over the (rewritten) md team order
        // — non-active members in team order, exactly production's
        // `canonical_switch_action_map`.
        let active_team_index = self_team_order.iter().position(|(_, active)| *active);
        let switch_targets: Vec<usize> = match active_team_index {
            Some(active_index) if self_team_order.len() >= 2 => (0..self_team_order.len())
                .filter(|index| *index != active_index)
                .collect(),
            // Force-switch with the active fainted: production requests keep
            // the fainted mon at its team slot, so the canonical map still
            // excludes it. When no row is active (unmapped engine active),
            // fall back to team order.
            _ => (0..self_team_order.len()).collect(),
        };
        for switch_slot in 0..(action_count - move_action_count) {
            let action_index = move_action_count + switch_slot;
            let team_index = switch_targets.get(switch_slot).copied();
            let legal = team_index
                .map(|index| {
                    self_team_order
                        .get(index)
                        .map(|(key, _)| legal_switch_keys.iter().any(|k| k == key))
                        .unwrap_or(false)
                })
                .unwrap_or(false);
            candidates.push(json!({
                "action_index": action_index,
                "kind": "switch",
                "legal": legal,
                "switch_slot": switch_slot + 1,
                "team_index": team_index,
            }));
        }

        Ok((Value::Array(candidates), Value::Array(payload_moves)))
    }

    /// See [`LeafContext::leaf_row_inputs`] for `engine_authoritative`.
    pub(crate) fn encode_leaf(
        &self,
        state: &State,
        fold: &FoldStateInner,
        turn: i64,
        self_order: Option<&[String]>,
        meta: Option<&LeafMeta>,
        engine_authoritative: bool,
    ) -> PyResult<EncodedArrays> {
        // tensor_s is 75% of encode, so split its three parts: building the row
        // inputs from engine state, materializing the fold's derived products
        // (transition/action token tails, rebuilt per leaf), and writing the
        // arrays. Timers are process-global because encode_leaf is called from
        // the leaf-pricing closure, which has no per-search context.
        let t0 = std::time::Instant::now();
        let row = self.leaf_row_inputs(state, turn, self_order, meta, engine_authoritative)?;
        ROW_INPUT_NANOS.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
        let t1 = std::time::Instant::now();
        let products = fold.products();
        PRODUCTS_NANOS.fetch_add(t1.elapsed().as_nanos() as u64, Ordering::Relaxed);
        let t2 = std::time::Instant::now();
        let encoded = encode_row_value(&self.tables, &row, Some(&products));
        ROW_WRITE_NANOS.fetch_add(t2.elapsed().as_nanos() as u64, Ordering::Relaxed);
        encoded
    }

    /// True when the acting seat is engine side one.
    pub(crate) fn self_is_side_one(&self) -> bool {
        self.self_is_p1
    }

    /// Map each of the acting seat's engine options to its action index in
    /// the observation's action block (schema v1: `layout_action_count`
    /// actions), or `None` when the option has no legal action-block slot.
    ///
    /// The correspondence is DERIVED from `action_surface` — the exact
    /// candidate/legal-mask builder the leaf encoder writes — never
    /// re-implemented: move options match the candidate at their engine move
    /// slot, switch options match the switch candidate whose (evolved) team
    /// position carries their species, and a recharge-shape `None` option
    /// matches the production "recharge" pseudo-move at action index 0.
    /// `options` must be the option list of the DECISION NODE this map is for
    /// (`root_get_all_options` at the root, `get_all_options` at interior
    /// nodes — same state, same order).
    pub(crate) fn self_action_map(
        &self,
        state: &State,
        options: &[MoveChoice],
        self_order: Option<&[String]>,
        meta: Option<&LeafMeta>,
        engine_authoritative: bool,
    ) -> PyResult<Vec<Option<usize>>> {
        self.seat_action_map(state, options, self_order, meta, engine_authoritative, true)
    }

    /// The OPPONENT seat's option list mapped onto action-block slots.
    ///
    /// Exists so the model's `opponent_action_logits` head can be gathered onto
    /// the arms the opponent actually owns. Orientation (the #937 lesson):
    /// priors are per-seat action distributions applied to the seat that owns
    /// the actions and are NEVER reflected — only *values* flip at the seat
    /// boundary. So this maps the opponent's own `get_all_options()` through
    /// the opponent's own action block; there is no `1-x` and no index remap
    /// through the self block anywhere in this path.
    ///
    /// Two conventions differ from the self side, both forced by the opponent
    /// being BELIEF state rather than known state:
    ///
    /// * **Display order.** Same shape as the self side, different base. The
    ///   root base is the sampled world's ENGINE PARTY order (the only base
    ///   that resolves engine `Switch(party)` indices), and it is then evolved
    ///   through the branch's switches by `evolve_self_order` with
    ///   `opponent_prefix()`, exactly as the self order is. Measured on the
    ///   golden corpus, that reproduces the opponent seat's own request order
    ///   — which is the head's training label space
    ///   (`rollout.py::_opponent_action_index`). Two nearby wrong answers,
    ///   both tried and both measured: `md["opponent_team"]` is the partial
    ///   belief view and resolves almost nothing, and the unevolved party
    ///   order is correct only until the opponent's first switch.
    /// * **PP base.** The self side can correct a switched-in mon's stale
    ///   cached PP from the public charging ledger (`self_ledger_uses`). There
    ///   is no such ledger for the opponent, so its base falls back to the
    ///   sampled world's own root-snapshot PP. A disagreement can only make an
    ///   engine-offered option fail to find a legal action slot, which returns
    ///   `None` for that option and leaves the whole node uniform — the
    ///   existing node-level fallback, i.e. it fails safe rather than shaping
    ///   priors from a wrong surface.
    pub(crate) fn opponent_action_map(
        &self,
        state: &State,
        options: &[MoveChoice],
        opponent_order: Option<&[String]>,
        meta: Option<&LeafMeta>,
        engine_authoritative: bool,
    ) -> PyResult<Vec<Option<usize>>> {
        self.seat_action_map(
            state,
            options,
            opponent_order,
            meta,
            engine_authoritative,
            false,
        )
    }

    /// The opponent's ROOT display order: the sampled world's engine party
    /// order. Always six entries and always resolves engine `Switch(party)`
    /// indices, unlike `md["opponent_team"]`, which is the partial belief view.
    /// Callers evolve this through a branch's switches with
    /// [`evolve_self_order`] and `opponent_prefix()`.
    pub(crate) fn root_opponent_order(&self) -> Vec<String> {
        // ACTIVE-FIRST, not the raw packed party order. The head's label space
        // is that seat's own Showdown request order, which keeps the active at
        // slot 0 and accumulates a slot-0 swap on every switch-in; the self
        // side gets this free because `root_self_order` comes from
        // `md["self_team"]`, the real request. The opponent has no request, so
        // the packed order must be corrected the same way before
        // `evolve_self_order` layers further swaps on top of it.
        //
        // Measured on the golden corpus: without this swap the crate's switch
        // slots disagree with `rollout.py::_opponent_action_index` on every row
        // whose opponent has already switched -- i.e. from the opponent's first
        // switch onward, which is most of a gen3 randbat. Applying the swap
        // reproduces the label order exactly.
        // Prefer the caller's explicit order. The fallback below -- packed
        // party order with the active swapped to slot 0 -- reproduces the
        // request order only while the opponent has made AT MOST ONE
        // switch-in, and is transposed from the second onward. Four review
        // rounds landed on that; it is kept only so ad-hoc callers that cannot
        // supply the order degrade to a documented approximation rather than
        // to nothing.
        if !self.root_opponent_request_order.is_empty() {
            return self.root_opponent_request_order.clone();
        }
        let engine_side = self.engine_side_index(false);
        let mut order = self.species_keys[engine_side].clone();
        let active = self.root_active_party[engine_side];
        if active < order.len() && active != 0 {
            order.swap(0, active);
        }
        order
    }

    pub(crate) fn opponent_prefix(&self) -> &'static str {
        if self.self_is_p1 {
            "p2"
        } else {
            "p1"
        }
    }

    /// Seat-generic core of the action map. `slot_is_self` picks which seat's
    /// side, engine index, party order and snapshot are read; every lookup
    /// below is already indexed by engine side, so the two seats differ only in
    /// that index and in the display-order convention documented above.
    fn seat_action_map(
        &self,
        state: &State,
        options: &[MoveChoice],
        self_order: Option<&[String]>,
        meta: Option<&LeafMeta>,
        engine_authoritative: bool,
        slot_is_self: bool,
    ) -> PyResult<Vec<Option<usize>>> {
        let meta = meta.unwrap_or(&self.root_meta);
        let side_is_p1 = if slot_is_self { self.self_is_p1 } else { !self.self_is_p1 };
        let self_side = side_ref_for(state, side_is_p1);
        let self_engine = self.engine_side_index(slot_is_self);
        let force_switch_shape =
            self_side.force_switch || self_side.get_active_immutable().hp <= 0;
        let recharging = self_side
            .volatile_statuses
            .contains(&PokemonVolatileStatus::MUSTRECHARGE);
        // The (evolved) self-team display order with active flags — the same
        // derivation leaf_row_inputs writes into the md team before
        // action_surface reads it back.
        //
        // The opponent uses the same shape: the sampled world's engine party
        // order at the root, EVOLVED through the branch's switches by the
        // caller. Both halves matter. The engine party order is the only base
        // that resolves engine `Switch(party)` indices -- `md["opponent_team"]`
        // is the partial belief view and resolves almost nothing. But leaving
        // it UNEVOLVED is also wrong: measured against the head's training
        // label (`rollout.py` `_opponent_action_index`, that seat's own
        // request-order action block), the two agree until the opponent's
        // first switch and are rotated by one from then on -- which in a gen3
        // randbat is most of the game, and permutes every opponent switch
        // prior.
        let root_opponent: Vec<String> =
            if slot_is_self { Vec::new() } else { self.root_opponent_order() };
        let order: &[String] = if slot_is_self {
            self_order.unwrap_or(&self.root_self_order)
        } else {
            self_order.unwrap_or(&root_opponent)
        };
        let active_party = active_index_usize(self_side);
        let team_flags: Vec<(String, bool)> = order
            .iter()
            .map(|key| {
                (
                    key.clone(),
                    self.party_index(self_engine, key) == Some(active_party),
                )
            })
            .collect();
        let (candidates, _) = self.action_surface(
            self_side,
            self_engine,
            options,
            &team_flags,
            force_switch_shape,
            meta,
            engine_authoritative,
        )?;
        let candidates = candidates.as_array().expect("action_surface returns an array");
        let legal_action_index = |predicate: &dyn Fn(&Map<String, Value>) -> bool| {
            candidates.iter().find_map(|candidate| {
                let obj = candidate.as_object()?;
                if !obj.get("legal").and_then(Value::as_bool).unwrap_or(false) {
                    return None;
                }
                if predicate(obj) {
                    obj.get("action_index").and_then(Value::as_u64).map(|v| v as usize)
                } else {
                    None
                }
            })
        };
        let mut map: Vec<Option<usize>> = Vec::with_capacity(options.len());
        for option in options {
            let index = match option {
                MoveChoice::Move(engine_index) => {
                    let slot = engine_index.serialize().parse::<usize>().unwrap_or(usize::MAX);
                    if recharging && !force_switch_shape {
                        None // recharge shape offers no real move candidates
                    } else {
                        legal_action_index(&|obj| {
                            obj.get("kind").and_then(Value::as_str) == Some("move")
                                && obj.get("move_slot").and_then(Value::as_u64)
                                    == Some(slot as u64 + 1)
                        })
                    }
                }
                MoveChoice::Switch(party) => {
                    let party = party.serialize().parse::<usize>().unwrap_or(usize::MAX);
                    match self.species_keys[self_engine].get(party) {
                        None => None,
                        Some(key) => legal_action_index(&|obj| {
                            obj.get("kind").and_then(Value::as_str) == Some("switch")
                                && obj
                                    .get("team_index")
                                    .and_then(Value::as_u64)
                                    .and_then(|team_index| team_flags.get(team_index as usize))
                                    .map(|(candidate_key, _)| candidate_key == key)
                                    .unwrap_or(false)
                        }),
                    }
                }
                MoveChoice::None => {
                    if recharging && !force_switch_shape {
                        // Production recharge request: one legal PP-less
                        // "recharge" pseudo-move at action index 0.
                        legal_action_index(&|obj| {
                            obj.get("move_id").and_then(Value::as_str) == Some("recharge")
                        })
                    } else {
                        None
                    }
                }
            };
            map.push(index);
        }
        Ok(map)
    }
}

/// Apply the self side's switch/drag lines to a display order: each switch
/// SWAPS the incoming mon with slot 0 — Showdown's exact `switchIn`
/// position semantics (sim/battle-actions.ts: `pokemon.position = pos;
/// side.pokemon[pos] = pokemon; side.pokemon[old.position] = old`). Species
/// come from the DETAILS field (nickname-proof).
pub(crate) fn evolve_self_order(
    order: &[String],
    lines: &[String],
    self_prefix: &str,
) -> Vec<String> {
    let mut order = order.to_vec();
    let switch_prefix = format!("|switch|{self_prefix}a: ");
    let drag_prefix = format!("|drag|{self_prefix}a: ");
    for line in lines {
        if !line.starts_with(&switch_prefix) && !line.starts_with(&drag_prefix) {
            continue;
        }
        let details = line.split('|').nth(3).unwrap_or("");
        let species = details.split(',').next().unwrap_or("").trim();
        let key = normalize_identifier(species);
        if key.is_empty() {
            continue;
        }
        if let Some(pos) = order.iter().position(|k| *k == key) {
            if pos != 0 {
                order.swap(0, pos);
            }
        }
    }
    order
}

/// Opponent `move_uses` evolution: root ledger uses + the branch's
/// LINE-REPLAYED PP charges for this mon (review F3: the engine only emits
/// DecrementPP below 10 PP, so engine PP deltas are root-frozen above 10;
/// the ledger's own counting rules — belief.py `_charge_move_use` — are
/// replayed over the synthesized |move| lines instead, Pressure
/// double-charges included).
fn rewrite_move_uses(obj: &mut Map<String, Value>, engine_side: usize, meta: &LeafMeta) {
    let mon_key = species_key(obj);
    let Some(uses) = obj.get_mut("move_uses").and_then(Value::as_array_mut) else {
        return;
    };
    for pair in uses.iter_mut() {
        let Some(items) = pair.as_array_mut() else { continue };
        if items.len() != 2 {
            continue;
        }
        let Some(move_id) = items[0].as_str().map(normalize_identifier) else {
            continue;
        };
        let root_uses = items[1].as_i64().unwrap_or(0);
        let charged = meta
            .move_charges
            .get(&(engine_side, mon_key.clone(), move_id))
            .copied()
            .unwrap_or(0);
        if charged > 0 {
            items[1] = json!(root_uses + charged);
        }
    }
}

fn boosts_value(side: &Side) -> Value {
    json!({
        "atk": side.attack_boost,
        "def": side.defense_boost,
        "spa": side.special_attack_boost,
        "spd": side.special_defense_boost,
        "spe": side.speed_boost,
        "accuracy": side.accuracy_boost,
        "evasion": side.evasion_boost,
    })
}

/// Active side-condition counts in the parser's id vocabulary (gen3 subset).
fn side_condition_counts(side: &Side) -> Value {
    let mut counts = Map::new();
    let sc = &side.side_conditions;
    for (id, value) in [
        ("spikes", sc.spikes as i64),
        ("reflect", sc.reflect as i64),
        ("lightscreen", sc.light_screen as i64),
        ("safeguard", sc.safeguard as i64),
        ("mist", sc.mist as i64),
    ] {
        if value > 0 {
            // Screens/safeguard/mist read as booleans downstream; spikes is a
            // layer count. The parser stores layer counts too.
            counts.insert(id.to_string(), json!(if id == "spikes" { value } else { 1 }));
        }
    }
    Value::Object(counts)
}

// ---------------------------------------------------------------------------
// PyO3 surface
// ---------------------------------------------------------------------------

/// Per-decision leaf encoder: constructed once at the root (tables + root row
/// inputs + party context + root engine state), then encodes leaf
/// observations from (leaf engine state, advanced fold state, leaf turn).
#[pyclass(name = "LeafEncoder", module = "pokezero_search")]
pub struct PyLeafEncoder {
    ctx: LeafContext,
}

#[pymethods]
impl PyLeafEncoder {
    #[new]
    fn new(
        tables_json: &str,
        root_inputs_json: &str,
        ctx_json: &str,
        root_state_str: &str,
    ) -> PyResult<Self> {
        let root_state = parse_state(root_state_str)?;
        Ok(PyLeafEncoder {
            ctx: LeafContext::new(tables_json, root_inputs_json, ctx_json, &root_state)?,
        })
    }

    /// A new encoder with different root inputs over the SAME parsed tables.
    ///
    /// Constructing an encoder is dominated by parsing the ~475 KB tables
    /// artifact (~3 ms, vs ~200 µs for an encode), so a caller that must
    /// re-root whenever its public-information surface changes — an
    /// environment learning which opponent Pokemon have been revealed — would
    /// otherwise spend most of its time re-parsing a constant. The tables are
    /// immutable after construction and are shared, not copied.
    fn rebased(
        &self,
        root_inputs_json: &str,
        ctx_json: &str,
        root_state_str: &str,
    ) -> PyResult<Self> {
        let root_state = parse_state(root_state_str)?;
        Ok(PyLeafEncoder {
            ctx: LeafContext::with_tables(
                Arc::clone(&self.ctx.tables),
                root_inputs_json,
                ctx_json,
                &root_state,
            )?,
        })
    }

    /// Encode a leaf observation: ENGINE-STATE columns from `state_str`,
    /// FOLD columns from `fold`, WORLD-CONSTANT columns from the root row
    /// inputs. `lines` are the branch's synthesized protocol lines from the
    /// root (they drive the self-team display order via Showdown's
    /// switch-swap semantics AND the line-driven metadata: toxic stages,
    /// active stints — None/empty keeps root values). At zero branches
    /// (`state_str` = the root state, `fold` = the root fold, `turn` = the
    /// root turn, no lines) this must reproduce the golden observation — the
    /// root-parity gate.
    ///
    /// `engine_authoritative = true` builds the action block from
    /// `root_get_all_options` instead of `get_all_options`. Search leaves are
    /// interior nodes and leave it `false` (the default keeps the search path
    /// byte-identical); an engine-as-environment caller, whose every encode is
    /// a real decision point, passes `true` so `force_trapped` and the
    /// slow-uturn/Baton-Pass shapes mask correctly.
    #[pyo3(signature = (state_str, fold, turn, lines = None, engine_authoritative = false))]
    fn encode_leaf(
        &self,
        py: Python<'_>,
        state_str: &str,
        fold: &PyFoldState,
        turn: i64,
        lines: Option<Vec<String>>,
        engine_authoritative: bool,
    ) -> PyResult<Py<PyDict>> {
        let state = parse_state(state_str)?;
        let (order, meta) = self.branch_context(lines.as_deref());
        let encoded = self.ctx.encode_leaf(
            &state,
            fold.inner(),
            turn,
            order.as_deref(),
            meta.as_ref(),
            engine_authoritative,
        )?;
        encoded_to_dict(py, &encoded)
    }

    /// The acting seat's option→action-index correspondence at a state: one
    /// `(display, action_index_or_None)` pair per engine option, in the
    /// engine's own option order (`root_get_all_options` when `root` is set —
    /// force-trapped / slow-uturn aware — else `get_all_options`). This is
    /// the exact map the model-prior wiring uses; the mapping-assertion test
    /// checks it against recorded request masks.
    /// `engine_authoritative` implies `root` and additionally drops the
    /// fresh-switch-in widening — see [`LeafContext::leaf_row_inputs`]. It
    /// must be set by callers that will submit the returned display back to
    /// the engine, so the map and the engine agree on what is playable; the
    /// prior-mapping gates compare against recorded SHOWDOWN request masks
    /// over constructed worlds and leave it `false`.
    #[pyo3(signature = (state_str, lines = None, root = false, engine_authoritative = false))]
    fn self_action_map(
        &self,
        state_str: &str,
        lines: Option<Vec<String>>,
        root: bool,
        engine_authoritative: bool,
    ) -> PyResult<Vec<(String, Option<usize>)>> {
        let state = parse_state(state_str)?;
        let (order, meta) = self.branch_context(lines.as_deref());
        let (s1_options, s2_options) = if root || engine_authoritative {
            state.root_get_all_options()
        } else {
            state.get_all_options()
        };
        let options = if self.ctx.self_is_side_one() {
            s1_options
        } else {
            s2_options
        };
        let side = if self.ctx.self_is_side_one() {
            &state.side_one
        } else {
            &state.side_two
        };
        let map = self.ctx.self_action_map(
            &state,
            &options,
            order.as_deref(),
            meta.as_ref(),
            engine_authoritative,
        )?;
        Ok(options
            .iter()
            .zip(map)
            .map(|(choice, index)| (crate::move_display(side, choice), index))
            .collect())
    }

    /// The OPPONENT seat's option→action-index correspondence, the mirror of
    /// `self_action_map`. Exposed so the orientation contract can be pinned
    /// from Python without a libtorch build: the opponent head must be
    /// gathered onto the arms the opponent owns, never reflected through the
    /// self block. See [`LeafContext::opponent_action_map`] for the two
    /// belief-state conventions (display order, PP base) that differ from the
    /// self side.
    #[pyo3(signature = (state_str, lines = None, root = false, engine_authoritative = false))]
    fn opponent_action_map(
        &self,
        state_str: &str,
        lines: Option<Vec<String>>,
        root: bool,
        engine_authoritative: bool,
    ) -> PyResult<Vec<(String, Option<usize>)>> {
        let state = parse_state(state_str)?;
        let (_order, meta) = self.branch_context(lines.as_deref());
        let (s1_options, s2_options) = if root || engine_authoritative {
            state.root_get_all_options()
        } else {
            state.get_all_options()
        };
        // The OPPONENT's options and the OPPONENT's side -- the whole point of
        // the mirror. `self_is_side_one()` selects the self seat, so both
        // picks below are inverted relative to `self_action_map`.
        let options = if self.ctx.self_is_side_one() {
            s2_options
        } else {
            s1_options
        };
        let side = if self.ctx.self_is_side_one() {
            &state.side_two
        } else {
            &state.side_one
        };
        // Evolve over the same branch lines the self side uses, with the
        // OPPONENT's protocol prefix.
        let opponent_order = lines.as_deref().map(|lines| {
            evolve_self_order(
                &self.ctx.root_opponent_order(),
                lines,
                self.ctx.opponent_prefix(),
            )
        });
        let map = self.ctx.opponent_action_map(
            &state,
            &options,
            opponent_order.as_deref(),
            meta.as_ref(),
            engine_authoritative,
        )?;
        Ok(options
            .iter()
            .zip(map)
            .map(|(choice, index)| (crate::move_display(side, choice), index))
            .collect())
    }

    /// The rewritten row-inputs JSON for a leaf state (divergence debugging:
    /// diff this against the root inputs to see exactly which state fields
    /// the engine recompute changed).
    #[pyo3(signature = (state_str, turn, lines = None, engine_authoritative = false))]
    fn leaf_inputs_json(
        &self,
        state_str: &str,
        turn: i64,
        lines: Option<Vec<String>>,
        engine_authoritative: bool,
    ) -> PyResult<String> {
        let state = parse_state(state_str)?;
        let (order, meta) = self.branch_context(lines.as_deref());
        let row =
            self.ctx
                .leaf_row_inputs(&state, turn, order.as_deref(), meta.as_ref(), engine_authoritative)?;
        serde_json::to_string(&row).map_err(|e| err(format!("serialize leaf inputs: {e}")))
    }
}

impl PyLeafEncoder {
    fn branch_context(
        &self,
        lines: Option<&[String]>,
    ) -> (Option<Vec<String>>, Option<LeafMeta>) {
        match lines {
            None => (None, None),
            Some(lines) => (
                Some(evolve_self_order(
                    self.ctx.root_self_order(),
                    lines,
                    self.ctx.self_prefix(),
                )),
                Some(evolve_leaf_meta(
                    self.ctx.root_meta(),
                    lines,
                    self.ctx.meta_ctx(),
                )),
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::events::{render_branch_events, EventContext};
    use poke_engine::choices::Choices;
    use poke_engine::engine::generate_instructions::generate_instructions_from_move_pair;
    use poke_engine::state::PokemonMoveIndex;

    fn lines(raw: &[&str]) -> Vec<String> {
        raw.iter().map(|s| s.to_string()).collect()
    }

    fn root_zero_attestation(p1: bool, p2: bool) -> Value {
        json!({
            "toxic_stage_zero_after_upkeep": {
                "p1": {
                    "proof": p1,
                    "pending": false,
                    "invalid": false,
                    "post_upkeep_window": false,
                },
                "p2": {
                    "proof": p2,
                    "pending": false,
                    "invalid": false,
                    "post_upkeep_window": false,
                },
            }
        })
    }

    #[test]
    fn root_zero_proof_requires_complete_exact_boolean_attestation() {
        for (field, value) in [
            ("proof", json!(1)),
            ("proof", json!(0)),
            ("proof", json!("true")),
            ("proof", Value::Null),
            ("proof", json!({"forged": true})),
            ("pending", json!(1)),
            ("pending", json!(0)),
            ("invalid", json!("false")),
            ("post_upkeep_window", json!({"forged": false})),
        ] {
            let mut ctx = root_zero_attestation(true, false);
            ctx["toxic_stage_zero_after_upkeep"]["p1"][field] = value;
            assert!(
                !root_toxic_zero_after_upkeep(&ctx)[0],
                "{field} must reject a non-boolean root attestation"
            );
        }

        for (field, value) in [("pending", json!(true)), ("invalid", json!(true))] {
            let mut ctx = root_zero_attestation(true, false);
            ctx["toxic_stage_zero_after_upkeep"]["p1"][field] = value;
            assert!(
                !root_toxic_zero_after_upkeep(&ctx)[0],
                "{field} must reject the wrong exact boolean"
            );
        }

        assert!(root_toxic_zero_after_upkeep(&root_zero_attestation(true, false))[0]);
        assert!(!root_toxic_zero_after_upkeep(&root_zero_attestation(true, false))[1]);
    }

    fn rendered_status_cure(choice: Choices) -> crate::events::RenderedEvents {
        let mut state = State::default();
        let active = state.side_one.get_active();
        active.maxhp = 200;
        active.hp = 100;
        active.status = PokemonStatus::TOXIC;
        active.replace_move(PokemonMoveIndex::M0, choice);
        let s1 = MoveChoice::Move(PokemonMoveIndex::M0);
        let s2 = MoveChoice::None;
        let branches = generate_instructions_from_move_pair(&mut state, &s1, &s2, false);
        assert_eq!(
            branches.len(),
            1,
            "expected deterministic status cure: {branches:?}"
        );
        render_branch_events(
            &mut state,
            &s1,
            &s2,
            &branches[0].instruction_list,
            false,
            &EventContext {
                species: [vec!["Rattata".to_string()], vec!["Chansey".to_string()]],
                turn: 1,
                hp_percent: [false, false],
            },
        )
    }

    #[test]
    fn live_type_source_tracks_in_tree_retypes_only_after_change() {
        let normal = (PokemonType::NORMAL, PokemonType::TYPELESS);
        let fire = (PokemonType::FIRE, PokemonType::TYPELESS);
        let water_flying = (PokemonType::WATER, PokemonType::FLYING);
        assert_eq!(changed_live_type_source(Some(normal), normal), None);
        assert_eq!(
            changed_live_type_source(Some(normal), fire).as_deref(),
            Some("type:FIRE")
        );
        assert_eq!(
            changed_live_type_source(Some(fire), water_flying).as_deref(),
            Some("type:WATER/FLYING")
        );
    }

    /// Review F1 repro shapes, replayed with the parser's own line rules.
    #[test]
    fn toxic_meta_fresh_apply_in_branch() {
        // arr69 shape: no toxic at root; Toxic lands in-branch and the turn
        // completes. Parser: |-status|tox -> 1, |turn| -> 2.
        let meta = evolve_leaf_meta(
            &LeafMeta::default(),
            &lines(&[
                "|move|p1a: Swampert|Toxic|p2a: Starmie",
                "|-status|p2a: Starmie|tox",
                "|upkeep",
                "|turn|32",
            ]),
            &LeafMetaCtx::default(),
        );
        assert_eq!(meta.toxic[1], 2);
        assert_eq!(meta.toxic[0], 0);
    }

    #[test]
    fn toxic_meta_reapply_after_switch() {
        // arr82 shape: toxic at root (stage 4); the mon switches out (reset)
        // and is re-poisoned; three completed turns later the stage is 4
        // again — never 5.
        let root = LeafMeta {
            toxic: [0, 4],
            stint: [0, 0],
            ..Default::default()
        };
        let meta = evolve_leaf_meta(
            &root,
            &lines(&[
                "|switch|p2a: Blissey|Blissey, L80, F|100/100",
                "|turn|20",
                "|switch|p2a: Starmie|Starmie, L77|68/227",
                "|-status|p2a: Starmie|tox",
                "|turn|21",
                "|turn|22",
                "|turn|23",
            ]),
            &LeafMetaCtx::default(),
        );
        assert_eq!(meta.toxic[1], 4);
        // Active tracking follows the switch DETAILS species.
        assert_eq!(meta.active[1], "starmie");
    }

    #[test]
    fn toxic_meta_rest_status_replacement_and_cures_clear_stage() {
        let root = LeafMeta {
            toxic: [0, 7],
            active_toxic: [false, true],
            ..Default::default()
        };
        for line in [
            "|-status|p2a: Starmie|slp|[from] move: Rest",
            "|-status|p2a: Starmie|par",
            "|-curestatus|p2a: Starmie|tox",
            "|-cureteam|p2a: Starmie|tox",
        ] {
            let meta = evolve_leaf_meta(&root, &lines(&[line]), &LeafMetaCtx::default());
            assert_eq!(meta.toxic[1], 0, "{line} must clear Toxic stage");
            assert!(!meta.active_toxic[1], "{line} must end active Toxic");
        }
    }

    #[test]
    fn rendered_status_cures_clear_toxic_meta_without_fake_protocol_lines() {
        let root = LeafMeta {
            toxic: [7, 0],
            active_toxic: [true, false],
            active_hp: [Some((100, 200)), None],
            ..Default::default()
        };

        let rest = rendered_status_cure(Choices::REST);
        assert!(
            rest.lines
                .iter()
                .any(|line| line.contains("|-heal|p1a: Rattata|200/200 slp|[silent]")),
            "Rest's production form carries sleep on its silent heal: {:?}",
            rest.lines
        );
        assert!(
            !rest
                .lines
                .iter()
                .any(|line| line.starts_with("|-status|p1a: Rattata|slp")),
            "the production renderer must not rely on a synthetic Rest status line: {:?}",
            rest.lines
        );
        let rest_meta = evolve_leaf_meta_with_status_transitions(
            &root,
            &rest.lines,
            &LeafMetaCtx::default(),
            &rest.active_status_transitions,
        );
        assert_eq!(rest_meta.toxic[0], 0);
        assert!(!rest_meta.active_toxic[0]);

        for choice in [Choices::REFRESH, Choices::HEALBELL] {
            let rendered = rendered_status_cure(choice);
            assert!(
                !rendered
                    .lines
                    .iter()
                    .any(|line| line.starts_with("|-curestatus|")),
                "the production renderer must not invent cure text: {:?}",
                rendered.lines
            );
            assert!(
                !rendered.active_status_transitions.is_empty(),
                "the leaf must receive the renderer-private cure transition"
            );
            let meta = evolve_leaf_meta_with_status_transitions(
                &root,
                &rendered.lines,
                &LeafMetaCtx::default(),
                &rendered.active_status_transitions,
            );
            assert_eq!(meta.toxic[0], 0, "{choice:?} must clear stale Toxic stage");
            assert!(
                !meta.active_toxic[0],
                "{choice:?} must clear Toxic provenance"
            );
        }
    }

    #[test]
    fn toxic_meta_switch_and_drag_reentry_residuals_recover_stage() {
        let root = LeafMeta {
            toxic: [0, 9],
            active_toxic: [false, true],
            active_hp: [None, Some((80, 240))],
            ..Default::default()
        };
        let switched = evolve_leaf_meta(
            &root,
            &lines(&[
                "|switch|p2a: Starmie|Starmie, L77|90/240 tox",
                "|-damage|p2a: Starmie|75/240|[from] psn",
            ]),
            &LeafMetaCtx::default(),
        );
        assert_eq!(switched.toxic[1], 1, "exact Toxic residual must recover stage one");
        let advanced = evolve_leaf_meta(
            &switched,
            &lines(&["|turn|21"]),
            &LeafMetaCtx::default(),
        );
        assert_eq!(advanced.toxic[1], 2);

        let ctx = LeafMetaCtx {
            hp_percent: [false, true],
            ..Default::default()
        };
        let dragged = evolve_leaf_meta(
            &root,
            &lines(&[
                "|drag|p2a: Starmie|Starmie, L77|90/100 tox",
                "|-damage|p2a: Starmie|85/100|[from] psn",
            ]),
            &ctx,
        );
        assert_eq!(dragged.toxic[1], 1, "reentry provenance must recover rounded stage one");

        let unproven = evolve_leaf_meta(
            &LeafMeta {
                toxic: [0, 0],
                active_toxic: [false, true],
                active_hp: [None, Some((90, 100))],
                ..Default::default()
            },
            &lines(&["|-damage|p2a: Starmie|85/100|[from] psn"]),
            &ctx,
        );
        assert_eq!(unproven.toxic[1], 0, "rounded residual without reentry provenance fails closed");

        for event in ["switch", "drag"] {
            let clean = evolve_leaf_meta(
                &root,
                &lines(&[
                    &format!("|{event}|p2a: Blissey|Blissey, L80, F|90/240"),
                    "|-damage|p2a: Blissey|75/240|[from] psn",
                ]),
                &LeafMetaCtx::default(),
            );
            assert_eq!(
                clean.toxic[1], 0,
                "{event} to a healthy mon must not resurrect Toxic"
            );
            assert!(
                !clean.active_toxic[1],
                "{event} must clear active Toxic provenance"
            );
            assert!(
                !clean.toxic_reentry_pending[1],
                "{event} clean entry has no reentry proof"
            );
        }
    }

    #[test]
    fn sanctioned_root_zero_proof_reaches_percent_leaf_for_both_seats() {
        for side in 0..2 {
            let mut root = LeafMeta {
                active_toxic: [side == 0, side == 1],
                active_hp: [Some((100, 100)), Some((100, 100))],
                ..Default::default()
            };
            let proof = root_toxic_zero_after_upkeep(&root_zero_attestation(side == 0, side == 1));
            seed_root_toxic_reentry_pending(&mut root, proof);
            assert!(root.toxic_reentry_pending[side], "root proof lost for side {side}");
            assert!(!root.toxic_reentry_pending[1 - side]);

            let mut ctx = LeafMetaCtx::default();
            ctx.hp_percent[side] = true;
            let ident = if side == 0 { "p1a: Replacement" } else { "p2a: Replacement" };
            let leaf = evolve_leaf_meta(
                &root,
                &lines(&[
                    &format!("|-damage|{ident}|94/100 tox|[from] psn"),
                    "|upkeep",
                    "|turn|2",
                ]),
                &ctx,
            );
            // The first rounded residual proves stage one; the following
            // parser turn boundary advances it to the next pending stage.
            assert_eq!(leaf.toxic[side], 2, "root-to-leaf recovery side {side}");
            assert!(!leaf.toxic_reentry_pending[side], "proof must be consumed");
        }
    }

    #[test]
    fn root_zero_proof_fails_closed_and_expires_on_lifecycle_transitions() {
        let mut root = LeafMeta {
            active_toxic: [true, false],
            active_hp: [Some((100, 100)), None],
            ..Default::default()
        };
        seed_root_toxic_reentry_pending(&mut root, root_toxic_zero_after_upkeep(&json!({})));
        let ctx = LeafMetaCtx {
            hp_percent: [true, false],
            ..Default::default()
        };
        let unproven = evolve_leaf_meta(
            &root,
            &lines(&["|-damage|p1a: Replacement|94/100 tox|[from] psn"]),
            &ctx,
        );
        assert_eq!(unproven.toxic[0], 0, "missing root proof must fail closed");

        for line in [
            "|-curestatus|p1a: Replacement|tox",
            "|switch|p1a: Other|Other, L80|100/100",
            "|faint|p1a: Replacement",
        ] {
            let mut proven = root.clone();
            seed_root_toxic_reentry_pending(
                &mut proven,
                root_toxic_zero_after_upkeep(&root_zero_attestation(true, false)),
            );
            let expired = evolve_leaf_meta(&proven, &lines(&[line]), &ctx);
            assert!(
                !expired.toxic_reentry_pending[0],
                "{line} must retire root zero proof"
            );
        }

        let mut forged = LeafMeta {
            toxic: [1, 0],
            active_toxic: [true, false],
            ..Default::default()
        };
        seed_root_toxic_reentry_pending(
            &mut forged,
            root_toxic_zero_after_upkeep(&root_zero_attestation(true, true)),
        );
        assert_eq!(forged.toxic_reentry_pending, [false, false]);
    }

    #[test]
    fn root_zero_proof_expires_at_first_missed_residual_opportunity() {
        for side in 0..2 {
            for boundary in ["|upkeep", "|turn|2"] {
                let mut root = LeafMeta {
                    active_toxic: [side == 0, side == 1],
                    active_hp: [Some((100, 100)), Some((100, 100))],
                    ..Default::default()
                };
                let proof =
                    root_toxic_zero_after_upkeep(&root_zero_attestation(side == 0, side == 1));
                seed_root_toxic_reentry_pending(&mut root, proof);
                let mut ctx = LeafMetaCtx::default();
                ctx.hp_percent[side] = true;
                let ident = if side == 0 { "p1a: Replacement" } else { "p2a: Replacement" };
                let expired = evolve_leaf_meta(
                    &root,
                    &lines(&[
                        boundary,
                        &format!("|-damage|{ident}|94/100 tox|[from] psn"),
                    ]),
                    &ctx,
                );
                assert!(
                    !expired.toxic_reentry_pending[side],
                    "{boundary} must retire side {side}'s root proof"
                );
                assert_eq!(
                    expired.toxic[side], 0,
                    "later rounded residual after {boundary} must not recover side {side}"
                );
            }
        }
    }

    #[test]
    fn toxic_meta_preserves_saturation_sentinel_sixteen() {
        let root = LeafMeta {
            toxic: [0, 15],
            active_toxic: [false, true],
            ..Default::default()
        };
        let meta = evolve_leaf_meta(
            &root,
            &lines(&["|upkeep", "|turn|32", "|turn|33"]),
            &LeafMetaCtx::default(),
        );
        assert_eq!(meta.toxic[1], 16);
    }

    /// toxic_stall repro: a faint-pending ply runs the ENGINE's end-of-turn
    /// tick but emits no |turn| line — the parser (and therefore the leaf)
    /// must NOT escalate.
    #[test]
    fn toxic_meta_faint_ply_does_not_escalate() {
        let root = LeafMeta {
            toxic: [0, 9],
            stint: [0, 0],
            ..Default::default()
        };
        let meta = evolve_leaf_meta(
            &root,
            &lines(&[
                "|move|p1a: Swampert|Earthquake|p2a: Starmie",
                "|-damage|p2a: Starmie|58/227 tox",
                "|-damage|p2a: Starmie|0 fnt|[from] psn",
                "|faint|p2a: Starmie",
                "|upkeep",
            ]),
            &LeafMetaCtx::default(),
        );
        assert_eq!(meta.toxic[1], 0);
    }

    /// Review F4: stint counting — +1 per |turn| line, reset on the side's
    /// own switch lines (Showdown `activeTurns = 0`).
    #[test]
    fn stint_meta_counts_turns_and_resets_on_switch() {
        let root = LeafMeta {
            toxic: [0, 0],
            stint: [3, 5],
            ..Default::default()
        };
        let meta = evolve_leaf_meta(
            &root,
            &lines(&[
                "|switch|p1a: Volbeat|Volbeat, L88, M|100/100",
                "|turn|10",
                "|turn|11",
            ]),
            &LeafMetaCtx::default(),
        );
        // p1 switched (reset) then two completed turns; p2 stayed in.
        assert_eq!(meta.stint[0], 2);
        assert_eq!(meta.stint[1], 7);
        assert_eq!(meta.turns_seen, 2);
    }

    /// Review F3 fix: PP charges replay the PARSER's rules over |move| lines
    /// — per (side, mon, move), called moves / locked continuations /
    /// Struggle exempt, opposing Pressure doubles foe-targeted charges.
    #[test]
    fn move_charges_replay_parser_rules() {
        let mut root = LeafMeta::default();
        root.active = ["swampert".to_string(), "zapdos".to_string()];
        let ctx = LeafMetaCtx {
            pressure: [Vec::new(), vec!["zapdos".to_string()]],
            ..Default::default()
        };
        let meta = evolve_leaf_meta(
            &root,
            &lines(&[
                // Foe-targeted into a Pressure active: 2.
                "|move|p1a: Swampert|icebeam|p2a: Zapdos",
                // Self-targeted: never pressured (1).
                "|move|p1a: Swampert|protect|p1a: Swampert",
                // Opponent's own move into us: no Pressure on our side (1).
                "|move|p2a: Zapdos|thunderbolt|p1a: Swampert",
                // Pressure mon leaves; the same foe-targeted move now costs 1.
                "|switch|p2a: Blissey|Blissey, L80, F|100/100",
                "|move|p1a: Swampert|icebeam|p2a: Blissey",
                // Sleep Talk called execution charges nothing...
                "|move|p1a: Swampert|surf|p2a: Blissey|[from] Sleep Talk",
                // ...locked continuations charge nothing...
                "|move|p1a: Swampert|thrash|p2a: Blissey|[from]lockedmove",
                // ...and Struggle has no PP.
                "|move|p1a: Swampert|struggle|p2a: Blissey",
            ]),
            &ctx,
        );
        let charge = |side: usize, mon: &str, mv: &str| {
            meta.move_charges
                .get(&(side, mon.to_string(), mv.to_string()))
                .copied()
                .unwrap_or(0)
        };
        assert_eq!(charge(0, "swampert", "icebeam"), 3); // 2 (pressured) + 1
        assert_eq!(charge(0, "swampert", "protect"), 1);
        assert_eq!(charge(1, "zapdos", "thunderbolt"), 1);
        assert_eq!(charge(0, "swampert", "surf"), 0);
        assert_eq!(charge(0, "swampert", "thrash"), 0);
        assert_eq!(charge(0, "swampert", "struggle"), 0);
        assert_eq!(meta.active[1], "blissey");
    }

    /// Timed side-condition set turns: |-sidestart| records the turn delta at
    /// set time (parser turn_number arithmetic), |-sideend| pops the entry;
    /// non-timed conditions (Spikes) are not tracked.
    #[test]
    fn side_condition_set_turn_replay() {
        let meta = evolve_leaf_meta(
            &LeafMeta::default(),
            &lines(&[
                "|-sidestart|p1: side|Reflect",
                "|turn|8",
                "|-sidestart|p1: side|move: Light Screen",
                "|-sidestart|p2: side|Spikes",
                "|turn|9",
                "|-sideend|p1: side|Reflect",
            ]),
            &LeafMetaCtx::default(),
        );
        assert_eq!(
            meta.side_condition_sets
                .get(&(0, "reflect".to_string()))
                .copied(),
            Some(None) // set then ended: the parser pops the entry
        );
        assert_eq!(
            meta.side_condition_sets
                .get(&(0, "lightscreen".to_string()))
                .copied(),
            Some(Some(1))
        );
        assert!(!meta
            .side_condition_sets
            .contains_key(&(1, "spikes".to_string())));
    }

    /// Review F2: Showdown's switch-swap position semantics.
    #[test]
    fn self_order_swaps_with_slot_zero() {
        let order: Vec<String> = ["kangaskhan", "volbeat", "snorlax"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        // Kangaskhan -> Volbeat: swap positions 0 and 1.
        let after = evolve_self_order(
            &order,
            &["|switch|p1a: Volbeat|Volbeat, L88, M|100/100".to_string()],
            "p1",
        );
        assert_eq!(after, ["volbeat", "kangaskhan", "snorlax"]);
        // Chain: then Volbeat -> Snorlax (swap 0 and 2) — NOT a rotation.
        let after2 = evolve_self_order(
            &after,
            &["|switch|p1a: Snorlax|Snorlax, L76, F|100/100".to_string()],
            "p1",
        );
        assert_eq!(after2, ["snorlax", "kangaskhan", "volbeat"]);
        // Opponent switches never touch the self order.
        let untouched = evolve_self_order(
            &order,
            &["|switch|p2a: Blissey|Blissey, L80, F|100/100".to_string()],
            "p1",
        );
        assert_eq!(untouched, order);
    }
}
