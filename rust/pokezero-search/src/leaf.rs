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
use pyo3::types::PyDict;
use serde_json::{json, Map, Value};

use poke_engine::engine::state::{MoveChoice, PokemonVolatileStatus, Weather};
use poke_engine::state::{PokemonStatus, Side, State};

use crate::encoder::{encode_row_value, encoded_to_dict, EncodedArrays, Tables};
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

fn tracked_volatiles(side: &Side) -> Vec<String> {
    use poke_engine::state::PokemonType;
    let mut out: Vec<String> = VOLATILE_MAP
        .iter()
        .filter(|(vs, _)| side.volatile_statuses.contains(vs))
        .map(|(_, id)| (*id).to_string())
        .collect();
    // CURSE Ghost gate (review F5): only a Ghost-typed curser's volatile is
    // real; the engine's spurious non-Ghost USER volatile is dropped.
    if side
        .volatile_statuses
        .contains(&PokemonVolatileStatus::CURSE)
    {
        let types = side.get_active_immutable().types;
        if types.0 == PokemonType::GHOST || types.1 == PokemonType::GHOST {
            out.push("curse".to_string());
        }
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

#[derive(Clone, Debug, Default)]
struct MonSnapshot {
    /// PP per move slot, keyed by showdown move id (delta base for
    /// `move_uses` / PP fractions).
    pp: Vec<(String, i8)>,
    hp: i16,
    status: Option<&'static str>,
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
/// `|-curestatus|` clears, the side's `|switch|`/`|drag|` resets stage AND
/// stint, `|turn|` escalates every nonzero stage (cap 15) and advances every
/// stint).
pub(crate) fn evolve_leaf_meta(meta: &LeafMeta, lines: &[String]) -> LeafMeta {
    let mut out = meta.clone();
    for line in lines {
        if line.starts_with("|turn|") {
            for side in 0..2 {
                if out.toxic[side] > 0 {
                    out.toxic[side] = (out.toxic[side] + 1).min(15);
                }
                out.stint[side] += 1;
            }
            continue;
        }
        if let Some(rest) = line.strip_prefix("|move|") {
            if let Some(side) = line_slot(rest) {
                out.fresh_active[side] = false;
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
                    Some(status) if status == "tox" => out.toxic[side] = 1,
                    Some(status) if status == "slp" => {
                        let key = ident_species_key(rest);
                        out.sleep.insert((side, key), (true, 0));
                    }
                    _ => {}
                }
            } else if is_cure {
                out.toxic[side] = 0;
            } else {
                out.toxic[side] = 0;
                out.stint[side] = 0;
                out.fresh_active[side] = true;
            }
            break;
        }
    }
    out
}

// ---------------------------------------------------------------------------
// The leaf context
// ---------------------------------------------------------------------------

pub(crate) struct LeafContext {
    pub(crate) tables: Tables,
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
        let tables = Tables::from_json(tables_json)?;
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
        let root_turn = ctx.get("turn").and_then(Value::as_i64).unwrap_or(0);
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
    pub(crate) fn leaf_row_inputs(
        &self,
        state: &State,
        turn: i64,
        self_order: Option<&[String]>,
        meta: Option<&LeafMeta>,
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
        // review F1), with a cure guard only for full cures the mapper does
        // not render (|-curestatus| is fold-invisible; a status REPLACEMENT
        // like Rest's tox->slp keeps the parser stage — golden-proven).
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
            if stage > 0 && active.hp > 0 && active.status == PokemonStatus::NONE {
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
            json!(tracked_volatiles(self_side)),
        );
        md.insert(
            "opponent_active_volatiles".into(),
            json!(tracked_volatiles(opp_side)),
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
                        rewrite_move_uses(obj, &snapshot, *p);
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
        let (s1_options, s2_options) = state.get_all_options();
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
            meta.fresh_active[self_engine],
        )?;
        md.insert("action_candidates".into(), candidates);

        // --- public materialization scalars the encoder reads ---
        let pm = row
            .get_mut("public_materialization")
            .and_then(Value::as_object_mut)
            .ok_or_else(|| err("row inputs missing public_materialization object"))?;
        pm.insert("turn".into(), json!(turn));
        pm.insert("selfActiveMoves".into(), payload_moves);
        // Timed side-condition SET TURNS stay root-frozen: remaining turns
        // derive from (leaf turn − set turn), which keeps ticking correctly
        // through simulated turns; only the ACTIVE counts are rewritten.
        let (self_slot_key, opp_slot_key) = if self_is_p1 { ("p1", "p2") } else { ("p2", "p1") };
        if let Some(sides) = pm.get_mut("sides").and_then(Value::as_object_mut) {
            for (slot, side) in [(self_slot_key, self_side), (opp_slot_key, opp_side)] {
                if let Some(side_obj) = sides.get_mut(slot).and_then(Value::as_object_mut) {
                    side_obj.insert("sideConditions".into(), side_condition_counts(side));
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
        _self_engine: usize,
        self_options: &[MoveChoice],
        self_team_order: &[(String, bool)],
        force_switch_shape: bool,
        fresh_switch_in: bool,
    ) -> PyResult<(Value, Value)> {
        let action_count = self.tables.layout_action_count();
        let move_action_count = self.tables.layout_move_action_count();

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
        let mut engine_moves: Vec<(String, bool, i8)> = Vec::new(); // (showdown id, disabled, pp)
        if !recharging {
            for mv in active.moves.into_iter() {
                let engine_id = format!("{:?}", mv.id).to_lowercase();
                if engine_id == "none" {
                    continue;
                }
                let disabled = if fresh_switch_in { false } else { mv.disabled };
                engine_moves.push((showdown_move_id(&engine_id), disabled, mv.pp));
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
                    let engine_side = self.engine_side_index(true);
                    if let Some(key) = self.species_keys[engine_side].get(party) {
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
                            "pp": (*pp as i64).max(0),
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

    pub(crate) fn encode_leaf(
        &self,
        state: &State,
        fold: &FoldStateInner,
        turn: i64,
        self_order: Option<&[String]>,
        meta: Option<&LeafMeta>,
    ) -> PyResult<EncodedArrays> {
        let row = self.leaf_row_inputs(state, turn, self_order, meta)?;
        let products = fold.products();
        encode_row_value(&self.tables, &row, Some(&products))
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

/// Opponent `move_uses` evolution: root ledger uses + engine PP consumed
/// since the root (the world seeds opponent PP at catalog full, so engine PP
/// alone cannot reproduce the ledger's observed-use counts).
fn rewrite_move_uses(
    obj: &mut Map<String, Value>,
    snapshot: &MonSnapshot,
    p: &poke_engine::state::Pokemon,
) {
    let mut engine_pp: HashMap<String, i8> = HashMap::new();
    for mv in p.moves.into_iter() {
        let engine_id = format!("{:?}", mv.id).to_lowercase();
        if engine_id == "none" {
            continue;
        }
        engine_pp.insert(showdown_move_id(&engine_id), mv.pp);
    }
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
        let root_pp = snapshot
            .pp
            .iter()
            .find(|(id, _)| *id == move_id)
            .map(|(_, pp)| *pp);
        let leaf_pp = engine_pp.get(&move_id).copied();
        if let (Some(root_pp), Some(leaf_pp)) = (root_pp, leaf_pp) {
            let consumed = (root_pp - leaf_pp).max(0) as i64;
            items[1] = json!(root_uses + consumed);
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

    /// Encode a leaf observation: ENGINE-STATE columns from `state_str`,
    /// FOLD columns from `fold`, WORLD-CONSTANT columns from the root row
    /// inputs. `lines` are the branch's synthesized protocol lines from the
    /// root (they drive the self-team display order via Showdown's
    /// switch-swap semantics AND the line-driven metadata: toxic stages,
    /// active stints — None/empty keeps root values). At zero branches
    /// (`state_str` = the root state, `fold` = the root fold, `turn` = the
    /// root turn, no lines) this must reproduce the golden observation — the
    /// root-parity gate.
    #[pyo3(signature = (state_str, fold, turn, lines = None))]
    fn encode_leaf(
        &self,
        py: Python<'_>,
        state_str: &str,
        fold: &PyFoldState,
        turn: i64,
        lines: Option<Vec<String>>,
    ) -> PyResult<Py<PyDict>> {
        let state = parse_state(state_str)?;
        let (order, meta) = self.branch_context(lines.as_deref());
        let encoded = self.ctx.encode_leaf(
            &state,
            fold.inner(),
            turn,
            order.as_deref(),
            meta.as_ref(),
        )?;
        encoded_to_dict(py, &encoded)
    }

    /// The rewritten row-inputs JSON for a leaf state (divergence debugging:
    /// diff this against the root inputs to see exactly which state fields
    /// the engine recompute changed).
    #[pyo3(signature = (state_str, turn, lines = None))]
    fn leaf_inputs_json(
        &self,
        state_str: &str,
        turn: i64,
        lines: Option<Vec<String>>,
    ) -> PyResult<String> {
        let state = parse_state(state_str)?;
        let (order, meta) = self.branch_context(lines.as_deref());
        let row = self
            .ctx
            .leaf_row_inputs(&state, turn, order.as_deref(), meta.as_ref())?;
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
                Some(evolve_leaf_meta(self.ctx.root_meta(), lines)),
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lines(raw: &[&str]) -> Vec<String> {
        raw.iter().map(|s| s.to_string()).collect()
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
        );
        assert_eq!(meta.toxic[1], 4);
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
        );
        assert_eq!(meta.toxic[1], 9);
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
        );
        // p1 switched (reset) then two completed turns; p2 stayed in.
        assert_eq!(meta.stint[0], 2);
        assert_eq!(meta.stint[1], 7);
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
