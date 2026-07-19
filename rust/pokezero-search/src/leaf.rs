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
//! Delta families: a few belief-ledger scalars (opponent `move_uses`, sleep
//! turn counts) cannot be recomputed from the leaf engine state alone because
//! the world constructor seeds them approximately (full PP, fresh sleep).
//! They evolve as ROOT VALUE + (leaf engine value − root engine value), which
//! reproduces the root exactly at zero branches and adds exactly the
//! simulated consumption at leaves.
//!
//! The root-parity gate (`scripts/leaf_root_parity.py`) drives this path at
//! zero branches over the golden corpus: world from the recorded public
//! payload + true teams, `encode_leaf` on the untouched root state, byte-diff
//! against the recorded golden arrays.

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
/// deliberately dropped — the parser never records them either.
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
    (PokemonVolatileStatus::CURSE, "curse"),
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
    VOLATILE_MAP
        .iter()
        .filter(|(vs, _)| side.volatile_statuses.contains(vs))
        .map(|(_, id)| (*id).to_string())
        .collect()
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

/// `p.status == SLEEP && p.hp > 0 && p.rest_turns == 0` — the engine's own
/// sleep-clause predicate (gen3 state.rs), matching the parser's semantics.
fn sleep_clause_used(side: &Side) -> bool {
    side.pokemon
        .into_iter()
        .any(|p| p.status == PokemonStatus::SLEEP && p.hp > 0 && p.rest_turns == 0)
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
    sleep_turns: i8,
    rest_turns: i8,
}

#[derive(Clone, Debug, Default)]
struct SideSnapshot {
    /// Party-index-aligned snapshots.
    mons: Vec<MonSnapshot>,
    /// Root engine toxic counter (delta base: the payload seeds the engine at
    /// the request-boundary convention, one below the parser's tracked stage).
    toxic_count: i8,
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
            sleep_turns: p.sleep_turns,
            rest_turns: p.rest_turns,
        });
    }
    SideSnapshot {
        mons,
        toxic_count: side.side_conditions.toxic_count,
        nonrest_sleepers: nonrest_sleepers(side),
    }
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
        Ok(LeafContext {
            tables,
            root,
            self_is_p1,
            species_keys,
            root_snapshot: [
                snapshot_side(&root_state.side_one),
                snapshot_side(&root_state.side_two),
            ],
        })
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
    /// leaf's battle turn (root turn + completed simulated turns).
    pub(crate) fn leaf_row_inputs(&self, state: &State, turn: i64) -> PyResult<Value> {
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

        // Root ledger values for the delta families (read before overwrite).
        let root_toxic = [
            md.get("self_toxic_stage").and_then(Value::as_i64).unwrap_or(0),
            md.get("opponent_toxic_stage").and_then(Value::as_i64).unwrap_or(0),
        ];
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
                md.insert(
                    "weather_turns_remaining".into(),
                    // Ability weather is permanent in gen3: production pins
                    // the counter at the full duration.
                    json!(if permanent {
                        self.tables.layout_timed_condition_duration()
                    } else {
                        state.weather.turns_remaining as i64
                    }),
                );
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
        // Sleep clause / toxic stage: delta families — root ledger value
        // evolved by the engine's change since the root (the world seeds both
        // at conventions the engine alone cannot invert; see the column map).
        for (index, (key_sc, key_tox, side, engine_side)) in [
            ("self_sleep_clause_used", "self_toxic_stage", self_side, self_engine),
            (
                "opponent_sleep_clause_used",
                "opponent_toxic_stage",
                opp_side,
                opp_engine,
            ),
        ]
        .into_iter()
        .enumerate()
        {
            let snapshot = &self.root_snapshot[engine_side];
            let clause = root_sleep_clause[index]
                || nonrest_sleepers(side) > snapshot.nonrest_sleepers;
            md.insert(key_sc.into(), json!(clause));
            let leaf_count = side.side_conditions.toxic_count as i64;
            let root_count = snapshot.toxic_count as i64;
            let stage = if leaf_count < root_count {
                // The toxic mon left the field during the branch: fresh stint,
                // engine and parser conventions agree from zero.
                leaf_count
            } else {
                root_toxic[index] + (leaf_count - root_count)
            };
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
                    }
                    obj.insert("active".into(), json!(party == active_party));
                }
            }
        }

        // --- belief-ledger evolution (exact-state fields only; belief FACTS
        //     are world-constants and stay untouched) ---
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
                        // Sleep bookkeeping: root ledger value + engine delta
                        // (the world seeds sleep approximately, so the engine
                        // alone cannot reproduce the ledger at the root).
                        let root_sleep = obj
                            .get("sleep_turns")
                            .and_then(Value::as_i64)
                            .unwrap_or(0);
                        let engine_delta = (p.sleep_turns - snapshot.sleep_turns).max(0) as i64
                            + rest_sleep_delta(snapshot.rest_turns, p.rest_turns);
                        obj.insert("sleep_turns".into(), json!(root_sleep + engine_delta));
                        let root_rest = obj
                            .get("rest_sleep")
                            .and_then(Value::as_bool)
                            .unwrap_or(false);
                        let rest_now = p.rest_turns > 0
                            || (root_rest && p.status == PokemonStatus::SLEEP);
                        obj.insert("rest_sleep".into(), json!(rest_now));
                    }
                    if !is_self {
                        rewrite_move_uses(obj, &snapshot, *p);
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
    fn action_surface(
        &self,
        self_side: &Side,
        _self_engine: usize,
        self_options: &[MoveChoice],
        self_team_order: &[(String, bool)],
        force_switch_shape: bool,
    ) -> PyResult<(Value, Value)> {
        let action_count = self.tables.layout_action_count();
        let move_action_count = self.tables.layout_move_action_count();

        // Engine move surface of the active mon, engine slot order.
        let active = self_side.get_active_immutable();
        let mut engine_moves: Vec<(String, bool, i8)> = Vec::new(); // (showdown id, disabled, pp)
        for mv in active.moves.into_iter() {
            let engine_id = format!("{:?}", mv.id).to_lowercase();
            if engine_id == "none" {
                continue;
            }
            engine_moves.push((showdown_move_id(&engine_id), mv.disabled, mv.pp));
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
        let moves_present = !force_switch_shape;
        for slot in 0..move_action_count {
            let entry = if moves_present { engine_moves.get(slot) } else { None };
            match entry {
                Some((move_id, disabled, pp)) => {
                    let legal = legal_moves.contains(&slot) && active.hp > 0;
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
    ) -> PyResult<EncodedArrays> {
        let row = self.leaf_row_inputs(state, turn)?;
        let products = fold.products();
        encode_row_value(&self.tables, &row, Some(&products))
    }
}

/// Rest-sleep contribution to the slept-turn delta: a fresh Rest sets the
/// engine counter to 3 and decrements per sleeping turn, so turns slept since
/// the root is (root_rest − rest) when resting at both ends, or (3 − rest)
/// when the Rest happened inside the branch.
fn rest_sleep_delta(root_rest: i8, rest: i8) -> i64 {
    if rest <= 0 {
        return 0;
    }
    if root_rest > 0 {
        (root_rest - rest).max(0) as i64
    } else {
        (3 - rest).max(0) as i64
    }
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
    /// inputs. At zero branches (`state_str` = the root state, `fold` = the
    /// root fold, `turn` = the root turn) this must reproduce the golden
    /// observation — the root-parity gate.
    fn encode_leaf(
        &self,
        py: Python<'_>,
        state_str: &str,
        fold: &PyFoldState,
        turn: i64,
    ) -> PyResult<Py<PyDict>> {
        let state = parse_state(state_str)?;
        let encoded = self.ctx.encode_leaf(&state, fold.inner(), turn)?;
        encoded_to_dict(py, &encoded)
    }

    /// The rewritten row-inputs JSON for a leaf state (divergence debugging:
    /// diff this against the root inputs to see exactly which state fields
    /// the engine recompute changed).
    fn leaf_inputs_json(&self, state_str: &str, turn: i64) -> PyResult<String> {
        let state = parse_state(state_str)?;
        let row = self.ctx.leaf_row_inputs(&state, turn)?;
        serde_json::to_string(&row).map_err(|e| err(format!("serialize leaf inputs: {e}")))
    }
}
