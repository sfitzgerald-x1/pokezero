//! Correctness-first synchronous own-policy provider for native model search.
//! The Python callback receives only a sampled OWN request plus PUBLIC branch
//! events. Neither a complete engine State nor the other seat's private facts
//! cross the boundary. Subject leaf/value/PUCT evaluation remains in model.rs.

use std::cell::{Cell, RefCell};
use std::collections::{HashMap, HashSet};
use std::time::Instant;

use poke_engine::choices::Choices;
use poke_engine::state::{Side, State};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use serde_json::json;

use crate::events::EventContext;
use crate::policy_opponent::ActionDistribution;
use crate::policy_request::PrivateTrapObservation;
use crate::tree::DecisionNode;

type BranchKey = (usize, usize);

#[derive(Clone)]
struct PublicPrefix {
    lines: Vec<String>,
    meta: crate::leaf::LeafMeta,
    own_pp: OwnPpLedger,
}

type MoveCharges = HashMap<(usize, String, String), i64>;

/// Original moves and Transform's temporary five-PP slots have separate banks.
/// Native tracks every copied-slot decrement (<10 PP), but deliberately omits
/// high-PP decrements. Retain only original-bank charges for root-base repair.
#[derive(Clone, Default)]
struct OwnPpLedger {
    original_charges: MoveCharges,
    transformed: HashSet<String>,
    active: String,
}

fn identity(name: &str) -> String {
    name.chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect()
}

impl OwnPpLedger {
    fn from_root(side: &Side, species: &[String]) -> Self {
        let transformed = species
            .iter()
            .enumerate()
            .filter(|(index, _)| side.pokemon.pkmn[*index].pre_transform.is_some())
            .map(|(_, name)| identity(name))
            .collect();
        let active = side
            .active_index
            .serialize()
            .parse::<usize>()
            .expect("native party index");
        Self {
            transformed,
            active: identity(&species[active]),
            ..Self::default()
        }
    }

    fn advance(
        &mut self,
        before: &MoveCharges,
        after: &MoveCharges,
        lines: &[String],
        side_one: bool,
    ) -> PyResult<()> {
        let seat = usize::from(!side_one);
        // A successful Transform charges the initiating move before changing
        // banks; a second Transform charges the already-copied bank. Classify
        // this edge's charges using the PRE-edge transformation state.
        for (key, total) in after {
            if key.0 != seat {
                continue;
            }
            let delta = total - before.get(key).copied().unwrap_or(0);
            if delta < 0 {
                return Err(PyValueError::new_err(
                    "policy opponent: PP ledger regressed",
                ));
            }
            if !self.transformed.contains(&key.1) {
                *self.original_charges.entry(key.clone()).or_insert(0) += delta;
            }
        }
        let slot = if side_one { "p1a: " } else { "p2a: " };
        for line in lines {
            let fields: Vec<_> = line.split('|').collect();
            let Some(actor) = fields.get(2).and_then(|value| value.strip_prefix(slot)) else {
                continue;
            };
            match fields.get(1).copied() {
                Some("switch" | "drag" | "replace") => {
                    self.transformed.remove(&self.active);
                    self.active = identity(actor);
                }
                Some("-transform") => {
                    self.transformed.insert(identity(actor));
                }
                _ => {}
            }
        }
        Ok(())
    }
}

pub(crate) struct PolicyOpponentBridge {
    callback: Py<PyAny>,
    opponent_side_one: bool,
    root_order: Vec<String>,
    max_pp: HashMap<String, i64>,
    base_pp: HashMap<String, i64>,
    root_own_side: Side,
    display_ctx: EventContext,
    // Immutable prefixes per branch; RefCell permits the leaf-pricing and
    // reached-node closures to share one ledger without changing the subject
    // fold's ownership or any incumbent encoder path.
    prefixes: RefCell<HashMap<BranchKey, PublicPrefix>>,
    pub(crate) evaluations: Cell<usize>,
    pub(crate) provider_calls: Cell<usize>,
    pub(crate) policy_nanos: Cell<u128>,
}

impl PolicyOpponentBridge {
    /// Isolate one terminal continuation's branch history. Copy ONLY the
    /// frontier prefix; future siblings must never share mutable PP/history.
    pub(crate) fn fork_at(&self, key: BranchKey) -> PyResult<Self> {
        let prefix = self.prefixes.borrow().get(&key).cloned().ok_or_else(||
            PyValueError::new_err("raw policy terminal: frontier public prefix missing"))?;
        let fork = Self::new(
            Python::attach(|py| self.callback.clone_ref(py)), self.opponent_side_one,
            self.root_order.clone(), self.max_pp.clone(), self.base_pp.clone(),
            self.root_own_side.clone(), self.display_ctx.clone(),
        );
        fork.prefixes.borrow_mut().insert(key, prefix);
        Ok(fork)
    }

    pub(crate) fn new(
        callback: Py<PyAny>,
        opponent_side_one: bool,
        root_order: Vec<String>,
        max_pp: HashMap<String, i64>,
        base_pp: HashMap<String, i64>,
        root_own_side: Side,
        display_ctx: EventContext,
    ) -> Self {
        let root_order = root_order
            .iter()
            .map(|name| {
                name.chars()
                    .filter(|c| c.is_ascii_alphanumeric())
                    .flat_map(char::to_lowercase)
                    .collect()
            })
            .collect();
        Self {
            callback,
            opponent_side_one,
            root_order,
            max_pp,
            base_pp,
            root_own_side,
            display_ctx,
            prefixes: RefCell::new(HashMap::new()),
            evaluations: Cell::new(0),
            provider_calls: Cell::new(0),
            policy_nanos: Cell::new(0),
        }
    }

    pub(crate) fn record(
        &self,
        parent: Option<BranchKey>,
        key: BranchKey,
        lines: &[String],
        ctx: &EventContext,
        meta: &crate::leaf::LeafMeta,
    ) -> PyResult<()> {
        let mut prefixes = self.prefixes.borrow_mut();
        let species = &self.display_ctx.species[usize::from(!self.opponent_side_one)];
        let (mut prefix, mut own_pp, before) = match parent {
            None => (
                Vec::new(),
                OwnPpLedger::from_root(&self.root_own_side, species),
                HashMap::new(),
            ),
            Some(parent) => prefixes
                .get(&parent)
                .map(|prefix| {
                    (
                        prefix.lines.clone(),
                        prefix.own_pp.clone(),
                        prefix.meta.move_charges.clone(),
                    )
                })
                .ok_or_else(|| {
                    PyValueError::new_err("policy opponent: parent public prefix missing")
                })?,
        };
        let canonical = canonical_branch_species(lines, &self.display_ctx);
        own_pp.advance(
            &before,
            &meta.move_charges,
            &canonical,
            self.opponent_side_one,
        )?;
        // Project BEFORE invoking user/Python code, rather than letting a
        // callback see exact hidden HP and trusting it not to use those facts.
        prefix.extend(project_public_hp(&canonical, ctx.hp_percent)?);
        prefixes.insert(
            key,
            PublicPrefix {
                lines: prefix,
                meta: meta.clone(),
                own_pp,
            },
        );
        Ok(())
    }

    /// A forked terminal continuation has exactly one live prefix, unlike a
    /// branching search. Drop old copies after recording the next boundary so
    /// a long continuation does not retain a quadratic history/PP ledger.
    pub(crate) fn record_linear(
        &self, parent: BranchKey, key: BranchKey, lines: &[String],
        ctx: &EventContext, meta: &crate::leaf::LeafMeta,
    ) -> PyResult<()> {
        self.record(Some(parent), key, lines, ctx, meta)?;
        self.prefixes.borrow_mut().retain(|candidate, _| *candidate == key);
        Ok(())
    }

    pub(crate) fn provide(
        &self,
        state: &State,
        node: &DecisionNode,
        parent: Option<BranchKey>,
        ctx: &EventContext,
    ) -> PyResult<ActionDistribution> {
        self.provide_with_trap(state, node, parent, ctx, PrivateTrapObservation::None)
    }

    /// Private trapping observations never enter public prefix/PP ledgers.
    /// Normal policy-opponent search retains its original strict native surface.
    pub(crate) fn provide_with_trap(
        &self,
        state: &State,
        node: &DecisionNode,
        parent: Option<BranchKey>,
        _ctx: &EventContext,
        trap_observation: PrivateTrapObservation,
    ) -> PyResult<ActionDistribution> {
        let started = Instant::now();
        let result = (|| {
            let (lines, own_pp) = match parent {
                None => (Vec::new(), None),
                Some(key) => self
                    .prefixes
                    .borrow()
                    .get(&key)
                    .map(|prefix| (prefix.lines.clone(), Some(prefix.own_pp.clone())))
                    .ok_or_else(|| {
                        PyValueError::new_err("policy opponent: reached public prefix missing")
                    })?,
            };
            let slot = if self.opponent_side_one { "p1" } else { "p2" };
            let order = crate::leaf::evolve_self_order(&self.root_order, &lines, slot);
            let (side, options, species) = if self.opponent_side_one {
                (
                    &state.side_one,
                    &node.s1_options,
                    &self.display_ctx.species[0],
                )
            } else {
                (
                    &state.side_two,
                    &node.s2_options,
                    &self.display_ctx.species[1],
                )
            };
            if let Some(ledger) = &own_pp {
                for (index, name) in species.iter().enumerate() {
                    if side.pokemon.pkmn[index].pre_transform.is_some()
                        != ledger.transformed.contains(&identity(name))
                    {
                        return Err(PyValueError::new_err(
                            "policy opponent: public Transform provenance mismatch",
                        ));
                    }
                }
            }
            let own = corrected_own_pp(
                side,
                &self.root_own_side,
                species,
                self.opponent_side_one,
                own_pp.as_ref().map(|ledger| &ledger.original_charges),
            )
            .map_err(|error| {
                Python::attach(|py| {
                    if let Ok(raw) = error.value(py).getattr("policy_opponent_diagnostic") {
                        if let Ok(raw) = raw.extract::<String>() {
                            if let Ok(mut witness) = serde_json::from_str::<serde_json::Value>(&raw)
                            {
                                witness["node_depth"] = json!(node.depth);
                                witness["public_branch_lines"] = json!(lines);
                                let _ = error
                                    .value(py)
                                    .setattr("policy_opponent_diagnostic", witness.to_string());
                            }
                        }
                    }
                });
                error
            })?;
            let opponent_replacing = if self.opponent_side_one {
                state.side_two.force_switch
            } else {
                state.side_one.force_switch
            };
            let bundle = crate::policy_request::sampled_side_request_with_trap(
                &own,
                slot,
                species,
                &order,
                options,
                &self.max_pp,
                parent.is_none(),
                opponent_replacing,
                Some(&self.base_pp),
                trap_observation,
            )
            .map_err(|error| {
                Python::attach(|py| {
                    if let Ok(raw) = error.value(py).getattr("policy_opponent_diagnostic") {
                        if let Ok(raw) = raw.extract::<String>() {
                            if let Ok(mut witness) = serde_json::from_str::<serde_json::Value>(&raw)
                            {
                                witness["node_depth"] = json!(node.depth);
                                witness["public_branch_lines"] = json!(lines);
                                // Own sampled request/PP only, never State or
                                // the opposing private team. No callback runs.
                                witness["sampled_own_side"] = json!(own.serialize());
                                let _ = error
                                    .value(py)
                                    .setattr("policy_opponent_diagnostic", witness.to_string());
                            }
                        }
                    }
                });
                error
            })?;
            let payload = json!({"native_request_bundle": bundle, "public_branch_lines": lines,
                "opponent_slot": slot})
            .to_string();
            self.provider_calls.set(self.provider_calls.get() + 1);
            // Certified singleton/WAIT providers return without a network
            // forward. Keep certification calls distinct from policy evals.
            if options.len() > 1 {
                self.evaluations.set(self.evaluations.get() + 1);
            }
            let weights: Vec<f32> =
                Python::attach(|py| self.callback.call1(py, (payload,))?.extract(py))?;
            ActionDistribution::new(&weights, options.len())
        })();
        self.policy_nanos
            .set(self.policy_nanos.get() + started.elapsed().as_nanos());
        result
    }
}

fn canonical_branch_species(lines: &[String], ctx: &EventContext) -> Vec<String> {
    let key = |name: &str| {
        name.chars()
            .filter(|c| c.is_ascii_alphanumeric())
            .flat_map(char::to_lowercase)
            .collect::<String>()
    };
    let displays: HashMap<_, _> = ctx
        .species
        .iter()
        .flatten()
        .map(|name| (key(name), name))
        .collect();
    lines
        .iter()
        .map(|line| {
            let mut fields: Vec<String> = line.split('|').map(String::from).collect();
            for field in &mut fields {
                if let Some((prefix, name)) = field.split_once(": ") {
                    if prefix.ends_with("p1a") || prefix.ends_with("p2a") {
                        if let Some(display) = displays.get(&key(name)) {
                            *field = format!("{prefix}: {display}");
                        }
                    }
                }
            }
            let event = fields.get(1).map(String::as_str).unwrap_or("");
            if matches!(event, "switch" | "drag" | "replace" | "-formechange") {
                if let Some(details) = fields.get_mut(3) {
                    let (name, rest) = details.split_once(',').unwrap_or((details, ""));
                    if let Some(display) = displays.get(&key(name)) {
                        *details = if rest.is_empty() {
                            display.to_string()
                        } else {
                            format!("{display},{rest}")
                        };
                    }
                }
            }
            fields.join("|")
        })
        .collect()
}

/// The engine omits DecrementPP above its low-PP threshold. Own requests must
/// nevertheless carry exact PP, using the same branch charge ledger as the
/// incumbent encoder. Subtract from the ROOT base once, not the already
/// decremented leaf PP: that would double-charge moves below the threshold.
fn corrected_own_pp(
    side: &Side,
    root: &Side,
    species: &[String],
    side_one: bool,
    charges: Option<&MoveCharges>,
) -> PyResult<Side> {
    let mut own = side.clone();
    let Some(charges) = charges else {
        return Ok(own);
    };
    for (physical, name) in species.iter().enumerate() {
        let species_key: String = name
            .chars()
            .filter(|c| c.is_ascii_alphanumeric())
            .flat_map(char::to_lowercase)
            .collect();
        for index in 0..4 {
            let slot = match index {
                0 => poke_engine::state::PokemonMoveIndex::M0,
                1 => poke_engine::state::PokemonMoveIndex::M1,
                2 => poke_engine::state::PokemonMoveIndex::M2,
                _ => poke_engine::state::PokemonMoveIndex::M3,
            };
            let mv = &own.pokemon.pkmn[physical].moves[&slot];
            if mv.id == Choices::NONE {
                continue;
            }
            // A transformed own Pokemon legitimately changes both species and
            // moves. Its copied slots start at five PP and native decrements
            // every use, including Pressure. Never subtract original charges
            // again, or treat Transform as a private-identity corruption.
            if own.pokemon.pkmn[physical].pre_transform.is_some() {
                if mv.pp > 5 {
                    return Err(PyValueError::new_err(
                        "policy opponent: invalid copied move PP",
                    ));
                }
                own.pokemon.pkmn[physical].moves[&slot].pp = mv.pp.max(0);
                continue;
            }
            let (id, _, _) = crate::policy_request::move_names(mv.id)?;
            let root_mon = &root.pokemon.pkmn[physical];
            let (base_id, base_pp) = root_mon
                .pre_transform
                .as_ref()
                .map(|snapshot| snapshot.moves[index])
                .unwrap_or((root_mon.moves[&slot].id, root_mon.moves[&slot].pp));
            if base_id != mv.id {
                let error = PyValueError::new_err("policy opponent: changed private move identity");
                let witness = json!({
                    "schema": "policy-opponent-refusal-v1",
                    "kind": "private_move_identity_mismatch",
                    "diagnostic_only_not_policy_input": true,
                    "seat": if side_one { "p1" } else { "p2" },
                    "species": name, "physical_index": physical, "move_index": index,
                    "root_move": format!("{:?}", base_id),
                    "branch_move": format!("{:?}", mv.id),
                    "transformed": own.pokemon.pkmn[physical].pre_transform.is_some()
                });
                Python::attach(|py| {
                    error
                        .value(py)
                        .setattr("policy_opponent_diagnostic", witness.to_string())
                })?;
                return Err(error);
            }
            let charged = charges
                .get(&(usize::from(!side_one), species_key.clone(), id))
                .copied()
                .unwrap_or(0);
            if charged < 0 {
                return Err(PyValueError::new_err(
                    "policy opponent: negative private PP charge",
                ));
            }
            own.pokemon.pkmn[physical].moves[&slot].pp =
                (i64::from(base_pp) - charged).max(0) as i8;
        }
    }
    Ok(own)
}

/// Native branch events use the registered context's HP provenance. The live
/// root prefix is supplied separately by the canonical Python view factory.
fn project_public_hp(lines: &[String], percentage: [bool; 2]) -> PyResult<Vec<String>> {
    lines
        .iter()
        .map(|line| {
            let mut parts: Vec<String> = line.split('|').map(String::from).collect();
            let event = parts.get(1).map(String::as_str).unwrap_or("");
            let fields = match event {
                "switch" | "drag" | "replace" => vec![4],
                "-damage" | "-heal" => vec![3],
                // Pain Split emits separate one-owner lines with optional
                // [from]/[silent] qualifiers, not necessarily two-owner HP.
                "-sethp" if parts.get(4).is_some_and(|s| !s.starts_with('[')) => {
                    vec![3, 5]
                }
                "-sethp" => vec![3],
                "request" | "split" => {
                    return Err(PyValueError::new_err(
                        "policy opponent: private branch event",
                    ))
                }
                _ => Vec::new(),
            };
            if event == "-sethp" {
                let qualifier_start = if fields.len() == 2 { 6 } else { 4 };
                if parts
                    .iter()
                    .skip(qualifier_start)
                    .any(|s| !s.starts_with('['))
                {
                    return Err(PyValueError::new_err(
                        "policy opponent: malformed public HP qualifiers",
                    ));
                }
            }
            for field in fields {
                let ident = if field == 4 { 2 } else { field - 1 };
                let side = match parts.get(ident).map(String::as_str) {
                    Some(s) if s.starts_with("p1") => 0,
                    Some(s) if s.starts_with("p2") => 1,
                    _ => {
                        return Err(PyValueError::new_err(
                            "policy opponent: unknown public HP owner",
                        ))
                    }
                };
                let condition = parts
                    .get(field)
                    .ok_or_else(|| PyValueError::new_err("policy opponent: missing public HP"))?;
                if condition == "0 fnt" {
                    continue;
                }
                let (fraction, status) = condition.split_once(' ').unwrap_or((condition, ""));
                let (hp, maxhp) = fraction
                    .split_once('/')
                    .ok_or_else(|| PyValueError::new_err("policy opponent: malformed public HP"))?;
                let hp: u64 = hp
                    .parse()
                    .map_err(|_| PyValueError::new_err("policy opponent: invalid public HP"))?;
                let maxhp: u64 = maxhp
                    .parse()
                    .map_err(|_| PyValueError::new_err("policy opponent: invalid public max HP"))?;
                if maxhp == 0 || hp > maxhp || (percentage[side] && maxhp != 100) {
                    return Err(PyValueError::new_err(
                        "policy opponent: invalid public HP range/provenance",
                    ));
                }
                let mut value =
                    ((u128::from(hp) * 100 + u128::from(maxhp) - 1) / u128::from(maxhp)) as u64;
                if value == 100 && hp < maxhp {
                    value = 99;
                }
                parts[field] = if status.is_empty() {
                    format!("{value}/100")
                } else {
                    format!("{value}/100 {status}")
                };
            }
            Ok(parts.join("|"))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_species_spelling_preserves_event_details_and_public_names() {
        let ctx = EventContext {
            species: [
                vec!["Deoxys-Defense".into(), "Mr. Mime".into()],
                vec!["Unown-Q".into()],
            ],
            turn: 1,
            hp_percent: [false, false],
        };
        let lines = [
            "|switch|p1a: deoxysdefense|deoxysdefense, L80, M|50/100",
            "|move|p1a: mrmime|Psychic|p2a: unownq",
            "|-formechange|p1a: deoxysdefense|deoxysdefense|[from] test",
            "|-damage|p2a: Unown-Q|70/100",
            "|message|Unrelated: mrmime",
        ]
        .map(String::from);
        assert_eq!(
            canonical_branch_species(&lines, &ctx),
            [
                "|switch|p1a: Deoxys-Defense|Deoxys-Defense, L80, M|50/100",
                "|move|p1a: Mr. Mime|Psychic|p2a: Unown-Q",
                "|-formechange|p1a: Deoxys-Defense|Deoxys-Defense|[from] test",
                "|-damage|p2a: Unown-Q|70/100",
                "|message|Unrelated: mrmime",
            ]
        );
    }

    #[test]
    fn projection_preserves_public_percent_and_hides_exact_maxima() {
        let lines = [
            "|switch|p1a: A|A, L80|999/1000",
            "|-sethp|p1a: A|5/10|p2a: B|10/20 brn",
            "|-sethp|p2a: Wigglytuff|128/407|[from] move: Pain Split|[silent]",
            "|-sethp|p1a: Dusclops|128/209|[from] move: Pain Split",
            "|-sethp|p1a: A|5/10|p2a: B|10/20 brn|[from] move: Pain Split",
            "|upkeep ",
        ]
        .map(String::from);
        assert_eq!(
            project_public_hp(&lines, [false, false]).unwrap(),
            [
                "|switch|p1a: A|A, L80|99/100",
                "|-sethp|p1a: A|50/100|p2a: B|50/100 brn",
                "|-sethp|p2a: Wigglytuff|32/100|[from] move: Pain Split|[silent]",
                "|-sethp|p1a: Dusclops|62/100|[from] move: Pain Split",
                "|-sethp|p1a: A|50/100|p2a: B|50/100 brn|[from] move: Pain Split",
                "|upkeep "
            ]
        );
        let percent = ["|-damage|p2a: B|67/100".into()];
        assert_eq!(project_public_hp(&percent, [false, true]).unwrap(), percent);
    }

    #[test]
    fn invalid_or_private_events_never_enter_callback_prefix() {
        Python::initialize();
        for line in [
            "|request|{}",
            "|split|p1",
            "|-damage|p3a: A|10/100",
            "|-heal|p1a: A|10/0",
            "|-sethp|p1a: A|10/20|p2a: B",
            "|-sethp|p1a: A|10/20|p3a: B|10/20",
            "|-sethp|p1a: A|10/20|[from] move: Pain Split|10/20",
        ] {
            assert!(project_public_hp(&[line.into()], [false, false]).is_err());
        }
        assert!(project_public_hp(&["|-damage|p2a: A|20/30".into()], [false, true]).is_err());
    }

    #[test]
    fn private_pp_uses_root_base_once_at_both_engine_thresholds_and_seats() {
        let state = crate::parse_state(include_str!("test_fixtures/minimal.state").trim()).unwrap();
        for side_one in [true, false] {
            let root = if side_one {
                &state.side_one
            } else {
                &state.side_two
            };
            let species = if side_one { "Charmander" } else { "Squirtle" };
            let move_id = if side_one { "ember" } else { "watergun" };
            let mut meta = crate::leaf::LeafMeta::default();
            meta.move_charges.insert(
                (
                    usize::from(!side_one),
                    species.to_lowercase(),
                    move_id.into(),
                ),
                2,
            );
            // A charge for the other seat must not affect this own request.
            meta.move_charges.insert(
                (
                    usize::from(side_one),
                    species.to_lowercase(),
                    move_id.into(),
                ),
                99,
            );
            let high = corrected_own_pp(
                root,
                root,
                &[species.into()],
                side_one,
                Some(&meta.move_charges),
            )
            .unwrap();
            assert_eq!(high.get_active_immutable().moves.m0.pp, 30);
            let mut low_root = root.clone();
            low_root.get_active().moves.m0.pp = 5;
            let mut low_leaf = low_root.clone();
            low_leaf.get_active().moves.m0.pp = 3;
            let low = corrected_own_pp(
                &low_leaf,
                &low_root,
                &[species.into()],
                side_one,
                Some(&meta.move_charges),
            )
            .unwrap();
            assert_eq!(low.get_active_immutable().moves.m0.pp, 3);
            assert_eq!(root.get_active_immutable().moves.m0.pp, 32);
            assert_eq!(low_leaf.get_active_immutable().moves.m0.pp, 3);
        }
    }

    #[test]
    fn transform_copied_pp_and_original_restoration_use_separate_banks() {
        use poke_engine::state::{PokemonMoveIndex, PreTransform};
        let state = crate::parse_state(include_str!("test_fixtures/minimal.state").trim()).unwrap();
        for side_one in [true, false] {
            let root = if side_one {
                &state.side_one
            } else {
                &state.side_two
            };
            let name = if side_one { "Charmander" } else { "Squirtle" };
            let original_move = if side_one { "ember" } else { "watergun" };
            let mut copied = root.clone();
            let snapshot = PreTransform::capture(copied.get_active_immutable());
            copied.get_active().pre_transform = Some(Box::new(snapshot));
            copied
                .get_active()
                .replace_move(PokemonMoveIndex::M0, Choices::THUNDERBOLT);
            copied.get_active().moves.m0.pp = 3;
            copied.get_active().moves.m1.pp = 5;
            let charges = HashMap::from([
                (
                    (
                        usize::from(!side_one),
                        name.to_lowercase(),
                        "thunderbolt".into(),
                    ),
                    9,
                ),
                (
                    (
                        usize::from(!side_one),
                        name.to_lowercase(),
                        original_move.into(),
                    ),
                    2,
                ),
            ]);
            let corrected =
                corrected_own_pp(&copied, root, &[name.into()], side_one, Some(&charges)).unwrap();
            assert_eq!(
                corrected.get_active_immutable().moves.m0.pp,
                3,
                "copied PP must not be double-charged"
            );
            assert_eq!(
                copied.get_active_immutable().moves.m0.pp,
                3,
                "caller remains unchanged"
            );
            let restored =
                corrected_own_pp(root, &copied, &[name.into()], side_one, Some(&charges)).unwrap();
            assert_eq!(
                restored.get_active_immutable().moves.m0.pp,
                root.get_active_immutable().moves.m0.pp - 2
            );
            copied.get_active().moves.m0.pp = 6;
            Python::initialize();
            assert!(
                corrected_own_pp(&copied, root, &[name.into()], side_one, Some(&charges)).is_err()
            );
        }
    }

    #[test]
    fn original_charge_ledger_excludes_retransform_and_pressure_on_copied_bank() {
        for side_one in [true, false] {
            let seat = usize::from(!side_one);
            let prefix = if side_one { "p1a" } else { "p2a" };
            let mut ledger = OwnPpLedger {
                active: "ditto".into(),
                ..OwnPpLedger::default()
            };
            let transform = (seat, "ditto".into(), "transform".into());
            let thunderbolt = (seat, "ditto".into(), "thunderbolt".into());
            let mut counts = HashMap::from([(transform.clone(), 1)]);
            ledger
                .advance(
                    &HashMap::new(),
                    &counts,
                    &[format!("|-transform|{prefix}: Ditto|p2a: Mew")],
                    side_one,
                )
                .unwrap();
            assert_eq!(ledger.original_charges[&transform], 1);
            let previous = counts.clone();
            counts.insert(transform.clone(), 2); // Copied Transform: a new temporary bank.
            counts.insert(thunderbolt.clone(), 2); // A copied move paid Pressure.
            counts.insert((1 - seat, "ditto".into(), "transform".into()), 99);
            ledger
                .advance(
                    &previous,
                    &counts,
                    &[format!("|-transform|{prefix}: Ditto|p2a: Zapdos")],
                    side_one,
                )
                .unwrap();
            assert_eq!(ledger.original_charges[&transform], 1);
            assert!(!ledger.original_charges.contains_key(&thunderbolt));
            // Switching out restores the original bank. Subsequent original
            // Transform uses cost that bank again, without counting copied uses.
            ledger
                .advance(
                    &counts,
                    &counts,
                    &[format!("|switch|{prefix}: Squirtle|Squirtle|100/100")],
                    side_one,
                )
                .unwrap();
            assert!(!ledger.transformed.contains("ditto"));
            ledger
                .advance(
                    &counts,
                    &counts,
                    &[format!("|switch|{prefix}: Ditto|Ditto|100/100")],
                    side_one,
                )
                .unwrap();
            let previous = counts.clone();
            counts.insert(transform.clone(), 3);
            ledger
                .advance(
                    &previous,
                    &counts,
                    &[format!("|-transform|{prefix}: Ditto|p2a: Gengar")],
                    side_one,
                )
                .unwrap();
            assert_eq!(ledger.original_charges[&transform], 2);
            let mut sibling = ledger.clone();
            sibling
                .advance(
                    &counts,
                    &counts,
                    &[format!("|drag|{prefix}: Squirtle|Squirtle|100/100")],
                    side_one,
                )
                .unwrap();
            assert!(
                ledger.transformed.contains("ditto"),
                "sibling must not mutate retained prefix"
            );
            assert!(!sibling.transformed.contains("ditto"));
        }
    }

    #[test]
    fn an_unexplained_private_move_change_still_refuses_with_own_only_witness() {
        use poke_engine::state::PokemonMoveIndex;
        Python::initialize();
        let state = crate::parse_state(include_str!("test_fixtures/minimal.state").trim()).unwrap();
        let root = &state.side_one;
        let mut invalid = root.clone();
        invalid
            .get_active()
            .replace_move(PokemonMoveIndex::M0, Choices::THUNDERBOLT);
        let error = corrected_own_pp(
            &invalid,
            root,
            &["Charmander".into()],
            true,
            Some(&HashMap::new()),
        )
        .unwrap_err();
        Python::attach(|py| {
            let raw: String = error
                .value(py)
                .getattr("policy_opponent_diagnostic")
                .unwrap()
                .extract()
                .unwrap();
            let value: serde_json::Value = serde_json::from_str(&raw).unwrap();
            assert_eq!(value["kind"], "private_move_identity_mismatch");
            assert_eq!(value["root_move"], "EMBER");
            assert_eq!(value["branch_move"], "THUNDERBOLT");
            assert_eq!(value["seat"], "p1");
            assert_eq!(value["diagnostic_only_not_policy_input"], true);
            assert!(
                !raw.contains("Squirtle"),
                "other seat must not enter the witness"
            );
        });
    }
}
