//! Correctness-first synchronous own-policy provider for native model search.
//! The Python callback receives only a sampled OWN request plus PUBLIC branch
//! events. Neither a complete engine State nor the other seat's private facts
//! cross the boundary. Subject leaf/value/PUCT evaluation remains in model.rs.

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::time::Instant;

use poke_engine::choices::Choices;
use poke_engine::state::{Side, State};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use serde_json::json;

use crate::events::EventContext;
use crate::policy_opponent::ActionDistribution;
use crate::tree::DecisionNode;

type BranchKey = (usize, usize);

struct PublicPrefix {
    lines: Vec<String>,
    meta: crate::leaf::LeafMeta,
}

pub(crate) struct PolicyOpponentBridge {
    callback: Py<PyAny>,
    opponent_side_one: bool,
    root_order: Vec<String>,
    max_pp: HashMap<String, i64>,
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
    pub(crate) fn new(
        callback: Py<PyAny>,
        opponent_side_one: bool,
        root_order: Vec<String>,
        max_pp: HashMap<String, i64>,
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
        let mut prefix = match parent {
            None => Vec::new(),
            Some(parent) => prefixes
                .get(&parent)
                .map(|prefix| prefix.lines.clone())
                .ok_or_else(|| {
                    PyValueError::new_err("policy opponent: parent public prefix missing")
                })?,
        };
        // Project BEFORE invoking user/Python code, rather than letting a
        // callback see exact hidden HP and trusting it not to use those facts.
        prefix.extend(project_public_hp(
            &canonical_branch_species(lines, &self.display_ctx),
            ctx.hp_percent,
        )?);
        prefixes.insert(
            key,
            PublicPrefix {
                lines: prefix,
                meta: meta.clone(),
            },
        );
        Ok(())
    }

    pub(crate) fn provide(
        &self,
        state: &State,
        node: &DecisionNode,
        parent: Option<BranchKey>,
        _ctx: &EventContext,
    ) -> PyResult<ActionDistribution> {
        let started = Instant::now();
        let result = (|| {
            let (lines, meta) = match parent {
                None => (Vec::new(), None),
                Some(key) => self
                    .prefixes
                    .borrow()
                    .get(&key)
                    .map(|prefix| (prefix.lines.clone(), Some(prefix.meta.clone())))
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
            let own = corrected_own_pp(
                side,
                &self.root_own_side,
                species,
                self.opponent_side_one,
                meta.as_ref(),
            )?;
            let opponent_replacing = if self.opponent_side_one {
                state.side_two.force_switch
            } else {
                state.side_one.force_switch
            };
            let bundle = crate::policy_request::sampled_side_request_in_phase(
                &own,
                slot,
                species,
                &order,
                options,
                &self.max_pp,
                parent.is_none(),
                opponent_replacing,
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
    meta: Option<&crate::leaf::LeafMeta>,
) -> PyResult<Side> {
    let mut own = side.clone();
    let Some(meta) = meta else {
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
            let (id, _, _) = crate::policy_request::move_names(mv.id)?;
            let base = &root.pokemon.pkmn[physical].moves[&slot];
            if base.id != mv.id {
                return Err(PyValueError::new_err(
                    "policy opponent: changed private move identity",
                ));
            }
            let charged = meta
                .move_charges
                .get(&(usize::from(!side_one), species_key.clone(), id))
                .copied()
                .unwrap_or(0);
            if charged < 0 {
                return Err(PyValueError::new_err(
                    "policy opponent: negative private PP charge",
                ));
            }
            own.pokemon.pkmn[physical].moves[&slot].pp =
                (i64::from(base.pp) - charged).max(0) as i8;
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
            let high =
                corrected_own_pp(root, root, &[species.into()], side_one, Some(&meta)).unwrap();
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
                Some(&meta),
            )
            .unwrap();
            assert_eq!(low.get_active_immutable().moves.m0.pp, 3);
            assert_eq!(root.get_active_immutable().moves.m0.pp, 32);
            assert_eq!(low_leaf.get_active_immutable().moves.m0.pp, 3);
        }
    }
}
