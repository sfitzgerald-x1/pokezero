//! Instruction→event mapping (track B): render one enumerated engine outcome
//! (a chance branch's `Vec<Instruction>`) as the Showdown protocol lines the
//! real game would have emitted, so search leaves can advance REAL fold state
//! per outcome (plan v3 search-tree contract, item 2: per-outcome fold-state
//! advance — no freezing, no stale history).
//!
//! # Position in the pipeline
//!
//! A chance branch (tree.rs) carries (pre-decision `State`, the joint
//! `MoveChoice` pair, the branch's instruction list). This module maps that
//! triple + an [`EventContext`] (display species per party slot, current turn
//! number) to protocol lines satisfying fold.rs's input contract (plain ASCII
//! integers, well-formed `p1a: Species` idents). The fold then advances a
//! CLONE of the root fold state over those lines — the seam
//! `multiply_batched_core` consumes via [`crate::tree::BranchSeam`].
//!
//! # Method: phase segmentation by re-generation
//!
//! The engine's instruction list is an unlabeled state-delta stream: it does
//! not say which move produced an instruction, who moved first, or where the
//! end-of-turn residual phase begins. All three are recovered EXACTLY by
//! re-running the engine's own (public) per-move generator: phase 1 =
//! `generate_instructions_from_move(first mover)` must produce a prefix of
//! the branch; phase 2 (second mover, `first_move=false`, phase-1 prefix as
//! `incoming`) must extend it; the remaining tail is the end-of-turn segment
//! (`add_end_of_turn_instructions` output). Generation is deterministic, so
//! the match is exact; a branch that fails to segment is reported as
//! attribution-unsafe and is never allowed through the fold/encoder path.
//!
//! # Honest limits (see docs/crate_search_design.md for the full table)
//!
//! Some real-protocol distinctions are NOT recoverable from the instruction
//! stream, because the engine itself merges outcomes with identical deltas
//! (`combine_duplicate_instructions`):
//! - full-paralysis vs. miss (both: empty delta) — rendered as `|cant|..|par`
//!   (the usually-larger probability mass), documented ambiguity;
//! - the KO-straddle branch conflates "high roll" and "crit" — no `|-crit|`
//!   is emitted for it;
//! - Sleep Talk's called move id is not in the delta — an unidentified call is
//!   attribution-unsafe rather than assigned to an invented action window.
//!
//! Lines the fold provably ignores (fold.rs `process_line`) are deliberately
//! NOT rendered: `|-singleturn|`, `|-curestatus|`, `|-fail|`, `|-ability|`,
//! `|-enditem|`, `|-mustrecharge|`, `|-start|` (except absorb signatures),
//! `|-anim|`, `|debug|`. Omissions are part of the documented contract.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use poke_engine::choices::{Boost, Choice, Choices, MoveCategory, MoveTarget};
use poke_engine::engine::abilities::Abilities;
use poke_engine::engine::damage_calc::type_effectiveness_modifier;
use poke_engine::engine::generate_instructions::{
    calculate_both_damage_rolls, generate_instructions_from_move,
    generate_instructions_from_move_pair,
};
use poke_engine::engine::items::Items;
use poke_engine::engine::state::{MoveChoice, PokemonVolatileStatus, Weather};
use poke_engine::instruction::{ChangeStatusInstruction, Instruction, StateInstructions};
use poke_engine::state::{
    PokemonBoostableStat, PokemonGender, PokemonIndex, PokemonSideCondition, PokemonStatus,
    PokemonType, SideReference, State,
};

use crate::parse_state;

// ---------------------------------------------------------------------------
// Context
// ---------------------------------------------------------------------------

/// Rendering context the engine state cannot supply: display species per
/// party slot (protocol species strings, e.g. "Mr. Mime", where the engine
/// only has enum names) and the current battle turn number.
#[derive(Clone, Debug)]
pub struct EventContext {
    /// Display species per side (index 0 = side one / p1), engine party order.
    pub species: [Vec<String>; 2],
    /// The fold's current turn number at the decision boundary.
    pub turn: i64,
    /// Sides whose HP lines render on the /100 base (Showdown HP Percentage
    /// Mod — what a live-ladder stream shows for the OPPONENT side). The
    /// local harness's omniscient stream reports exact HP for both sides, so
    /// the default is exact everywhere; set the opponent side when the root
    /// fold was built from a ladder (player-view) stream so leaf-synthesized
    /// fractions land on the same /100 grid the root fold consumed.
    pub hp_percent: [bool; 2],
}

impl EventContext {
    pub fn from_json(json: &str) -> Result<EventContext, String> {
        let value: serde_json::Value =
            serde_json::from_str(json).map_err(|e| format!("ctx json: {e}"))?;
        let mut species: [Vec<String>; 2] = [Vec::new(), Vec::new()];
        for (key, out) in [("p1", 0usize), ("p2", 1usize)] {
            let arr = value
                .get(key)
                .and_then(|v| v.as_array())
                .ok_or_else(|| format!("ctx json: missing {key} species array"))?;
            out_species(&mut species[out], arr)?;
        }
        let turn = value
            .get("turn")
            .and_then(|v| v.as_i64())
            .ok_or("ctx json: missing integer turn")?;
        let mut hp_percent = [false, false];
        if let Some(sides) = value.get("hp_percent").and_then(|v| v.as_array()) {
            for entry in sides {
                match entry.as_str() {
                    Some("p1") => hp_percent[0] = true,
                    Some("p2") => hp_percent[1] = true,
                    other => return Err(format!("ctx json: bad hp_percent entry {other:?}")),
                }
            }
        }
        Ok(EventContext {
            species,
            turn,
            hp_percent,
        })
    }

    fn display(&self, side: SideReference, index: PokemonIndex) -> String {
        let list = &self.species[side_usize(side)];
        let i = pokemon_index_usize(index);
        list.get(i)
            .cloned()
            .unwrap_or_else(|| format!("unknown{}", i))
    }

    fn details(&self, state: &State, side: SideReference, index: PokemonIndex) -> String {
        let pokemon = &state.get_side_immutable(&side).pokemon[index];
        let level = if pokemon.level == 100 {
            String::new()
        } else {
            format!(", L{}", pokemon.level)
        };
        let gender = match pokemon.gender {
            PokemonGender::MALE => ", M",
            PokemonGender::FEMALE => ", F",
            PokemonGender::NONE => "",
        };
        format!("{}{level}{gender}", self.display(side, index))
    }

    fn ident(&self, side: SideReference, index: PokemonIndex) -> String {
        format!("{}a: {}", side_prefix(side), self.display(side, index))
    }

    fn active_ident(&self, state: &State, side: SideReference) -> String {
        let active = match side {
            SideReference::SideOne => state.side_one.active_index,
            SideReference::SideTwo => state.side_two.active_index,
        };
        self.ident(side, active)
    }
}

fn out_species(out: &mut Vec<String>, arr: &[serde_json::Value]) -> Result<(), String> {
    for entry in arr {
        out.push(
            entry
                .as_str()
                .ok_or("ctx json: species entries must be strings")?
                .to_string(),
        );
    }
    Ok(())
}

fn side_usize(side: SideReference) -> usize {
    match side {
        SideReference::SideOne => 0,
        SideReference::SideTwo => 1,
    }
}

fn side_prefix(side: SideReference) -> &'static str {
    match side {
        SideReference::SideOne => "p1",
        SideReference::SideTwo => "p2",
    }
}

fn pokemon_index_usize(index: PokemonIndex) -> usize {
    index.serialize().parse::<usize>().unwrap_or(0)
}

// ---------------------------------------------------------------------------
// Rendered output
// ---------------------------------------------------------------------------

/// One branch's rendered protocol lines plus bookkeeping the caller needs.
#[derive(Clone, Debug, Default)]
pub struct RenderedEvents {
    pub lines: Vec<String>,
    /// True when this ply emitted `|turn|N+1` (the caller advances its turn
    /// counter for deeper plies).
    pub turn_completed: bool,
    /// Non-empty when rendering lost some diagnostic fidelity; each entry is
    /// a stable reason slug. A reason may be telemetry-only when the emitted
    /// action attribution remains exact.
    pub lossy: Vec<String>,
    /// Stable subset of [`Self::lossy`] which cannot safely cross the
    /// renderer-to-fold boundary. Production model search and env stepping
    /// reject these branches before fold/encoder advancement instead of
    /// silently inventing action evidence or dropping chance mass.
    pub attribution_unsafe: Vec<String>,
    /// Internal status transitions for the leaf's line-driven ledgers. These
    /// deliberately do not add protocol text for fold-ignored cure events.
    pub(crate) active_status_transitions: Vec<ActiveStatusTransition>,
}

#[derive(Clone, Debug)]
pub(crate) struct ActiveStatusTransition {
    /// Apply after this many rendered lines, preserving instruction order.
    pub(crate) line_offset: usize,
    pub(crate) side: usize,
    pub(crate) new_status: PokemonStatus,
}

fn active_status_transition(
    state: &State,
    change: &ChangeStatusInstruction,
) -> Option<ActiveStatusTransition> {
    (state.get_side_immutable(&change.side_ref).active_index == change.pokemon_index).then_some(
        ActiveStatusTransition {
            line_offset: 0,
            side: side_usize(change.side_ref),
            new_status: change.new_status,
        },
    )
}

impl RenderedEvents {
    // `&str`, not `&'static str`. Both of these immediately `.to_string()`, so
    // the 'static bound was incidental rather than a deliberate guard against
    // unbounded reason cardinality -- and nothing downstream is a metrics sink:
    // every aggregator over `world_failure_reasons` is a plain Counter/dict
    // merge with no top-N, no label cap and no Prometheus/wandb export. Widened
    // so the attract refusal can name the JOINT set of live predicates, which a
    // fixed table of 31 boolean combinations could not do readably.
    fn mark_lossy(&mut self, reason: &str) {
        self.lossy.push(reason.to_string());
    }

    fn mark_attribution_unsafe(&mut self, reason: &str) {
        self.mark_lossy(reason);
        self.attribution_unsafe.push(reason.to_string());
    }

    pub fn is_attribution_unsafe(&self) -> bool {
        !self.attribution_unsafe.is_empty()
    }
}

/// Refuse an event stream whose action attribution is not observable from the
/// engine delta. Callers must do this before advancing a fold or encoding a
/// leaf: treating a rejected chance branch as a zero-weight branch would lose
/// probability mass, so model search lets the normal world-fallback path own
/// the whole world instead.
pub fn reject_attribution_unsafe(rendered: &RenderedEvents, lane: &str) -> PyResult<()> {
    if rendered.is_attribution_unsafe() {
        // DEDUPE before joining. The Python seam truncates this message at 160
        // chars to build a `world_failure_reasons` key, and the prefix eats 68 of
        // them -- so with two sides refusing for the SAME reason the duplicate
        // pushed the second copy past the cliff and minted a garbage bucket
        // (`...paralyzed+mis`, `...paralyzed+can`). A truncated key silently drops
        // the label it cut, which for the attract sub-cases means hiding exactly
        // the non-downgradeable mass the measurement exists to find.
        //
        // Both sides refusing identically is the common case, not the exotic one,
        // so deduping removes the overflow at its source rather than widening the
        // limit and waiting for a longer slug to cross it again.
        let mut reasons: Vec<&str> = Vec::with_capacity(rendered.attribution_unsafe.len());
        for reason in &rendered.attribution_unsafe {
            if !reasons.contains(&reason.as_str()) {
                reasons.push(reason);
            }
        }
        return Err(PyValueError::new_err(format!(
            "attribution-unsafe renderer branch rejected before {lane}: {}",
            reasons.join(",")
        )));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Choice preparation + move order (replicas of private engine helpers)
// ---------------------------------------------------------------------------

/// Replica of `generate_instructions_from_move_pair`'s choice preparation.
fn build_choice(state: &State, side: SideReference, mc: &MoveChoice) -> Choice {
    let side_ref = match side {
        SideReference::SideOne => &state.side_one,
        SideReference::SideTwo => &state.side_two,
    };
    match mc {
        MoveChoice::Switch(switch_id) => {
            let mut c = Choice::default();
            c.switch_id = *switch_id;
            c.category = MoveCategory::Switch;
            c
        }
        MoveChoice::Move(move_index) => {
            let mut c = side_ref.get_active_immutable().moves[move_index]
                .choice
                .clone();
            c.move_index = *move_index;
            c
        }
        MoveChoice::None => Choice::default(),
    }
}

/// Replica of the private `get_effective_speed` (gen3).
fn effective_speed(state: &State, side: SideReference) -> i16 {
    let side_ref = match side {
        SideReference::SideOne => &state.side_one,
        SideReference::SideTwo => &state.side_two,
    };
    let active = side_ref.get_active_immutable();
    let mut speed = side_ref.calculate_boosted_stat(PokemonBoostableStat::Speed) as f32;
    match state.weather.weather_type {
        Weather::SUN if active.ability == Abilities::CHLOROPHYLL => speed *= 2.0,
        Weather::RAIN if active.ability == Abilities::SWIFTSWIM => speed *= 2.0,
        _ => {}
    }
    if side_ref
        .volatile_statuses
        .contains(&PokemonVolatileStatus::SLOWSTART)
    {
        speed *= 0.5;
    }
    if active.status == PokemonStatus::PARALYZE {
        speed *= 0.25;
    }
    speed as i16
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Order {
    SideOne,
    SideTwo,
    Tie,
}

/// Replica of the private `moves_first` (gen3).
fn move_order(state: &State, c1: &Choice, c2: &Choice) -> Order {
    let s1 = effective_speed(state, SideReference::SideOne);
    let s2 = effective_speed(state, SideReference::SideTwo);
    if c1.category == MoveCategory::Switch && c2.category == MoveCategory::Switch {
        return if s1 > s2 {
            Order::SideOne
        } else if s1 == s2 {
            Order::Tie
        } else {
            Order::SideTwo
        };
    } else if c1.category == MoveCategory::Switch {
        return if c2.move_id != Choices::PURSUIT {
            Order::SideOne
        } else {
            Order::SideTwo
        };
    } else if c2.category == MoveCategory::Switch {
        return if c1.move_id == Choices::PURSUIT {
            Order::SideOne
        } else {
            Order::SideTwo
        };
    }
    if c1.priority == c2.priority {
        if s1 == s2 {
            Order::Tie
        } else if s1 > s2 {
            Order::SideOne
        } else {
            Order::SideTwo
        }
    } else if c1.priority > c2.priority {
        Order::SideOne
    } else {
        Order::SideTwo
    }
}

/// Replica of the private `end_of_turn_triggered`.
///
/// Mirrors the engine's recharge-turn-residuals patch: a (switch, None) ply
/// where the None side still carries MUSTRECHARGE is a full turn — the engine
/// now emits the whole end-of-turn block on it, so segmentation must expect
/// an end-of-turn phase there (previously these plies ended at the volatile
/// removal and the residual instructions failed the grammar as
/// `segmentation_failed`).
fn end_of_turn_triggered(state: &State, s1: &MoveChoice, s2: &MoveChoice) -> bool {
    if state.side_one.force_switch || state.side_two.force_switch {
        return true;
    }
    if (s1 == &MoveChoice::None
        && state
            .side_one
            .volatile_statuses
            .contains(&PokemonVolatileStatus::MUSTRECHARGE))
        || (s2 == &MoveChoice::None
            && state
                .side_two
                .volatile_statuses
                .contains(&PokemonVolatileStatus::MUSTRECHARGE))
    {
        return true;
    }
    !(matches!(s1, MoveChoice::Switch(_)) && s2 == &MoveChoice::None)
        && !(s1 == &MoveChoice::None && matches!(s2, MoveChoice::Switch(_)))
}

// ---------------------------------------------------------------------------
// Segmentation by re-generation
// ---------------------------------------------------------------------------

struct Segmentation {
    first: SideReference,
    /// End (exclusive) of the first mover's phase in the branch list.
    p1_end: usize,
    /// End (exclusive) of the second mover's phase; the rest is end-of-turn.
    p2_end: usize,
    /// The first mover's choice AFTER the engine's own mutation pass
    /// (encore redirection, protect stripping, charge conversion...).
    first_choice: Choice,
    /// The second mover's mutated choice.
    second_choice: Choice,
}

fn is_prefix(prefix: &[Instruction], full: &[Instruction]) -> bool {
    prefix.len() <= full.len() && prefix == &full[..prefix.len()]
}

/// Segment `full` into first-move / second-move / end-of-turn phases by
/// re-running the engine's own per-move generation and prefix-matching.
fn segment(
    state: &mut State,
    s1_move: &MoveChoice,
    s2_move: &MoveChoice,
    full: &[Instruction],
    branch_on_damage: bool,
) -> Option<Segmentation> {
    let c1 = build_choice(state, SideReference::SideOne, s1_move);
    let c2 = build_choice(state, SideReference::SideTwo, s2_move);
    let orders: Vec<Order> = match move_order(state, &c1, &c2) {
        Order::Tie => vec![Order::SideOne, Order::SideTwo],
        other => vec![other],
    };
    let eot = end_of_turn_triggered(state, s1_move, s2_move);

    for order in orders {
        let (first_ref, mut first_choice, mut second_choice) = match order {
            Order::SideTwo => (SideReference::SideTwo, c2.clone(), c1.clone()),
            _ => (SideReference::SideOne, c1.clone(), c2.clone()),
        };
        let second_ref = match first_ref {
            SideReference::SideOne => SideReference::SideTwo,
            SideReference::SideTwo => SideReference::SideOne,
        };

        let mut phase1: Vec<StateInstructions> = Vec::with_capacity(4);
        generate_instructions_from_move(
            state,
            &mut first_choice,
            &second_choice,
            first_ref,
            StateInstructions::default(),
            &mut phase1,
            branch_on_damage,
        );
        second_choice.first_move = false;

        // Longest matching phase-1 prefix first: greedy but verified by the
        // phase-2 continuation, so a shorter prefix still wins when it is the
        // only one with a consistent continuation.
        let mut candidates: Vec<&StateInstructions> = phase1
            .iter()
            .filter(|b| is_prefix(&b.instruction_list, full))
            .collect();
        candidates.sort_by_key(|b| std::cmp::Reverse(b.instruction_list.len()));

        for p1 in candidates {
            let incoming = StateInstructions {
                percentage: 100.0,
                instruction_list: p1.instruction_list.clone(),
            };
            let mut phase2: Vec<StateInstructions> = Vec::with_capacity(4);
            let mut second_mut = second_choice.clone();
            generate_instructions_from_move(
                state,
                &mut second_mut,
                &first_choice,
                second_ref,
                incoming,
                &mut phase2,
                branch_on_damage,
            );
            let mut best: Option<usize> = None; // p2 end index
            for p2 in &phase2 {
                let list = &p2.instruction_list;
                if list.len() < p1.instruction_list.len() || !is_prefix(list, full) {
                    continue;
                }
                if !eot && list.len() != full.len() {
                    continue;
                }
                if best.map_or(true, |b| list.len() > b) {
                    best = Some(list.len());
                }
            }
            if let Some(p2_end) = best {
                return Some(Segmentation {
                    first: first_ref,
                    p1_end: p1.instruction_list.len(),
                    p2_end,
                    first_choice,
                    second_choice: second_mut,
                });
            }
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Sim: incremental application with rendering reads
// ---------------------------------------------------------------------------

struct Sim<'a> {
    state: &'a mut State,
    applied: Vec<Instruction>,
    /// Per-side /100 HP rendering (live-ladder streams show opponent HP under
    /// the HP Percentage Mod; the local harness's omniscient stream is exact).
    hp_percent: [bool; 2],
}

impl<'a> Sim<'a> {
    fn new(state: &'a mut State, hp_percent: [bool; 2]) -> Sim<'a> {
        Sim {
            state,
            applied: Vec::new(),
            hp_percent,
        }
    }

    fn apply(&mut self, instruction: &Instruction) {
        self.state.apply_one_instruction(instruction);
        self.applied.push(instruction.clone());
    }

    fn active_hp(&self, side: SideReference) -> (i16, i16) {
        let s = match side {
            SideReference::SideOne => &self.state.side_one,
            SideReference::SideTwo => &self.state.side_two,
        };
        let active = s.get_active_immutable();
        (active.hp, active.maxhp)
    }

    fn hp_condition(&self, side: SideReference) -> String {
        let (hp, maxhp) = self.active_hp(side);
        if hp <= 0 {
            return "0 fnt".to_string();
        }
        if self.hp_percent[side_usize(side)] && maxhp > 0 {
            return hp_percent_condition(hp, maxhp);
        }
        format!("{hp}/{maxhp}")
    }

    fn finish(self) {
        // Restore the caller's state exactly (reverse in reverse order).
        self.state.reverse_instructions(&self.applied);
    }
}

/// Showdown's HP Percentage Mod rendering (sim/pokemon.ts `getHealth`):
/// `ceil(100 * hp / maxhp)`, with 100 shown as 99 while hp < maxhp.
fn hp_percent_condition(hp: i16, maxhp: i16) -> String {
    let mut pct = (100 * hp as i32 + maxhp as i32 - 1) / maxhp as i32;
    if pct == 100 && hp < maxhp {
        pct = 99;
    }
    format!("{pct}/100")
}

fn other_side(side: SideReference) -> SideReference {
    match side {
        SideReference::SideOne => SideReference::SideTwo,
        SideReference::SideTwo => SideReference::SideOne,
    }
}

fn instruction_side(ins: &Instruction) -> Option<SideReference> {
    Some(match ins {
        Instruction::Switch(i) => i.side_ref,
        Instruction::ApplyVolatileStatus(i) => i.side_ref,
        Instruction::RemoveVolatileStatus(i) => i.side_ref,
        Instruction::ChangeStatus(i) => i.side_ref,
        Instruction::Heal(i) => i.side_ref,
        Instruction::Damage(i) => i.side_ref,
        Instruction::Boost(i) => i.side_ref,
        Instruction::ChangeSideCondition(i) => i.side_ref,
        Instruction::ChangeVolatileStatusDuration(i) => i.side_ref,
        Instruction::DamageSubstitute(i) => i.side_ref,
        Instruction::DecrementRestTurns(i) => i.side_ref,
        Instruction::SetRestTurns(i) => i.side_ref,
        Instruction::SetSleepTurns(i) => i.side_ref,
        Instruction::ChangeSubstituteHealth(i) => i.side_ref,
        Instruction::DecrementPP(i) => i.side_ref,
        Instruction::ChangeItem(i) => i.side_ref,
        Instruction::ChangeAbility(i) => i.side_ref,
        Instruction::ChangeType(i) => i.side_ref,
        Instruction::FormeChange(i) => i.side_ref,
        Instruction::ChangeWish(i) => i.side_ref,
        Instruction::DecrementWish(i) => i.side_ref,
        Instruction::SetFutureSight(i) => i.side_ref,
        Instruction::DecrementFutureSight(i) => i.side_ref,
        Instruction::DisableMove(i) => i.side_ref,
        Instruction::EnableMove(i) => i.side_ref,
        Instruction::SetLastUsedMove(i) => i.side_ref,
        Instruction::ChangeDamageDealtDamage(i) => i.side_ref,
        Instruction::ChangeDamageDealtMoveCatagory(i) => i.side_ref,
        Instruction::ToggleDamageDealtHitSubstitute(i) => i.side_ref,
        Instruction::ToggleBatonPassing(i) => i.side_ref,
        Instruction::ToggleShedTailing(i) => i.side_ref,
        Instruction::ChangeAttack(i) => i.side_ref,
        Instruction::ChangeDefense(i) => i.side_ref,
        Instruction::ChangeSpecialAttack(i) => i.side_ref,
        Instruction::ChangeSpecialDefense(i) => i.side_ref,
        Instruction::ChangeSpeed(i) => i.side_ref,
        _ => return None,
    })
}

// ---------------------------------------------------------------------------
// Display helpers
// ---------------------------------------------------------------------------

/// Move display for `|move|` lines: the engine's enum name lowercased is
/// normalize-equal to the real display name ("Double-Edge" -> "doubleedge"),
/// with the two engine-id aliases the real protocol never shows mapped back
/// ("hiddenpower<type><bp>" -> "hiddenpower", "return102" -> "return").
fn move_display(id: Choices) -> String {
    let name = format!("{:?}", id).to_lowercase();
    if name.starts_with("hiddenpower") {
        return "hiddenpower".to_string();
    }
    if name == "return102" {
        return "return".to_string();
    }
    name
}

fn status_code(status: PokemonStatus) -> Option<&'static str> {
    Some(match status {
        PokemonStatus::BURN => "brn",
        PokemonStatus::SLEEP => "slp",
        PokemonStatus::FREEZE => "frz",
        PokemonStatus::PARALYZE => "par",
        PokemonStatus::POISON => "psn",
        PokemonStatus::TOXIC => "tox",
        PokemonStatus::NONE => return None,
    })
}

fn boost_stat_code(stat: PokemonBoostableStat) -> &'static str {
    match stat {
        PokemonBoostableStat::Attack => "atk",
        PokemonBoostableStat::Defense => "def",
        PokemonBoostableStat::SpecialAttack => "spa",
        PokemonBoostableStat::SpecialDefense => "spd",
        PokemonBoostableStat::Speed => "spe",
        PokemonBoostableStat::Accuracy => "accuracy",
        PokemonBoostableStat::Evasion => "evasion",
    }
}

fn weather_display(weather: Weather) -> Option<&'static str> {
    Some(match weather {
        Weather::SUN => "SunnyDay",
        Weather::RAIN => "RainDance",
        Weather::SAND => "Sandstorm",
        Weather::HAIL => "Hail",
        Weather::NONE => return None,
    })
}

fn side_condition_display(condition: PokemonSideCondition) -> Option<&'static str> {
    Some(match condition {
        PokemonSideCondition::Spikes => "Spikes",
        PokemonSideCondition::Reflect => "Reflect",
        PokemonSideCondition::LightScreen => "Light Screen",
        PokemonSideCondition::Safeguard => "Safeguard",
        PokemonSideCondition::Mist => "Mist",
        // Engine-internal counters with no protocol line.
        _ => return None,
    })
}

/// Charge-move volatiles (gen3): `|-prepare|` is rendered for these.
fn charge_volatile_move(vs: PokemonVolatileStatus) -> Option<Choices> {
    Some(match vs {
        PokemonVolatileStatus::SOLARBEAM => Choices::SOLARBEAM,
        PokemonVolatileStatus::SKULLBASH => Choices::SKULLBASH,
        PokemonVolatileStatus::RAZORWIND => Choices::RAZORWIND,
        PokemonVolatileStatus::SKYATTACK => Choices::SKYATTACK,
        PokemonVolatileStatus::FLY => Choices::FLY,
        PokemonVolatileStatus::DIG => Choices::DIG,
        PokemonVolatileStatus::BOUNCE => Choices::BOUNCE,
        PokemonVolatileStatus::DIVE => Choices::DIVE,
        _ => return None,
    })
}

fn is_absorb_ability(ability: Abilities) -> Option<&'static str> {
    Some(match ability {
        Abilities::VOLTABSORB => "Volt Absorb",
        Abilities::WATERABSORB => "Water Absorb",
        Abilities::FLASHFIRE => "Flash Fire",
        _ => return None,
    })
}

/// Non-absorb ability immunities the gen3 engine models as "no effect"
/// (empty delta): the real protocol shows `|-immune|..|[from] ability: X`.
fn ability_immunity(
    ability: Abilities,
    choice: &Choice,
    effectiveness: f32,
) -> Option<&'static str> {
    use poke_engine::state::PokemonType;
    let damaging = choice.category != MoveCategory::Status;
    let inflicts =
        |status: PokemonStatus| choice.status.as_ref().map_or(false, |s| s.status == status);
    Some(match ability {
        Abilities::LEVITATE if damaging && choice.move_type == PokemonType::GROUND => "Levitate",
        Abilities::WONDERGUARD if damaging && effectiveness <= 1.0 => "Wonder Guard",
        Abilities::IMMUNITY
            if inflicts(PokemonStatus::POISON) || inflicts(PokemonStatus::TOXIC) =>
        {
            "Immunity"
        }
        Abilities::INSOMNIA if inflicts(PokemonStatus::SLEEP) => "Insomnia",
        Abilities::VITALSPIRIT if inflicts(PokemonStatus::SLEEP) => "Vital Spirit",
        Abilities::LIMBER if inflicts(PokemonStatus::PARALYZE) => "Limber",
        Abilities::WATERVEIL if inflicts(PokemonStatus::BURN) => "Water Veil",
        Abilities::MAGMAARMOR if inflicts(PokemonStatus::FREEZE) => "Magma Armor",
        _ => return None,
    })
}

// ---------------------------------------------------------------------------
// The renderer
// ---------------------------------------------------------------------------

/// Render one enumerated outcome as protocol lines. `state` must be the
/// pre-decision state; it is mutated during rendering and restored before
/// returning. `turn` is the fold's current turn number.
pub fn render_branch_events(
    state: &mut State,
    s1_move: &MoveChoice,
    s2_move: &MoveChoice,
    instructions: &[Instruction],
    branch_on_damage: bool,
    ctx: &EventContext,
) -> RenderedEvents {
    let mut out = RenderedEvents::default();
    out.lines.push("|".to_string());

    // Replacement / pivot plies ((switch, none) shapes): no end-of-turn phase.
    let eot_triggered = end_of_turn_triggered(state, s1_move, s2_move);
    // For (switch, none) shapes: was the switching side's active already
    // fainted (faint replacement — the faint ply already ran residuals +
    // upkeep) or alive (pivot — the engine never runs the pivot turn's
    // residuals; documented deviation)?
    let pre_ply_replacement =
        if matches!(s1_move, MoveChoice::Switch(_)) && s2_move == &MoveChoice::None {
            state.side_one.get_active_immutable().hp <= 0
        } else if s1_move == &MoveChoice::None && matches!(s2_move, MoveChoice::Switch(_)) {
            state.side_two.get_active_immutable().hp <= 0
        } else {
            false
        };

    let seg = match segment(state, s1_move, s2_move, instructions, branch_on_damage) {
        Some(seg) => seg,
        None => {
            // The renderer cannot identify action/residual boundaries. Keep
            // the simulation reversible for diagnostic post-state reporting,
            // but emit no partially attributed protocol stream: model/env
            // callers reject this branch before it reaches fold/encoder.
            out.mark_attribution_unsafe("segmentation_failed");
            let mut sim = Sim::new(state, ctx.hp_percent);
            for ins in instructions {
                sim.apply(ins);
            }
            sim.finish();
            return out;
        }
    };

    let second_ref = other_side(seg.first);
    let (first_mc, second_mc) = match seg.first {
        SideReference::SideOne => (s1_move, s2_move),
        SideReference::SideTwo => (s2_move, s1_move),
    };

    let mut sim = Sim::new(state, ctx.hp_percent);
    render_action_phase(
        &mut sim,
        seg.first,
        first_mc,
        &seg.first_choice,
        &instructions[..seg.p1_end],
        branch_on_damage,
        ctx,
        &mut out,
    );
    render_action_phase(
        &mut sim,
        second_ref,
        second_mc,
        &seg.second_choice,
        &instructions[seg.p1_end..seg.p2_end],
        branch_on_damage,
        ctx,
        &mut out,
    );

    let residual_segment = &instructions[seg.p2_end..];
    if eot_triggered {
        out.lines.push("|".to_string());
        let mut plan = ResidualPlan::build(sim.state, residual_segment);
        for (index, ins) in residual_segment.iter().enumerate() {
            render_residual_instruction(
                &mut sim,
                ins,
                residual_segment.get(index + 1),
                &mut plan,
                ctx,
                &mut out,
            );
        }
        // A pivot in flight (U-turn/Baton Pass chose to switch, the engine
        // skipped residuals): the turn is not over — no |upkeep yet.
        if !(sim.state.side_one.force_switch || sim.state.side_two.force_switch) {
            out.lines.push("|upkeep".to_string());
        }
    }
    finish_ply(
        &mut sim,
        s1_move,
        s2_move,
        eot_triggered,
        pre_ply_replacement,
        ctx,
        &mut out,
    );
    sim.finish();
    out
}

/// Emit the `|turn|N+1` line when the ply completes the battle turn:
/// - an end-of-turn ply with no pending replacement completes it;
/// - a faint-replacement ply ((switch, none) with the switcher's active at
///   0 HP before the switch) completes the turn its faint began (the real
///   protocol places the replacement before `|turn|`).
/// Pivot plies (force-switch, attacker alive) also complete the turn — the
/// engine never runs residuals for pivot turns (documented deviation), so
/// `|upkeep` + `|turn|` are emitted with no residual lines.
fn finish_ply(
    sim: &mut Sim<'_>,
    s1_move: &MoveChoice,
    s2_move: &MoveChoice,
    eot_triggered: bool,
    pre_ply_replacement: bool,
    ctx: &EventContext,
    out: &mut RenderedEvents,
) {
    let s1_hp = sim.active_hp(SideReference::SideOne).0;
    let s2_hp = sim.active_hp(SideReference::SideTwo).0;
    let replacement_pending = s1_hp <= 0 || s2_hp <= 0;
    let force_switch_pending = sim.state.side_one.force_switch || sim.state.side_two.force_switch;
    if eot_triggered {
        if !replacement_pending && !force_switch_pending {
            out.lines.push(format!("|turn|{}", ctx.turn + 1));
            out.turn_completed = true;
        }
        return;
    }
    // (switch, none) shapes: replacement or pivot ply.
    let switching_side = if matches!(s1_move, MoveChoice::Switch(_)) {
        Some(SideReference::SideOne)
    } else if matches!(s2_move, MoveChoice::Switch(_)) {
        Some(SideReference::SideTwo)
    } else {
        None
    };
    if switching_side.is_some() && !replacement_pending && !force_switch_pending {
        if !pre_ply_replacement {
            // Pivot follow-up: the engine skipped the pivot turn's residuals
            // entirely (documented deviation), so the turn boundary — with
            // no residual lines — lands here.
            out.lines.push("|".to_string());
            out.lines.push("|upkeep".to_string());
        }
        // Faint replacement: the faint ply already carried residuals +
        // |upkeep; the real protocol places the replacement switch before
        // |turn|, which is exactly where we are now.
        out.lines.push(format!("|turn|{}", ctx.turn + 1));
        out.turn_completed = true;
    }
}

// ---------------------------------------------------------------------------
// Action-phase rendering
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn render_action_phase(
    sim: &mut Sim<'_>,
    side: SideReference,
    mc: &MoveChoice,
    mutated_choice: &Choice,
    segment: &[Instruction],
    branch_on_damage: bool,
    ctx: &EventContext,
    out: &mut RenderedEvents,
) {
    match mc {
        MoveChoice::Switch(_) => render_switch_phase(sim, side, segment, ctx, out),
        MoveChoice::None => render_none_phase(sim, side, segment, ctx, out),
        MoveChoice::Move(_) => render_move_phase(
            sim,
            side,
            mutated_choice,
            segment,
            branch_on_damage,
            ctx,
            out,
            None,
        ),
    }
}

/// A `(switch, ...)` action: `|switch|` with display details, hazard damage
/// with `[from] Spikes`, switch-in ability lines the fold consumes (weather).
fn render_switch_phase(
    sim: &mut Sim<'_>,
    side: SideReference,
    segment: &[Instruction],
    ctx: &EventContext,
    out: &mut RenderedEvents,
) {
    let mut baton_pass = false;
    let mut switched = false;
    for ins in segment {
        match ins {
            Instruction::ToggleBatonPassing(_) => {
                baton_pass = true;
                sim.apply(ins);
            }
            Instruction::Switch(switch) => {
                sim.apply(ins);
                let details = ctx.details(sim.state, switch.side_ref, switch.next_index);
                let ident = ctx.ident(switch.side_ref, switch.next_index);
                let condition = sim.hp_condition(switch.side_ref);
                let mut line = format!("|switch|{ident}|{details}|{condition}");
                if baton_pass {
                    line.push_str("|[from] Baton Pass");
                }
                out.lines.push(line);
                switched = true;
            }
            Instruction::Damage(damage) if switched && damage.side_ref == side => {
                // Spikes chip on the way in.
                sim.apply(ins);
                let ident = ctx.active_ident(sim.state, side);
                let condition = sim.hp_condition(side);
                out.lines
                    .push(format!("|-damage|{ident}|{condition}|[from] Spikes"));
                emit_faint_if_dead(sim, side, ctx, out);
            }
            Instruction::ChangeWeather(change) if switched => {
                sim.apply(ins);
                if let Some(name) = weather_display(change.new_weather) {
                    let ident = ctx.active_ident(sim.state, side);
                    let ability = ability_display_of_active(sim.state, side);
                    out.lines.push(format!(
                        "|-weather|{name}|[from] ability: {ability}|[of] {ident}"
                    ));
                } else {
                    out.lines.push("|-weather|none".to_string());
                }
            }
            Instruction::Boost(boost) if switched && boost.side_ref != side => {
                // Intimidate on entry (real: |-ability| then |-unboost|; the
                // fold only reads the boost line).
                sim.apply(ins);
                out.lines.push(render_boost_line(
                    ctx,
                    sim,
                    boost.side_ref,
                    boost.stat,
                    boost.amount,
                    None,
                ));
            }
            // Pre-switch bookkeeping (volatile clears, boost resets, toxic
            // reset, PARTIALLYTRAPPED release, ability change...): no lines.
            _ => sim.apply(ins),
        }
    }
}

/// A forced `none` action: recharge (`|cant|..|recharge`) or true no-op.
fn render_none_phase(
    sim: &mut Sim<'_>,
    side: SideReference,
    segment: &[Instruction],
    ctx: &EventContext,
    out: &mut RenderedEvents,
) {
    for ins in segment {
        if let Instruction::RemoveVolatileStatus(remove) = ins {
            if remove.side_ref == side
                && remove.volatile_status == PokemonVolatileStatus::MUSTRECHARGE
            {
                let ident = ctx.active_ident(sim.state, side);
                out.lines.push(format!("|cant|{ident}|recharge"));
            }
        }
        sim.apply(ins);
    }
}

/// Restore amount for a deferred confusion snap-out. Mirrors
/// `-CONFUSION_SNAP_OUT_PENDING` in the gen3 engine patch, and is deliberately
/// NOT the ladder's `+1` "check ran" marker: at `+1` the two are
/// indistinguishable and the snap-out renders a fabricated `-activate` with no
/// `-end`, which the world layer then never clears.
const CONFUSION_SNAP_OUT_RESTORE: i8 = 4;

struct MovePrelude {
    used_move: bool,
    woke_up: bool,
    /// The ladder's deferred snap-out was consumed in this phase: emit `-end`
    /// and no `-activate`, and let the move through with no self-hit roll.
    confusion_snapped_out: bool,
    /// The exact 40-power damage the engine would emit for the pre-move
    /// confusion self-hit, if this phase actually reached that handler.
    ///
    /// This is deliberately recorded from the duration `+1` instruction,
    /// rather than inferred from PP or last-move bookkeeping. Both of those
    /// instructions are omitted by the engine in common legal states.
    confusion_self_hit_damage: Option<i16>,
}

/// Consume the pre-move bookkeeping (PP, last-used-move, sleep/freeze/rest
/// counters, charge-release volatile) and decide whether the mon acts.
fn consume_move_prelude(
    sim: &mut Sim<'_>,
    side: SideReference,
    choice: &Choice,
    segment: &[Instruction],
    cursor: &mut usize,
    ctx: &EventContext,
    out: &mut RenderedEvents,
) -> MovePrelude {
    let mut prelude = MovePrelude {
        used_move: true,
        woke_up: false,
        confusion_snapped_out: false,
        confusion_self_hit_damage: None,
    };
    // Attacker already fainted (first mover KO'd it before it could act) or
    // the opponent's pivot (U-turn/Baton Pass) saved this move for after the
    // replacement: the engine skips the phase and the real protocol shows
    // nothing.
    let attacker_dead = sim.active_hp(side).0 <= 0;
    let opponent_pivot_pending = match other_side(side) {
        SideReference::SideOne => sim.state.side_one.force_switch,
        SideReference::SideTwo => sim.state.side_two.force_switch,
    };
    if attacker_dead || opponent_pivot_pending {
        prelude.used_move = false;
        while *cursor < segment.len() {
            sim.apply(&segment[*cursor]);
            *cursor += 1;
        }
        return prelude;
    }
    let pre_status = {
        let s = match side {
            SideReference::SideOne => &sim.state.side_one,
            SideReference::SideTwo => &sim.state.side_two,
        };
        s.get_active_immutable().status
    };
    let flinched = {
        let s = match side {
            SideReference::SideOne => &sim.state.side_one,
            SideReference::SideTwo => &sim.state.side_two,
        };
        s.volatile_statuses.contains(&PokemonVolatileStatus::FLINCH)
    };
    let taunt_blocked = {
        let s = match side {
            SideReference::SideOne => &sim.state.side_one,
            SideReference::SideTwo => &sim.state.side_two,
        };
        s.volatile_statuses.contains(&PokemonVolatileStatus::TAUNT)
            && choice.category == MoveCategory::Status
    };
    let defender_dead = sim.active_hp(other_side(side)).0 <= 0;

    // Truant loaf: the whole phase is the volatile removal.
    if segment.len() == 1 {
        if let Instruction::RemoveVolatileStatus(remove) = &segment[0] {
            if remove.side_ref == side && remove.volatile_status == PokemonVolatileStatus::TRUANT {
                let ident = ctx.active_ident(sim.state, side);
                out.lines.push(format!("|cant|{ident}|ability: Truant"));
                sim.apply(&segment[0]);
                *cursor = 1;
                prelude.used_move = false;
                return prelude;
            }
        }
    }
    if defender_dead {
        // The engine skips the second mover entirely; the real protocol shows
        // nothing for it.
        prelude.used_move = false;
        while *cursor < segment.len() {
            sim.apply(&segment[*cursor]);
            *cursor += 1;
        }
        return prelude;
    }
    if flinched {
        let ident = ctx.active_ident(sim.state, side);
        out.lines.push(format!("|cant|{ident}|flinch"));
        prelude.used_move = false;
        while *cursor < segment.len() {
            sim.apply(&segment[*cursor]);
            *cursor += 1;
        }
        return prelude;
    }
    if taunt_blocked {
        let ident = ctx.active_ident(sim.state, side);
        out.lines.push(format!(
            "|cant|{ident}|move: Taunt|{}",
            move_display(choice.move_id)
        ));
        prelude.used_move = false;
        while *cursor < segment.len() {
            sim.apply(&segment[*cursor]);
            *cursor += 1;
        }
        return prelude;
    }

    // Bookkeeping instructions that precede the status gate.
    let mut sleep_gate_seen = false;
    while *cursor < segment.len() {
        let ins = &segment[*cursor];
        match ins {
            Instruction::DecrementPP(_)
            | Instruction::SetLastUsedMove(_)
            | Instruction::ChangeDamageDealtDamage(_)
            | Instruction::ChangeDamageDealtMoveCatagory(_)
            | Instruction::ToggleDamageDealtHitSubstitute(_) => {
                sim.apply(ins);
                *cursor += 1;
            }
            Instruction::RemoveVolatileStatus(remove)
                if remove.side_ref == side
                    && remove.volatile_status == PokemonVolatileStatus::DESTINYBOND =>
            {
                // Choosing any other move silently clears the previous
                // Destiny Bond before the status gate. It is real state
                // bookkeeping, but has no public protocol line.
                sim.apply(ins);
                *cursor += 1;
            }
            Instruction::DisableMove(disable)
                if disable.side_ref == side
                    && (sim
                        .state
                        .get_side_immutable(&side)
                        .get_active_immutable()
                        .item
                        == Items::CHOICEBAND
                        || matches!(
                            choice.volatile_status.as_ref(),
                            Some(volatile)
                                if volatile.volatile_status
                                    == PokemonVolatileStatus::LOCKEDMOVE
                                    && volatile.target == MoveTarget::User
                        )) =>
            {
                // Choice Band and locked-move setup disable the unused slots
                // before confusion runs. The disables are silent and may
                // legally appear even when confusion cancels the move.
                sim.apply(ins);
                *cursor += 1;
            }
            Instruction::SetFutureSight(set)
                if set.side_ref == side && choice.move_id == Choices::FUTURESIGHT =>
            {
                // The engine records Future Sight's pending-slot bookkeeping
                // in choice_before_move. The selected move still has to pass
                // the later confusion gate before a |move| line is emitted.
                sim.apply(ins);
                *cursor += 1;
            }
            // Confusion's onBeforeMove handler increments its bounded-duration
            // counter before it chooses the self-hit branch. This is silent
            // protocol bookkeeping, like PP and last-move updates above. If it
            // remains in the move segment, the following lone self-damage is
            // no longer recognized as a cancelled move and can be rendered as
            // the selected move's self-cost (notably Substitute).
            // The deferred snap-out, published where Showdown publishes it.
            // The ladder already decided this confusion ends; the engine parks
            // that as a negative duration and consumes it at the victim's next
            // move attempt, which is where `confusion.onBeforeMove` snaps out.
            // Showdown emits `-end` and NO `-activate` on this turn, and lets
            // the move through with no self-hit roll -- so this arm must not
            // set `confusion_self_hit_damage`. Matched on the restore amount,
            // which is deliberately not the ladder's `+1` marker.
            Instruction::ChangeVolatileStatusDuration(change)
                if change.side_ref == side
                    && change.volatile_status == PokemonVolatileStatus::CONFUSION
                    && change.amount == CONFUSION_SNAP_OUT_RESTORE =>
            {
                prelude.confusion_snapped_out = true;
                sim.apply(ins);
                *cursor += 1;
            }
            Instruction::RemoveVolatileStatus(remove)
                if remove.side_ref == side
                    && remove.volatile_status == PokemonVolatileStatus::CONFUSION
                    && prelude.confusion_snapped_out =>
            {
                let ident = ctx.active_ident(sim.state, side);
                out.lines.push(format!("|-end|{ident}|confusion"));
                sim.apply(ins);
                *cursor += 1;
            }
            Instruction::ChangeVolatileStatusDuration(change)
                if change.side_ref == side
                    && change.volatile_status == PokemonVolatileStatus::CONFUSION
                    && change.amount == 1 =>
            {
                prelude.confusion_self_hit_damage =
                    Some(confusion_self_hit_damage(sim.state, side));
                sim.apply(ins);
                *cursor += 1;
            }
            Instruction::RemoveVolatileStatus(remove)
                if remove.side_ref == side
                    && charge_volatile_move(remove.volatile_status).is_some() =>
            {
                // Charge release (Solar Beam turn 2 etc.): consumed silently;
                // the |move| line follows.
                sim.apply(ins);
                *cursor += 1;
            }
            Instruction::ChangeStatus(change)
                if change.side_ref == side
                    && change.old_status == PokemonStatus::SLEEP
                    && change.new_status == PokemonStatus::NONE =>
            {
                let transition = active_status_transition(sim.state, change);
                // Natural / Rest wake. Real protocol: |-curestatus| (ignored
                // by the fold — omitted).
                sim.apply(ins);
                *cursor += 1;
                prelude.woke_up = true;
                sleep_gate_seen = true;
                if let Some(mut transition) = transition {
                    transition.line_offset = out.lines.len();
                    out.active_status_transitions.push(transition);
                }
            }
            Instruction::ChangeStatus(change)
                if change.side_ref == side
                    && change.old_status == PokemonStatus::FREEZE
                    && change.new_status == PokemonStatus::NONE =>
            {
                let transition = active_status_transition(sim.state, change);
                // Thaw (real: |-curestatus|..|frz| — fold-ignored).
                sim.apply(ins);
                *cursor += 1;
                sleep_gate_seen = true;
                if let Some(mut transition) = transition {
                    transition.line_offset = out.lines.len();
                    out.active_status_transitions.push(transition);
                }
            }
            Instruction::SetSleepTurns(set) if set.side_ref == side => {
                sim.apply(ins);
                *cursor += 1;
                if set.new_turns > set.previous_turns && !prelude.woke_up {
                    let ident = ctx.active_ident(sim.state, side);
                    out.lines.push(format!("|cant|{ident}|slp"));
                    // Showdown emits the sleep gate even for Sleep Talk. The
                    // click is sleep-usable, so only ordinary moves stop
                    // here; Sleep Talk continues through the lower-priority
                    // confusion gate and then emits its own move line.
                    if choice.move_id != Choices::SLEEPTALK {
                        prelude.used_move = false;
                    }
                }
                sleep_gate_seen = true;
            }
            Instruction::DecrementRestTurns(_) => {
                sim.apply(ins);
                *cursor += 1;
                if !prelude.woke_up {
                    let ident = ctx.active_ident(sim.state, side);
                    out.lines.push(format!("|cant|{ident}|slp"));
                    // Rest sleep uses the same public sleep gate and the
                    // same Sleep Talk continuation rule as ordinary sleep.
                    if choice.move_id != Choices::SLEEPTALK {
                        prelude.used_move = false;
                    }
                }
                sleep_gate_seen = true;
            }
            _ => break,
        }
        if !prelude.used_move {
            // consume the rest silently (there should be nothing left).
            while *cursor < segment.len() {
                sim.apply(&segment[*cursor]);
                *cursor += 1;
            }
            return prelude;
        }
    }

    // Asleep with no wake/sleep-talk instructions at all: the engine's
    // "still asleep" branch when chance_to_wake == 0 emits SetSleepTurns, but
    // a rest sleep at 0 pp etc. may reach here with an empty tail.
    if pre_status == PokemonStatus::SLEEP && !sleep_gate_seen && segment[*cursor..].is_empty() {
        let ident = ctx.active_ident(sim.state, side);
        out.lines.push(format!("|cant|{ident}|slp"));
        prelude.used_move = false;
        return prelude;
    }
    if pre_status == PokemonStatus::FREEZE && !sleep_gate_seen && segment[*cursor..].is_empty() {
        let ident = ctx.active_ident(sim.state, side);
        out.lines.push(format!("|cant|{ident}|frz"));
        prelude.used_move = false;
        return prelude;
    }

    prelude
}

/// The Gen 3 confusion self-hit is a deterministic, typeless 40-power
/// physical hit against the user's own boosted Attack and Defense. This is a
/// direct mirror of the engine's `generate_instructions_from_existing_status_conditions`
/// calculation, including burn and the current-HP cap.
fn confusion_self_hit_damage(state: &State, side: SideReference) -> i16 {
    let active_side = state.get_side_immutable(&side);
    let active = active_side.get_active_immutable();
    let attack = active_side.calculate_boosted_stat(PokemonBoostableStat::Attack);
    let defense = active_side
        .calculate_boosted_stat(PokemonBoostableStat::Defense)
        .max(1);

    let mut damage = 2.0 * active.level as f32;
    damage = damage.floor() / 5.0;
    damage = damage.floor() + 2.0;
    damage = damage.floor() * 40.0;
    damage = damage * attack as f32 / defense as f32;
    damage = damage.floor() / 50.0;
    damage = damage.floor() + 2.0;
    if active.status == PokemonStatus::BURN {
        damage /= 2.0;
    }
    std::cmp::min(damage as i16, active.hp)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ConfusionSelfHitEvidence {
    Exact,
    AmbiguousExecutedSelfDamage,
}

/// Decide whether a one-damage action tail is provably a confusion self-hit.
///
/// A pre-move duration increment proves the confusion handler ran, but it does
/// not alone prove that it stopped the chosen move: a missed Jump Kick can
/// produce the same bare self-damage, and a self-destructing move can do so
/// behind Protect or immunity. We only render the public `|-activate|...|
/// confusion` line when the fixed 40-power damage identity is unique. When the
/// engine has collapsed an executed self-damage branch onto that identity, the
/// caller must preserve the endpoint behind a stable lossy marker instead.
fn classify_confusion_self_hit(
    state: &State,
    side: SideReference,
    choice: &Choice,
    action_tail: &[Instruction],
    expected_damage: Option<i16>,
) -> Option<ConfusionSelfHitEvidence> {
    let expected_damage = expected_damage?;
    let damage = match action_tail {
        [Instruction::Damage(damage)] if damage.side_ref == side => damage.damage_amount,
        _ => return None,
    };
    if damage != expected_damage {
        return None;
    }

    let active = state.get_side_immutable(&side).get_active_immutable();
    let crash_matches = choice.crash.map_or(false, |fraction| {
        let crash = (active.maxhp as f32 * fraction) as i16;
        damage == std::cmp::min(crash, active.hp)
    });
    if crash_matches || self_faint_move_can_be_self_only(state, side, choice, damage) {
        return Some(ConfusionSelfHitEvidence::AmbiguousExecutedSelfDamage);
    }
    Some(ConfusionSelfHitEvidence::Exact)
}

/// Explosion/Self-Destruct normally leave target damage in their action tail,
/// which makes a bare user damage impossible to confuse with their execution.
/// Protect, type/ability immunity are the exceptions: the engine records only
/// the user's faint, so a lethal confusion self-hit has the same delta.
/// Memento is deliberately excluded: success first lowers the target's stats
/// before fainting its user, while a protected/immune Memento does not faint;
/// it can never produce this one-instruction self-damage collision.
fn self_faint_move_can_be_self_only(
    state: &State,
    side: SideReference,
    choice: &Choice,
    damage: i16,
) -> bool {
    if !matches!(choice.move_id, Choices::EXPLOSION | Choices::SELFDESTRUCT) {
        return false;
    }
    let attacker_hp = state.get_side_immutable(&side).get_active_immutable().hp;
    if damage != attacker_hp {
        return false;
    }
    let defender = other_side(side);
    let defender_active = state.get_side_immutable(&defender).get_active_immutable();
    let protected = state
        .get_side_immutable(&defender)
        .volatile_statuses
        .contains(&PokemonVolatileStatus::PROTECT)
        && choice.flags.protect;
    let effectiveness = type_effectiveness_modifier(&choice.move_type, defender_active);
    let ability_blocks = ability_immunity(defender_active.ability, choice, effectiveness).is_some();
    protected || effectiveness == 0.0 || ability_blocks
}

/// A `(move, ...)` action phase. `called_tag` marks a caller-invoked move
/// (Sleep Talk): the prelude is skipped and the `|move|` line carries the
/// `[from]` caller attribution (fold: `called` token flag).
#[allow(clippy::too_many_arguments)]
fn render_move_phase(
    sim: &mut Sim<'_>,
    side: SideReference,
    choice: &Choice,
    segment: &[Instruction],
    branch_on_damage: bool,
    ctx: &EventContext,
    out: &mut RenderedEvents,
    called_tag: Option<&str>,
) {
    let defender = other_side(side);
    let mut cursor = 0usize;

    let locked_continuation = called_tag.is_none() && {
        let s = match side {
            SideReference::SideOne => &sim.state.side_one,
            SideReference::SideTwo => &sim.state.side_two,
        };
        s.volatile_statuses
            .contains(&PokemonVolatileStatus::LOCKEDMOVE)
    };

    let mut confusion_self_hit_damage = None;
    if called_tag.is_none() {
        let prelude = consume_move_prelude(sim, side, choice, segment, &mut cursor, ctx, out);
        if !prelude.used_move {
            return;
        }
        confusion_self_hit_damage = prelude.confusion_self_hit_damage;
    }

    let tail = &segment[cursor..];
    match classify_confusion_self_hit(sim.state, side, choice, tail, confusion_self_hit_damage) {
        Some(ConfusionSelfHitEvidence::Exact) => {
            let ident = ctx.active_ident(sim.state, side);
            out.lines.push(format!("|-activate|{ident}|confusion"));
            sim.apply(&tail[0]);
            let condition = sim.hp_condition(side);
            out.lines.push(format!("|-damage|{ident}|{condition}"));
            emit_faint_if_dead(sim, side, ctx, out);
            return;
        }
        Some(ConfusionSelfHitEvidence::AmbiguousExecutedSelfDamage) => {
            // The native engine combines chance outcomes with equal state
            // deltas. Do not invent either a confusion activation or a move
            // line when a crash/self-faint execution shares this exact delta.
            // In particular, never emit a causeless bare -damage: it would
            // be charged to the preceding move window by the fold. The exact
            // endpoint remains available to diagnostics, while production
            // callers fail closed before using this stream as evidence.
            out.mark_attribution_unsafe("confusion_selfhit_ambiguous_executed_self_damage");
            sim.apply(&tail[0]);
            return;
        }
        None => {}
    }

    // Showdown announces a surviving confusion check before the selected
    // move (including Sleep Talk) proceeds. The exact self-hit arm above
    // already emitted the same activation before returning.
    if confusion_self_hit_damage.is_some() {
        let ident = ctx.active_ident(sim.state, side);
        out.lines.push(format!("|-activate|{ident}|confusion"));
    }

    // Sleep Talk while asleep: the instruction list carries the CALLED
    // move's effects but not its identity — recover it by re-generating each
    // sleep-talk candidate and matching the segment tail exactly, then
    // render both |move| lines (fold: the called move opens the window with
    // `called=true`).
    if called_tag.is_none() && choice.move_id == Choices::SLEEPTALK {
        let still_asleep = {
            let s = match side {
                SideReference::SideOne => &sim.state.side_one,
                SideReference::SideTwo => &sim.state.side_two,
            };
            s.get_active_immutable().status == PokemonStatus::SLEEP
        };
        if still_asleep {
            let attacker_ident = ctx.active_ident(sim.state, side);
            out.lines
                .push(format!("|move|{attacker_ident}|sleeptalk|{attacker_ident}"));
            let called_tail: Vec<Instruction> = tail.to_vec();
            match identify_sleep_talk_called(sim.state, side, &called_tail, branch_on_damage) {
                Some(called_choice) => {
                    render_move_phase(
                        sim,
                        side,
                        &called_choice,
                        &called_tail,
                        branch_on_damage,
                        ctx,
                        out,
                        Some("Sleep Talk"),
                    );
                }
                None => {
                    // The called move is not provable from this delta, so its
                    // HP change gets no public action owner. It must still be
                    // DESCRIBED. Emitting nothing left the consumer's running
                    // HP baseline stale, and the next end-of-turn heal absorbed
                    // the whole drop -- surfacing as an impossible component
                    // such as a Leftovers tick of -134 on a mon that was at
                    // full HP (reports/c52_impossible_heal_component.json).
                    //
                    // This block's old comment asserted "callers reject before
                    // fold/encoder", but the differential deliberately does NOT
                    // reject a branch whose only lossy marker is this one --
                    // it treats the damage as real and generically attributed,
                    // and passes unattributed_damage_as_roll so that
                    // damage_component_events can retag it. The two sides
                    // disagreed about the contract; this satisfies the
                    // consumer's half by emitting the generic tag it already
                    // knows how to read
                    // (reports/c54_sleeptalk_render_contract_mismatch.json).
                    out.mark_attribution_unsafe("sleeptalk_called_unidentified");
                    // Walk the tail IN ORDER, rendering a drag at the moment
                    // it happens and re-baselining that side, then describing
                    // whatever HP movement follows. The previous version applied
                    // the whole tail first and read HP afterwards, so the drag
                    // line carried the mon's FINAL hp, any post-switch damage
                    // (Roar into Spikes is a common gen3 line) was swallowed,
                    // and emit_faint_if_dead could never fire for a switched
                    // side because before[] had already been set to its final
                    // value. This mirrors the proven-callee path below.
                    //
                    // The callee itself stays unattributed: no |move| line is
                    // emitted and the damage carries the generic [from] residual
                    // tag, which is what the differential retags as
                    // move_unknown_callee. A drag names no move, so rendering it
                    // invents nothing.
                    let mut before = [
                        sim.active_hp(SideReference::SideOne).0,
                        sim.active_hp(SideReference::SideTwo).0,
                    ];
                    // Once a side has been dragged, HP it loses is hazard chip
                    // on the way in, NOT a residual. Showdown emits
                    // `[from] Spikes` there, and the proven-callee path above
                    // already gets this right (the `switched && damage.side_ref
                    // == side` arm). Tagging it `residual` put the observation
                    // in the exact-component bucket as ("spikes", -n) while the
                    // engine line was retagged move_unknown_callee into the
                    // roll-scaled bucket, so the two never compared and the row
                    // could not match -- on "Roar into Spikes", which is the
                    // very line this walk was written to render.
                    // gen3-only inference: Spikes is the sole entry hazard that
                    // deals damage in this generation, and the crate is built
                    // with features = ["gen3"]. Under a gen4+ build this would
                    // need to distinguish Stealth Rock and Toxic Spikes.
                    let mut dragged = [false, false];
                    macro_rules! emit_residuals {
                        () => {
                            for (index, hp_side) in
                                [SideReference::SideOne, SideReference::SideTwo]
                                    .into_iter()
                                    .enumerate()
                            {
                                // DECREASES ONLY, deliberately. Rendering the heal
                                // direction was shipped once and emitted lines for
                                // the wrong Pokemon; it needs the same per-side
                                // re-baselining this walk now has, plus its own pin.
                                // Until then an unrendered rise leaves the row
                                // divergent, which is safe -- it cannot manufacture
                                // a false match -- but it does leave C52's impossible
                                // component alive in mirror image.
                                if sim.active_hp(hp_side).0 < before[index] {
                                    let ident = ctx.active_ident(sim.state, hp_side);
                                    let condition = sim.hp_condition(hp_side);
                                    let source = if dragged[index] {
                                        "Spikes"
                                    } else {
                                        "residual"
                                    };
                                    out.lines.push(format!(
                                        "|-damage|{ident}|{condition}|[from] {source}"
                                    ));
                                    emit_faint_if_dead(sim, hp_side, ctx, out);
                                    before[index] = sim.active_hp(hp_side).0;
                                }
                            }
                        };
                    }
                    for instruction in &called_tail {
                        if let Instruction::Switch(switch) = instruction {
                            emit_residuals!();
                            sim.apply(instruction);
                            let details =
                                ctx.details(sim.state, switch.side_ref, switch.next_index);
                            let ident = ctx.ident(switch.side_ref, switch.next_index);
                            let condition = sim.hp_condition(switch.side_ref);
                            out.lines
                                .push(format!("|drag|{ident}|{details}|{condition}"));
                            before[side_usize(switch.side_ref)] =
                                sim.active_hp(switch.side_ref).0;
                            dragged[side_usize(switch.side_ref)] = true;
                        } else {
                            sim.apply(instruction);
                        }
                    }
                    emit_residuals!();
                }
            }
            return;
        }
        // Awake Sleep Talk always fails, and (measured) the real sim keeps
        // the explicit self target on its move line — no [still] blanking.
        if segment[cursor..].is_empty() {
            let attacker_ident = ctx.active_ident(sim.state, side);
            out.lines
                .push(format!("|move|{attacker_ident}|sleeptalk|{attacker_ident}"));
            return;
        }
    }

    let attacker_ident = ctx.active_ident(sim.state, side);
    let defender_ident = ctx.active_ident(sim.state, defender);
    let move_name = move_display(choice.move_id);
    let is_damaging = choice.category != MoveCategory::Status;

    // Type effectiveness of the (mutated) choice against the live defender.
    let effectiveness = {
        let d = match defender {
            SideReference::SideOne => &sim.state.side_one,
            SideReference::SideTwo => &sim.state.side_two,
        };
        let active = d.get_active_immutable();
        type_effectiveness_modifier(&choice.move_type, active)
    };
    let defender_protected = {
        let d = match defender {
            SideReference::SideOne => &sim.state.side_one,
            SideReference::SideTwo => &sim.state.side_two,
        };
        d.volatile_statuses
            .contains(&PokemonVolatileStatus::PROTECT)
    };
    let (absorb, defender_ability) = {
        let d = match defender {
            SideReference::SideOne => &sim.state.side_one,
            SideReference::SideTwo => &sim.state.side_two,
        };
        let ability = d.get_active_immutable().ability;
        (is_absorb_ability(ability), ability)
    };
    let ability_immune = ability_immunity(defender_ability, choice, effectiveness);

    // Expected collapsed damage values for crit labeling.
    let (regular_collapsed, crit_collapsed) =
        expected_damage_values(sim.state, side, choice, branch_on_damage);

    // Classify the remaining tail.
    let deals_damage_to_defender = tail.iter().any(|ins| match ins {
        Instruction::Damage(d) => d.side_ref == defender,
        Instruction::DamageSubstitute(d) => d.side_ref == defender,
        _ => false,
    });
    let has_any_effect = !tail.is_empty();
    let is_self_faint_move = matches!(
        choice.move_id,
        Choices::EXPLOSION | Choices::SELFDESTRUCT | Choices::MEMENTO
    );

    // The |move| line. Target rendering (measured against the golden corpus):
    // opponent-target moves always show the target; self-target moves show
    // the user on success and a blank target + [still] on failure. Curse by
    // a non-Ghost is engine-targeted at the opponent (Ghost semantics) but
    // renders as a self-target move in the real protocol.
    let non_ghost_curse = choice.move_id == Choices::CURSE && {
        let s = match side {
            SideReference::SideOne => &sim.state.side_one,
            SideReference::SideTwo => &sim.state.side_two,
        };
        !s.get_active_immutable()
            .has_type(&poke_engine::state::PokemonType::GHOST)
    };
    let ghost_curse = choice.move_id == Choices::CURSE && !non_ghost_curse;
    let self_target = choice.target == MoveTarget::User || non_ghost_curse;
    // A status-inflicting move against an already-statused defender cannot
    // work; the engine merges its no-op "hit" branch with the miss branch,
    // and the fail outcome carries most of the probability mass — render the
    // real protocol's fail form (blank target + [still]), never [miss]
    // (documented ambiguity: a real 15%-miss renders identically here).
    // Type-based status immunity (Steel/Poison vs psn, Fire vs brn, Ice vs
    // frz): the real protocol shows |-immune| and it wins over the
    // already-statused fail (PS checks immunity first).
    let status_type_immune = choice.status.as_ref().map_or(false, |status| {
        use poke_engine::state::PokemonType;
        let d = match defender {
            SideReference::SideOne => &sim.state.side_one,
            SideReference::SideTwo => &sim.state.side_two,
        };
        let active = d.get_active_immutable();
        match status.status {
            PokemonStatus::BURN => active.has_type(&PokemonType::FIRE),
            PokemonStatus::FREEZE => active.has_type(&PokemonType::ICE),
            PokemonStatus::POISON | PokemonStatus::TOXIC => {
                active.has_type(&PokemonType::POISON) || active.has_type(&PokemonType::STEEL)
            }
            _ => false,
        }
    });
    let status_fail = choice.category == MoveCategory::Status
        && choice.status.is_some()
        && !has_any_effect
        && !status_type_immune
        && {
            let d = match defender {
                SideReference::SideOne => &sim.state.side_one,
                SideReference::SideTwo => &sim.state.side_two,
            };
            d.get_active_immutable().status != PokemonStatus::NONE
        };
    // A boost can be empty because every requested stat is capped OR an
    // opponent-side stat drop is blocked by Clear Body/White Smoke,
    // Hyper Cutter/Keen Eye, or Substitute. Both are indistinguishable from
    // Attract's empty immobilization branch. Only the self-target cap keeps
    // the renderer's established zero-amount boost-line behavior.
    let boost_has_no_effect = !has_any_effect
        && choice.category == MoveCategory::Status
        && choice
            .boost
            .as_ref()
            .map_or(false, |boost| !boost_would_apply(sim.state, side, boost));
    let capped_boost_move = self_target && boost_has_no_effect;
    // Most volatile moves have move-specific no-op paths (failed Protect,
    // no eligible Encore target, an already-present volatile). Substitute is
    // the one pure volatile whose public pre-state proves an executed move
    // would change state, so it remains a sound Attract immobilization cue.
    let volatile_empty_tail_ambiguous = !has_any_effect
        && choice.volatile_status.as_ref().map_or(false, |volatile| {
            if volatile.volatile_status != PokemonVolatileStatus::SUBSTITUTE {
                return true;
            }
            let target = match &volatile.target {
                MoveTarget::User => side,
                MoveTarget::Opponent => defender,
            };
            let target_side = sim.state.get_side_immutable(&target);
            !target_side
                .get_active_immutable()
                .volatile_status_can_be_applied(
                    &volatile.volatile_status,
                    &target_side.volatile_statuses,
                    choice.first_move,
                )
        });
    // A pure side-condition move whose condition is at CAP (spikes: 3
    // layers; screens/safeguard/mist: 1) fails with the real protocol's
    // blank-target form: `|move|..|Spikes||[still]` + `|-fail|user`
    // (corpus-measured).
    let side_condition_fail = !has_any_effect
        && choice.category == MoveCategory::Status
        && choice.status.is_none()
        && choice.side_condition.as_ref().map_or(false, |sc| {
            let target_side = match sc.target {
                MoveTarget::User => side,
                MoveTarget::Opponent => defender,
            };
            let cap = if sc.condition == PokemonSideCondition::Spikes {
                3
            } else {
                1
            };
            side_condition_value(sim.state, target_side, sc.condition) >= cap
        });
    // Empty tails need two independent predicates. The same engine delta can
    // represent an immobilizer OR a successful move that left no state change.
    // Full paralysis has a documented probability-based tie break; Attract
    // does not, so the latter must reject rather than invent either action.
    let deterministic_noop = (defender_protected && choice.flags.protect)
        || (is_damaging && effectiveness == 0.0)
        || (is_damaging && absorb.is_some())
        || ability_immune.is_some()
        || status_fail
        || status_type_immune
        || boost_has_no_effect
        || side_condition_fail;
    let (attacker_hp, attacker_maxhp) = sim.active_hp(side);
    let move_could_act = is_damaging
        || choice.status.is_some()
        || (choice.heal.is_some() && attacker_hp < attacker_maxhp)
        || choice.volatile_status.is_some()
        || choice.side_condition.is_some()
        || choice.boost.is_some();
    let empty_tail_can_be_accuracy_miss = choice.target == MoveTarget::Opponent
        && !status_fail
        && !non_ghost_curse
        && ability_immune.is_none()
        && effectiveness > 0.0
        && choice.accuracy < 100.0;

    // Full paralysis: the engine merges the 25% fully-paralyzed branch with
    // any same-delta branch (notably the miss branch). When the empty delta
    // is not deterministically explained and the move WOULD have acted, the
    // paralysis outcome carries the larger probability mass — render
    // |cant|..|par| (documented ambiguity: a real miss renders identically).
    let attacker_paralyzed = {
        let s = match side {
            SideReference::SideOne => &sim.state.side_one,
            SideReference::SideTwo => &sim.state.side_two,
        };
        s.get_active_immutable().status == PokemonStatus::PARALYZE
    };
    let attacker_attracted = {
        let s = match side {
            SideReference::SideOne => &sim.state.side_one,
            SideReference::SideTwo => &sim.state.side_two,
        };
        s.volatile_statuses
            .contains(&PokemonVolatileStatus::ATTRACT)
            && s.get_active_immutable().ability != Abilities::OBLIVIOUS
    };

    // Attract's immobilized branch is also an empty tail, including after the
    // higher-priority confusion handler has already incremented its duration.
    // An empty tail is only evidence of immobilization when the selected move
    // could otherwise change state and no deterministic no-op/miss explains
    // the same endpoint. Protect, immunity, misses, capped boosts/statuses,
    // capped side conditions, and intrinsically no-effect moves therefore
    // fail closed: rendering either |cant| or |move| would invent attribution.
    if attacker_attracted && !has_any_effect && called_tag.is_none() {
        if deterministic_noop
            || volatile_empty_tail_ambiguous
            || empty_tail_can_be_accuracy_miss
            // Attract resolves before full paralysis, but the engine merges
            // their identical empty endpoints. The aggregate branch cannot
            // prove which immobilizer stopped this action.
            || attacker_paralyzed
            || !move_could_act
        {
            // Name WHICH ambiguity refused, not just that one did. The five
            // predicates are function-local and were discarded at the refusal, so
            // no artifact recorded the split and no script could recover it --
            // which left the only available plan "patch the engine and hope".
            //
            // The split decides the fix, and the two answers are far apart. If
            // `paralyzed` dominates, this is downgradeable to lossy in a few
            // lines: both outcomes are "no move used, no reveal, no PP", Attract
            // dominates 4:1 (50% vs 12.5%), and that is a WIDER margin than the
            // par-over-miss guess this renderer already ships. If the noop/miss
            // arms dominate it is not downgradeable at any price -- those erase a
            // `|move|` reveal, and the miss arm also suppresses a PP decrement the
            // fold tracks -- and only then is an engine marker instruction worth
            // its patch-stack and digest cost.
            //
            // Emit EVERY live predicate, not the first match. They are not
            // mutually exclusive, and a first-match bucket answers the wrong
            // question in the expensive direction: `attacker_paralyzed` is a
            // property of the ATTACKER while `miss`/`noop` are properties of the
            // MOVE, so they co-occur freely, and testing paralysis first hides
            // the non-downgradeable arms inside the one bucket that looks safe
            // to downgrade.
            //
            // Measured on the fork masses (`ATTRACT_IMMOBILIZE_CHANCE` 1/2, then
            // the 0.25 paralysis roll on the surviving half):
            //   clean paralyzed-only      attract .500 / par .125          -> 80/20
            //   paralyzed + Thunder       + miss .1125                     -> 15.3% miss
            //   paralyzed + immune target + noop .375                      -> 37.5% noop
            // The contamination is unrecoverable once collapsed, so a
            // `paralyzed`-dominant read would say "ship the lossy downgrade"
            // while a third of that mass is the case that erases a `|move|`
            // reveal. Emitting the joint set keeps the probe able to answer its
            // own question, and the realized cardinality is small (~8).
            //
            // Order within the slug is FIXED, not predicate-evaluation order, so
            // the key is stable across runs and aggregators can sum it.
            let mut parts: Vec<&str> = Vec::new();
            if attacker_paralyzed {
                parts.push("paralyzed");
            }
            if empty_tail_can_be_accuracy_miss {
                parts.push("miss");
            }
            if deterministic_noop {
                parts.push("noop");
            }
            if volatile_empty_tail_ambiguous {
                parts.push("volatile");
            }
            if !move_could_act {
                parts.push("cannot_act");
            }
            // Unreachable: the enclosing `if` fired, so at least one predicate is
            // live. Named rather than silently empty so a future edit that breaks
            // that correspondence is visible in the measurement.
            if parts.is_empty() {
                parts.push("unclassified");
            }
            out.mark_attribution_unsafe(&format!(
                "attract_empty_tail_ambiguous:{}",
                parts.join("+")
            ));
            return;
        }
        // The action is uniquely immobilized, but the engine does not retain
        // Attract's source/gender attribution. That omission is telemetry-only
        // because the public action window itself is exact.
        out.mark_lossy("attract_immobilization_source_unknown");
        out.lines.push(format!("|cant|{attacker_ident}|Attract"));
        return;
    }

    if attacker_paralyzed && !has_any_effect && called_tag.is_none() {
        if !deterministic_noop && move_could_act {
            out.lines.push(format!("|cant|{attacker_ident}|par"));
            return;
        }
    }

    // Caller-invoked moves (Sleep Talk) render their explicit target even on
    // failure (measured; the [still] blanking does not apply to them).
    //
    // Corpus-measured fail forms (leaf_vs_reality PP follow-up, 2026-07-19):
    // an ALREADY-STATUSED status fail keeps the explicit target
    // (`|move|p1a: Piloswine|Toxic|p2a: Deoxys` + `|-fail|p2a: Deoxys|tox`) —
    // the pre-fix blank+[still] form dropped the target, breaking BOTH the
    // fold's defender_species and the parser's foe-targeted Pressure PP
    // double-charge; a side-condition-at-cap fail uses the blank form
    // (`|move|p2a: Deoxys|Spikes||[still]` + `|-fail|p2a: Deoxys`).
    let mut move_line = if self_target {
        if has_any_effect || capped_boost_move || called_tag.is_some() {
            format!("|move|{attacker_ident}|{move_name}|{attacker_ident}")
        } else {
            format!("|move|{attacker_ident}|{move_name}||[still]")
        }
    } else if side_condition_fail && called_tag.is_none() {
        format!("|move|{attacker_ident}|{move_name}||[still]")
    } else {
        format!("|move|{attacker_ident}|{move_name}|{defender_ident}")
    };
    if locked_continuation {
        move_line.push_str("|[from]lockedmove");
    }
    if let Some(tag) = called_tag {
        move_line.push_str(&format!("|[from] {tag}"));
    }

    // Miss inference: an opponent-target move with accuracy < 100 whose tail
    // shows no effect on the defender, with deterministic causes (immunity,
    // protect, absorb) ruled out. NOTE: for a paralyzed/frozen attacker the
    // engine merges the full-para branch with the miss branch — that case
    // never reaches here (the prelude renders |cant| first), so the residual
    // ambiguity is para-vs-miss only, documented in the module docs.
    let mut missed = false;
    if choice.target == MoveTarget::Opponent
        && !status_fail
        && !non_ghost_curse
        && ability_immune.is_none()
        && !deals_damage_to_defender
        && effectiveness > 0.0
        && choice.accuracy < 100.0
    {
        let defender_affected = tail.iter().any(|ins| {
            instruction_side(ins) == Some(defender)
                || matches!(ins, Instruction::ChangeStatus(c) if c.side_ref == defender)
        });
        let crash_only = tail
            .iter()
            .all(|ins| matches!(ins, Instruction::Damage(d) if d.side_ref == side));
        if !defender_affected && (tail.is_empty() || (choice.crash.is_some() && crash_only)) {
            missed = true;
        }
    }
    if missed {
        move_line.push_str("|[miss]");
    }
    out.lines.push(move_line);

    // Ghost-typed Curse (live-game protocol probe, 2026-07-19): the real
    // protocol starts the curse on the TARGET and charges the user a bare
    // self |-damage| HP cut:
    //     |move|p1a: Gengar|Curse|p2a: Blissey
    //     |-start|p2a: Blissey|Curse|[of] p1a: Gengar
    //     |-damage|p1a: Gengar|114/227
    // The gen3 engine has NO Ghost split — it applies the base Curse
    // choice's delta (user stat boosts, user volatile, no HP cut, no
    // residual). Render the real-protocol |-start| marker (fold-ignored),
    // suppress the spurious |-boost| lines below (reality never shows them),
    // and flag the branch lossy: the true self-cost is not derivable from
    // the engine delta and is knowingly missing (engine-model deviation,
    // never silently mis-attributed).
    if ghost_curse && has_any_effect {
        out.lines.push(format!(
            "|-start|{defender_ident}|Curse|[of] {attacker_ident}"
        ));
        out.mark_attribution_unsafe("ghost_curse_engine_model");
    }

    // Real fail lines (fold-ignored; kept for line-stream fidelity with the
    // measured protocol shapes above).
    if called_tag.is_none() {
        if status_fail {
            let code = {
                let d = match defender {
                    SideReference::SideOne => &sim.state.side_one,
                    SideReference::SideTwo => &sim.state.side_two,
                };
                status_code(d.get_active_immutable().status)
            };
            match code {
                Some(code) => out.lines.push(format!("|-fail|{defender_ident}|{code}")),
                None => out.lines.push(format!("|-fail|{defender_ident}")),
            }
        } else if side_condition_fail {
            out.lines.push(format!("|-fail|{attacker_ident}"));
        }
    }

    if missed {
        out.lines
            .push(format!("|-miss|{attacker_ident}|{defender_ident}"));
        // Crash damage (High Jump Kick class).
        for ins in tail {
            if let Instruction::Damage(damage) = ins {
                if damage.side_ref == side {
                    sim.apply(ins);
                    let condition = sim.hp_condition(side);
                    let ident = ctx.active_ident(sim.state, side);
                    out.lines
                        .push(format!("|-damage|{ident}|{condition}|[from] {move_name}"));
                    emit_faint_if_dead(sim, side, ctx, out);
                    continue;
                }
            }
            sim.apply(ins);
        }
        return;
    }

    // Explosion/Self-Destruct faint the user before the hit check. Their
    // resulting tail is therefore nonempty even when the target was protected
    // or immune; render that target outcome before walking the user's faint.
    let mut self_faint_target_outcome_rendered = false;
    if is_self_faint_move && has_any_effect && !deals_damage_to_defender {
        if defender_protected && choice.flags.protect {
            out.lines
                .push(format!("|-activate|{defender_ident}|Protect"));
            self_faint_target_outcome_rendered = true;
        } else if is_damaging && effectiveness == 0.0 {
            out.lines.push(format!("|-immune|{defender_ident}"));
            self_faint_target_outcome_rendered = true;
        } else if let Some(ability) = ability_immune {
            out.lines.push(format!(
                "|-immune|{defender_ident}|[from] ability: {ability}"
            ));
            self_faint_target_outcome_rendered = true;
        }
    }

    // Deterministic no-effect renders.
    if !has_any_effect {
        if capped_boost_move {
            // All requested stats at cap: the real protocol still shows the
            // 0-amount boost lines (fold: Boost side effect).
            if let Some(boost) = &choice.boost {
                for (stat, amount) in [
                    (PokemonBoostableStat::Attack, boost.boosts.attack),
                    (PokemonBoostableStat::Defense, boost.boosts.defense),
                    (
                        PokemonBoostableStat::SpecialAttack,
                        boost.boosts.special_attack,
                    ),
                    (
                        PokemonBoostableStat::SpecialDefense,
                        boost.boosts.special_defense,
                    ),
                    (PokemonBoostableStat::Speed, boost.boosts.speed),
                    (PokemonBoostableStat::Accuracy, boost.boosts.accuracy),
                ] {
                    if amount != 0 {
                        let head = if amount > 0 { "-boost" } else { "-unboost" };
                        let code = boost_stat_code(stat);
                        out.lines.push(format!("|{head}|{attacker_ident}|{code}|0"));
                    }
                }
            }
            return;
        }
        if defender_protected && choice.flags.protect {
            out.lines
                .push(format!("|-activate|{defender_ident}|Protect"));
            return;
        }
        if is_damaging && effectiveness == 0.0 {
            out.lines.push(format!("|-immune|{defender_ident}"));
            return;
        }
        if choice.status.is_some()
            && (status_type_immune
                || (effectiveness == 0.0 && choice.target == MoveTarget::Opponent))
        {
            out.lines.push(format!("|-immune|{defender_ident}"));
            return;
        }
        if let Some(ability) = ability_immune {
            out.lines.push(format!(
                "|-immune|{defender_ident}|[from] ability: {ability}"
            ));
            return;
        }
        if is_damaging {
            if let Some(ability) = absorb {
                out.lines.push(format!(
                    "|-immune|{defender_ident}|[from] ability: {ability}"
                ));
                return;
            }
        }
        // A failed status move (already statused, boost at cap, no last move
        // to encore...): real protocol = blank-target [still] (already
        // rendered for self-target); the fold ignores |-fail|.
        return;
    }

    // Effectiveness annotations precede the damage lines (PS ordering).
    // Fixed-damage moves (Seismic Toss class: base_power 0) never show
    // effectiveness in the real protocol, only immunity.
    if is_damaging && deals_damage_to_defender && choice.base_power > 0.0 {
        if effectiveness > 1.0 {
            out.lines.push(format!("|-supereffective|{defender_ident}"));
        } else if effectiveness > 0.0 && effectiveness < 1.0 {
            out.lines.push(format!("|-resisted|{defender_ident}"));
        }
    }

    // Walk the effect tail. Faints are DEFERRED to the end of the phase
    // (real protocol: recoil/drain lines come before the |faint| lines).
    let mut defender_hits: i64 = 0;
    let mut crit_emitted = false;
    let mut damage_lines_done = false;
    let mut roughskin_emitted = false;
    let mut pending_faints: Vec<SideReference> = Vec::new();
    macro_rules! note_faint {
        ($side:expr) => {
            if sim.active_hp($side).0 <= 0 && !pending_faints.contains(&$side) {
                pending_faints.push($side);
            }
        };
    }
    let is_transform = choice.move_id == Choices::TRANSFORM;
    if is_transform {
        out.lines
            .push(format!("|-transform|{attacker_ident}|{defender_ident}"));
    }
    for ins in tail {
        match ins {
            // Pain Split is not damage: Showdown expresses BOTH halves as
            // `|-sethp|..|[from] move: Pain Split` (target first and
            // `[silent]`, then the user). It is the only move in the pool that
            // emits the tag. Rendering it bare made the engine path disagree
            // with the sim on a deterministic quantity, and — because
            // `fold.rs` keys its Pain Split self-cost branch on exactly
            // `-sethp` + that `[from]` payload — meant `self_hp_cost` was
            // never charged on the engine-as-environment path at all.
            Instruction::Damage(damage)
                if damage.side_ref == defender && choice.move_id == Choices::PAINSPLIT =>
            {
                sim.apply(ins);
                let condition = sim.hp_condition(defender);
                out.lines.push(format!(
                    "|-sethp|{defender_ident}|{condition}|[from] move: Pain Split|[silent]"
                ));
                defender_hits += 1;
                note_faint!(defender);
            }
            Instruction::Damage(damage) if damage.side_ref == defender => {
                let (pre_hp, _max) = sim.active_hp(defender);
                sim.apply(ins);
                // Crit labeling: exact-value match against the engine's own
                // collapsed crit damage, never on KO-capped values.
                if !crit_emitted
                    && !damage_lines_done
                    && crit_collapsed.is_some()
                    && Some(damage.damage_amount) == crit_collapsed
                    && crit_collapsed != regular_collapsed
                    && damage.damage_amount < pre_hp
                {
                    out.lines.push(format!("|-crit|{defender_ident}"));
                    crit_emitted = true;
                    // -crit precedes -damage in the real protocol; reorder by
                    // inserting before the damage line we are about to push.
                }
                let condition = sim.hp_condition(defender);
                out.lines
                    .push(format!("|-damage|{defender_ident}|{condition}"));
                defender_hits += 1;
                note_faint!(defender);
            }
            Instruction::DamageSubstitute(_) => {
                sim.apply(ins);
                out.lines
                    .push(format!("|-activate|{defender_ident}|Substitute|[damage]"));
                defender_hits += 1;
            }
            Instruction::RemoveVolatileStatus(remove)
                if remove.side_ref == defender
                    && remove.volatile_status == PokemonVolatileStatus::SUBSTITUTE =>
            {
                sim.apply(ins);
                out.lines.push(format!("|-end|{defender_ident}|Substitute"));
            }
            Instruction::Damage(damage) if damage.side_ref == side => {
                // Attacker-side damage attribution ladder. A bare render is
                // read by the fold as SELF-COST, so opponent-inflicted
                // damage (Rough Skin, Destiny Bond) must carry its [from]
                // tag; anything unexplained is rendered bare but flagged
                // lossy — never silently mis-attributed.
                let (pre_hp, pre_maxhp) = sim.active_hp(side);
                let roughskin_expected = std::cmp::min(pre_maxhp / 16, pre_hp);
                let is_roughskin = !roughskin_emitted
                    && defender_ability == Abilities::ROUGHSKIN
                    && choice.flags.contact
                    && deals_damage_to_defender
                    && damage.damage_amount == roughskin_expected;
                let is_destiny_bond = {
                    let d = match defender {
                        SideReference::SideOne => &sim.state.side_one,
                        SideReference::SideTwo => &sim.state.side_two,
                    };
                    d.volatile_statuses
                        .contains(&PokemonVolatileStatus::DESTINYBOND)
                        && damage.damage_amount == pre_hp
                        && deals_damage_to_defender
                };
                if is_self_faint_move {
                    // Explosion-class: the real protocol shows only |faint|.
                    sim.apply(ins);
                    note_faint!(side);
                } else if is_roughskin {
                    // Engine order: the contact-punish damage lands BEFORE
                    // any recoil damage (ability_after_damage_hit precedes
                    // the recoil push in generate_instructions_from_damage).
                    roughskin_emitted = true;
                    sim.apply(ins);
                    let condition = sim.hp_condition(side);
                    out.lines.push(format!(
                        "|-damage|{attacker_ident}|{condition}|[from] ability: Rough Skin|[of] {defender_ident}"
                    ));
                    note_faint!(side);
                } else if is_destiny_bond {
                    sim.apply(ins);
                    let condition = sim.hp_condition(side);
                    out.lines.push(format!(
                        "|-damage|{attacker_ident}|{condition}|[from] move: Destiny Bond"
                    ));
                    note_faint!(side);
                } else if choice.recoil.is_some() {
                    sim.apply(ins);
                    let condition = sim.hp_condition(side);
                    out.lines.push(format!(
                        "|-damage|{attacker_ident}|{condition}|[from] Recoil|[of] {defender_ident}"
                    ));
                    note_faint!(side);
                } else if choice.move_id == Choices::PAINSPLIT {
                    // The user's half, rendered as the sim does: `-sethp` with
                    // the move tag and NO `[silent]`. This is the line
                    // `fold.rs` charges `self_hp_cost` from.
                    sim.apply(ins);
                    let condition = sim.hp_condition(side);
                    out.lines.push(format!(
                        "|-sethp|{attacker_ident}|{condition}|[from] move: Pain Split"
                    ));
                    note_faint!(side);
                } else if matches!(
                    choice.move_id,
                    Choices::SUBSTITUTE | Choices::BELLYDRUM | Choices::CURSE
                ) {
                    // Genuine self-costs (the fold SHOULD count these, and
                    // the real protocol renders them bare).
                    sim.apply(ins);
                    let condition = sim.hp_condition(side);
                    out.lines
                        .push(format!("|-damage|{attacker_ident}|{condition}"));
                    note_faint!(side);
                } else {
                    // Unexplained attacker-side damage has no observable
                    // owner. Do not emit a bare line that the fold would
                    // charge as self-cost; fail closed before encoding.
                    out.mark_attribution_unsafe("unattributed_self_damage");
                    sim.apply(ins);
                }
            }
            Instruction::Heal(heal) => {
                if heal.heal_amount == 0 && heal.side_ref == defender {
                    if self_faint_target_outcome_rendered {
                        continue;
                    } else if defender_protected {
                        out.lines
                            .push(format!("|-activate|{defender_ident}|Protect"));
                    } else if let Some(ability) = absorb {
                        out.lines.push(format!(
                            "|-immune|{defender_ident}|[from] ability: {ability}"
                        ));
                    } else {
                        out.mark_attribution_unsafe("unattributed_noop_heal_marker");
                    }
                    continue;
                }
                if choice.move_id == Choices::MEMENTO
                    && heal.side_ref == side
                    && heal.heal_amount < 0
                {
                    // Memento is represented as a negative self-Heal so the
                    // reversible engine can restore the user. It is not
                    // Liquid Ooze: Showdown first shows the target stat drops
                    // and then silently faints the user. Deferring the faint
                    // preserves that order and lets the fold charge the
                    // action's remaining HP through its Memento rule.
                    sim.apply(ins);
                    note_faint!(side);
                    continue;
                }
                sim.apply(ins);
                let target_ident = ctx.active_ident(sim.state, heal.side_ref);
                let condition = sim.hp_condition(heal.side_ref);
                if heal.heal_amount < 0 {
                    // Liquid Ooze reverses drain into damage. The engine uses
                    // a reversible negative Heal instruction, but Showdown's
                    // public event is damage and can be lethal.
                    out.lines.push(format!(
                        "|-damage|{target_ident}|{condition}|[from] ability: Liquid Ooze|[of] {defender_ident}"
                    ));
                    note_faint!(heal.side_ref);
                } else if heal.side_ref == side {
                    if choice.drain.is_some() && deals_damage_to_defender {
                        out.lines.push(format!(
                            "|-heal|{target_ident}|{condition}|[from] drain|[of] {defender_ident}"
                        ));
                    } else if choice.move_id == Choices::REST {
                        // Rest's heal is [silent] in the real protocol — the
                        // fold must NOT read it as a Heal side effect.
                        out.lines
                            .push(format!("|-heal|{target_ident}|{condition} slp|[silent]"));
                    } else {
                        out.lines.push(format!("|-heal|{target_ident}|{condition}"));
                    }
                } else {
                    // Heal on the DEFENDER inside our move phase: absorb
                    // ability soak (Volt/Water Absorb).
                    if let Some(ability) = absorb {
                        out.lines.push(format!(
                            "|-heal|{target_ident}|{condition}|[from] ability: {ability}|[of] {attacker_ident}"
                        ));
                    } else {
                        out.lines.push(format!("|-heal|{target_ident}|{condition}"));
                    }
                }
            }
            Instruction::ChangeStatus(change) => {
                let transition = active_status_transition(sim.state, change);
                sim.apply(ins);
                let target_ident = ctx.active_ident(sim.state, change.side_ref);
                if change.old_status == PokemonStatus::NONE {
                    if let Some(code) = status_code(change.new_status) {
                        if change.side_ref == side && choice.move_id == Choices::REST {
                            out.lines
                                .push(format!("|-status|{target_ident}|slp|[from] move: Rest"));
                        } else {
                            out.lines.push(format!("|-status|{target_ident}|{code}"));
                        }
                    }
                }
                // Status cures (Heal Bell / Refresh / lum): |-curestatus| is
                // fold-ignored — omitted.
                if let Some(mut transition) = transition {
                    transition.line_offset = out.lines.len();
                    out.active_status_transitions.push(transition);
                }
            }
            Instruction::Boost(boost) => {
                sim.apply(ins);
                if ghost_curse {
                    // Engine-model artifact: the gen3 engine applies the
                    // non-Ghost stats-up delta for a Ghost curser; the real
                    // protocol shows no boost lines (see the ghost_curse
                    // block above — branch already flagged lossy).
                    continue;
                }
                out.lines.push(render_boost_line(
                    ctx,
                    sim,
                    boost.side_ref,
                    boost.stat,
                    boost.amount,
                    None,
                ));
            }
            Instruction::ChangeSideCondition(change) => {
                sim.apply(ins);
                render_side_condition_change(change, sim, ctx, out, Some(&move_name));
            }
            Instruction::ChangeWeather(change) => {
                sim.apply(ins);
                match weather_display(change.new_weather) {
                    Some(name) => out.lines.push(format!("|-weather|{name}")),
                    None => out.lines.push("|-weather|none".to_string()),
                }
            }
            Instruction::ApplyVolatileStatus(apply) => {
                sim.apply(ins);
                let target_ident = ctx.active_ident(sim.state, apply.side_ref);
                if let Some(charge_move) = charge_volatile_move(apply.volatile_status) {
                    // Charge turn: the |move| line was already emitted; the
                    // fold reads |-prepare| (Charging side effect +
                    // pending_charge).
                    out.lines.push(format!(
                        "|-prepare|{target_ident}|{}",
                        move_display(charge_move)
                    ));
                }
                if apply.volatile_status == PokemonVolatileStatus::FLASHFIRE {
                    // Flash Fire FIRST activation: the real protocol's
                    // boost-state form (live capture: `|-start|p2a: Houndoom|
                    // ability: Flash Fire`) — an ABSORB SIGNATURE the fold
                    // consumes (`_is_absorb_start`), so omitting it would
                    // leave a bare |move| line and lose the Absorbed outcome.
                    // (Repeat activations are an empty delta and render
                    // `|-immune|..|[from] ability: Flash Fire` upstream.)
                    out.lines
                        .push(format!("|-start|{target_ident}|ability: Flash Fire"));
                }
                // Substitute/Protect/Leech Seed/Encore/confusion starts render
                // as |-start|/|-singleturn| in the real protocol — all
                // fold-ignored, deliberately omitted (module docs).
            }
            Instruction::ChangeType(_)
            | Instruction::ChangeAbility(_)
            | Instruction::FormeChange(_) => {
                // Transform internals / trace: single |-transform| line
                // already rendered; the rest is silent.
                sim.apply(ins);
            }
            Instruction::Switch(switch) => {
                // Drag (Whirlwind/Roar): the forced switch renders as |drag|.
                sim.apply(ins);
                let details = ctx.details(sim.state, switch.side_ref, switch.next_index);
                let ident = ctx.ident(switch.side_ref, switch.next_index);
                let condition = sim.hp_condition(switch.side_ref);
                out.lines
                    .push(format!("|drag|{ident}|{details}|{condition}"));
            }
            _ => sim.apply(ins),
        }
        if let Instruction::Damage(_) | Instruction::DamageSubstitute(_) = ins {
            damage_lines_done = defender_hits > 0;
        }
    }

    // Multi-hit count (fold: n_hits).
    if defender_hits >= 1 && !matches!(choice.multi_hit(), poke_engine::choices::MultiHitMove::None)
    {
        out.lines
            .push(format!("|-hitcount|{defender_ident}|{defender_hits}"));
    }
    // Deferred faints, in the order the KOs landed.
    for fainted in pending_faints {
        emit_faint_if_dead(sim, fainted, ctx, out);
    }
}

/// Identify which move Sleep Talk called by re-generating each sleep-talk
/// candidate's instructions from the current (prelude-applied) state and
/// matching the branch tail exactly. Returns the MUTATED candidate choice
/// (the engine's own modification pass applied) or None when zero or
/// multiple candidates match (ambiguous delta — documented insufficiency).
fn identify_sleep_talk_called(
    state: &mut State,
    side: SideReference,
    tail: &[Instruction],
    branch_on_damage: bool,
) -> Option<Choice> {
    let candidates = {
        let s = match side {
            SideReference::SideOne => &state.side_one,
            SideReference::SideTwo => &state.side_two,
        };
        s.get_active_immutable().get_sleep_talk_choices()
    };
    let mut matched: Option<Choice> = None;
    for candidate in candidates {
        let mut choice = candidate.clone();
        choice.sleep_talk_move = true;
        let mut generated: Vec<StateInstructions> = Vec::with_capacity(4);
        generate_instructions_from_move(
            state,
            &mut choice,
            &Choice::default(),
            side,
            StateInstructions::default(),
            &mut generated,
            branch_on_damage,
        );
        if generated
            .iter()
            .any(|branch| branch.instruction_list.as_slice() == tail)
        {
            if matched.is_some() {
                return None; // ambiguous
            }
            matched = Some(choice);
        }
    }
    matched
}

/// Expected collapsed damage values (regular, crit) for the attacking side's
/// move, used ONLY to label `|-crit|` branches. Mirrors the engine's own
/// collapsing (0.925 * max roll).
fn expected_damage_values(
    state: &State,
    side: SideReference,
    choice: &Choice,
    _branch_on_damage: bool,
) -> (Option<i16>, Option<i16>) {
    if choice.category == MoveCategory::Status {
        return (None, None);
    }
    let s1_first = match side {
        SideReference::SideOne => true,
        SideReference::SideTwo => false,
    };
    let (rolls_s1, rolls_s2) = calculate_both_damage_rolls(
        state,
        if s1_first {
            choice.clone()
        } else {
            Choice::default()
        },
        if s1_first {
            Choice::default()
        } else {
            choice.clone()
        },
        s1_first,
    );
    let rolls = match side {
        SideReference::SideOne => rolls_s1,
        SideReference::SideTwo => rolls_s2,
    };
    match rolls {
        Some(values) if values.len() >= 2 => (
            Some((values[0] as f32 * 0.925) as i16),
            Some((values[1] as f32 * 0.925) as i16),
        ),
        Some(values) if values.len() == 1 => (Some((values[0] as f32 * 0.925) as i16), None),
        _ => (None, None),
    }
}

fn render_boost_line(
    ctx: &EventContext,
    sim: &Sim<'_>,
    side: SideReference,
    stat: PokemonBoostableStat,
    amount: i8,
    from: Option<&str>,
) -> String {
    let ident = ctx.active_ident(sim.state, side);
    let stat_code = boost_stat_code(stat);
    let magnitude = amount.unsigned_abs();
    let head = if amount >= 0 { "-boost" } else { "-unboost" };
    match from {
        Some(tag) => format!("|{head}|{ident}|{stat_code}|{magnitude}|[from] {tag}"),
        None => format!("|{head}|{ident}|{stat_code}|{magnitude}"),
    }
}

fn render_side_condition_change(
    change: &poke_engine::instruction::ChangeSideConditionInstruction,
    sim: &Sim<'_>,
    _ctx: &EventContext,
    out: &mut RenderedEvents,
    from_move: Option<&str>,
) {
    let Some(display) = side_condition_display(change.side_condition) else {
        return; // Protect counter / ToxicCount: engine-internal, no line.
    };
    let side_ident = side_prefix(change.side_ref);
    if change.amount > 0 {
        out.lines
            .push(format!("|-sidestart|{side_ident}: side|{display}"));
    } else {
        // Removal renders |-sideend| only when the counter reached zero
        // (screens expiring / Rapid Spin); mid-count decrements (screen
        // timers at end of turn) are silent.
        let remaining = side_condition_value(sim.state, change.side_ref, change.side_condition);
        if remaining <= 0 {
            match from_move {
                Some(name) if name == "rapidspin" => out.lines.push(format!(
                    "|-sideend|{side_ident}: side|{display}|[from] move: Rapid Spin"
                )),
                _ => out
                    .lines
                    .push(format!("|-sideend|{side_ident}: side|{display}")),
            }
        }
    }
}

fn side_condition_value(state: &State, side: SideReference, condition: PokemonSideCondition) -> i8 {
    let s = match side {
        SideReference::SideOne => &state.side_one,
        SideReference::SideTwo => &state.side_two,
    };
    match condition {
        PokemonSideCondition::Spikes => s.side_conditions.spikes,
        PokemonSideCondition::Reflect => s.side_conditions.reflect,
        PokemonSideCondition::LightScreen => s.side_conditions.light_screen,
        PokemonSideCondition::Safeguard => s.side_conditions.safeguard,
        PokemonSideCondition::Mist => s.side_conditions.mist,
        _ => 0,
    }
}

/// Whether an attempted primary boost can mutate the exact current engine
/// state. The engine suppresses opponent-side stat drops behind Substitute
/// and the Gen 3 stat-drop immunities before it emits any instruction.
fn boost_would_apply(state: &State, attacker: SideReference, boost: &Boost) -> bool {
    let target = match &boost.target {
        MoveTarget::User => attacker,
        MoveTarget::Opponent => other_side(attacker),
    };
    let target_side = state.get_side_immutable(&target);
    let target_pokemon = target_side.get_active_immutable();
    if target_pokemon.hp <= 0 {
        return false;
    }
    boost
        .boosts
        .get_as_pokemon_boostable()
        .iter()
        .any(|(stat, amount)| {
            if *amount == 0 {
                return false;
            }
            if *amount < 0
                && target != attacker
                && target_pokemon
                    .immune_to_stats_lowered_by_opponent(stat, &target_side.volatile_statuses)
            {
                return false;
            }
            let current = target_side.get_boost_from_boost_enum(stat);
            (*amount > 0 && current < 6) || (*amount < 0 && current > -6)
        })
}

fn emit_faint_if_dead(
    sim: &Sim<'_>,
    side: SideReference,
    ctx: &EventContext,
    out: &mut RenderedEvents,
) {
    if sim.active_hp(side).0 <= 0 {
        let ident = ctx.active_ident(sim.state, side);
        let line = format!("|faint|{ident}");
        if out.lines.last() != Some(&line) {
            out.lines.push(line);
        }
    }
}

// ---------------------------------------------------------------------------
// End-of-turn (residual) rendering
// ---------------------------------------------------------------------------

/// Render one end-of-turn instruction. Windows are already closed (a blank
/// `|` line precedes the residual segment, as in the real protocol), so the
/// fold consumes only HP fractions, faints, weather transitions and
/// side-condition expiry from this segment; `[from]` tags are attached on a
/// best-effort basis for stream realism and to stay inert if a caller ever
/// feeds residuals into an open window.
fn render_residual_instruction(
    sim: &mut Sim<'_>,
    ins: &Instruction,
    next_ins: Option<&Instruction>,
    plan: &mut ResidualPlan,
    ctx: &EventContext,
    out: &mut RenderedEvents,
) {
    match ins {
        Instruction::Damage(damage) => {
            let side = damage.side_ref;
            // Positional first; the state guess is only a fallback for sides
            // whose plan did not reconcile (see ResidualPlan).
            let cause = plan
                .take(side, false)
                .unwrap_or_else(|| residual_damage_cause(sim.state, side, damage.damage_amount));
            sim.apply(ins);
            let ident = ctx.active_ident(sim.state, side);
            let condition = sim.hp_condition(side);
            out.lines
                .push(format!("|-damage|{ident}|{condition}|[from] {cause}"));
            emit_faint_if_dead(sim, side, ctx, out);
        }
        Instruction::Heal(heal) => {
            let side = heal.side_ref;
            sim.apply(ins);
            let ident = ctx.active_ident(sim.state, side);
            let condition = sim.hp_condition(side);
            if heal.heal_amount < 0 {
                let source = ctx.active_ident(sim.state, other_side(side));
                out.lines.push(format!(
                    "|-damage|{ident}|{condition}|[from] ability: Liquid Ooze|[of] {source}"
                ));
                emit_faint_if_dead(sim, side, ctx, out);
            } else {
                let cause = plan
                    .take(side, true)
                    .unwrap_or_else(|| residual_heal_cause(sim.state, side, next_ins));
                out.lines.push(if cause.is_empty() {
                    // Showdown renders the Leech Seed sap on the SEEDER as a
                    // bare silent heal — the `[from] Leech Seed` tag goes on
                    // the victim's damage line, not the drainer's heal
                    // (verified against a live trace:
                    // `|-heal|p1a: Bellossom|259/293|[silent]`).
                    format!("|-heal|{ident}|{condition}|[silent]")
                } else {
                    format!("|-heal|{ident}|{condition}|[from] {cause}")
                });
            }
        }
        Instruction::ChangeWeather(change) => {
            sim.apply(ins);
            match weather_display(change.new_weather) {
                None => out.lines.push("|-weather|none".to_string()),
                Some(name) => out.lines.push(format!("|-weather|{name}")),
            }
        }
        Instruction::DecrementWeatherTurnsRemaining => {
            sim.apply(ins);
            if let Some(name) = weather_display(sim.state.weather.weather_type) {
                out.lines.push(format!("|-weather|{name}|[upkeep]"));
            }
        }
        Instruction::ChangeSideCondition(change) => {
            sim.apply(ins);
            render_side_condition_change(change, sim, ctx, out, None);
        }
        Instruction::ChangeStatus(change) => {
            let transition = active_status_transition(sim.state, change);
            // Yawn falling asleep at end of turn.
            sim.apply(ins);
            if change.old_status == PokemonStatus::NONE {
                if let Some(code) = status_code(change.new_status) {
                    let ident = ctx.active_ident(sim.state, change.side_ref);
                    out.lines.push(format!("|-status|{ident}|{code}"));
                }
            }
            if let Some(mut transition) = transition {
                transition.line_offset = out.lines.len();
                out.active_status_transitions.push(transition);
            }
        }
        Instruction::Boost(boost) => {
            // End-of-turn boosts are item/ability sourced (Salac/Petaya,
            // Speed Boost): the [from] tag keeps the fold from reading them
            // as move side effects if a window were open.
            sim.apply(ins);
            let item_name = active_item_display(sim.state, boost.side_ref);
            out.lines.push(render_boost_line(
                ctx,
                sim,
                boost.side_ref,
                boost.stat,
                boost.amount,
                Some(&item_name),
            ));
        }
        Instruction::RemoveVolatileStatus(remove)
            if remove.volatile_status == PokemonVolatileStatus::CONFUSION =>
        {
            // The engine resolves its bounded confusion ladder at end of turn
            // for implementation convenience, but Showdown keeps the inert
            // volatile visible through the next decision boundary and snaps
            // out only when that mon next attempts to move. Emitting -end here
            // would leak the engine's future branch into public state early.
            // The engine now defers the snap-out itself, so this arm is no
            // longer reachable from engine-generated instructions -- the ladder
            // parks the counter instead of removing the volatile here. It is
            // KEPT as a fail-closed backstop: if the deferral ever regresses, or
            // a hand-built instruction list removes CONFUSION at end of turn,
            // refusing the world is still the right answer. Do not delete it in
            // a dead-code sweep.
            out.mark_attribution_unsafe("confusion_expiry_timing_unobservable");
            sim.apply(ins);
        }
        _ => {
            sim.apply(ins);
        }
    }
}

/// Positional attribution of end-of-turn residuals.
///
/// `residual_damage_cause` / `residual_heal_cause` guessed a source by testing
/// the side's STATE in a fixed priority order and returning the first match for
/// EVERY residual instruction on that side. A mon with two simultaneous sources
/// therefore had all of its ticks labelled with the highest-priority one: a
/// poisoned mon in sand had its sand tick rendered `[from] psn`, a trapped mon
/// in sand had its trap tick rendered `[from] Sandstorm`, and a Leftovers holder
/// whose opponent was seeded had its heal rendered `[from] Leech Seed`. The HP
/// arithmetic was always right; only the labels were wrong (ledger H.1).
///
/// The fix is POSITIONAL, never amount-based. Amounts cannot disambiguate: the
/// sand chip and the partial-trap tick are BOTH `maxhp/16`, which is a permanent
/// counterexample, not an edge case (pinned by `sand_and_trap_collide_on_amount`).
///
/// The engine emits residuals in Showdown's own speed-major order, from
/// `gen3/generate_instructions.rs::add_end_of_turn_instructions`. PER SIDE that
/// order is:
///
///   damage: weather chip (8) -> Leech Seed (10.5) -> status (10.6)
///           -> partial trap (10.9) -> the OPPONENT's Future Sight (11)
///   heal:   Wish (7) -> Leftovers (10.4) -> the seeder's Leech Seed drain
///
/// Each phase emits at most one HP instruction per side, so the k-th damage on a
/// side is the k-th firing damage phase for that side. The plan below predicts
/// which phases fire using PRESENCE predicates only — never damage formulas —
/// and is used ONLY when its predicted counts match the counts actually emitted.
/// On any mismatch the side falls back to the generic `residual` tag, which is
/// loud (it diverges) rather than confidently wrong.
///
/// TWO of those entries are cross-side, which is why this plan has to know the
/// engine's speed order and is not simply a per-side constant:
///
/// * The Leech Seed sap DAMAGES the seeded side and HEALS the seeder, both at
///   the seeded side's 10.5 slot. So the seeder's drain heal lands BEFORE its own
///   Leftovers heal when the victim is faster, and AFTER it when the seeder is.
/// * Future Sight is owned by one side and damages the other, at order 11 —
///   after every order-10 handler on BOTH sides, so it is always last in the
///   damaged side's sequence. (The pre-speed-major plan put this label on the
///   OWNER's list, which is the side that takes no damage from it.)
///
/// Both are handled by reading the segment rather than by predicting a speed
/// order, so the plan stays correct on the exact tie the engine forks on.
#[derive(Default)]
pub(crate) struct ResidualPlan {
    damage: [Vec<String>; 2],
    heal: [Vec<String>; 2],
    usable: [bool; 2],
    damage_seen: [usize; 2],
    heal_seen: [usize; 2],
}

fn side_index(side: SideReference) -> usize {
    match side {
        SideReference::SideOne => 0,
        SideReference::SideTwo => 1,
    }
}

fn weather_chips(state: &State, side: SideReference) -> Option<&'static str> {
    let s = match side {
        SideReference::SideOne => &state.side_one,
        SideReference::SideTwo => &state.side_two,
    };
    let active = s.get_active_immutable();
    if active.hp <= 0 {
        return None;
    }
    if state.weather_is_active(&Weather::HAIL) {
        if active.has_type(&PokemonType::ICE) {
            return None;
        }
        return Some("Hail");
    }
    if state.weather_is_active(&Weather::SAND) {
        if active.has_type(&PokemonType::ROCK)
            || active.has_type(&PokemonType::GROUND)
            || active.has_type(&PokemonType::STEEL)
        {
            return None;
        }
        return Some("Sandstorm");
    }
    None
}

impl ResidualPlan {
    /// Build from the PRE-residual state, in the engine's own emission order.
    pub(crate) fn build(state: &State, segment: &[Instruction]) -> ResidualPlan {
        let mut plan = ResidualPlan::default();
        let mut drains_opponent = [false; 2];
        for side in [SideReference::SideOne, SideReference::SideTwo] {
            let i = side_index(side);
            let (s, opponent) = match side {
                SideReference::SideOne => (&state.side_one, &state.side_two),
                SideReference::SideTwo => (&state.side_two, &state.side_one),
            };
            let active = s.get_active_immutable();

            // --- damage phases, in order ---
            if let Some(label) = weather_chips(state, side) {
                plan.damage[i].push(label.to_string());
            }
            if s.volatile_statuses
                .contains(&PokemonVolatileStatus::LEECHSEED)
                && opponent.get_active_immutable().hp > 0
            {
                plan.damage[i].push("Leech Seed".to_string());
            }
            match active.status {
                PokemonStatus::BURN => plan.damage[i].push("brn".to_string()),
                PokemonStatus::POISON | PokemonStatus::TOXIC => {
                    plan.damage[i].push("psn".to_string())
                }
                _ => {}
            }
            if s.volatile_statuses
                .contains(&PokemonVolatileStatus::PARTIALLYTRAPPED)
            {
                plan.damage[i].push("partiallytrapped".to_string());
            }
            // Future Sight is order 11 — after every order-10 handler on BOTH
            // sides, so it is always last — and it is the OPPONENT's, because the
            // engine emits `Damage { side_ref: owner.get_other_side() }`. The
            // pre-speed-major plan listed it on the owner's side, which takes no
            // damage from it at all.
            if opponent.future_sight.0 == 1 {
                plan.damage[i].push("move: Future Sight".to_string());
            }

            // --- heal phases, in order ---
            if s.wish.0 == 1 {
                plan.heal[i].push("move: Wish".to_string());
            }
            if active.item == Items::LEFTOVERS {
                plan.heal[i].push("item: Leftovers".to_string());
            }
            drains_opponent[i] = opponent
                .volatile_statuses
                .contains(&PokemonVolatileStatus::LEECHSEED)
                && active.hp > 0
                && opponent.get_active_immutable().hp > 0;
        }

        // The seeder's silent drain heal is emitted at the SEEDED side's 10.5
        // slot, not at the seeder's own, so where it lands among the seeder's
        // heals depends on which side resolved first — before its Leftovers when
        // the victim is faster, after it when the seeder is. That is not a
        // per-side constant, and on an exact speed tie the engine forks and BOTH
        // orders are live, so it cannot be predicted from speed either.
        //
        // Read it off the segment instead: walk once, and the drain is the
        // seeder's next heal after the sap damage on the victim's side. Still
        // positional, still never amount-based, and correct on a tie without
        // having to know which fork this segment came from.
        let leech_at: [Option<usize>; 2] = [
            plan.damage[0].iter().position(|tag| tag == "Leech Seed"),
            plan.damage[1].iter().position(|tag| tag == "Leech Seed"),
        ];
        let mut drain_at: [Option<usize>; 2] = [None; 2];
        {
            let mut damage_seen = [0usize; 2];
            let mut heal_seen = [0usize; 2];
            for ins in segment {
                match ins {
                    Instruction::Damage(d) => {
                        let victim = side_index(d.side_ref);
                        if leech_at[victim] == Some(damage_seen[victim]) {
                            drain_at[1 - victim] = Some(heal_seen[1 - victim]);
                        }
                        damage_seen[victim] += 1;
                    }
                    Instruction::Heal(h) if h.heal_amount > 0 => {
                        heal_seen[side_index(h.side_ref)] += 1;
                    }
                    _ => {}
                }
            }
        }
        for i in 0..2 {
            if !drains_opponent[i] {
                continue;
            }
            // Silent: Showdown tags the victim's DAMAGE with Leech Seed and
            // emits the seeder's heal bare.
            let at = drain_at[i]
                .unwrap_or(plan.heal[i].len())
                .min(plan.heal[i].len());
            plan.heal[i].insert(at, String::new());
        }

        // Only trust the plan for a side when it predicts EXACTLY the number of
        // HP instructions that segment actually emits for that side. Predicting
        // a phase that did not fire (or missing one that did) would shift every
        // later label on that side, so a count mismatch disables the plan there.
        let mut emitted_damage = [0usize; 2];
        let mut emitted_heal = [0usize; 2];
        for ins in segment {
            match ins {
                Instruction::Damage(d) => emitted_damage[side_index(d.side_ref)] += 1,
                Instruction::Heal(h) if h.heal_amount > 0 => {
                    emitted_heal[side_index(h.side_ref)] += 1
                }
                _ => {}
            }
        }
        for i in 0..2 {
            plan.usable[i] =
                plan.damage[i].len() == emitted_damage[i] && plan.heal[i].len() == emitted_heal[i];
        }
        plan
    }

    fn take(&mut self, side: SideReference, is_heal: bool) -> Option<String> {
        let i = side_index(side);
        if !self.usable[i] {
            return None;
        }
        let (list, seen) = if is_heal {
            (&self.heal[i], &mut self.heal_seen[i])
        } else {
            (&self.damage[i], &mut self.damage_seen[i])
        };
        let label = list.get(*seen).cloned();
        *seen += 1;
        label
    }
}

/// Best-effort residual damage attribution from the pre-application state.
fn residual_damage_cause(state: &State, side: SideReference, amount: i16) -> String {
    let s = match side {
        SideReference::SideOne => &state.side_one,
        SideReference::SideTwo => &state.side_two,
    };
    let active = s.get_active_immutable();
    match active.status {
        PokemonStatus::BURN => return "brn".to_string(),
        PokemonStatus::POISON | PokemonStatus::TOXIC => return "psn".to_string(),
        _ => {}
    }
    if s.volatile_statuses
        .contains(&PokemonVolatileStatus::LEECHSEED)
    {
        return "Leech Seed".to_string();
    }
    match state.weather.weather_type {
        Weather::SAND => return "Sandstorm".to_string(),
        Weather::HAIL => return "Hail".to_string(),
        _ => {}
    }
    if s.volatile_statuses
        .contains(&PokemonVolatileStatus::PARTIALLYTRAPPED)
    {
        return "partiallytrapped".to_string();
    }
    let _ = amount;
    "residual".to_string()
}

/// Attribute a residual heal.
///
/// The Wish test is ADJACENCY, not `wish.0 > 0`. `wish.0` is the pending-turn
/// counter, so a side merely *carrying* a wish mislabels every ordinary
/// Leftovers tick as `[from] move: Wish` — the engine emits `DecrementWish`
/// ahead of the Leftovers heal on a non-resolving turn, leaving the counter
/// positive when the heal is rendered. Verified against the instruction stream:
///
///   resolving      : `Heal`, `DecrementWish`, [`Heal` (Leftovers)]
///   pending only   : `DecrementWish`, `Heal` (Leftovers)
///   no wish        : `Heal` (Leftovers)
///
/// so the wish heal is exactly the one IMMEDIATELY FOLLOWED by `DecrementWish`
/// for the same side. `next_ins` is the lookahead the caller supplies.
///
/// WHY THIS SURVIVES #876's RESIDUAL DEFERRAL. The deferral relocates the WHOLE
/// end-of-turn residual block past a forced replacement — it does not reorder,
/// split, or interleave the block's contents. `Heal`/`DecrementWish` are emitted
/// as a unit at the wish's residual slot, so wherever the block is emitted the
/// pair stays adjacent. The invariant is therefore structural (a property of how
/// the block is constructed), not an observation about where the block happens
/// to land, and it holds equally on the deferred ply.
fn residual_heal_cause(
    state: &State,
    side: SideReference,
    next_ins: Option<&Instruction>,
) -> String {
    let s = match side {
        SideReference::SideOne => &state.side_one,
        SideReference::SideTwo => &state.side_two,
    };
    let wish_resolving = matches!(
        next_ins,
        Some(Instruction::DecrementWish(d)) if d.side_ref == side
    );
    if wish_resolving {
        return "move: Wish".to_string();
    }
    let opponent = match side {
        SideReference::SideOne => &state.side_two,
        SideReference::SideTwo => &state.side_one,
    };
    if opponent
        .volatile_statuses
        .contains(&PokemonVolatileStatus::LEECHSEED)
    {
        return "Leech Seed".to_string();
    }
    if s.get_active_immutable().item == Items::LEFTOVERS {
        return "item: Leftovers".to_string();
    }
    "item: Leftovers".to_string()
}

fn active_item_display(state: &State, side: SideReference) -> String {
    let s = match side {
        SideReference::SideOne => &state.side_one,
        SideReference::SideTwo => &state.side_two,
    };
    format!("item: {:?}", s.get_active_immutable().item)
}

fn ability_display_of_active(state: &State, side: SideReference) -> String {
    let s = match side {
        SideReference::SideOne => &state.side_one,
        SideReference::SideTwo => &state.side_two,
    };
    let raw = format!("{:?}", s.get_active_immutable().ability);
    // "SANDSTREAM" -> "Sand Stream" is not recoverable without a table; the
    // fold only normalizes ([a-z0-9]), so the enum name is fold-equivalent.
    raw
}

// ---------------------------------------------------------------------------
// Python surface
// ---------------------------------------------------------------------------

pub(crate) fn move_choice_from_str(
    name: &str,
    state: &State,
    side: SideReference,
) -> PyResult<MoveChoice> {
    let side_ref = match side {
        SideReference::SideOne => &state.side_one,
        SideReference::SideTwo => &state.side_two,
    };
    MoveChoice::from_string(name, side_ref)
        .ok_or_else(|| PyValueError::new_err(format!("invalid move for {:?}: {name}", side)))
}

fn post_state_summary(state: &State) -> serde_json::Value {
    let mut sides = serde_json::Map::new();
    for (key, side, force_switch) in [
        ("p1", &state.side_one, state.side_one.force_switch),
        ("p2", &state.side_two, state.side_two.force_switch),
    ] {
        let active = side.get_active_immutable();
        let mut mons = Vec::new();
        let mut iter = side.pokemon.into_iter();
        while let Some(p) = iter.next() {
            if format!("{:?}", p.id) == "NONE" {
                continue;
            }
            mons.push(serde_json::json!({
                "hp": p.hp,
                "maxhp": p.maxhp,
                "status": format!("{:?}", p.status).to_lowercase(),
            }));
        }
        sides.insert(
            key.to_string(),
            serde_json::json!({
                "active_index": side.active_index.serialize().parse::<i64>().unwrap_or(-1),
                "active_hp": active.hp,
                "active_maxhp": active.maxhp,
                "active_status": format!("{:?}", active.status).to_lowercase(),
                "force_switch": force_switch,
                "boosts": {
                    "atk": side.attack_boost,
                    "def": side.defense_boost,
                    "spa": side.special_attack_boost,
                    "spd": side.special_defense_boost,
                    "spe": side.speed_boost,
                    "accuracy": side.accuracy_boost,
                    "evasion": side.evasion_boost,
                },
                "pokemon": mons,
            }),
        );
    }
    serde_json::Value::Object(sides)
}

/// Serialize the exact branch prefix that prices a direct hit after a
/// same-turn switch or stat-stage change.
///
/// The transition differential cannot use a branch's completed post-state for
/// this: post-state may include the hit itself plus reaction effects such as
/// Knock Off's item removal.  Re-segment the engine instruction list, locate
/// the first damage to each acting phase's defender, and snapshot only the
/// instructions before that hit.  A snapshot is emitted only when that prefix
/// contains a switch or boost; ordinary branches keep using their pre-boundary
/// damage support.
fn legal_roll_state_before_direct_damage(
    state: &mut State,
    s1_move: &MoveChoice,
    s2_move: &MoveChoice,
    full: &[Instruction],
    branch_on_damage: bool,
) -> Option<String> {
    let segmentation = segment(state, s1_move, s2_move, full, branch_on_damage)?;
    let phases = [
        (0, segmentation.p1_end, segmentation.first),
        (
            segmentation.p1_end,
            segmentation.p2_end,
            other_side(segmentation.first),
        ),
    ];

    for (start, end, attacker) in phases {
        let defender = other_side(attacker);
        let Some(direct_damage_index) =
            full[start..end]
                .iter()
                .position(|instruction| match instruction {
                    Instruction::Damage(damage) | Instruction::DamageSubstitute(damage) => {
                        damage.side_ref == defender
                    }
                    _ => false,
                })
        else {
            continue;
        };
        let prefix = full[..start + direct_damage_index].to_vec();
        let requires_reprice = prefix.iter().any(|instruction| {
            matches!(instruction, Instruction::Switch(_) | Instruction::Boost(_))
        });
        if !requires_reprice {
            continue;
        }

        state.apply_instructions(&prefix);
        let serialized = state.serialize();
        state.reverse_instructions(&prefix);
        return Some(serialized);
    }
    None
}

/// Enumerate the engine's chance outcomes for a joint action and render each
/// as protocol lines (the instruction→event mapping), returning JSON:
/// `{"end_of_turn": bool, "branches": [{"percentage", "events", "turn_completed",
///   "lossy", "attribution_unsafe", "attribution_unsafe_reasons", "post",
///   "post_state", "legal_roll_state"}]}`.
///
/// `ctx_json`: `{"p1": [display species...], "p2": [...], "turn": N}` with
/// species in ENGINE PARTY ORDER (see `EngineWorld.party_species`).
#[pyfunction]
#[pyo3(signature = (state_str, s1_move, s2_move, ctx_json, branch_on_damage = true, include_post_state = false))]
pub fn branch_events(
    state_str: &str,
    s1_move: &str,
    s2_move: &str,
    ctx_json: &str,
    branch_on_damage: bool,
    include_post_state: bool,
) -> PyResult<String> {
    let mut state = parse_state(state_str)?;
    let ctx = EventContext::from_json(ctx_json).map_err(PyValueError::new_err)?;
    let s1 = move_choice_from_str(s1_move, &state, SideReference::SideOne)?;
    let s2 = move_choice_from_str(s2_move, &state, SideReference::SideTwo)?;

    let generated = generate_instructions_from_move_pair(&mut state, &s1, &s2, branch_on_damage);
    let mut branches = Vec::new();
    if generated.is_empty() {
        branches.push(serde_json::json!({
            "percentage": 100.0,
            "events": ["|"],
            "turn_completed": false,
            "lossy": ["empty_instruction_list"],
            "attribution_unsafe": false,
            "attribution_unsafe_reasons": [],
            "post": post_state_summary(&state),
        }));
    }
    for branch in &generated {
        let legal_roll_state = if include_post_state {
            legal_roll_state_before_direct_damage(
                &mut state,
                &s1,
                &s2,
                &branch.instruction_list,
                branch_on_damage,
            )
        } else {
            None
        };
        let rendered = render_branch_events(
            &mut state,
            &s1,
            &s2,
            &branch.instruction_list,
            branch_on_damage,
            &ctx,
        );
        state.apply_instructions(&branch.instruction_list);
        let post = post_state_summary(&state);
        let post_state = if include_post_state {
            Some(state.serialize())
        } else {
            None
        };
        state.reverse_instructions(&branch.instruction_list);
        let attribution_unsafe = rendered.is_attribution_unsafe();
        let mut obj = serde_json::json!({
            "percentage": branch.percentage,
            "events": rendered.lines,
            "turn_completed": rendered.turn_completed,
            "lossy": rendered.lossy,
            "attribution_unsafe": attribution_unsafe,
            "attribution_unsafe_reasons": rendered.attribution_unsafe,
            "post": post,
        });
        if let Some(post_state) = post_state {
            obj["post_state"] = serde_json::Value::String(post_state);
        }
        if let Some(legal_roll_state) = legal_roll_state {
            obj["legal_roll_state"] = serde_json::Value::String(legal_roll_state);
        }
        branches.push(obj);
    }
    let report = serde_json::json!({
        "end_of_turn": end_of_turn_triggered(&state, &s1, &s2),
        "branches": branches,
    });
    serde_json::to_string(&report)
        .map_err(|e| PyValueError::new_err(format!("serialize report: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    const MINIMAL: &str = include_str!("test_fixtures/minimal.state");

    fn ctx() -> EventContext {
        EventContext {
            species: [vec!["Charmander".to_string()], vec!["Squirtle".to_string()]],
            turn: 4,
            hp_percent: [false, false],
        }
    }

    #[test]
    fn force_switch_keeps_end_of_turn_open() {
        let mut state = parse_state(MINIMAL.trim()).expect("fixture parses");
        state.side_one.force_switch = true;
        assert!(end_of_turn_triggered(
            &state,
            &MoveChoice::Switch(PokemonIndex::P1),
            &MoveChoice::None,
        ));
    }

    /// A recharge on an ORDINARY turn must still be consumed and rendered.
    ///
    /// The companion to the forced-replacement case. The fix suppresses recharge
    /// consumption while a replacement boundary is open, and the obvious way to
    /// get that wrong is to suppress it always -- which would make MUSTRECHARGE
    /// permanent and hand the recharger a free pass every turn forever. This
    /// pins the other direction.
    #[test]
    fn an_ordinary_recharge_turn_still_consumes_and_renders() {
        let fixture = MINIMAL
            .trim()
            .replacen("EMBER;false;32", "HYPERBEAM;false;8", 1);
        let mut state = parse_state(&fixture).expect("fixture parses");
        state
            .side_one
            .volatile_statuses
            .insert(PokemonVolatileStatus::MUSTRECHARGE);
        assert!(!state.side_one.force_switch, "no replacement boundary");
        assert!(!state.side_two.force_switch, "no replacement boundary");

        let ctx = EventContext {
            species: [vec!["Charmander".to_string()], vec!["Squirtle".to_string()]],
            turn: 7,
            hp_percent: [false, false],
        };
        let tackle =
            MoveChoice::from_string("tackle", &state.side_two).expect("Tackle is available");
        let branches =
            generate_instructions_from_move_pair(&mut state, &MoveChoice::None, &tackle, true);
        assert!(!branches.is_empty());
        for branch in &branches {
            assert!(
                branch.instruction_list.iter().any(|i| matches!(
                    i,
                    Instruction::RemoveVolatileStatus(r)
                        if r.volatile_status == PokemonVolatileStatus::MUSTRECHARGE
                )),
                "an ordinary recharge turn must CONSUME the recharge: {:?}",
                branch.instruction_list
            );
            let rendered = render_branch_events(
                &mut state,
                &MoveChoice::None,
                &tackle,
                &branch.instruction_list,
                true,
                &ctx,
            );
            let text = rendered.lines.join("\n");
            assert_eq!(
                text.matches("|cant|p1a: Charmander|recharge").count(),
                1,
                "exactly one recharge line on an ordinary turn: {text}"
            );
        }
    }

    #[test]
    fn forced_replacement_defers_hyper_beam_recharge_cant_until_next_turn() {
        let fixture = MINIMAL
            .trim()
            .replacen("EMBER;false;32", "HYPERBEAM;false;8", 1);
        let mut state = parse_state(&fixture).expect("fixture parses");
        state.side_one.get_active().speed = 500;
        state.side_two.get_active().speed = 1;
        state.side_two.pokemon[PokemonIndex::P1] = state.side_two.pokemon[PokemonIndex::P0].clone();
        state.side_two.get_active().hp = 1;
        let ctx = EventContext {
            species: [
                vec!["Charmander".to_string()],
                vec!["Squirtle".to_string(), "Squirtle".to_string()],
            ],
            turn: 4,
            hp_percent: [false, false],
        };

        // A guaranteed Hyper Beam KO opens the opponent's forced-replacement
        // boundary while the attacker still owes its recharge turn.
        let hyper_beam =
            MoveChoice::from_string("hyperbeam", &state.side_one).expect("Hyper Beam is available");
        let ko =
            generate_instructions_from_move_pair(&mut state, &hyper_beam, &MoveChoice::None, true)
                .into_iter()
                .find(|branch| {
                    branch.instruction_list.iter().any(|instruction| {
                        matches!(instruction, Instruction::ToggleSideTwoForceSwitch)
                    })
                })
                .expect("Hyper Beam KO creates a forced replacement");
        state.apply_instructions(&ko.instruction_list);
        assert!(state.side_two.force_switch);
        assert!(state
            .side_one
            .volatile_statuses
            .contains(&PokemonVolatileStatus::MUSTRECHARGE));

        let forced_replacement = generate_instructions_from_move_pair(
            &mut state,
            &MoveChoice::None,
            &MoveChoice::Switch(PokemonIndex::P1),
            true,
        );
        assert_eq!(forced_replacement.len(), 1);
        assert!(end_of_turn_triggered(
            &state,
            &MoveChoice::None,
            &MoveChoice::Switch(PokemonIndex::P1),
        ));
        let replacement = &forced_replacement[0].instruction_list;
        let rendered_replacement = render_branch_events(
            &mut state,
            &MoveChoice::None,
            &MoveChoice::Switch(PokemonIndex::P1),
            replacement,
            true,
            &ctx,
        );
        let replacement_text = rendered_replacement.lines.join("\n");
        assert!(
            replacement_text.contains("|switch|p2a: Squirtle|Squirtle|100/100"),
            "{replacement_text}"
        );
        assert!(replacement_text.contains("|upkeep"), "{replacement_text}");
        assert!(
            !replacement_text.contains("|cant|p1a: Charmander|recharge"),
            "the replacement is not the recharger turn: {replacement_text}"
        );
        state.apply_instructions(replacement);
        assert!(!state.side_two.force_switch);
        assert!(
            state
                .side_one
                .volatile_statuses
                .contains(&PokemonVolatileStatus::MUSTRECHARGE),
            "recharge must survive until side one next acts"
        );

        let tackle =
            MoveChoice::from_string("tackle", &state.side_two).expect("Tackle is available");
        let recharge_turn =
            generate_instructions_from_move_pair(&mut state, &MoveChoice::None, &tackle, true);
        assert!(!recharge_turn.is_empty());
        for branch in recharge_turn {
            let rendered = render_branch_events(
                &mut state,
                &MoveChoice::None,
                &tackle,
                &branch.instruction_list,
                true,
                &EventContext {
                    turn: 5,
                    ..ctx.clone()
                },
            );
            let text = rendered.lines.join("\n");
            assert_eq!(
                text.matches("|cant|p1a: Charmander|recharge").count(),
                1,
                "the next turn is the sole recharge render: {text}"
            );
        }
    }

    #[test]
    fn legal_roll_snapshot_includes_an_earlier_stat_boost_but_not_the_hit() {
        let fixture = MINIMAL
            .trim()
            .replacen("EMBER;false;32", "FIREBLAST;false;8", 1)
            .replacen("WATERGUN;false;32", "CALMMIND;false;32", 1);
        let mut state = parse_state(&fixture).expect("fixture parses");
        let before = state.serialize();
        let s1 = MoveChoice::from_string("fireblast", &state.side_one).expect("Fire Blast");
        let s2 = MoveChoice::from_string("calmmind", &state.side_two).expect("Calm Mind");
        let branches = generate_instructions_from_move_pair(&mut state, &s1, &s2, true);

        let snapshot = branches.iter().find_map(|branch| {
            legal_roll_state_before_direct_damage(
                &mut state,
                &s1,
                &s2,
                &branch.instruction_list,
                true,
            )
        });
        let snapshot = snapshot.expect("Calm Mind before Fire Blast has a local roll state");
        let priced = State::deserialize(&snapshot);

        assert_eq!(priced.side_two.special_attack_boost, 1);
        assert_eq!(priced.side_two.special_defense_boost, 1);
        assert_eq!(priced.side_two.get_active_immutable().hp, 100);
        assert_eq!(
            state.serialize(),
            before,
            "snapshotting must restore the source state"
        );
    }

    #[test]
    fn legal_roll_snapshot_replaces_the_defender_before_a_same_turn_hit() {
        let mut state = parse_state(MINIMAL.trim()).expect("fixture parses");
        state.side_two.pokemon[PokemonIndex::P1] = state.side_two.pokemon[PokemonIndex::P0].clone();
        let before = state.serialize();
        let s1 = MoveChoice::from_string("tackle", &state.side_one).expect("Tackle");
        let s2 = MoveChoice::Switch(PokemonIndex::P1);
        let branches = generate_instructions_from_move_pair(&mut state, &s1, &s2, true);

        let snapshot = branches.iter().find_map(|branch| {
            legal_roll_state_before_direct_damage(
                &mut state,
                &s1,
                &s2,
                &branch.instruction_list,
                true,
            )
        });
        let snapshot = snapshot.expect("switch before Tackle has a local roll state");
        let priced = State::deserialize(&snapshot);

        assert_eq!(priced.side_two.active_index, PokemonIndex::P1);
        assert_eq!(priced.side_two.get_active_immutable().hp, 100);
        assert_eq!(
            state.serialize(),
            before,
            "snapshotting must restore the source state"
        );
    }

    #[test]
    fn switch_details_preserve_showdown_level_and_gender() {
        let mut state = parse_state(MINIMAL.trim()).expect("fixture parses");
        let pokemon = &mut state.side_one.pokemon[PokemonIndex::P1];
        pokemon.hp = 100;
        pokemon.maxhp = 100;
        pokemon.level = 84;
        pokemon.gender = PokemonGender::MALE;
        let mut context = ctx();
        context.species[0].push("Charmeleon".to_string());
        assert_eq!(
            context.details(&state, SideReference::SideOne, PokemonIndex::P0),
            "Charmander"
        );
        let serialized = state.serialize();
        let restored = State::deserialize(&serialized);
        assert_eq!(
            restored.side_one.pokemon[PokemonIndex::P1].gender,
            PokemonGender::MALE
        );

        let segment = vec![Instruction::Switch(
            poke_engine::instruction::SwitchInstruction {
                side_ref: SideReference::SideOne,
                previous_index: PokemonIndex::P0,
                next_index: PokemonIndex::P1,
            },
        )];
        let mut rendered = RenderedEvents::default();
        let mut sim = Sim::new(&mut state, [false, false]);
        render_switch_phase(
            &mut sim,
            SideReference::SideOne,
            &segment,
            &context,
            &mut rendered,
        );
        assert_eq!(
            rendered.lines,
            ["|switch|p1a: Charmeleon|Charmeleon, L84, M|100/100"]
        );
        sim.finish();
        assert_eq!(state.serialize(), serialized);
    }

    /// ONE SIDE's emission order, pinned against the REAL engine.
    ///
    /// The per-source tests below each pin one pair. None of them would catch a
    /// future engine that REORDERS the end-of-turn sections: the counts would
    /// still reconcile, the plan would still be "usable", and every tick would
    /// be silently mislabelled. This test is the guard for that — it drives
    /// `generate_instructions_from_move_pair` and asserts the rendered sequence,
    /// so a reorder in `add_end_of_turn_instructions` fails here.
    ///
    /// Five sources fire on side one at once (weather chip, Leftovers, Leech
    /// Seed, status, partial trap), which pins every ordering relationship
    /// between them WITHIN one Pokemon. Note that the sandstorm chip and the
    /// partial-trap tick are BOTH 20 here — the amount collision, live, in the
    /// same trace.
    ///
    /// It filters to `p1a`, and that is a real limitation: the residual phase is
    /// speed-major across the two sides, so a within-side sequence can be
    /// perfectly right while the interleaving is wrong. That half is pinned by
    /// `dual_side_emission_order_interleaves_by_speed` below.
    #[test]
    fn end_of_turn_section_order_is_pinned_against_the_engine() {
        let mut state = parse_state(MINIMAL.trim()).expect("fixture parses");
        state.weather.weather_type = Weather::SAND;
        state.weather.turns_remaining = 5;
        state
            .side_one
            .volatile_statuses
            .insert(PokemonVolatileStatus::LEECHSEED);
        state
            .side_one
            .volatile_statuses
            .insert(PokemonVolatileStatus::PARTIALLYTRAPPED);
        {
            let active = state.side_one.get_active();
            active.maxhp = 320;
            active.hp = 200;
            active.item = Items::LEFTOVERS;
            active.status = PokemonStatus::POISON;
            active.types = (PokemonType::NORMAL, PokemonType::TYPELESS);
        }
        {
            let active = state.side_two.get_active();
            active.maxhp = 320;
            active.hp = 150;
            active.item = Items::LEFTOVERS;
            active.types = (PokemonType::NORMAL, PokemonType::TYPELESS);
        }

        let s1 = MoveChoice::from_string("splash", &state.side_one)
            .or_else(|| MoveChoice::from_string("tackle", &state.side_one))
            .expect("a move");
        let s2 = MoveChoice::from_string("splash", &state.side_two)
            .or_else(|| MoveChoice::from_string("tackle", &state.side_two))
            .expect("a move");
        let branches = generate_instructions_from_move_pair(&mut state, &s1, &s2, true);
        let rendered = render_branch_events(
            &mut state,
            &s1,
            &s2,
            &branches[0].instruction_list,
            true,
            &ctx(),
        );

        let p1_tags: Vec<String> = rendered
            .lines
            .iter()
            .filter(|l| l.contains("p1a") && (l.contains("-damage") || l.contains("-heal")))
            .filter_map(|l| l.split("[from]").nth(1).map(|t| t.trim().to_string()))
            .collect();
        assert_eq!(
            p1_tags,
            vec![
                "Sandstorm".to_string(),
                "item: Leftovers".to_string(),
                "Leech Seed".to_string(),
                "psn".to_string(),
                "partiallytrapped".to_string(),
            ],
            "end-of-turn section order changed; ResidualPlan's order must be \
             updated in lock-step or every residual tick is mislabelled. \
             Rendered: {:?}",
            rendered.lines
        );
    }

    /// THE INTERLEAVING, pinned against the REAL engine — the half the single-side
    /// test above cannot see.
    ///
    /// gen3 resolves the residual phase speed-major: within an order class the
    /// faster Pokemon runs its WHOLE set before the slower one runs any. So with
    /// Leftovers on both sides and a status tick on the fast one, the rendered
    /// sequence has to be `fast heal, fast status, slow heal` — NOT the
    /// section-major `both heals, both statuses`.
    ///
    /// Transcribed from the real sim (`scripts/gen3_switch_differential.py::
    /// residualspeedmajorfast`; 206-speed Aipom with Leftovers and a burn vs a
    /// 96-speed badly poisoned Snorlax):
    ///
    /// ```text
    /// |-heal|p2a: Aipom|235/251 brn|[from] item: Leftovers
    /// |-damage|p2a: Aipom|204/251 brn|[from] brn
    /// |-damage|p1a: Snorlax|277/461 tox|[from] psn
    /// ```
    ///
    /// Seats are mirrored here (side ONE is the fast holder) so that a renderer
    /// that happened to key on the seat rather than on emission order would fail.
    #[test]
    fn dual_side_emission_order_interleaves_by_speed() {
        let mut state = parse_state(MINIMAL.trim()).expect("fixture parses");
        {
            let fast = state.side_one.get_active();
            fast.maxhp = 320;
            fast.hp = 200;
            fast.speed = 206;
            fast.item = Items::LEFTOVERS;
            fast.status = PokemonStatus::BURN;
            fast.types = (PokemonType::NORMAL, PokemonType::TYPELESS);
        }
        {
            let slow = state.side_two.get_active();
            slow.maxhp = 320;
            slow.hp = 200;
            slow.speed = 96;
            slow.status = PokemonStatus::POISON;
            slow.types = (PokemonType::NORMAL, PokemonType::TYPELESS);
        }

        let s1 = MoveChoice::from_string("splash", &state.side_one)
            .or_else(|| MoveChoice::from_string("tackle", &state.side_one))
            .expect("a move");
        let s2 = MoveChoice::from_string("splash", &state.side_two)
            .or_else(|| MoveChoice::from_string("tackle", &state.side_two))
            .expect("a move");
        let branches = generate_instructions_from_move_pair(&mut state, &s1, &s2, true);
        let rendered = render_branch_events(
            &mut state,
            &s1,
            &s2,
            &branches[0].instruction_list,
            true,
            &ctx(),
        );

        let tagged: Vec<String> = rendered
            .lines
            .iter()
            .filter(|l| l.contains("-damage") || l.contains("-heal"))
            .filter_map(|l| {
                let seat = l.split('|').nth(2)?.split(':').next()?.to_string();
                let tag = l.split("[from]").nth(1)?.trim().to_string();
                Some(format!("{seat} {tag}"))
            })
            .collect();
        assert_eq!(
            tagged,
            vec![
                "p1a item: Leftovers".to_string(),
                "p1a brn".to_string(),
                "p2a psn".to_string(),
            ],
            "the faster side's whole set must precede the slower side's. Rendered: {:?}",
            rendered.lines
        );
    }

    /// The seeder's silent drain heal is emitted at the SEEDED side's slot, so
    /// when the victim is faster it arrives BEFORE the seeder's own Leftovers —
    /// and `ResidualPlan` must not label the Leftovers tick with it.
    ///
    /// Transcribed from the sim (`residualspeedleech`):
    ///
    /// ```text
    /// |-heal|p2a: Aipom|135/251|[from] item: Leftovers
    /// |-damage|p2a: Aipom|104/251|[from] Leech Seed|[of] p1a: Cacturne
    /// |-heal|p1a: Cacturne|260/281|[silent]
    /// |-heal|p1a: Cacturne|277/281|[from] item: Leftovers
    /// ```
    #[test]
    fn a_seeders_drain_heal_does_not_steal_its_own_leftovers_tag() {
        let mut state = parse_state(MINIMAL.trim()).expect("fixture parses");
        {
            let seeder = state.side_one.get_active();
            seeder.maxhp = 320;
            seeder.hp = 200;
            seeder.item = Items::LEFTOVERS;
        }
        state
            .side_two
            .volatile_statuses
            .insert(PokemonVolatileStatus::LEECHSEED);
        // Victim resolves first: sap damage on side two, drain heal on side one,
        // then side one's own Leftovers.
        let segment = vec![
            Instruction::Damage(poke_engine::instruction::DamageInstruction {
                side_ref: SideReference::SideTwo,
                damage_amount: 40,
            }),
            heal_one(40),
            heal_one(20),
        ];
        assert_eq!(
            residual_tags(&mut state, &segment, "p1a"),
            vec!["item: Leftovers".to_string()],
            "only the Leftovers tick carries a tag; the drain is silent"
        );
    }

    // --- positional residual attribution (ledger H.1) ---------------------

    /// Render a whole residual segment through the plan, returning the `[from]`
    /// tags in emission order for the requested side.
    fn residual_tags(state: &mut State, segment: &[Instruction], side: &str) -> Vec<String> {
        let mut rendered = RenderedEvents::default();
        let mut sim = Sim::new(state, [false, false]);
        let mut plan = ResidualPlan::build(sim.state, segment);
        for (index, ins) in segment.iter().enumerate() {
            render_residual_instruction(
                &mut sim,
                ins,
                segment.get(index + 1),
                &mut plan,
                &ctx(),
                &mut rendered,
            );
        }
        sim.finish();
        rendered
            .lines
            .iter()
            .filter(|l| l.contains(side))
            .filter_map(|l| l.split("[from]").nth(1).map(|t| t.trim().to_string()))
            .collect()
    }

    fn damage_one(amount: i16) -> Instruction {
        Instruction::Damage(poke_engine::instruction::DamageInstruction {
            side_ref: SideReference::SideOne,
            damage_amount: amount,
        })
    }

    fn heal_one(amount: i16) -> Instruction {
        Instruction::Heal(poke_engine::instruction::HealInstruction {
            side_ref: SideReference::SideOne,
            heal_amount: amount,
        })
    }

    /// A poisoned mon in a sandstorm takes BOTH ticks. The old attributor tested
    /// status before weather and labelled both `psn`.
    #[test]
    fn poisoned_in_sand_attributes_each_tick_separately() {
        let mut state = parse_state(MINIMAL.trim()).expect("fixture parses");
        state.weather.weather_type = Weather::SAND;
        state.weather.turns_remaining = 5;
        {
            let active = state.side_one.get_active();
            active.maxhp = 320;
            active.hp = 320;
            active.status = PokemonStatus::POISON;
            active.item = Items::NONE;
        }
        let segment = vec![damage_one(20), damage_one(40)];
        assert_eq!(
            residual_tags(&mut state, &segment, "p1a"),
            vec!["Sandstorm".to_string(), "psn".to_string()],
        );
    }

    /// THE COUNTEREXAMPLE TO AMOUNT MATCHING, pinned on purpose.
    ///
    /// The sandstorm chip and the partial-trap tick are BOTH `maxhp/16`, so on a
    /// mon carrying both they are numerically IDENTICAL and no amount-based
    /// attributor can ever separate them. Only the engine's emission order can.
    #[test]
    fn sand_and_trap_collide_on_amount_and_only_order_separates_them() {
        let mut state = parse_state(MINIMAL.trim()).expect("fixture parses");
        state.weather.weather_type = Weather::SAND;
        state.weather.turns_remaining = 5;
        state
            .side_one
            .volatile_statuses
            .insert(PokemonVolatileStatus::PARTIALLYTRAPPED);
        {
            let active = state.side_one.get_active();
            active.maxhp = 320;
            active.hp = 320;
            active.item = Items::NONE;
        }
        // Both ticks are 320/16 = 20. Identical numbers, different sources.
        let segment = vec![damage_one(20), damage_one(20)];
        assert_eq!(
            residual_tags(&mut state, &segment, "p1a"),
            vec!["Sandstorm".to_string(), "partiallytrapped".to_string()],
            "amounts are equal; attribution must come from emission order alone"
        );
    }

    /// A Leftovers holder whose OPPONENT is seeded: the old attributor checked
    /// the opponent's LEECHSEED before Leftovers and tagged the holder's own
    /// tick `Leech Seed`.
    #[test]
    fn leftovers_holder_facing_a_seeded_opponent_keeps_its_own_tag() {
        let mut state = parse_state(MINIMAL.trim()).expect("fixture parses");
        {
            let active = state.side_one.get_active();
            active.maxhp = 320;
            active.hp = 200;
            active.item = Items::LEFTOVERS;
        }
        state
            .side_two
            .volatile_statuses
            .insert(PokemonVolatileStatus::LEECHSEED);
        // Leftovers (order 5) then the Leech Seed sap heal (order 8).
        let segment = vec![heal_one(20), heal_one(40)];
        // The sap heal on the SEEDER is silent in Showdown, so it carries no
        // `[from]` tag at all — only the Leftovers tick does.
        assert_eq!(
            residual_tags(&mut state, &segment, "p1a"),
            vec!["item: Leftovers".to_string()],
        );
    }

    /// Three simultaneous sources on one side, in the engine's order:
    /// weather chip -> order-5 Leftovers -> status damage.
    #[test]
    fn triple_source_side_attributes_all_three_in_order() {
        let mut state = parse_state(MINIMAL.trim()).expect("fixture parses");
        state.weather.weather_type = Weather::SAND;
        state.weather.turns_remaining = 5;
        {
            let active = state.side_one.get_active();
            active.maxhp = 320;
            active.hp = 200;
            active.item = Items::LEFTOVERS;
            active.status = PokemonStatus::POISON;
        }
        let segment = vec![damage_one(20), heal_one(20), damage_one(40)];
        assert_eq!(
            residual_tags(&mut state, &segment, "p1a"),
            vec![
                "Sandstorm".to_string(),
                "item: Leftovers".to_string(),
                "psn".to_string()
            ],
        );
    }

    /// The plan is only trusted when it predicts EXACTLY what the segment
    /// emits. An unexpected extra tick must fall back to the generic tag —
    /// loud (it diverges) rather than confidently wrong.
    #[test]
    fn count_mismatch_falls_back_to_the_generic_tag() {
        let mut state = parse_state(MINIMAL.trim()).expect("fixture parses");
        {
            let active = state.side_one.get_active();
            active.maxhp = 320;
            active.hp = 320;
            active.item = Items::NONE;
            active.status = PokemonStatus::POISON;
        }
        // The plan predicts ONE damage (psn); the segment emits two.
        let segment = vec![damage_one(40), damage_one(20)];
        let tags = residual_tags(&mut state, &segment, "p1a");
        assert!(
            tags.iter().all(|t| t == "psn"),
            "a desynced plan must not invent labels; got {tags:?}"
        );
    }

    /// A side merely CARRYING a pending wish must not have its ordinary
    /// Leftovers tick attributed to Wish. Regression for the mapper mis-tag that
    /// inflated the strict differential's divergence count (ledger Appendix B.5):
    /// `residual_heal_cause` keyed on `wish.0 > 0`, but the engine emits
    /// `DecrementWish` BEFORE the Leftovers heal on a non-resolving turn, so the
    /// counter is still positive when the heal renders. Only a heal IMMEDIATELY
    /// FOLLOWED by `DecrementWish` is the wish landing.
    #[test]
    fn pending_wish_does_not_steal_the_leftovers_tag() {
        let mut state = parse_state(MINIMAL.trim()).expect("fixture parses");
        state.side_one.get_active().item = Items::LEFTOVERS;
        state.side_one.wish = (2, 50);
        let heal = Instruction::Heal(poke_engine::instruction::HealInstruction {
            side_ref: SideReference::SideOne,
            heal_amount: 6,
        });

        // Pending but NOT resolving: the next instruction is not DecrementWish.
        let mut rendered = RenderedEvents::default();
        let mut sim = Sim::new(&mut state, [false, false]);
        let mut plan = ResidualPlan::default();
        render_residual_instruction(&mut sim, &heal, None, &mut plan, &ctx(), &mut rendered);
        sim.finish();
        assert!(
            rendered.lines[0].contains("[from] item: Leftovers"),
            "pending wish stole the Leftovers tag: {:?}",
            rendered.lines
        );

        // Resolving: DecrementWish for the SAME side follows immediately.
        let decrement =
            Instruction::DecrementWish(poke_engine::instruction::DecrementWishInstruction {
                side_ref: SideReference::SideOne,
            });
        let mut rendered = RenderedEvents::default();
        let mut sim = Sim::new(&mut state, [false, false]);
        let mut plan = ResidualPlan::default();
        render_residual_instruction(
            &mut sim,
            &heal,
            Some(&decrement),
            &mut plan,
            &ctx(),
            &mut rendered,
        );
        sim.finish();
        assert!(
            rendered.lines[0].contains("[from] move: Wish"),
            "resolving wish not attributed: {:?}",
            rendered.lines
        );

        // A DecrementWish for the OTHER side must not claim this heal.
        let other =
            Instruction::DecrementWish(poke_engine::instruction::DecrementWishInstruction {
                side_ref: SideReference::SideTwo,
            });
        let mut rendered = RenderedEvents::default();
        let mut sim = Sim::new(&mut state, [false, false]);
        let mut plan = ResidualPlan::default();
        render_residual_instruction(
            &mut sim,
            &heal,
            Some(&other),
            &mut plan,
            &ctx(),
            &mut rendered,
        );
        sim.finish();
        assert!(
            rendered.lines[0].contains("[from] item: Leftovers"),
            "other side's wish stole the tag: {:?}",
            rendered.lines
        );
    }

    #[test]
    fn liquid_ooze_negative_heal_renders_as_lethal_damage() {
        let mut state = parse_state(MINIMAL.trim()).expect("fixture parses");
        state.side_one.get_active().hp = 10;
        let before = state.serialize();
        let instruction = Instruction::Heal(poke_engine::instruction::HealInstruction {
            side_ref: SideReference::SideOne,
            heal_amount: -10,
        });
        let mut rendered = RenderedEvents::default();
        let mut sim = Sim::new(&mut state, [false, false]);
        let mut plan = ResidualPlan::default();
        render_residual_instruction(
            &mut sim,
            &instruction,
            None,
            &mut plan,
            &ctx(),
            &mut rendered,
        );
        assert_eq!(
            rendered.lines,
            [
                "|-damage|p1a: Charmander|0 fnt|[from] ability: Liquid Ooze|[of] p2a: Squirtle",
                "|faint|p1a: Charmander",
            ]
        );
        sim.finish();
        assert_eq!(state.serialize(), before);
    }

    #[test]
    fn renders_simple_damaging_turn() {
        let mut state = parse_state(MINIMAL.trim()).expect("fixture parses");
        let s1 = MoveChoice::from_string("tackle", &state.side_one).unwrap();
        let s2 = MoveChoice::from_string("tackle", &state.side_two).unwrap();
        let before = state.serialize();
        let branches = generate_instructions_from_move_pair(&mut state, &s1, &s2, true);
        assert!(!branches.is_empty());
        for branch in &branches {
            let rendered =
                render_branch_events(&mut state, &s1, &s2, &branch.instruction_list, true, &ctx());
            assert!(
                rendered.lossy.is_empty(),
                "branch failed to segment: {:?} / {:?}",
                rendered.lossy,
                branch.instruction_list
            );
            let text = rendered.lines.join("\n");
            assert!(
                text.contains("|move|p1a: Charmander|tackle|p2a: Squirtle"),
                "{text}"
            );
            assert!(
                text.contains("|move|p2a: Squirtle|tackle|p1a: Charmander"),
                "{text}"
            );
            assert!(text.contains("|-damage|"), "{text}");
            assert!(text.contains("|upkeep"), "{text}");
            assert!(rendered.turn_completed, "{text}");
            assert!(text.contains("|turn|5"), "{text}");
            // Damage lines carry plain ASCII cur/max integers (fold input
            // contract).
            for line in &rendered.lines {
                if line.starts_with("|-damage|") {
                    let hp = line.split('|').nth(3).unwrap();
                    assert!(
                        hp == "0 fnt" || hp.split('/').all(|p| p.parse::<i64>().is_ok()),
                        "malformed hp field {hp} in {line}"
                    );
                }
            }
        }
        // State restored exactly.
        assert_eq!(before, state.serialize());
    }

    /// Ghost-typed Curse (live-game protocol probe, 2026-07-19): the real
    /// protocol targets the opponent, starts the curse on the TARGET, and
    /// never shows boost lines. The gen3 engine applies the non-Ghost
    /// stats-up delta instead — the renderer emits the real-protocol shape,
    /// suppresses the spurious boosts, and flags the branch lossy (the true
    /// self-HP cut is not derivable from the engine delta).
    #[test]
    fn ghost_curse_renders_target_start_and_no_boosts() {
        let fixture = MINIMAL
            .trim()
            .replace("FIRE", "GHOST")
            .replace("EMBER", "CURSE");
        let mut state = parse_state(&fixture).expect("fixture parses");
        let s1 = MoveChoice::from_string("curse", &state.side_one).unwrap();
        let s2 = MoveChoice::from_string("tackle", &state.side_two).unwrap();
        let branches = generate_instructions_from_move_pair(&mut state, &s1, &s2, true);
        assert!(!branches.is_empty());
        for branch in &branches {
            let rendered =
                render_branch_events(&mut state, &s1, &s2, &branch.instruction_list, true, &ctx());
            let text = rendered.lines.join("\n");
            assert!(
                text.contains("|move|p1a: Charmander|curse|p2a: Squirtle"),
                "{text}"
            );
            assert!(
                text.contains("|-start|p2a: Squirtle|Curse|[of] p1a: Charmander"),
                "{text}"
            );
            assert!(!text.contains("|-boost|p1a: Charmander"), "{text}");
            assert!(!text.contains("|-unboost|p1a: Charmander"), "{text}");
            assert!(
                rendered
                    .lossy
                    .iter()
                    .any(|tag| tag == "ghost_curse_engine_model"),
                "ghost curse must be flagged lossy: {:?}",
                rendered.lossy
            );
        }
    }

    /// Non-Ghost Curse keeps the corpus-measured self-target render with the
    /// real boost lines (regression guard for the Ghost gate).
    #[test]
    fn non_ghost_curse_renders_self_target_boosts() {
        let fixture = MINIMAL.trim().replace("EMBER", "CURSE");
        let mut state = parse_state(&fixture).expect("fixture parses");
        let s1 = MoveChoice::from_string("curse", &state.side_one).unwrap();
        let s2 = MoveChoice::from_string("tackle", &state.side_two).unwrap();
        let branches = generate_instructions_from_move_pair(&mut state, &s1, &s2, true);
        assert!(!branches.is_empty());
        for branch in &branches {
            let rendered =
                render_branch_events(&mut state, &s1, &s2, &branch.instruction_list, true, &ctx());
            let text = rendered.lines.join("\n");
            assert!(
                text.contains("|move|p1a: Charmander|curse|p1a: Charmander"),
                "{text}"
            );
            assert!(text.contains("|-boost|p1a: Charmander|atk|1"), "{text}");
            assert!(
                !rendered.lossy.iter().any(|t| t.contains("curse")),
                "{:?}",
                rendered.lossy
            );
        }
    }

    /// Flash Fire FIRST activation (absorb-class audit): the engine's delta
    /// is ApplyVolatileStatus(FLASHFIRE) on the defender; the real protocol's
    /// shape is the boost-state form `|-start|p2a: Houndoom|ability: Flash
    /// Fire` (live capture, absorb-audit probe 3) — an absorb SIGNATURE the
    /// fold consumes. A bare |move| render would lose the Absorbed outcome.
    #[test]
    fn flash_fire_first_activation_renders_absorb_start() {
        let mut state = parse_state(MINIMAL.trim()).expect("fixture parses");
        let active = state.side_two.active_index;
        state.side_two.pokemon[active].ability = Abilities::FLASHFIRE;
        let s1 = MoveChoice::from_string("ember", &state.side_one).unwrap();
        let s2 = MoveChoice::from_string("tackle", &state.side_two).unwrap();
        let branches = generate_instructions_from_move_pair(&mut state, &s1, &s2, true);
        assert!(!branches.is_empty());
        let mut checked = 0;
        for branch in &branches {
            let applies_flashfire = branch.instruction_list.iter().any(|ins| {
                matches!(
                    ins,
                    Instruction::ApplyVolatileStatus(apply)
                        if apply.volatile_status == PokemonVolatileStatus::FLASHFIRE
                )
            });
            if !applies_flashfire {
                continue;
            }
            let rendered =
                render_branch_events(&mut state, &s1, &s2, &branch.instruction_list, true, &ctx());
            let text = rendered.lines.join("\n");
            assert!(
                text.contains("|-start|p2a: Squirtle|ability: Flash Fire"),
                "{text}"
            );
            // Through the fold: the -start form is the absorb signature —
            // the attacker's move token must read Absorbed, not a bare move.
            let mut fold = crate::fold::FoldStateInner::initial(0, 128, 512);
            fold.advance_in_place(&rendered.lines)
                .expect("fold advances over the rendered lines");
            let products = fold.products();
            let ember = products
                .transition_tokens
                .iter()
                .find(|token| token.action == "ember")
                .expect("ember token present");
            assert_eq!(ember.damage_outcome, crate::fold::Outcome::Absorbed);
            checked += 1;
        }
        assert!(checked > 0, "no branch applied the FLASHFIRE volatile");
    }

    /// Memento's engine-side `Heal(-remaining_hp)` is reversible machinery,
    /// not a Liquid Ooze drain reversal. The protocol must preserve the target
    /// stat drops and defer the user faint; the fold then charges the user's
    /// actual remaining fraction, not a synthetic damage line.
    #[test]
    fn memento_negative_heal_preserves_order_and_fold_self_cost() {
        let fixture = MINIMAL
            .trim()
            .replace("EMBER;false;32", "SPLASH;false;32")
            .replace("WATERGUN;false;32", "MEMENTO;false;32");
        let mut state = parse_state(&fixture).expect("fixture parses");
        state.side_one.get_active().speed = 500;
        state.side_two.get_active().speed = 1;
        state.side_two.get_active().maxhp = 100;
        state.side_two.get_active().hp = 60;
        let s1 = MoveChoice::from_string("splash", &state.side_one).expect("Splash");
        let s2 = MoveChoice::from_string("memento", &state.side_two).expect("Memento");
        let branches = generate_instructions_from_move_pair(&mut state, &s1, &s2, false);
        let branch = branches
            .iter()
            .find(|branch| {
                branch.instruction_list.iter().any(|instruction| {
                    matches!(instruction, Instruction::Boost(boost)
                        if boost.side_ref == SideReference::SideOne && boost.amount == -2)
                })
            })
            .expect("successful Memento branch with target stat drops");
        let rendered = render_branch_events(
            &mut state,
            &s1,
            &s2,
            &branch.instruction_list,
            false,
            &ctx(),
        );
        assert_eq!(
            rendered.lines,
            [
                "|",
                "|move|p1a: Charmander|splash||[still]",
                "|move|p2a: Squirtle|memento|p1a: Charmander",
                "|-unboost|p1a: Charmander|atk|2",
                "|-unboost|p1a: Charmander|spa|2",
                "|faint|p2a: Squirtle",
                "|",
                "|upkeep",
            ],
            "Memento must not render its self-faint as Liquid Ooze damage"
        );
        let mut fold = crate::fold::FoldStateInner::initial(0, 128, 512);
        let mut lines = vec![
            "|switch|p1a: Charmander|Charmander, L100|100/100".to_string(),
            "|switch|p2a: Squirtle|Squirtle, L100|60/100".to_string(),
            "|turn|4".to_string(),
        ];
        lines.extend(rendered.lines);
        fold.advance_in_place(&lines)
            .expect("fold advances over the Memento protocol");
        let memento = fold
            .products()
            .transition_tokens
            .into_iter()
            .find(|token| token.action == "memento")
            .expect("Memento token present");
        assert!(
            (memento.self_hp_cost - 0.60).abs() < f64::EPSILON,
            "Memento must charge the user's remaining HP fraction: {memento:?}"
        );
    }

    /// The /100 base reconciliation (ladder streams): the exact Showdown HP
    /// Percentage Mod formula, including both special cases.
    #[test]
    fn hp_percent_condition_matches_showdown() {
        // ceil rounding: 1/335 -> 1%, never 0 while alive.
        assert_eq!(hp_percent_condition(1, 335), "1/100");
        // near-full clamps to 99 while hp < maxhp (pokemon.ts getHealth).
        assert_eq!(hp_percent_condition(334, 335), "99/100");
        assert_eq!(hp_percent_condition(335, 335), "100/100");
        assert_eq!(hp_percent_condition(148, 196), "76/100");
        assert_eq!(hp_percent_condition(100, 100), "100/100");
        assert_eq!(hp_percent_condition(99, 100), "99/100");
    }

    /// With `hp_percent` set for side two, every rendered HP condition about
    /// side two's mons lands on the /100 grid — the same grid a ladder root
    /// fold consumed — while side one keeps the exact base. The fixture's
    /// side-two mon is rescaled to 335 max HP so the two bases are visibly
    /// different strings.
    #[test]
    fn renders_side_two_on_percent_base() {
        let mut state = parse_state(MINIMAL.trim()).expect("fixture parses");
        let active = state.side_two.active_index;
        state.side_two.pokemon[active].hp = 335;
        state.side_two.pokemon[active].maxhp = 335;
        let s1 = MoveChoice::from_string("tackle", &state.side_one).unwrap();
        let s2 = MoveChoice::from_string("tackle", &state.side_two).unwrap();
        let mut ctx = ctx();
        ctx.hp_percent = [false, true];
        let branches = generate_instructions_from_move_pair(&mut state, &s1, &s2, true);
        let mut saw_p2_damage = false;
        for branch in &branches {
            let rendered =
                render_branch_events(&mut state, &s1, &s2, &branch.instruction_list, true, &ctx);
            for line in &rendered.lines {
                if !line.starts_with("|-damage|") && !line.starts_with("|-heal|") {
                    continue;
                }
                let ident = line.split('|').nth(2).unwrap_or("");
                let hp = line.split('|').nth(3).unwrap_or("");
                if hp == "0 fnt" || !ident.starts_with("p2a:") {
                    continue;
                }
                saw_p2_damage = true;
                let (cur, base) = hp.split_once('/').expect("condition has a base");
                assert_eq!(base, "100", "side-two condition {hp} not /100 in {line}");
                let pct: i32 = cur.parse().expect("percent parses");
                assert!((1..=99).contains(&pct), "damaged percent {pct} in {line}");
            }
        }
        assert!(saw_p2_damage, "fixture must produce side-two damage lines");
    }

    /// Pain Split renders as the sim renders it: TWO `-sethp` lines carrying an
    /// IDENTICAL `[from] move: Pain Split` payload, the target's `[silent]` and
    /// the user's not.
    ///
    /// Transcribed from the sim (`data/moves.ts`, the only `-sethp` emitter in
    /// the pool; no gen3/4/5 mod overrides it):
    ///
    /// ```text
    /// |-sethp|p2a: Wigglytuff|128/407|[from] move: Pain Split|[silent]
    /// |-sethp|p1a: Dusclops|128/209|[from] move: Pain Split
    /// ```
    ///
    /// The pairing is the load-bearing part. The differential compares
    /// components by their normalized `[from]` source, so if the two halves
    /// were tagged differently — or one were left bare, as they both were
    /// before this — the rows come back as attribution mismatches instead of
    /// matching. `fold.rs` keys its Pain Split `self_hp_cost` branch on the
    /// same tag, so a bare render also silently skipped that charge on the
    /// engine-as-environment path.
    ///
    /// One known and deliberate difference from the sim: the engine emits the
    /// USER's half first and the sim emits the TARGET's first (the engine's own
    /// instruction order, kept per the positional-attribution rule). This is
    /// not load-bearing — the two halves land on different slots, and the
    /// differential compares components per slot, so cross-slot order cannot
    /// affect a verdict.
    #[test]
    fn pain_split_renders_paired_sethp_with_identical_attribution() {
        let fixture = MINIMAL.trim().replace("EMBER", "PAINSPLIT");
        let mut state = parse_state(&fixture).expect("fixture parses");
        // Asymmetric HP, or Pain Split moves nothing and emits no lines.
        state.side_one.get_active().maxhp = 209;
        state.side_one.get_active().hp = 132;
        state.side_two.get_active().maxhp = 407;
        state.side_two.get_active().hp = 125;
        let s1 = MoveChoice::from_string("painsplit", &state.side_one).unwrap();
        let s2 = MoveChoice::from_string("tackle", &state.side_two).unwrap();
        let branches = generate_instructions_from_move_pair(&mut state, &s1, &s2, true);
        assert!(!branches.is_empty());
        for branch in &branches {
            let rendered =
                render_branch_events(&mut state, &s1, &s2, &branch.instruction_list, true, &ctx());
            let sethp: Vec<&String> = rendered
                .lines
                .iter()
                .filter(|l| l.starts_with("|-sethp|"))
                .collect();
            assert_eq!(
                sethp.len(),
                2,
                "both halves must be -sethp. Rendered: {:?}",
                rendered.lines
            );

            // The pairing pin: one identical [from] payload across both halves.
            let tags: Vec<String> = sethp
                .iter()
                .map(|l| l.split("[from]").nth(1).unwrap().trim().to_string())
                .map(|t| t.trim_end_matches("|[silent]").trim().to_string())
                .collect();
            assert_eq!(
                tags,
                vec![
                    "move: Pain Split".to_string(),
                    "move: Pain Split".to_string()
                ],
                "the two halves must carry the SAME attribution: {sethp:?}"
            );

            // Target silent, user visible — exactly the sim's split.
            let silent: Vec<bool> = sethp.iter().map(|l| l.contains("[silent]")).collect();
            assert_eq!(
                silent.iter().filter(|s| **s).count(),
                1,
                "exactly one half is [silent]: {sethp:?}"
            );
            assert!(
                sethp
                    .iter()
                    .any(|l| l.contains("p2a") && l.contains("[silent]")),
                "the TARGET's half is the silent one: {sethp:?}"
            );

            // And neither half may still be rendered as bare damage. Scoped to
            // the Pain Split window — a later bare `-damage` is the OPPONENT's
            // move landing, which is correctly bare.
            let window: Vec<&String> = rendered
                .lines
                .iter()
                .skip_while(|l| !l.contains("|painsplit|"))
                .skip(1)
                .take_while(|l| !l.starts_with("|move|"))
                .collect();
            assert!(
                !window
                    .iter()
                    .any(|l| l.starts_with("|-damage|") && !l.contains("[from]")),
                "no bare -damage may survive inside the Pain Split window: {window:?}"
            );
        }
    }
}
