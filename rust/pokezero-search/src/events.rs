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
//! the match is exact; a branch that fails to segment is reported as such
//! (`lossy`), never silently mis-rendered.
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
//! - Sleep Talk's called move id is not in the delta — the called move's
//!   effects are attributed to the Sleep Talk window (flagged lossy).
//!
//! Lines the fold provably ignores (fold.rs `process_line`) are deliberately
//! NOT rendered: `|-singleturn|`, `|-curestatus|`, `|-fail|`, `|-ability|`,
//! `|-enditem|`, `|-mustrecharge|`, `|-start|` (except absorb signatures),
//! `|-anim|`, `|debug|`. Omissions are part of the documented contract.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use poke_engine::choices::{Choice, Choices, MoveCategory, MoveTarget};
use poke_engine::engine::abilities::Abilities;
use poke_engine::engine::damage_calc::type_effectiveness_modifier;
use poke_engine::engine::generate_instructions::{
    calculate_both_damage_rolls, generate_instructions_from_move,
    generate_instructions_from_move_pair,
};
use poke_engine::engine::items::Items;
use poke_engine::engine::state::{MoveChoice, PokemonVolatileStatus, Weather};
use poke_engine::instruction::{Instruction, StateInstructions};
use poke_engine::state::{
    PokemonBoostableStat, PokemonIndex, PokemonSideCondition, PokemonStatus, SideReference, State,
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
        Ok(EventContext { species, turn })
    }

    fn display(&self, side: SideReference, index: PokemonIndex) -> String {
        let list = &self.species[side_usize(side)];
        let i = pokemon_index_usize(index);
        list.get(i)
            .cloned()
            .unwrap_or_else(|| format!("unknown{}", i))
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
    /// Non-empty when part of the branch could not be attributed exactly;
    /// each entry is a stable reason slug. Rendering is still fold-safe
    /// (unattributed residual damage carries a `[from]` tag), but the branch
    /// should be counted, not trusted blindly.
    pub lossy: Vec<String>,
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
            let mut c = side_ref.get_active_immutable().moves[move_index].choice.clone();
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
fn end_of_turn_triggered(s1: &MoveChoice, s2: &MoveChoice) -> bool {
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
    let eot = end_of_turn_triggered(s1_move, s2_move);

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
}

impl<'a> Sim<'a> {
    fn new(state: &'a mut State) -> Sim<'a> {
        Sim {
            state,
            applied: Vec::new(),
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
            "0 fnt".to_string()
        } else {
            format!("{hp}/{maxhp}")
        }
    }

    fn finish(self) {
        // Restore the caller's state exactly (reverse in reverse order).
        self.state.reverse_instructions(&self.applied);
    }
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
    let inflicts = |status: PokemonStatus| {
        choice
            .status
            .as_ref()
            .map_or(false, |s| s.status == status)
    };
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
    let eot_triggered = end_of_turn_triggered(s1_move, s2_move);
    // For (switch, none) shapes: was the switching side's active already
    // fainted (faint replacement — the faint ply already ran residuals +
    // upkeep) or alive (pivot — the engine never runs the pivot turn's
    // residuals; documented deviation)?
    let pre_ply_replacement = if matches!(s1_move, MoveChoice::Switch(_)) && s2_move == &MoveChoice::None
    {
        state.side_one.get_active_immutable().hp <= 0
    } else if s1_move == &MoveChoice::None && matches!(s2_move, MoveChoice::Switch(_)) {
        state.side_two.get_active_immutable().hp <= 0
    } else {
        false
    };

    let seg = match segment(state, s1_move, s2_move, instructions, branch_on_damage) {
        Some(seg) => seg,
        None => {
            // Fold-safe fallback: apply everything, render nothing but the
            // state-tracking lines we can attribute without phases (hp only),
            // and mark the branch lossy so callers can count it.
            out.lossy.push("segmentation_failed".to_string());
            let mut sim = Sim::new(state);
            for ins in instructions {
                render_residual_instruction(&mut sim, ins, ctx, &mut out, true);
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
            return out;
        }
    };

    let second_ref = other_side(seg.first);
    let (first_mc, second_mc) = match seg.first {
        SideReference::SideOne => (s1_move, s2_move),
        SideReference::SideTwo => (s2_move, s1_move),
    };

    let mut sim = Sim::new(state);
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
        for ins in residual_segment {
            render_residual_instruction(&mut sim, ins, ctx, &mut out, false);
        }
        // A pivot in flight (U-turn/Baton Pass chose to switch, the engine
        // skipped residuals): the turn is not over — no |upkeep| yet.
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
    let force_switch_pending =
        sim.state.side_one.force_switch || sim.state.side_two.force_switch;
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
        // |upkeep|; the real protocol places the replacement switch before
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
                let display = ctx.display(switch.side_ref, switch.next_index);
                let ident = ctx.ident(switch.side_ref, switch.next_index);
                let condition = sim.hp_condition(switch.side_ref);
                let mut line =
                    format!("|switch|{ident}|{display}|{condition}");
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
                out.lines.push(render_boost_line(ctx, sim, boost.side_ref, boost.stat, boost.amount, None));
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

struct MovePrelude {
    used_move: bool,
    woke_up: bool,
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
            if remove.side_ref == side
                && remove.volatile_status == PokemonVolatileStatus::TRUANT
            {
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
                // Natural / Rest wake. Real protocol: |-curestatus| (ignored
                // by the fold — omitted).
                sim.apply(ins);
                *cursor += 1;
                prelude.woke_up = true;
                sleep_gate_seen = true;
            }
            Instruction::ChangeStatus(change)
                if change.side_ref == side
                    && change.old_status == PokemonStatus::FREEZE
                    && change.new_status == PokemonStatus::NONE =>
            {
                // Thaw (real: |-curestatus|..|frz| — fold-ignored).
                sim.apply(ins);
                *cursor += 1;
                sleep_gate_seen = true;
            }
            Instruction::SetSleepTurns(set) if set.side_ref == side => {
                sim.apply(ins);
                *cursor += 1;
                if set.new_turns > set.previous_turns && !prelude.woke_up {
                    // Stayed asleep (sleep-turn counter advanced, no wake).
                    // Sleep Talk still acts through the sleep — the real
                    // protocol shows |cant|..|slp| AND the Sleep Talk lines.
                    let ident = ctx.active_ident(sim.state, side);
                    out.lines.push(format!("|cant|{ident}|slp"));
                    if choice.move_id == Choices::SLEEPTALK {
                        // Everything after the sleep gate belongs to the
                        // CALLED move (e.g. a called Refresh's status cure
                        // must not be mis-read as a natural wake).
                        return prelude;
                    }
                    prelude.used_move = false;
                }
                sleep_gate_seen = true;
            }
            Instruction::DecrementRestTurns(_) => {
                sim.apply(ins);
                *cursor += 1;
                if !prelude.woke_up {
                    // Still resting.
                    let ident = ctx.active_ident(sim.state, side);
                    out.lines.push(format!("|cant|{ident}|slp"));
                    if choice.move_id == Choices::SLEEPTALK {
                        return prelude; // see above
                    }
                    prelude.used_move = false;
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

    if called_tag.is_none() {
        let prelude = consume_move_prelude(sim, side, choice, segment, &mut cursor, ctx, out);
        if !prelude.used_move {
            return;
        }
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
            let tail: Vec<Instruction> = segment[cursor..].to_vec();
            match identify_sleep_talk_called(sim.state, side, &tail, branch_on_damage) {
                Some(called_choice) => {
                    render_move_phase(
                        sim,
                        side,
                        &called_choice,
                        &tail,
                        branch_on_damage,
                        ctx,
                        out,
                        Some("Sleep Talk"),
                    );
                }
                None => {
                    // Unrecoverable from the delta (documented insufficiency):
                    // fold-safe fallback — the effects (if any) accrue to
                    // the Sleep Talk window. ALWAYS flagged, even for an
                    // empty delta (an ambiguous no-op call still means the
                    // real stream had a called-move line we cannot emit).
                    out.lossy.push("sleeptalk_called_unidentified".to_string());
                    for ins in &tail {
                        render_residual_instruction(sim, ins, ctx, out, true);
                    }
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

    // Confusion self-hit: the residue after the prelude is a single Damage on
    // the user (the engine's hit-yourself branch).
    if segment[cursor..].len() == 1 {
        if let Instruction::Damage(damage) = &segment[cursor] {
            if damage.side_ref == side {
                let confused = {
                    let s = match side {
                        SideReference::SideOne => &sim.state.side_one,
                        SideReference::SideTwo => &sim.state.side_two,
                    };
                    s.volatile_statuses
                        .contains(&PokemonVolatileStatus::CONFUSION)
                };
                if confused && choice.crash.is_none() {
                    let ident = ctx.active_ident(sim.state, side);
                    sim.apply(&segment[cursor]);
                    let condition = sim.hp_condition(side);
                    out.lines
                        .push(format!("|-damage|{ident}|{condition}|[from] confusion"));
                    emit_faint_if_dead(sim, side, ctx, out);
                    return;
                }
            }
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
    let tail = &segment[cursor..];
    let deals_damage_to_defender = tail.iter().any(|ins| match ins {
        Instruction::Damage(d) => d.side_ref == defender,
        Instruction::DamageSubstitute(d) => d.side_ref == defender,
        _ => false,
    });
    let has_any_effect = !tail.is_empty();

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
    let self_target = choice.target == MoveTarget::User || non_ghost_curse;
    // A status-inflicting move against an already-statused defender cannot
    // work; the engine merges its no-op "hit" branch with the miss branch,
    // and the fail outcome carries most of the probability mass — render the
    // real protocol's fail form (blank target + [still]), never [miss]
    // (documented ambiguity: a real 15%-miss renders identically here).
    // Type-based status immunity (Steel/Poison vs psn, Fire vs brn, Ice vs
    // frz): the real protocol shows |-immune| and it wins over the
    // already-statused fail (PS checks immunity first).
    let status_type_immune = choice
        .status
        .as_ref()
        .map_or(false, |status| {
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
                    active.has_type(&PokemonType::POISON)
                        || active.has_type(&PokemonType::STEEL)
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
    // Boost moves whose stats are all at cap generate no instructions but
    // still "succeed" in the real protocol (explicit self target + 0-amount
    // boost lines).
    let capped_boost_move =
        self_target && !has_any_effect && choice.boost.is_some() && choice.category == MoveCategory::Status;
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
    if attacker_paralyzed && !has_any_effect && called_tag.is_none() {
        let side_condition_already_up = choice.side_condition.as_ref().map_or(false, |sc| {
            let target_side = match sc.target {
                MoveTarget::User => side,
                MoveTarget::Opponent => defender,
            };
            side_condition_value(sim.state, target_side, sc.condition) > 0
        });
        let deterministic_noop = (defender_protected && choice.flags.protect)
            || (is_damaging && effectiveness == 0.0)
            || (is_damaging && absorb.is_some())
            || ability_immune.is_some()
            || status_fail
            || status_type_immune
            || capped_boost_move
            || side_condition_already_up;
        let (attacker_hp, attacker_maxhp) = sim.active_hp(side);
        let move_could_act = is_damaging
            || choice.status.is_some()
            || (choice.heal.is_some() && attacker_hp < attacker_maxhp)
            || choice.volatile_status.is_some()
            || choice.side_condition.is_some()
            || choice.boost.is_some();
        if !deterministic_noop && move_could_act {
            out.lines.push(format!("|cant|{attacker_ident}|par"));
            return;
        }
    }

    // Caller-invoked moves (Sleep Talk) render their explicit target even on
    // failure (measured; the [still] blanking does not apply to them).
    let mut move_line = if self_target {
        if has_any_effect || capped_boost_move || called_tag.is_some() {
            format!("|move|{attacker_ident}|{move_name}|{attacker_ident}")
        } else {
            format!("|move|{attacker_ident}|{move_name}||[still]")
        }
    } else if status_fail && called_tag.is_none() {
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
        && !defender_protected
        && effectiveness > 0.0
        && choice.accuracy < 100.0
    {
        let defender_affected = tail.iter().any(|ins| {
            instruction_side(ins) == Some(defender)
                || matches!(ins, Instruction::ChangeStatus(c) if c.side_ref == defender)
        });
        let crash_only = tail.iter().all(|ins| {
            matches!(ins, Instruction::Damage(d) if d.side_ref == side)
        });
        if !defender_affected && (tail.is_empty() || (choice.crash.is_some() && crash_only)) {
            missed = true;
        }
    }
    if missed {
        move_line.push_str("|[miss]");
    }
    out.lines.push(move_line);

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
                    out.lines.push(format!(
                        "|-damage|{ident}|{condition}|[from] {move_name}"
                    ));
                    emit_faint_if_dead(sim, side, ctx, out);
                    continue;
                }
            }
            sim.apply(ins);
        }
        return;
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
                    (PokemonBoostableStat::SpecialAttack, boost.boosts.special_attack),
                    (PokemonBoostableStat::SpecialDefense, boost.boosts.special_defense),
                    (PokemonBoostableStat::Speed, boost.boosts.speed),
                    (PokemonBoostableStat::Accuracy, boost.boosts.accuracy),
                ] {
                    if amount != 0 {
                        let head = if amount > 0 { "-boost" } else { "-unboost" };
                        let code = boost_stat_code(stat);
                        out.lines
                            .push(format!("|{head}|{attacker_ident}|{code}|0"));
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
            out.lines
                .push(format!("|-supereffective|{defender_ident}"));
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
    let is_self_faint_move = matches!(
        choice.move_id,
        Choices::EXPLOSION | Choices::SELFDESTRUCT | Choices::MEMENTO
    );
    let is_transform = choice.move_id == Choices::TRANSFORM;
    if is_transform {
        out.lines
            .push(format!("|-transform|{attacker_ident}|{defender_ident}"));
    }
    for ins in tail {
        match ins {
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
                out.lines.push(format!(
                    "|-activate|{defender_ident}|Substitute|[damage]"
                ));
                defender_hits += 1;
            }
            Instruction::RemoveVolatileStatus(remove)
                if remove.side_ref == defender
                    && remove.volatile_status == PokemonVolatileStatus::SUBSTITUTE =>
            {
                sim.apply(ins);
                out.lines
                    .push(format!("|-end|{defender_ident}|Substitute"));
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
                } else if matches!(
                    choice.move_id,
                    Choices::SUBSTITUTE | Choices::BELLYDRUM | Choices::CURSE | Choices::PAINSPLIT
                ) {
                    // Genuine self-costs (the fold SHOULD count these, and
                    // the real protocol renders them bare).
                    sim.apply(ins);
                    let condition = sim.hp_condition(side);
                    out.lines
                        .push(format!("|-damage|{attacker_ident}|{condition}"));
                    note_faint!(side);
                } else {
                    // Unexplained attacker-side damage: render bare (the
                    // status-quo reading) but FLAG it — a bare line charges
                    // the window's self_hp_cost, which is wrong if the true
                    // source was opponent-inflicted.
                    out.lossy.push("unattributed_self_damage".to_string());
                    sim.apply(ins);
                    let condition = sim.hp_condition(side);
                    out.lines
                        .push(format!("|-damage|{attacker_ident}|{condition}"));
                    note_faint!(side);
                }
            }
            Instruction::Heal(heal) => {
                sim.apply(ins);
                let target_ident = ctx.active_ident(sim.state, heal.side_ref);
                let condition = sim.hp_condition(heal.side_ref);
                if heal.side_ref == side {
                    if choice.drain.is_some() && deals_damage_to_defender {
                        out.lines.push(format!(
                            "|-heal|{target_ident}|{condition}|[from] drain|[of] {defender_ident}"
                        ));
                    } else if choice.move_id == Choices::REST {
                        // Rest's heal is [silent] in the real protocol — the
                        // fold must NOT read it as a Heal side effect.
                        out.lines
                            .push(format!("|-heal|{target_ident}|{condition} slp|[silent]"));
                    } else if heal.heal_amount >= 0 {
                        out.lines
                            .push(format!("|-heal|{target_ident}|{condition}"));
                    } else {
                        // Negative heal (Struggle-class HP loss modeled as
                        // heal): render as recoil-style damage.
                        out.lines.push(format!(
                            "|-damage|{target_ident}|{condition}|[from] Recoil|[of] {defender_ident}"
                        ));
                        note_faint!(side);
                    }
                } else {
                    // Heal on the DEFENDER inside our move phase: absorb
                    // ability soak (Volt/Water Absorb).
                    if let Some(ability) = absorb {
                        out.lines.push(format!(
                            "|-heal|{target_ident}|{condition}|[from] ability: {ability}|[of] {attacker_ident}"
                        ));
                    } else {
                        out.lines
                            .push(format!("|-heal|{target_ident}|{condition}"));
                    }
                }
            }
            Instruction::ChangeStatus(change) => {
                sim.apply(ins);
                let target_ident = ctx.active_ident(sim.state, change.side_ref);
                if change.old_status == PokemonStatus::NONE {
                    if let Some(code) = status_code(change.new_status) {
                        if change.side_ref == side && choice.move_id == Choices::REST {
                            out.lines.push(format!(
                                "|-status|{target_ident}|slp|[from] move: Rest"
                            ));
                        } else {
                            out.lines.push(format!("|-status|{target_ident}|{code}"));
                        }
                    }
                }
                // Status cures (Heal Bell / Refresh / lum): |-curestatus| is
                // fold-ignored — omitted.
            }
            Instruction::Boost(boost) => {
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
                // Substitute/Protect/Leech Seed/Encore/confusion starts render
                // as |-start|/|-singleturn| in the real protocol — all
                // fold-ignored, deliberately omitted (module docs).
            }
            Instruction::ChangeType(_) | Instruction::ChangeAbility(_) | Instruction::FormeChange(_) => {
                // Transform internals / trace: single |-transform| line
                // already rendered; the rest is silent.
                sim.apply(ins);
            }
            Instruction::Switch(switch) => {
                // Drag (Whirlwind/Roar): the forced switch renders as |drag|.
                sim.apply(ins);
                let display = ctx.display(switch.side_ref, switch.next_index);
                let ident = ctx.ident(switch.side_ref, switch.next_index);
                let condition = sim.hp_condition(switch.side_ref);
                out.lines
                    .push(format!("|drag|{ident}|{display}|{condition}"));
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
        if s1_first { choice.clone() } else { Choice::default() },
        if s1_first { Choice::default() } else { choice.clone() },
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

fn side_condition_value(
    state: &State,
    side: SideReference,
    condition: PokemonSideCondition,
) -> i8 {
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
    ctx: &EventContext,
    out: &mut RenderedEvents,
    lossy_mode: bool,
) {
    match ins {
        Instruction::Damage(damage) => {
            let side = damage.side_ref;
            let cause = residual_damage_cause(sim.state, side, damage.damage_amount);
            sim.apply(ins);
            let ident = ctx.active_ident(sim.state, side);
            let condition = sim.hp_condition(side);
            out.lines
                .push(format!("|-damage|{ident}|{condition}|[from] {cause}"));
            emit_faint_if_dead(sim, side, ctx, out);
        }
        Instruction::Heal(heal) => {
            let side = heal.side_ref;
            let cause = residual_heal_cause(sim.state, side);
            sim.apply(ins);
            let ident = ctx.active_ident(sim.state, side);
            let condition = sim.hp_condition(side);
            out.lines
                .push(format!("|-heal|{ident}|{condition}|[from] {cause}"));
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
            // Yawn falling asleep at end of turn.
            sim.apply(ins);
            if change.old_status == PokemonStatus::NONE {
                if let Some(code) = status_code(change.new_status) {
                    let ident = ctx.active_ident(sim.state, change.side_ref);
                    out.lines.push(format!("|-status|{ident}|{code}"));
                }
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
        _ => {
            if lossy_mode {
                // In segmentation-failure fallback the whole list flows
                // through here; everything else is silent state-keeping.
            }
            sim.apply(ins);
        }
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

fn residual_heal_cause(state: &State, side: SideReference) -> String {
    let s = match side {
        SideReference::SideOne => &state.side_one,
        SideReference::SideTwo => &state.side_two,
    };
    if s.wish.0 > 0 {
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

fn move_choice_from_str(name: &str, state: &State, side: SideReference) -> PyResult<MoveChoice> {
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

/// Enumerate the engine's chance outcomes for a joint action and render each
/// as protocol lines (the instruction→event mapping), returning JSON:
/// `{"end_of_turn": bool, "branches": [{"percentage", "events", "turn_completed",
///   "lossy", "post", "post_state"}]}`.
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
            "post": post_state_summary(&state),
        }));
    }
    for branch in &generated {
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
        let mut obj = serde_json::json!({
            "percentage": branch.percentage,
            "events": rendered.lines,
            "turn_completed": rendered.turn_completed,
            "lossy": rendered.lossy,
            "post": post,
        });
        if let Some(post_state) = post_state {
            obj["post_state"] = serde_json::Value::String(post_state);
        }
        branches.push(obj);
    }
    let report = serde_json::json!({
        "end_of_turn": end_of_turn_triggered(&s1, &s2),
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
        }
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
            let rendered = render_branch_events(
                &mut state,
                &s1,
                &s2,
                &branch.instruction_list,
                true,
                &ctx(),
            );
            assert!(
                rendered.lossy.is_empty(),
                "branch failed to segment: {:?} / {:?}",
                rendered.lossy,
                branch.instruction_list
            );
            let text = rendered.lines.join("\n");
            assert!(text.contains("|move|p1a: Charmander|tackle|p2a: Squirtle"), "{text}");
            assert!(text.contains("|move|p2a: Squirtle|tackle|p1a: Charmander"), "{text}");
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
}
