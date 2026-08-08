//! Census of `sleeptalk_called_unidentified:none_matched:*` by SHAPE, on the only
//! path that can reach the Sleep Talk double `damage_dealt` reset.
//!
//! WHY THIS EXISTS. The guard in
//! `third_party/poke-engine-gen3-sleeptalk-damage-dealt-double-reset.patch` was
//! landed with its headline figures living only in prose. Independent review found
//! no committed measurement of the class anywhere in the repo, and found that the
//! transition-differential sweep corpus **structurally cannot observe the fix**:
//! `Side::serialize` (`third_party/poke-engine-src/src/state.rs`) does not emit
//! `damage_dealt` at all and `Side::deserialize` hardcodes
//! `damage_dealt: DamageDealt::default()`, while the corpus reaches the engine only
//! through `pokezero_search.branch_events`, which opens with
//! `parse_state` -> `State::deserialize`. Every corpus boundary therefore presents a
//! ZERO carry-over, and a zero carry-over cannot be doubled.
//!
//! So this census does two things a sweep cannot:
//!
//!   1. It measures the class on a LIVE `State` -- the in-memory tree fold's regime
//!      (`model.rs` / `tree.rs` apply and reverse instructions without re-serialising),
//!      at depth >= 2, where a ply-1 `set_damage_dealt` leaves the carry that ply 2
//!      then doubles.
//!   2. It runs the SAME population twice, once with the carry-over present and once
//!      with it zeroed, so the artifact carries its own reachability control. The
//!      `carry_over: 0` arm is exactly what a deserialized corpus boundary looks
//!      like; it must be identical between the guarded and guard-reverted engines,
//!      and that identity is what PREDICTS the null sweep result rather than merely
//!      reporting it.
//!
//! POPULATION. The builder is the #1048 attribution oracle's
//! (`events::tests::every_sleeptalk_attribution_names_the_callee_the_engine_used`)
//! with two deliberate additions: `state.use_damage_dealt = true` and a non-default
//! `side_two.damage_dealt`. Both are set DIRECTLY rather than via a Counter moveset,
//! because `set_conditional_mechanics` runs inside `State::deserialize` and not on
//! field assignment -- a probe that only added Counter to a moveset would leave the
//! flag false and measure nothing, against either engine.
//!
//! WHAT IT DOES NOT CLAIM. These are branch counts over a synthetic
//! `State::default()` stat block, not a campaign-era rate and not a world-failure
//! count. No fallback-rate reduction is derived from them.
//!
//! Usage:
//!   cargo run --release --example gen3_sleeptalk_none_matched_census -- <out.json>

use std::collections::BTreeMap;

use poke_engine::choices::Choices;
use poke_engine::engine::generate_instructions::generate_instructions_from_move_pair;
use poke_engine::engine::state::MoveChoice;
use poke_engine::state::{PokemonMoveIndex, PokemonStatus, State};

use pokezero_search::events::{render_branch_events, EventContext};

const SLOTS: [PokemonMoveIndex; 3] = [
    PokemonMoveIndex::M1,
    PokemonMoveIndex::M2,
    PokemonMoveIndex::M3,
];

/// The carry-over a ply-1 `set_damage_dealt` leaves behind. Any non-zero value
/// reproduces the defect; 137 is the value the original reproduction used.
const CARRIED_DAMAGE: i16 = 137;

/// The #1048 oracle's collision-prone movesets, unchanged: same-power/type pairs
/// that regenerate byte-identical tails and must be REFUSED rather than guessed,
/// plus status, boost, drain, self-KO and locked-move shapes.
const MOVESETS: [[Choices; 3]; 10] = [
    [Choices::BODYSLAM, Choices::EARTHQUAKE, Choices::REST],
    [Choices::BODYSLAM, Choices::EARTHQUAKE, Choices::PETALDANCE],
    [Choices::THUNDER, Choices::SURF, Choices::REST],
    [Choices::HARDEN, Choices::WITHDRAW, Choices::BODYSLAM],
    [Choices::TACKLE, Choices::SCRATCH, Choices::REST],
    [Choices::TOXIC, Choices::WILLOWISP, Choices::EARTHQUAKE],
    [Choices::GIGADRAIN, Choices::BODYSLAM, Choices::REST],
    [Choices::EXPLOSION, Choices::BODYSLAM, Choices::REST],
    [Choices::THRASH, Choices::EARTHQUAKE, Choices::REST],
    [Choices::SPLASH, Choices::BODYSLAM, Choices::EARTHQUAKE],
];

/// Counter and Mirror Coat are the two `damage_dealt` readers that can be driven
/// from the defender's slot; Substitute is the oracle's own control.
const DEFENDERS: [Choices; 4] = [
    Choices::COUNTER,
    Choices::MIRRORCOAT,
    Choices::SUBSTITUTE,
    Choices::FLAIL,
];

fn build(callees: &[Choices; 3], defender: Choices, sleeper_first: bool, carry: i16) -> State {
    let mut st = State::default();
    // The two lines that make this the in-memory fold's regime rather than a
    // deserialized corpus boundary's.
    st.use_damage_dealt = true;
    st.side_two.damage_dealt.damage = carry;

    // HIGH-DAMAGE stats, from the oracle and for its reason: at `State::default()`
    // magnitudes `floor(0.925*M) == floor(M*92/100)`, so the average-collapse tail
    // is byte-identical to roll 92 and move-order effects look unpinnable.
    st.side_two.get_active().attack = 318;
    st.side_two.get_active().special_attack = 318;
    st.side_one.get_active().attack = 200;
    st.side_one.get_active().defense = 96;
    st.side_one.get_active().special_defense = 96;
    st.side_one.get_active().maxhp = 404;
    st.side_one.get_active().hp = 404;
    st.side_two.get_active().maxhp = 404;
    st.side_two.get_active().hp = 404;
    st.side_two.get_active().speed = if sleeper_first { 500 } else { 1 };
    st.side_one.get_active().speed = if sleeper_first { 1 } else { 500 };
    st.side_two.get_active().status = PokemonStatus::SLEEP;
    st.side_two.get_active().rest_turns = 0;
    st.side_two.get_active().sleep_turns = 0;
    st.side_two
        .get_active()
        .replace_move(PokemonMoveIndex::M0, Choices::SLEEPTALK);
    for (i, slot) in SLOTS.iter().enumerate() {
        st.side_two.get_active().replace_move(*slot, callees[i]);
    }
    st.side_one
        .get_active()
        .replace_move(PokemonMoveIndex::M0, defender);
    st
}

/// One arm of the census: the whole population at a fixed carry-over.
fn census(carry: i16) -> serde_json::Value {
    let ctx = EventContext {
        species: [vec!["Lead".into()], vec!["Opponent".into()]],
        turn: 1,
        hp_percent: [false, false],
    };
    let s1 = MoveChoice::Move(PokemonMoveIndex::M0);
    let s2 = MoveChoice::Move(PokemonMoveIndex::M0);

    let mut total_branches = 0usize;
    let mut refused_branches = 0usize;
    let mut none_matched_branches = 0usize;
    let mut attributed_branches = 0usize;
    let mut slugs: BTreeMap<String, usize> = BTreeMap::new();
    let mut cells = 0usize;
    // END-STATE NEUTRALITY, recorded so it can be CHECKED across the two artifacts
    // rather than asserted in prose. `Side::serialize` omits `damage_dealt`, so this
    // projection deliberately EXCLUDES the field being repaired: it answers "does the
    // observable outcome distribution move?", not "did the bug get fixed?". Mass is
    // bucketed per observable end state and summed, because the guard may repartition
    // branches (`combine_duplicate_instructions` folds byte-identical arms) without
    // moving any probability.
    let mut mass_by_end_state: BTreeMap<String, i64> = BTreeMap::new();

    for sleeper_first in [true, false] {
        for defender in DEFENDERS {
            for callees in &MOVESETS {
                cells += 1;
                let mut state = build(callees, defender, sleeper_first, carry);
                let branches = generate_instructions_from_move_pair(&mut state, &s1, &s2, true);
                for branch in &branches {
                    total_branches += 1;
                    {
                        // Mass in micro-percent, as an INTEGER: f32 percentages differ in
                        // the last bit between two builds that repartition, and a float
                        // sum would report a difference that is not one. 1e-6 pp
                        // granularity is far finer than any real repartition.
                        let mut end = state.clone();
                        end.apply_instructions(&branch.instruction_list);
                        let key = format!("{}|{}", cells, end.serialize());
                        *mass_by_end_state.entry(key).or_insert(0) +=
                            (f64::from(branch.percentage) * 1_000_000.0).round() as i64;
                    }
                    let rendered = render_branch_events(
                        &mut state.clone(),
                        &s1,
                        &s2,
                        &branch.instruction_list,
                        true,
                        &ctx,
                    );
                    let sleeptalk_refusals: Vec<&String> = rendered
                        .attribution_unsafe
                        .iter()
                        .filter(|r| r.starts_with("sleeptalk_called_unidentified"))
                        .collect();
                    if sleeptalk_refusals.is_empty() {
                        // A branch that reached the callee walk and named its callee
                        // renders the `[from] Sleep Talk` line; one that never reached
                        // the walk renders nothing and is neither refused nor attributed.
                        if rendered
                            .lines
                            .iter()
                            .any(|l| l.contains("[from] Sleep Talk"))
                        {
                            attributed_branches += 1;
                        }
                        continue;
                    }
                    refused_branches += 1;
                    if sleeptalk_refusals
                        .iter()
                        .any(|r| r.contains(":none_matched"))
                    {
                        none_matched_branches += 1;
                    }
                    // One branch can carry several shape slugs; count each, and count
                    // the branch once above. Both figures are reported because they
                    // answer different questions and have been conflated before.
                    for slug in sleeptalk_refusals {
                        *slugs.entry(slug.clone()).or_insert(0) += 1;
                    }
                }
            }
        }
    }

    // A single digest over the whole (cell, end state) -> mass map. Two builds agree on
    // the observable outcome distribution iff these match; the map itself is far too
    // large to commit, and a digest cannot be eyeballed into agreement.
    let mut digest_input = String::new();
    for (key, mass) in &mass_by_end_state {
        digest_input.push_str(key);
        digest_input.push('\u{1}');
        digest_input.push_str(&mass.to_string());
        digest_input.push('\u{2}');
    }
    let mut hasher = <blake2::Blake2s256 as blake2::Digest>::new();
    blake2::Digest::update(&mut hasher, digest_input.as_bytes());
    let mass_digest = format!("{:x}", blake2::Digest::finalize(hasher));

    serde_json::json!({
        "carry_over": carry,
        "cells": cells,
        "total_branches": total_branches,
        "refused_branches": refused_branches,
        "none_matched_branches": none_matched_branches,
        "attributed_branches": attributed_branches,
        "refusal_slug_counts": slugs,
        "distinct_end_states": mass_by_end_state.len(),
        "mass_by_end_state_digest": mass_digest,
        "total_mass_micropercent": mass_by_end_state.values().sum::<i64>(),
    })
}

fn main() {
    let with_carry = census(CARRIED_DAMAGE);
    let without_carry = census(0);

    let shape_length_key =
        "sleeptalk_called_unidentified:none_matched:shape_length".to_string();
    let shape_length = |arm: &serde_json::Value| -> u64 {
        arm["refusal_slug_counts"]
            .get(&shape_length_key)
            .and_then(|v| v.as_u64())
            .unwrap_or(0)
    };

    let out = serde_json::json!({
        "probe": "gen3_sleeptalk_none_matched_census",
        "population": {
            "movesets": MOVESETS.len(),
            "defenders": DEFENDERS.iter().map(|d| format!("{d:?}")).collect::<Vec<_>>(),
            "move_orders": 2,
            "path": "render_branch_events on a live State (the in-memory tree fold's regime)",
        },
        "scope_note": "Branch counts over a synthetic State::default() stat block. \
NOT a campaign-era rate, NOT a world-failure count, and no fallback-rate reduction is \
derived from them. The `carry_over: 0` arm is the reachability control: it is what a \
deserialized corpus boundary presents, because Side::serialize omits damage_dealt and \
Side::deserialize hardcodes DamageDealt::default().",
        "with_carry_over": with_carry,
        "without_carry_over": without_carry,
        "shape_length_with_carry_over": shape_length(&with_carry),
        "shape_length_without_carry_over": shape_length(&without_carry),
    });

    let text = serde_json::to_string_pretty(&out).expect("census serializes");
    match std::env::args().nth(1) {
        Some(path) => std::fs::write(&path, text + "\n").expect("census artifact is writable"),
        None => println!("{text}"),
    }
}
