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
//! - "it hit and did nothing" vs. a miss (both: empty delta). Split by STATE where the
//!   state decides it — an already-statused defender (`status_fail`) or one already
//!   carrying the move's volatile (`volatile_fail`) — and the residual ambiguity is that
//!   a real miss renders identically. Which render wins is a MASS comparison, made in the
//!   code (`no_effect_hit_outweighs_miss`) rather than asserted in a comment:
//!   `P(hit, no effect) = accuracy` against `P(miss) = 1 - accuracy`, crossing at 50%.
//!   NOT covered, and measured rather than believed absent: a blocked OPPONENT-side boost
//!   is the same shape and is still labelled `|[miss]|`; 0 of 1682 gen3 randbat variants
//!   carry a sub-100%-accuracy member of that family (`scripts/c157_no_effect_hit_reach.py`);
//! - full-paralysis vs. miss was ALSO this shape and is now decided, not guessed: the
//!   engine marks both move-time immobilizers (`Instruction::MoveImmobilized`), so
//!   `|cant|..|par` is emitted from the marker and an unmarked empty tail is provably not
//!   an immobilization. The old probability-mass guess, and PR #1140's proposal to gate it,
//!   both describe a branch that no longer exists;
//! - the KO-straddle branch conflates "high roll" and "crit" at the level of
//!   BRANCH STRUCTURE — one arm carries both masses. Its damage IS now labelled
//!   `|-crit|` when it exceeds the maximum non-crit roll, since that is decidable;
//!   what remains conflated is the probability, not the tag;
//! - Sleep Talk's called move id is not in the delta — an unidentified call is
//!   attribution-unsafe rather than assigned to an invented action window.
//!
//! Lines the fold provably ignores (fold.rs `process_line`) are deliberately
//! NOT rendered: `|-singleturn|`, `|-curestatus|`, `|-ability|`,
//! `|-enditem|`, `|-mustrecharge|`, `|-start|` (except absorb signatures),
//! `|-anim|`, `|debug|`. Omissions are part of the documented contract.
//!
//! `|-fail|` WAS ON THAT LIST AND DOES NOT BELONG ON IT. The fold reads it —
//! `fold.rs`'s `process_line` sets `window.fail` on `-fail` (and
//! `transitions_fold.py` mirrors it), which reaches the encoder as its own numeric
//! column (`encoder.rs`, `columns.fail`). It is also RENDERED, on three paths
//! (`status_fail`, `side_condition_fail`, `volatile_fail`), so the list was wrong in
//! both directions at once. Corrected here rather than left standing because the
//! fail-vs-miss choice below turns on which flag the fold ends up carrying.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use poke_engine::choices::{Boost, Choice, Choices, MoveCategory, MoveTarget};
use poke_engine::engine::abilities::Abilities;
use poke_engine::engine::damage_calc::type_effectiveness_modifier;
use poke_engine::engine::generate_instructions::{
    calculate_both_damage_rolls, generate_instructions_from_move,
    generate_instructions_from_move_pair, residual_speed_order,
};
use poke_engine::engine::items::Items;
use poke_engine::engine::state::{MoveChoice, PokemonVolatileStatus, Weather};
use poke_engine::instruction::{
    BoostInstruction, ChangeStatusInstruction, DamageInstruction, ImmobilizeReason, Instruction,
    StateInstructions,
};
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
    /// Sub-cases of [`Self::lossy`] that do NOT refuse the branch. Measurement
    /// only: no consumer keys behaviour off this, which is exactly the point --
    /// a class can be counted without being refused.
    pub lossy_subcases: Vec<String>,
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

    /// Record a DIAGNOSTIC sub-case WITHOUT refusing the branch.
    ///
    /// The lossy-only sibling of [`Self::mark_attribution_unsafe_subcase`], and the
    /// distinction is the whole point. Use this when the transition is PROVEN and only
    /// its label is unknown; use the refusing one when the renderer cannot reproduce
    /// what the engine did, because then the description itself may be wrong.
    ///
    /// Same stable tag to `lossy`, so the differential's contract
    /// (`set(lossy) == {_SLEEPTALK_LOSSY_MARKER}`) is unchanged.
    fn mark_lossy_subcase(&mut self, lossy_tag: &'static str, subcase: &'static str) {
        assert!(
            subcase.starts_with(lossy_tag),
            "sub-case {subcase:?} does not belong to lossy tag {lossy_tag:?}"
        );
        self.mark_lossy(lossy_tag);
        self.lossy_subcases.push(subcase.to_string());
    }

    /// Refuse with a DIAGNOSTIC sub-case while keeping the `lossy` tag stable.
    ///
    /// The two channels have different consumers and different contracts.
    /// `attribution_unsafe` is what `reject_attribution_unsafe` joins into the
    /// error the Python seam keys `world_failure_reasons` by -- the probe's
    /// measurement channel, where splitting a class costs nothing.
    ///
    /// `lossy` is a CONTRACT. `scripts/engine_transition_differential.py` matches
    /// it exactly (`set(lossy) == {_SLEEPTALK_LOSSY_MARKER}`) to decide whether a
    /// branch is still usable, and `tests/test_matcher_tolerance_promotion.py`
    /// pins that. Splitting the lossy tag would silently change which branches
    /// the differential accepts -- and that file's bytes are pinned by the
    /// certification lifecycle, so it cannot be edited to follow along without
    /// its own attestation.
    ///
    /// So: sub-case to the measurement channel, stable tag to the contract.
    ///
    /// `lossy_tag` stays `&'static str`: it is the contract label and must be a literal.
    ///
    /// `subcase` was `&'static str` too, and the reason given was sound -- a formatted
    /// string could "mint an unbounded set of `world_failure_reasons` keys", and an
    /// unbounded key set is what makes a measurement channel useless. It is relaxed to
    /// `&str` so a sub-case can name a COMPOSITION (which effect families blocked a
    /// render), which a literal cannot express.
    ///
    /// WHAT IS AND IS NOT GUARANTEED, stated precisely because the first version of this
    /// comment claimed the wrong thing twice and review disproved both halves:
    ///
    /// * `assert_subcase_vocabulary` bounds the token ALPHABET, not the key SET. It admits
    ///   any composition of registered tokens, which is an infinite language under
    ///   repetition -- `...:boost+boost` passes it, proven by mutation.
    /// * What actually bounds cardinality is the dedup and fixed-order sort in
    ///   `unrenderable_tail_families`: each family appears at most once, in one order, so
    ///   the key set for this class is the non-empty subsets of the REACHABLE token set.
    /// * This is therefore NOT "stronger than `'static`". `'static` restricted keys to
    ///   literals present in the source: finite, greppable, reviewable. The honest
    ///   statement is that the ceiling rises from 1 key to 2^17 - 1 = 131,071, and
    ///   that the REALIZED count is small because tails are short -- the oracle corpus
    ///   yielded two (`boost`, `substitute+volatile`), though the second can no longer be
    ///   emitted now that `substitute` is deregistered.
    ///
    /// Note 17, not 18: `UNRENDERABLE_FAMILY_ORDER` has 18 entries but `unclassified` is
    /// emitted by NO classifier arm -- it is reachable only through the degradation below.
    /// `immobilizer`, the 18th, IS emitted by an arm -- one that is structurally unreachable
    /// in production -- so it counts toward the ceiling even though its realized volume is
    /// zero. Registered-and-unreachable and unregistered-and-unemittable are different
    /// states and the count follows registration, not reachability.
    ///
    /// That degradation is NOT redundant, and an earlier version of this sentence implied it
    /// was: it said the path is "unreachable while every arm's token is registered", which
    /// the heal split falsified. `heal_subcase` returns the UNREGISTERED token `"heal"` from
    /// two arms, so the degradation is the only thing standing between them and a
    /// `PanicException` in the release wheel. Those two arms are unreachable on their own
    /// terms -- non-negative `Damage` is admitted upstream and every `Heal` sign is covered
    /// -- which is a different and much narrower guarantee. Do not delete
    /// `registered_family_or_unclassified` on the strength of the old sentence. Counting the order list
    /// instead of the reachable token set overstated this 2x in an earlier version, in the
    /// one comment block whose entire purpose is precision.
    ///
    /// RECOUNT FROM THE ARRAY, never by adding to this number -- and the recount is now
    /// ENFORCED by `the_cardinality_ceiling_matches_the_array`, because being told to
    /// recount did not work. This figure has been wrong three times: it read "2^13 - 1 =
    /// 8,191, 14 entries" while the array held 13; the heal split took it to 18 without
    /// the arithmetic moving; and the correction to 2^17 was itself stale, because
    /// deregistering `heal` had already taken 18 back to 17. Each time the error was the
    /// same -- treating a COUNT as prose. A test is the only thing that has held -- and it
    /// is what carried the fourth move, `immobilizer`, from 17 entries to 18.
    ///
    /// A 131k ceiling is a real cost, accepted because a class that was 51.6% of the abort
    /// channel could not be ranked at all as one key. It is bounded, greppable via the
    /// order list, and every token maps to a named renderer gap.
    ///
    /// Note the sibling `mark_attribution_unsafe` already takes a `&str` and is already
    /// called with `&format!(...)` by the attract path, with NO vocabulary check and no
    /// paired-tag assert at all -- though that path is itself bounded at 2^5 by
    /// construction, so it is not a live unbounded hole either.
    fn mark_attribution_unsafe_subcase(&mut self, lossy_tag: &'static str, subcase: &str) {
        // The two arguments must name the SAME class, or this helper quietly
        // becomes the bug it exists to prevent: a branch whose contract tag and
        // measurement reason disagree changes which branches the differential
        // accepts, with nothing to notice. Nothing else relates them.
        //
        // A plain `assert!`, NOT `debug_assert!`: the campaign wheels are built
        // `maturin build --release` (scripts/build_search_crate_engine.sh,
        // build_search_crate_model.sh), where debug assertions compile out --
        // so a debug_assert would guard only `cargo test`, which is the one
        // place the single call site is already correct by construction. The
        // refusal path is rare, so one `starts_with` costs nothing on the
        // artifact we actually ship.
        assert!(
            subcase.starts_with(lossy_tag),
            "sub-case {subcase:?} does not belong to lossy tag {lossy_tag:?}"
        );
        assert_subcase_vocabulary(lossy_tag, subcase);
        self.mark_lossy(lossy_tag);
        self.attribution_unsafe.push(subcase.to_string());
    }

    pub fn is_attribution_unsafe(&self) -> bool {
        !self.attribution_unsafe.is_empty()
    }
}

/// Every token a sub-case slug may contain, beyond the lossy tag itself.
///
/// This bounds the token ALPHABET a sub-case may draw on. It is NOT what bounds the key
/// SET -- an earlier version of this line claimed it was, contradicting the corrected
/// analysis on `mark_attribution_unsafe_subcase` sixty lines above. Cardinality is bounded
/// by the dedup and fixed-order sort in `unrenderable_tail_families`. Two comments
/// disagreeing about one claim is a failure this file already records twice. `attract_empty_tail_ambiguous`'s own tokens are
/// listed here as well, so if that path is ever moved onto this helper it does not have
/// to be discovered first.
const SUBCASE_VOCABULARY: &[&str] = &[
    // sleeptalk
    "ambiguous",
    "ambiguous_unrenderable",
    "none_matched",
    // The `none_matched` DIVERGENCE SHAPES. era 60 measured that class at 3,595 world
    // failures with no way to say why, and the era-60 measurement states it "must be
    // classified before it can be fixed". These SEVEN are that classification.
    //
    // The two-way ownership split this comment once drew -- `values_only` unfixable here,
    // `structure`/`length` a candidate-set bug -- is RETRACTED in both directions. The
    // `NoneMatchedShape` doc records `ValuesOnly` measured 132/132 RENDERER-side in C31, so
    // it is no ownership verdict; and the containment split means a length difference can
    // indicate an over-long TAIL rather than a wrong candidate. These name PREDICATES.
    // PREFIXED. `SUBCASE_VOCABULARY` is shared across every lossy tag and
    // `assert_subcase_vocabulary` validates per token with no tag scoping, so registering the
    // bare words `structure`, `length` and `empty` would weaken the gate for unrelated
    // families -- a mis-composed attract or `ambiguous_unrenderable` slug containing "length"
    // would start passing. That gate is a PRODUCTION assert, so this reaches past tests.
    "shape_same_variants_and_sides",
    "shape_structure",
    "shape_branch_is_prefix_of_tail",
    "shape_tail_is_prefix_of_branch",
    "shape_length",
    "shape_empty",
    "shape_no_candidates",
    // attract: DEREGISTERED. `cannot_act`, `miss`, `noop`, `paralyzed` and `volatile` were
    // the five sub-case tokens of `attract_empty_tail_ambiguous`, and that whole class is
    // gone -- the engine now marks both move-time immobilizers, so an empty tail is
    // provably not an immobilization and there is nothing left to refuse. Removed by the
    // same rule that removed `heal` and `substitute`: a token belongs here only if some arm
    // can emit it, and this vocabulary's value is being a closed, greppable set.
    //
    // `volatile` is NOT lost -- it is still registered in `UNRENDERABLE_FAMILY_ORDER`, which
    // `assert_subcase_vocabulary` also accepts, and the sleeptalk family path still emits
    // it. The other four had no other producer.
    // the `heal` sub-cases. PREFIXED for the same reason the `shape_*` tokens are: this
    // vocabulary is shared across every lossy tag and `assert_subcase_vocabulary` validates
    // per token with no tag scoping, so registering bare `drain` or `defender` would weaken
    // the gate for unrelated families.
    "heal_paindmg",
    "heal_liquidooze",
    "heal_defender",
    "heal_drain_or_shellbell",
    "heal_zero_marker",
    // The SUCCESS-side counter for the Protect marker. Registered even though the
    // caller does not currently reach this gate: `mark_lossy_subcase` asserts only
    // `starts_with(lossy_tag)` and, unlike `mark_attribution_unsafe_subcase`, never
    // calls `assert_subcase_vocabulary`.
    //
    // WHAT THAT ASYMMETRY MEANS, corrected. This block used to say an unregistered token
    // "becomes a PRODUCTION panic", present tense, which overstates it in the direction that
    // makes the entry look self-enforcing. It is not: for THESE tokens the registration is
    // INERT on the production path, and review measured the consequence -- deregistering one
    // of them survived the whole suite. The conditional is what is true: registration
    // matters the moment anyone routes `mark_lossy_subcase` through the gate, which is a
    // sensible hardening since the lossy sub-case channel is otherwise unbounded.
    //
    // WHY THAT ROUTING IS NOT DONE HERE, so the next reader does not assume it was an
    // oversight: `the_paired_tag_assert_in_mark_lossy_subcase` deliberately passes
    // `attract_empty_tail_ambiguous:miss`, whose tokens this vocabulary DEREGISTERED, so
    // routing would panic an existing pin that is testing something else entirely. Closing
    // the asymmetry means dealing with that pin first.
    //
    // What makes these entries load-bearing today is `PROTECT_MARKER_COUNTERS`: the emit
    // site's own array, iterated by `the_live_subcase_slugs_are_all_in_vocabulary`, so a
    // token that reaches the emit site and not this list fails there.
    //
    // Prefixed for the same reason as the `shape_*` and `heal_*` tokens: this
    // vocabulary is shared across every lossy tag and validated per token with no
    // tag scoping, so a bare `protect` or `rendered` would weaken the gate for
    // unrelated families.
    "protect_marker_rendered",
    // The #1211 half of that counter: a Protect marker rendered on a defender that HAS a
    // zero-heal-capable absorb ability but had the HP headroom that makes the absorb no-op
    // impossible. Separate from the bare token so the two reclaims can be differenced;
    // see the emit site for why summing them would make this one unmeasurable.
    "protect_marker_rendered_absorb_headroom",
    // The third half of that counter, and the one this campaign's census block actually
    // reaches: a Protect marker rendered on a FULL-HP absorber, where the HP axis says the
    // absorb no-op WAS possible and the callee scan says no candidate could have produced it.
    // Its own token for the reason the header above gives -- `_absorb_headroom` would be a
    // false statement about this render (there is no headroom), and summing it into either
    // existing token would make the reclaim unmeasurable inside a series a reader is
    // differencing across eras.
    "protect_marker_rendered_absorb_full_hp",
    // the escape hatch both paths use when no predicate fired
    "unclassified",
];

/// Refuse a sub-case slug built from tokens nobody registered.
///
/// The `'static` bound on `subcase` used to make an unbounded key set unrepresentable.
/// With composition allowed, this does the same job explicitly and more tightly: it
/// admits a fixed vocabulary rather than any literal a caller cares to write. A
/// mis-composed slug therefore fails LOUDLY at the call site instead of quietly becoming
/// a 37th aggregate key that nobody can trace back to a code path.
///
/// SCOPE, corrected. This block used to say the assert "can no longer fire at all" on the
/// sleeptalk path because `unrenderable_tail_families` degrades an unregistered token to
/// `unclassified` first. That is true of the FAMILY path and FALSE of the SHAPE path:
/// `none_matched_slugs` composes its slug directly and reaches
/// `mark_attribution_unsafe_subcase` with NO degrade in between. The belief that this could
/// not fire is what shipped a half-applied rename whose first world would have panicked the
/// release wheel -- the assert was doing exactly the job this comment said it no longer had.
/// LIVE for: the shape path, paired-tag misuse, and any future caller that does not degrade.
///
/// A plain `assert!` for the same reason the paired-tag check above is one: the campaign
/// wheels are built `--release`, where `debug_assert!` compiles out, so a debug assert
/// would guard only `cargo test` -- the one place the call sites are already correct.
fn assert_subcase_vocabulary(lossy_tag: &str, subcase: &str) {
    let tail = &subcase[lossy_tag.len()..];
    for token in tail.split([':', '+']) {
        if token.is_empty() {
            continue;
        }
        assert!(
            SUBCASE_VOCABULARY.contains(&token)
                || UNRENDERABLE_FAMILY_ORDER.contains(&token),
            "sub-case {subcase:?} contains unregistered token {token:?}; \
             add it to SUBCASE_VOCABULARY or UNRENDERABLE_FAMILY_ORDER"
        );
    }
}

/// Canonical `world_failure_reasons` label for a refused event stream.
///
/// This is a MEASUREMENT KEY, not prose: the Python seam counts it verbatim
/// into `world_failure_reasons`, so two refusals with the same content must
/// produce the same bytes. Two rules earn that:
///
/// **Dedupe** — both sides refusing for the SAME reason is the common case, and
/// the duplicate copy is pure length with no information in it.
///
/// **Sort** — push order here is RENDER order, which is SPEED order. Without a
/// sort the identical pair `{miss, cannot_act}` keys as `...:miss,...:cannot_act`
/// or `...:cannot_act,...:miss` depending only on who moved first, splitting one
/// measurement across two buckets and halving each count. The order WITHIN a
/// slug was already fixed for exactly this reason (see the attract sub-case
/// emitter below); the order BETWEEN slugs was not, which left the bug
/// half-fixed — found by independent review on #1030.
///
/// Length is bounded at the seam rather than here, by `_bounded_reason_detail`
/// in `src/pokezero/engine_search.py`. That truncation is non-aliasing, so a
/// slug set that does overflow can never masquerade as a different one.
///
/// Split out of [`reject_attribution_unsafe`] so it is testable without a
/// Python interpreter: the label is the thing under test, the `PyErr` wrapper
/// is not.
pub fn attribution_unsafe_label(rendered: &RenderedEvents) -> String {
    let mut reasons: Vec<&str> = Vec::with_capacity(rendered.attribution_unsafe.len());
    for reason in &rendered.attribution_unsafe {
        if !reasons.contains(&reason.as_str()) {
            reasons.push(reason);
        }
    }
    reasons.sort_unstable();
    reasons.join(",")
}

/// Refuse an event stream whose action attribution is not observable from the
/// engine delta. Callers must do this before advancing a fold or encoding a
/// leaf: treating a rejected chance branch as a zero-weight branch would lose
/// probability mass, so model search lets the normal world-fallback path own
/// the whole world instead.
pub fn reject_attribution_unsafe(rendered: &RenderedEvents, lane: &str) -> PyResult<()> {
    if rendered.is_attribution_unsafe() {
        return Err(PyValueError::new_err(format!(
            "attribution-unsafe renderer branch rejected before {lane}: {}",
            attribution_unsafe_label(rendered)
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
    // GUARDED on weather_is_active, mirroring the engine's get_effective_speed.
    // AIR LOCK and CLOUD NINE suppress weather entirely, so Swift Swim must not
    // double a speed while either is on the field. Unguarded this flipped the
    // computed move order on 19000093/51 (Rayquaza AIR LOCK vs Seaking SWIFT
    // SWIM in RAIN), and since segment() tries only the order it computes, the
    // whole branch was voided as segmentation_failed. reports/c108.
    if state.weather_is_active(&state.weather.weather_type) {
        match state.weather.weather_type {
            Weather::SUN if active.ability == Abilities::CHLOROPHYLL => speed *= 2.0,
            Weather::RAIN if active.ability == Abilities::SWIFTSWIM => speed *= 2.0,
            _ => {}
        }
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
        // Added by #1105 with no arm here, which was invisible while nothing
        // emitted it: the catch-all below returns None, and an unattributable
        // instruction breaks the sleep/Sleep-Talk prelude the renderer is
        // walking. It only surfaced once the engine started banking a refund on
        // a sleepUsable attempt. Bookkeeping only -- it maps to no public line.
        Instruction::SetRestSleepPendingRefund(i) => i.side_ref,
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
        // pokezero gen3 fidelity fix (immobilization markers). The catch-all below
        // returns `None`, i.e. "this instruction belongs to no side", and a marker
        // does belong to one -- the side whose action it aborted.
        //
        // THE REAL CALLERS, named because the first version of this comment said
        // "the prelude/segment walks" and that is FALSE: `consume_move_prelude` never
        // calls this function. The two callers are the MISS-INFERENCE predicate
        // (`defender_affected`, which asks whether any tail instruction touched the
        // defender) and the `NoneMatchedShape` diagnostic (`divergence_shape`, which
        // compares per-instruction sides between a candidate branch and the tail).
        //
        // Neither can currently observe this arm: a marker is consumed and returned
        // on far above the miss inference, and it cannot appear in a Sleep Talk callee
        // tail at all. So the arm is pinned DIRECTLY, by
        // `a_marker_is_attributed_to_the_side_whose_action_it_aborted`, rather than
        // through a production path -- deleting it with only the end-to-end tests in
        // place leaves the suite green.
        Instruction::MoveImmobilized(i) => i.side_ref,
        _ => return None,
    })
}

/// How `side`'s move-time immobilization marker sits in `tail`, if the engine set one.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
enum ImmobilizationMarker {
    /// The marker is the ENTIRE remaining move phase: `[marker]`. Renderable.
    Terminal,
    /// A marker plus something else. The something else has no render arm here, so the
    /// branch is refused rather than described with that part dropped.
    NotTerminal,
}

/// Classify `side`'s move-time immobilization marker within a post-prelude move tail.
///
/// SPLIT OUT SO THE TERMINAL/NON-TERMINAL SPLIT CAN BE UNIT-TESTED. The renderer is only
/// reachable through `segment`, which prefix-matches against RE-GENERATED engine branches,
/// so a "marker plus something else" tail cannot be hand-built end to end -- the engine
/// appends the marker last and every instruction `before_move` may have pushed ahead of it
/// in gen3 (`SetFutureSight` for Future Sight, Choice Band's `DisableMove`s, the sleep and
/// freeze gates, the confusion counter) is consumed by `consume_move_prelude`. Without this
/// seam the refusal arm would have no test at all. See
/// `the_attract_marker_is_classified_terminal_only_when_it_is_the_whole_tail`.
///
/// Matches on `side_ref` rather than trusting the segment boundary. The marker is only ever
/// pushed for the acting side, but both plies share ONE instruction list and a segmentation
/// slip would otherwise credit one side's `|cant|` line to the other -- the worst available
/// failure for a line whose entire purpose is attribution.
fn move_immobilization_marker(
    tail: &[Instruction],
    side: SideReference,
) -> Option<(ImmobilizationMarker, ImmobilizeReason)> {
    let position = tail.iter().position(|instruction| {
        matches!(instruction, Instruction::MoveImmobilized(marker) if marker.side_ref == side)
    })?;
    let reason = match &tail[position] {
        Instruction::MoveImmobilized(marker) => marker.reason,
        // `position` came from the `matches!` above, so this is unreachable. Written
        // as an explicit panic rather than a silent default because a default would
        // pick a `|cant|` REASON TAG, and the tag is a different action id
        // downstream -- `public_action_capture.py` keys `cant:{reason}`.
        other => unreachable!("marker search returned a non-marker: {other:?}"),
    };
    // `tail.len() == 1` rather than `position == tail.len() - 1`: a marker that is LAST but
    // preceded by instructions is exactly the case that must refuse, so "is it last" is the
    // wrong question. Both conditions are written out because `position == 0` alone would
    // admit `[marker, boost]` and `tail.len() == 1` alone would admit a one-element tail
    // holding some other side's marker -- which `position` has already excluded, but the
    // pair is what makes that independent of the search above.
    if tail.len() == 1 && position == 0 {
        Some((ImmobilizationMarker::Terminal, reason))
    } else {
        Some((ImmobilizationMarker::NotTerminal, reason))
    }
}

/// The `|cant|` reason tag Showdown prints for each marked immobilizer.
///
/// A total match with NO catch-all, on purpose. The tag is not cosmetic:
/// `src/pokezero/public_action_capture.py` builds `event_id = f"cant:{reason}"`, so
/// `cant:Attract` and `cant:par` are DIFFERENT public action ids, and
/// `public_replay_materializer.py` gates on a closed reason set. A wrong tag is a wrong
/// action, not a cosmetic slip, so a new `ImmobilizeReason` must fail to compile here
/// rather than fall through to a plausible-looking default.
fn immobilize_cant_reason(reason: ImmobilizeReason) -> &'static str {
    match reason {
        ImmobilizeReason::Attract => "Attract",
        ImmobilizeReason::Paralysis => "par",
    }
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
        // The DEFENDER's mutated choice. Sleep Talk's identifier needs it: the
        // engine gates its 32-roll damage enumeration on the defender's move.
        &seg.second_choice,
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
        &seg.first_choice,
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
    defender_choice: &Choice,
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
            defender_choice,
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
            // pokezero row 2 (Rest skippedTime): pure bookkeeping, no public
            // line. It rides INSIDE the sleep prelude, because the engine banks
            // or discards the refund on the very attempt that emits |cant|. The
            // catch-all below is `break`, so without this arm the walk truncates
            // right after the sleep gate and drops everything after it -- the
            // confusion activation and the Sleep Talk call line both vanish.
            Instruction::SetRestSleepPendingRefund(_) => {
                sim.apply(ins);
                *cursor += 1;
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
    //
    // "EMPTY" MUST TOLERATE AN IMMOBILIZATION MARKER, or the markers fabricate a
    // protocol line here. At `chance_to_wake == 0.0` with a non-`sleepUsable` move the
    // engine pushes the real "still asleep" outcome to `final_instructions` and then
    // zeroes `incoming`'s mass WITHOUT waking the Pokemon and WITHOUT clearing
    // `reaches_confusion_handler` -- so a ZERO-MASS phantom branch flows on into the
    // Attract roll with the mon still asleep, and comes back carrying a marker. Before
    // the markers that branch was an empty delta and landed here as `|cant|..|slp`,
    // which is right; with a marker in the way this check missed and the marker arm in
    // `render_move_phase` emitted `|cant|..|Attract` instead.
    //
    // That line CANNOT EXIST. Showdown gen3's `slp.onBeforeMove` returns FALSE for a
    // non-`sleepUsable` move, which short-circuits the whole BeforeMove chain, so
    // attract's handler (priority 2) never runs on a turn the sleep gate blocked. The
    // mass is 0.0, but `branch_render_is_usable` allowlists this branch's lossy set and
    // `sample_branch_index` can still land on a zero-weight branch, so a fabricated
    // line in a searched world was reachable. Deferring to the sleep gate -- rather than
    // refusing -- restores exactly the pre-marker render.
    //
    // The engine-side zero-mass phantom is a SEPARATE pre-existing defect (it also
    // advances the confusion counter on a turn Showdown never reaches) and is left
    // alone here: gating it needs `reaches_confusion_handler` to change, which moves
    // instruction lists on branches this change has no business touching.
    let remainder_is_only_immobilization_markers = segment[*cursor..]
        .iter()
        .all(|ins| matches!(ins, Instruction::MoveImmobilized(_)));
    if pre_status == PokemonStatus::SLEEP
        && !sleep_gate_seen
        && remainder_is_only_immobilization_markers
    {
        let ident = ctx.active_ident(sim.state, side);
        out.lines.push(format!("|cant|{ident}|slp"));
        prelude.used_move = false;
        return prelude;
    }
    if pre_status == PokemonStatus::FREEZE
        && !sleep_gate_seen
        && remainder_is_only_immobilization_markers
    {
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

/// Does "it hit and did nothing" outweigh "it missed", for one empty delta?
///
/// The engine merges same-delta branches, so when a move's ENTIRE effect cannot apply to
/// the current defender its successful no-op hit and its miss are one branch. Within that
/// branch, conditioned on the move having been attempted:
///
/// ```text
/// P(hit, no effect) = accuracy
/// P(miss)           = 1 - accuracy
/// ```
///
/// They cross at 50%. Any immobilizer factor (the 0.25 paralysis roll, Attract's 1/2)
/// multiplies BOTH and cancels, so the crossover does not move with the attacker's status
/// — the immobilized branch is a separate, marked branch anyway.
///
/// WHY THE MASSES ARE COMPARED HERE RATHER THAN A THRESHOLD HARD-CODED. The
/// `|cant|..|par` guess this renderer used to carry was justified by "the paralysis
/// outcome carries the larger probability mass" and never checked it, and PR #1140 was
/// opened to gate it. The lesson that outlived that branch is that a dominance claim in a
/// comment and a dominance TEST in the code are different things, so this one is a test.
fn no_effect_hit_outweighs_miss(accuracy: f32) -> bool {
    accuracy > 100.0 - accuracy
}

/// A `(move, ...)` action phase. `called_tag` marks a caller-invoked move
/// (Sleep Talk): the prelude is skipped and the `|move|` line carries the
/// `[from]` caller attribution (fold: `called` token flag).
#[allow(clippy::too_many_arguments)]
fn render_move_phase(
    sim: &mut Sim<'_>,
    side: SideReference,
    choice: &Choice,
    // The DEFENDER's mutated choice for this ply. Only Sleep Talk's identifier
    // consumes it today, but it is threaded generically because the engine's
    // damage enumeration is a function of BOTH choices, so any future
    // regeneration needs it for the same reason.
    defender_choice: &Choice,
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

    // MOVE-TIME IMMOBILIZATION, read off the engine's marker instead of guessed.
    //
    // `Instruction::MoveImmobilized` is the gen3 attract-marker patch's whole point.
    // Both gen3 immobilizers that abort a move produce an EMPTY delta -- "the move did
    // not happen" has no state representation -- so the Attract branch and the
    // fully-paralyzed branch used to be BYTE-IDENTICAL. This renderer could not name
    // the cause: it refused the attracted case (`attract_empty_tail_ambiguous`) and
    // GUESSED the paralyzed one on a probability-mass argument. Now each branch names
    // itself and neither inference is needed.
    //
    // BOTH REASONS ARE MARKED, and marking only one would have been pointless:
    // `reject_attribution_unsafe` aborts the WHOLE WORLD rather than the branch, so an
    // unmarked paralysis sibling kept every attracted world falling back. That is the
    // review finding that shaped this arm -- two refusing branches became one and the
    // fallback rate did not move.
    //
    // POSITION IS LOAD-BEARING, three ways:
    //
    //   * BEFORE the Sleep Talk block below. Showdown resolves attract's onBeforeMove
    //     at priority 2 -- before the move is used at all -- so an immobilized Sleep
    //     Talk click emits NO `|move|...|sleeptalk|` line. It is also the trap: the
    //     sleep-talk unnamed-callee walk ends in a bare `else { sim.apply(instruction) }`
    //     and a pure marker has no state effect, so that fall-through renders NOTHING
    //     and swallows the marker in silence -- the fix would fail with the suite green.
    //     `an_attracted_sleeptalk_user_never_reaches_the_unnamed_callee_walk` is the
    //     pin; moving this block below the Sleep Talk block turns it red.
    //   * BEFORE `has_any_effect` is computed below as `!tail.is_empty()`. A marker
    //     makes the tail non-empty, so every `!has_any_effect` predicate below would
    //     flip and the branch would render as a MOVE THAT HAPPENED.
    //   * AFTER the confusion activation above. Confusion is onBeforeMove priority 3
    //     and both marked immobilizers are lower (attract 2, par 1), so its
    //     `|-activate|` precedes the `|cant|` on the same turn.
    //
    // It is also AFTER `consume_move_prelude`, which is what stops it fabricating a
    // line on a turn the sleep gate already blocked -- see the marker-tolerant
    // "remainder is empty" check there, and why that branch can exist at all.
    //
    // TERMINAL-ONLY, and it refuses otherwise. The engine appends the marker last,
    // after everything `before_move` may have pushed, and the prelude has already
    // consumed the bookkeeping it knows (PP, last-move, the sleep gate, the confusion
    // counter). So the expected tail is exactly `[marker]`. Anything ELSE left in it
    // means the prelude broke on an instruction this renderer cannot express, and
    // emitting only the `|cant|` line would silently drop it. Refusing there is a
    // change in the conservative direction: before the markers such a branch had a
    // non-empty tail, missed the immobilizer paths entirely, and rendered a `|move|`
    // line for a move that never happened.
    if called_tag.is_none() {
        if let Some((marker, reason)) = move_immobilization_marker(tail, side) {
            let attacker_ident = ctx.active_ident(sim.state, side);
            if marker == ImmobilizationMarker::Terminal {
                // Apply it for Sim bookkeeping symmetry: it is a no-op in the engine,
                // but `Sim::apply` also records the instruction for the reverse in
                // `finish`, and leaving one instruction of a consumed segment
                // unrecorded is the kind of asymmetry that only shows up once someone
                // gives the variant a state effect. KNOWN UNPINNED: deleting this call
                // leaves the suite green, because a no-op's reverse is also a no-op.
                sim.apply(&tail[0]);
                if reason == ImmobilizeReason::Attract {
                    // Telemetry-only, and UNCHANGED by this patch: the engine does not
                    // track WHO the mon is infatuated with, so Showdown's companion
                    // `|-activate|<ident>|move: Attract|[of] <source>` line stays
                    // unrenderable. The public action window (`cant:Attract`) is exact,
                    // which is why this is `mark_lossy` and not a refusal -- and why
                    // `engine_transition_differential.py` allowlists this exact tag in
                    // `_TELEMETRY_ONLY_LOSSY_MARKERS`. Do not add a NEW lossy tag on
                    // this path: that allowlist is an equality check on the tag SET and
                    // its file is byte-pinned by the certification lifecycle, so a new
                    // tag would make every marked branch unusable for matching.
                    //
                    // Paralysis gets NO tag: `|cant|<ident>|par` is the complete
                    // Showdown line, so that render is exact and not lossy at all.
                    out.mark_lossy("attract_immobilization_source_unknown");
                }
                let tag = immobilize_cant_reason(reason);
                out.lines.push(format!("|cant|{attacker_ident}|{tag}"));
            } else {
                // `mark_attribution_unsafe`, not the sub-case helper: this is a
                // DIFFERENT class from the old `attract_empty_tail_ambiguous` -- that
                // slug meant "the tail is empty and several immobilizers explain it",
                // this one means "a marker fired but the tail also carries something
                // unrenderable", i.e. an engine/renderer contract violation rather than
                // a known ambiguity.
                out.mark_attribution_unsafe("immobilization_marker_tail_not_terminal");
                for instruction in tail {
                    sim.apply(instruction);
                }
            }
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
            let called_tail: Vec<Instruction> = tail.to_vec();
            let SleepTalkProbe {
                ident,
                callee_can_convert_an_opponent_heal,
            } = identify_sleep_talk_called(
                sim.state,
                side,
                defender_choice,
                choice,
                &called_tail,
                branch_on_damage,
            );
            match ident {
                SleepTalkIdent::Matched(called_choice) => {
                    render_move_phase(
                        sim,
                        side,
                        &called_choice,
                        // Same defender for the callee's ply -- it is the same
                        // ply, one level down.
                        defender_choice,
                        &called_tail,
                        branch_on_damage,
                        ctx,
                        out,
                        Some("Sleep Talk"),
                    );
                }
                SleepTalkIdent::NoneMatched(_) | SleepTalkIdent::Ambiguous => {
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
                    // Name WHICH cause. `ambiguous` can only be fixed by the
                    // engine recording the called move; `none_matched` means the
                    // replay diverges from what the engine did and may be fixable
                    // in the renderer. Splitting them is what decides that, and
                    // this class is 48.9% of all world failures.
                    // AMBIGUOUS and NONE_MATCHED are not the same defect, and only
                    // one of them justifies throwing the world away.
                    //
                    // Ambiguous means two or more candidate callees each regenerated a
                    // branch whose instruction list equals `tail` EXACTLY. So the state
                    // transition is proven -- it is `tail`, whichever candidate is
                    // named -- and the only thing not known is the callee's NAME. The
                    // block below already renders that honestly: no `|move|` line, the
                    // damage carried on the generic tag the differential retags as
                    // move_unknown_callee. Nothing is invented, so nothing is unsafe.
                    //
                    // None-matched means NO candidate reproduced the tail: the
                    // renderer's model of the engine diverges, and a description built
                    // on that divergence may be wrong. That is unsafe and still refuses.
                    //
                    // The differential already agreed with this split and the renderer
                    // did not. `engine_transition_differential.py` lists
                    // `sleeptalk_called_unidentified` in `_TELEMETRY_ONLY_LOSSY_MARKERS`
                    // -- "its damage is real, only its attribution is unknown" -- so it
                    // accepts these branches for matching while the renderer marked them
                    // attribution-unsafe and search discarded the whole world. The
                    // comment below this one has recorded that disagreement for two
                    // eras. Era 57 priced it: this class is 49.5% of all world failures
                    // and 86.4% of the abort channel, and aborts are ~76% of fallback.
                    // ONE predicate decides refuse-vs-count, so the decision is testable
                    // without reaching an arm the engine cannot currently produce.
                    // THE STATE FACTS the tail cannot supply, read once. All are
                    // public: Showdown announces `|-singleturn|...|Protect` on use, an
                    // absorb ability announces itself when it fires, and HP is on the
                    // protocol every turn. Reading them fabricates no belief. See
                    // `protect_blocked_marker_side` for why each is needed.
                    let (
                        defender_protected,
                        defender_has_absorb_ability,
                        defender_absorb_heal_clamps_to_zero,
                        defender_absorb_zero_heal_possible,
                    ) = {
                        let d = match other_side(side) {
                            SideReference::SideOne => &sim.state.side_one,
                            SideReference::SideTwo => &sim.state.side_two,
                        };
                        let active = d.get_active_immutable();
                        let has_absorb =
                            absorb_ability_can_emit_a_zero_heal(active.ability);
                        // NARROWED from ability PRESENCE to "that ability could have
                        // produced THIS instruction". The absorb no-op only exists
                        // when the engine's own clamp took the 25% heal to zero.
                        let clamps =
                            has_absorb && absorb_heal_clamps_to_zero(active.hp, active.maxhp);
                        (
                            d.volatile_statuses
                                .contains(&PokemonVolatileStatus::PROTECT),
                            has_absorb,
                            clamps,
                            // NARROWED AGAIN, from a NECESSARY condition on the defender to
                            // the producer's OWN condition on the callee. The HP clamp answers
                            // "could a converted absorb heal have come out as zero"; it does
                            // not answer "was there a converted absorb heal at all", and on
                            // the census block the answer to the second question is no at
                            // every occurrence of this refusal.
                            //
                            // MEASURED, on the 31 decisions of
                            // `ambiguous_unrenderable:heal_zero_marker` at `--truth-sims 64`:
                            // all 31 hold PROTECT, are at FULL HP with `WATERABSORB`, and have
                            // exactly two matching callees, BOTH protect-flagged with
                            // `heal == None`. Two callee pairs account for all of them --
                            // (Ice Beam, Toxic) x19 and (Surf, Toxic) x12 -- and Ice Beam is
                            // not even an absorbed type. So the marker was the Protect-blocked
                            // branch at every one, and the guard was refusing on the
                            // defender's HP rather than on anything the producer needed.
                            //
                            // FAIL-CLOSED IS PRESERVED, and this is the load-bearing claim:
                            // #1211's `the_absorb_bypass_producer_is_real` counterexample --
                            // a protect-BYPASSING absorbed move such as `WATERSPORT`, whose
                            // converted heal survives `remove_effects_for_protect` -- comes
                            // back from the scan with `heal == Some(Heal{Opponent, 0.25})`,
                            // so it still refuses. That is not an argument: the scan reports
                            // that field set on 101 candidate evaluations in the exemplar
                            // battle alone, so the conjunct is live rather than vacuous, and
                            // `the_bypassing_callee_still_refuses` pins it.
                            clamps && callee_can_convert_an_opponent_heal,
                        )
                    };
                    if !sleeptalk_refusal_is_unsafe_with_protect(
                        &ident, &called_tail, side,
                        defender_protected, defender_absorb_zero_heal_possible,
                    ) {
                        out.mark_lossy_subcase(
                            SLEEPTALK_LOSSY_TAG,
                            sleeptalk_subcase_slug(&ident),
                        );
                    } else if matches!(ident, SleepTalkIdent::Ambiguous) {
                        // Ambiguous but the tail carries an effect the walk would DROP
                        // (a boost, status, heal, side condition...). Distinct sub-case so
                        // the cost of the remaining gap stays measurable.
                        //
                        // NAME WHICH ONE. Era 59 measured this key at 8,149 world failures
                        // -- the largest world-level refusal in the era and 51.6% of the
                        // abort channel -- as a single opaque token, so nothing said whether
                        // closing it needed `-boost`, `-status`, `-heal` or the Substitute
                        // family. The parenthetical above listed the candidates and the
                        // measurement could not rank them.
                        out.mark_attribution_unsafe_subcase(
                            SLEEPTALK_LOSSY_TAG,
                            &ambiguous_unrenderable_slug_with_protect(
                                &called_tail, side,
                                defender_protected, defender_absorb_zero_heal_possible,
                            ),
                        );
                    } else if let SleepTalkIdent::NoneMatched(shapes) = ident {
                        // ONE SLUG PER OBSERVED SHAPE. `attribution_unsafe_label` sorts and
                        // joins, so the set composes into a single deterministic key -- and
                        // `same_variants_and_sides` alone is a different diagnosis from
                        // `same_variants_and_sides + structure`, which a reduction to one shape
                        // could not express.
                        for slug in none_matched_slugs(shapes) {
                            out.mark_attribution_unsafe_subcase(SLEEPTALK_LOSSY_TAG, slug);
                        }
                    } else {
                        out.mark_attribution_unsafe_subcase(
                            SLEEPTALK_LOSSY_TAG,
                            sleeptalk_subcase_slug(&ident),
                        );
                    }
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
                    // ENUMERATED, because renderability is a property of the tail and not of
                    // a lone instruction -- see `substitute_break_side`, where the same
                    // engine variant is a narrated break in one position and a silent
                    // switch-out cleanup in another.
                    for (index, instruction) in called_tail.iter().enumerate() {
                        if boost_may_be_a_switch_out_reset(&called_tail, index) {
                            // LOAD-BEARING. Do not delete this as dead code.
                            //
                            // The first version of this comment said it was "unreachable while
                            // the classifier refuses these tails". That is FALSE, and review
                            // proved it by deleting the arm and keeping the classifier: the
                            // phantom `|-unboost|p1a: Lead|atk|2` came straight back. Refusal
                            // goes through `mark_attribution_unsafe_subcase`, which records a
                            // reason and does NOT short-circuit the walk -- the walk runs on
                            // refused branches too, which the sibling test's own docstring
                            // says out loud. So this arm is the only thing suppressing the
                            // line, not a second layer behind the classifier.
                            //
                            // KNOWN UNTESTED, same as the substitute break's arm: deleting the
                            // `emit_residuals!()` below survives the suite, because nothing can
                            // be pending at this point in any tail the corpus produces.
                            emit_residuals!();
                            sim.apply(instruction);
                        } else if let Instruction::Boost(boost) = instruction {
                            // RENDER the boost, which is what lets an ambiguous
                            // Harden/Withdraw tail be searched instead of thrown away.
                            //
                            // `ambiguous_unrenderable` was 8,149 world failures in era 59 --
                            // 51.6% of the abort channel and the largest single world-level
                            // refusal in the era. #1124 split it by blocking effect family
                            // precisely so the work could be scoped, and `boost` is the family
                            // the oracle corpus shows dominating: 10 of its 16 refused tails
                            // are a bare `[Boost]`, from identical-boost pairs like
                            // Harden/Withdraw where no candidate can be named but the
                            // transition is proven.
                            //
                            // RESIDUALS FIRST, exactly as the `Switch` arm does, and for the
                            // same reason: the walk's contract is IN ORDER. Without this a
                            // `[Damage, Boost..]` tail rendered the boosts BEFORE the damage
                            // they follow -- reproduced on Overheat/Psycho Boost (-2 spa) and
                            // Ancient Power/Silver Wind (+1 all five), both of which this
                            // change newly admits, so the misordering would have been a
                            // regression introduced by the fix. The whole suite passed with
                            // and without it, so `renders_a_damage_then_boost_tail_in_order`
                            // exists to pin it.
                            emit_residuals!();
                            sim.apply(instruction);
                            // NO `amount != 0` guard, mirroring the named path, which has
                            // none either. An earlier version had one, justified by two claims
                            // that are both FALSE: the engine never writes a 0-delta `Boost`
                            // (every construction site is zero-guarded --
                            // `gen3/generate_instructions.rs`'s `if boost_amount != 0`, plus
                            // the ability, item and Belly Drum sites), and Showdown DOES show
                            // a line for a capped boost -- `|-boost|IDENT|stat|0`, which this
                            // renderer already emits from its own `capped_boost_move` block,
                            // driven by the move rather than by a `Boost` instruction. So the
                            // guard was dead code resting on a wrong premise, and removing it
                            // also removes the `!= 0` -> `> 0` mutation that silently dropped
                            // every `|-unboost|`.
                            out.lines.push(render_boost_line(
                                ctx,
                                sim,
                                boost.side_ref,
                                boost.stat,
                                boost.amount,
                                None,
                            ));
                            // GHOST CURSE is the one suppression the named path has and this
                            // arm does not: it skips the boost lines and marks the branch
                            // `ghost_curse_engine_model` attribution-unsafe. Unreachable here
                            // only by accident -- `Choices::CURSE` carries a
                            // `volatile_status` alongside its boosts, so such a tail still
                            // holds `ApplyVolatileStatus` and is blocked by the `volatile`
                            // family. Recorded because admitting `volatile` is a plausible
                            // next step, and it would turn this into three phantom boost lines
                            // with no refusal.
                        } else if let Instruction::DamageSubstitute(dmg) = instruction {
                            // RENDER the substitute hit. This is the other half of
                            // `ambiguous_unrenderable`: #1131 closed the `boost` family, and the
                            // oracle's surviving 6 of 16 refusals are all
                            // `[DamageSubstitute, RemoveVolatileStatus]` -- a substitute BREAK,
                            // which the doc block above correctly said the obvious
                            // "-boost/-status/-heal/-sidestart" plan does not cover.
                            //
                            // Residuals first, same contract as the Boost and Switch arms.
                            emit_residuals!();
                            sim.apply(instruction);
                            let ident = ctx.active_ident(sim.state, dmg.side_ref);
                            out.lines
                                .push(format!("|-activate|{ident}|Substitute|[damage]"));
                        } else if let Some(protected_side) = protect_blocked_marker_side(
                            &called_tail, index, side,
                            defender_protected, defender_absorb_zero_heal_possible,
                        ) {
                            // RENDER the Protect activation. Era 62 measured this shape at
                            // 3,365 worlds -- 33.0% of world failures and the whole of the
                            // `heal` family, which turned out to contain no drain, no absorb,
                            // no Pain Split and no Liquid Ooze.
                            //
                            // Residuals first, same contract as the Boost, Switch and
                            // substitute arms.
                            emit_residuals!();
                            sim.apply(instruction);
                            let ident = ctx.active_ident(sim.state, protected_side);
                            out.lines.push(format!("|-activate|{ident}|Protect"));
                            // COUNT IT. A class that stops refusing must not stop being
                            // visible -- `engine_search.py`'s `lossy_subcase_renders` exists
                            // for exactly this, and its comment records the price of the
                            // alternative: "Two eras were spent unable to say what had
                            // changed in a class."
                            //
                            // Before this line, closing the `heal` family DELETED its only
                            // number. Era 62 measured the shape at 3,365 worlds solely
                            // because it aborted and landed in `world_failure_reasons`.
                            //
                            // WHAT THIS NUMBER IS, stated exactly, because the first version
                            // of this comment claimed something the plumbing does not deliver
                            // and review disproved it:
                            //
                            // * It counts BRANCH RENDERS, not worlds. The enclosing `price`
                            //   closure runs once per expanded branch seam (`tree.rs`
                            //   `expand_edge`), summed over every SEARCH INVOCATION and
                            //   decision in the shard -- invocation, not world: with
                            //   `early_stop` on, a stopped world is replayed at full budget
                            //   and `_absorb_lossy_subcases` runs on both passes over a
                            //   freshly re-expanded tree, so that world contributes twice.
                            //   `early_stop` defaults off, so this is latent, not live.
                            //   Era 62's 3,365 is a WORLD count. One world expands
                            //   many branches carrying the same Protect-blocked tail, so the
                            //   two are NOT commensurable and must not be differenced.
                            //
                            // * IT USED TO SURVIVE ONLY FOR WORLDS WHOSE SEARCH COMPLETED,
                            //   AND THAT IS NOW LARGELY RETIRED. `model.rs` accumulates
                            //   these counts and then, on any attribution-unsafe branch,
                            //   `return Err(error)` before the report is built. The Python
                            //   seam used to catch that and salvage only
                            //   `attribution_unsafe_renders`, so a world that rendered the
                            //   marker and later died at a DIFFERENT unsafe branch
                            //   contributed ZERO here, exactly like a world where the marker
                            //   never fired -- which is what made zero unreadable.
                            //
                            //   The abort-path sub-case carry closes that. The counts live in
                            //   a `LossySubcaseLedger` owned by the caller
                            //   (`crate::abort_telemetry`), which attaches it to the aborting
                            //   exception as a Python attribute, and
                            //   `EngineMctsPolicy._absorb_aborted_lossy_subcases` folds it
                            //   into the SAME `lossy_subcase_renders` map the clean path
                            //   feeds. An attribution-unsafe abort, any other mid-search
                            //   error, and a contained poke-engine panic all now carry this
                            //   slug out.
                            //
                            //   TWO GAPS REMAIN, so this is "largely" and not "entirely".
                            //   `model.rs` has six argument-validation and root-parse
                            //   `return Err`s AHEAD of the ledger's construction and those
                            //   still attach nothing -- but none of them can have rendered a
                            //   branch yet, so there is nothing of this kind to lose. The
                            //   real one: the wiring is behind the crate's `model` cargo
                            //   feature, which no CI job and no sweep builds, so the carry is
                            //   exercised by unit tests rather than by any measurement run
                            //   this repository currently makes.
                            //
                            // WHAT THE NUMBER MEANS NOW. A nonzero value is still direct
                            // positive evidence that Protect-blocked worlds are being
                            // RECLAIMED and searched rather than re-refused one branch later.
                            // The change is on the other side: on a `model`-feature build,
                            // zero is no longer the ambiguous reading it was, because a world
                            // that rendered the marker and then aborted now reports it
                            // instead of vanishing. So the old "zero is ambiguous; nonzero is
                            // not" asymmetry is retired ON THAT PATH and MUST NOT be re-cited
                            // as a live limitation -- the abort-vs-never-fired confound is
                            // what #1158 could not distinguish and what this carry exists to
                            // remove. Zero is still not a proof of absence for ordinary
                            // reasons (the shard may simply never have reached the arm), and
                            // on a build WITHOUT the `model` feature the old caveat stands
                            // unchanged.
                            //
                            // The fallback RATE still cannot supply any of this: it moves
                            // between eras 62 and 63 for nine commits' worth of reasons, so a
                            // fall in it is not evidence about #1157. Note the scope -- the
                            // MAGNITUDE is a raw volume and moves with search_sims, batch
                            // size, decisions and games per shard, and the early-stop replay
                            // factor, so it must not be read as a rate or differenced across
                            // eras. That is unaffected by the carry, and the carry ADDS to it:
                            // aborted worlds now contribute too, so a magnitude measured
                            // before this change and one measured after are not commensurable
                            // either.
                            //
                            // `mark_lossy_subcase`, NOT a new lossy tag. It pushes the SAME
                            // `SLEEPTALK_LOSSY_TAG` that the accepting path above already
                            // pushed, so `set(lossy)` is unchanged and
                            // `engine_transition_differential.py`'s
                            // `set(lossy) == {_SLEEPTALK_LOSSY_MARKER}` contract still holds
                            // -- that file's bytes are pinned by the certification
                            // lifecycle and cannot be edited to follow along. A NEW tag
                            // would silently narrow which branches the differential accepts.
                            //
                            // This is telemetry only. Nothing keys behaviour off
                            // `lossy_subcases`, so counting a render cannot refuse one.
                            //
                            // TWO TOKENS, so #1211's reclaim is separable from #1157's. The
                            // stop condition for this class is a fall in
                            // `world_failure_reasons[...:heal_zero_marker]`, and the
                            // positive-evidence counter for it was a single key that both
                            // changes feed. Summing them would make the new admission
                            // unmeasurable inside the old one -- the same "the class stopped
                            // refusing and stopped being visible" failure that
                            // `lossy_subcase_renders` exists to prevent. `_absorb` marks a
                            // render the ability axis WOULD have refused before this change;
                            // the bare token is unchanged and still means what era 62 and 63
                            // measured, so no series is redefined under a reader.
                            //
                            // THREE TOKENS NOW, for the reason the two-token note above
                            // gives, applied once more. This change admits a render on a
                            // FULL-HP absorber, which the old expression would have counted
                            // as `_absorb_headroom` -- a token whose whole meaning is that
                            // the defender HAD headroom. That would have been a false label
                            // on a live series AND would have hidden this reclaim inside
                            // #1211's, which is the "the class stopped refusing and stopped
                            // being visible" failure `lossy_subcase_renders` exists to
                            // prevent. `_absorb_full_hp` is therefore the counter the stop
                            // condition for THIS change is read from, against the fall in
                            // `world_failure_reasons[...:heal_zero_marker]`.
                            out.mark_lossy_subcase(
                                SLEEPTALK_LOSSY_TAG,
                                protect_marker_counter_slug(
                                    defender_has_absorb_ability,
                                    defender_absorb_heal_clamps_to_zero,
                                ),
                            );
                        } else if heal_is_a_direct_self_heal(&called_tail, index, side) {
                            // RENDER the direct self-heal. Bare `|-heal|{ident}|{cond}` with no
                            // `[from]` tag, which is what Showdown emits for Recover,
                            // Soft-Boiled, Moonlight, Synthesis and Morning Sun.
                            //
                            // BARE IS LOAD-BEARING, not laziness. The fold READS the tag:
                            // `[from] item: Leftovers` is an item reveal and `[from] ability:
                            // X` an ability reveal on the healed mon, whose own comment records
                            // a live capture where a misread overwrote a protocol-confirmed
                            // Pressure. Inventing a tag FABRICATES a belief, which is worse
                            // than refusing. The predicate exists precisely to admit only the
                            // shape whose correct tag is no tag.
                            // TWO MUTANTS STILL SURVIVE HERE, stated rather than implied,
                            // because an earlier version of this arm claimed a pin it did not
                            // have and review caught exactly that:
                            //
                            //   * passing `0` instead of `index` to the predicate. Survives
                            //     because in every fixture that reaches this arm the `Heal` IS
                            //     at index 0 -- a gen3 Sleep Talk callee tail for a direct
                            //     healing move contains nothing else. Killing it needs a tail
                            //     with an instruction before the heal, which no reachable gen3
                            //     callee produces.
                            //   * deleting the re-baseline below. Survives because making it
                            //     observable needs a same-side HP DECREASE later in the same
                            //     tail, and the Leech Seed fixture written for it does not
                            //     produce one -- the residual lands outside the callee tail.
                            //     The sibling test records that non-firing rather than passing
                            //     silently.
                            //
                            // Both are the same shape as the `emit_residuals!()` survivor the
                            // substitute arm documents: unreachable-today rather than untested
                            // in principle.
                            emit_residuals!();
                            sim.apply(instruction);
                            let ident = ctx.active_ident(sim.state, side);
                            let condition = sim.hp_condition(side);
                            out.lines.push(format!("|-heal|{ident}|{condition}"));
                            // RE-BASELINE, which is the half of the deferral comment that was
                            // already satisfied and the half this arm must not forget:
                            // `emit_residuals!()` compares against `before[]`, so an unbaselined
                            // increase would leave the next comparison reading a phantom
                            // decrease. The `Switch` arm does the same thing for the same
                            // reason.
                            before[side_usize(side)] = sim.active_hp(side).0;
                        } else if let Some(break_side) = substitute_break_side(&called_tail, index) {
                            // SUBSTITUTE ONLY, matched through a helper rather than by
                            // destructuring the variant here ON PURPOSE. Every other volatile is
                            // still unexpressible by this walk and stays in the `volatile`
                            // family, so admitting the whole variant would be the C52-mirror
                            // defect.
                            //
                            // Routing through `substitute_break_side` keeps this a flat arm in
                            // the chain, so a non-substitute volatile falls through to the same
                            // `else { sim.apply(instruction) }` it always did -- byte-identical
                            // behaviour, including NOT calling `emit_residuals!()` at that
                            // point. Deciding it inside the body would have made this change
                            // touch tails it has no business touching, reachable or not, and
                            // "the predicate blocks those anyway" is a reachability argument,
                            // not an invariant. Edition 2021 here, so no let-chain.
                            //
                            // `ChangeSubstituteHealth` stays `silent` -- see the classifier --
                            // because a creation or a break always carries a companion whose
                            // line is the one that matters.
                            //
                            // KNOWN UNTESTED, stated rather than hidden: mutation testing kills
                            // six of seven mutants of these two arms (dropped line, dropped
                            // -activate, break credited to the attacker, spurious `[from]` tag,
                            // lowercased keyword, admission widened to every volatile) but
                            // DELETING the `emit_residuals!()` below SURVIVES the whole suite.
                            //
                            // The first version of this note claimed it survives because "the
                            // sibling `DamageSubstitute` arm flushes first, in every tail this
                            // corpus produces". Review FALSIFIED that: it built a phaze tail
                            // reaching this arm with no `DamageSubstitute` present at all. The
                            // tail-pairing guard now makes the claim true by CONSTRUCTION rather
                            // than by assertion -- a break is admitted only when a same-side
                            // `DamageSubstitute` precedes it, so the flush has always happened.
                            // The mutant still survives because nothing can be pending between
                            // the two, which is now a property of the predicate.
                            //
                            // The call stays anyway. #1131 shipped precisely this omission for
                            // `[Damage, Boost..]`, the suite was green with AND without the fix,
                            // and the result was boosts rendered before the damage that caused
                            // them. An unreachable-today ordering guard costs one macro call;
                            // its absence cost a fidelity regression that only a hand-read of
                            // the diff caught. Making the unreachability a PINNED assertion over
                            // the corpus rather than a paragraph is the honest follow-up.
                            emit_residuals!();
                            sim.apply(instruction);
                            let ident = ctx.active_ident(sim.state, break_side);
                            out.lines.push(format!("|-end|{ident}|Substitute"));
                        } else if let Instruction::Switch(switch) = instruction {
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
                        } else if matches!(instruction, Instruction::MoveImmobilized(_)) {
                            // EXPLICIT ARM FOR AN UNREACHABLE CASE. Do not delete it
                            // as dead code, and do not delete it on the strength of
                            // "the caller handles it" either -- that is exactly the
                            // reachability argument the boost arm above records being
                            // FALSIFIED by review.
                            //
                            // Why it cannot fire today: `render_move_phase` consumes
                            // any `MoveImmobilized` for the acting side (or refuses)
                            // BEFORE the Sleep Talk block that owns this walk, and
                            // `called_tail` is a copy of that same `tail`, so a marker
                            // that reaches here would have had to survive an earlier
                            // `return`. That is a property of statement order in one
                            // function, which is the weakest kind of invariant there
                            // is: it is one cut-and-paste away from being false, the
                            // matches in this walk are NOT exhaustive (a catch-all
                            // `else` closes them), so the compiler will not object.
                            //
                            // Why it must not be the fall-through: the marker is a
                            // pure no-op, so `sim.apply` renders nothing and the walk
                            // would SWALLOW it -- a `|cant|` line silently replaced by
                            // an unattributed callee tail, with every test green.
                            // Refusing instead is right rather than rendering the
                            // `|cant|` line here: by this point the walk has already
                            // emitted `|move|...|sleeptalk|`, so the turn would read as
                            // both a move and a cant.
                            emit_residuals!();
                            sim.apply(instruction);
                            out.mark_attribution_unsafe(
                                "attract_marker_reached_unnamed_callee_walk",
                            );
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
    // `choice` here is the MUTATED choice: `before_move` has already run on it
    // inside `generate_instructions_from_move`, and `segment()` stored the
    // result. Handing it to `expected_damage_values` makes the damage-roll path
    // apply every `before_move` modifier a SECOND time -- for Thick Fat,
    // `abilities.rs` halves `base_power` in place, so the base power ends up
    // quartered and the expectations come back at roughly half their true value.
    // Rebuild a pristine Choice from the move index, which survives the
    // mutation. See reports/c102.
    // GUARDED on identity. `render_move_phase` is also called recursively for a
    // Sleep Talk callee, and a callee's Choice comes from
    // `get_sleep_talk_choices`, which clones raw move-table Choices — those
    // carry `move_index: M0` and nothing ever sets it. So rebuilding from
    // `choice.move_index` would silently return move slot 0 for every callee:
    // Sleep Talk itself (Status, so no maxima at all, and no `|-crit|` could
    // ever be emitted) or an unrelated move (wrong, too-low maxima, so the gate
    // fires on non-crits). Both failure modes were measured — one row each.
    //
    // `move_id` is the discriminator: it survives the `before_move` mutation and
    // is re-resolved by `change_move_id` for Transform, so it agrees on every
    // non-callee path and disagrees on exactly the callee path.
    let rebuilt = build_choice(sim.state, side, &MoveChoice::Move(choice.move_index));
    let expectation_choice = if rebuilt.move_id == choice.move_id {
        rebuilt
    } else {
        // A callee. Use its own Choice rather than a wrong slot.
        //
        // NOT parity, and an earlier comment here wrongly claimed it was: the
        // callee's Choice was already mutated inside
        // `identify_sleep_talk_called`'s `generate_instructions_from_move`, and
        // `expected_damage_values` applies `before_move` again — so this path
        // totals TWO applications where every other path now totals one. The
        // double-mutation defect part 1 exists to fix is still live here, scoped
        // to Sleep Talk callees.
        //
        // Reachable, though unexercised in the 200-game window (0 rows opened):
        // a Sleep-Talk-called Fire or Ice move into a Thick Fat defender gets
        // `base_power` quartered, `max_regular` roughly halved, and the gate then
        // over-fires on ordinary non-crit rolls. The dangerous direction is a
        // false MATCH — Showdown crits, a non-crit arm is stamped `|-crit|`, and
        // the boundary accepts on the wrong arm, which the divergence count
        // cannot see.
        //
        // The clean fix needs no new machinery: `identify_sleep_talk_called`
        // already holds the pristine `candidate` before cloning and mutating it,
        // so returning it alongside the match would remove this fallback
        // entirely. Follow-up, registered in `reports/c102`.
        choice.clone()
    };
    let (_regular_collapsed, _crit_collapsed, max_regular, max_crit) =
        expected_damage_values(sim.state, side, &expectation_choice, branch_on_damage);

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
    // A volatile-only move whose volatile the defender ALREADY CARRIES cannot work, exactly
    // as `status_fail` above cannot for an already-statused defender. The engine merges its
    // no-op hit with its miss, so one empty delta carries both — and below, the miss
    // inference used to label the whole thing `|[miss]|` without asking which half was
    // bigger.
    //
    // MEASURED, from the engine's own `MOVES` table rather than a hand-copied one
    // (`cargo run -p pokezero-search --example dump_move_table`) crossed with the 1682-set
    // gen3 randbat pool (`scripts/c157_no_effect_hit_reach.py`): this arm's family is FIVE
    // moves — SUPERSONIC 55, SWEETKISS 75, DISABLE 80, SWAGGER 85, LEECHSEED 90 — and the
    // pool carries exactly one of them, LEECHSEED, on 45 of 1682 variants (2.68%). At 90%
    // the split inside the merged branch is 0.90 hit-no-op against 0.10 miss, so `|[miss]|`
    // was the MINORITY render at 9:1. The other four carry 0 variants in this pool; the arm
    // covers them because the mechanism is theirs too, not because they are reachable today.
    //
    // The 45 is the OBSERVED-PLAY floor, not the reach. The renderer also runs on searched
    // worlds, and a search enumerates re-clicking Leech Seed at an already-seeded foe as a
    // legal action, so every node with a seeded foe generates this branch.
    //
    // NOT COSMETIC. `fold.rs`'s `process_line` turns `|-miss|` into `window.miss` and
    // `|-fail|` into `window.fail`, and `encoder.rs` gives each its own numeric column, so
    // the wrong label wrote false data into two features the search consumes.
    //
    // THE LINE SHAPE IS CORPUS-MEASURED, not inferred:
    // `tests/fixtures/showdown/capture/lines-battle-gen3randombattle-controlled-20260710004.log`
    // turn 7 has real Showdown emitting `|move|p1a: Jumpluff|Leech Seed||[still]` followed
    // by `|-fail|p1a: Jumpluff` — the blank-target form, failing on the USER, which is the
    // same pair `side_condition_fail` already renders and NOT the `|-fail|<defender>|<code>`
    // form `status_fail` uses.
    //
    // EXISTENCE IS CHECKED, NOT ASSUMED. #1140 shipped and then reverted a blanket
    // "a volatile is present, so a competitor exists" arm; at Protect counter 0 the
    // competing mass is ZERO and that arm invented a `|move|` line for a branch that was
    // 100% something else. So this reads the DEFENDER'S OWN volatile set: no already-present
    // volatile, no competitor, and the empty tail really is the miss.
    //
    // DOMINANCE IS CHECKED TOO, for the same reason. Below 50% accuracy the miss is the
    // larger half and `|[miss]|` stays correct.
    //
    // KNOWN UNPINNED AT THIS CALL SITE, measured rather than assumed: deleting the
    // `no_effect_hit_outweighs_miss` conjunct SURVIVES the whole crate suite (37 binaries,
    // `cargo test --no-fail-fast`, 0 failures). It cannot be pinned by a fixture, and the
    // reason is a property of the move table rather than of the tests — NO move in
    // `poke_engine::choices::MOVES`, in any generation, is an opponent-target volatile
    // Status move at or below 50% accuracy, so no `Choice` the renderer can be handed
    // reaches the false branch. The crossover itself is pinned directly instead, on both
    // sides and at the tie, by `no_effect_hit_outweighs_miss`'s unit test. Dropping the
    // conjunct is a mutation toward the SAFER behaviour and it survives; that is declared
    // here rather than counted as caught.
    //
    // The deterministic causes are excluded the way the miss inference below excludes them,
    // so the two are mutually exclusive by construction: a protected, ability-immune,
    // type-immune or absorbing defender keeps its own exact render.
    //
    // SCOPED TO ACCURACY < 100, which is where the mislabel lives. At 100% accuracy the same
    // shape is DETERMINISTIC (no miss exists), the miss inference cannot fire, and today's
    // render is a plain `|move|<attacker>|<move>|<defender>` with no fail line — a MISSING
    // line rather than a wrong one. That sibling is real and larger (ATTRACT, CONFUSERAY,
    // MEANLOOK, SPIDERWEB, TAUNT, TORMENT, YAWN, NIGHTMARE, FORESIGHT, LOCKON, INGRAIN and
    // ENCORE are all 100%), and it is left for a change that measures its `|move|`-line and
    // `fail`-column movement against the fidelity corpus instead of riding along here. Its
    // reach is in `scripts/c157_no_effect_hit_reach.py` so it is a bounded, named gap.
    let volatile_fail = !has_any_effect
        && choice.category == MoveCategory::Status
        && choice.status.is_none()
        && choice.side_condition.is_none()
        && choice.target == MoveTarget::Opponent
        && !non_ghost_curse
        && choice.accuracy < 100.0
        && !defender_protected
        && ability_immune.is_none()
        && absorb.is_none()
        && effectiveness > 0.0
        && no_effect_hit_outweighs_miss(choice.accuracy)
        && choice.volatile_status.as_ref().map_or(false, |vs| {
            vs.target == MoveTarget::Opponent && {
                let d = match defender {
                    SideReference::SideOne => &sim.state.side_one,
                    SideReference::SideTwo => &sim.state.side_two,
                };
                d.volatile_statuses.contains(&vs.volatile_status)
            }
        });
    // FOUR PREDICATES DELETED HERE, not merely unused: `volatile_empty_tail_ambiguous`,
    // `deterministic_noop`, `move_could_act` and `empty_tail_can_be_accuracy_miss`
    // existed ONLY to decide the two empty-tail immobilizer inferences below, and both
    // inferences are gone now that the engine marks its immobilizers. Their inputs
    // (`status_fail`, `status_type_immune`, `boost_has_no_effect`, `side_condition_fail`,
    // `capped_boost_move`, `defender_protected`, `absorb`, `ability_immune`) are all
    // still live and still read directly by the deterministic no-effect renders and the
    // miss inference below, which is where an empty tail is now handled with no
    // reference to the attacker's status at all.
    //
    // Left as a deletion rather than `let _ =`: a predicate kept alive for no reader is
    // how the next person concludes the inference is still there.

    // BOTH EMPTY-TAIL IMMOBILIZER INFERENCES ARE GONE, and their removal is the half
    // of the attract-marker change that actually reclaims worlds.
    //
    // What stood here:
    //
    //   * `attract_empty_tail_ambiguous` -- a REFUSAL. An attracted attacker with an
    //     empty tail could be Attract, full paralysis, a miss or a deterministic no-op,
    //     and the renderer would not pick. Because `reject_attribution_unsafe` aborts
    //     the whole WORLD rather than the branch, every one of those worlds fell back.
    //   * a `|cant|..|par|` GUESS -- rendered whenever a paralyzed attacker had an empty
    //     tail that no deterministic no-op explained, justified by full paralysis
    //     carrying more mass than a miss, with its own comment conceding "a real miss
    //     renders identically".
    //
    // Both existed only because the two immobilizers had no state representation. They
    // are now marked (`Instruction::MoveImmobilized`, consumed far above), so:
    //
    //   * an immobilized branch never reaches here -- it returned with its own exact
    //     `|cant|` line;
    //   * an empty tail that DOES reach here is provably NOT an immobilization, and the
    //     deterministic no-effect renders and the miss inference below -- which already
    //     handle exactly this shape for an unattracted, unparalyzed attacker -- are
    //     correct for it without any attract- or paralysis-specific special case.
    //
    // So the ambiguity is not being downgraded or guessed away; it stopped existing.
    // The module docs' "the residual ambiguity is para-vs-miss only" note is settled the
    // same way: the para branch is now separable, so a surviving empty tail on a
    // sub-100%-accuracy move IS the miss.
    //
    // DO NOT reintroduce an `attacker_paralyzed`/`attacker_attracted` empty-tail arm
    // here. Both reads are still available from the live state, and both are now
    // ANTI-EVIDENCE: reaching this point with either status set means that immobilizer
    // specifically did not fire.
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
    } else if (side_condition_fail || volatile_fail) && called_tag.is_none() {
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
    // protect, absorb) ruled out.
    //
    // THE IMMOBILIZERS NO LONGER COMPETE HERE, and the claim that used to stand in this
    // comment — "for a paralyzed/frozen attacker the engine merges the full-para branch
    // with the miss branch, that case never reaches here (the prelude renders |cant|
    // first)" — outlived the code that made it true, twice over. Both gen3 move-time
    // immobilizers now carry `Instruction::MoveImmobilized`, so a full-paralysis or
    // Attract branch returns far above with its own exact `|cant|` line and an empty tail
    // that DOES reach here is provably not an immobilization. The sleep and freeze gates
    // still return from `consume_move_prelude`. Nothing about the attacker's status is
    // read here, deliberately.
    //
    // WHAT DOES COMPETE is "it hit and did nothing", which the engine merges into the same
    // empty delta and which is usually the LARGER half. Three shapes of it are separated
    // before this point and are excluded below so the two decisions cannot both fire:
    // `status_fail` (already-statused defender), `volatile_fail` (already-carried
    // volatile), and a Ghost-target Curse.
    //
    // ONE SHAPE IS NOT SEPARATED, and it is disclosed rather than guessed at: an
    // OPPONENT-side boost that cannot apply — every requested stat already at floor, or
    // Clear Body / White Smoke / Hyper Cutter / Keen Eye / Substitute blocking it —
    // produces the same empty delta, and `boost_has_no_effect` above is computed for it but
    // only consumed for the SELF-target case (`capped_boost_move`). It is left alone on
    // measured reach, not on belief: the family is KINESIS 80, COTTONSPORE / METALSOUND /
    // SCREECH / SWAGGER 85, SCARYFACE 90 and STRINGSHOT 95, and
    // `scripts/c157_no_effect_hit_reach.py` measures **0 of 1682** gen3 randbat variants
    // carrying any of them. The three accuracy-droppers that ARE common (FLASH,
    // SANDATTACK, SMOKESCREEN) are all 100% accuracy, so they cannot reach this block at
    // all. Handling it needs the real protocol's THREE different failure lines
    // (`|-fail|<user>` at floor, `|-fail|<target>|unboost` + `[from] ability:` for Clear
    // Body, `|-activate|<target>|move: Substitute` behind a sub), which is a separate
    // change with its own corpus measurement — and #1140's reverted arm is the standing
    // reason not to fold three distinguishable outcomes into one guess in passing.
    let mut missed = false;
    if choice.target == MoveTarget::Opponent
        && !status_fail
        && !volatile_fail
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

    // Real fail lines. NOT fold-ignored, which an earlier version of this comment claimed:
    // `process_line` sets `window.fail` on `-fail` and the encoder gives it a column, so
    // these lines carry a feature and not only line-stream fidelity.
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
        } else if side_condition_fail || volatile_fail {
            // Same pair as the side-condition-at-cap fail, and corpus-measured for the
            // volatile case too: the blank-target `|move|` line above plus `|-fail|<USER>`.
            // NOT `|-fail|<defender>` — real Showdown fails a fizzled volatile on the
            // attacker, which the captured turn cited at the predicate reads out directly.
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
        // rendered for self-target, and now for the sub-100%-accuracy
        // already-carried-volatile case via `volatile_fail`).
        //
        // "the fold ignores |-fail|" STOOD HERE AND IS FALSE: `process_line` sets
        // `window.fail` and the encoder gives it a column. So the OPPONENT-target 100%
        // -accuracy fails that still reach this return are missing both the blank target
        // and a real feature. Named, bounded and left for its own change — see the
        // `volatile_fail` predicate's scope note.
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
    // Named distinctly from the Sleep Talk walk's own `dragged` further up: two
    // locals of the same name in one function is the trap the arm-order pin exists
    // for. Review flagged it.
    let mut dragged_this_phase = [false, false];
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
            // Hazard chip on a side that has just been dragged in. This arm must
            // precede the generic defender-damage arm below, which would otherwise
            // render it as a bare `|-damage|` with no `[from]` -- and Showdown says
            // `[from] Spikes`, so the differential filed the observation as an exact
            // ("spikes", -n) component while the engine line went untagged into the
            // roll-scaled bucket and the two never compared.
            //
            // Both other paths already get this right: the voluntary-switch arm
            // (`switched && damage.side_ref == side`) and the Sleep Talk walk. The
            // phazing path in render_move_phase was the one missed.
            //
            // gen3-only inference, same as the Sleep Talk walk states: Spikes is the
            // sole entry hazard that deals damage in this generation, and the crate is
            // built with features = ["gen3"]. A gen4+ build would need to distinguish
            // Stealth Rock and Toxic Spikes.
            //
            // Worth being precise about what this does NOT fix: the classifier at
            // engine_transition_differential.py:1761 still short-circuits to
            // limit:world_sample_drag_target on the mere presence of a `|drag|` line,
            // before any component test. That is the second half of B1 and is a
            // separate commit, measured separately, per the program's rule against
            // mixing a classifier change with a fidelity change.
            // Guarded on the DEFENDER as well as the flag. Review proved the
            // attacker-side path is dead code -- get_instructions_from_drag always
            // switches `attacking_side.get_other_side()` and returns immediately, and
            // the pivot path only toggles flags -- but narrowing costs nothing and
            // closes it by construction rather than by argument.
            Instruction::Damage(damage)
                if dragged_this_phase[side_usize(damage.side_ref)]
                    && damage.side_ref == defender =>
            {
                sim.apply(ins);
                let ident = ctx.active_ident(sim.state, damage.side_ref);
                let condition = sim.hp_condition(damage.side_ref);
                out.lines
                    .push(format!("|-damage|{ident}|{condition}|[from] Spikes"));
                emit_faint_if_dead(sim, damage.side_ref, ctx, out);
            }
            Instruction::Damage(damage) if damage.side_ref == defender => {
                sim.apply(ins);
                // Crit labeling by REACHABILITY. A damage strictly above the
                // maximum possible non-crit roll cannot have come from a
                // non-crit, whichever representative the branch carries -- which
                // covers the crit-kill arm (defender HP), the crit-survive arm
                // (average non-kill crit damage) and the 16-roll enumeration
                // alike. The old exact-value test against a single collapsed
                // value missed all three (reports/c93).
                //
                // This is only sound because `max_regular` now comes from a
                // pristine Choice; against the doubly-mutated value it fired on
                // ordinary non-crit rolls (reports/c100, reports/c102).
                if !crit_emitted
                    && !damage_lines_done
                    && max_regular.is_some()
                    && max_crit.is_some()
                    && max_crit > max_regular
                    && damage.damage_amount > max_regular.unwrap()
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
                // Record it: HP the dragged side loses after this point is entry
                // hazard chip, not move damage. Without this the chip fell through
                // to the generic Damage arm below and rendered as a bare
                // `|-damage|` with no `[from]`. reports/c117 cause B1.
                dragged_this_phase[side_usize(switch.side_ref)] = true;
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

/// Whether an unidentified-callee outcome must REFUSE the branch, or may merely be
/// counted.
///
/// Extracted as a pure function ON PURPOSE. When this decision lived inline in
/// `render_move_phase`, making `NoneMatched` lossy-only -- which would let a render the
/// engine cannot reproduce reach the fold -- was INDISTINGUISHABLE from the correct code:
/// `NoneMatched` is unreachable from any state the crate tests build, so nothing observed
/// the routing. A synthesised test of the refusing SEAM did not help either, because the
/// seam is downstream of the choice. A pure predicate is testable without reaching the arm.
fn sleeptalk_refusal_is_unsafe(
    ident: &SleepTalkIdent,
    tail: &[Instruction],
    // The ATTACKER's side, i.e. the sleeping mon that used Sleep Talk. Threaded in because
    // renderability is not a property of the tail alone: a `Heal` on the attacker with no
    // damage to the defender is a direct healing move and renderable, while the same `Heal`
    // on the DEFENDER is an absorb ability and is not. See `heal_is_a_direct_self_heal`.
    attacker: SideReference,
) -> bool {
    // FAIL-CLOSED default, as everywhere else in this chain: no Protect, assume an absorb
    // ability. That is the pre-existing behaviour exactly.
    sleeptalk_refusal_is_unsafe_with_protect(ident, tail, attacker, false, true)
}

fn sleeptalk_refusal_is_unsafe_with_protect(
    ident: &SleepTalkIdent,
    tail: &[Instruction],
    attacker: SideReference,
    defender_protected: bool,
    defender_absorb_zero_heal_possible: bool,
) -> bool {
    match ident {
        // Proven transition; unsafe only if the walk would silently drop part of it.
        SleepTalkIdent::Ambiguous => !ambiguous_tail_is_fully_renderable_with_protect(
            tail, attacker, defender_protected, defender_absorb_zero_heal_possible),
        // The renderer could not reproduce the engine's tail at all, so any description
        // built on it may be wrong. Always unsafe.
        SleepTalkIdent::NoneMatched(_) => true,
        // Handled by the naming path; never reaches the refusal decision.
        SleepTalkIdent::Matched(_) => false,
    }
}

/// Whether the UNNAMED-callee walk can express this tail completely.
///
/// This is the acceptance test for Sleep Talk ambiguity, and it is derived from what
/// the walk actually emits rather than from what the engine proved. Independent review
/// rejected the first version of that split for exactly this gap: `Ambiguous` proves the
/// engine TRANSITION -- every matching candidate regenerated this tail -- but the walk
/// below emits only HP DECREASES, drags and faints. It emits no `-boost`, `-status`,
/// `-heal`, `-sidestart` or `-start`. So a Harden/Withdraw ambiguity left the engine
/// state holding `def +1` while the rendered observation said nothing happened, and once
/// the branch stopped being refused that mismatch reached `fold.advance_in_place` and
/// `encode_leaf`. Reproduced downstream: an unrendered heal leaves the fold's
/// `hp_fraction` stale, so a LATER, unrelated move's damage records as zero -- the C52
/// impossible-component defect in mirror image.
///
/// So: an ambiguous tail is usable ONLY if it contains nothing the walk would drop.
/// ALLOWLIST, and fail-closed against unknown INSTRUCTIONS -- a new variant refuses until
/// someone decides it is renderable, which is the direction that cannot silently corrupt an
/// observation. Note the limit of that guarantee, learned the hard way with `Damage` above:
/// it is NOT fail-closed against a new USE of an already-admitted variant.
///
/// What the refused tails actually contain, measured over the oracle corpus rather than
/// assumed: 10 are `[Boost]` (identical-boost pairs like Harden/Withdraw) and 6 are
/// `[DamageSubstitute, RemoveVolatileStatus]` (a substitute break). The second group is
/// worth naming because the obvious completeness plan -- emit `-boost`/`-status`/`-heal`/
/// `-sidestart` -- does NOT cover it: a substitute break needs `-activate|...|Substitute`
/// and `-end|...|Substitute`, so 6 of the 16 need a fifth family that plan omits.
/// `DamageSubstitute` is correctly excluded here: substitute hits use that variant rather
/// than `Damage`, and the walk renders neither.
///
/// COVERAGE, stated because five of these six entries have none. The corpus exercises
/// `Damage` and the empty tail only; `Switch`, `SetLastUsedMove` and the three
/// `ChangeDamageDealt*` variants are admitted on a structural argument with no fixture.
/// Review checked all five by hand and they hold today: `Switch` renders as a complete,
/// correctly-tagged drag (probed: `|drag|` followed by `|-damage|...|[from] Spikes`), and
/// the other four carry no protocol line on any path and are read by neither the fold nor
/// the native encoder -- the v4 `last_used_move` and damage-dealt features are written only
/// from the Python world-state path. Engine-side Counter/Mirror Coat state stays correct
/// because `sim.apply` runs every instruction regardless of what is rendered. A structural
/// argument is weaker than a fixture, so anyone extending this list should add one.
///
/// `Damage` is admitted ONLY when `damage_amount >= 0`. It is a SIGNED instruction: the
/// engine's own comment at `gen3/choice_effects.rs` records that "a negative
/// `damage_amount` is the engine's existing spelling for a heal on this instruction", which
/// is how Pain Split is expressed. The walk below emits on `active_hp < before` only, so a
/// heal-direction `Damage` renders NOTHING -- and an earlier version of this predicate
/// admitted it on the strength of the comment "Damage is rendered as an HP decrease", which
/// was simply false. Review reproduced the consequence through the production render path:
/// a tail of `[Damage -130, Damage +130]` came back USABLE with the sleeper's 40 -> 170 heal
/// absent from the lines, leaving the fold's `hp_fraction` stale so a LATER, unrelated hit
/// records as zero damage. That is the C52-mirror defect this whole predicate exists to
/// prevent, admitted by the predicate itself.
///
/// The lesson generalises past this one variant: fail-closed against unknown INSTRUCTIONS
/// is not the same as fail-closed against unknown USES of a known one.
///
/// `Switch` is rendered as a drag. The remaining
/// members carry no protocol representation at all in Gen 3: they are the engine's own
/// bookkeeping for Counter/Mirror Coat damage accounting and last-move tracking, and the
/// renderer emits no line for them on ANY path, named or unnamed -- so omitting them
/// loses nothing that the named path would have shown.
fn ambiguous_tail_is_fully_renderable(tail: &[Instruction], attacker: SideReference) -> bool {
    ambiguous_tail_is_fully_renderable_with_protect(tail, attacker, false, true)
}

fn ambiguous_tail_is_fully_renderable_with_protect(
    tail: &[Instruction],
    attacker: SideReference,
    defender_protected: bool,
    defender_absorb_zero_heal_possible: bool,
) -> bool {
    // DEFINED as "nothing blocks it", so the predicate and the diagnostic below cannot
    // disagree about which instructions are renderable. They were two independent
    // matches in the first version, and this file already records what that costs: the
    // renderer and `engine_transition_differential.py` held opposite views of the
    // sleeptalk contract for two eras with nothing to notice. One list, one answer.
    unrenderable_tail_families_with_protect(
        tail, attacker, defender_protected, defender_absorb_zero_heal_possible).is_empty()
}

/// Fixed slug order, so the emitted key is stable across runs and aggregators can sum
/// it. Encounter order would make the same tail composition produce different keys
/// depending on instruction sequence -- the requirement attract's slug records too.
const UNRENDERABLE_FAMILY_ORDER: &[&str] = &[
    // `boost` is BACK, after #1131 removed it. The removal reasoned "the walk renders it, so
    // no arm can emit it" -- true for a move's own stat change, false for the switch-out reset
    // Showdown does not narrate. The narrowed classifier can emit this token again, so leaving
    // it out would push an emittable family through `registered_family_or_unclassified` and
    // bucket a KNOWN cause as `unclassified`.
    //
    // Position is unchanged from before #1131 removed it, and no slug emitted in the interval
    // could contain the BARE token.
    //
    // "So no era-over-era key moves" was the first version of that sentence, and it is
    // OVERSTATED: COMPOSITE keys do move, for any tail already refused under another family.
    // Review demonstrated `[RemoveVolatileStatus(CONFUSION, S1), Boost(S1), Switch(S1)]`
    // keying `…:volatile` before this change and `…:boost+volatile` after. The reason there is
    // no practical drift is NOT the token position -- it is that the whole family is
    // unreachable in the current randbat pool, so the volume is zero.
    "boost",
    "statrecalc",
    "status",
    "sleepcounter",
    // The `heal` SUB-CASES.
    //
    // ERA-OVER-ERA DRIFT, stated because the `boost` note above understated exactly this:
    // every key containing `heal` MOVES with this change, and unlike the `boost` case the
    // volume is NOT zero -- era 61 measured 3,533. `ambiguous_unrenderable:heal` becomes
    // `…:heal_drain_or_shellbell` and friends, so an era-over-era diff keyed on the bare
    // token will read the old class as vanished and the new ones as novel. It is one class
    // being partitioned, and the SUM is the quantity that is comparable across the boundary.
    "heal_paindmg",
    "heal_liquidooze",
    "heal_defender",
    "heal_drain_or_shellbell",
    "heal_zero_marker",
    // `heal` is deliberately ABSENT, by the SAME rule that removed `substitute` below and
    // that put `boost` back above: a token belongs here only if some classifier arm can
    // emit it. After the sub-case split every reachable shape routes to a sub-case --
    // negative Damage to `heal_paindmg`, negative Heal to `heal_liquidooze`, zero Heal to
    // `heal_zero_marker`, defender heal to `heal_defender`, attacker heal with foe damage
    // to `heal_drain_or_shellbell`, and attacker heal WITHOUT foe damage is admitted by
    // `heal_is_a_direct_self_heal` before it ever reaches the classifier.
    //
    // The two `"heal"` returns that remain in `heal_subcase` are both unreachable
    // fall-throughs. Leaving the token registered would have made
    // `the_renderable_allowlist_is_exactly_what_it_was` demand a representative for a
    // shape no input can produce. Unregistered, either fall-through degrades through
    // `registered_family_or_unclassified` to a measurable `unclassified` bucket rather
    // than panicking -- which is the whole reason that degradation exists.
    // `substitute` is deliberately ABSENT, for the same reason as `boost` and by the same
    // rule: `DamageSubstitute` was its ONLY producer and the walk now renders it, so the
    // token is dead weight in a vocabulary whose job is to be a closed, greppable set.
    // `ChangeSubstituteHealth` never produced it -- that is `silent`, deliberately, and the
    // representative below records why.
    "volatile",
    "sidecondition",
    "weather",
    "field",
    "moveslot",
    "item",
    "silent",
    // APPENDED, deliberately last before the escape hatch, so no existing composition
    // changes relative order and no era-over-era key moves. Its producer is the
    // `MoveImmobilized` arm in `unrenderable_family_at_with_protect`, which is
    // structurally unreachable in production -- registered anyway, by the same rule that
    // put `boost` back: a token belongs here if some classifier arm can emit it, and an
    // unregistered one would degrade a NAMED contract violation into `unclassified`.
    "immobilizer",
    "unclassified",
];

/// The substitute BREAK and nothing else, as an `Option` the walk's else-if chain and the
/// renderability classifier can both match on.
///
/// One predicate, two callers, deliberately: the walk RENDERS exactly what the classifier
/// ADMITS. When those were separate expressions the pair could drift, and the drift is
/// silent in the direction that matters -- admit a tail the walk cannot express and the
/// world is searched against a protocol log missing a line, which is a wrong world rather
/// than a refused one.
///
/// # Why this takes the TAIL and an INDEX rather than one instruction
///
/// Because `RemoveVolatileStatus(SUBSTITUTE)` has TWO producers in the engine and only one
/// of them is a break that Showdown narrates:
///
///   * `generate_instructions.rs` emits it directly after a same-side `DamageSubstitute`.
///     That is the real break, and Showdown's `onEnd` fires: `|-end|<ident>|Substitute`.
///   * `state.rs`'s `remove_volatile_statuses_on_switch` emits it on EVERY non-Baton-Pass
///     switch-out, including a phazing drag. Showdown clears volatiles there with
///     `this.volatiles = {}` and does NOT run `onEnd`, so it emits NOTHING.
///
/// The first version of this predicate keyed on the volatile identity alone and therefore
/// admitted the second case, making a `[RemoveVolatileStatus(SUBSTITUTE), Switch]` phaze
/// tail render a PHANTOM `|-end|` and be SEARCHED where it used to refuse. Review
/// reproduced that end to end through `render_branch_events`. An extra line is the same
/// defect class as a missing one: a wrong world instead of a refused one.
///
/// That is the mistake this file already warns about further up -- "fail-closed
/// against unknown INSTRUCTIONS is not the same as fail-closed against unknown USES of a
/// known one" -- committed against the very sentence that names it.
///
/// So the test is CONSTRUCTIONAL, not nominal: a break is a substitute removal preceded in
/// the same tail by a same-side `DamageSubstitute`, because that is how the engine builds
/// one. A switch-out removal has no such predecessor and stays in the `volatile` family.
///
/// # The one hypothetical this rule does not close by itself
///
/// A hit that does NOT break, followed later by a switch-out clear of the same still-standing
/// substitute, would pair spuriously and re-emit the phantom. That needs a single move which
/// both damages a substitute and phazes, and the ENGINE closes it, not this predicate:
/// `generate_instructions.rs` clears `choice.flags.drag` whenever the target holds a
/// Substitute and the move is not a status move, so the only damaging drag moves can never
/// reach the drag path against a sub (and both are post-gen3 regardless).
///
/// Recorded because it is a dependency on upstream behaviour that the engine itself marks
/// TODO, and it is the only thing standing between this pairing rule and a repeat of the
/// phantom-line defect. If that block ever moves, this predicate needs a same-tail "and no
/// later same-side Switch" clause too.
///
/// Reused already: `Boost` has the same multi-producer problem -- a switch-out pushes a boost
/// RESET that Showdown does not narrate -- and `boost_may_be_a_switch_out_reset` below is that
/// fix, built on this same tail-and-index shape rather than a second bespoke rule. See its doc
/// for the full producer list, which is FIVE and not two.
fn substitute_break_side(tail: &[Instruction], index: usize) -> Option<SideReference> {
    let remove = match tail.get(index)? {
        Instruction::RemoveVolatileStatus(remove)
            if remove.volatile_status == PokemonVolatileStatus::SUBSTITUTE =>
        {
            remove
        }
        _ => return None,
    };
    // A same-side `DamageSubstitute` EARLIER in this tail. Same-side matters: a double
    // battle is out of scope for gen3 singles, but the sides are still distinct objects and
    // pairing across them would be a phantom in the other direction.
    tail[..index]
        .iter()
        .any(|earlier| match earlier {
            Instruction::DamageSubstitute(dmg) => dmg.side_ref == remove.side_ref,
            _ => false,
        })
        .then_some(remove.side_ref)
}

/// Is this `Boost` possibly a switch-out RESET rather than a narrated stat change?
///
/// `Boost` has the same MULTI-producer problem the substitute break has -- five producers in
/// gen3, enumerated below -- and #1131 admitted every one of them unconditionally. The two that
/// matter for THIS predicate:
///
///   * A move's own stat change. Showdown narrates `|-boost|` / `|-unboost|`.
///   * `generate_instructions.rs`'s switch path calls `state.reset_boosts(&switching_side_ref,
///     ..)` when `!baton_passing`, in the pre-switch block beside the volatile clears and the
///     toxic reset. Showdown drops boosts inside `clearVolatile()` and narrates NOTHING. This
///     crate's own `render_switch_phase` already gets that right: it renders only the
///     `switched && boost.side_ref != side` Intimidate case and sends everything else through
///     `_ => sim.apply(ins)`, commented "Pre-switch bookkeeping (volatile clears, boost
///     resets, ...): no lines."
///
/// So the unnamed-callee walk contradicted sibling code in the same file, and a phaze tail
/// rendered a phantom `|-unboost|`.
///
/// # FIVE producers, not two -- this predicate closes ONE of the three open ones
///
/// The first version of this doc called it "the two-producer problem". Review enumerated the
/// gen3 construction sites and there are FIVE. Only the first narrates `-boost`/`-unboost`;
/// the other four do something else, and three of those four are still open (White Herb
/// self-closes). Two counts of three are easy to confuse here, so: four are mis-narrated, one
/// of those four is now fixed, and three remain.
///
///   * move's own stat change -- `-boost`/`-unboost`. Correctly rendered.
///   * switch-out reset -- no line. CLOSED by this predicate.
///   * **Haze** (`choice_effects.rs`) -- Showdown emits `|-clearallboost|`. STILL ADMITTED, and
///     reproduced end to end: a Haze/Charm ambiguity renders the Charm line and is SEARCHED,
///     so if the callee was Haze the world is silently wrong. No `clearallboost` exists
///     anywhere in this crate, so the NAMED path is wrong for Haze too.
///   * **Psych Up** (`choice_effects.rs`) -- `|-copyboost|`. STILL ADMITTED, same shape.
///   * **White Herb** (`items.rs`) -- `|-clearnegativeboost|[silent]`. Self-closing, because
///     its tail also carries `ChangeItem`, which is the `item` family.
///
/// Haze and Psych Up are PRE-EXISTING from #1131 and not widened here, and unreachable on
/// today's data (of 350 Sleep Talk sets across three cached randbat universes, zero pair it
/// with Roar, Whirlwind, Haze, Psych Up or Baton Pass). They are named rather than left
/// implied because this file's rule is that reachability is not an invariant, and an
/// enumeration that says "two" when it is five is the kind of claim the next author builds on.
///
/// # What this actually does, stated precisely because the first version overclaimed
///
/// A `Boost` with a same-side `Switch` later in the tail is classified `boost` and the walk
/// emits no line for it. Measured cost on the attribution oracle: ZERO searchable worlds --
/// the tally is unchanged at (2720, 2483, 237, 0), because no corpus tail pairs the two.
///
/// The first version of this block called that "failing closed", and argued that refusing is
/// safe where rendering nothing would need a reachability premise. **Review showed that
/// framing is wrong for one of the two consumers**, so it is corrected here rather than
/// quoted forward:
///
///   * The SEARCH consumer does refuse. `mark_attribution_unsafe_subcase` populates
///     `attribution_unsafe`, and a branch with a Sleep-Talk-shaped entry there is discarded.
///   * The TRANSITION DIFFERENTIAL does NOT. It gates usability on the `lossy` set alone, and
///     this path leaves `lossy` at exactly the bare marker, which is in the telemetry-only
///     allowlist. So the differential accepts these branches -- WHEN that marker is the only
///     one they carry, since usability is a property of the whole branch and any other lossy
///     entry drops it -- and reads their rendered events -- meaning for that consumer this ships EXACTLY the render-nothing
///     behaviour the first version declined to commit to, resting on EXACTLY the reachability
///     premise it said it would not accept. The two consumers disagreeing is documented
///     elsewhere in this file; it is not new here, but it does invalidate the old argument.
///
/// The reachability premise, for the record, since half the behaviour now depends on it: no
/// gen3 move both boosts and phazes (the four `drag: true` moves -- Circle Throw, Dragon Tail,
/// Roar, Whirlwind -- carry no `boost`), and Baton Pass is the only other `Switch` producer in
/// a callee tail and is excluded from the reset by `!baton_passing`. So a legitimate boost and
/// a same-side switch cannot co-occur, and rendering nothing is correct for the tails that
/// reach it.
///
/// What the classification still buys, given that: the family REPORTS ITS OWN SIZE. Folding
/// these tails into `None` would make them silently indistinguishable from a rendered boost,
/// and a family that reports its size is how we would learn whether the contiguous-pre-switch
/// refinement is worth writing.
fn boost_may_be_a_switch_out_reset(tail: &[Instruction], index: usize) -> bool {
    let boost = match tail.get(index) {
        Some(Instruction::Boost(boost)) => boost,
        _ => return false,
    };
    tail[index + 1..].iter().any(|later| match later {
        Instruction::Switch(switch) => switch.side_ref == boost.side_ref,
        _ => false,
    })
}

/// Is this HP increase a DIRECT healing move on the attacker -- the one heal shape the walk
/// can express -- rather than drain, an absorb ability, or Rest?
///
/// `heal` is the family for HP INCREASES, because `emit_residuals!()` is decreases-only by
/// design (its own comment: "DECREASES ONLY, deliberately. Rendering the heal direction was
/// shipped once and emitted lines for the wrong Pokemon; it needs the same per-side
/// re-baselining this walk now has, plus its own pin."). Era-60 production ranks it second
/// among the walk's gaps, at 117 records / 461 world failures.
///
/// # Why this is NOT a mirror of the boost or substitute fixes
///
/// A `-heal` line's `[from]` tag is not decoration: the FOLD READS IT. `belief.py` treats
/// `[from] item: Leftovers` as an item reveal, and `[from] ability: X` as an ability reveal on
/// the healed mon -- its comment records a live capture where misreading `[of]` pinned an
/// ability on the attacker and overwrote a protocol-confirmed Pressure. So an invented tag
/// FABRICATES a belief, which is worse than refusing. Four shapes exist and the named path
/// distinguishes all four:
///
///   * drain (Absorb, Giga Drain): `|-heal|{attacker}|{cond}|[from] drain|[of] {defender}`
///   * absorb ability (Volt/Water Absorb): `|-heal|{defender}|{cond}|[from] ability: X|[of] ..`
///   * direct healing move (Recover, Soft-Boiled, Moonlight): BARE `|-heal|{ident}|{cond}`
///   * Rest: `|-heal|{ident}|{cond} slp|[silent]`
///
/// Only the third is admitted here, and the discrimination is a property of the TAIL:
///
///   * The heal must be on the ATTACKER. A heal on the defender is an absorb ability.
///   * The tail must carry NO damage to the defender. Damage plus a heal on the attacker is
///     drain, whose line needs `[from] drain|[of] ..`.
///   * The amount must be POSITIVE. A negative `Heal` is the engine's spelling for Liquid
///     Ooze, which the named path renders as `-damage`, not `-heal`.
///
/// Rest needs no clause: its tail always carries `ChangeStatus` and `SetSleepTurns`, which are
/// the still-blocked `status` and `sleepcounter` families, so those tails refuse regardless.
/// That is a companion-blocks-it argument rather than a reachability one -- the companion is
/// emitted by construction, the way the substitute break's `DamageSubstitute` is.
///
/// This is a PARTIAL close of the family, deliberately. Every tail this does not admit is
/// reported under a `heal_*` SUB-CASE by `heal_subcase`, so the remainder stays rankable
/// instead of vanishing. The bare `heal` token is deregistered -- no reachable input
/// produces it once the sub-cases exist.
fn heal_is_a_direct_self_heal(tail: &[Instruction], index: usize, attacker: SideReference) -> bool {
    let healed = match tail.get(index) {
        // POSITIVE only: a negative `Heal` is Liquid Ooze and renders as `-damage`.
        Some(Instruction::Heal(heal)) if heal.heal_amount > 0 => heal.side_ref,
        _ => return false,
    };
    if healed != attacker {
        return false;
    }
    // No damage to the OTHER side anywhere in the tail, or this is drain.
    !tail_damages_the_foe(tail, attacker)
}

/// Does this tail damage the side that is NOT the attacker?
///
/// Factored out so `heal_is_a_direct_self_heal` (which REFUSES on it) and `heal_subcase`
/// (which CLASSIFIES on it) cannot drift apart. Two copies of this predicate would let the
/// admit-set and the diagnostic disagree about the same tail, which is the failure mode
/// where a bucket named `drain` does not contain the tails the renderer treats as drain --
/// and the ranking built on it sends the fix to the wrong place.
fn tail_damages_the_foe(tail: &[Instruction], attacker: SideReference) -> bool {
    tail.iter().any(|other| match other {
        Instruction::Damage(dmg) => dmg.side_ref != attacker && dmg.damage_amount > 0,
        Instruction::DamageSubstitute(dmg) => dmg.side_ref != attacker,
        _ => false,
    })
}

/// Can this ability produce the ZERO-HEAL absorb no-op that the Protect marker must not be
/// confused with?
///
/// NARROWER than `is_absorb_ability` on purpose. That set includes `FLASHFIRE`, which sets a
/// VOLATILE and never a heal (`gen3/abilities.rs`), so it cannot emit this shape at all --
/// including it would refuse Protect-blocked worlds for a Flash Fire defender and buy nothing.
/// The producer at `gen3/generate_instructions.rs:1367-1375` needs `health_recovered == 0`
/// from a heal-carrying absorb, which in gen3 is Water Absorb and Volt Absorb. Dry Skin
/// carries one too but appears only in gen9 data.
fn absorb_ability_can_emit_a_zero_heal(ability: Abilities) -> bool {
    matches!(
        ability,
        Abilities::WATERABSORB | Abilities::VOLTABSORB | Abilities::DRYSKIN
    )
}

/// The fraction every zero-heal-capable absorb ability restores. All three arms of
/// `absorb_ability_can_emit_a_zero_heal` set `Heal { target: Opponent, amount: 0.25 }`
/// (`gen3/abilities.rs`), so ONE constant covers the producer.
const ABSORB_HEAL_FRACTION: f32 = 0.25;

/// Would an absorb ability's heal on a defender at this HP CLAMP to zero?
///
/// This is the second half of the absorb guard, and it is what turns "this defender has an
/// ability that can emit the no-op" into "that ability could have emitted THIS instruction".
///
/// Producer 2 is not "an absorb ability fired". Read the site
/// (`gen3/generate_instructions.rs:1405-1424`): the engine computes the heal, clamps it to
/// the target's remaining headroom, and pushes the zero-amount `Heal` ONLY in the
/// `health_recovered == 0` else-branch. A defender below full HP takes a REAL heal with a
/// nonzero amount, which is a different instruction that `heal_subcase` routes to
/// `heal_defender` and which this predicate never sees. So a zero `Heal` on a below-full-HP
/// defender cannot be producer 2 at all, whatever ability that defender has.
///
/// The arithmetic is copied from the engine rather than simplified to `hp == maxhp`,
/// deliberately. They agree for every gen3 randbat Pokemon, but they part company when
/// `(0.25 * maxhp) as i16` truncates to 0 -- and a simplification that is only true for the
/// current pool is the kind of premise this file has been burned by twice. Mirroring the
/// producer keeps the two in step by construction.
fn absorb_heal_clamps_to_zero(hp: i16, maxhp: i16) -> bool {
    let mut health_recovered = (ABSORB_HEAL_FRACTION * maxhp as f32) as i16;
    let final_health = hp + health_recovered;
    if final_health > maxhp {
        health_recovered -= final_health - maxhp;
    } else if final_health < 0 {
        health_recovered -= final_health;
    }
    health_recovered == 0
}

/// Are the defender state facts, read from the state BEFORE the tail was applied, still
/// true at `index`?
///
/// The three facts (`PROTECT` held, absorb ability, HP) are read ONCE at the top of the
/// unnamed-callee block, because the CLASSIFIER
/// (`ambiguous_tail_is_fully_renderable_with_protect`) runs before any of the tail has been
/// applied and the WALK must reach the same verdict for the same tail -- one list, one
/// answer, the rule this file states for `unrenderable_tail_families_with_protect`.
///
/// That makes a pre-tail read load-bearing at a LATER index, and the direction of the error
/// is the silent one: a tail that heals the defender to full before the marker would make
/// producer 2 possible at the marker while the pre-tail HP said it was not, and the walk
/// would emit `|-activate|...|Protect` over an ability activation. Rendering a wrong line
/// is worse than refusing, so the prefix is checked and anything that could move the
/// defender's active Pokemon, its HP or its volatiles refuses.
///
/// ALLOWLIST with a fail-closed default, the same discipline as
/// `unrenderable_family_at_with_protect`. Note the set is small for a reason: the enclosing
/// predicate only ever admits a tail EVERY index of which is renderable, so the prefix can
/// only be drawn from that allowlist in the first place.
fn defender_facts_survive_tail_prefix(
    tail: &[Instruction],
    index: usize,
    defender: SideReference,
) -> bool {
    tail.iter().take(index).all(|earlier| match earlier {
        Instruction::Damage(damage) => damage.side_ref != defender,
        Instruction::Heal(heal) => heal.side_ref != defender,
        Instruction::Switch(switch) => switch.side_ref != defender,
        // Stat stages and the engine's damage-dealt bookkeeping move neither the active
        // Pokemon, its HP nor its volatiles. `DamageSubstitute` moves the SUBSTITUTE's
        // health, which is a separate field from the Pokemon's `hp`.
        Instruction::Boost(_)
        | Instruction::DamageSubstitute(_)
        | Instruction::SetLastUsedMove(_)
        | Instruction::ChangeDamageDealtDamage(_)
        | Instruction::ChangeDamageDealtMoveCatagory(_)
        | Instruction::ToggleDamageDealtHitSubstitute(_) => true,
        // Anything else -- a volatile change that could clear PROTECT, or a future
        // variant nobody has audited -- refuses.
        _ => false,
    })
}

/// Is this instruction a PROTECT-BLOCKED branch marker that the walk can render?
///
/// gen3 emits a zero-amount `Heal` from exactly TWO sites, and BOTH push on the DEFENDER,
/// so the side does not discriminate:
///
///   * `gen3/generate_instructions.rs:3436-3444` -- the Protect-blocked branch, gated on
///     `blocked_by_protect` and pushed on `attacking_side.get_other_side()`. Its own comment:
///     "Mark only the successful accuracy branch so protocol rendering can emit Protect
///     instead of collapsing both outcomes." Showdown's line is `|-activate|<target>|Protect`.
///   * `gen3/generate_instructions.rs:1367-1375` -- a full-HP absorb activation, kept as "a
///     reversible no-op so event consumers can keep the public histories distinct". Its line
///     is an ABILITY activation, not Protect.
///
/// Rendering Protect for the second would FABRICATE a line, which is the defect that hit the
/// substitute-break arm and the boost arm before it. So the discriminator is a STATE fact the
/// tail cannot supply: the defender holds `PokemonVolatileStatus::PROTECT`. That is public --
/// Showdown announces `|-singleturn|...|Protect` when Protect is used -- so reading it
/// fabricates no belief.
///
/// FAIL-CLOSED on both axes, and the absorb axis is LOAD-BEARING rather than paranoia.
///
/// Without the volatile: None, tail keeps refusing, pre-existing behaviour.
///
/// With an absorb ability on the defender: NARROWED, and the narrowing is the whole of
/// #1211. An earlier version of this doc said "Protect strips the move before an absorb
/// could fire, so the combination should be unreachable" -- FALSE. The correction was to
/// refuse on ability PRESENCE, which is also wrong, in the other direction: it refused every
/// Water Absorb or Volt Absorb mon that uses Protect, which is routine. The capture at
/// `fb3m21-946004` round 45 is exactly that shape -- Registeel's Sleep Talk into a
/// PROTECTING Mantine at 192/252 -- and the ability there could not have produced the
/// instruction being refused.
///
/// What producer 2 actually requires, read off its site rather than inferred from the
/// ability list, is `health_recovered == 0`: an absorb heal on a defender with HEADROOM is a
/// REAL heal with a nonzero amount, a different instruction entirely. So the guard is on
/// `absorb_heal_clamps_to_zero`, not on the ability alone, and a below-full-HP absorber
/// stops costing a world.
///
/// WHAT STILL REFUSES, and why the axis cannot simply be deleted. A FULL-HP absorber that
/// holds PROTECT is ambiguous WHEN SOME CALLEE COULD HAVE CONVERTED A HEAL, and only then.
/// Both producers can reach that state: producer 1 through any protect-flagged callee, and
/// producer 2 through a callee that BYPASSES Protect, because `ability_modify_attack_against`
/// runs BEFORE the Protect gate in `before_move` and deliberately RESTORES `flags.protect` --
/// so an unflagged move keeps its converted heal where a flagged one has it stripped by
/// `remove_effects_for_protect`. That is not hypothetical: WATERSPORT is Water-typed,
/// `target: Opponent` and carries no protect flag, so Water Absorb converts it and Protect
/// does not strip it. `the_absorb_bypass_producer_is_real` pins that counterexample against
/// the engine's own move table, and
/// `protect_plus_a_bypassing_absorbed_callee_refuses_rather_than_guessing` now drives it
/// through the production render path, so a future reader cannot retire this axis on prose.
///
/// NARROWED AGAIN, and the caller is where it happened. This function still refuses whenever
/// `defender_absorb_zero_heal_possible`, unchanged; what changed is what the production read
/// puts in that argument. The HP clamp is a NECESSARY condition on producer 2, not a
/// sufficient one, and the sufficient one -- the callee's own post-modification
/// `heal == Some(Heal{Opponent, > 0})`, see `choice_can_convert_an_opponent_heal` -- is now
/// ANDed in at `render_move_phase`'s read site. On the census block that is the difference
/// between refusing 31 decisions and refusing none of them: all 31 hold PROTECT at full HP
/// with `WATERABSORB` and have two matching callees, both protect-flagged with `heal == None`,
/// so producer 2 could not have written the instruction being refused.
///
/// EQUIVALENT-MUTANT NOTE. Hardcoding `defender_protected = true` at the PRODUCTION READ
/// SITE survives the crate suite -- the unit tests below pin the parameter, not the read --
/// and it is now a genuine fail-open mutant rather than the overlapping one it used to be:
/// before #1211 the absorb axis refused every tail the PROTECT axis would, and it no longer
/// does. `tests/test_crate_protect_marker_state_reads.py` is where that read is pinned.
fn protect_blocked_marker_side(
    tail: &[Instruction],
    index: usize,
    attacker: SideReference,
    defender_protected: bool,
    defender_absorb_zero_heal_possible: bool,
) -> Option<SideReference> {
    if !defender_protected || defender_absorb_zero_heal_possible {
        return None;
    }
    match tail.get(index) {
        Some(Instruction::Heal(heal)) if heal.heal_amount == 0 => {
            let defender = other_side(attacker);
            if heal.side_ref != defender {
                return None;
            }
            // The three facts were read before ANY of the tail was applied. Refuse rather
            // than render on a fact the prefix could have invalidated.
            defender_facts_survive_tail_prefix(tail, index, defender).then_some(defender)
        }
        _ => None,
    }
}

/// Which SUB-CASE of the `heal` family a REFUSED HP-increase belongs to.
///
/// `heal` is the second-largest `ambiguous_unrenderable` family (era 61 final, 64/64 shards:
/// 3,533 world failures, 24.6% of all world-failure classes) and it survived the partial close
/// in `heal_is_a_direct_self_heal`. That close admitted exactly one shape -- a positive
/// heal on the attacker with no foe damage -- and the ranking cannot say which of the
/// remaining shapes a refusal is without this split. There are FIVE buckets below against
/// the FOUR shapes `heal_is_a_direct_self_heal`'s doc block enumerates: that block predates
/// `heal_paindmg` and `heal_zero_marker`. Rest -- its fourth shape -- has no bucket of its
/// own because its `Heal` is ADMITTED by `heal_is_a_direct_self_heal`; its companion
/// `status`/`sleepcounter` instructions are separately refused by their own arms, which is
/// what makes that admission safe. There is no short-circuit: `unrenderable_tail_families`
/// visits every index.
///
/// This is DIAGNOSTIC ONLY. Every token returned is still a blocking family, so the set
/// of refused tails is byte-identical to before. Nothing here changes what is searched.
///
/// The `drain_or_shellbell` name is deliberately honest about an ambiguity the tail cannot
/// resolve. Both a drain move and a SHELLBELL holder produce `[foe damage, attacker heal]`,
/// and poke-engine models both in gen3 (`src/gen3/items.rs` SHELLBELL). Showdown renders
/// them differently -- `[from] drain|[of] <foe>` versus `[from] item: Shell Bell` -- and the
/// second is an ITEM REVEAL. Guessing between them would FABRICATE a belief, which the
/// render arm's own comment states is worse than refusing. So this bucket is named for what
/// is actually known, and disambiguating it needs the candidate move set or the item, not
/// the tail.
fn heal_subcase(tail: &[Instruction], index: usize, attacker: SideReference) -> &'static str {
    match tail.get(index) {
        // The engine spells an HP INCREASE as a NEGATIVE `Damage` (Pain Split). A distinct
        // producer from `Heal`, and one whose protocol line differs, so it is worth its own
        // bucket even though the walk drops the same thing for both.
        Some(Instruction::Damage(dmg)) if dmg.damage_amount < 0 => "heal_paindmg",
        // A NEGATIVE `Heal` is Liquid Ooze; the named path renders it as `-damage`.
        Some(Instruction::Heal(heal)) if heal.heal_amount < 0 => "heal_liquidooze",
        Some(Instruction::Heal(heal)) if heal.heal_amount > 0 => {
            if heal.side_ref != attacker {
                // A heal on the DEFENDER inside our move phase is an absorb ability
                // (Volt Absorb, Water Absorb), whose line is an ABILITY reveal.
                "heal_defender"
            } else if tail_damages_the_foe(tail, attacker) {
                "heal_drain_or_shellbell"
            } else {
                // UNREACHABLE from the classifier: a positive attacker heal with no foe
                // damage is exactly what `heal_is_a_direct_self_heal` admits, so it
                // returns `None` before reaching here. Kept as the honest remainder
                // rather than an `unreachable!()`, because a panic in this crate is
                // mapped by pyo3 to `PanicException` -- past `except Exception` -- and
                // kills the campaign worker instead of producing a measurable key.
                "heal"
            }
        }
        // A ZERO-amount `Heal` is not a heal at all -- it is a BRANCH MARKER, and gen3
        // emits it from two places on purpose:
        //   * `gen3/generate_instructions.rs:3369` -- a PROTECT-blocked branch. Its own
        //     comment: "Mark only the successful accuracy branch so protocol rendering
        //     can emit Protect instead of collapsing both outcomes." Pushed on
        //     `attacking_side.get_other_side()`, i.e. the DEFENDER.
        //   * `gen3/generate_instructions.rs:1373` -- a full-HP absorb activation kept as
        //     "a reversible no-op so event consumers can keep the public histories
        //     distinct".
        // The line the walk owes for the first is `|-activate|<target>|Protect`, NOT a
        // `-heal`. Leaving these in the bare `heal` bucket would break this classifier's
        // own rule -- grouped by PROTOCOL LINE, not by engine instruction kind -- and
        // would mis-scope the very fix this split exists to aim. Review caught the first
        // version of this arm doing exactly that, with a test pinning the mislabel.
        Some(Instruction::Heal(heal)) if heal.heal_amount == 0 => "heal_zero_marker",
        // Any future producer. NOT silent, deliberately -- but note this no longer keeps
        // the remainder RANKABLE the way the bare `heal` bucket once did: the token is
        // deregistered, so `registered_family_or_unclassified` maps it to `unclassified`.
        // That is the intended outcome for a shape nothing can currently produce (both
        // `"heal"` returns here are unreachable), and it is measurable rather than a panic.
        _ => "heal",
    }
}

/// Which effect FAMILY, if any, the unnamed-callee walk cannot express for this
/// instruction. `None` means the walk renders it (or correctly renders nothing).
///
/// EXHAUSTIVE on purpose -- no `_` arm. A new engine variant becomes a compile error
/// here instead of silently classifying as one more `unclassified`, which on the largest
/// failure class in the program would be a mis-diagnosis rather than a crash. This is
/// the same reasoning `sleeptalk_subcase_slug` states for its own exhaustive match.
/// FAIL-CLOSED wrapper. Keeps the three-argument shape for the TESTS -- there are no
/// production callers left, they all moved to the `_with_protect` form -- and passes
/// `defender_protected: false` / `defender_absorb_zero_heal_possible: true` -- the
/// combination that renders NOTHING and preserves the pre-existing refusal exactly. Only the
/// production walk, which can read the live state, calls the `_with_protect` form.
fn unrenderable_family_at(
    tail: &[Instruction],
    index: usize,
    attacker: SideReference,
) -> Option<&'static str> {
    unrenderable_family_at_with_protect(tail, index, attacker, false, true)
}

fn unrenderable_family_at_with_protect(
    tail: &[Instruction],
    index: usize,
    attacker: SideReference,
    defender_protected: bool,
    defender_absorb_zero_heal_possible: bool,
) -> Option<&'static str> {
    // RENDERED NOW when the state says Protect. Checked before the match so the zero-Heal
    // reaches `heal_subcase` only when it is NOT a renderable Protect marker -- otherwise the
    // classifier would report a blocking family for a tail the walk successfully renders, and
    // `ambiguous_tail_is_fully_renderable` would keep refusing it.
    if protect_blocked_marker_side(tail, index, attacker, defender_protected,
                                   defender_absorb_zero_heal_possible).is_some() {
        return None;
    }
    // `get`, not `tail[index]`. Every caller is in bounds today, but an out-of-bounds index
    // would PANIC, and this file spends a long comment below on why that specific outcome is
    // the worst one available: pyo3 maps a Rust panic to `PanicException`, which derives from
    // `BaseException` so it propagates past `engine_search.py`'s `except Exception`, killing
    // the campaign worker instead of producing a measurable key. `None` here means "nothing
    // blocks", which is the wrong answer to give for a nonexistent instruction -- so it is
    // deliberately paired with the sibling predicate's `tail.get(index)?`, and both are
    // unreachable rather than merely harmless.
    let instruction = tail.get(index)?;
    match instruction {
        // --- rendered, or provably nothing to render -------------------------------
        //
        // This arm set must stay EXACTLY the previous allowlist. Widening it here does
        // not just add a diagnostic: it stops the branch being refused, which is a
        // behaviour change to the largest failure class. `ambiguous_unrenderable_*`
        // tests pin the membership in both directions.
        //
        // `damage_amount >= 0` is load-bearing, not defensive: `Damage` is SIGNED, and a
        // negative amount is the engine's spelling for a heal (Pain Split). The walk
        // emits on `active_hp < before` only, so a heal-direction `Damage` renders
        // NOTHING -- see this function's caller doc for the reproduced C52-mirror defect.
        Instruction::Damage(damage) if damage.damage_amount >= 0 => None,
        // A DIRECT self-heal is renderable; drain, an absorb ability and Liquid Ooze are not.
        // The discrimination needs the whole tail and the attacker -- see
        // `heal_is_a_direct_self_heal` for why, and for why this is a deliberately PARTIAL
        // close of the family rather than a whole one.
        Instruction::Heal(_) if heal_is_a_direct_self_heal(tail, index, attacker) => None,
        Instruction::Switch(_) => None,
        Instruction::SetLastUsedMove(_)
        | Instruction::ChangeDamageDealtDamage(_)
        | Instruction::ChangeDamageDealtMoveCatagory(_)
        | Instruction::ToggleDamageDealtHitSubstitute(_) => None,

        // --- blocked, grouped by the PROTOCOL LINE the walk would have to emit -----
        //
        // Grouped by protocol line, NOT by engine instruction kind, because the point of
        // this classifier is to scope renderer work: a ranking whose buckets do not
        // correspond to lines to emit mis-scopes the fix it exists to enable.
        //
        // Review caught three groups violating that, all of which would have sent a
        // reader to write the wrong thing:
        //   * `Change<Stat>` is NOT a `-boost`. The engine emits it from
        //     `recalculate_stats` on mega/forme/transform, and it carries no protocol
        //     line at all. Split out as `statrecalc`.
        //   * `SetSleepTurns` / `SetRestTurns` / `DecrementRestTurns` are NOT `-status`.
        //     The NAMED path renders them as `|cant|...|slp`. Split out as
        //     `sleepcounter`.
        //   * `DecrementPP` and `SetRestSleepPendingRefund` have NO public line on any
        //     path -- the latter's own comment says "pure bookkeeping". A `pp` bucket
        //     named a protocol line that does not exist in the protocol. Both are now
        //     `silent`, which says the truth: renderable by doing nothing, and blocked
        //     only because the allowlist is conservative about tails it has not audited.
        //
        // A heal-direction `Damage` lands with `Heal` because what the walk drops is
        // identical: an HP INCREASE. That grouping IS by protocol line, so it stays.
        //
        // SUB-CASED, not widened. `heal_subcase` only ever returns a blocking family, so
        // the refused set is unchanged -- this splits the bucket for ranking, it does not
        // admit anything. See `heal_subcase` for why the drain bucket keeps its ambiguous
        // name instead of guessing between drain and Shell Bell.
        Instruction::Damage(_) | Instruction::Heal(_) => {
            Some(heal_subcase(tail, index, attacker))
        }
        // RENDERED NOW, so no longer a blocker. The unnamed-callee walk emits the
        // `|-boost|`/`|-unboost|` line above, which is the whole reason this family existed:
        // a bare `[Boost]` tail is fully expressible and must not refuse a world. Moving an
        // arm into the `None` set is a BEHAVIOUR CHANGE by design -- it stops refusing a
        // class -- which is why `the_renderable_allowlist_is_exactly_what_it_was` had to be
        // updated deliberately rather than silently widened.
        // REOPENED, and NARROWED rather than reverted. #1131 admitted every `Boost`: right
        // for a move's own stat change, wrong for the switch-out reset Showdown does not
        // narrate. See `boost_may_be_a_switch_out_reset` for why this refuses rather than
        // rendering silence.
        Instruction::Boost(_) if boost_may_be_a_switch_out_reset(tail, index) => Some("boost"),
        Instruction::Boost(_) => None,
        Instruction::ChangeAttack(_)
        | Instruction::ChangeDefense(_)
        | Instruction::ChangeSpecialAttack(_)
        | Instruction::ChangeSpecialDefense(_)
        | Instruction::ChangeSpeed(_) => Some("statrecalc"),
        Instruction::ChangeStatus(_) => Some("status"),
        // `SetSleepTurns` and `DecrementRestTurns` are the two the NAMED path renders as
        // `|cant|...|slp` (events.rs render arms). `SetRestTurns` is NOT one: the engine
        // emits it immediately after `ChangeStatus(-> SLEEP)` when Rest is used, so the
        // lines belong to that `ChangeStatus` and the accompanying `Heal`. It has no
        // render arm and would have inflated this bucket.
        Instruction::SetSleepTurns(_) | Instruction::DecrementRestTurns(_) => {
            Some("sleepcounter")
        }
        // Only a substitute HIT pairs `DamageSubstitute` with a missing
        // `-activate`/`-end` pair. `ChangeSubstituteHealth` is `silent`, not here:
        // creation emits it alongside `Damage` + `ApplyVolatileStatus(SUBSTITUTE)` and
        // break alongside `RemoveVolatileStatus(SUBSTITUTE)`, so a `volatile` or
        // `substitute` companion is ALWAYS present and the line to emit is theirs.
        // Filing it here made a Substitute-CREATION tail report `substitute+volatile`
        // when the only missing line is the `volatile` one.
        // RENDERED NOW: the walk emits `|-activate|...|Substitute|[damage]`, so a substitute
        // hit is expressible. This closes the oracle's surviving 6 of 16 `ambiguous_unrenderable`
        // refusals, all of which are `[DamageSubstitute, RemoveVolatileStatus]`.
        Instruction::DamageSubstitute(_) => None,
        // The SUBSTITUTE break is rendered; every OTHER volatile is not. Admitting
        // `RemoveVolatileStatus` wholesale would be the C52-mirror defect, because the walk has
        // no line for Leech Seed, Confusion, Encore or the rest -- so the guard is on the
        // volatile IDENTITY, not on the variant.
        // Same predicate the walk matches on -- see `substitute_break_side`. Written as a
        // concrete pattern with a guard rather than a bare `_`, so the "EXHAUSTIVE on
        // purpose -- no `_` arm" claim above stays literally true and a NEW engine variant
        // is still a compile error rather than falling into this arm.
        Instruction::RemoveVolatileStatus(_) if substitute_break_side(tail, index).is_some() => {
            None
        }
        Instruction::ApplyVolatileStatus(_)
        | Instruction::RemoveVolatileStatus(_)
        | Instruction::ChangeVolatileStatusDuration(_) => Some("volatile"),
        Instruction::ChangeSideCondition(_) => Some("sidecondition"),
        Instruction::ChangeWeather(_) | Instruction::DecrementWeatherTurnsRemaining => {
            Some("weather")
        }
        Instruction::ChangeTerrain(_)
        | Instruction::DecrementTerrainTurnsRemaining
        | Instruction::ToggleTrickRoom(_)
        | Instruction::DecrementTrickRoomTurnsRemaining => Some("field"),
        Instruction::DisableMove(_) | Instruction::EnableMove(_) | Instruction::ChangeMoveId(_) => {
            Some("moveslot")
        }
        // Split out of the old `identity` group, which bundled ONE genuine gap with five
        // silents and so violated the rule stated above as badly as `status` did before
        // `sleepcounter` was split off. `ChangeItem` is the genuine one: Knock Off, Trick
        // and consumed berries need `|-item|` / `|-enditem|`.
        Instruction::ChangeItem(_) => Some("item"),
        // NO PUBLIC LINE ON ANY PATH. This is the most decision-relevant bucket in the
        // scheme: a `silent`-only tail needs ZERO renderer work, just an allowlist audit,
        // so filing anything here that does need a line under-reports work, and filing a
        // silent under a named family books "write a protocol line" for something that
        // needs none.
        //
        //   * `SetRestSleepPendingRefund` -- its own comment says "pure bookkeeping".
        //   * `DecrementPP` -- there is no `-pp` line in the protocol.
        //   * `SetRestTurns` -- see `sleepcounter` above.
        //   * `ChangeSubstituteHealth` -- see `substitute` above.
        //   * `ChangeWish` / `DecrementWish` -- `DecrementWish` is read only as a
        //     LOOKAHEAD, to tag an existing `|-heal|` with `[from] move: Wish`. The
        //     renderable unit is that `Heal`, already accounted for -- admitted by
        //     `heal_is_a_direct_self_heal`, or bucketed under one of the `heal_*` sub-cases.
        //   * `SetFutureSight` / `DecrementFutureSight` -- pending-slot bookkeeping;
        //     `SetFutureSight`'s arm is `sim.apply` only and the decrement has no arm.
        //   * `ChangeType` / `ChangeAbility` / `FormeChange` -- the named path's own arm
        //     renders a single `|-transform|` and applies the rest silently.
        //   * `ChangeTransformSnapshot`, `ToggleTerastallized` (gen9, unreachable in
        //     gen3), and the switch-flow flags -- no render arm at all.
        //   * `ToggleBatonPassing` -- sets a flag DECORATING an existing `|switch|`.
        Instruction::SetRestSleepPendingRefund(_)
        | Instruction::DecrementPP(_)
        | Instruction::SetRestTurns(_)
        | Instruction::ChangeSubstituteHealth(_)
        | Instruction::ChangeWish(_)
        | Instruction::DecrementWish(_)
        | Instruction::SetFutureSight(_)
        | Instruction::DecrementFutureSight(_)
        | Instruction::ChangeType(_)
        | Instruction::ChangeAbility(_)
        | Instruction::FormeChange(_)
        | Instruction::ChangeTransformSnapshot(_)
        | Instruction::ToggleTerastallized(_)
        | Instruction::SetSideOneMoveSecondSwitchOutMove(_)
        | Instruction::SetSideTwoMoveSecondSwitchOutMove(_)
        | Instruction::ToggleBatonPassing(_)
        | Instruction::ToggleShedTailing(_)
        | Instruction::ToggleSideOneForceSwitch
        | Instruction::ToggleSideTwoForceSwitch => Some("silent"),
        // NOT `silent`, and the distinction is exactly the one that group's comment
        // insists on: `MoveImmobilized` DOES have a public line
        // (`|cant|<ident>|Attract`), it is just emitted by a different path --
        // `render_move_phase`'s dedicated arm, which returns before this walk is
        // reachable. Filing it under `silent` would book "needs zero renderer work,
        // just an allowlist audit" for the one instruction in this match whose
        // presence HERE means the renderer's own statement order broke.
        //
        // STRUCTURALLY UNREACHABLE, which is stronger than the walk's other
        // unreachable arms and is why it gets a family of its own rather than sharing
        // one. The engine pushes this marker from
        // `generate_instructions_from_existing_status_conditions`, and that call is
        // guarded by `!choice.sleep_talk_move` -- so a Sleep Talk CALLEE's generation
        // cannot produce one, and the caller has already consumed or refused any
        // marker in the enclosing tail before this walk starts.
        Instruction::MoveImmobilized(_) => Some("immobilizer"),
    }
}

/// The effect families this ambiguous tail contains that the walk would DROP.
///
/// Empty means fully renderable, and the branch is kept as telemetry-only. Non-empty
/// means the branch is refused, and the slug names WHY -- which is the whole point.
///
/// Era 59 measured `sleeptalk_called_unidentified:ambiguous_unrenderable` at 8,149 world
/// failures, the single largest world-level refusal in the era and 51.6% of the abort
/// channel, as ONE opaque key. There was no way to tell whether closing it needed
/// `-boost`, `-status`, `-heal`, `-sidestart` or the Substitute family, so no fix could
/// be scoped or ranked. This is the treatment #1030 gave `attract_empty_tail_ambiguous`,
/// which is why that class arrives already broken into 17 measurable sub-cases and this
/// one did not.
///
/// The only prior figure was 10 `[Boost]` and 6 `[DamageSubstitute, RemoveVolatileStatus]`
/// over the 16-tail ORACLE CORPUS. That is not a production distribution and is not
/// assumed to be one here: 16 hand-built tails cannot rank a class of 8,149.
/// Keep a family only if it is REGISTERED; otherwise degrade to `unclassified`.
///
/// Extracted as a seam so the `else` branch is reachable from a test. Inlined, it was not:
/// every family the classifier emits is registered, so no input could take that branch and
/// DELETING the degradation left the whole suite green -- while the behaviour difference is
/// real and severe. Proven by paired mutation: with an unregistered token, keeping this
/// yields a measurable `unclassified` key; removing it panics through
/// `assert_subcase_vocabulary` on the production render path, and a pyo3 panic is
/// `BaseException`-derived while `engine_search.py` catches only `Exception` -- so the
/// failure mode was a dead campaign worker.
///
/// `order` is a parameter rather than a direct read of `UNRENDERABLE_FAMILY_ORDER` for
/// exactly one reason: a test can pass a deliberately short list and reach the branch.
fn registered_family_or_unclassified(family: &'static str, order: &[&str]) -> &'static str {
    if order.contains(&family) {
        family
    } else {
        "unclassified"
    }
}

fn unrenderable_tail_families(
    tail: &[Instruction],
    attacker: SideReference,
) -> Vec<&'static str> {
    unrenderable_tail_families_with_protect(tail, attacker, false, true)
}

fn unrenderable_tail_families_with_protect(
    tail: &[Instruction],
    attacker: SideReference,
    defender_protected: bool,
    defender_absorb_zero_heal_possible: bool,
) -> Vec<&'static str> {
    let mut families: Vec<&'static str> = Vec::new();
    for index in 0..tail.len() {
        // An UNREGISTERED token degrades to `unclassified` here, which is in both the
        // order list and the vocabulary. That makes `assert_subcase_vocabulary`
        // UNREACHABLE from this path by construction, and the difference is not
        // cosmetic: pyo3 maps a Rust panic to `PanicException`, which derives from
        // `BaseException` specifically so it propagates past ordinary handlers, and
        // `engine_search.py` catches only `except Exception`. So the old behaviour --
        // sort an unknown token last and let the assert fire -- meant a DEAD CAMPAIGN
        // WORKER, which is strictly worse than the bad aggregate key the assert exists
        // to prevent.
        //
        // The realistic sequence that would have hit it: the engine adds a variant, the
        // exhaustive match forces an author to write `Some("newfamily")`, the author
        // forgets `UNRENDERABLE_FAMILY_ORDER`, and the release wheel aborts mid-campaign.
        // Now that path yields a measurable `unclassified` bucket instead, and the test
        // below still fails in CI so the omission is caught before it ships.
        if let Some(family) = unrenderable_family_at_with_protect(
            tail, index, attacker, defender_protected, defender_absorb_zero_heal_possible) {
            let family = registered_family_or_unclassified(family, UNRENDERABLE_FAMILY_ORDER);
            if !families.contains(&family) {
                families.push(family);
            }
        }
    }
    families.sort_by_key(|family| {
        UNRENDERABLE_FAMILY_ORDER
            .iter()
            .position(|known| known == family)
            // Now genuinely unreachable: every token reaching here was just checked for
            // membership. `usize::MAX` rather than a panic for the same reason as above.
            .unwrap_or(usize::MAX)
    });
    families
}

/// `ambiguous_unrenderable`, with the blocking families named.
///
/// Kept next to the classifier so the `:`-joined shape and the contract-tag prefix stay
/// visible together. The prefix must remain `SLEEPTALK_LOSSY_TAG`, since
/// `mark_attribution_unsafe_subcase` asserts it and
/// `engine_transition_differential.py` matches the bare tag exactly.
fn ambiguous_unrenderable_slug(tail: &[Instruction], attacker: SideReference) -> String {
    ambiguous_unrenderable_slug_with_protect(tail, attacker, false, true)
}

fn ambiguous_unrenderable_slug_with_protect(
    tail: &[Instruction],
    attacker: SideReference,
    defender_protected: bool,
    defender_absorb_zero_heal_possible: bool,
) -> String {
    // THE SAME FACTS THE WALK USED. An earlier version called the fail-closed 2-arg form, so a
    // tail where the Protect marker is now RENDERED but something else still blocks would be
    // keyed `...:heal_zero_marker` -- naming a family the walk no longer refuses. Era 63 would
    // then rank a closed family as open. The PR's own risk note predicts exactly this
    // population: worlds whose FIRST refuser was the marker and whose second is something else.
    let families = unrenderable_tail_families_with_protect(
        tail, attacker, defender_protected, defender_absorb_zero_heal_possible);
    // Unreachable: this slug is only built when the tail is NOT fully renderable, which
    // is defined as a non-empty family list. Named rather than emitting a bare trailing
    // colon so a future edit that breaks that correspondence shows up in the measurement.
    let joined = if families.is_empty() {
        "unclassified".to_string()
    } else {
        families.join("+")
    };
    format!("{SLEEPTALK_LOSSY_TAG}:ambiguous_unrenderable:{joined}")
}

/// Why `identify_sleep_talk_called` could not name the called move.
///
/// The two causes need OPPOSITE fixes and the single `None` hid which one was
/// happening. `Ambiguous` means two candidates regenerate byte-identical tails,
/// which no amount of renderer cleverness can separate -- only the engine
/// recording which move it actually called can. `NoneMatched` means the
/// regeneration reproduced NO candidate's tail, which is a different defect: the
/// replay diverges from what the engine really did, and it is potentially fixable
/// without touching the engine.
///
/// `sleeptalk_called_unidentified` is 48.9% of world failures on the era-55
/// probe -- larger than every other class combined -- so which of these two it
/// actually is decides the whole next phase of work.
enum SleepTalkIdent {
    Matched(Box<Choice>),
    /// No candidate regenerated the observed tail.
    NoneMatched(NoneMatchedShapes),
    /// Two or more candidates regenerate the SAME tail.
    Ambiguous,
}

/// What the callee scan learned, beyond WHICH callee it was.
///
/// The scan regenerates every Sleep Talk candidate through the engine's own modification
/// pass, so it already holds the one fact the zero-heal guard was approximating with the
/// defender's HP: whether any callee reaches the absorb no-op's producer at all. Returning
/// it costs nothing and is strictly more informative than the proxy -- see
/// `callee_can_convert_an_opponent_heal` for why the proxy was not enough.
struct SleepTalkProbe {
    ident: SleepTalkIdent,
    /// Does ANY candidate's post-modification choice still carry an OPPONENT-targeted
    /// positive heal -- i.e. the exact and only precondition of the engine's full-HP absorb
    /// no-op (`gen3/generate_instructions.rs` `get_instructions_from_heal`)?
    ///
    /// OVER ALL CANDIDATES, not just the matching ones, and that is deliberate: it is the
    /// fail-closed direction (more candidates can only make this MORE true, hence refuse
    /// more), and it does not depend on the match set, which the ambiguity means is not a
    /// singleton anyway.
    ///
    /// EQUIVALENT-MUTANT NOTE, recorded so the next reader does not chase it -- and SCOPED,
    /// because the first version of this note was stated for the whole function and is only
    /// true of one arm. Review caught that.
    ///
    /// **On the `Ambiguous` arm**, narrowing the scan to MATCHING candidates is equivalent,
    /// not merely unkilled: if producer 2 fired, the callee that fired is the one that
    /// generated this tail, so it matches by construction and a matching-only scan cannot
    /// miss it. A candidate that carries the converted heal and does NOT match did not
    /// produce THIS tail, so its heal is irrelevant to it.
    ///
    /// **On the `NoneMatched` arm that argument is FALSE as stated**, because the matching set
    /// is empty by definition -- "the callee that fired matches" is exactly what did not
    /// happen there. The arm does reach this flag: `sleeptalk_refusal_is_unsafe_with_protect`
    /// answers `NoneMatched(_) => true`, and `mark_attribution_unsafe_subcase` does not
    /// short-circuit, so the walk runs and calls `protect_blocked_marker_side` with it. What
    /// bounds the consequence is that the refusal is UNCONDITIONAL on that arm: the flag
    /// cannot turn a refused `NoneMatched` branch into a searched one. It can only change
    /// lines on a branch nothing consumes, and fire the render counter there -- which is why
    /// that counter is documented as counting BRANCH RENDERS and not reclaimed worlds.
    /// Measured: `…_absorb_full_hp` and `none_matched` decisions do not co-occur in any shard
    /// of any arm run for this change (9900068 carries 19 `none_matched` decisions and 0
    /// full-HP renders; every shard with full-HP renders carries 0 `none_matched`). That is
    /// NOT a discharge -- nothing shows the marker could have fired in 9900068 -- and it is
    /// recorded as an observation, not a proof.
    ///
    /// **`NoCandidates` has no behavioural coverage and cannot have any**, which is why a
    /// strictly-safer mutant that forces this flag `true` on an empty candidate list also
    /// survives. Measured against a control rather than argued: with the sleeper's only move
    /// being Sleep Talk the engine emits ONE branch, the 50% "nothing happened" arm, and no
    /// branch carrying a callee tail -- so no branch exists on which the flag is consulted.
    /// The control with two callees emits the second branch and does refuse. An arm with no
    /// reachable branch cannot be distinguished by any fixture.
    ///
    /// The all-candidates form is kept because it is the strictly more conservative of the two
    /// and does not rest on the `Ambiguous`-only argument. Restoring the old EARLY RETURN on
    /// the second match is a different mutant and is NOT equivalent -- it skips candidates
    /// outright, and `the_callee_scan_covers_candidates_after_the_second_match` kills it.
    callee_can_convert_an_opponent_heal: bool,
}

/// The CONTRACT tag. `engine_transition_differential.py` matches this exactly
/// (`set(lossy) == {_SLEEPTALK_LOSSY_MARKER}`) to decide branch usability, so it
/// must never carry a sub-case suffix. Named once so the two call sites cannot
/// drift apart.
const SLEEPTALK_LOSSY_TAG: &str = "sleeptalk_called_unidentified";

/// Every literal the Protect-marker counter can emit, named ONCE so the emit site and the
/// vocabulary gate cannot drift.
///
/// They used to be inline literals at the emit site and hand-copied, one per `assert`, into
/// `the_live_subcase_slugs_are_all_in_vocabulary`. That gate's own comment says "**BOTH** have
/// to clear it or the branch that fires less often is the one that panics a release wheel" --
/// and when a THIRD literal was added for the full-HP reclaim, the gate was not updated and
/// its stated invariant was silently violated. Review found it; a mutant deregistering the new
/// token survived the whole suite.
///
/// Hand-copying was the defect, so the copy is gone: the gate calls
/// `protect_marker_counter_slug` over every input and validates whatever comes back.
const PROTECT_MARKER_RENDERED: &str = "sleeptalk_called_unidentified:protect_marker_rendered";
const PROTECT_MARKER_RENDERED_ABSORB_HEADROOM: &str =
    "sleeptalk_called_unidentified:protect_marker_rendered_absorb_headroom";
const PROTECT_MARKER_RENDERED_ABSORB_FULL_HP: &str =
    "sleeptalk_called_unidentified:protect_marker_rendered_absorb_full_hp";
/// WHICH counter literal a rendered Protect marker belongs to.
///
/// Extracted from the emit site so the vocabulary gate can DERIVE the set it validates by
/// calling this over every input, instead of mirroring a hand-written list. The mirror is
/// what went stale: the gate spelled out two literals under the comment "BOTH have to clear
/// it", a third was added at the emit site, and the gate was not updated.
fn protect_marker_counter_slug(
    defender_has_absorb_ability: bool,
    defender_absorb_heal_clamps_to_zero: bool,
) -> &'static str {
    match (
        defender_has_absorb_ability,
        defender_absorb_heal_clamps_to_zero,
    ) {
        (true, true) => PROTECT_MARKER_RENDERED_ABSORB_FULL_HP,
        (true, false) => PROTECT_MARKER_RENDERED_ABSORB_HEADROOM,
        (false, _) => PROTECT_MARKER_RENDERED,
    }
}

/// Measurement label for a failed Sleep Talk identification.
///
/// Split out of the render path so the ident-to-label mapping is testable
/// WITHOUT needing an engine state that reaches each variant. That matters:
/// independent review showed the previous end-to-end `none_matched` fixture only
/// reached that arm on a STALE vendored engine missing patch C87
/// (`poke-engine-gen3-sleeptalk-crit-arm.patch`), and on a faithful build the
/// arm is not reachable at all on the gen3 randbats set pool. Pinning the
/// mapping here keeps the label honest even for a variant production may never
/// produce -- an unreachable arm must still be labelled correctly if the engine
/// ever regresses into producing it.
fn sleeptalk_subcase_slug(ident: &SleepTalkIdent) -> &'static str {
    match ident {
        SleepTalkIdent::Ambiguous => "sleeptalk_called_unidentified:ambiguous",
        // A SET cannot be one static string, so `NoneMatched` is emitted by
        // `none_matched_slugs` at the marking site -- one slug per observed shape, which the
        // existing sort-and-join in `attribution_unsafe_label` composes into one deterministic
        // key. Routing it through here would force either a `format!` (ungreppable, and this
        // returns `&'static str`) or a lossy reduction back to a single shape, which is exactly
        // what the set replaced.
        SleepTalkIdent::NoneMatched(_) => {
            unreachable!("NoneMatched is emitted per shape by `none_matched_slugs`")
        }
        // Exhaustive on purpose. A `_` arm would send a future variant into
        // `none_matched` with no compiler error -- a silent MIS-DIAGNOSIS of the
        // largest failure class rather than a crash. `Matched` cannot reach here:
        // the caller destructures it in the success arm.
        SleepTalkIdent::Matched(_) => {
            unreachable!("Matched is handled by the identifying arm above")
        }
    }
}

/// Identify which move Sleep Talk called by re-generating each sleep-talk
/// candidate's instructions from the current (prelude-applied) state and
/// matching the branch tail exactly.
///
/// Returns [`SleepTalkIdent::Matched`] with the MUTATED candidate choice (the
/// engine's own modification pass applied), or one of the two failure variants.
/// Those two used to be a single `None`, which conflated causes that need
/// opposite fixes -- see [`SleepTalkIdent`].
fn identify_sleep_talk_called(
    state: &mut State,
    side: SideReference,
    defender_choice: &Choice,
    outer_choice: &Choice,
    tail: &[Instruction],
    branch_on_damage: bool,
) -> SleepTalkProbe {
    let candidates = {
        let s = match side {
            SideReference::SideOne => &state.side_one,
            SideReference::SideTwo => &state.side_two,
        };
        s.get_active_immutable().get_sleep_talk_choices()
    };
    let mut matched: Option<Choice> = None;
    // AMBIGUITY IS NOW COUNTED RATHER THAN RETURNED EARLY. The old code returned
    // `Ambiguous` the instant a second candidate matched, which threw away the remaining
    // candidates' modified choices -- and `callee_can_convert_an_opponent_heal` must be
    // computed over ALL of them or it is not the fail-closed direction. The returned VARIANT
    // is unchanged (`Ambiguous` iff two or more matched), and the extra `shapes` this now
    // records are unreachable: `shapes` is read only on the `matched == 0` path.
    let mut match_count = 0usize;
    let mut can_convert_an_opponent_heal = false;
    // The CLOSEST miss across all candidates. Seeded at the least informative shape so a
    // candidate list that produces nothing still yields a token rather than a default that
    // reads as a diagnosis.
    // EVERY shape observed, not the closest one. See `NoneMatchedShapes`.
    let mut shapes = NoneMatchedShapes::default();
    for candidate in candidates {
        let mut choice = candidate.clone();
        choice.sleep_talk_move = true;
        // Inherit the OUTER Sleep Talk choice's move order, exactly as the engine
        // does for the callee (`generate_instructions.rs`:
        // `new_choice.first_move = choice.first_move`). The move table's default
        // is `true` (`choices.rs`), so a second-moving Sleep Talk regenerated its
        // callee as if it had moved first.
        //
        // PINNED by a test. This comment used to read "NOT pinned by a test --
        // reverting this line alone leaves the suite green", which was false and
        // false in the direction that invites deletion. Measured: reverting this
        // line alone fails `every_sleeptalk_attribution_names_the_callee_the_engine_used`
        // with `INPUT FIDELITY: 96 tail(s) ... left: 96, right: 0`. The test's own
        // comment says the same ("that revert drives it 0 -> 96"); the two
        // contradicted each other and this one was the stale half.
        //
        // It does flip the value: the candidate's move-table default is `true`,
        // and `outer_choice.first_move` is `false` whenever Sleep Talk moves
        // second, which four crate tests reach. It changes no observable outcome
        // for a STRUCTURAL reason: C31's enumeration exists to preserve rolls
        // because a PENDING HP-reading move will read HP later in the turn. When
        // the sleeper moves second the defender's Substitute/Flail/Reversal has
        // already resolved, so nothing is pending, the gate's purpose is moot,
        // and its preconditions cannot produce a differing tail.
        //
        // Keep it anyway: it mirrors the engine one-for-one, and the two other
        // `!choice.first_move` sites it makes reachable inside the probe (drag
        // and force-switch early returns) are blocked only because they stop the
        // attacker's move executing, leaving no callee tail to identify. That is
        // a structural argument about today's engine, not an invariant.
        choice.first_move = outer_choice.first_move;
        let mut generated: Vec<StateInstructions> = Vec::with_capacity(4);
        generate_instructions_from_move(
            state,
            &mut choice,
            // The REAL defender choice, not `Choice::default()`. The engine gates
            // its 32-roll damage enumeration on
            // `pending_hp_reading_move(defender_choice)` -- {SUBSTITUTE, FLAIL,
            // REVERSAL} -- so regenerating against a `NONE` move produced the
            // ordinary 2-branch max/crit collapse while the engine had emitted one
            // of 32 rolls. Nothing matched, the callee was unidentifiable, and the
            // whole world was refused as `sleeptalk_called_unidentified:none_matched`.
            defender_choice,
            side,
            StateInstructions::default(),
            &mut generated,
            branch_on_damage,
        );
        // READ THE PRODUCER'S OWN INPUT, from the choice the engine's modification pass just
        // finished mutating. `choice_can_convert_an_opponent_heal` documents why this field
        // and not the defender's HP is the discriminator.
        can_convert_an_opponent_heal |= choice_can_convert_an_opponent_heal(&choice);
        if generated
            .iter()
            .any(|branch| branch.instruction_list.as_slice() == tail)
        {
            match_count += 1;
            if matched.is_none() {
                matched = Some(choice);
            }
        } else {
            // NO MATCH for this candidate. Record HOW CLOSE it came, because the match is
            // byte-exact on the whole instruction list and so a single differing numeric field
            // is indistinguishable from a wholly different transition -- and those are
            // different bugs with different owners.
            //
            // era 60 measured `none_matched` at 3,595 world failures, third largest, and the
            // era-60 measurement says in as many words that it "must be classified before it
            // can be fixed". This is that classification, and it is the same move that turned
            // `ambiguous_unrenderable` from one opaque key into a ranked family list.
            for branch in &generated {
                shapes.insert(divergence_shape(branch.instruction_list.as_slice(), tail));
            }
        }
    }
    let ident = match matched {
        Some(_) if match_count > 1 => SleepTalkIdent::Ambiguous,
        Some(choice) => SleepTalkIdent::Matched(Box::new(choice)),
        None => SleepTalkIdent::NoneMatched(if shapes.is_empty() {
            // No candidate produced ANY branch to classify, which is the empty-candidate-list
            // case. `NoCandidates` is reachable only from here.
            let mut only = NoneMatchedShapes::default();
            only.insert(NoneMatchedShape::NoCandidates);
            only
        } else {
            shapes
        }),
    };
    SleepTalkProbe {
        ident,
        callee_can_convert_an_opponent_heal: can_convert_an_opponent_heal,
    }
}

/// Could THIS callee, as the engine's modification pass left it, reach the full-HP absorb
/// no-op?
///
/// This is the discriminator the zero-heal guard was missing, and it is not a new belief: it
/// is the producer's own `if` condition, read off the same struct the producer reads.
///
/// gen3 emits a zero-amount `Heal` from exactly two sites and both push on the DEFENDER, so
/// the instruction cannot say which fired. The guard therefore has to decide from state. It
/// used to decide from the defender's HP alone (`absorb_heal_clamps_to_zero`, #1211), which
/// answers "could an absorb heal have clamped to zero IF one had been converted" -- a
/// NECESSARY condition, and the census block shows it is nowhere near sufficient.
///
/// The SUFFICIENT one is here. `get_instructions_from_heal` pushes the zero-amount `Heal`
/// only in its `heal.target == MoveTarget::Opponent && heal.amount > 0.0` else-branch, and
/// its `heal` is `choice.heal`. So a callee whose modified choice carries no
/// opponent-targeted positive heal cannot be producer 2, whatever the defender's HP is.
///
/// THREE FACTS make that a proof rather than an argument, all read from the engine rather
/// than assumed, because this is the direction that renders a wrong line if it is wrong:
///
///   1. NO MOVE IN THE TABLE carries an opponent-targeted heal natively. Every
///      `heal: Some(Heal { target: .. })` in `choices.rs` targets `User` (measured: 17 of
///      17). The only writer of `target: Opponent` is the absorb abilities' conversion in
///      `gen3/abilities.rs` (`WATERABSORB`, `VOLTABSORB`, `DRYSKIN`) -- exactly
///      `absorb_ability_can_emit_a_zero_heal`'s set. So this field being set IS "an absorb
///      ability converted this callee".
///   2. PROTECT CLEARS IT. `Choice::remove_effects_for_protect` sets `heal = None`, and it
///      runs in `before_move` AFTER `ability_modify_attack_against`. So a protect-BLOCKED
///      callee -- which is what produces producer 1's marker -- provably cannot also be
///      producer 2, and the two producers are mutually exclusive per callee rather than
///      merely unlikely to coincide.
///   3. THE CANDIDATE SET IS THE ENGINE'S. Both the engine's Sleep Talk dispatch
///      (`gen3/generate_instructions.rs`) and the scan above call
///      `Pokemon::get_sleep_talk_choices`, with the same `sleep_talk_move`, `first_move` and
///      `branch_on_damage` threading. So the callee that actually fired is in the scanned
///      set, and scanning all of it cannot miss it.
///
/// WHY NOT REIMPLEMENT THE CONDITION FROM THE MOVE TABLE. The obvious cheaper form -- "does
/// the sleeper know a move of the absorbed type without the protect flag" -- reads the
/// UNMODIFIED table entry and is a fail-OPEN: gen3's `WEATHERBALL` is `Normal` in the table
/// and `Water` in rain, so a rain Weather Ball into a full-HP Water Absorb defender would be
/// judged unable to convert, and the walk would render `|-activate|..|Protect` over an
/// ability activation. Reading the post-modification field cannot make that mistake because
/// the modification is what it reads.
fn choice_can_convert_an_opponent_heal(choice: &Choice) -> bool {
    matches!(
        choice.heal,
        Some(poke_engine::choices::Heal {
            target: MoveTarget::Opponent,
            amount
        }) if amount > 0.0
    )
}

/// How a regenerated candidate branch DIFFERS from the observed tail.
///
/// Reordering these variants COMPILES and is caught by `the_ordering_keeps_the_closest_miss`.
/// An earlier commit message claimed reordering "does not compile, so that property is
/// compile-time rather than tested" -- false in both halves, and stated as fact. Coverage was
/// better than claimed, which is the less harmful direction to be wrong in but still wrong.
///
/// Ordered from most to least diagnostic, so `min` over all candidates keeps the closest miss:
/// if any candidate reproduced the transition's SHAPE and differed only in a number, that is
/// far more informative than a candidate that produced something structurally unrelated.
///
/// The point is ownership, the same argument `choices_unmapped_causes` makes on the Python
/// side. `ValuesOnly` means the renderer regenerated the right transition and disagreed about a
/// ROLL -- the engine's damage enumeration, or a merged chance branch, neither of which the
/// renderer can fix. `Structure` means it regenerated a different transition altogether, which
/// is a candidate-set or state-input bug and IS fixable here. The code above records one
/// instance already: passing `Choice::default()` for the defender made the engine's 32-roll
/// enumeration mismatch, and every such world refused as `none_matched`.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum NoneMatchedShape {
    /// Same variant sequence and same sides; something in the payload differs.
    ///
    /// **Named for the PREDICATE, not a cause — twice corrected to get here.** The first
    /// version claimed "numeric field differs", which review measured false a second time even
    /// after sides were added: `Boost(Attack)` vs `Boost(Speed)`, `ChangeStatus(→SLEEP)` vs
    /// `(→BURN)`, and `Switch(→P1)` vs `(→P4)` all land here. Those are a wrong stat, a wrong
    /// status and a wrong Pokémon — categorical, not numeric, and renderer-side.
    ///
    /// **It also does NOT mean the engine owns it, and an earlier version of this doc said it
    /// did.** Review measured the one `none_matched` population this repo can reproduce -- the
    /// C31 bug, where `Choice::default()` was passed instead of the real defender choice, so
    /// the engine's 32-roll damage enumeration did not match -- and it lands here **132 of
    /// 132**. That bug was entirely RENDERER-side and was fixed in this file. So a numeric
    /// divergence is a roll-enumeration disagreement whose cause may be engine-side (merged
    /// chance branches) or renderer-side (a wrong state input, as C31 was), and this token
    /// must not be read as an ownership verdict.
    ValuesOnly,
    /// Same number of instructions, different variant sequence.
    Structure,
    /// Different lengths, and the SHORTER list is a prefix of the longer -- with the
    /// regenerated BRANCH the shorter one. The candidate reproduced the head of the tail
    /// exactly and the tail continues past it.
    ///
    /// CONSISTENT WITH an over-long tail rather than a wrong callee: `tail` is
    /// `&segment[cursor..]`, which runs to the end of the segment, while
    /// `generate_instructions_from_move` produces only the callee's own instructions.
    ///
    /// It does NOT establish that. Prefix containment is not callee identity: a WRONG callee
    /// whose head coincidentally matches -- one `Damage` of the right amount on the right
    /// side, which short branches make likely -- lands here too. A high count narrows the
    /// search; it does not license bounding the tail on its own. Whether it dominates at all
    /// is an OPEN QUESTION this token exists to answer.
    BranchIsPrefix,
    /// Different lengths, shorter is a prefix of longer, and the TAIL is the shorter one.
    /// The candidate generated MORE than happened -- a state-input or candidate-set fault.
    TailIsPrefix,
    /// Different number of instructions, and neither list is a prefix of the other -- PLUS
    /// the empty-tail case, which is routed here deliberately. An empty tail IS vacuously a
    /// prefix, so admitting it to a containment bucket would fill that bucket with rows
    /// carrying no containment evidence; see the guard in `divergence_shape`.
    ///
    /// Otherwise: a genuinely different transition, which is the reading the bare token was
    /// always assumed to carry and, before the containment split, could not establish.
    Length,
    /// A candidate produced an empty instruction list.
    Empty,
    /// There were NO candidates to regenerate. `get_sleep_talk_choices` returns an empty list
    /// when every non-Sleep-Talk slot is `Choices::NONE`, so the loop body never runs.
    ///
    /// Split out of `Empty`, which conflated it with "a candidate produced nothing" -- and this
    /// one is different in kind: there was nothing to look at, it is knowable with certainty,
    /// and it is trivially fixable. LAST in declaration order, so `min` never prefers it.
    NoCandidates,
}

/// The SET of divergence shapes observed across a decision's candidates.
///
/// A `min` was the first design and review argued it out on three grounds:
///
///   1. It costs nothing to carry the set. `attribution_unsafe` is already a `Vec<String>`
///      that `attribution_unsafe_label` sorts and joins, so emitting one `&'static str` per
///      observed shape composes into a single deterministic key -- no counter, and no change to
///      the Python seam, which counts `f"crate_search: {reason}"` verbatim. That is the
///      mechanism `ambiguous_unrenderable` already uses via `unrenderable_tail_families`'
///      fixed-order dedupe: this file's own precedent, not a new pattern.
///   2. **It DELETES the one defect nothing could pin.** Inverting the cross-candidate `min`
///      to `max` survived the entire suite, reporting the FARTHEST miss -- near-universally the
///      least informative bucket. A union has no "closest" semantics to get backwards, so the
///      failure mode stops existing rather than being tested around. Strictly less code and
///      strictly less risk.
///   3. A min is a lossy projection of a distribution collectable ONCE per campaign.
///      `same_variants_and_sides` alone and `same_variants_and_sides + structure` are different
///      diagnoses, and only the set separates them. The min is recoverable from the set, so the
///      set strictly dominates.
///
/// Cardinality is bounded at 2^7 - 1 = 127, inside the discipline attract already accepts.
#[derive(Clone, Copy, PartialEq, Eq, Default, Debug)]
pub struct NoneMatchedShapes(u8);

impl NoneMatchedShapes {
    fn insert(&mut self, shape: NoneMatchedShape) {
        self.0 |= 1u8 << shape.bit();
    }

    fn contains(self, shape: NoneMatchedShape) -> bool {
        self.0 & (1u8 << shape.bit()) != 0
    }

    fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// In declaration order, so the emitted slug set is deterministic across runs -- the same
    /// requirement `UNRENDERABLE_FAMILY_ORDER` exists for.
    fn iter(self) -> impl Iterator<Item = NoneMatchedShape> {
        NoneMatchedShape::ALL
            .into_iter()
            .filter(move |shape| self.contains(*shape))
    }
}

impl NoneMatchedShape {
    /// EVERY variant, in declaration order. The `iter` above and the vocabulary test both walk
    /// this, so a new variant that is not added here is invisible to both.
    const ALL: [NoneMatchedShape; 7] = [
        NoneMatchedShape::ValuesOnly,
        NoneMatchedShape::Structure,
        NoneMatchedShape::BranchIsPrefix,
        NoneMatchedShape::TailIsPrefix,
        NoneMatchedShape::Length,
        NoneMatchedShape::Empty,
        NoneMatchedShape::NoCandidates,
    ];

    fn bit(self) -> u8 {
        match self {
            NoneMatchedShape::ValuesOnly => 0,
            NoneMatchedShape::Structure => 1,
            NoneMatchedShape::BranchIsPrefix => 2,
            NoneMatchedShape::TailIsPrefix => 3,
            NoneMatchedShape::Length => 4,
            NoneMatchedShape::Empty => 5,
            NoneMatchedShape::NoCandidates => 6,
        }
    }
}

impl NoneMatchedShape {
    /// The sub-case token. Closed and greppable, like the renderer's family order.
    pub fn token(self) -> &'static str {
        match self {
            NoneMatchedShape::ValuesOnly => "shape_same_variants_and_sides",
            NoneMatchedShape::Structure => "shape_structure",
            NoneMatchedShape::BranchIsPrefix => "shape_branch_is_prefix_of_tail",
            NoneMatchedShape::TailIsPrefix => "shape_tail_is_prefix_of_branch",
            NoneMatchedShape::Length => "shape_length",
            NoneMatchedShape::Empty => "shape_empty",
            NoneMatchedShape::NoCandidates => "shape_no_candidates",
        }
    }
}

/// One registered slug per observed shape.
///
/// Static literals rather than a `format!`, for the same reason the rest of this vocabulary is:
/// a formatted key is ungreppable, which is how a class stops being rankable.
fn none_matched_slugs(shapes: NoneMatchedShapes) -> impl Iterator<Item = &'static str> {
    shapes.iter().map(|shape| match shape {
        NoneMatchedShape::ValuesOnly => {
            "sleeptalk_called_unidentified:none_matched:shape_same_variants_and_sides"
        }
        NoneMatchedShape::Structure => "sleeptalk_called_unidentified:none_matched:shape_structure",
        NoneMatchedShape::BranchIsPrefix => "sleeptalk_called_unidentified:none_matched:shape_branch_is_prefix_of_tail",
        NoneMatchedShape::TailIsPrefix => "sleeptalk_called_unidentified:none_matched:shape_tail_is_prefix_of_branch",
        NoneMatchedShape::Length => "sleeptalk_called_unidentified:none_matched:shape_length",
        NoneMatchedShape::Empty => "sleeptalk_called_unidentified:none_matched:shape_empty",
        NoneMatchedShape::NoCandidates => {
            "sleeptalk_called_unidentified:none_matched:shape_no_candidates"
        }
    })
}

/// The CLOSEST divergence across a candidate's regenerated branches.
///
/// Extracted from the loop specifically so it can be tested. Review's mutation battery found
/// every kill landing on `divergence_shape` or the enum, and every SURVIVOR in the aggregation
/// that computes what an era actually reads: swapping the argument order, and -- worst --
/// `min` becoming `max`, which inverts the measurement to report the FARTHEST miss with a
/// fully green suite. The enum's ORDERING was pinned; the USE of `min` was not, because no test
/// crossed from the helper into the loop. It does now.
///
/// The C31 fixture cannot serve here: on a correct build it identifies its callee, so it
/// produces no `none_matched` at all, and the only way to reach this code naturally is to
/// reintroduce the bug. A pure function is testable without that.
fn nearest_divergence<'a>(
    branches: impl Iterator<Item = &'a [Instruction]>,
    tail: &[Instruction],
) -> NoneMatchedShape {
    branches
        .map(|branch| divergence_shape(branch, tail))
        .min()
        // `Empty`, NOT `NoCandidates`. This fires when a candidate that DOES exist generated
        // zero branches, which is "the candidate produced nothing" -- exactly what `Empty`
        // means. Returning `NoCandidates` here reintroduced the very conflation that variant
        // was split out to remove, so `NoCandidates` is now reachable ONLY from the seed, where
        // it means the candidate list itself was empty.
        .unwrap_or(NoneMatchedShape::Empty)
}

/// Classify one candidate branch against the observed tail.
fn divergence_shape(branch: &[Instruction], tail: &[Instruction]) -> NoneMatchedShape {
    if branch.is_empty() {
        return NoneMatchedShape::Empty;
    }
    if branch.len() != tail.len() {
        // SPLIT BY CONTAINMENT before falling back to a bare length mismatch.
        //
        // `shape_length` was era 61's largest world-failure class -- 4,786 worlds, 33.3% --
        // and it says only "the lists are different sizes", which names no fix. The question
        // it cannot answer is whether the SHORTER list is a PREFIX of the longer one, and
        // that distinction is the whole diagnosis:
        //
        //   * `BranchIsPrefix` -- the branch reproduces the head of the tail exactly and the
        //     tail continues past it. CONSISTENT WITH an over-long tail (`&segment[cursor..]`
        //     runs to the END OF THE SEGMENT while the branch is only the callee's own
        //     instructions), but it does NOT prove the callee was right: a wrong callee with
        //     a coincidentally-matching head lands here too, and short branches make that
        //     likely. A high count narrows where to look; on its own it does not license
        //     bounding the tail.
        //   * `TailIsPrefix` -- the branch reproduces the tail and then continues. Consistent
        //     with the candidate generating MORE than happened -- a state-input or
        //     candidate-set fault -- subject to the same coincidence caveat.
        //   * `Length` -- neither contains the other. A genuinely different transition, which
        //     is the reading the bare token was always assumed to carry.
        //
        // DELIBERATELY NOT a mechanism claim. An earlier read of era 61 asserted a
        // "constant-offset signature" from the absence of `ValuesOnly`, which the 460
        // `Structure` worlds refuted -- `Structure` is returned only AFTER the length check
        // passes, so same-length branches demonstrably exist. This split MEASURES the thing
        // that inference guessed at.
        // An EMPTY TAIL is vacuously a prefix of anything, so without this guard
        // `divergence_shape(&[dmg], &[])` returns a containment shape carrying ZERO
        // containment evidence. Empty tails are real here -- `tail` is `&segment[cursor..]`
        // and this file handles the empty case elsewhere -- so the bucket whose doc says
        // "the tail is reproduced and the branch continues" would be contaminated.
        // The branch-empty mirror is caught by the `is_empty` return above.
        if tail.is_empty() {
            return NoneMatchedShape::Length;
        }
        let (shorter, longer) = if branch.len() < tail.len() {
            (branch, tail)
        } else {
            (tail, branch)
        };
        if longer.starts_with(shorter) {
            return if branch.len() < tail.len() {
                NoneMatchedShape::BranchIsPrefix
            } else {
                NoneMatchedShape::TailIsPrefix
            };
        }
        return NoneMatchedShape::Length;
    }
    // VARIANT **and SIDE**. `std::mem::discriminant` alone ignores the entire payload, and
    // `side_ref` lives in the payload -- so an earlier version reported
    // `Damage(SideOne)` vs `Damage(SideTwo)` as a mere "value" difference. Review measured the
    // same collapse for a different stat on `Boost`, a different volatile on
    // `ApplyVolatileStatus`, a different status on `ChangeStatus` and a different weather on
    // `ChangeWeather`. Every one of those is a WRONG-TARGET or WRONG-EFFECT renderer bug, and
    // reporting them as numeric divergence pointed the diagnosis at the wrong owner.
    //
    // Side is checked via `instruction_side`, which is exhaustive over the variants that carry
    // one. A `None` side (field-wide instructions like weather) compares equal, which is
    // correct: those have no side to disagree about.
    if branch.iter().zip(tail).all(|(a, b)| {
        std::mem::discriminant(a) == std::mem::discriminant(b) && instruction_side(a) == instruction_side(b)
    }) {
        return NoneMatchedShape::ValuesOnly;
    }
    NoneMatchedShape::Structure
}

/// Expected collapsed damage values (regular, crit) for the attacking side's
/// move, used ONLY to label `|-crit|` branches. Mirrors the engine's own
/// collapsing (0.925 * max roll).
/// Returns `(regular_collapsed, crit_collapsed, max_regular, max_crit)`.
///
/// The caller MUST pass a pristine Choice. This function's damage-roll path runs
/// `before_move`, so a Choice that has already been through it gets every
/// modifier applied twice (reports/c102).
fn expected_damage_values(
    state: &State,
    side: SideReference,
    choice: &Choice,
    _branch_on_damage: bool,
) -> (Option<i16>, Option<i16>, Option<i16>, Option<i16>) {
    if choice.category == MoveCategory::Status {
        return (None, None, None, None);
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
            Some(values[0]),
            Some(values[1]),
        ),
        Some(values) if values.len() == 1 => (
            Some((values[0] as f32 * 0.925) as i16),
            None,
            Some(values[0]),
            None,
        ),
        _ => (None, None, None, None),
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
/// On any mismatch the side falls back to the per-instruction cause helpers,
/// and BOTH of them guess from fixed-priority state rather than from position.
/// That guessing is the H.1 mechanism, and it bites on both sides: of the 30
/// H.1 rows in `docs/engine_divergence_ledger_20260728.md`, 21 are damage-side
/// (`sandstorm|psn`, `partialtrap|sandstorm`, `leechseed|psn`, `sandstorm|brn`)
/// and 9 are heal-side.
///
/// The two helpers share no predicate — damage goes own-status, own-LeechSeed,
/// weather, partialtrap; heal goes Wish-by-instruction-lookahead,
/// OPPONENT-LeechSeed, own Leftovers, and only the heal side looks ahead at
/// all. The difference that matters FOR THIS HAZARD is the last resort, and
/// only when no predicate matches: `residual_damage_cause` ends in a generic
/// `residual` at its terminal that diverges loudly, while `residual_heal_cause`
/// terminates in a specific `item: Leftovers` and so is confidently wrong even
/// in the fall-through case. A narrow extra hazard on the heal side, not the
/// whole mechanism.
///
/// The predicates below must therefore mirror the engine's gates AS THEY WILL
/// EVALUATE WHEN THE TICK FIRES — which is NOT the same as transcribing them.
/// This plan is built on the PRE-RESIDUAL state, while the engine's gates run
/// after earlier phases have already moved HP. For an HP-dependent gate the two
/// disagree, and copying it across is a measured 5-row regression: see the NOTE
/// on the Leftovers slot below. A slot booked that the engine never fills is
/// not a harmless over-count — it silently corrupts the tag on a sibling
/// heal — but the cure is to model the phase order, not to copy the gate.
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
    // WEATHER EXPIRY, the same class of defect as the Sand Veil gate below and
    // found by the same review. `weather_is_active` does NOT read
    // `turns_remaining` (`gen3/state.rs:1050-1060`), and this function reads the
    // PRE-residual state -- but the engine decrements and clears the weather at
    // `generate_instructions.rs:4144-4163`, BEFORE its chip loop at `:4193`. So on
    // the turn the weather expires the engine emits NO chip while the plan books
    // one, the side's plan goes unusable, and every label on it drops to the
    // constant fallback.
    //
    // Exactly `== 1`, never `<= 1`. The engine's decrement is gated on
    // `turns_remaining > 0`, so any value at or below 0 skips the decrement, keeps
    // its `weather_type`, and therefore KEEPS CHIPPING.
    //
    // PERMANENT gen3 weather is `-1`, not `0`: `gen3/abilities.rs:20`
    // `WEATHER_ABILITY_TURNS: i8 = -1`, which is what Sand Stream, Drizzle and
    // Drought write. An earlier version of this comment said `0`, naming a value
    // that is merely also non-decrementing while missing the one the pool actually
    // produces -- and Tyranitar, Kyogre and Groudon are all in the pool. Row
    // `19100014/35`, one of the two rows this change closes, is Tyranitar switching
    // into its own sand.
    //
    // So `<= 1` is not a stylistic variant, it is a regression that reintroduces the
    // `19100193/46` mislabel across the whole permanent-weather region. Pinned by
    // `expiring_weather_books_no_chip_so_the_drain_keeps_its_label`'s second arm,
    // because a review changed `==` to `<=` and all 375 tests stayed green.
    if state.weather.turns_remaining == 1 {
        return None;
    }
    if state.weather_is_active(&Weather::HAIL) {
        if active.has_type(&PokemonType::ICE) {
            return None;
        }
        return Some("Hail");
    }
    if state.weather_is_active(&Weather::SAND) {
        // SAND VEIL. The engine exempts it at
        // `gen3/generate_instructions.rs:4223`, and this function did not, so it
        // booked a sandstorm chip that never fired. One unfilled slot makes the
        // whole side's plan unusable, which drops EVERY heal on that side into
        // `residual_heal_cause`'s fallback -- and the fallback is a constant
        // function of state, so it cannot label two different heals differently.
        //
        // That is the root cause of `19100014/35`, not the fallback ordering.
        // Cacturne has Sand Veil, so the plan reserved a chip it never emitted,
        // `plan.usable` went false, and both of Cacturne's heals fell through.
        // Reordering the fallback only fixed whichever of the two happened to be
        // the Leftovers tick; the drain stayed mislabelled. With this gate the plan
        // reconciles, the row's 90% arm matches, and the row closes.
        //
        // NOT "both arms" -- an earlier version of this comment said that. The 10%
        // arm is the engine's Leech-Seed-MISSED branch against a Showdown hit
        // (`observed_only=[('leechseed', -33)] engine_only=[]`), and no harness
        // RENDERING change can make a miss branch reproduce a hit. It closes anyway
        // because one matching branch closes a boundary.
        if active.has_type(&PokemonType::ROCK)
            || active.has_type(&PokemonType::GROUND)
            || active.has_type(&PokemonType::STEEL)
            || active.ability == Abilities::SANDVEIL
        {
            return None;
        }
        return Some("Sandstorm");
    }
    None
}

/// G33b — whether each side's order-**10.4** Leftovers slot is UNREACHABLE because
/// the residual phase was truncated by the opposing active's battle-ending faint.
///
/// This is the one over-booking the `NOTE:` on the Leftovers slot below cannot fix.
/// That note refuses an `hp < maxhp` guard because the plan is built on the
/// PRE-residual state and Leftovers fires later, so HP moves underneath the
/// predicate. The truncation is worse than that: the slot is not skipped because a
/// predicate evaluated false, it is skipped because the engine **never reached the
/// handler**. No HP predicate can see it — in the row that motivated this
/// (`19200244/115`, `reports/c143_heal_attribution_diagnosis.md`) the winner ends at
/// 260/268, comfortably below max.
///
/// Ground truth, and it is the engine's own structure rather than an inference:
/// `add_end_of_turn_instructions` runs `stop_residuals_if_battle_ended!` at every
/// entry boundary, mirroring `sim/battle.ts:565-566`'s
/// `this.faintMessages(); if (this.ended) return;`. Order 10 is the SPEED-MAJOR
/// class — one Pokemon at a time, fastest first, each running its whole 10.x set in
/// subOrder before the other side runs any of its own — so a faster loser whose own
/// 10.5 Leech Seed sap kills it ends the battle before the slower winner's 10.3 is
/// ever entered, and the winner's 10.4 Leftovers tick never fires. `ResidualPlan`
/// books it anyway, the count mismatch sets `plan.usable[winner] = false`, and every
/// heal on that side drops to `residual_heal_cause` — a constant function of state
/// which, since C131 change 3, tests Leftovers FIRST. So the bare Leech Seed drain
/// mirror comes back tagged `[from] item: Leftovers`. The HP arithmetic is right;
/// only the attribution is wrong.
///
/// The predicate walks the segment for the instruction that ends the battle and then
/// asks ONE question: was the winner's 10.4 behind that point? Everything that can
/// deliver a battle-ending faint is enumerated from the engine's own section order,
/// and only two of the five arms can gate at all:
///
/// The third column is the FACT and the fourth is what this predicate does about it.
/// They differ on exactly one row, deliberately, and the difference is the
/// weather under-reach recorded below — read the fourth column for what the code
/// does, because there is ONE speed test and it guards both gating rows:
///
/// | what delivered it | where it sits | is the winner's 10.4 behind it? | gated here |
/// |---|---|---|---|
/// | the shared weather entry | order 8 | **yes, always** — order 8 precedes every order-10 handler on both sides | only when the loser is faster, so the winner-faster half is NOT gated |
/// | the loser's own 10.5 / 10.6 / 10.9 | order 10, loser's bucket | yes **iff the loser is faster** | yes, on exactly that condition |
/// | the winner's 10.5 Liquid Ooze recoil | order 10, winner's bucket | no, it already fired | no |
/// | Future Sight | order 11 | no, it already fired | no |
/// | Perish Song | order 12 | no, it already fired | no |
///
/// The two "not gated" order-10+ arms are excluded by state predicate rather than by
/// classifying the instruction, because a lethal residual damage always equals the
/// victim's remaining HP exactly and therefore carries no information about which
/// phase produced it. Liquid Ooze is separable structurally instead: it is the only
/// residual source that writes a NEGATIVE `Heal`, never a `Damage`.
///
/// Two deliberate under-reaches, both leaving the pre-gate booking in place:
///
/// * **A speed TIE is not gated.** `residual_speed_order` returns `None` on an exact
///   tie because `speedSort` shuffles it (`sim/battle.ts:455-457`) and the engine
///   forks BOTH orders, keeping both when they differ. One of the two live orders
///   fires the winner's tick and the other does not, so there is no single answer to
///   give and this declines to guess.
/// * **A fatal weather chip with the WINNER faster is not gated.** Order 8 precedes
///   all of order 10 unconditionally, so that case is a real instance of the same
///   family, and the third column of the table above says so. It is left unshipped
///   because it was not measured, and the single speed test below is what keeps it
///   out: that test guards BOTH order-<=10 arms, so declining to guess a tie also
///   declines the winner-faster weather case. Do not read the table's first row as
///   an unconditional gate -- the fourth column is the code.
fn leftovers_slot_truncated(state: &State, segment: &[Instruction]) -> [bool; 2] {
    const NO_TRUNCATION: [bool; 2] = [false, false];

    let sides = [&state.side_one, &state.side_two];
    let mut hp = [
        sides[0].get_active_immutable().hp,
        sides[1].get_active_immutable().hp,
    ];
    // Reserves cannot enter during the residual phase, so an active that faints
    // with no living reserve loses the battle then and there. COUNTED rather than
    // indexed: the active is itself a party member, so "a living reserve exists"
    // is "more living Pokemon than the active contributes".
    let mut has_reserve = [false; 2];
    for i in 0..2 {
        let mut living = 0usize;
        let mut iter = sides[i].pokemon.into_iter();
        while let Some(mon) = iter.next() {
            if mon.hp > 0 {
                living += 1;
            }
        }
        has_reserve[i] = living > usize::from(hp[i] > 0);
        // Already over before the residual block began. `add_end_of_turn_instructions`
        // returns at its own entry guard in that case, so the segment carries nothing
        // and there is no label to get wrong.
        if hp[i] <= 0 && !has_reserve[i] {
            return NO_TRUNCATION;
        }
    }

    for ins in segment {
        let (loser, by_damage) = match ins {
            Instruction::Damage(d) => {
                let side = side_index(d.side_ref);
                hp[side] -= d.damage_amount;
                (side, true)
            }
            Instruction::Heal(h) => {
                let side = side_index(h.side_ref);
                hp[side] += h.heal_amount;
                (side, false)
            }
            _ => continue,
        };
        if hp[loser] > 0 || has_reserve[loser] {
            continue;
        }
        // The battle ends on this instruction: the entry that caused it runs to
        // completion, `faintMessages()` resolves the faint, `checkWin` sets `ended`,
        // and every REMAINING entry is skipped.
        if !by_damage {
            // Liquid Ooze is the only residual effect that writes a negative `Heal`,
            // and it writes it at the SEEDED side's 10.5 — inside the winner's own
            // bucket, after the winner's 10.4. The tick fired.
            return NO_TRUNCATION;
        }
        let winner = 1 - loser;
        if sides[winner].future_sight.0 == 1 {
            // Order 11, after every order-10 handler on BOTH sides.
            return NO_TRUNCATION;
        }
        if sides[loser]
            .volatile_statuses
            .contains(&PokemonVolatileStatus::PERISH1)
        {
            // Order 12, likewise after all of order 10.
            return NO_TRUNCATION;
        }
        return match residual_speed_order(state) {
            Some(first) if side_index(first) == loser => {
                let mut truncated = NO_TRUNCATION;
                truncated[winner] = true;
                truncated
            }
            _ => NO_TRUNCATION,
        };
    }
    NO_TRUNCATION
}

impl ResidualPlan {
    /// Build from the PRE-residual state, in the engine's own emission order.
    pub(crate) fn build(state: &State, segment: &[Instruction]) -> ResidualPlan {
        let mut plan = ResidualPlan::default();
        let mut drains_opponent = [false; 2];
        let leftovers_truncated = leftovers_slot_truncated(state, segment);
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
            // NOTE: it is tempting to add `&& active.hp < active.maxhp` here,
            // mirroring the engine's gate at `gen3/items.rs:352`. Do not. The
            // plan is built on the PRE-RESIDUAL state, and Leftovers fires at
            // phase 10.4 — after weather chip at phase 8 — so a mon at full HP
            // when the plan is built is routinely below max by the time the
            // tick actually fires, and the engine does emit it. Measured: that
            // guard alone costs 5 rows on seeds 19000000-19000199: matched
            // 15187 -> 15182 AND diverged 36 -> 41 (component_missing_in_engine
            // :psn 2 -> 7). The rows became divergences, not skips. Same trap as the drain slot documented in
            // `a_near_full_hp_seeder_still_over_books_the_drain_slot`: these
            // predicates cannot use HP without modelling the phase order.
            //
            // G33b: the ONE case where the slot must not be booked, and it is not an
            // HP question at all — the handler is never reached, because the residual
            // block was truncated by the opposing active's battle-ending faint. See
            // `leftovers_slot_truncated` for the enumeration of what can deliver that
            // faint and where each sits in the engine's section order.
            if active.item == Items::LEFTOVERS && !leftovers_truncated[i] {
                plan.heal[i].push("item: Leftovers".to_string());
            }
            // Liquid Ooze reverses the drain: the seeder takes damage instead of
            // healing, and the engine emits that as a NEGATIVE Heal on the
            // seeder (gen3/generate_instructions.rs:3624-3647, where the
            // positive-drain branch is the `else`). There is no drain heal to
            // place, so planning a slot for one leaves `plan.heal` one longer
            // than `emitted_heal` — which counts only `heal_amount > 0` — the
            // reconcile below marks the whole side unusable, and the seeder's
            // Leftovers tick falls through to the `[from] Leech Seed` label.
            // That is the legacy H.1 bug this plan exists to prevent.
            drains_opponent[i] = opponent
                .volatile_statuses
                .contains(&PokemonVolatileStatus::LEECHSEED)
                && active.hp > 0
                && opponent.get_active_immutable().hp > 0
                && opponent.get_active_immutable().ability != Abilities::LIQUIDOOZE;
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
    // Liquid Ooze reverses the drain, so a seeded opponent carrying it produces
    // no drain heal on this side at all — any positive heal here is something
    // else. The plan handles this shape now, but this fallback is still reached
    // whenever the plan fails reconciliation for an unrelated reason, and
    // without the guard it re-arms exactly the H.1 mislabel the plan exists to
    // prevent.
    //
    // WHY LEFTOVERS FIRST, stated correctly this time. An earlier version of this
    // comment claimed the rule was "answer with the earlier residual phase",
    // Leftovers being 10.4 and the drain 10.5. That reasoning is wrong twice and
    // a review caught both:
    //
    //   1. This function cannot implement it. It takes `(state, side, next_ins)`
    //      and no heal index, so it is a CONSTANT FUNCTION OF STATE -- it returns
    //      the same answer for every heal on the side. It has no notion of
    //      "first" to reason about.
    //   2. The premise is false in general. The drain is emitted at the VICTIM's
    //      10.5 slot inside the speed-major loop, so when the victim is faster the
    //      drain heal PRECEDES the seeder's own Leftovers tick. See the note at
    //      the top of `ResidualPlan`, which says exactly this and is why the plan
    //      needs the speed order to label heals at all.
    //
    // The real reason Leftovers wins is narrower and does not depend on ordering:
    // since the drain is rendered SILENTLY (see below), `"Leech Seed"` is never a
    // correct answer for a `[from]`-tagged heal, so nothing is displaced by
    // preferring Leftovers. Ordering is the plan's job; this fallback only has to
    // avoid asserting a label that cannot be right.
    //
    // Measured on holdout `19100193/46`. The engine state has the opponent
    // seeded and both actives alive when the plan is built, so a drain slot is
    // reserved — but the SEEDER dies to poison at 10.6 before the 10.5 sap can
    // run, `emitted_heal` is one short of `plan.heal`, reconciliation fails, and
    // this fallback is reached. Checking Leech Seed first labelled Cacturne's
    // Leftovers tick `[from] Leech Seed`: magnitude right (18 = 290/16), source
    // wrong. A real drain from Miltank would have been 273/8 = 34.
    if s.get_active_immutable().item == Items::LEFTOVERS {
        return "item: Leftovers".to_string();
    }
    if opponent
        .volatile_statuses
        .contains(&PokemonVolatileStatus::LEECHSEED)
        && opponent.get_active_immutable().ability != Abilities::LIQUIDOOZE
    {
        // EMPTY, not "Leech Seed". Showdown renders the drain heal SILENTLY:
        // `sim/battle.ts:2293-2296` switches on `effect.id` and
        // `case 'leechseed'` emits `('-heal', target, getHealth, '[silent]')`,
        // reached from the move's own handler at `data/moves.ts:10218-10221`.
        // There is no `[from] Leech Seed` heal line anywhere in Showdown, so the
        // string this used to return was never a correct answer for any state --
        // it could only ever trade one wrong label for another.
        //
        // `ResidualPlan` already knew this: it inserts `String::new()` for its
        // own drain slot. This makes the fallback agree with the plan.
        return String::new();
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
///   "lossy", "attribution_unsafe", "attribution_unsafe_reasons",
///   "lossy_subcases" (counted, not refused), "post",
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
            "lossy_subcases": [],
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
            "lossy_subcases": rendered.lossy_subcases,
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

    // Two of the SIX shapes the renderable allowlist admits are only constructible with
    // these, and `the_renderable_allowlist_is_exactly_what_it_was` pins all six. Imported
    // here rather than at module scope so the non-test build keeps its current import set.
    use poke_engine::instruction::{
        ApplyVolatileStatusInstruction, ChangeDamageDealtDamageInstruction,
        ChangeDamageDealtMoveCategoryInstruction, ChangeItemInstruction,
        ChangeSideConditionInstruction, ChangeStatInstruction, ChangeWishInstruction,
        ChangeSubsituteHealthInstruction, ChangeType, DecrementPPInstruction,
        DisableMoveInstruction,
        HealInstruction, ImmobilizeReason, MoveImmobilizedInstruction,
        RemoveVolatileStatusInstruction,
        SetFutureSightInstruction, SetLastUsedMoveInstruction, SetSleepTurnsInstruction,
        SwitchInstruction, ToggleBatonPassingInstruction,
        ToggleDamageDealtHitSubstituteInstruction,
    };
    use poke_engine::state::{LastUsedMove, PokemonMoveIndex};

    /// #1048 VALIDATED: every confident Sleep Talk attribution names the callee
    /// the ENGINE actually used. 0 wrong attributions over 1,271 branches.
    ///
    /// (An earlier version of this line said 1,278 -- a figure lifted from a
    /// reviewer's independently-built matrix rather than derived from this one.
    /// Recorded because carrying an unverified number is the failure mode this
    /// whole test exists to guard against.)
    ///
    /// #1048 converted ~5,000 refusals into confident `|move| … [from] Sleep Talk`
    /// lines and nothing checked whether they were RIGHT. The feared failure: the
    /// true callee fails to reproduce its own tail while exactly one WRONG
    /// candidate reproduces it, yielding a confident wrong protocol line where
    /// there used to be a loud refusal. That is strictly worse than a refusal,
    /// which is loud, because a wrong attribution silently poisons the fold.
    ///
    /// GROUND TRUTH — the engine labels the tail, not a re-implementation of the
    /// matcher. For each callee C, re-run the FULL pair generator on a state whose
    /// only non-Sleep-Talk slot is C. Every instruction list that run emits is one
    /// the engine itself emits when the callee is C. That is independent of
    /// `identify_sleep_talk_called` by construction: it never calls it, and it
    /// reconstructs none of its inputs.
    ///
    /// A first version of this test labelled tails by re-running the identifier's
    /// own predicate — same `get_sleep_talk_choices`, same
    /// `generate_instructions_from_move`, same byte-equality — which made the
    /// assertions TAUTOLOGIES: the identifier's verdicts are defined as functions
    /// of exactly that result set, so `Matched(C)` on a tail labelled only D was
    /// unreachable. Independent review caught it and supplied this construction.
    ///
    /// IMPLEMENTATION TRAP, learned the hard way: keep the single callee in its
    /// ORIGINAL move slot. Locked-move and PP bookkeeping carry the move index, so
    /// slot-shifting silently unlabels tails.
    ///
    /// The renderer is driven through the REAL path (`render_branch_events`), so
    /// no cursor, prelude slicing, `defender_choice` reconstruction, engine-state
    /// reconstruction or Node process is involved.
    #[test]
    fn every_sleeptalk_attribution_names_the_callee_the_engine_used() {
        use poke_engine::engine::generate_instructions::generate_instructions_from_move_pair;
        use poke_engine::state::PokemonMoveIndex;

        const SLOTS: [PokemonMoveIndex; 3] =
            [PokemonMoveIndex::M1, PokemonMoveIndex::M2, PokemonMoveIndex::M3];

        // Collision-prone sets on purpose (same power/type pairs regenerate
        // byte-identical tails and must be REFUSED, not guessed), plus status,
        // boost, drain, self-KO and locked-move shapes. Petal Dance is in because
        // its crit arm restructures the tail rather than changing an integer, and
        // every bug found in this area was invisible to integer-only fixtures.
        let movesets: [[Choices; 3]; 10] = [
            [Choices::BODYSLAM, Choices::EARTHQUAKE, Choices::REST],
            [Choices::BODYSLAM, Choices::EARTHQUAKE, Choices::PETALDANCE],
            [Choices::THUNDER, Choices::SURF, Choices::REST],
            [Choices::HARDEN, Choices::WITHDRAW, Choices::BODYSLAM],   // identical boosts
            [Choices::TACKLE, Choices::SCRATCH, Choices::REST],        // identical damage
            [Choices::TOXIC, Choices::WILLOWISP, Choices::EARTHQUAKE],
            [Choices::GIGADRAIN, Choices::BODYSLAM, Choices::REST],
            [Choices::EXPLOSION, Choices::BODYSLAM, Choices::REST],
            [Choices::THRASH, Choices::EARTHQUAKE, Choices::REST],
            [Choices::SPLASH, Choices::BODYSLAM, Choices::EARTHQUAKE],
        ];

        let ctx = EventContext {
            species: [vec!["Lead".into()], vec!["Opponent".into()]],
            turn: 1,
            hp_percent: [false, false],
        };
        let build = |callees: &[Choices], keep: Option<usize>, sleeper_first: bool| {
            let mut st = State::default();
            // BOTH move orders, and this DOES pin
            // `choice.first_move = outer_choice.first_move` -- #1048's second half,
            // which every earlier version of this test left green under revert.
            //
            // My recorded reason for that was WRONG. I claimed the engine's gate
            // "does not fire either" when the sleeper moves second. It does:
            // `pending_hp_reading_move` is pure data
            // (`branch_on_damage && choice.first_move && pending_hp_reading_move(..)
            // && fixed_damage.is_none()`) and consults nothing about whether the
            // defender's move already resolved. Instrumented on the 639
            // sleeper-second calls, reverting the line takes generated branches from
            // 2-3 to 15-22, so the roll enumeration runs.
            //
            // The real reason the revert used to survive is a NUMERIC COINCIDENCE:
            // the gate block ends `combine_duplicate_instructions(); return;`, so it
            // REPLACES the average-collapse path, and at `State::default()`
            // magnitudes `floor(0.925*M) == floor(M*92/100)` -- the average IS one of
            // the sixteen rolls, so the byte-match survived either way. Found by
            // independent review after I had recorded the wrong mechanism.
            // HIGH-DAMAGE stats, deliberately. At `State::default()` magnitudes
            // `floor(0.925*M) == floor(M*92/100)`, i.e. the average-collapse tail is
            // byte-identical to roll 92, so a `first_move` revert still matched and
            // the line looked unpinnable. Separating them needs ~0.01*M > 2, i.e.
            // max damage above ~200, which default stats never reach.
            st.side_two.get_active().attack = 318;
            st.side_two.get_active().special_attack = 318;
            st.side_one.get_active().defense = 96;
            st.side_one.get_active().special_defense = 96;
            st.side_one.get_active().maxhp = 404;
            st.side_one.get_active().hp = 404;
            st.side_two.get_active().maxhp = 404;
            st.side_two.get_active().hp = 404;
            st.side_two.get_active().speed = if sleeper_first { 500 } else { 1 };
            st.side_two.get_active().status = PokemonStatus::SLEEP;
            st.side_two.get_active().rest_turns = 0;
            st.side_two
                .get_active()
                .replace_move(PokemonMoveIndex::M0, Choices::SLEEPTALK);
            for (i, slot) in SLOTS.iter().enumerate() {
                // Keep the retained callee IN ITS ORIGINAL SLOT.
                let mv = match keep {
                    None => callees[i],
                    Some(k) if k == i => callees[i],
                    Some(_) => Choices::NONE,
                };
                st.side_two.get_active().replace_move(*slot, mv);
            }
            st.side_one.get_active().speed = if sleeper_first { 1 } else { 500 };
            st
        };

        let mut agree = 0usize;
        let mut wrong: Vec<String> = Vec::new();
        let mut multi_label_refused = 0usize;
        let mut multi_label_unattributed = 0usize;
        let mut single_label_refused = 0usize;
        let mut unlabelled: Vec<String> = Vec::new();
        let mut agree_by_defender: std::collections::BTreeMap<String, usize> =
            std::collections::BTreeMap::new();
        let mut total_branches = 0usize;

        for sleeper_first in [true, false] {
        for defender in [Choices::SUBSTITUTE, Choices::FLAIL, Choices::REVERSAL] {
            for callees in &movesets {
                let s1 = MoveChoice::Move(PokemonMoveIndex::M0);
                let s2 = MoveChoice::Move(PokemonMoveIndex::M0);

                // ENGINE-SIDE LABELS: one restricted run per callee.
                let mut labels: Vec<(Vec<Instruction>, Choices)> = Vec::new();
                for (i, callee) in callees.iter().enumerate() {
                    let mut st = build(callees, Some(i), sleeper_first);
                    st.side_one
                        .get_active()
                        .replace_move(PokemonMoveIndex::M0, defender);
                    for b in generate_instructions_from_move_pair(&mut st, &s1, &s2, true) {
                        labels.push((b.instruction_list.clone(), *callee));
                    }
                }

                // The real, full-moveset branch set.
                let mut full_state = build(callees, None, sleeper_first);
                full_state
                    .side_one
                    .get_active()
                    .replace_move(PokemonMoveIndex::M0, defender);
                let branches =
                    generate_instructions_from_move_pair(&mut full_state, &s1, &s2, true);

                for branch in &branches {
                    total_branches += 1;
                    let mut named: Vec<Choices> = labels
                        .iter()
                        .filter(|(list, _)| list.as_slice() == branch.instruction_list.as_slice())
                        .map(|(_, c)| *c)
                        .collect();
                    named.sort_by_key(|c| format!("{c:?}"));
                    named.dedup();

                    let rendered = render_branch_events(
                        &mut full_state.clone(),
                        &s1,
                        &s2,
                        &branch.instruction_list,
                        true,
                        &ctx,
                    );
                    let refused = rendered
                        .attribution_unsafe
                        .iter()
                        .any(|r| r.starts_with("sleeptalk_called_unidentified"));
                    let attributed: Option<String> = rendered
                        .lines
                        .iter()
                        .find(|l| l.contains("[from] Sleep Talk"))
                        // Field 3, not 2. The line is
                        //   |move|p2a: Opponent|bodyslam|p1a: Lead|[from] Sleep Talk
                        // so split('|') yields ["", "move", ACTOR, MOVE, TARGET, ...].
                        // Reading field 2 returns the ACTOR, which made a first run
                        // report 904 "wrong attributions" that were all
                        // `attributed "p2a: opponent"` -- my off-by-one, not a defect
                        // in #1048. Recorded because that mistake was one assertion
                        // away from being published as "#1048 is broken".
                        .and_then(|l| l.split('|').nth(3).map(|m| m.trim().to_lowercase()));

                    // EVERY bucket is asserted or counted. A `_ => {}` catch-all
                    // here previously swallowed two of them, and one is populated by
                    // a #1048 revert: 6 cases where the engine's own labelling says
                    // TWO callees produce the tail and the renderer named one anyway
                    // -- a coin-flip attribution, and it named DIFFERENT callees on
                    // identical-label tails, so they cannot all be right. The test
                    // reported `WRONG 0` for that revert purely because the bucket
                    // was unrouted, which made a claim of "never misattributions"
                    // look established when it was not.
                    match (named.len(), &attributed, refused) {
                        // Unambiguous engine label + a confident attribution:
                        // the names MUST agree. This is the core case.
                        (1, Some(name), false) => {
                            let truth = move_display(named[0]).to_lowercase().replace(' ', "");
                            if truth == name.replace('_', "").replace(' ', "") {
                                let key = format!(
                                    "{defender:?}/{}",
                                    if sleeper_first { "first" } else { "second" }
                                );
                                *agree_by_defender.entry(key).or_insert(0) += 1;
                                agree += 1;
                            } else {
                                wrong.push(format!(
                                    "defender {defender:?} set {callees:?}: engine used {:?} but \
                                     the renderer attributed {name:?}",
                                    named[0]
                                ));
                            }
                        }
                        // Engine says AMBIGUOUS, renderer named one anyway. The
                        // oracle says the evidence cannot single out a callee, so a
                        // confident name here is a guess and cannot be vindicated.
                        (n, Some(name), false) if n >= 2 => wrong.push(format!(
                            "defender {defender:?} set {callees:?}: engine labels are \
                             {named:?} (AMBIGUOUS) but the renderer confidently \
                             attributed {name:?}"
                        )),
                        // One label, no attribution AND no refusal marker: the
                        // attribution was dropped with nothing loud attached.
                        (1, None, false) => wrong.push(format!(
                            "defender {defender:?} set {callees:?}: engine used {:?} but the \
                             renderer emitted neither an attribution nor a refusal",
                            named[0]
                        )),
                        // Two callees emit byte-identical lists and the renderer names
                        // NOBODY and does NOT refuse. This is the correct outcome and the
                        // one the ambiguity split exists to produce: the oracle says the
                        // evidence cannot single out a callee, so naming one would be a
                        // guess (the arm above catches that) -- but the transition is
                        // proven, since every matching candidate regenerated exactly this
                        // tail, so there is nothing unsafe to refuse.
                        (n, None, false) if n >= 2 => multi_label_unattributed += 1,
                        // Two callees emit byte-identical lists and the branch was
                        // REFUSED. Correct before the split; now it means something
                        // reached the refusing path that should not have.
                        (n, _, true) if n >= 2 => multi_label_refused += 1,
                        // Unambiguous label but refused. LOUD, so not a correctness
                        // failure -- but it is the none_matched/input-fidelity class,
                        // which is the residual risk, so it gets its own counter
                        // rather than being folded into genuine ambiguity.
                        (1, _, true) => single_label_refused += 1,
                        (0, _, _) => unlabelled.push(format!(
                            "defender {defender:?} set {callees:?}: no engine label"
                        )),
                        // Catch-all that ASSERTS rather than passing. Guard arms do
                        // not satisfy exhaustiveness, so without this the compiler
                        // would demand a `_ => {}` -- reintroducing exactly the
                        // silent bucket this rewrite removed.
                        (n, attr, ref_) => wrong.push(format!(
                            "defender {defender:?} set {callees:?}: unasserted shape \
                             (labels {n} = {named:?}, attributed {attr:?}, refused {ref_}) \
                             -- every combination must be a verdict, not a fall-through"
                        )),
                    }
                }
            }
        }
        }

        assert!(
            wrong.is_empty(),
            "#1048 WRONG ATTRIBUTION -- {} case(s):\n  {}",
            wrong.len(),
            wrong.join("\n  ")
        );
        // The oracle's OWN validity check. If the restricted-moveset labelling
        // stops reproducing the real branch set, tails go unlabelled, `agree`
        // falls, and the coverage gate below fires with a message blaming the
        // identifier -- misdiagnosing a broken oracle as a broken subject.
        assert!(
            unlabelled.is_empty(),
            "ORACLE BROKEN: {} tail(s) carry no engine label, so the restricted-moveset \
             labelling no longer reproduces the real branch set. Do NOT read the \
             coverage gate below as a statement about the identifier.\n  {}",
            unlabelled.len(),
            unlabelled.join("\n  ")
        );

        // A single-label refusal means the engine produced this tail from callee C
        // and the identifier could not reproduce it -- the INPUT-FIDELITY class, and
        // the residual risk this whole area is about. It is 0 today, and it is the
        // FIRST and most specific assertion to catch a `first_move` revert: with
        // high-damage stats that revert drives it 0 -> 96. NOT the only one, measured:
        // with this assert neutered the partition below fires too ("213 usable + 24
        // refused != 333 unattributed"), because `single_label_refused` counts toward
        // `total - agree` but not toward `multi_unattr + multi_refused`, so any nonzero
        // value breaks that identity; and the tally pin catches it a third time
        // (`agree 2281 != 2377`).
        //
        // An earlier version of this comment said "ONLY", and cited a per-defender
        // floor of 932/932/417 -- a THIRD stale unasserted triple in this test. That
        // revert moves only the sleeper-SECOND column; sleeper-first is unchanged at
        // 404/900/900. The floor is defence-in-depth against a per-defender collapse
        // the aggregate would hide, not the #1048 tripwire, so its values are not
        // cited here.
        //
        // This is what retires the "first_move is unpinned" note -- see the corrected
        // comment at the assignment itself.
        assert_eq!(
            single_label_refused, 0,
            "INPUT FIDELITY: {single_label_refused} tail(s) carry ONE unambiguous \
             engine label and were still refused -- the identifier cannot reproduce \
             a tail the engine demonstrably produced from that callee. Check the \
             reconstructed inputs (defender_choice, first_move, branch_on_damage)."
        );

        // COVERAGE GATE, CALIBRATED AGAINST A #1048 REVERT -- and scoped to the
        // SLEEPER-FIRST half, which is where that revert actually bites.
        //
        // `pending_hp_reading_move` is a 3-member set, so "one member dropped" is
        // the likeliest real regression, and a bare aggregate threshold misses it.
        // Measured on the sleeper-FIRST subset, with #1048 vs reverted:
        //
        //     SUBSTITUTE  404     FLAIL  900     REVERSAL  900     (with #1048)
        //
        // The reverted column is DELIBERATELY ABSENT. It read 37 / 60 / 53 here and
        // 234 / 363 / 307 in the with-#1048 column; both were unasserted and both had
        // drifted -- the exact failure the tally pin below closes. Replacing them with
        // fresh unasserted numbers would just restart the clock, and the reverted
        // column cannot be produced by this test at all: it needs a `defender_choice`
        // revert (`&Choice::default()` at the `identify_sleep_talk_called` call site),
        // which trips the WRONG-attribution assert long before this print. Reproduce it
        // deliberately if you need it; do not transcribe it into a comment.
        //
        // What matters for the gate is the live column against the floor of 150, and
        // that is measured every run by the print below. Scoping matters: on the
        // FULL matrix the revert leaves {FLAIL 142, REVERSAL 135, SUBSTITUTE 50},
        // an 8-branch margin, and it degrades predictably -- sleeper-second agrees
        // are INVARIANT under a `defender_choice` revert (the gate needs
        // `first_move`, false there), so every moveset added to that half raises the
        // without-#1048 floor while leaving the signal flat. Two or three more and
        // the gate would silently stop separating: the same silent-vacuity family
        // this test exists to prevent.
        for defender in ["SUBSTITUTE", "FLAIL", "REVERSAL"] {
            let key = format!("{defender}/first");
            let n = agree_by_defender.get(&key).copied().unwrap_or(0);
            assert!(
                n >= 150,
                "VACUOUS or #1048 REGRESSED for {key}: only {n} confident \
                 attributions (need >= 150). A per-defender collapse means the \
                 identifier lost the real defender choice for that move; the \
                 aggregate can still look healthy. All: {:?}",
                agree_by_defender
            );
        }
        let pct = agree * 100 / total_branches.max(1);
        assert!(
            pct > 60,
            "VACUOUS or #1048 REGRESSED: only {agree}/{total_branches} branches \
             ({pct}%) produced a confident attribution (refused multi \
             {multi_label_refused}, refused single {single_label_refused}). Expect \
             ~86% with #1048 in place; ~14% means the identifier lost the real \
             defender choice."
        );

        // THE FIX, pinned. Genuine ambiguity -- two callees whose branches are
        // byte-identical -- must no longer refuse. The transition is proven (both
        // regenerated exactly this tail) and the renderer names nobody, so there is
        // nothing unsafe to reject. Refusing it discarded the whole world, and on the
        // cluster that class was 49.5% of all world failures.
        //
        // Reverting the split in `render_move_phase` sends these back down the
        // refusing arm and fires this assertion.
        // Ambiguity splits THREE ways now, and both halves of the split are pinned.
        //
        // A renderable tail must be USABLE: refusing it discards a proven transition the
        // walk can express completely. An unrenderable tail must still REFUSE: the walk
        // drops boosts, statuses, heals and side conditions, so accepting one would hand
        // the fold an observation that contradicts the state. Review rejected the first
        // version of this change for exactly that, and these two assertions are what
        // stop either half drifting.
        assert!(
            multi_label_unattributed > 0,
            "VACUOUS: no ambiguous branch reached the usable arm, so this test cannot \
             show the split works at all."
        );
        // The fail-closed arm must also be EXERCISED, or the predicate is untested and
        // could be `|_| true` without anything noticing.
        // THE CORPUS IS NOW EXHAUSTED, so this guard has to move rather than be weakened.
        //
        // It used to assert `multi_label_refused > 0`, to stop
        // `ambiguous_tail_is_fully_renderable` degenerating to "accept everything" unnoticed.
        // That was the right guard while the corpus still contained a refusable ambiguity.
        // It no longer does: #1131 rendered the `boost` family (16 -> 6) and the substitute
        // break closes the last 6 (6 -> 0), so every ambiguous tail this corpus produces is
        // fully expressible. Keeping `> 0` would force either a weakened assertion or a
        // fabricated fixture, and both are worse than moving the guard.
        //
        // WHERE THE FAIL-CLOSED ARM IS EXERCISED NOW, so this is a relocation and not a loss:
        //   * `the_renderable_allowlist_is_exactly_what_it_was` asserts, for one representative
        //     of EVERY still-blocked family, that `ambiguous_tail_is_fully_renderable` returns
        //     FALSE. An "accept everything" predicate fails there on all of them.
        //   * `the_sleeptalk_refusal_subcases_without_moving_the_lossy_contract` drives a real
        //     Mean Look/Spider Web ambiguity end to end and requires it to refuse under
        //     `volatile`. It used to use Recover/Soft-Boiled and refuse under `heal`; the
        //     direct-self-heal change admits exactly that shape, so the guard had to move to a
        //     family that still blocks -- a knock-on the heal work predicted in advance rather
        //     than discovered when the suite went red.
        // The pinned tuple below still catches any drift, in both directions: a refusal
        // REAPPEARING moves the 0 as loudly as a usable one moving the 237.
        assert_eq!(
            multi_label_refused, 0,
            "an ambiguous refusal reappeared in a corpus that is now fully renderable -- \
             see the relocation note above before updating this number"
        );
        // ...and the two must partition the ambiguous population exactly.
        assert_eq!(
            multi_label_unattributed + multi_label_refused,
            total_branches - agree,
            "ambiguous branches lost: {multi_label_unattributed} usable + \
             {multi_label_refused} refused != {} unattributed",
            total_branches - agree
        );

        // PRINT BEFORE THE PIN BELOW, deliberately. The pin's own message tells you to
        // decide whether a change was intended, and that decision needs the
        // per-defender map -- which an `assert_eq!` placed first would suppress,
        // because a panic never reaches this line.
        println!(
            "#1048 attribution: branches {total_branches}  agree {agree} ({pct}%)  WRONG 0  \
             ambiguous-USABLE {multi_label_unattributed}  ambiguous-UNRENDERABLE \
             {multi_label_refused}  refused-single {single_label_refused}  \
             per-defender {agree_by_defender:?}"
        );

        // PIN THE TALLIES, not just their shape. Everything above asserts the partition
        // is non-vacuous and exact -- and both hold for ANY split of the same total, so
        // a vendored-engine or renderer change can move these numbers with the suite
        // green. Demonstrated: narrowing the renderability allowlist so `Damage` stops
        // qualifying gives 20 usable / 217 unrenderable, and every other assertion in
        // this test still passes.
        //
        // They are quoted outside this repo -- the deploy campaign's fallback ledger
        // states them as the measured effect of #1070 -- so drift is a reporting defect
        // elsewhere, not merely a test update.
        //
        // If you are here because this failed: that is the signal working. The
        // breakdown printed above says WHERE it moved. Decide whether the change was
        // intended, then update these values AND the ledger's row-1 disposition
        // together. Do not update one without the other.
        assert_eq!(
            (total_branches, agree, multi_label_unattributed, multi_label_refused),
            // MOVED DELIBERATELY by the immobilizer-marker change: branches 2614 -> 2720 and
            // agree 2377 -> 2483, both +106, with usable and unrenderable UNCHANGED at 237/0
            // and WRONG still 0. This is the change's own measurement and the reason it is
            // recorded here rather than only in the PR.
            //
            // MECHANISM, verified by reverting just the paralysis push and watching the pin
            // go back to 2614: several callees in this matrix (`bodyslam`, `thunder`)
            // paralyse the DEFENDER, who then rolls full paralysis on its own ply. Its
            // fully-paralyzed branch was an EMPTY delta, byte-identical to its same-delta
            // sibling, so `combine_duplicate_instructions` MERGED the two into one branch --
            // which is precisely why the renderer could not tell "no move happened" from "a
            // move happened and changed nothing". Marking the immobilizer separates them:
            // +106 branches, and all 106 land in `agree` because each now carries its own
            // exact attribution. Nothing moved OUT of agree, which is the claim that matters.
            //
            // The deploy campaign's fallback ledger quotes the OLD pair. That row needs the
            // same update; it lives outside this repo, so this comment is the handoff.
            //
            // 221 -> 231 -> 237 usable, 16 -> 6 -> 0 unrenderable, across two earlier
            // changes: #1131
            // rendered the ten `[Boost]` tails, and the substitute break closes the last six
            // `[DamageSubstitute, RemoveVolatileStatus]`. `branches`, `agree` and WRONG were
            // UNCHANGED throughout those two, which is the claim that matters -- no attribution moved,
            // only the refuse-versus-count decision.
            //
            // `ambiguous_unrenderable` is therefore CLOSED for this corpus. It is not closed in
            // production: the corpus contains only the two shapes above, and the era-59 family
            // split exists precisely because the reachable surface is wider than the corpus
            // that ranked it.
            (2720, 2483, 237, 0),
            "the Sleep Talk attribution oracle moved; see the per-defender breakdown \
             printed above, and the comment here on what else must be updated."
        );
    }

    /// The refactor must not change WHICH branches are refused.
    ///
    /// `ambiguous_tail_is_fully_renderable` was an inline `matches!` pair and is now
    /// defined as `unrenderable_tail_families(tail, attacker).is_empty()`. That is only a
    /// refactor if the admitted set is byte-identical -- widening it by one variant
    /// stops refusing a class of worlds, which is a behaviour change to the largest
    /// failure class in the program, and narrowing it starts refusing worlds that
    /// already worked. Neither would fail any other test in this file.
    ///
    /// Pinned in BOTH directions, and this time the claim is true. The first version said
    /// "the six admitted shapes must render, and one representative of every blocked
    /// family must not" while testing FOUR of six and FOUR of fourteen. Review found four
    /// allowlist mutations that left all 376 tests green:
    ///
    /// * `Heal(_) => None` admits an explicit HP INCREASE -- exactly the C52-mirror defect
    ///   this predicate exists for, and the variant the doc block spends 11 lines on had
    ///   no representative at all.
    /// * The volatile family `=> None` lets an unrendered `-start`/`-end` reach the fold.
    /// * `Switch(_) => Some(..)` starts refusing Roar/Whirlwind ambiguities that worked.
    /// * `ChangeDamageDealt* => Some(..)` starts refusing Counter/Mirror Coat tails.
    ///
    /// The oracle test is only a partial backstop and is blind to SINGLE-family
    /// widenings: widening `substitute` alone stays green because `RemoveVolatileStatus`
    /// still blocks those tails, and widening `volatile` alone stays green because
    /// `DamageSubstitute` does. So coverage has to be exhaustive here, not sampled.
    #[test]
    fn the_renderable_allowlist_is_exactly_what_it_was() {
        // ALL SIX admitted shapes. Three were missing before, so narrowing mutations on
        // them passed.
        let admitted: Vec<Instruction> = vec![
            Instruction::Damage(DamageInstruction {
                side_ref: SideReference::SideOne,
                damage_amount: 10,
            }),
            // Zero is a no-op and was admissible before; `>= 0`, not `> 0`.
            Instruction::Damage(DamageInstruction {
                side_ref: SideReference::SideOne,
                damage_amount: 0,
            }),
            Instruction::Switch(SwitchInstruction {
                side_ref: SideReference::SideOne,
                previous_index: PokemonIndex::P0,
                next_index: PokemonIndex::P1,
            }),
            Instruction::SetLastUsedMove(SetLastUsedMoveInstruction {
                side_ref: SideReference::SideOne,
                last_used_move: LastUsedMove::None,
                previous_last_used_move: LastUsedMove::None,
            }),
            Instruction::ChangeDamageDealtDamage(ChangeDamageDealtDamageInstruction {
                side_ref: SideReference::SideOne,
                damage_change: 5,
            }),
            Instruction::ChangeDamageDealtMoveCatagory(ChangeDamageDealtMoveCategoryInstruction {
                side_ref: SideReference::SideOne,
                move_category: MoveCategory::Physical,
                previous_move_category: MoveCategory::Status,
            }),
            Instruction::ToggleDamageDealtHitSubstitute(
                ToggleDamageDealtHitSubstituteInstruction {
                    side_ref: SideReference::SideOne,
                },
            ),
            // NEWLY ADMITTED, and the reason this test exists. The unnamed-callee walk now
            // emits `|-boost|`/`|-unboost|`, so a bare `[Boost]` tail is fully expressible and
            // must stop refusing worlds. Measured on the attribution oracle: usable 221 -> 231,
            // unrenderable 16 -> 6, with zero change to `agree` or WRONG. Moving an arm into
            // the admitted set is a BEHAVIOUR change, which is precisely why this test had to
            // be edited by hand rather than quietly widened.
            Instruction::Boost(BoostInstruction {
                side_ref: SideReference::SideOne,
                stat: PokemonBoostableStat::Defense,
                amount: 1,
            }),
            // NEWLY ADMITTED, second batch. The walk now emits
            // `|-activate|{ident}|Substitute|[damage]` and, for the SUBSTITUTE volatile only,
            // `|-end|{ident}|Substitute`. That closes the oracle's last 6 of 16 refusals, all
            // `[DamageSubstitute, RemoveVolatileStatus]`: usable 231 -> 237, unrenderable
            // 6 -> 0, again with zero change to `agree` or WRONG.
            Instruction::DamageSubstitute(DamageInstruction {
                side_ref: SideReference::SideOne,
                damage_amount: 20,
            }),
            // `RemoveVolatileStatus(SUBSTITUTE)` is deliberately NOT in this list. It is
            // admitted only when a same-side `DamageSubstitute` precedes it in the tail, so it
            // is not an unconditionally-renderable INSTRUCTION and belongs in the tail-level
            // pin below rather than here. Review found the first version of this change
            // admitting it unconditionally, which made a switch-out volatile cleanup render a
            // phantom `|-end|` and be searched where it used to refuse.
        ];
        for (index, instruction) in admitted.iter().enumerate() {
            assert_eq!(
                unrenderable_family_at(&admitted, index, SideReference::SideOne),
                None,
                "{instruction:?} was admitted by the previous allowlist and must stay \
                 admitted -- widening or narrowing this set changes which worlds refuse"
            );
        }
        assert!(
            ambiguous_tail_is_fully_renderable(&admitted, SideReference::SideOne),
            "a tail built only from admitted instructions must be fully renderable"
        );

        // THE TAIL-PAIRING GUARD, pinned in both directions. `RemoveVolatileStatus(SUBSTITUTE)`
        // is a narrated break in one position and a silent switch-out cleanup in another, and
        // the first version of this change could not tell them apart.
        let hit = Instruction::DamageSubstitute(DamageInstruction {
            side_ref: SideReference::SideOne,
            damage_amount: 20,
        });
        let removal = Instruction::RemoveVolatileStatus(RemoveVolatileStatusInstruction {
            side_ref: SideReference::SideOne,
            volatile_status: PokemonVolatileStatus::SUBSTITUTE,
        });
        let switch = Instruction::Switch(SwitchInstruction {
            side_ref: SideReference::SideOne,
            previous_index: PokemonIndex::P0,
            next_index: PokemonIndex::P1,
        });

        // A real break: hit then removal. Showdown narrates `|-end|`, so this is renderable.
        assert!(
            ambiguous_tail_is_fully_renderable(&[hit.clone(), removal.clone()], SideReference::SideOne),
            "a `DamageSubstitute` followed by the substitute removal IS a break and must be \
             renderable -- this is the whole point of the change"
        );

        // A PHAZE: removal then switch, no hit. `remove_volatile_statuses_on_switch` emits this
        // on every non-Baton-Pass switch-out and Showdown clears volatiles with
        // `this.volatiles = {}` WITHOUT running `onEnd`, so it emits no line at all. Rendering
        // `|-end|` here is a phantom, and searching the world against a protocol log with an
        // extra line is the same defect class as one missing a line.
        assert_eq!(
            unrenderable_family_at(&[removal.clone(), switch.clone()], 0, SideReference::SideOne),
            Some("volatile"),
            "a substitute removal with NO preceding same-side `DamageSubstitute` is a \
             switch-out cleanup, not a break, and must stay blocked"
        );
        assert!(
            !ambiguous_tail_is_fully_renderable(&[removal.clone(), switch], SideReference::SideOne),
            "a phaze tail must keep REFUSING -- review reproduced this rendering a phantom \
             `|-end|` end to end through `render_branch_events`"
        );

        // ORDER is load-bearing, not just presence: the hit must come BEFORE the removal.
        assert_eq!(
            unrenderable_family_at(&[removal.clone(), hit.clone()], 0, SideReference::SideOne),
            Some("volatile"),
            "a removal that PRECEDES the hit is not a break that hit caused"
        );

        // SIDE is load-bearing too: the sub that broke must belong to the mon whose volatile
        // is being removed, or the pairing invents a break across sides.
        let other_side_hit = Instruction::DamageSubstitute(DamageInstruction {
            side_ref: SideReference::SideTwo,
            damage_amount: 20,
        });
        assert_eq!(
            // index 1: the REMOVAL is the instruction under test here, not the hit.
            unrenderable_family_at(&[other_side_hit, removal], 1, SideReference::SideOne),
            Some("volatile"),
            "a `DamageSubstitute` on the OTHER side does not make this removal a break"
        );

        // ONE REPRESENTATIVE PER BLOCKED FAMILY -- 14 distinct families in this vec, and the
        // full sixteen only counting `blocked_in_tail` below. `Heal` is first because it
        // is the variant review found most likely to be mistakenly admitted -- the whole
        // C52-mirror doc block is about it -- and it previously had no representative at
        // all, so `Heal(_) => None` left every test green.
        let blocked: Vec<(Instruction, &str)> = vec![
            (
                // The DEFENDER's side, deliberately. An attacker-side heal with no damage
                // to the defender is now a direct self-heal and RENDERS; this one is an
                // absorb ability, whose Showdown line carries `[from] ability: X|[of] ..`,
                // which the walk cannot construct without knowing the ability. So the family
                // is partially closed and this representative is what remains of it.
                Instruction::Heal(HealInstruction {
                    side_ref: SideReference::SideTwo,
                    heal_amount: 40,
                }),
                "heal_defender",
            ),
            (
                // The load-bearing sign case: `Damage` is SIGNED, and negative is the
                // engine's spelling for a heal (Pain Split). Same family as `Heal`
                // because what the walk drops is identical -- an HP increase.
                Instruction::Damage(DamageInstruction {
                    side_ref: SideReference::SideOne,
                    damage_amount: -130,
                }),
                "heal_paindmg",
            ),
            (
                // NOT a boost family any more -- see the admitted list above. Kept here only
                // as the contrast for `statrecalc`, which looks like a boost and is not.
                // The engine emits `Change<Stat>` from `recalculate_stats` on
                // mega/forme/transform and it carries no `-boost` line.
                Instruction::ChangeAttack(ChangeStatInstruction {
                    side_ref: SideReference::SideOne,
                    amount: 20,
                }),
                "statrecalc",
            ),
            (
                Instruction::ChangeStatus(ChangeStatusInstruction {
                    side_ref: SideReference::SideOne,
                    pokemon_index: PokemonIndex::P0,
                    old_status: PokemonStatus::NONE,
                    new_status: PokemonStatus::SLEEP,
                }),
                "status",
            ),
            (
                // NOT `status`: the named path renders these as `|cant|...|slp`.
                Instruction::SetSleepTurns(SetSleepTurnsInstruction {
                    side_ref: SideReference::SideOne,
                    pokemon_index: PokemonIndex::P0,
                    new_turns: 3,
                    previous_turns: 0,
                }),
                "sleepcounter",
            ),
            (
                // NOT `sleepcounter`, though it looks like it: `SetRestTurns` has no render
                // arm. The engine emits it right after `ChangeStatus(-> SLEEP)` when Rest is
                // used, so the lines belong to that ChangeStatus and the accompanying Heal.
                // Mutation showed moving it into `sleepcounter` survived until this existed.
                Instruction::SetRestTurns(SetSleepTurnsInstruction {
                    side_ref: SideReference::SideOne,
                    pokemon_index: PokemonIndex::P0,
                    new_turns: 3,
                    previous_turns: 0,
                }),
                "silent",
            ),
            (
                // Was `identity`. The named path's own arm renders a single `|-transform|`
                // and applies ChangeType/ChangeAbility/FormeChange silently, so only
                // `ChangeItem` in that old group named real missing work.
                Instruction::ChangeType(ChangeType {
                    side_ref: SideReference::SideOne,
                    new_types: (PokemonType::NORMAL, PokemonType::TYPELESS),
                    old_types: (PokemonType::WATER, PokemonType::TYPELESS),
                }),
                "silent",
            ),
            (
                // THE GUARD ON THE ADMISSION ABOVE, and the reason it is written as a match
                // GUARD on `volatile_status` rather than on the variant. `RemoveVolatileStatus`
                // is admitted for SUBSTITUTE only; the walk has no line for Leech Seed,
                // Confusion, Encore or any other volatile, so those must keep refusing. Widen
                // the admission to the whole variant and this representative fails -- which is
                // the C52-mirror defect caught at compile-adjacent cost instead of in a
                // campaign. `DamageSubstitute` USED to sit here under "substitute"; it is now
                // in the admitted list above.
                Instruction::RemoveVolatileStatus(RemoveVolatileStatusInstruction {
                    side_ref: SideReference::SideOne,
                    volatile_status: PokemonVolatileStatus::LEECHSEED,
                }),
                "volatile",
            ),
            (
                // NOT `substitute`. Creation emits this with `Damage` +
                // `ApplyVolatileStatus(SUBSTITUTE)`, break with
                // `RemoveVolatileStatus(SUBSTITUTE)` -- so a `volatile` or `substitute`
                // companion is ALWAYS present and the line to emit is theirs. Filed under
                // `substitute` it made a Substitute-CREATION tail report
                // `substitute+volatile` when only the `volatile` line is missing: a false
                // positive in the bucket most likely to be ranked first. Mutation showed
                // this misfiling survived until this representative existed.
                Instruction::ChangeSubstituteHealth(ChangeSubsituteHealthInstruction {
                    side_ref: SideReference::SideOne,
                    health_change: -20,
                }),
                "silent",
            ),
            (
                Instruction::ApplyVolatileStatus(ApplyVolatileStatusInstruction {
                    side_ref: SideReference::SideOne,
                    volatile_status: PokemonVolatileStatus::SUBSTITUTE,
                }),
                "volatile",
            ),
            (
                Instruction::ChangeSideCondition(ChangeSideConditionInstruction {
                    side_ref: SideReference::SideOne,
                    side_condition: PokemonSideCondition::Spikes,
                    amount: 1,
                }),
                "sidecondition",
            ),
            (
                Instruction::DecrementWeatherTurnsRemaining,
                "weather",
            ),
            (
                Instruction::DecrementTerrainTurnsRemaining,
                "field",
            ),
            (
                Instruction::DisableMove(DisableMoveInstruction {
                    side_ref: SideReference::SideOne,
                    move_index: PokemonMoveIndex::M0,
                }),
                "moveslot",
            ),
            (
                Instruction::ChangeWish(ChangeWishInstruction {
                    side_ref: SideReference::SideOne,
                    wish_amount_change: 50,
                }),
                "silent",
            ),
            (
                Instruction::SetFutureSight(SetFutureSightInstruction {
                    side_ref: SideReference::SideOne,
                    pokemon_index: PokemonIndex::P0,
                    previous_pokemon_index: PokemonIndex::P0,
                }),
                "silent",
            ),
            (
                Instruction::ChangeItem(ChangeItemInstruction {
                    side_ref: SideReference::SideOne,
                    current_item: Items::NONE,
                    new_item: Items::LEFTOVERS,
                }),
                "item",
            ),
            (
                // NOT `silent`, which is where every other line-less instruction goes. The
                // attract marker HAS a public line -- `|cant|<ident>|Attract` -- emitted by
                // `render_move_phase`'s own arm, which returns before this walk can start.
                // A marker reaching the classifier therefore means the renderer's statement
                // order broke, not that a family needs a new protocol line, and `silent`
                // reads as "zero renderer work, just an allowlist audit". Mutation: filing
                // it under `silent` passes every other test in this file.
                Instruction::MoveImmobilized(MoveImmobilizedInstruction {
                    side_ref: SideReference::SideOne,
                    reason: ImmobilizeReason::Attract,
                }),
                "immobilizer",
            ),
            (
                Instruction::ToggleBatonPassing(ToggleBatonPassingInstruction {
                    side_ref: SideReference::SideOne,
                }),
                "silent",
            ),
            (
                // PAIN SPLIT at the boundary: a heal-direction `Damage` of just -1, on the
                // ATTACKER's side (SideOne is the attacker this loop passes). Pins that the
                // sign test is `< 0`, not a magnitude threshold.
                Instruction::Damage(DamageInstruction {
                    side_ref: SideReference::SideOne,
                    damage_amount: -1,
                }),
                "heal_paindmg",
            ),
            (
                // LIQUID OOZE: a NEGATIVE `Heal`, which the named path renders as
                // `-damage`, not `-heal`. Blocked standalone, so it belongs here and
                // not in `blocked_in_tail`.
                Instruction::Heal(HealInstruction {
                    side_ref: SideReference::SideOne,
                    heal_amount: -40,
                }),
                "heal_liquidooze",
            ),
            (
                // A ZERO-amount `Heal` is a gen3 BRANCH MARKER -- Protect-blocked, or a
                // full-HP absorb no-op -- not a heal. The bare `heal` remainder
                // deliberately has NO representative: it is unreachable, which is why the
                // token is deregistered from UNRENDERABLE_FAMILY_ORDER.
                Instruction::Heal(HealInstruction {
                    side_ref: SideReference::SideOne,
                    heal_amount: 0,
                }),
                "heal_zero_marker",
            ),
            (
                // NOT `pp`: there is no `-pp` line in the protocol. `silent` says the
                // truth -- no public line on any path.
                Instruction::DecrementPP(DecrementPPInstruction {
                    side_ref: SideReference::SideOne,
                    move_index: PokemonMoveIndex::M0,
                    amount: 1,
                }),
                "silent",
            ),
        ];
        for (instruction, family) in &blocked {
            assert_eq!(
                unrenderable_family_at(std::slice::from_ref(instruction), 0, SideReference::SideOne),
                Some(*family),
                "{instruction:?} must be blocked and classified as {family:?}"
            );
            assert!(
                !ambiguous_tail_is_fully_renderable(std::slice::from_ref(instruction), SideReference::SideOne),
                "{instruction:?} carries an effect the walk drops, so its tail is not \
                 fully renderable"
            );
        }

        // TAIL-CONTEXT families, which a lone instruction cannot represent because the
        // classifier's answer depends on what surrounds it. `boost` is one: a move's own stat
        // change is renderable, the switch-out RESET is not, and the only difference is a
        // later same-side `Switch`.
        //
        // A separate list rather than forced into `blocked` above, because that loop asserts
        // on `from_ref(instruction), 0` and a lone `Boost` legitimately answers `None`.
        // Collapsing the two would mean weakening that loop or writing a representative that
        // lies about its own family.
        let blocked_in_tail: Vec<(Vec<Instruction>, usize, &str)> = vec![(
            vec![
                Instruction::Boost(BoostInstruction {
                    side_ref: SideReference::SideOne,
                    stat: PokemonBoostableStat::Attack,
                    amount: -2,
                }),
                Instruction::Switch(SwitchInstruction {
                    side_ref: SideReference::SideOne,
                    previous_index: PokemonIndex::P0,
                    next_index: PokemonIndex::P1,
                }),
            ],
            0,
            "boost",
        ),
        (
            // DRAIN SHAPE. The `Heal` ALONE is admitted -- it is a direct self-heal --
            // so this belongs here rather than in `blocked`, and the loop's
            // "same instruction without the tail must stay admitted" assertion is the
            // one that proves the discrimination is a property of the TAIL.
            vec![
                Instruction::Damage(DamageInstruction {
                    side_ref: SideReference::SideTwo,
                    damage_amount: 60,
                }),
                Instruction::Heal(HealInstruction {
                    side_ref: SideReference::SideOne,
                    heal_amount: 30,
                }),
            ],
            1,
            "heal_drain_or_shellbell",
        ),
        (
            // DRAIN AGAINST A SUBSTITUTE. Review's mutation DELETED the
            // `DamageSubstitute` arm from `tail_damages_the_foe` and the whole suite
            // stayed GREEN -- while that deletion flips this tail from refused to
            // ADMITTED, making the walk emit a bare `|-heal|` for a drain. That is
            // precisely the fabricated-`[from]`-tag harm `heal_is_a_direct_self_heal`
            // exists to prevent, and it was the one clause the refactor's doc claimed to
            // protect with nothing testing it.
            vec![
                Instruction::DamageSubstitute(DamageInstruction {
                    side_ref: SideReference::SideTwo,
                    damage_amount: 25,
                }),
                Instruction::Heal(HealInstruction {
                    side_ref: SideReference::SideOne,
                    heal_amount: 30,
                }),
            ],
            1,
            "heal_drain_or_shellbell",
        )];
        // THE HEAL PREDICATE, pinned in every direction it discriminates on. This family
        // is only PARTIALLY closed, and each clause is what keeps a mis-tagged `-heal` -- which
        // FABRICATES a belief in the fold -- out of a searched world.
        let self_heal = Instruction::Heal(HealInstruction {
            side_ref: SideReference::SideOne,
            heal_amount: 40,
        });
        let foe_damage = Instruction::Damage(DamageInstruction {
            side_ref: SideReference::SideTwo,
            damage_amount: 30,
        });
        // 1. Direct self-heal: renders bare. The one admitted shape.
        assert_eq!(
            unrenderable_family_at(std::slice::from_ref(&self_heal), 0, SideReference::SideOne),
            None,
            "an attacker-side heal with no foe damage is a direct healing move"
        );
        // 2. DRAIN: same heal, but the tail damages the foe. Needs `[from] drain|[of] ..`.
        assert_eq!(
            unrenderable_family_at(&[foe_damage, self_heal.clone()], 1, SideReference::SideOne),
            Some("heal_drain_or_shellbell"),
            "damage to the foe plus a heal on the attacker is DRAIN-SHAPED, not a direct \
             heal -- and the tail alone cannot tell drain from a Shell Bell holder"
        );
        // 3. ABSORB ABILITY: the heal is on the defender. Needs `[from] ability: X`.
        assert_eq!(
            unrenderable_family_at(std::slice::from_ref(&self_heal), 0, SideReference::SideTwo),
            Some("heal_defender"),
            "a heal on the DEFENDER is an absorb ability, not a direct heal"
        );
        // ARM ORDER, pinned. A heal on the DEFENDER *with* foe damage in the tail must
        // bucket as `heal_defender`, not as drain -- an absorb ability's line is an
        // ABILITY reveal whatever else the tail did. NOT in `blocked_in_tail`, because
        // that loop also asserts the instruction alone stays ADMITTED and a defender heal
        // is refused standalone. Review's mutation swapped the two checks and the suite
        // stayed green, since no other fixture carries both.
        assert_eq!(
            unrenderable_family_at(
                &[
                    Instruction::Damage(DamageInstruction {
                        side_ref: SideReference::SideTwo,
                        damage_amount: 60,
                    }),
                    Instruction::Heal(HealInstruction {
                        side_ref: SideReference::SideTwo,
                        heal_amount: 30,
                    }),
                ],
                1,
                SideReference::SideOne
            ),
            Some("heal_defender"),
            "a defender heal stays an absorb ability even when the tail damages the foe"
        );

        // ZERO-DAMAGE CONTROL, pinning `> 0` rather than `>= 0` in `tail_damages_the_foe`.
        // Review mutated that comparison and the whole suite stayed GREEN, while the change
        // flips this tail from ADMITTED to refused -- an admission boundary with nothing
        // watching. The two drain fixtures use 60 and 25, which cannot discriminate.
        assert_eq!(
            unrenderable_family_at(
                &[
                    Instruction::Damage(DamageInstruction {
                        side_ref: SideReference::SideTwo,
                        damage_amount: 0,
                    }),
                    Instruction::Heal(HealInstruction {
                        side_ref: SideReference::SideOne,
                        heal_amount: 30,
                    }),
                ],
                1,
                SideReference::SideOne
            ),
            None,
            "a ZERO-amount foe Damage is not damage, so this stays a direct self-heal"
        );

        // 4. LIQUID OOZE: a negative heal, which the named path renders as `-damage`.
        assert_eq!(
            unrenderable_family_at(
                &[Instruction::Heal(HealInstruction {
                    side_ref: SideReference::SideOne,
                    heal_amount: -40,
                })],
                0,
                SideReference::SideOne
            ),
            Some("heal_liquidooze"),
            "a NEGATIVE heal is Liquid Ooze and renders as damage, not as a heal"
        );

        // CROSS-SIDE control. Review's mutation replaced the predicate's
        // `switch.side_ref == boost.side_ref` with `true` and SURVIVED the whole suite, because
        // every fixture above pairs SideOne with SideOne. A cross-side pair must stay ADMITTED:
        // side two switching out does not reset side one's boosts.
        let cross_side = vec![
            Instruction::Boost(BoostInstruction {
                side_ref: SideReference::SideOne,
                stat: PokemonBoostableStat::Attack,
                amount: -2,
            }),
            Instruction::Switch(SwitchInstruction {
                side_ref: SideReference::SideTwo,
                previous_index: PokemonIndex::P0,
                next_index: PokemonIndex::P1,
            }),
        ];
        assert_eq!(
            unrenderable_family_at(&cross_side, 0, SideReference::SideOne),
            None,
            "a `Switch` on the OTHER side does not reset this side's boosts, so the boost \
             stays renderable"
        );

        for (tail, index, family) in &blocked_in_tail {
            assert_eq!(
                unrenderable_family_at(tail, *index, SideReference::SideOne),
                Some(*family),
                "{tail:?} at {index} must be blocked as {family:?}"
            );
            assert!(
                !ambiguous_tail_is_fully_renderable(tail, SideReference::SideOne),
                "{tail:?} carries an effect the walk drops, so it is not fully renderable"
            );
            // ...and the SAME instruction WITHOUT the tail context must stay admitted, or the
            // narrowing is a blanket revert of #1131 wearing a guard's clothes.
            assert_eq!(
                unrenderable_family_at(std::slice::from_ref(&tail[*index]), 0, SideReference::SideOne),
                None,
                "{:?} ALONE is admitted, so the refusal is a property of the TAIL",
                tail[*index]
            );
            // ...and the COMPOSED SLUG must name the family, not just the raw classifier.
            // Review's mutation dropped `"boost"` from UNRENDERABLE_FAMILY_ORDER *and* from
            // the order pin together, and SURVIVED: the family then degrades to
            // `unclassified` through `registered_family_or_unclassified`, which is exactly the
            // outcome reinstating the token is supposed to prevent. Asserting the raw family
            // cannot see that, because the degradation happens one layer up.
            assert!(
                ambiguous_unrenderable_slug(tail, SideReference::SideOne).ends_with(&format!(":{family}")),
                "the composed slug must name {family:?} rather than degrade to \
                 `unclassified`: {}",
                ambiguous_unrenderable_slug(tail, SideReference::SideOne)
            );
        }

        // EVERY family in the order list has a representative above, except the
        // `unclassified` escape hatch which no instruction maps to. Without this, adding
        // a family and forgetting to cover it silently reopens the gap review found.
        let covered: Vec<&str> = blocked
            .iter()
            .map(|(_, f)| *f)
            .chain(blocked_in_tail.iter().map(|(_, _, f)| *f))
            .collect();
        for family in UNRENDERABLE_FAMILY_ORDER {
            if *family == "unclassified" {
                continue;
            }
            assert!(
                covered.contains(family),
                "family {family:?} is in UNRENDERABLE_FAMILY_ORDER but has no \
                 representative instruction in this test, so a mutation admitting it \
                 would pass"
            );
        }
    }

    /// The slug names the blocking families, in FIXED order, deduplicated.
    ///
    /// Era 59 measured `ambiguous_unrenderable` at 8,149 world failures as one opaque
    /// key, so nothing said whether closing it needed `-boost`, `-status`, `-heal` or
    /// the Substitute family. These three properties are what make the replacement key
    /// summable: a stable order (encounter order would split one composition across
    /// several keys), dedup (two Boosts are one family), and the contract-tag prefix.
    #[test]
    fn the_unrenderable_slug_is_stable_deduplicated_and_tag_prefixed() {
        // `statrecalc`, not `boost`: Boost is now RENDERED and therefore admitted, so it can
        // no longer stand in for "a blocked family". `Change<Stat>` still carries no line.
        let boost = Instruction::ChangeAttack(ChangeStatInstruction {
            side_ref: SideReference::SideOne,
            amount: 20,
        });
        let second_boost = Instruction::ChangeDefense(ChangeStatInstruction {
            side_ref: SideReference::SideTwo,
            amount: 15,
        });
        let status = Instruction::ChangeStatus(ChangeStatusInstruction {
            side_ref: SideReference::SideOne,
            pokemon_index: PokemonIndex::P0,
            old_status: PokemonStatus::NONE,
            new_status: PokemonStatus::SLEEP,
        });

        // Two Boosts are ONE family token.
        assert_eq!(
            unrenderable_tail_families(&[boost.clone(), second_boost], SideReference::SideOne),
            vec!["statrecalc"],
            "repeated instructions in the same family must collapse to one token"
        );

        // FIXED order, not encounter order. `boost` precedes `status` in
        // UNRENDERABLE_FAMILY_ORDER, so both instruction orders give the same slug --
        // otherwise one composition splits across two keys and neither sums.
        let forward = ambiguous_unrenderable_slug(&[boost.clone(), status.clone()], SideReference::SideOne);
        let reversed = ambiguous_unrenderable_slug(&[status, boost.clone()], SideReference::SideOne);
        assert_eq!(forward, reversed, "slug must not depend on instruction order");
        assert_eq!(
            forward,
            "sleeptalk_called_unidentified:ambiguous_unrenderable:statrecalc+status"
        );

        // The prefix is the CONTRACT tag. `engine_transition_differential.py` matches
        // the bare tag exactly, and `mark_attribution_unsafe_subcase` asserts this
        // relationship, so a slug that lost the prefix would panic in production.
        assert!(ambiguous_unrenderable_slug(&[boost], SideReference::SideOne).starts_with(SLEEPTALK_LOSSY_TAG));
    }

    /// Every token the classifier can emit must be ORDERABLE and REGISTERED.
    ///
    /// `unrenderable_tail_families` sorts by position in `UNRENDERABLE_FAMILY_ORDER` and
    /// falls back to `usize::MAX` for an unknown token. That fallback is deliberate --
    /// losing slug ordering is not worth aborting a search over -- but it means a family
    /// added to the classifier and forgotten in the order list degrades SILENTLY to
    /// encounter-order-dependent keys. And `assert_subcase_vocabulary` would then panic
    /// in production, on the release wheel, where the assert is compiled in.
    ///
    /// Limit stated honestly: this checks the families a representative instruction can
    /// reach, not all 50 engine variants. The exhaustive `match` in `unrenderable_family`
    /// is what makes a NEW variant a compile error; this is what makes a new TOKEN a
    /// test failure.
    #[test]
    fn every_classifier_token_is_registered_and_orderable() {
        for family in UNRENDERABLE_FAMILY_ORDER {
            assert_eq!(
                UNRENDERABLE_FAMILY_ORDER
                    .iter()
                    .filter(|other| *other == family)
                    .count(),
                1,
                "{family:?} appears twice in UNRENDERABLE_FAMILY_ORDER, so sort position \
                 depends on which copy `position` finds"
            );
        }
        let reachable = [
            Instruction::ChangeAttack(ChangeStatInstruction {
                side_ref: SideReference::SideOne,
                amount: 20,
            }),
            Instruction::Damage(DamageInstruction {
                side_ref: SideReference::SideOne,
                damage_amount: -1,
            }),
            Instruction::ChangeStatus(ChangeStatusInstruction {
                side_ref: SideReference::SideOne,
                pokemon_index: PokemonIndex::P0,
                old_status: PokemonStatus::NONE,
                new_status: PokemonStatus::SLEEP,
            }),
            // Was `DamageSubstitute`, which the walk now renders. A NON-substitute volatile
            // is the reachable blocked representative in its place.
            Instruction::RemoveVolatileStatus(RemoveVolatileStatusInstruction {
                side_ref: SideReference::SideOne,
                volatile_status: PokemonVolatileStatus::LEECHSEED,
            }),
        ];
        for instruction in &reachable {
            let family = unrenderable_family_at(std::slice::from_ref(instruction), 0, SideReference::SideOne)
                .expect("these representatives are all blocked families");
            assert!(
                UNRENDERABLE_FAMILY_ORDER.contains(&family),
                "{family:?} is emitted by the classifier but missing from \
                 UNRENDERABLE_FAMILY_ORDER, so its slug position is unstable and \
                 `assert_subcase_vocabulary` will panic on the release wheel"
            );
        }
    }

    /// An unregistered token must panic, not mint a 37th untraceable aggregate key.
    ///
    /// This is what replaces the `&'static str` bound on `subcase`. That bound made an
    /// unbounded key set unrepresentable; relaxing it to allow a COMPOSED slug would
    /// have given that protection up for nothing if the vocabulary check did not
    /// actually reject. Note it is a plain `assert!`, so it holds on the `--release`
    /// campaign wheel too -- a `debug_assert!` would guard only `cargo test`.
    #[test]
    #[should_panic(expected = "unregistered token")]
    fn a_subcase_token_outside_the_vocabulary_is_refused() {
        assert_subcase_vocabulary(
            SLEEPTALK_LOSSY_TAG,
            "sleeptalk_called_unidentified:ambiguous_unrenderable:invented_family",
        );
    }

    /// The PAIRED-TAG assert, which nothing pinned.
    ///
    /// `mark_attribution_unsafe_subcase` asserts `subcase.starts_with(lossy_tag)`, and its
    /// own comment says why: a branch whose contract tag and measurement reason disagree
    /// changes which branches `engine_transition_differential.py` accepts, with nothing to
    /// notice. Review found the assert survives DELETION with all 376 tests green -- this
    /// PR added a `should_panic` for the new vocabulary check and left the older, more
    /// consequential one uncovered.
    #[test]
    #[should_panic(expected = "does not belong to lossy tag")]
    fn a_subcase_naming_a_different_class_is_refused() {
        let mut out = RenderedEvents::default();
        out.mark_attribution_unsafe_subcase(SLEEPTALK_LOSSY_TAG, "attract_empty_tail_ambiguous:miss");
    }

    /// The SIBLING paired-tag assert, in `mark_lossy_subcase`.
    ///
    /// Found while mutation-testing the one above: `mark_attribution_unsafe_subcase` and
    /// `mark_lossy_subcase` each assert the same tag/sub-case relationship, and
    /// neutralising the one in `mark_lossy_subcase` left every test green. Pre-existing
    /// gap, not introduced here, but it is the same assert guarding the same contract on
    /// the branch-USABLE path -- where a mismatched tag changes which branches the
    /// differential accepts without refusing anything, so nothing would be loud.
    #[test]
    #[should_panic(expected = "does not belong to lossy tag")]
    fn a_lossy_subcase_naming_a_different_class_is_refused() {
        let mut out = RenderedEvents::default();
        out.mark_lossy_subcase(SLEEPTALK_LOSSY_TAG, "attract_empty_tail_ambiguous:miss");
    }

    /// The cardinality ceiling quoted in `mark_attribution_unsafe_subcase`'s doc block,
    /// ENFORCED.
    ///
    /// That figure has been wrong THREE times -- "2^13 - 1 = 8,191, 14 entries" while the
    /// array held 13; then 18 entries after the heal split with the arithmetic unmoved;
    /// then 2^17 after deregistering `heal` had already taken 18 back to 17. Each time the
    /// error was treating a COUNT as prose, and each time the fix was a comment telling the
    /// next author to recount. Telling did not work. A test does.
    ///
    /// FOURTH move, and the first one that did not start as an error: the attract-marker
    /// patch appends `immobilizer`, taking 17 entries to 18 and the ceiling to 2^17 - 1.
    /// The figure is RECOUNTED from the array here rather than edited to match.
    #[test]
    fn the_cardinality_ceiling_matches_the_array() {
        let reachable = UNRENDERABLE_FAMILY_ORDER.len() - 1; // `unclassified` is unemittable
        assert_eq!(
            (UNRENDERABLE_FAMILY_ORDER.len(), reachable, 2usize.pow(reachable as u32) - 1),
            (18, 17, 131_071),
            "the order list changed size -- update THREE places in \
             `mark_attribution_unsafe_subcase`'s doc block (the `2^17 - 1 = 131,071` \
             figure, the `Note 17, not 18` line, and the `A 131k ceiling` sentence) plus \
             this test's own doc block"
        );
    }

    /// The full slug ORDER, not just one adjacent pair.
    ///
    /// `the_unrenderable_slug_is_stable_deduplicated_and_tag_prefixed` pins `boost` before
    /// `status`, which leaves 15 of 17 token positions free: review showed that swapping
    /// two adjacent tokens in `UNRENDERABLE_FAMILY_ORDER` silently changes emitted keys
    /// era-over-era with every test green. Cross-era comparability is the entire value of
    /// a stable slug, so the whole sequence is pinned.
    ///
    /// If you are DELIBERATELY reordering, update this list and say so in the commit --
    /// era N and era N+1 keys stop matching for any composition spanning the moved token.
    #[test]
    fn the_family_order_is_pinned_in_full() {
        assert_eq!(
            UNRENDERABLE_FAMILY_ORDER,
            &[
                "boost",
                "statrecalc",
                "status",
                "sleepcounter",
                // DELIBERATE change, per this test's own instruction to say so. Not a
                // reorder: the `heal` family is PARTITIONED into sub-cases and the bare
                // token is REMOVED. Unlike the "substitute" removal below, this one DOES move
                // real keys: era 61 measured 3,533 world failures under `heal`, and every
                // one of them now reports a sub-case instead. The sum across the five
                // tokens is what compares to the old bare count. The BARE token is gone
                // from this array: after the split nothing can emit it.
                "heal_paindmg",
                "heal_liquidooze",
                "heal_defender",
                "heal_drain_or_shellbell",
                "heal_zero_marker",
                // "substitute" removed by hand: its only producer, `DamageSubstitute`, is
                // now rendered. Every token AFTER it keeps its relative order, so no slug
                // that does not contain "substitute" changes -- and no slug can contain it.
                "volatile",
                "sidecondition",
                "weather",
                "field",
                "moveslot",
                "item",
                "silent",
                // DELIBERATE change, per this test's own instruction to say so. APPENDED,
                // not reordered: `immobilizer` goes last before the escape hatch, so every
                // token keeps its relative order and no existing slug changes. Its
                // producer -- the `MoveImmobilized` arm -- is structurally unreachable in
                // production, so the emitted-key volume is zero on both sides of the
                // boundary.
                "immobilizer",
                "unclassified",
            ],
            "the emitted slug order changed; era-over-era keys will not match for any \
             composition spanning a moved token"
        );
    }

    /// An unregistered family must degrade to a measurable key, NOT abort the worker.
    ///
    /// pyo3 maps a Rust panic to `PanicException`, which derives from `BaseException`
    /// precisely so it propagates past ordinary handlers, and `engine_search.py` catches
    /// only `except Exception`. So a vocabulary miss on the release wheel -- where this is
    /// a plain `assert!`, by design -- killed the campaign worker rather than producing a
    /// bad aggregate key. Strictly worse than the thing the assert prevents.
    ///
    /// `unrenderable_tail_families` now maps an unregistered token to `unclassified`,
    /// which is in both lists, making the assert unreachable from this path by
    /// construction.
    ///
    /// The DEGRADATION itself, reachable now that it lives behind a seam.
    ///
    /// Deleting the degradation used to leave the whole suite green. Not because the
    /// behaviour difference was unobservable -- it is severe, a dead campaign worker versus
    /// a measurable key -- but because no input could reach the `else` branch through the
    /// inlined form, since every family the classifier emits is registered. Review's fix
    /// was a seam, not a 50-variant enumeration: enumerating variants would have been
    /// exactly as vacuous, 50x longer, because every variant maps to a REGISTERED token.
    #[test]
    fn an_unregistered_family_degrades_to_unclassified() {
        // A deliberately short order list makes `boost` unregistered, which no real input
        // can achieve. This is the branch that stands between a future forgotten
        // registration and an aborted worker.
        assert_eq!(
            registered_family_or_unclassified("statrecalc", &["status", "heal"]),
            "unclassified"
        );
        // And a registered family passes through untouched.
        assert_eq!(
            registered_family_or_unclassified("statrecalc", UNRENDERABLE_FAMILY_ORDER),
            "statrecalc"
        );
        // `unclassified` must itself be registered, or the degradation would produce a
        // token that panics in `assert_subcase_vocabulary` -- turning the guard into the
        // very abort it exists to prevent.
        assert!(UNRENDERABLE_FAMILY_ORDER.contains(&"unclassified"));
        assert!(SUBCASE_VOCABULARY.contains(&"unclassified"));
        // Same for the Protect counter's token. It is registered DEFENSIVELY -- the only
        // caller goes through `mark_lossy_subcase`, which does not currently reach
        // `assert_subcase_vocabulary` -- and review showed that made the registration
        // deletable with all 423 tests green. An "unused token" cleanup would then arm a
        // production panic for whoever later closes that asymmetry, on a --release wheel,
        // where a pyo3 panic escapes `except Exception` and kills the campaign worker.
        // Pinned here rather than in a new test so the CI count floor does not move.
        //
        // DELIBERATELY REDUNDANT. `the_live_subcase_slugs_are_all_in_vocabulary` now runs the
        // same literal through `assert_subcase_vocabulary`, which SUBSUMES this membership
        // check -- deleting the entry fails both. Kept as defence in depth, and labelled so,
        // because the comment above justifies only this assert's PLACEMENT and review noted
        // it no longer explains its EXISTENCE.
        assert!(SUBCASE_VOCABULARY.contains(&"protect_marker_rendered"));
    }

    /// A marker is attributed to the side whose action it aborted.
    ///
    /// PINNED DIRECTLY because no production path can observe it. `instruction_side`'s two
    /// callers are the miss-inference predicate and the `NoneMatchedShape` diagnostic, and a
    /// marker is consumed and returned on before either runs -- so deleting the arm and
    /// letting the catch-all answer `None` leaves every end-to-end test green. Review found
    /// exactly that.
    ///
    /// `None` would mean "belongs to no side", which for a `|cant|` line's own instruction
    /// is the one answer that could let a future caller credit the line to the wrong
    /// Pokemon.
    #[test]
    fn a_marker_is_attributed_to_the_side_whose_action_it_aborted() {
        for side_ref in [SideReference::SideOne, SideReference::SideTwo] {
            for reason in [ImmobilizeReason::Attract, ImmobilizeReason::Paralysis] {
                assert_eq!(
                    instruction_side(&Instruction::MoveImmobilized(MoveImmobilizedInstruction {
                        side_ref,
                        reason,
                    })),
                    Some(side_ref),
                    "{side_ref:?}/{reason:?} must be attributable, not fall to the catch-all"
                );
            }
        }
    }

    /// `attribution_unsafe_label` dedupes AND sorts. RELOCATED, not deleted.
    ///
    /// This property was pinned end to end by
    /// `a_two_sided_refusal_keys_canonically_and_fits_the_python_seam` in the renderer
    /// suite, through a fixture where BOTH sides refused with different
    /// `attract_empty_tail_ambiguous` sub-cases. That class no longer exists -- the
    /// immobilizer markers closed it -- so the fixture cannot be rebuilt and the test
    /// could not be updated. Pinning the function directly is strictly stronger anyway:
    /// the old version depended on a state that happened to produce two refusals, and
    /// review had already recorded that such a fixture "must produce two DIFFERENT
    /// sub-case sets" as an assertion inside the test rather than a property of it.
    ///
    /// Both rules are load-bearing and both were real bugs review found:
    ///
    /// * SORT -- push order is RENDER order, which is SPEED order, so without it the same
    ///   pair of reasons keyed two ways depending only on who moved first, splitting one
    ///   `world_failure_reasons` measurement across two buckets and halving each.
    /// * DEDUPE -- both sides refusing for the SAME reason is the common case, and the
    ///   duplicate is pure length with no information in it.
    ///
    /// The length budget is pinned here too, mirroring `_REASON_DETAIL_LIMIT` in
    /// `src/pokezero/engine_search.py`: that seam TRUNCATES, and a truncated key aliases
    /// two different diagnoses into one bucket.
    #[test]
    fn the_attribution_unsafe_label_is_deduplicated_and_sorted() {
        /// Mirror of `_REASON_DETAIL_LIMIT` in `src/pokezero/engine_search.py`.
        const PY_REASON_DETAIL_LIMIT: usize = 512;

        let mut out = RenderedEvents::default();
        // Deliberately pushed in NON-alphabetical order, with a duplicate, which is
        // exactly what two sides refusing produces.
        out.mark_attribution_unsafe("segmentation_failed");
        out.mark_attribution_unsafe("immobilization_marker_tail_not_terminal");
        out.mark_attribution_unsafe("segmentation_failed");
        let label = attribution_unsafe_label(&out);
        assert_eq!(
            label,
            "immobilization_marker_tail_not_terminal,segmentation_failed",
            "the label must be deduplicated and sorted, never render/speed order"
        );

        // Order of PUSHES must not change the key. This is the whole point of the sort.
        let mut reversed = RenderedEvents::default();
        reversed.mark_attribution_unsafe("immobilization_marker_tail_not_terminal");
        reversed.mark_attribution_unsafe("segmentation_failed");
        assert_eq!(attribution_unsafe_label(&reversed), label);

        // ...and it fits the Python seam WITH the prefix that side prepends. The lane
        // string varies; `tree/model fold` is the longest in use.
        let full =
            format!("attribution-unsafe renderer branch rejected before tree/model fold: {label}");
        assert!(
            full.len() <= PY_REASON_DETAIL_LIMIT,
            "refusal message is {} chars, over the {PY_REASON_DETAIL_LIMIT}-char seam \
             budget -- it would be truncated into a `world_failure_reasons` key: {full}",
            full.len()
        );
    }

    /// The marker's TERMINAL/NON-TERMINAL split and its REASON, pinned at the only seam
    /// where they can be pinned.
    ///
    /// The renderer arm that consumes a marker is reachable only through `segment`, which
    /// prefix-matches against re-generated engine branches -- and the engine appends the
    /// marker LAST, after every instruction gen3's `before_move` can push, all of which
    /// `consume_move_prelude` consumes. So `[marker, something]` is not constructible end
    /// to end, the `NotTerminal` refusal is a FAIL-CLOSED guard rather than a live path,
    /// and this classifier is the whole of what a test can reach.
    ///
    /// Stated as a coverage limit rather than implied: mutating the refusal arm's BODY (the
    /// `immobilization_marker_tail_not_terminal` reason, the `sim.apply` loop) survives the
    /// suite, because nothing reaches it. Mutating this classifier does not -- collapsing it
    /// to always-`Terminal`, dropping the `side_ref` match, widening `tail.len() == 1` to
    /// `position == tail.len() - 1`, or swapping the two `|cant|` tags each fail here.
    #[test]
    fn a_move_immobilization_marker_is_classified_by_position_and_reason() {
        let marker = |side_ref, reason| {
            Instruction::MoveImmobilized(MoveImmobilizedInstruction { side_ref, reason })
        };
        let boost = Instruction::Boost(BoostInstruction {
            side_ref: SideReference::SideOne,
            stat: PokemonBoostableStat::Defense,
            amount: 1,
        });

        for reason in [ImmobilizeReason::Attract, ImmobilizeReason::Paralysis] {
            // The shape the engine actually produces.
            assert_eq!(
                move_immobilization_marker(
                    std::slice::from_ref(&marker(SideReference::SideOne, reason)),
                    SideReference::SideOne
                ),
                Some((ImmobilizationMarker::Terminal, reason)),
                "{reason:?}"
            );

            // Anything else in the tail refuses -- in BOTH orders, so the guard is "the
            // marker is the whole tail" and not the weaker "the marker is last".
            for tail in [
                vec![marker(SideReference::SideOne, reason), boost.clone()],
                vec![boost.clone(), marker(SideReference::SideOne, reason)],
            ] {
                assert_eq!(
                    move_immobilization_marker(&tail, SideReference::SideOne),
                    Some((ImmobilizationMarker::NotTerminal, reason)),
                    "{tail:?} must refuse rather than render a bare |cant| and drop the rest"
                );
            }

            // SIDE is load-bearing: the other side's marker is not this side's cant line.
            // Without the `side_ref` match a segmentation slip would credit the `|cant|`
            // to the wrong Pokemon, the worst failure available for an attribution line.
            assert_eq!(
                move_immobilization_marker(
                    std::slice::from_ref(&marker(SideReference::SideTwo, reason)),
                    SideReference::SideOne
                ),
                None,
                "{reason:?}"
            );
        }

        // No marker at all is the overwhelmingly common case and must stay untouched.
        assert_eq!(move_immobilization_marker(&[], SideReference::SideOne), None);
        assert_eq!(
            move_immobilization_marker(std::slice::from_ref(&boost), SideReference::SideOne),
            None
        );

        // THE TAGS, pinned exactly. `public_action_capture.py` keys public actions as
        // `cant:{reason}`, so swapping these two is not a cosmetic slip -- it reports a
        // different ACTION to everything downstream.
        assert_eq!(immobilize_cant_reason(ImmobilizeReason::Attract), "Attract");
        assert_eq!(immobilize_cant_reason(ImmobilizeReason::Paralysis), "par");
    }

    /// COVERAGE LIMIT, stated because the first version of this test overstated it. That
    /// version re-implemented the registered/unclassified choice inline in the test body
    /// and asserted on its own copy -- so it passed whether or not the production code did
    /// the same thing, and mutation confirmed it: DELETING the degradation from
    /// `unrenderable_tail_families` leaves the whole suite green. It was the vacuous shape
    /// this file already records twice.
    ///
    /// PRECISELY WHAT IS AND IS NOT CAUGHT, because I got this wrong twice.
    ///
    /// `an_unregistered_family_degrades_to_unclassified` pins the HELPER
    /// (`registered_family_or_unclassified`) through the seam review suggested: inverting
    /// its branch fails that test. What is still NOT caught is deleting the CALL from
    /// `unrenderable_tail_families` -- mutation confirms that leaves the suite green. That
    /// is the same direct-test-versus-wiring-test gap as the vocabulary check, and unlike
    /// there it cannot be closed the same way: no real input reaches the branch, since
    /// every family the classifier emits is registered, so a wiring test would need a
    /// test-only injection point in production code.
    ///
    /// My earlier claim that "no test CAN catch that deletion" was wrong in one direction
    /// and my replacement claim that "that deletion is now caught" was wrong in the other.
    /// The accurate statement is the two sentences above. The reasoning below still explains
    /// why the branch takes no real input:
    /// the degradation is only reachable via a token absent from
    /// `UNRENDERABLE_FAMILY_ORDER`, and no such token exists -- every family the classifier
    /// can emit is registered, which `every_classifier_token_is_registered_and_orderable`
    /// and the coverage loop in `the_renderable_allowlist_is_exactly_what_it_was` both
    /// depend on. So the degradation is DEFENCE IN DEPTH for a future edit that adds a
    /// family and forgets to register it. Removing it is harmful only together with that
    /// second mistake.
    ///
    /// What this test does assert, through the real production functions: a registered
    /// family produces a slug the marker accepts without panicking. What it does not
    /// assert is the degradation's own existence. Anyone deleting it should know they are
    /// removing a guard the suite cannot miss.
    #[test]
    fn a_registered_family_reaches_the_marker_without_panicking() {
        // `statrecalc`: Boost is admitted now, so it produces NO family at all.
        let boost = Instruction::ChangeAttack(ChangeStatInstruction {
            side_ref: SideReference::SideOne,
            amount: 20,
        });
        let families = unrenderable_tail_families(std::slice::from_ref(&boost), SideReference::SideOne);
        assert_eq!(families, vec!["statrecalc"]);
        for family in &families {
            assert!(
                UNRENDERABLE_FAMILY_ORDER.contains(family),
                "{family:?} is emitted but unregistered, so the release wheel would abort"
            );
        }
        let mut out = RenderedEvents::default();
        out.mark_attribution_unsafe_subcase(
            SLEEPTALK_LOSSY_TAG,
            &ambiguous_unrenderable_slug(std::slice::from_ref(&boost), SideReference::SideOne),
        );
        assert!(out.is_attribution_unsafe());
    }

    /// The vocabulary check must be WIRED IN, not merely present.
    ///
    /// The test above calls `assert_subcase_vocabulary` directly, so it stays green if the
    /// call is deleted from `mark_attribution_unsafe_subcase` -- it would pin a function
    /// nothing invokes, which is the vacuous-test shape that has already cost this program
    /// a wheel break and a negative control that asserted nothing. This one goes through
    /// the marker, so removing the call fails it.
    #[test]
    #[should_panic(expected = "unregistered token")]
    fn the_marker_itself_rejects_an_unregistered_token() {
        let mut out = RenderedEvents::default();
        out.mark_attribution_unsafe_subcase(
            SLEEPTALK_LOSSY_TAG,
            "sleeptalk_called_unidentified:ambiguous_unrenderable:invented_family",
        );
    }

    /// The slugs actually shipped must pass their own gate.
    ///
    /// EVERY shape, through the PRODUCTION assert. This looped ONE hand-picked shape
    /// (`Structure`), so six of seven slugs never touched the gate they must clear at
    /// runtime -- and the sibling test asserts only a PROXY for it (starts-with-tag plus
    /// ends-with-token), which a slug carrying an extra unregistered segment satisfies while
    /// the real gate panics. Review demonstrated exactly that survivor.
    /// `assert_subcase_vocabulary` is a plain `assert!` kept out of `debug_assert!` so it
    /// survives `--release`; a slug that passes only the proxy kills the campaign worker.
    #[test]
    fn the_live_subcase_slugs_are_all_in_vocabulary() {
        assert_subcase_vocabulary(
            SLEEPTALK_LOSSY_TAG,
            sleeptalk_subcase_slug(&SleepTalkIdent::Ambiguous),
        );
        for shape in NoneMatchedShape::ALL {
            let slug = none_matched_slugs(one_shape(shape)).next().unwrap();
            assert_subcase_vocabulary(SLEEPTALK_LOSSY_TAG, slug);
        }
        // EVERY Protect-counter literal, through the PRODUCTION gate rather than a membership
        // check. Strictly stronger: membership passes a re-composed literal that the gate
        // would reject. Its caller is `mark_lossy_subcase`, which does NOT reach this gate
        // today -- the `&'static str` bound on its `subcase` keeps that caller set
        // literals-only and greppable, so running them through here is what makes closing
        // that asymmetry safe later.
        //
        // ITERATED, not hand-copied, and that is the fix for a defect review found here. This
        // block used to spell out two literals with the comment "BOTH have to clear it or the
        // branch that fires less often is the one that panics a release wheel". A third
        // literal was then added at the emit site for the full-HP reclaim and this gate was
        // not updated, so the block's own stated invariant was violated and a mutant
        // deregistering the new token survived the entire suite. `PROTECT_MARKER_COUNTERS` is
        // the emit site's own array, so the copy that could go stale no longer exists.
        let mut seen = std::collections::BTreeSet::new();
        for has_absorb in [true, false] {
            for clamps in [true, false] {
                let slug = protect_marker_counter_slug(has_absorb, clamps);
                assert_subcase_vocabulary(SLEEPTALK_LOSSY_TAG, slug);
                seen.insert(slug);
            }
        }
        assert_eq!(
            seen.len(), 3,
            "the Protect counter no longer has exactly three distinct literals: {seen:?}. \
             That is fine, but this gate derives the set from \
             `protect_marker_counter_slug` and the count is what tells a reader the emit \
             site's arity changed"
        );
        // The MULTI-shape composition too: `none_matched_slugs` yields one slug per observed
        // shape and a real world can carry several, so each must clear the gate.
        let mut several = NoneMatchedShapes::default();
        several.insert(NoneMatchedShape::BranchIsPrefix);
        several.insert(NoneMatchedShape::TailIsPrefix);
        several.insert(NoneMatchedShape::Length);
        for slug in none_matched_slugs(several) {
            assert_subcase_vocabulary(SLEEPTALK_LOSSY_TAG, slug);
        }
        let boost = Instruction::Boost(BoostInstruction {
            side_ref: SideReference::SideOne,
            stat: PokemonBoostableStat::Defense,
            amount: 1,
        });
        assert_subcase_vocabulary(SLEEPTALK_LOSSY_TAG, &ambiguous_unrenderable_slug(&[boost], SideReference::SideOne));
    }

    /// The ROUTING decision, tested directly for every variant.
    ///
    /// `NoneMatched` must be unsafe REGARDLESS of how renderable its tail is: the tail is
    /// not the problem there, the renderer's inability to reproduce it is. Making it
    /// lossy-only was indistinguishable from correct code until this existed.
    #[test]
    fn none_matched_is_always_unsafe_and_renderable_ambiguity_never_is() {
        let renderable = vec![Instruction::Damage(DamageInstruction {
            side_ref: SideReference::SideOne,
            damage_amount: 10,
        })];
        // `Change<Stat>`, not Boost: Boost is RENDERED now and so no longer unrenderable.
        // Using it here would assert the opposite of the intended behaviour, which is the
        // exact trap this test's own comment records from an earlier split.
        let unrenderable = vec![Instruction::ChangeAttack(ChangeStatInstruction {
            side_ref: SideReference::SideOne,
            amount: 20,
        })];

        // A HEAL-DIRECTION Damage must refuse. `Damage` is signed and the walk emits on
        // decreases only, so admitting a negative amount renders nothing while the state
        // moves -- review reproduced exactly that and it is the defect this predicate is
        // for. Pinned here because no natural gen3 move pair produces such a tail, so no
        // corpus test can reach it.
        let heal_shaped = vec![Instruction::Damage(DamageInstruction {
            side_ref: SideReference::SideOne,
            damage_amount: -130,
        })];
        assert!(
            !ambiguous_tail_is_fully_renderable(&heal_shaped, SideReference::SideOne),
            "a negative `damage_amount` is a HEAL and the walk drops it, so the tail is \
             NOT fully renderable"
        );
        assert!(
            sleeptalk_refusal_is_unsafe(&SleepTalkIdent::Ambiguous, &heal_shaped, SideReference::SideOne),
            "a heal-shaped ambiguous tail must REFUSE"
        );
        // Mixed sign refuses too: one dropped component is enough.
        let mixed = vec![
            Instruction::Damage(DamageInstruction {
                side_ref: SideReference::SideOne,
                damage_amount: 130,
            }),
            Instruction::Damage(DamageInstruction {
                side_ref: SideReference::SideTwo,
                damage_amount: -130,
            }),
        ];
        assert!(sleeptalk_refusal_is_unsafe(&SleepTalkIdent::Ambiguous, &mixed, SideReference::SideOne));
        // Zero is a no-op and stays admissible.
        assert!(!sleeptalk_refusal_is_unsafe(
            &SleepTalkIdent::Ambiguous,
            &[Instruction::Damage(DamageInstruction {
                side_ref: SideReference::SideOne,
                damage_amount: 0,
            })],
            SideReference::SideOne
        ));

        // Ambiguous: renderable is USABLE, unrenderable REFUSES.
        assert!(!sleeptalk_refusal_is_unsafe(&SleepTalkIdent::Ambiguous, &renderable, SideReference::SideOne));
        assert!(sleeptalk_refusal_is_unsafe(&SleepTalkIdent::Ambiguous, &unrenderable, SideReference::SideOne));

        // NoneMatched: unsafe either way. A renderable tail must NOT rescue it.
        assert!(
            sleeptalk_refusal_is_unsafe(&SleepTalkIdent::NoneMatched(one_shape(NoneMatchedShape::Structure)), &renderable, SideReference::SideOne),
            "none_matched with a renderable tail must STILL refuse -- the tail is not the \
             defect, the renderer's failure to reproduce it is"
        );
        assert!(sleeptalk_refusal_is_unsafe(&SleepTalkIdent::NoneMatched(one_shape(NoneMatchedShape::Structure)), &unrenderable, SideReference::SideOne));

        // An empty tail is renderable by construction, so it must not rescue it either.
        assert!(sleeptalk_refusal_is_unsafe(&SleepTalkIdent::NoneMatched(one_shape(NoneMatchedShape::Structure)), &[], SideReference::SideOne));
        assert!(!sleeptalk_refusal_is_unsafe(&SleepTalkIdent::Ambiguous, &[], SideReference::SideOne));
    }

    /// The refusing seam must still refuse `none_matched`, and must NOT refuse
    /// `ambiguous`. Synthesised, so it does not depend on reaching either arm.
    ///
    /// `none_matched` is genuinely hard to reach from a real state -- two crate tests
    /// assert it stays at zero, and the identifier's own comment records the arm as
    /// unreachable today -- so a test that waits for the engine to produce one pins
    /// nothing. Review found the consequence: making `NoneMatched` lossy-only too, which
    /// would let a divergent render reach the fold, was INDISTINGUISHABLE from the
    /// correct code because the only witness was a test already red for another reason.
    /// This closes that by constructing the two shapes directly.
    #[test]
    fn the_refusing_seam_separates_ambiguous_from_none_matched() {
        // AMBIGUOUS: sub-case on the measurement channel, bare tag on the contract
        // channel, nothing on the refusing channel.
        let usable = RenderedEvents {
            lines: Vec::new(),
            turn_completed: false,
            lossy: vec![SLEEPTALK_LOSSY_TAG.to_string()],
            attribution_unsafe: Vec::new(),
            lossy_subcases: vec![sleeptalk_subcase_slug(&SleepTalkIdent::Ambiguous).to_string()],
            active_status_transitions: Vec::new(),
        };
        assert!(
            !usable.is_attribution_unsafe(),
            "a renderable ambiguous branch must pass the refusing seam"
        );
        reject_attribution_unsafe(&usable, "test")
            .expect("ambiguous-and-renderable must not be refused");

        // NONE_MATCHED: the renderer could not reproduce the engine's tail, so the
        // description may be wrong. This one MUST still refuse.
        let unsafe_render = RenderedEvents {
            lines: Vec::new(),
            turn_completed: false,
            lossy: vec![SLEEPTALK_LOSSY_TAG.to_string()],
            attribution_unsafe: vec![
                none_matched_slugs(one_shape(NoneMatchedShape::Structure)).next().unwrap().to_string(),
            ],
            lossy_subcases: Vec::new(),
            active_status_transitions: Vec::new(),
        };
        assert!(unsafe_render.is_attribution_unsafe());
        let error = reject_attribution_unsafe(&unsafe_render, "test")
            .expect_err("none_matched must not reach the fold");
        assert!(
            error.to_string().contains("sleeptalk_called_unidentified:none_matched"),
            "the refusal must name the cause: {error}"
        );

        // And the two slugs must not be the same string, or the split is cosmetic.
        assert_ne!(
            sleeptalk_subcase_slug(&SleepTalkIdent::Ambiguous),
            none_matched_slugs(one_shape(NoneMatchedShape::Structure)).next().unwrap()
        );
    }

    /// Pin WHICH cause maps to which label, without needing an engine state
    /// that reaches each variant.
    ///
    /// Independent review found this mapping was completely unpinned: swapping
    /// the two literals left the entire crate suite green. The obvious fix --
    /// an end-to-end fixture per arm -- turned out to be impossible for
    /// `none_matched`: the fixture that appeared to reach it only did so on a
    /// STALE vendored engine missing `poke-engine-gen3-sleeptalk-crit-arm.patch`
    /// (C87). On a faithful build that patch makes the callee's regenerated tail
    /// match, so the arm is not reachable at all on the gen3 randbats set pool
    /// (measured: 7,560 state combinations across all 70 Sleep Talk variants,
    /// zero `none_matched`).
    ///
    /// An unreachable arm still has to be labelled correctly if the engine ever
    /// regresses into producing it -- and mislabelling is expensive in a
    /// specific direction, because `ambiguous` is the arm fixable ONLY by an
    /// engine change. So pin the mapping directly rather than pinning nothing.
    #[test]
    fn sleeptalk_subcases_map_to_their_own_labels() {
        assert_eq!(
            sleeptalk_subcase_slug(&SleepTalkIdent::Ambiguous),
            "sleeptalk_called_unidentified:ambiguous"
        );
        assert_eq!(
            none_matched_slugs(one_shape(NoneMatchedShape::Structure))
                .next()
                .unwrap(),
            "sleeptalk_called_unidentified:none_matched:shape_structure"
        );
    }

    /// Both labels must stay inside the contract tag's namespace, or
    /// `mark_attribution_unsafe_subcase`'s assertion trips in production and the
    /// differential starts seeing a tag it does not recognise.
    #[test]
    fn every_sleeptalk_subcase_belongs_to_the_lossy_contract_tag() {
        // Ambiguous through the slug fn; every NoneMatched shape through the set emitter,
        // which is where they are produced now. Both must stay inside the contract tag.
        let mut slugs: Vec<&'static str> = vec![sleeptalk_subcase_slug(&SleepTalkIdent::Ambiguous)];
        for shape in NoneMatchedShape::ALL {
            slugs.extend(none_matched_slugs(one_shape(shape)));
        }
        for slug in slugs {
            assert!(
                slug.starts_with(SLEEPTALK_LOSSY_TAG),
                "{slug} escapes the contract tag {SLEEPTALK_LOSSY_TAG}"
            );
            assert_ne!(
                slug, SLEEPTALK_LOSSY_TAG,
                "the bare tag carries no cause and cannot be measured"
            );
        }
    }

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

    /// Sleep Talk callees carry `move_index: M0`, which is why the pristine
    /// rebuild in `render_move_phase` must be guarded on `move_id`.
    ///
    /// `get_sleep_talk_choices` clones raw move-table `Choice`s, and those carry
    /// the `Choice::default()` `move_index` — `M0` — which nothing ever sets.
    /// Rebuilding damage expectations from `choice.move_index` therefore returns
    /// move SLOT 0 for every callee, not the callee. Measured cost when that
    /// guard was absent: two boundaries, one per failure mode — slot 0 being
    /// Sleep Talk itself (Status, so no maxima at all and `|-crit|` becomes
    /// unemittable) and slot 0 being an unrelated weaker move (maxima too low, so
    /// the crit gate fires on a non-crit).
    ///
    /// This pins the upstream invariant rather than the guard, deliberately: if a
    /// future engine version starts setting `move_index` on these clones the
    /// assert below fails, which is the signal that the guard is no longer
    /// load-bearing. See `reports/c102`.
    #[test]
    fn sleep_talk_callee_choices_carry_slot_zero_move_index() {
        let state = parse_state(MINIMAL.trim()).expect("fixture parses");
        let callees = state.side_one.get_active_immutable().get_sleep_talk_choices();
        assert!(
            !callees.is_empty(),
            "fixture must expose at least one Sleep Talk callee"
        );
        for callee in &callees {
            assert_eq!(
                callee.move_index,
                poke_engine::state::PokemonMoveIndex::M0,
                "callee {:?} carries a real move_index — the pristine-rebuild \
                 guard in render_move_phase may no longer be needed",
                callee.move_id
            );
        }
    }

    /// KNOWN OPEN — documents a defect that is still live. `#[ignore]`d rather
    /// than deleted so it is not rediscovered from scratch.
    ///
    /// The seeder is at 307/312 with Leftovers. Heals resolve in phase order,
    /// Leftovers (10.4) before the drain (10.5), so the +5 tick fills it to 312
    /// and the drain then recovers nothing: the engine emits one heal, the plan
    /// books two, the reconcile disables the side, and the tick renders
    /// `Leech Seed`.
    ///
    /// An `active.hp < active.maxhp` guard does NOT close this, which is why
    /// this PR does not add one (see the NOTE on the Leftovers slot above, and
    /// the 5-row cost measured there). Evaluated on the pre-residual state,
    /// 307 < 312 holds and the drain slot would still be booked. Closing it needs the drain predicate to know the
    /// seeder's HP AFTER its own earlier heal phases, which this plan
    /// deliberately avoids — it is documented as using presence predicates
    /// only, never HP formulas. Doing it properly means modelling the heal
    /// phases cumulatively, and deciding what to do about Wish; that is a
    /// larger change than this one and is not attempted here.
    ///
    /// Zero occurrences in seeds 19000000-19000199, reachable in ordinary gen3
    /// stall play, and present identically before this change.
    #[test]
    #[ignore = "known open: drain slot booked from pre-residual HP; see doc comment"]
    fn a_near_full_hp_seeder_still_over_books_the_drain_slot() {
        let mut state = parse_state(MINIMAL.trim()).expect("fixture parses");
        {
            let seeder = state.side_one.get_active();
            seeder.maxhp = 312;
            seeder.hp = 307;
            seeder.item = Items::LEFTOVERS;
        }
        state
            .side_two
            .volatile_statuses
            .insert(PokemonVolatileStatus::LEECHSEED);
        let segment = vec![
            heal_one(5),
            Instruction::Damage(poke_engine::instruction::DamageInstruction {
                side_ref: SideReference::SideTwo,
                damage_amount: 12,
            }),
        ];
        assert_eq!(
            residual_tags(&mut state, &segment, "p1a"),
            vec!["item: Leftovers".to_string()],
            "the +5 is the Leftovers tick; the drain recovered nothing"
        );
    }

    /// AIR LOCK suppresses weather, so SWIFT SWIM must not double a speed while
    /// it is on the field.
    ///
    /// The engine's `get_effective_speed` wraps its weather match in
    /// `state.weather_is_active(...)`; this replica did not. Unguarded, Rayquaza
    /// (AIR LOCK) against Seaking (SWIFT SWIM) in RAIN gave Seaking 348 against
    /// Rayquaza's 261 and flipped the computed move order. `segment()` tries only
    /// the order it computes, so it regenerated the wrong side as phase 1,
    /// nothing was a prefix of the real instruction list, and the whole branch
    /// was voided as `segmentation_failed`. Two rows (reports/c108).
    ///
    /// Reverting the `weather_is_active` guard makes the rain assertion below
    /// fail: Seaking comes back doubled.
    #[test]
    fn air_lock_suppresses_swift_swim_in_the_speed_replica() {
        let mut state = parse_state(MINIMAL.trim()).expect("fixture parses");
        state.weather.weather_type = Weather::RAIN;
        state.side_two.get_active().ability = Abilities::SWIFTSWIM;
        let base = effective_speed(&state, SideReference::SideTwo);

        // Anchor on the UN-doubled read. An earlier version of this test set
        // side_one's ability to NONE and asserted the result equalled `base` --
        // but NONE is already the fixture's ability, so it compared two
        // identical pure computations over an unmutated state and could not
        // fail. It never asserted that Swift Swim doubles anything.
        let doubled = base;
        state.side_two.get_active().ability = Abilities::NONE;
        let undoubled = effective_speed(&state, SideReference::SideTwo);
        state.side_two.get_active().ability = Abilities::SWIFTSWIM;
        assert_eq!(
            doubled,
            undoubled * 2,
            "SWIFT SWIM must double in rain with no suppressor: {undoubled} -> {doubled}"
        );

        // AIR LOCK on the OPPONENT must suppress it.
        state.side_one.get_active().ability = Abilities::AIRLOCK;
        let suppressed = effective_speed(&state, SideReference::SideTwo);
        assert!(
            suppressed < doubled,
            "AIR LOCK must suppress the Swift Swim doubling: {suppressed} vs {doubled}"
        );
        assert_eq!(
            suppressed * 2,
            doubled,
            "suppression must remove exactly the 2x, not some other factor"
        );
    }

    /// Same shape, but the SEEDED mon has Liquid Ooze, so there is no drain
    /// heal to be silent about — the engine emits the reversed drain as a
    /// negative Heal on the seeder instead
    /// (`gen3/generate_instructions.rs:3624-3647`).
    ///
    /// Planning a drain slot anyway leaves `plan.heal` one longer than
    /// `emitted_heal` (which counts only `heal_amount > 0`), the reconcile
    /// marks the side unusable, and the seeder's Leftovers tick falls through
    /// to the `[from] Leech Seed` fallback — ledger H.1, in the one case the
    /// test above does not cover. Reverting the `LIQUIDOOZE` conjunct in
    /// `drains_opponent` makes this fail with `Leech Seed` in place of
    /// `item: Leftovers`; verified in both directions.
    #[test]
    fn liquid_ooze_leaves_the_seeders_leftovers_tick_correctly_tagged() {
        let mut state = parse_state(MINIMAL.trim()).expect("fixture parses");
        {
            let seeder = state.side_one.get_active();
            seeder.maxhp = 312;
            seeder.hp = 200;
            seeder.item = Items::LEFTOVERS;
        }
        state.side_two.get_active().ability = Abilities::LIQUIDOOZE;
        state
            .side_two
            .volatile_statuses
            .insert(PokemonVolatileStatus::LEECHSEED);
        // Sap damage on side two, the reversed drain as a NEGATIVE heal on the
        // seeder, then the seeder's own Leftovers tick (312 / 16 = 19).
        let segment = vec![
            Instruction::Damage(poke_engine::instruction::DamageInstruction {
                side_ref: SideReference::SideTwo,
                damage_amount: 39,
            }),
            heal_one(-39),
            heal_one(19),
        ];
        assert_eq!(
            residual_tags(&mut state, &segment, "p1a"),
            vec![
                "ability: Liquid Ooze|[of] p2a: Squirtle".to_string(),
                "item: Leftovers".to_string(),
            ],
            "the reversed drain is tagged Liquid Ooze, and the +19 is the \
             seeder's own Leftovers tick — not a Leech Seed drain"
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

    /// SAND VEIL, and this pin is the whole point of the change.
    ///
    /// Round 2's review deleted `|| active.ability == Abilities::SANDVEIL` from
    /// `weather_chips` and the ENTIRE crate suite stayed green: 373 passed, 0
    /// failed. The line was pinned by nothing, while the LIQUID OOZE guard worth zero
    /// measured rows got a pin the round before.
    ///
    /// WORTH ONE ROW, NOT TWO, AND VIA ONE ARM, NOT BOTH. Earlier versions of this
    /// docstring said two rows, then said "both arms". Both were wrong.
    ///
    /// The branch's own artifact at the reorder-only revision `87bcf351` -- whose
    /// `events.rs` has zero `SANDVEIL` occurrences, and which predates the `[silent]`
    /// change (that entered at `c9f6839b`) -- records holdout **4** with
    /// `19100193/46` already closed. So the FALLBACK REORDER ALONE closes that row;
    /// this exemption closes `19100014/35`, 4 -> 3.
    ///
    /// And it closes it through the 90% arm only. The 10% arm is the engine's
    /// Leech-Seed-MISSED branch against a Showdown hit
    /// (`observed_only=[('leechseed', -33)] engine_only=[]`); no harness RENDERING
    /// change can make a miss branch reproduce a hit. One matching branch closes a
    /// boundary (`engine_transition_differential.py` returns "matched" on the first
    /// fully matching branch), which is why the row closes anyway.
    ///
    /// MY FIRST VERSION OF THIS PIN WAS ALSO VACUOUS, and I wrote it in the same
    /// commit as an expiry pin with the identical flaw. It asserted on DAMAGE tags
    /// with a poison-only segment, and deleting the exemption still left it green.
    /// An unfilled chip slot does not corrupt damage attribution -- it corrupts the
    /// HEAL labels, because that is where the fallback has to guess between
    /// Leftovers and the cross-side drain. Twice in one commit I asserted on the
    /// wrong side of the mechanism and called it a pin.
    ///
    /// Measured: without the exemption this state yields
    /// `["item: Leftovers", "item: Leftovers"]` -- the genuine drain mislabelled as
    /// a second Leftovers tick, which is the `19100193/46` signature.
    #[test]
    fn sand_veil_is_exempt_so_the_plan_does_not_book_a_chip_that_never_fires() {
        let mut state = parse_state(MINIMAL.trim()).expect("fixture parses");
        state.weather.weather_type = Weather::SAND;
        state.weather.turns_remaining = 5;
        {
            let active = state.side_one.get_active();
            active.maxhp = 320;
            active.hp = 200;
            active.item = Items::LEFTOVERS;
            active.ability = Abilities::SANDVEIL;
        }
        state
            .side_two
            .volatile_statuses
            .insert(PokemonVolatileStatus::LEECHSEED);

        // Sand is active with turns to spare, but this mon is EXEMPT, so the engine
        // emits no chip for it: the segment is the Leftovers tick plus the drain.
        let segment = vec![heal_one(20), heal_one(40)];
        assert_eq!(
            residual_tags(&mut state, &segment, "p1a"),
            vec!["item: Leftovers".to_string()],
            "the drain heal must stay silent; a second `item: Leftovers` means the \
             plan booked a sandstorm chip for a Sand Veil mon"
        );

        // Anti-vacuity: the same state WITHOUT Sand Veil really does book a chip,
        // so the fixture is exercising the exemption and not an empty plan.
        let mut control = parse_state(MINIMAL.trim()).expect("fixture parses");
        control.weather.weather_type = Weather::SAND;
        control.weather.turns_remaining = 5;
        {
            let active = control.side_one.get_active();
            active.maxhp = 320;
            active.hp = 200;
            active.item = Items::LEFTOVERS;
        }
        control
            .side_two
            .volatile_statuses
            .insert(PokemonVolatileStatus::LEECHSEED);
        let with_chip = vec![damage_one(20), heal_one(20), heal_one(40)];
        assert_eq!(
            residual_tags(&mut control, &with_chip, "p1a"),
            vec!["Sandstorm".to_string(), "item: Leftovers".to_string()],
            "without Sand Veil the chip is booked and the plan reconciles"
        );
    }

    /// WEATHER EXPIRY, the same class as Sand Veil -- and NOT reachable in the
    /// current pool, which an earlier version of this docstring claimed it was.
    ///
    /// Measured rather than assumed this time: `weather_chips` can only return `Some`
    /// for SAND or HAIL, and `data/random-battles/gen3/sets.json` has **0 of 220**
    /// species carrying `sandstorm` or `hail` (`raindance` 7 and `sunnyday` 4 exist,
    /// and neither chips; Snow Warning does not exist in gen3). So sand only ever
    /// comes from Sand Stream, which writes `WEATHER_ABILITY_TURNS = -1`, and
    /// `generate_instructions.rs:4144` never decrements a non-positive value --
    /// `turns_remaining == 1` cannot occur. The same status as the Liquid Ooze guard.
    ///
    /// It is fixed for FIDELITY, not for a row. The `== 1` boundary still matters,
    /// because the permanent region IS reachable and `<= 1` breaks it across all of it.
    ///
    /// `weather_is_active` ignores `turns_remaining` (`gen3/state.rs:1050-1060`)
    /// and this function reads the PRE-residual state, but the engine decrements
    /// and clears the weather at `generate_instructions.rs:4144-4163` BEFORE its
    /// chip loop at `:4193`. So on the expiring turn the engine emits no chip while
    /// the plan books one, and the side falls to the constant fallback.
    ///
    /// MY FIRST VERSION OF THIS PIN WAS VACUOUS AND PASSED WITHOUT THE FIX. It
    /// asserted on DAMAGE tags, and the defect does not show there -- an unfilled
    /// chip slot corrupts the HEAL labels, because that is where the fallback has
    /// to guess between Leftovers and the drain. Asserting on the wrong side of the
    /// mechanism looked like a test and constrained nothing.
    ///
    /// Measured: without the `turns_remaining == 1` gate this state yields
    /// `["item: Leftovers", "item: Leftovers"]` -- the genuine drain mislabelled as
    /// a second Leftovers tick, the `19100193/46` signature. With it, the drain
    /// renders `[silent]` as Showdown does.
    #[test]
    fn expiring_weather_books_no_chip_so_the_drain_keeps_its_label() {
        let mut state = parse_state(MINIMAL.trim()).expect("fixture parses");
        state.weather.weather_type = Weather::SAND;
        state.weather.turns_remaining = 1;
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

        // The engine cleared the weather first, so the segment carries NO chip:
        // just the Leftovers tick and the cross-side drain heal.
        let segment = vec![heal_one(20), heal_one(40)];
        let tags = residual_tags(&mut state, &segment, "p1a");
        assert_eq!(
            tags,
            vec!["item: Leftovers".to_string()],
            "the drain heal must stay silent; a second `item: Leftovers` means the \
             plan was disabled by a chip the engine never emitted"
        );

        // THE BOUNDARY, and it was unpinned until a review broke it: `== 1` and not
        // `<= 1`. PERMANENT weather is `turns_remaining == -1`
        // (`gen3/abilities.rs:20` `WEATHER_ABILITY_TURNS`), written by Sand Stream,
        // Drizzle and Drought. The engine's decrement is gated on
        // `turns_remaining > 0`, so permanent weather never clears and DOES chip --
        // the plan must book it. Under `<= 1` the chip is skipped, the plan comes up
        // one short, and the drain is mislabelled as a second Leftovers tick across
        // the entire permanent-weather region. That region is reachable: Tyranitar,
        // Kyogre and Groudon are all in the pool, and row `19100014/35` is Tyranitar
        // switching into its own sand.
        let mut permanent = parse_state(MINIMAL.trim()).expect("fixture parses");
        permanent.weather.weather_type = Weather::SAND;
        permanent.weather.turns_remaining = -1;
        {
            let active = permanent.side_one.get_active();
            active.maxhp = 320;
            active.hp = 200;
            active.item = Items::LEFTOVERS;
        }
        permanent
            .side_two
            .volatile_statuses
            .insert(PokemonVolatileStatus::LEECHSEED);
        assert_eq!(
            residual_tags(
                &mut permanent,
                &vec![damage_one(20), heal_one(20), heal_one(40)],
                "p1a",
            ),
            vec!["Sandstorm".to_string(), "item: Leftovers".to_string()],
            "permanent weather (turns_remaining -1) never expires, so its chip MUST \
             still be booked; skipping it re-arms the 19100193/46 mislabel"
        );

        // HAIL, because the gate sits above BOTH weather branches and the property is
        // claimed for both. Scoping the gate to `weather_type == Weather::SAND`, or
        // moving it below the hail branch, leaves all 375 tests green while making
        // hail-expiry book a phantom chip. Pinned even though hail is unreachable in
        // the current pool (0 of 220 sets carry it), because the gate's placement --
        // above the branches rather than inside one -- is the thing worth keeping.
        let mut hail = parse_state(MINIMAL.trim()).expect("fixture parses");
        hail.weather.weather_type = Weather::HAIL;
        hail.weather.turns_remaining = 1;
        {
            let active = hail.side_one.get_active();
            active.maxhp = 320;
            active.hp = 200;
            active.item = Items::LEFTOVERS;
        }
        hail.side_two
            .volatile_statuses
            .insert(PokemonVolatileStatus::LEECHSEED);
        assert_eq!(
            residual_tags(&mut hail, &vec![heal_one(20), heal_one(40)], "p1a"),
            vec!["item: Leftovers".to_string()],
            "expiring HAIL must book no chip either; the gate is above both branches"
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

    /// A side whose OPPONENT is seeded must not have its Leftovers tick
    /// attributed to the Leech Seed drain. Regression for holdout row
    /// `19100193/46`.
    ///
    /// `residual_heal_cause` is the fallback reached when the residual plan fails
    /// reconciliation, and it used to test Leech Seed BEFORE Leftovers.
    ///
    /// NOTE: an earlier version of this docstring justified the order by residual
    /// PHASE -- Leftovers 10.4 before the drain 10.5 -- and that reasoning is
    /// RETRACTED where the function is defined, ~2,100 lines up. This function
    /// receives no heal index, so it is a constant function of state and has no
    /// notion of "first"; and a faster victim's drain is emitted BEFORE the
    /// seeder's Leftovers tick, so the premise is false regardless. The real reason
    /// Leftovers wins is that the drain renders silently, so `"Leech Seed"` is
    /// never a correct answer for a `[from]`-tagged heal.
    ///
    /// On that row the
    /// plan reserved a drain slot — opponent seeded, both actives alive when it
    /// was built — but the seeder died to poison at 10.6 before the 10.5 sap
    /// could run, so `emitted_heal` came up one short, the plan was discarded,
    /// and this fallback tagged an 18-point Leftovers tick (290/16) as
    /// `[from] Leech Seed`. A real drain would have been 273/8 = 34.
    #[test]
    fn a_seeded_opponent_does_not_steal_the_leftovers_tag() {
        let mut state = parse_state(MINIMAL.trim()).expect("fixture parses");
        state.side_one.get_active().item = Items::LEFTOVERS;
        // The opponent is seeded, so a drain heal on side one is POSSIBLE — which
        // is exactly the condition that used to win the fallback outright.
        state
            .side_two
            .volatile_statuses
            .insert(PokemonVolatileStatus::LEECHSEED);
        let heal = Instruction::Heal(poke_engine::instruction::HealInstruction {
            side_ref: SideReference::SideOne,
            heal_amount: 6,
        });

        let mut rendered = RenderedEvents::default();
        let mut sim = Sim::new(&mut state, [false, false]);
        let mut plan = ResidualPlan::default();
        render_residual_instruction(&mut sim, &heal, None, &mut plan, &ctx(), &mut rendered);
        sim.finish();
        assert!(
            rendered.lines[0].contains("[from] item: Leftovers"),
            "a seeded opponent stole the Leftovers tag: {:?}",
            rendered.lines
        );
    }

    /// The converse, so the reorder cannot pass by always answering Leftovers:
    /// a side WITHOUT Leftovers whose opponent is seeded still gets the drain
    /// label.
    #[test]
    fn without_leftovers_a_seeded_opponent_still_yields_the_drain_label() {
        let mut state = parse_state(MINIMAL.trim()).expect("fixture parses");
        state.side_one.get_active().item = Items::NONE;
        state
            .side_two
            .volatile_statuses
            .insert(PokemonVolatileStatus::LEECHSEED);
        let heal = Instruction::Heal(poke_engine::instruction::HealInstruction {
            side_ref: SideReference::SideOne,
            heal_amount: 6,
        });

        let mut rendered = RenderedEvents::default();
        let mut sim = Sim::new(&mut state, [false, false]);
        let mut plan = ResidualPlan::default();
        render_residual_instruction(&mut sim, &heal, None, &mut plan, &ctx(), &mut rendered);
        sim.finish();
        // `[silent]`, NOT `[from] Leech Seed`. This assertion used to demand the
        // latter, which Showdown never emits for a drain heal
        // (`sim/battle.ts:2293-2296`, `case 'leechseed'` renders
        // `('-heal', target, getHealth, '[silent]')`). A review built the correct
        // fix and found this pin was the only thing failing -- so the pin was
        // enshrining a wrong label and blocking the right one at zero measured
        // cost. Its purpose stands: a seeded opponent WITHOUT Leftovers must not
        // have its drain heal attributed to Leftovers.
        assert!(
            rendered.lines[0].contains("[silent]"),
            "a drain heal must render silently, not with a [from] tag: {:?}",
            rendered.lines
        );
        assert!(
            !rendered.lines[0].contains("item: Leftovers"),
            "the drain heal was attributed to Leftovers: {:?}",
            rendered.lines
        );
    }

    // ---------------------------------------------------------------- G33b
    //
    // The residual block truncated by the opposing active's battle-ending faint.
    // `leftovers_slot_truncated`'s seven arms, three end-to-end through the
    // renderer and four directly on the predicate. Only the FIRST is
    // revert-failing; the other six exist because every one of them passes on
    // `main` too, and a gate that fires unconditionally would break them.
    //
    // Red/green measured in `reports/c147_g33b_residual_bucket_gate.md` §4 by
    // checking out `origin/main` into its own worktree, rebuilding, and running --
    // not by reasoning about which of them the predicate must reach.

    /// Set up the G33b shape: side two is seeded, its active is its LAST living
    /// Pokemon, and side one holds Leftovers below max HP.
    ///
    /// `speed_of_side_two` is the whole experiment. The engine's order-10 class is
    /// speed-major, so a faster seeded victim resolves its entire 10.x set --
    /// including the 10.5 sap that kills it -- before side one's 10.3 is entered,
    /// and `stop_residuals_if_battle_ended!` then skips side one's 10.4 Leftovers.
    fn g33b_state(speed_of_side_two: i16) -> State {
        let mut state = parse_state(MINIMAL.trim()).expect("fixture parses");
        {
            let active = state.side_one.get_active();
            active.maxhp = 268;
            active.hp = 219;
            active.item = Items::LEFTOVERS;
            active.speed = 100;
        }
        {
            let active = state.side_two.get_active();
            active.maxhp = 96;
            active.hp = 12;
            active.item = Items::NONE;
            active.speed = speed_of_side_two;
        }
        state
            .side_two
            .volatile_statuses
            .insert(PokemonVolatileStatus::LEECHSEED);
        state
    }

    fn damage_two(amount: i16) -> Instruction {
        Instruction::Damage(poke_engine::instruction::DamageInstruction {
            side_ref: SideReference::SideTwo,
            damage_amount: amount,
        })
    }

    fn heal_two(amount: i16) -> Instruction {
        Instruction::Heal(poke_engine::instruction::HealInstruction {
            side_ref: SideReference::SideTwo,
            heal_amount: amount,
        })
    }

    /// **The revert-failing pin.** G33b, and the row it comes from is
    /// `19200244/115` (`reports/c143_heal_attribution_diagnosis.md`).
    ///
    /// Side two is the FASTER seeded victim and its own 10.5 sap kills it as its
    /// last living Pokemon. The battle ends inside side two's own order-10 bucket,
    /// so side one's 10.4 Leftovers tick is never reached -- the segment carries
    /// side one's silent drain mirror and nothing else.
    ///
    /// On `main` `ResidualPlan::build` books that unreached slot anyway, `plan.heal`
    /// comes out one longer than `emitted_heal`, `plan.usable[0]` goes false, and the
    /// bare drain falls through to `residual_heal_cause` -- which since C131 change 3
    /// tests Leftovers FIRST. So the drain comes back tagged `[from] item: Leftovers`
    /// on a heal Showdown renders `[silent]`.
    #[test]
    fn a_truncated_leftovers_slot_is_not_booked_so_the_drain_stays_silent() {
        let mut state = g33b_state(200);
        // The sap kills side two (12 of 12) and heals side one by the same amount.
        // Nothing follows: the battle is over and side one's 10.4 never runs.
        let segment = vec![damage_two(12), heal_one(12)];
        assert_eq!(
            residual_tags(&mut state, &segment, "p1a"),
            Vec::<String>::new(),
            "the drain mirror must render bare; a [from] tag here is the G33b mislabel"
        );
    }

    /// The converse, and it is what stops the gate from being "never book
    /// Leftovers". Side one is the FASTER seeder, so its whole order-10 bucket --
    /// 10.4 Leftovers included -- resolves before side two's 10.5 kills side two.
    /// The tick really did fire and must keep its tag. c143 §1a variant C.
    #[test]
    fn a_faster_seeder_keeps_its_leftovers_tag() {
        let mut state = g33b_state(50);
        let segment = vec![heal_one(16), damage_two(12), heal_one(12)];
        assert_eq!(
            residual_tags(&mut state, &segment, "p1a"),
            vec!["item: Leftovers".to_string()],
            "the faster seeder's tick fired before the victim's sap and keeps its tag"
        );
    }

    /// A spare Pokemon behind the victim, which is the ONE bit c143 §1a varied
    /// between its variants A and B. The victim still faints to its own sap, but the
    /// battle does not end, so nothing is truncated and the slower seeder's 10.4
    /// still runs. Pins the living-reserve half of the predicate: without it the
    /// gate would fire on every residual faint.
    #[test]
    fn a_spare_pokemon_behind_the_victim_keeps_the_leftovers_tag() {
        let mut state = g33b_state(200);
        state.side_two.pokemon[PokemonIndex::P1].hp = 100;
        let segment = vec![damage_two(12), heal_one(12), heal_one(16)];
        assert_eq!(
            residual_tags(&mut state, &segment, "p1a"),
            vec!["item: Leftovers".to_string()],
            "a non-final faint truncates nothing, so the seeder's tick still fires"
        );
    }

    /// An exact speed tie is deliberately NOT gated. `speedSort` shuffles a tie
    /// (`sim/battle.ts:455-457`) and the engine forks BOTH residual orders, keeping
    /// both when they differ -- so one live order fires the tick and the other does
    /// not, and there is no single answer. The pre-gate booking is retained rather
    /// than guessed, which is a documented under-reach and not an oversight.
    #[test]
    fn a_speed_tie_is_not_gated() {
        let state = g33b_state(100);
        assert_eq!(
            leftovers_slot_truncated(&state, &[damage_two(12), heal_one(12)]),
            [false, false],
            "a tie forks into both orders; the gate must not pick one"
        );
    }

    /// Future Sight is order **11**, after every order-10 handler on BOTH sides, so
    /// a kill by the winner's Future Sight lands AFTER the winner's own 10.4. The
    /// tick fired and its slot must stay booked.
    ///
    /// Excluded by state predicate rather than by classifying the instruction: a
    /// lethal residual damage always equals the victim's remaining HP exactly, so
    /// the instruction itself carries no information about which phase produced it.
    #[test]
    fn a_future_sight_kill_is_not_gated() {
        let mut state = g33b_state(200);
        state.side_one.future_sight.0 = 1;
        // Side two's sap is survivable here; the Future Sight tick at order 11 is
        // what finishes it, one entry after side one's Leftovers.
        state.side_two.get_active().hp = 20;
        let segment = vec![damage_two(12), heal_one(12), heal_one(16), damage_two(8)];
        assert_eq!(
            leftovers_slot_truncated(&state, &segment),
            [false, false],
            "order 11 is after the winner's 10.4, so nothing was truncated"
        );
    }

    /// Perish Song is order **12**, likewise after all of order 10. Same argument
    /// as Future Sight, different section.
    #[test]
    fn a_perish_song_kill_is_not_gated() {
        let mut state = g33b_state(200);
        state
            .side_two
            .volatile_statuses
            .insert(PokemonVolatileStatus::PERISH1);
        state.side_two.get_active().hp = 20;
        let segment = vec![damage_two(12), heal_one(12), heal_one(16), damage_two(8)];
        assert_eq!(
            leftovers_slot_truncated(&state, &segment),
            [false, false],
            "order 12 is after the winner's 10.4, so nothing was truncated"
        );
    }

    /// Liquid Ooze is the only residual effect that writes a NEGATIVE `Heal`, and it
    /// writes it at the SEEDED side's 10.5 -- inside the winner's own bucket, one
    /// sub-order after the winner's 10.4. So a battle ended by ooze recoil comes
    /// after the tick, and this is the one arm separated structurally (by
    /// instruction kind) rather than by a state predicate.
    #[test]
    fn a_liquid_ooze_kill_is_not_gated() {
        let mut state = g33b_state(200);
        // Side ONE is the seeded one here, and its ooze turns the sap into recoil on
        // the seeder. Side two is the faster loser.
        state
            .side_one
            .volatile_statuses
            .insert(PokemonVolatileStatus::LEECHSEED);
        state.side_one.get_active().ability = Abilities::LIQUIDOOZE;
        let segment = vec![heal_one(16), damage_one(33), heal_two(-12)];
        assert_eq!(
            leftovers_slot_truncated(&state, &segment),
            [false, false],
            "ooze recoil fires at 10.5, after the winner's 10.4"
        );
    }

    /// NB-3 from the review of #1120: deleting the LIQUID OOZE guard from
    /// `residual_heal_cause` left the ENTIRE crate suite green, so the guard was
    /// unpinned. Liquid Ooze reverses the drain, so a seeded opponent carrying it
    /// produces no drain heal on this side at all -- any positive heal must be
    /// something else, and with the reorder that something else is Leftovers.
    #[test]
    fn liquid_ooze_on_the_seeder_means_a_heal_here_is_not_the_drain() {
        let mut state = parse_state(MINIMAL.trim()).expect("fixture parses");
        state.side_one.get_active().item = Items::NONE;
        state
            .side_two
            .volatile_statuses
            .insert(PokemonVolatileStatus::LEECHSEED);
        state.side_two.get_active().ability = Abilities::LIQUIDOOZE;
        let heal = Instruction::Heal(poke_engine::instruction::HealInstruction {
            side_ref: SideReference::SideOne,
            heal_amount: 6,
        });

        let mut rendered = RenderedEvents::default();
        let mut sim = Sim::new(&mut state, [false, false]);
        let mut plan = ResidualPlan::default();
        render_residual_instruction(&mut sim, &heal, None, &mut plan, &ctx(), &mut rendered);
        sim.finish();
        // With Liquid Ooze there is no drain to attribute, so this must NOT go
        // silent -- going silent is what dropping the guard would cause.
        assert!(
            !rendered.lines[0].contains("[silent]"),
            "a heal that cannot be the drain was rendered as one: {:?}",
            rendered.lines
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

#[cfg(test)]
mod none_matched_shape_tests {
    use super::*;
    use poke_engine::instruction::{
        BoostInstruction, DamageInstruction, HealInstruction, RemoveVolatileStatusInstruction,
        SetLastUsedMoveInstruction, SwitchInstruction,
    };
    use poke_engine::state::{
        LastUsedMove, PokemonBoostableStat, PokemonIndex, PokemonMoveIndex, PokemonType,
    };

    fn dmg(amount: i16) -> Instruction {
        Instruction::Damage(DamageInstruction {
            side_ref: SideReference::SideOne,
            damage_amount: amount,
        })
    }

    fn heal(amount: i16) -> Instruction {
        Instruction::Heal(HealInstruction {
            side_ref: SideReference::SideOne,
            heal_amount: amount,
        })
    }

    /// The whole point of the split is OWNERSHIP, so the two verdicts with different owners
    /// must not collapse.
    ///
    /// `values_only` says the renderer regenerated the right transition and disagreed about a
    /// NUMBER -- a damage roll, or a chance branch the engine merged -- neither of which the
    /// renderer can fix. `structure` says it regenerated a different transition, which is a
    /// candidate-set or state-input bug and IS fixable here. The code records one instance
    /// already: passing `Choice::default()` for the defender made the engine's 32-roll
    /// enumeration mismatch, and every affected world refused as `none_matched`.
    #[test]
    fn a_numeric_disagreement_is_values_only_and_a_variant_swap_is_structure() {
        assert_eq!(
            divergence_shape(&[dmg(30)], &[dmg(31)]),
            NoneMatchedShape::ValuesOnly,
            "same variant, different number: a roll disagreement"
        );
        assert_eq!(
            divergence_shape(&[dmg(30)], &[heal(30)]),
            NoneMatchedShape::Structure,
            "different variant at the same position: a different transition"
        );
    }

    /// SIDE is part of the shape, not a value. Review measured every one of these reporting
    /// `ValuesOnly` before the predicate compared `instruction_side`, which pointed a
    /// wrong-target renderer bug at the engine.
    #[test]
    fn a_side_difference_is_structural_not_numeric() {
        let one = Instruction::Damage(DamageInstruction {
            side_ref: SideReference::SideOne,
            damage_amount: 30,
        });
        let two = Instruction::Damage(DamageInstruction {
            side_ref: SideReference::SideTwo,
            damage_amount: 30,
        });
        assert_eq!(
            divergence_shape(std::slice::from_ref(&one), std::slice::from_ref(&two)),
            NoneMatchedShape::Structure,
            "the same variant on the OTHER side is a wrong-target bug, not a roll disagreement"
        );
        // ...and the same side with a different number is still numeric, so the fix did not
        // simply collapse everything into `Structure`.
        let one_bigger = Instruction::Damage(DamageInstruction {
            side_ref: SideReference::SideOne,
            damage_amount: 31,
        });
        assert_eq!(
            divergence_shape(std::slice::from_ref(&one), std::slice::from_ref(&one_bigger)),
            NoneMatchedShape::ValuesOnly
        );
    }

    /// An empty candidate LIST is not an empty candidate BRANCH.
    #[test]
    fn no_candidates_is_distinct_from_an_empty_branch() {
        assert_ne!(
            NoneMatchedShape::NoCandidates.token(),
            NoneMatchedShape::Empty.token()
        );
        // LAST in declaration order, so `min` never prefers it over a real observation.
        assert_eq!(
            NoneMatchedShape::NoCandidates.min(NoneMatchedShape::Empty),
            NoneMatchedShape::Empty
        );
    }

    #[test]
    fn a_length_difference_outranks_a_variant_difference() {
        // Checked BEFORE the variant scan, because zipping unequal lengths would silently
        // compare only the shorter prefix and could report `values_only` for a tail that is
        // missing instructions entirely. That ordering is unchanged by the containment
        // split -- all three land ahead of the scan; they only say WHICH length difference.
        //
        // These two fixtures were asserted as bare `Length` before the split. Both are
        // containment cases, which is exactly why the bare token could not be acted on.
        assert_eq!(
            divergence_shape(&[dmg(30)], &[dmg(30), heal(10)]),
            NoneMatchedShape::BranchIsPrefix,
            "the branch reproduces the head of the tail and the tail continues past it"
        );
        assert_eq!(
            divergence_shape(&[dmg(30), heal(10)], &[dmg(30)]),
            NoneMatchedShape::TailIsPrefix,
            "the tail is reproduced and the branch continues past it -- the mirror case, and \
             it must NOT collapse into the branch-shorter bucket"
        );
        // NEITHER contains the other: a genuinely different transition. This is the reading
        // the bare `Length` token was always assumed to carry and, before the split, could
        // not establish -- era 61's 4,786 worlds were all reported under it.
        assert_eq!(
            divergence_shape(&[heal(10)], &[dmg(30), heal(10)]),
            NoneMatchedShape::Length,
            "a shorter list that is not a PREFIX of the longer is a real structural miss"
        );
    }

    /// `bit()` must be UNIQUE, CONTIGUOUS and inside the `u8` bitset.
    ///
    /// Review mutated `BranchIsPrefix => 2` to `=> 0` and the whole suite stayed GREEN. Two
    /// shapes sharing a bit makes them indistinguishable in `NoneMatchedShapes`, so inserting
    /// `BranchIsPrefix` would ALSO emit `shape_same_variants_and_sides` -- the one token whose
    /// doc spends fourteen lines warning it must not be read as an ownership verdict --
    /// silently mislabelling the largest failure class. This PR moved three of these values
    /// and added two, which is exactly when the guard was missing.
    #[test]
    fn the_shape_bits_are_unique_contiguous_and_fit_the_bitset() {
        let bits: Vec<u8> = NoneMatchedShape::ALL.iter().map(|s| s.bit()).collect();
        assert_eq!(
            bits,
            (0..NoneMatchedShape::ALL.len() as u8).collect::<Vec<u8>>(),
            "shape bits must be unique and contiguous from 0"
        );
        assert!(
            NoneMatchedShape::ALL.len() <= 8,
            "NoneMatchedShapes is a u8 bitset: {} shapes will not fit",
            NoneMatchedShape::ALL.len()
        );
        // ROUND TRIP: a set holding exactly one shape yields exactly that shape. This is what
        // a duplicated bit actually breaks, and asserting the bits alone would not catch a
        // mismatch between `bit()` and `iter()`.
        for shape in NoneMatchedShape::ALL {
            let mut only = NoneMatchedShapes::default();
            only.insert(shape);
            assert_eq!(
                only.iter().collect::<Vec<_>>(),
                vec![shape],
                "{shape:?} did not round-trip through the bitset alone"
            );
        }
    }

    /// The PROTECT marker: rendered only when the state proves it, refused otherwise.
    ///
    /// Era 62 measured this shape at 3,365 worlds -- 33.0% of world failures and the ENTIRE
    /// `heal` family. It is a TWO-PRODUCER shape (Protect-blocked branch vs full-HP absorb
    /// no-op) and both producers push a zero `Heal` on the DEFENDER, so every one of these
    /// cases turns on the state facts rather than the tail.
    #[test]
    fn the_protect_marker_renders_only_when_the_state_proves_it() {
        let zero_heal_on_defender = [Instruction::Heal(HealInstruction {
            side_ref: SideReference::SideTwo,
            heal_amount: 0,
        })];
        let atk = SideReference::SideOne;

        // PROTECTED, no absorb ability -> RENDERABLE, and it names the DEFENDER.
        assert_eq!(
            protect_blocked_marker_side(&zero_heal_on_defender, 0, atk, true, false),
            Some(SideReference::SideTwo)
        );
        // ...and the classifier must then report NOTHING blocking, or the walk renders the
        // line while the tail still refuses -- the two would disagree about one tail.
        assert_eq!(
            unrenderable_family_at_with_protect(&zero_heal_on_defender, 0, atk, true, false),
            None
        );
        assert!(ambiguous_tail_is_fully_renderable_with_protect(
            &zero_heal_on_defender, atk, true, false
        ));

        // NO PROTECT VOLATILE -> refuse. This is the pre-existing behaviour and the
        // fail-closed default every three-argument caller still gets.
        assert_eq!(
            protect_blocked_marker_side(&zero_heal_on_defender, 0, atk, false, false),
            None
        );
        assert_eq!(
            unrenderable_family_at_with_protect(&zero_heal_on_defender, 0, atk, false, false),
            Some("heal_zero_marker")
        );
        assert_eq!(
            unrenderable_family_at(&zero_heal_on_defender, 0, atk),
            Some("heal_zero_marker"),
            "the three-argument wrapper must keep refusing -- it is what every REMAINING \
             caller uses, and they are all tests: production moved to _with_protect"
        );

        // THE ABSORB NO-OP IS POSSIBLE -> refuse EVEN WITH the volatile. The flag now means
        // "ability present AND its 25% heal clamps to zero AND some callee's modified choice
        // still carries the converted opponent heal" -- the HP clause is #1211's, the callee
        // clause is this PR's, and both are computed at the production read site. The axis is
        // still right: `ability_modify_attack_against` runs BEFORE the Protect gate and
        // RESTORES `flags.protect`, so a protect-bypassing Water move (WATERSPORT, pinned by
        // `the_absorb_bypass_producer_is_real` and driven end-to-end by
        // `protect_plus_a_bypassing_absorbed_callee_refuses_rather_than_guessing`) keeps its
        // converted heal while the volatile is set. Rendering `Protect` over an ability
        // activation corrupts a searched world.
        assert_eq!(
            protect_blocked_marker_side(&zero_heal_on_defender, 0, atk, true, true),
            None
        );
        assert_eq!(
            unrenderable_family_at_with_protect(&zero_heal_on_defender, 0, atk, true, true),
            Some("heal_zero_marker")
        );
    }

    /// The CONJUNCT IS LIVE, not vacuous -- pinned through the ENGINE's own modification pass
    /// rather than a hand-built `Choice`.
    ///
    /// This test exists because `render_move_phase` cited it by name and it did not exist.
    /// Review found the citation resolving to nothing but its own comment, which is the
    /// "instrument that cannot report failure" shape wearing a citation's clothes. The claim it
    /// was cited for is the load-bearing half of the whole guard: that
    /// `choice_can_convert_an_opponent_heal` can actually be TRUE in production, so the new
    /// conjunct still refuses the genuinely ambiguous case rather than being a constant
    /// `false` dressed as a predicate.
    ///
    /// Pinned at the SCAN, not at the predicate, and that is the point. The sibling
    /// `only_an_opponent_targeted_positive_heal_can_reach_the_absorb_no_op` feeds the
    /// predicate a `Choice` this test file constructed, so it cannot show that anything in the
    /// engine ever sets the field. Here `identify_sleep_talk_called` regenerates the callees
    /// through `generate_instructions_from_move`, so the flag is TRUE only if the absorb
    /// conversion really wrote `heal` and `remove_effects_for_protect` really left it alone.
    /// The control is the same fixture with protect-flagged callees, where the strip does
    /// happen and the flag must be FALSE -- without it, a mutant hardcoding the flag `true`
    /// would pass.
    #[test]
    fn the_bypassing_callee_still_refuses() {
        let mut state = State::default();
        state.side_two.get_active().ability = Abilities::WATERABSORB;
        let maxhp = state.side_two.get_active().maxhp;
        state.side_two.get_active().hp = maxhp;
        state
            .side_two
            .volatile_statuses
            .insert(PokemonVolatileStatus::PROTECT);
        state.side_one.get_active().status = PokemonStatus::SLEEP;

        let mut probe_flag = |callees: [Choices; 2]| {
            state
                .side_one
                .get_active()
                .replace_move(PokemonMoveIndex::M0, Choices::SLEEPTALK);
            state
                .side_one
                .get_active()
                .replace_move(PokemonMoveIndex::M1, callees[0]);
            state
                .side_one
                .get_active()
                .replace_move(PokemonMoveIndex::M2, callees[1]);
            let mut outer = poke_engine::choices::MOVES
                .get(&Choices::SLEEPTALK)
                .unwrap()
                .clone();
            outer.move_id = Choices::SLEEPTALK;
            let tail = [Instruction::Heal(HealInstruction {
                side_ref: SideReference::SideTwo,
                heal_amount: 0,
            })];
            identify_sleep_talk_called(
                &mut state,
                SideReference::SideOne,
                &Choice::default(),
                &outer,
                &tail,
                false,
            )
            .callee_can_convert_an_opponent_heal
        };

        assert!(
            probe_flag([Choices::WATERSPORT, Choices::TACKLE]),
            "the engine's own modification pass did not leave an opponent-targeted heal on a \
             protect-BYPASSING absorbed callee, so the conjunct that keeps the genuinely \
             ambiguous case refused is vacuous and the guard is a fail-open"
        );
        assert!(
            !probe_flag([Choices::SURF, Choices::TACKLE]),
            "a protect-FLAGGED absorbed callee still carried its converted heal, so \
             remove_effects_for_protect no longer strips it and the two producers are no \
             longer mutually exclusive per callee"
        );
    }

    /// The bypassing-producer set, DERIVED from the engine's gates instead of from the move
    /// type alone -- and the correction to a figure this change published wrong.
    ///
    /// The PR body first reported the set as "the five Water moves without the protect flag",
    /// counted 27 / 1682 pool carriers for `raindance`, and then said Rain Dance never reaches
    /// the producer. Review was right that this is incoherent: a move that cannot reach the
    /// producer is not in the set, so the count was of something irrelevant. The gate the
    /// first enumeration missed is `ability_modify_attack_against`'s own first statement --
    /// `if attacker_choice.target != MoveTarget::Opponent { return; }` -- whose comment names
    /// Rain Dance as the reason it exists.
    ///
    /// So the set is: absorbed move TYPE, `target: Opponent`, and no protect flag. For
    /// `VOLTABSORB` gen3 adds `category != Status`, which no unflagged Electric move can
    /// satisfy. Derived here so the PR's pool claim rests on a gate-accurate set.
    #[test]
    fn the_bypassing_producer_set_is_derived_from_every_gate() {
        let bypassing = |want: PokemonType, status_gated: bool| -> Vec<Choices> {
            poke_engine::choices::MOVES
                .iter()
                .filter(|(_, c)| {
                    c.move_type == want
                        && c.target == MoveTarget::Opponent
                        && !c.flags.protect
                        && !(status_gated && c.category == MoveCategory::Status)
                })
                .map(|(id, _)| *id)
                .collect()
        };
        let mut water = bypassing(PokemonType::WATER, false);
        water.sort_by_key(|c| format!("{c:?}"));
        assert_eq!(
            water,
            vec![Choices::WATERSPORT],
            "the Water-side bypassing-producer set changed; the PR's pool-reachability claim \
             is derived from exactly this list and must be recounted"
        );
        assert!(
            bypassing(PokemonType::ELECTRIC, true).is_empty(),
            "gen3 gates Volt Absorb on `category != Status` and every unflagged Electric move \
             is a Status move; an Electric bypassing producer now exists: {:?}",
            bypassing(PokemonType::ELECTRIC, true)
        );
        // THE NEAR MISS, kept explicit: Rain Dance is Water-typed and unflagged, and is
        // excluded by the TARGET gate alone. Dropping that gate is what produced the wrong
        // published figure.
        let raindance = poke_engine::choices::MOVES.get(&Choices::RAINDANCE).unwrap();
        assert_eq!(raindance.move_type, PokemonType::WATER);
        assert!(!raindance.flags.protect);
        assert_ne!(raindance.target, MoveTarget::Opponent);
    }

    /// The DISCRIMINATOR: producer 2's own `if`, read off the callee rather than approximated
    /// from the defender.
    ///
    /// Pinned as a pure predicate, separately from the read site that ANDs it in, for the
    /// reason `sleeptalk_refusal_is_unsafe` states about itself: the two halves have to be
    /// able to fail apart, and a wrong bound here is a WRONG RENDERED LINE rather than an
    /// extra refusal.
    #[test]
    fn only_an_opponent_targeted_positive_heal_can_reach_the_absorb_no_op() {
        let with = |heal| {
            let mut choice = Choice::default();
            choice.heal = heal;
            choice
        };
        // No heal at all -- every protect-BLOCKED callee, because
        // `remove_effects_for_protect` sets `heal = None`. This is the shape all 31 census
        // refusals present on both of their matching callees.
        assert!(!choice_can_convert_an_opponent_heal(&with(None)));
        // A SELF heal: Rest, Recover, Softboiled, Moonlight... i.e. every native `heal` in
        // the move table. Never producer 2, whose site reads `target == Opponent`.
        assert!(!choice_can_convert_an_opponent_heal(&with(Some(
            poke_engine::choices::Heal { target: MoveTarget::User, amount: 0.5 }
        ))));
        // THE ABSORB CONVERSION, and the only writer of this shape: Water Absorb, Volt
        // Absorb and Dry Skin all set exactly this.
        assert!(choice_can_convert_an_opponent_heal(&with(Some(
            poke_engine::choices::Heal { target: MoveTarget::Opponent, amount: 0.25 }
        ))));
        // `> 0.0` and not `>= 0.0`, mirroring the producer. A zero-fraction heal writes no
        // instruction at all, so admitting it would refuse worlds for nothing -- and a
        // NEGATIVE opponent heal is not this producer either.
        assert!(!choice_can_convert_an_opponent_heal(&with(Some(
            poke_engine::choices::Heal { target: MoveTarget::Opponent, amount: 0.0 }
        ))));
        assert!(!choice_can_convert_an_opponent_heal(&with(Some(
            poke_engine::choices::Heal { target: MoveTarget::Opponent, amount: -0.5 }
        ))));
    }

    /// The claim the discriminator rests on, MACHINE-CHECKED against the engine's own table
    /// instead of asserted in a comment.
    ///
    /// `choice_can_convert_an_opponent_heal` is read as "an absorb ability converted this
    /// callee", and that reading is only sound while NO move carries an opponent-targeted
    /// heal natively. A future engine bump that adds one (Pollen Puff is the obvious
    /// candidate) would make the field ambiguous again and must fail here rather than
    /// silently widen what the walk renders.
    ///
    /// The `amount > 0.0` filter matches the predicate: a nonpositive native opponent heal
    /// would be harmless because the producer ignores it too.
    #[test]
    fn no_move_in_the_table_natively_heals_the_opponent() {
        let offenders: Vec<Choices> = poke_engine::choices::MOVES
            .iter()
            .filter(|(_, choice)| choice_can_convert_an_opponent_heal(choice))
            .map(|(id, _)| *id)
            .collect();
        assert!(
            offenders.is_empty(),
            "an opponent-targeted positive heal is no longer unique to the absorb \
             abilities' conversion, so it can no longer discriminate the two zero-heal \
             producers: {offenders:?}"
        );
    }

    /// #1211: an absorb ability with HP HEADROOM cannot have produced the marker.
    ///
    /// This is the whole behaviour change, pinned at the predicate that decides it, and
    /// SEPARATELY from the flag that carries it -- the flag is computed at the production
    /// read site (`render_move_phase`) and the two halves have to be able to fail apart.
    ///
    /// Producer 2 pushes a zero `Heal` only in the `health_recovered == 0` else-branch of
    /// `gen3/generate_instructions.rs:1405-1424`. A defender below full HP takes a REAL heal
    /// with a nonzero amount, which is a different instruction routed to `heal_defender`. So
    /// the capture at `fb3m21-946004` round 45 -- Mantine, Water Absorb, PROTECT held,
    /// 192/252 -- was a world thrown away over an instruction its ability could not emit.
    #[test]
    fn absorb_headroom_makes_the_no_op_impossible() {
        // 192/252, the captured HP: 25% is 63, and 192 + 63 = 255 > 252 clamps to 60, not
        // to zero. Producer 2 could not have written a zero here.
        assert!(!absorb_heal_clamps_to_zero(192, 252));
        assert!(!absorb_heal_clamps_to_zero(237, 252));
        // FULL HP is the case that still refuses -- 227/227 (Jolteon) and 335/335 (Vaporeon)
        // are both from the same capture, at the production budget.
        assert!(absorb_heal_clamps_to_zero(227, 227));
        assert!(absorb_heal_clamps_to_zero(335, 335));
        // ONE POINT BELOW FULL is the boundary, and it is the direction that matters: the
        // clamp yields exactly 1, so the engine writes a REAL heal and never the marker.
        assert!(!absorb_heal_clamps_to_zero(251, 252));
        assert!(absorb_heal_clamps_to_zero(252, 252));
        // TRUNCATION, which is why this mirrors the engine's arithmetic instead of testing
        // `hp == maxhp`. At maxhp 3 the 25% heal truncates to 0 and the no-op is possible
        // even with headroom, so the simplification would be fail-OPEN here. Unreachable in
        // the gen3 randbat pool, pinned so it stays that way if the pool changes.
        assert!(absorb_heal_clamps_to_zero(1, 3));
        assert!(!absorb_heal_clamps_to_zero(1, 4));
    }

    /// The marker is the DEFENDER's, and only a ZERO heal.
    ///
    /// Both producers push on `get_other_side()` / the target, so a zero `Heal` on the
    /// ATTACKER is neither of them and must not be dressed as Protect. And a NON-zero heal is
    /// an ordinary heal whose own arms already decide it.
    #[test]
    fn the_protect_marker_is_defender_side_and_zero_amount_only() {
        let atk = SideReference::SideOne;
        let on_attacker = [Instruction::Heal(HealInstruction {
            side_ref: SideReference::SideOne,
            heal_amount: 0,
        })];
        assert_eq!(
            protect_blocked_marker_side(&on_attacker, 0, atk, true, false),
            None,
            "a zero heal on the ATTACKER is neither producer"
        );
        let nonzero_on_defender = [Instruction::Heal(HealInstruction {
            side_ref: SideReference::SideTwo,
            heal_amount: 30,
        })];
        assert_eq!(
            protect_blocked_marker_side(&nonzero_on_defender, 0, atk, true, false),
            None,
            "a POSITIVE heal on the defender is an absorb, not a Protect marker"
        );
        // Out of bounds returns None rather than panicking: pyo3 maps a panic to
        // PanicException, which escapes `except Exception` and kills the campaign worker.
        assert_eq!(protect_blocked_marker_side(&on_attacker, 9, atk, true, false), None);

        // NEGATIVE heal on the defender is Liquid Ooze, not a Protect marker. Pins `== 0`
        // against `<= 0`, which survived the mutation battery: era 62 measured
        // heal_liquidooze at zero so it is unreachable today, but widening the test would
        // dress a Liquid Ooze tail as Protect the moment one appears.
        let liquid_ooze = [Instruction::Heal(HealInstruction {
            side_ref: SideReference::SideTwo,
            heal_amount: -40,
        })];
        assert_eq!(
            protect_blocked_marker_side(&liquid_ooze, 0, atk, true, false),
            None,
            "a NEGATIVE heal is Liquid Ooze; the marker is zero-amount only"
        );
    }

    /// The absorb set is exactly the abilities that can emit a zero `Heal`.
    ///
    /// Pins both directions, because both mutations survived. Narrowing it drops the guard
    /// for a real producer and would render Protect over a Water Absorb activation; widening
    /// it back to `is_absorb_ability` re-adds FLASHFIRE, which sets a VOLATILE and never a
    /// heal, refusing Protect-blocked worlds for a Flash Fire defender and buying nothing.
    #[test]
    fn the_absorb_guard_covers_exactly_the_zero_heal_producers() {
        for a in [Abilities::WATERABSORB, Abilities::VOLTABSORB, Abilities::DRYSKIN] {
            assert!(
                absorb_ability_can_emit_a_zero_heal(a),
                "{a:?} carries a heal and CAN emit the zero-heal no-op"
            );
        }
        assert!(
            !absorb_ability_can_emit_a_zero_heal(Abilities::FLASHFIRE),
            "FLASHFIRE sets a volatile and never a heal, so guarding on it is pure loss"
        );
        assert!(!absorb_ability_can_emit_a_zero_heal(Abilities::NONE));
    }

    /// The absorb axis cannot be retired on the argument that Protect always gets there
    /// first. A protect-BYPASSING absorb-triggering move exists.
    ///
    /// This is the counterexample that keeps a FULL-HP absorber refused, and it is pinned
    /// against the engine's own move table rather than asserted in prose, because the prose
    /// version of this claim has already been wrong twice in
    /// `protect_blocked_marker_side`'s doc -- once in each direction.
    ///
    /// The mechanism it stands for: `before_move` calls `ability_modify_attack_against`
    /// BEFORE the Protect gate, and the absorb arms deliberately RESTORE `flags.protect`
    /// after `remove_all_effects()`. A move that carries the flag then has its converted
    /// heal stripped again by `remove_effects_for_protect` (`heal = None`), so producer 2
    /// dies and producer 1 writes the marker. A move WITHOUT the flag keeps the heal, so a
    /// full-HP defender gets a zero `Heal` that means ability activation, not Protect --
    /// while the PROTECT volatile is set. Both producers reach the same instruction from the
    /// same visible state, which is why full HP has to stay refused.
    ///
    /// `target: Opponent` is part of the requirement, not decoration: the gen3 fidelity fix
    /// at the top of `ability_modify_attack_against` returns early for self/field moves, and
    /// it is what stops Rain Dance -- also Water-typed -- from being a second counterexample.
    ///
    /// WHAT WATERSPORT IS, stated because "a gen3 mechanic makes this reachable" would be the
    /// wrong lesson. In real Showdown Water Sport is SELF-targeting; it reaches
    /// `ability_modify_attack_against` at all only because poke-engine's `Choice::default()`
    /// sets `target: Opponent` and this move's entry does not override it. The counterexample
    /// is therefore an ENGINE-DATA ARTIFACT, not a property of the generation. The guard is
    /// still right, for a reason that does not weaken with that correction: the renderer must
    /// match the engine that EMITTED the instruction it is describing, not the cartridge. If
    /// the entry is ever given `target: MoveTarget::User` -- the faithful fix, and the one
    /// Rain Dance already received -- this test fails LOUDLY and the absorb axis can be
    /// revisited on evidence rather than quietly kept.
    ///
    /// The third leg is the one that would have broken `ABSORB_HEAL_FRACTION`: producer 2 is
    /// reachable ONLY from an ability, because no move in the table carries a heal aimed at
    /// the opponent. One fraction constant is sufficient exactly while that holds.
    #[test]
    fn the_absorb_bypass_producer_is_real() {
        // NO MOVE is a producer-2 source, so the ability arms -- all `amount: 0.25` -- are
        // the complete producer set and one constant covers them.
        let move_borne: Vec<Choices> = poke_engine::choices::MOVES
            .iter()
            .filter(|(_, choice)| {
                choice
                    .heal
                    .as_ref()
                    .is_some_and(|heal| heal.target == MoveTarget::Opponent)
            })
            .map(|(id, _)| *id)
            .collect();
        assert!(
            move_borne.is_empty(),
            "a MOVE now carries a heal aimed at the opponent ({move_borne:?}); producer 2 is \
             no longer ability-only and ABSORB_HEAL_FRACTION is no longer one number"
        );
        let watersport = poke_engine::choices::MOVES
            .get(&Choices::WATERSPORT)
            .expect("WATERSPORT is in the engine's move table");
        assert_eq!(watersport.move_type, PokemonType::WATER);
        assert_eq!(watersport.target, MoveTarget::Opponent);
        assert!(
            !watersport.flags.protect,
            "WATERSPORT carrying the protect flag would retire the absorb axis -- if the \
             engine data changes here, re-derive the guard rather than deleting it"
        );
        // Rain Dance is the near miss, and the reason the early return is load-bearing.
        let raindance = poke_engine::choices::MOVES
            .get(&Choices::RAINDANCE)
            .expect("RAINDANCE is in the engine's move table");
        assert_eq!(raindance.move_type, PokemonType::WATER);
        assert_ne!(
            raindance.target,
            MoveTarget::Opponent,
            "the gen3 weather-move targeting fix is what keeps Rain Dance out of \
             ability_modify_attack_against"
        );
    }

    /// The pre-tail state read has to still be true at the marker's index.
    ///
    /// The three defender facts are read ONCE, before any of the tail is applied, because
    /// the classifier and the walk must agree on one tail. That makes them stale-able, and
    /// the stale direction is the SILENT one: a tail that heals the defender to full before
    /// the marker makes producer 2 possible at the marker while the pre-tail HP said it was
    /// not, and the walk would then emit `|-activate|...|Protect` over an ability
    /// activation. A wrong rendered line is worse than an abort, so the prefix refuses.
    #[test]
    fn a_prefix_that_could_move_the_defender_refuses_the_marker() {
        let atk = SideReference::SideOne;
        let marker = Instruction::Heal(HealInstruction {
            side_ref: SideReference::SideTwo,
            heal_amount: 0,
        });
        // BASELINE: the marker alone renders, so any failure below is the prefix.
        assert_eq!(
            protect_blocked_marker_side(&[marker.clone()], 0, atk, true, false),
            Some(SideReference::SideTwo)
        );
        // A heal ON THE DEFENDER before the marker could have closed the headroom the
        // `defender_absorb_zero_heal_possible: false` argument was computed from.
        let healed_first = [
            Instruction::Heal(HealInstruction {
                side_ref: SideReference::SideTwo,
                heal_amount: 60,
            }),
            marker.clone(),
        ];
        assert_eq!(
            protect_blocked_marker_side(&healed_first, 1, atk, true, false),
            None
        );
        // A defender SWITCH invalidates all three facts at once -- different Pokemon,
        // different ability, different HP.
        let switched_first = [
            Instruction::Switch(SwitchInstruction {
                side_ref: SideReference::SideTwo,
                previous_index: PokemonIndex::P0,
                next_index: PokemonIndex::P1,
            }),
            marker.clone(),
        ];
        assert_eq!(
            protect_blocked_marker_side(&switched_first, 1, atk, true, false),
            None
        );
        // The classifier must AGREE, or the walk and the acceptance test disagree about one
        // tail -- the failure this file's "one list, one answer" rule exists to prevent.
        assert_eq!(
            unrenderable_family_at_with_protect(&switched_first, 1, atk, true, false),
            Some("heal_zero_marker")
        );
        // THE FAIL-CLOSED DEFAULT ARM, pinned separately because the battery caught it
        // unpinned: flipping `_ => false` to `_ => true` survived every other assertion
        // here. A volatile removal is the case that makes the arm matter -- it can clear
        // the very `PROTECT` the `defender_protected: true` argument was read from, so
        // admitting unaudited variants renders `Protect` for a defender that no longer
        // holds it.
        let protect_cleared_first = [
            Instruction::RemoveVolatileStatus(RemoveVolatileStatusInstruction {
                side_ref: SideReference::SideTwo,
                volatile_status: PokemonVolatileStatus::PROTECT,
            }),
            marker.clone(),
        ];
        assert_eq!(
            protect_blocked_marker_side(&protect_cleared_first, 1, atk, true, false),
            None
        );
        // AND THE OTHER DIRECTION, so the guard is not simply "any prefix refuses": the
        // same instructions on the ATTACKER's side move none of the three facts, and a
        // tail that only re-baselines the attacker still renders.
        let attacker_side_prefix = [
            Instruction::Heal(HealInstruction {
                side_ref: SideReference::SideOne,
                heal_amount: 60,
            }),
            marker.clone(),
        ];
        assert_eq!(
            protect_blocked_marker_side(&attacker_side_prefix, 1, atk, true, false),
            Some(SideReference::SideTwo)
        );
        // THE POSITIVE ARMS OF THE ALLOWLIST, pinned because independent review showed
        // they were not. Deleting every variant but `SetLastUsedMove` from the `=> true`
        // group survived the whole suite: a safe-direction mutant (lost reclaim, not a
        // wrong render) but exactly the "the suite does not pin the boundary" signal, and
        // the boundary is the whole point of an allowlist. A `Boost` on the DEFENDER is
        // the sharpest case -- it names the refusing side and still moves none of the
        // three facts, because a stat stage is neither HP, nor the active Pokemon, nor a
        // volatile.
        for benign in [
            Instruction::Boost(BoostInstruction {
                side_ref: SideReference::SideTwo,
                stat: PokemonBoostableStat::Defense,
                amount: 1,
            }),
            Instruction::DamageSubstitute(DamageInstruction {
                side_ref: SideReference::SideTwo,
                damage_amount: 25,
            }),
            Instruction::SetLastUsedMove(SetLastUsedMoveInstruction {
                side_ref: SideReference::SideTwo,
                last_used_move: LastUsedMove::Move(PokemonMoveIndex::M0),
                previous_last_used_move: LastUsedMove::None,
            }),
        ] {
            let prefixed = [benign.clone(), marker.clone()];
            assert_eq!(
                protect_blocked_marker_side(&prefixed, 1, atk, true, false),
                Some(SideReference::SideTwo),
                "{benign:?} moves none of the three defender facts and must not cost the \
                 render -- if it is dropped from the allowlist the reclaim silently shrinks"
            );
        }
    }

    /// Containment is checked on FULL instruction equality, not on variant alone.
    ///
    /// `starts_with` uses `PartialEq`, so a branch whose head has the right VARIANTS but wrong
    /// payloads is `Length`, not a containment shape. Getting this wrong would be the worse direction:
    /// it would report "the callee was identified and the tail is over-long" for a tail whose
    /// head the candidate did not actually reproduce, sending the fix at the tail bound when
    /// the candidate is wrong.
    #[test]
    fn containment_compares_payloads_not_just_variants() {
        assert_eq!(
            divergence_shape(&[dmg(30)], &[dmg(31), heal(10)]),
            NoneMatchedShape::Length
        );
        assert_eq!(
            divergence_shape(&[dmg(30)], &[dmg(30), heal(10)]),
            NoneMatchedShape::BranchIsPrefix
        );
    }

    /// An EMPTY branch stays `Empty`, not a containment shape.
    ///
    /// The empty slice is a prefix of everything, so ordering matters: the `is_empty` check
    /// runs first. Collapsing these would fold "the candidate generated nothing" -- the move
    /// did not execute at all -- into "the candidate reproduced the tail's head", which is a
    /// different question with a different owner.
    #[test]
    fn an_empty_branch_is_not_reported_as_containment() {
        assert_eq!(divergence_shape(&[], &[dmg(30)]), NoneMatchedShape::Empty);
    }

    #[test]
    fn an_empty_candidate_branch_is_its_own_shape() {
        // Distinct from `length` deliberately: a candidate that generated NOTHING means the
        // move did not execute in regeneration at all, which is a different question from one
        // that executed and produced a different number of instructions.
        assert_eq!(divergence_shape(&[], &[dmg(30)]), NoneMatchedShape::Empty);
    }

    /// `min` over candidates must keep the CLOSEST miss, or a structurally-unrelated candidate
    /// would mask the one that nearly matched -- and the near-miss is the whole diagnosis.
    #[test]
    fn the_ordering_keeps_the_closest_miss() {
        // THE FULL SEQUENCE over `ALL`, not a hand-picked subset. The previous version listed
        // four variants and so said nothing about any variant added later: review swapped
        // `BranchIsPrefix` and `TailIsPrefix` in declaration order and the suite stayed GREEN,
        // while this test's NAME claims to catch exactly that.
        let mut shapes = NoneMatchedShape::ALL;
        shapes.sort();
        assert_eq!(
            shapes,
            [
                NoneMatchedShape::ValuesOnly,
                NoneMatchedShape::Structure,
                NoneMatchedShape::BranchIsPrefix,
                NoneMatchedShape::TailIsPrefix,
                NoneMatchedShape::Length,
                NoneMatchedShape::Empty,
                NoneMatchedShape::NoCandidates,
            ],
            "the closest-miss ordering changed; `min` over candidates now keeps a different \
             shape and era-over-era keys move"
        );
        assert_eq!(shapes[0], NoneMatchedShape::ValuesOnly);
        assert_eq!(
            NoneMatchedShape::Empty.min(NoneMatchedShape::ValuesOnly),
            NoneMatchedShape::ValuesOnly
        );
    }

    /// An empty TAIL is not a containment case.
    ///
    /// The empty slice is a prefix of everything, so `longer.starts_with(shorter)` is
    /// vacuously true and the containment buckets would absorb tails carrying ZERO
    /// containment evidence. The branch-empty mirror is `Empty`; this is its counterpart.
    #[test]
    fn an_empty_tail_is_not_reported_as_containment() {
        assert_eq!(
            divergence_shape(&[dmg(30)], &[]),
            NoneMatchedShape::Length,
            "an empty tail carries no containment evidence"
        );
    }

    /// Every shape's token must be registered, or the class silently stops being rankable --
    /// the failure the family split exists to prevent.
    #[test]
    fn every_shape_token_is_in_the_subcase_vocabulary() {
        // ITERATE `ALL`, not a hand-picked list. The previous version looped four variants
        // and so said nothing about any variant added later -- which is how a HALF-APPLIED
        // rename shipped: `token()` and `SUBCASE_VOCABULARY` carried the new names while
        // `none_matched_slugs` still emitted the old ones, and `assert_subcase_vocabulary` is
        // a plain `assert!` kept out of `debug_assert!` ON PURPOSE so it survives --release.
        // The first world of the largest failure class would have panicked the wheel.
        for shape in NoneMatchedShape::ALL {
            assert!(
                SUBCASE_VOCABULARY.contains(&shape.token()),
                "{shape:?}'s token {:?} is not registered, so the class stops being rankable",
                shape.token()
            );
            let mut only = NoneMatchedShapes::default();
            only.insert(shape);
            let slug = none_matched_slugs(only).next().expect("one shape, one slug");
            // The SLUG must end with the token. This is what catches a rename applied to
            // `token()` but not to `none_matched_slugs`, and a token-string swap between two
            // variants -- both of which the four-variant loop missed.
            assert!(
                slug.ends_with(shape.token()),
                "{shape:?}: slug {slug:?} does not end with its token {:?}",
                shape.token()
            );
            assert!(
                slug.starts_with(SLEEPTALK_LOSSY_TAG),
                "{shape:?}: slug {slug:?} lost the contract tag prefix"
            );
        }
    }
}

#[cfg(test)]
mod nearest_divergence_tests {
    use super::*;
    use poke_engine::instruction::{DamageInstruction, HealInstruction};

    fn dmg(side: SideReference, amount: i16) -> Instruction {
        Instruction::Damage(DamageInstruction {
            side_ref: side,
            damage_amount: amount,
        })
    }

    fn heal(amount: i16) -> Instruction {
        Instruction::Heal(HealInstruction {
            side_ref: SideReference::SideOne,
            heal_amount: amount,
        })
    }

    /// The aggregation must keep the CLOSEST miss. `max` here reports the farthest, which is
    /// near-universally the least informative bucket -- and it survived the whole suite before
    /// this test existed, because nothing crossed from `divergence_shape` into the loop.
    #[test]
    fn the_aggregation_keeps_the_closest_miss_not_the_farthest() {
        let tail = [dmg(SideReference::SideOne, 30)];
        // One structurally-unrelated candidate branch, one that differs only numerically.
        let far = [heal(10)];
        let near = [dmg(SideReference::SideOne, 31)];
        let branches: Vec<&[Instruction]> = vec![&far, &near];
        assert_eq!(
            nearest_divergence(branches.into_iter(), &tail),
            NoneMatchedShape::ValuesOnly,
            "a structurally-unrelated candidate must not mask the near miss"
        );
    }

    /// Argument order is load-bearing and was unobserved: `divergence_shape(&[], &[d])` is
    /// `Empty` while `divergence_shape(&[d], &[])` is `Length`.
    #[test]
    fn the_argument_order_is_branch_then_tail() {
        let tail = [dmg(SideReference::SideOne, 30)];
        let empty: [Instruction; 0] = [];
        let branches: Vec<&[Instruction]> = vec![&empty];
        assert_eq!(
            nearest_divergence(branches.into_iter(), &tail),
            NoneMatchedShape::Empty,
            "an empty BRANCH against a non-empty tail is `Empty`; swapping the arguments \
             would report `Length`"
        );
    }

    /// A candidate that generated NO branches is `Empty`, not `NoCandidates`.
    ///
    /// Those are different facts: the candidate existed and produced nothing, versus there
    /// being no candidate to regenerate. An earlier version returned `NoCandidates` here,
    /// reintroducing the exact conflation that variant was split out to remove. `NoCandidates`
    /// is now reachable only from the seed.
    #[test]
    fn a_candidate_with_no_branches_is_empty_not_no_candidates() {
        let tail = [dmg(SideReference::SideOne, 30)];
        let branches: Vec<&[Instruction]> = vec![];
        assert_eq!(
            nearest_divergence(branches.into_iter(), &tail),
            NoneMatchedShape::Empty
        );
        assert_ne!(NoneMatchedShape::Empty, NoneMatchedShape::NoCandidates);
    }
}

#[cfg(test)]
fn one_shape(shape: NoneMatchedShape) -> NoneMatchedShapes {
    let mut set = NoneMatchedShapes::default();
    set.insert(shape);
    set
}

/// The dominance test behind `volatile_fail`, pinned on BOTH sides of its crossover.
///
/// A fixture cannot pin the below-crossover side: no gen3 move in the family sits at or
/// below 50% accuracy (measured — `scripts/c157_no_effect_hit_reach.py`), so a
/// renderer-level test would only ever exercise the true branch and a mutant that
/// returned `true` unconditionally would survive it. Pinning the predicate directly is
/// what makes the boundary a boundary rather than a coincidence of the move table.
#[cfg(test)]
mod no_effect_hit_dominance {
    use super::*;

    /// Above the crossover the successful no-op hit is the larger half, below it the miss
    /// is, and AT it neither is — which is the arm a `>=` mutation would open.
    ///
    /// `assert!(!f(50.0))` is the mutate-toward-SAFER control. Suppressing `|[miss]|` at a
    /// dead tie is the "safer-looking" variant (it refuses to call a coin flip a miss), and
    /// a boundary that tolerates it is a boundary nothing pins. The three constant mutants
    /// die here too: `true` on 30.0, `false` on 90.0, and `>=` on 50.0.
    #[test]
    fn the_crossover_is_at_fifty_percent_and_the_tie_is_not_dominance() {
        assert!(no_effect_hit_outweighs_miss(90.0), "0.90 hit vs 0.10 miss");
        assert!(no_effect_hit_outweighs_miss(55.0), "0.55 hit vs 0.45 miss");
        assert!(
            no_effect_hit_outweighs_miss(50.000_01),
            "just above the crossover must still be dominance"
        );
        assert!(
            !no_effect_hit_outweighs_miss(50.0),
            "a dead tie is NOT dominance: 0.50 hit vs 0.50 miss renders either way and \
             this predicate must not claim one of them"
        );
        assert!(
            !no_effect_hit_outweighs_miss(30.0),
            "0.30 hit vs 0.70 miss: |[miss]| is the CORRECT label there and must survive"
        );
        assert!(!no_effect_hit_outweighs_miss(0.0));
    }

    /// The whole sub-100% opponent-target volatile family, read out of the ENGINE'S OWN
    /// move table rather than transcribed, is above the crossover — so `volatile_fail`
    /// suppressing `|[miss]|` is never the minority render in gen3.
    ///
    /// A renamed or retuned move is a loud failure (`expect`) rather than a silent skip,
    /// which is the failure mode a hand-copied accuracy table had. The count is asserted
    /// so the loop cannot pass over zero moves.
    #[test]
    fn every_gen3_volatile_fail_carrier_is_above_the_crossover() {
        let family = [
            Choices::SUPERSONIC,
            Choices::SWEETKISS,
            Choices::DISABLE,
            Choices::LEECHSEED,
            Choices::SWAGGER,
        ];
        let mut checked = 0;
        for move_id in family {
            let accuracy = poke_engine::choices::MOVES
                .get(&move_id)
                .unwrap_or_else(|| panic!("{move_id:?} is absent from the engine move table"))
                .accuracy;
            assert!(
                accuracy < 100.0,
                "{move_id:?} at {accuracy} is no longer in the sub-100% family; the \
                 predicate's scope note is stale"
            );
            assert!(
                no_effect_hit_outweighs_miss(accuracy),
                "{move_id:?} at {accuracy}% would make |[miss]| suppression the MINORITY \
                 render: hit {} vs miss {}",
                accuracy / 100.0,
                1.0 - accuracy / 100.0
            );
            checked += 1;
        }
        assert_eq!(checked, 5, "the family loop must not pass over zero moves");
    }
}
