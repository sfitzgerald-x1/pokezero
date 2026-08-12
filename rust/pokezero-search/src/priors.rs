//! Model-prior GATHER and APPLY: the seam between a model's flat per-action
//! prior rows and a decision node's own per-arm PUCT priors.
//!
//! Deliberately OUTSIDE the `model` cargo feature even though its only caller
//! is `crate::model`. The logic here is pure arithmetic over `&[f32]` and
//! `&[Option<usize>]` — it has no tch dependency — and behind the feature gate
//! it could only be tested on a host with a working libtorch, which is exactly
//! the "unrun in the environment where it matters" failure the opponent-priors
//! work is trying not to repeat. Keeping it here makes the whole path testable
//! under a plain `cargo test`.
//!
//! Two rules hold everywhere in this module, and every test below is written
//! to fail if either is broken:
//!
//! * **SLOT ORDER.** A gathered vector is in the DECISION NODE's option order,
//!   and the value at option `i` is the model's prior for action-block slot
//!   `map[i]`. The map is not sorted, not contiguous, and not an identity —
//!   `map[i]` is wherever the leaf encoder's action surface put that option.
//! * **NO REFLECTION AT THE SEAT BOUNDARY.** Priors are per-seat action
//!   distributions applied to the seat that OWNS the actions. The opponent
//!   head is gathered through the OPPONENT's map and written to the
//!   OPPONENT's stat vector. Only *values* flip between seats, and that
//!   happens elsewhere (`multiply_batched_encoded_core`). The single `!` that
//!   turns the searching seat into the opponent seat lives in
//!   [`PriorSeat::owning_side_one`] and NOWHERE else — see the note below.
//!
//! ## Why the seat and the head are one parameter
//!
//! An earlier shape took `side_one: bool` and `seat: PriorSeat` as independent
//! arguments and let the caller pick the flat array too. That put the
//! invariant coupling them — acting seat ⇔ acting head ⇔ `self_side_one` —
//! in `model.rs`, behind the `model` feature, where no test on a
//! libtorch-less host can reach it. Independent review demonstrated the cost:
//! four separate one-token mutations at those call sites (opponent head onto
//! the searching seat's arms, opponent vector into the acting seat's storage
//! slot, the same swap at the root, and feeding the acting head to the
//! opponent resolve) all COMPILED and all passed the entire suite, including
//! under `--features model`. That is precisely the defect class the
//! opponent-priors gate exists to rule out.
//!
//! So the coupling is structural here instead. [`PriorSeat`] derives its own
//! side flag and selects its own head out of a [`HeadPair`], and the searching
//! seat crosses the gate as [`SearchingSeat`] rather than as a `bool` — a
//! `&dyn SearchingSeat` cannot be negated at a call site the way a `bool` can.
//!
//! What that does and does not buy, MEASURED rather than predicted (an earlier
//! version of this note predicted three of the four would become
//! unrepresentable; a mutation run falsified it):
//!
//! * one of the four is unrepresentable — `PriorSeat` no longer appears in
//!   `model.rs`;
//! * the four `self_side_one` arguments that used to cross the gate are now
//!   type errors to negate, and the knob they carried survives in ONE place:
//!   the `impl SearchingSeat for LeafContext` below;
//! * the two-slice head construction is likewise collapsed into
//!   `impl HeadSource for LeafBatchOutput`.
//!
//! A seat or head swap at the `model.rs` boundary is therefore reduced and
//! concentrated, NOT eliminated. Nothing in THIS module can kill one; that
//! needs an integration test over the real core, which
//! `tests/test_model_priors_search.py::OpponentPriorsEncodedSearchTest` now is
//! — an encoded search with `use_opponent_priors=True`, against a
//! random-weights artifact at the real v3 shape, with both heads recomputed in
//! torch as the oracle.
//!
//! MEASURED OUT OF IMAGE. Every outcome below comes from a local macOS/arm64
//! `--features model` wheel (torch 2.12.1), not from the campaign image. The
//! campaign's item-3 gate asks for an IN-IMAGE run; this table says the gate
//! is sharp, it does not say the gate has been run where it counts. Do not
//! read a discharged blocker off this comment.
//!
//! One wheel rebuilt and reinstalled per world; each child hashes the `.so` it
//! imported and a world whose hash matches the baseline's is refused:
//!
//! | boundary mutation                                   | outcome |
//! |-----------------------------------------------------|---------|
//! | `impl HeadSource` heads transposed                   | KILLED  |
//! | `impl SearchingSeat` negated                         | KILLED  |
//! | root: acting map passed as the opponent map          | KILLED  |
//! | root: `opponent_action_map` -> `self_action_map`     | KILLED  |
//! | root: opponent options are the acting options        | KILLED (second fixture only) |
//! | branch: opponent options are the acting options      | KILLED  |
//! | branch: `opponent_action_map` -> `self_action_map`   | KILLED  |
//! | round: the two seats' pending map lists swapped      | KILLED  |
//! | branch: `opponent_prefix()` -> `self_prefix()`       | **MISSED** |
//!
//! TWO FINDINGS THE TABLE COMPRESSES, both of which cost a wrong answer first.
//!
//! 1. WHY THE ROOT OPTION SWAP NEEDS ITS OWN FIXTURE. `MoveChoice` is
//!    index-valued (`Move(slot)` / `Switch(party_index)`) and `seat_action_map`
//!    picks the seat from `slot_is_self`, never from the options — so where the
//!    two seats' option lists are the same VALUE, substituting one for the
//!    other is a no-op. What makes them the same value is not the same option
//!    SHAPE (an earlier version of this note said shape, and it was wrong):
//!    `Side::add_switches` pushes every alive party index EXCEPT
//!    `active_index`, so it is the same ACTIVE PARTY INDEX. Row 0 of the
//!    committed sample has both actives at slot 0 and is blind to the mutation
//!    (24/24 identical reports); rows 2-4 have them at different slots and see
//!    it immediately (20/20 differ, `prior_fallbacks` 0 -> 1). Row 0 is the
//!    unusual case — after any switch the two actives sit apart — so the gate
//!    now drives BOTH, and no fainted reserve, narrowed move list or forced
//!    switch is needed.
//!
//!    "Identical reports" above means every field of the search report EXCEPT
//!    the wall-clock telemetry, which is nondeterministic on one wheel:
//!    `elapsed_s`, `iterations_per_s`, `encode_s`, `model_s`, `tree_s`,
//!    `fold_clone_s`, `render_s`, `fold_advance_s`, `tensor_s`,
//!    `action_map_s`, `row_input_s`, `products_s`, `row_write_s`. Two runs of
//!    ONE wheel agree on everything else (24/24), which is what makes the
//!    filtered comparison a differential rather than a fudge.
//!
//! 2. WHAT IS STILL UNCOVERED, and it is in the channel the campaign's
//!    label-space half depends on. Swapping `leaf_ctx.opponent_prefix()` for
//!    `self_prefix()` in the opponent's per-branch order evolution is MISSED by
//!    every test above. Self switch lines name the SELF party's species, which
//!    never appear in the opponent's order, so `evolve_self_order` matches
//!    nothing and the mutation degenerates to "the opponent's request order is
//!    frozen at its root value" — precisely the fail-open
//!    `root_opponent_order`'s docstring warns about, correct until the
//!    opponent's first switch and permuted from then on. It changes the ACTING
//!    seat's visits while leaving the opponent's ROOT visit order — the gate's
//!    only opponent observable — untouched, because the ROOT order is not
//!    evolved at all. Raising the budget does not help: checked at sims in
//!    {48, 96, 192, 512, 1024} x batch in {1, 8} x 4 seeds, the mutant is
//!    concordant wherever the baseline is.
//!
//!    WHY `prior_branches` CANNOT SERVE AS THE ORACLE, which is not the reason
//!    an earlier version of this note gave. M9 is a BRANCH defect and its
//!    applications DO reach that counter — the number moves, and it moves a
//!    lot. It is that [`resolve_pending_priors`] sums both seats over the whole
//!    search into ONE number, which a reshaped tree moves regardless, and not
//!    even in a consistent direction (HEAD -> M9 at sims=48, batch=1,
//!    seeds 5/11/17/23: 256 -> 260, 260 -> 252, 258 -> 196, 256 -> 189). There
//!    is no invariant there to pin, only a golden number that any legitimate
//!    change to the tree would also break.
//!
//!    THE FIX DIRECTION, so the next reader does not chase the wrong one: a
//!    crate field carrying a DIGEST of the opponent's gathered prior vectors in
//!    resolution order, per seat. Not an applied count — a count is exactly the
//!    summed-scalar shape that fails above. A per-seat digest is order- and
//!    value-sensitive, so a frozen opponent order changes it while a
//!    legitimately reshaped tree that gathers the same vectors does not.
//!
//! 3. A SEPARATE, SMALLER GAP on the ROOT side, which is NOT M9's cause and is
//!    recorded here only so it is not rediscovered as one.
//!    [`RootPriorResolution`] carries no `applied` count, so the root's
//!    opponent application — unlike every branch application — never reaches
//!    `prior_branches` at all. Nothing in the battery turns on this; the root
//!    seat is covered by the visit-order and seat-asymmetric oracles instead.

use crate::tree::{apply_self_priors, DecisionNode, Tree};
use poke_engine::engine::state::MoveChoice;
use poke_engine::state::State;

/// Which engine side the SEARCHING seat occupies.
///
/// Taken as a trait object rather than a `bool` for one reason: a `bool` handed
/// across the `model` feature gate can be negated at the call site, and four
/// such arguments used to cross it. `&dyn SearchingSeat` cannot be negated, so
/// those four one-token mutations become type errors. The knob does not vanish
/// — it relocates into the single impl below, which is one line in a module
/// that has tests. That is the whole claim; it is a reduction from four sites
/// to one, not an elimination.
pub(crate) trait SearchingSeat {
    fn searching_side_one(&self) -> bool;
}

impl SearchingSeat for crate::leaf::LeafContext {
    fn searching_side_one(&self) -> bool {
        self.self_is_side_one()
    }
}

/// The two policy heads of one batched forward, named rather than positional.
///
/// Same reasoning as [`SearchingSeat`]: `HeadPair::new(a, b)` let a caller swap
/// two same-typed slice arguments. Behind the feature gate that is a silent
/// head swap. Naming them collapses the swap into the implementing type.
pub(crate) trait HeadSource {
    fn acting_head(&self) -> &[f32];
    fn opponent_head(&self) -> &[f32];
    fn action_count(&self) -> usize;
}

/// Gather a leaf's action-block priors onto a seat's option list.
///
/// `priors_row` is one row of the UNMASKED softmax; the gathered subset is
/// renormalized over the mapped options — mathematically identical to the
/// masked softmax restricted to those actions (exp(l_i)/Σ_mapped exp(l_j)).
/// Returns `None` — leaving the node uniform — when any option lacks an
/// action-block slot (the whole node falls back rather than zeroing arms the
/// model cannot see) or the mapped mass underflows.
fn gather_self_priors(priors_row: &[f32], map: &[Option<usize>]) -> Option<Vec<f32>> {
    if map.is_empty() {
        return None;
    }
    let mut gathered = Vec::with_capacity(map.len());
    let mut sum = 0.0f32;
    for entry in map {
        let index = (*entry)?;
        let prior = *priors_row.get(index)?;
        sum += prior;
        gathered.push(prior);
    }
    // NaN comparisons are false, so a non-finite logit would slip past the
    // underflow guard alone and propagate into stat.prior.
    if !sum.is_finite() || sum <= 1e-8 {
        return None;
    }
    for prior in &mut gathered {
        *prior /= sum;
    }
    Some(gathered)
}

/// One row of a flat `[n_rows, action_count]` prior block.
///
/// `None` rather than a slice panic. This is DEFENSIVE ONLY: [`HeadPair::new`]
/// rejects any block whose length is not a whole number of `action_count`-wide
/// rows, so by the time a row is requested the arithmetic cannot overrun. It
/// stays fallible because "silently read a window straddling two leaves'
/// distributions" is the failure this module exists to prevent, and a total
/// signature would have to invent a value to do it.
fn prior_row(flat: &[f32], row: usize, action_count: usize) -> Option<&[f32]> {
    if action_count == 0 {
        return None;
    }
    let base = row.checked_mul(action_count)?;
    flat.get(base..base.checked_add(action_count)?)
}

/// Which seat a resolution is for. Carries EVERYTHING that differs between the
/// two seats: the side flag, the model head, and the branch storage slot.
///
/// Nothing outside this enum's methods may compute a seat's side or choose a
/// seat's head — see the module header.
#[cfg_attr(not(feature = "model"), allow(dead_code))]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum PriorSeat {
    /// The searching seat's own policy head (`LeafBatchOutput::priors`).
    Acting,
    /// The opponent action head (`LeafBatchOutput::opponent_priors`).
    Opponent,
}

impl PriorSeat {
    /// The engine side this seat occupies, given the side the SEARCHING seat
    /// occupies. The one and only `!` in the opponent-priors path.
    #[cfg_attr(not(feature = "model"), allow(dead_code))]
    pub(crate) fn owning_side_one(self, self_side_one: bool) -> bool {
        match self {
            PriorSeat::Acting => self_side_one,
            PriorSeat::Opponent => !self_side_one,
        }
    }
}

/// The two policy heads of one batched forward, checked against each other.
///
/// `acting` is the softmaxed policy head, `[n, action_count]`. `opponent` is
/// the softmaxed opponent action head, which is a SEPARATE tensor: nothing in
/// the model contract makes its width equal to `action_count`, and
/// `action_count` is read off the POLICY head's last dimension. Striding one
/// head by the other's width does not fail loudly — it reads windows that
/// straddle two leaves' distributions and are almost all in range:
///
/// ```text
/// opponent width 5, action_count 4, 3 rows, flat[r*5+s] = 100*r+s
///   row 0 -> [  0,   1,   2,   3]   slot 4 silently dropped
///   row 1 -> [  4, 100, 101, 102]   leaf 0 + leaf 1
///   row 2 -> [103, 104, 200, 201]   leaf 1 + leaf 2
/// ```
///
/// Two of three branches priced off a blend of two different leaves, reported
/// clean. So the widths are checked ONCE, here, and a mismatch is an error —
/// same standing as the fully-illegal legal-mask row in
/// `TorchScriptLeafEval::eval_batch`, and for the same reason: a wrong gather
/// does not fail, it returns a confident number.
#[cfg_attr(not(feature = "model"), allow(dead_code))]
#[derive(Debug)]
pub(crate) struct HeadPair<'a> {
    acting: &'a [f32],
    opponent: &'a [f32],
    action_count: usize,
}

impl<'a> HeadPair<'a> {
    /// The only public constructor. Callers name a SOURCE, never two slices, so
    /// the two heads cannot be transposed at the call site.
    ///
    /// `use_opponent_priors` is the flag the search runs under. With it OFF the
    /// opponent head is dropped unread, so a model whose opponent head has the
    /// wrong width still runs exactly the search it always ran — flag-off
    /// equivalence is the campaign's anchor and must not acquire a new way to
    /// fail.
    ///
    /// Both halves are load-bearing and both are pinned by
    /// `tests::from_source_names_each_head_and_honours_the_flag`: transposing
    /// the two `source` reads swaps the heads on every flag-on search (the
    /// width check compares the same two lengths, so it still passes), and
    /// hard-coding the flag breaks flag-off equivalence one way and silently
    /// makes the whole feature inert the other — the latter being exactly how
    /// cells B and E would read "opponent priors do not help" off a feature
    /// that never ran.
    #[cfg_attr(not(feature = "model"), allow(dead_code))]
    pub(crate) fn from_source(
        source: &'a dyn HeadSource,
        use_opponent_priors: bool,
    ) -> Result<Self, String> {
        Self::new(
            source.acting_head(),
            source.opponent_head(),
            source.action_count(),
            use_opponent_priors,
        )
    }

    /// Private: the two-slice form, reachable only through
    /// [`Self::from_source`] in production and directly from this module's
    /// tests. Keeping it private is what makes `from_source` the single seam
    /// where the heads are named.
    fn new(
        acting: &'a [f32],
        opponent: &'a [f32],
        action_count: usize,
        use_opponent_priors: bool,
    ) -> Result<Self, String> {
        if action_count == 0 {
            return Err("model returned action_count = 0".to_string());
        }
        if acting.len() % action_count != 0 {
            return Err(format!(
                "policy head is not a whole number of rows: {} values over {action_count} actions",
                acting.len()
            ));
        }
        let opponent = if use_opponent_priors { opponent } else { &[] };
        // Empty is "this evaluator has no opponent head" (the throughput-bench
        // cores), which stays uniform. Non-empty and the wrong width is a
        // model-contract break.
        if !opponent.is_empty() && opponent.len() != acting.len() {
            return Err(format!(
                "opponent action head width does not match the policy head: {} vs {} values over \
                 {action_count} actions. Striding one by the other prices branches off windows \
                 spanning two leaves' distributions, so this refuses instead. Re-export the \
                 artifact with matching action widths (scripts/export_model.py)",
                opponent.len(),
                acting.len()
            ));
        }
        Ok(Self {
            acting,
            opponent,
            action_count,
        })
    }

    /// One seat's row. The head is chosen BY THE SEAT — a caller cannot pair
    /// the opponent seat with the acting head.
    fn row(&self, seat: PriorSeat, row: usize) -> Option<&[f32]> {
        let flat = match seat {
            PriorSeat::Acting => self.acting,
            PriorSeat::Opponent => self.opponent,
        };
        prior_row(flat, row, self.action_count)
    }
}

/// Number of branches whose priors were gathered AND landed, and the number
/// that fell back to uniform.
///
/// `applied`/`fallbacks` remain BOTH-SEAT sums so existing consumers
/// (`prior_branches`, `prior_fallbacks`) are unchanged. The seat-attributed
/// fields beside them exist because those sums are exactly the shape that
/// cannot answer "did the opponent half run at all": a shard in which the
/// opponent map refused every time is indistinguishable, in the sums, from one
/// where it applied cleanly. Section 2 of this module's header says a count
/// alone still cannot serve as M9's oracle -- `opponent_digest` is the
/// order- and value-sensitive observable it prescribes instead.
#[cfg_attr(not(feature = "model"), allow(dead_code))]
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub(crate) struct PriorResolution {
    pub applied: usize,
    pub fallbacks: usize,
    pub acting_applied: usize,
    pub acting_fallbacks: usize,
    pub opponent_applied: usize,
    pub opponent_fallbacks: usize,
    /// FNV-1a over every OPPONENT prior vector this resolution gathered, folded
    /// in resolution order with its position. Order- and value-sensitive by
    /// construction: freezing the opponent's request order permutes the gathered
    /// vectors and moves the digest, while a legitimately reshaped tree that
    /// gathers the SAME vectors in the same order does not. `0` means "nothing
    /// gathered", which is distinct from any gathered value in practice.
    pub opponent_digest: u64,
}

/// FNV-1a fold of one already-computed digest into a running one.
///
/// Used to chain per-round opponent digests across a search. Kept beside
/// [`fold_prior_digest`] and using the same constants so the two cannot drift.
#[cfg_attr(not(feature = "model"), allow(dead_code))]
pub(crate) fn fold_digest_u64(digest: u64, position: usize, value: u64) -> u64 {
    const PRIME: u64 = 0x1000_0000_01b3;
    let mut acc = if digest == 0 { 0xcbf2_9ce4_8422_2325 } else { digest };
    for byte in (position as u64).to_le_bytes() {
        acc = (acc ^ u64::from(byte)).wrapping_mul(PRIME);
    }
    for byte in value.to_le_bytes() {
        acc = (acc ^ u64::from(byte)).wrapping_mul(PRIME);
    }
    acc
}

/// FNV-1a fold of one prior vector into a running digest, position included.
///
/// Position matters: two resolutions that gather the same multiset of vectors
/// in a different order are different events, and the frozen-order mutant is
/// precisely a reordering.
#[cfg_attr(not(feature = "model"), allow(dead_code))]
pub(crate) fn fold_prior_digest(digest: u64, position: usize, priors: &[f32]) -> u64 {
    const PRIME: u64 = 0x1000_0000_01b3;
    let mut acc = if digest == 0 { 0xcbf2_9ce4_8422_2325 } else { digest };
    for byte in (position as u64).to_le_bytes() {
        acc = (acc ^ u64::from(byte)).wrapping_mul(PRIME);
    }
    for value in priors {
        // Bit pattern, not the float: identical vectors must fold identically
        // and NaN never appears here (gather rejects non-finite rows upstream).
        for byte in value.to_bits().to_le_bytes() {
            acc = (acc ^ u64::from(byte)).wrapping_mul(PRIME);
        }
    }
    acc
}

/// Resolve one batch round's pending prior maps against the batch's own prior
/// rows: gather per branch, apply onto a child decision node that already
/// exists, and store on the branch for the child that does not exist yet.
///
/// `pending` entries are `(chance-branch key, BATCH ROW, option→action-slot
/// map)`. The row is the branch's index in THIS round's batch, so the prior
/// row is `flat[row * action_count .. +action_count]` — a branch reading any
/// other row would price its arms from a different leaf's policy.
///
/// `self_side_one` is the side the SEARCHING seat occupies. The side that owns
/// these arms is derived from it by the seat, and stored on the branch
/// verbatim so the lazy application at child creation
/// ([`crate::tree::apply_branch_child_priors`]) is a plain write to that
/// seat's stats with no negation anywhere.
fn resolve_pending_priors(
    tree: &mut Tree,
    pending: &[((usize, usize), usize, Vec<Option<usize>>)],
    heads: &HeadPair<'_>,
    self_side_one: bool,
    seat: PriorSeat,
) -> PriorResolution {
    let side_one = seat.owning_side_one(self_side_one);
    let mut resolution = PriorResolution::default();
    for (position, (key, row, map)) in pending.iter().enumerate() {
        let priors = heads
            .row(seat, *row)
            .and_then(|priors_row| gather_self_priors(priors_row, map));
        let Some(priors) = priors else {
            resolution.fallbacks += 1;
            match seat {
                PriorSeat::Acting => resolution.acting_fallbacks += 1,
                PriorSeat::Opponent => resolution.opponent_fallbacks += 1,
            }
            continue;
        };
        // Folded at GATHER, before the apply can reject on arity: the digest
        // answers "which vectors did this seat gather, in what order", which is
        // the question the frozen-order mutant changes. Whether the child
        // existed yet is a separate fact, carried by the counts.
        if matches!(seat, PriorSeat::Opponent) {
            resolution.opponent_digest =
                fold_prior_digest(resolution.opponent_digest, position, &priors);
        }
        let child = tree.chances[key.0].branches[key.1].child;
        let applied = match child {
            Some(child) => apply_self_priors(&mut tree.decisions[child], side_one, &priors),
            None => true,
        };
        // Parked on the branch whether or not the child exists yet: a child
        // created LATER in this search takes its priors from here (batch > 1
        // can descend a sibling before its priors resolve), and an arity
        // mismatch with TODAY's child is not necessarily one with a later one.
        let branch = &mut tree.chances[key.0].branches[key.1];
        let slot = match seat {
            PriorSeat::Acting => &mut branch.child_self_priors,
            PriorSeat::Opponent => &mut branch.child_opponent_priors,
        };
        *slot = Some((side_one, priors));
        if applied {
            resolution.applied += 1;
            match seat {
                PriorSeat::Acting => resolution.acting_applied += 1,
                PriorSeat::Opponent => resolution.opponent_applied += 1,
            }
        } else {
            resolution.fallbacks += 1;
            match seat {
                PriorSeat::Acting => resolution.acting_fallbacks += 1,
                PriorSeat::Opponent => resolution.opponent_fallbacks += 1,
            }
        }
    }
    resolution
}

/// Resolve ONE batch round for BOTH seats off the one forward that produced
/// them. This is the whole per-round prior surface: the caller supplies each
/// seat's pending maps, the heads, and the side the SEARCHING seat occupies,
/// and never names a seat's side, a seat's head, or a seat's storage slot.
///
/// `opponent_pending` is empty whenever the flag is off, so flag-off runs pay
/// one empty loop and change nothing.
#[cfg_attr(not(feature = "model"), allow(dead_code))]
pub(crate) fn resolve_round_priors(
    tree: &mut Tree,
    acting_pending: &[((usize, usize), usize, Vec<Option<usize>>)],
    opponent_pending: &[((usize, usize), usize, Vec<Option<usize>>)],
    heads: &HeadPair<'_>,
    seat_source: &dyn SearchingSeat,
) -> PriorResolution {
    let self_side_one = seat_source.searching_side_one();
    let acting = resolve_pending_priors(
        tree,
        acting_pending,
        heads,
        self_side_one,
        PriorSeat::Acting,
    );
    let opponent = resolve_pending_priors(
        tree,
        opponent_pending,
        heads,
        self_side_one,
        PriorSeat::Opponent,
    );
    // The sums stay, for the existing both-seat counters. The seat fields are
    // taken from the seat that produced them and are never added together --
    // folding them here would rebuild, one level up, the exact blindness that
    // made `prior_fallbacks` unable to say whether the opponent half ran.
    PriorResolution {
        applied: acting.applied + opponent.applied,
        fallbacks: acting.fallbacks + opponent.fallbacks,
        acting_applied: acting.acting_applied,
        acting_fallbacks: acting.acting_fallbacks,
        opponent_applied: opponent.opponent_applied,
        opponent_fallbacks: opponent.opponent_fallbacks,
        opponent_digest: opponent.opponent_digest,
    }
}

/// The root decision node's two option lists, selected by the side the
/// SEARCHING seat occupies.
///
/// Cloned rather than borrowed because the caller needs the tree mutably again
/// to apply what it gathers. Exists so that no caller writes
/// `if self_side_one { s2_options } else { s1_options }` — that expression is
/// a seat swap waiting to happen, and behind the `model` feature it is a seat
/// swap no test can see.
#[cfg_attr(not(feature = "model"), allow(dead_code))]
pub(crate) struct RootSeats {
    pub acting_options: Vec<MoveChoice>,
    pub opponent_options: Vec<MoveChoice>,
}

/// True for the one-arm "no real choice" shape, which is never given priors.
#[cfg_attr(not(feature = "model"), allow(dead_code))]
pub(crate) fn is_single_none(options: &[MoveChoice]) -> bool {
    options.len() == 1 && matches!(options[0], MoveChoice::None)
}

/// Both seats' option lists at an INTERIOR node's future child decision node,
/// selected by the side the SEARCHING seat occupies — the `root_seats`
/// equivalent one ply down, and it exists for the same reason.
///
/// Mirrors `new_decision_node`'s defensive empty-side handling so prior vectors
/// align with the node's own arm order, and calls `get_all_options` ONCE for
/// both seats (the two-call version could not disagree, but it could be edited
/// to).
#[cfg_attr(not(feature = "model"), allow(dead_code))]
pub(crate) fn branch_seats(state: &State, seat: &dyn SearchingSeat) -> RootSeats {
    let (s1_options, s2_options) = state.get_all_options();
    seat_split(s1_options, s2_options, seat.searching_side_one())
}

/// The pure half of [`branch_seats`], split out so the empty-side defence below
/// is reachable from a test.
///
/// It is NOT reachable through a real `State`, and the reason is structural
/// rather than "not observed in the states I tried". `get_all_options` cannot
/// return an empty vector on the vendored tree: every exit either carries its
/// own `len() == 0` guard or routes through `Side::add_switches`
/// (`third_party/poke-engine-src/src/gen3/state.rs`), which ends by pushing
/// `MoveChoice::None` onto a still-empty vector. That invariant is established
/// by `third_party/poke-engine-gen3-terminal-options.patch` — see its header
/// for the condition it closed (a forced replacement on one side plus an
/// unsatisfiable `switch_out_move_second_saved_move` on the other, which is
/// NOT the same as "a side with no living reserve") — and it is pinned by
/// `tests/gen3_terminal_options.rs` and `tests/test_engine_search_no_panic.py`.
///
/// So the defence below is belt-and-braces mirroring `new_decision_node`'s, and
/// the two must not disagree about how many arms a node has. It stays, and it
/// is tested, for a specific reason: the invariant is held by a POKEZERO PATCH
/// TO VENDORED CODE, which is exactly the kind of thing that regresses silently
/// on an engine bump. `vendor_poke_engine_src.sh` re-applies the patch stack
/// against whatever upstream ships, and a patch that stops applying cleanly is
/// a louder failure than one that applies to a rewritten function.
fn seat_split(
    s1_options: Vec<MoveChoice>,
    s2_options: Vec<MoveChoice>,
    self_side_one: bool,
) -> RootSeats {
    let (acting, opponent) = if self_side_one {
        (s1_options, s2_options)
    } else {
        (s2_options, s1_options)
    };
    RootSeats {
        acting_options: non_empty(acting),
        opponent_options: non_empty(opponent),
    }
}

/// A decision node must always offer at least one arm, or the prior vector and
/// the node's stat vector disagree on length and every gather falls back.
fn non_empty(mut options: Vec<MoveChoice>) -> Vec<MoveChoice> {
    if options.is_empty() {
        options.push(MoveChoice::None);
    }
    options
}

#[cfg_attr(not(feature = "model"), allow(dead_code))]
pub(crate) fn root_seats(root: &DecisionNode, seat: &dyn SearchingSeat) -> RootSeats {
    let (acting_options, opponent_options) = if seat.searching_side_one() {
        (&root.s1_options, &root.s2_options)
    } else {
        (&root.s2_options, &root.s1_options)
    };
    RootSeats {
        acting_options: acting_options.clone(),
        opponent_options: opponent_options.clone(),
    }
}

/// What the root resolution produced: the acting seat's applied prior vector
/// (the `root_priors` the report echoes) and the fallback count.
#[cfg_attr(not(feature = "model"), allow(dead_code))]
#[derive(Clone, PartialEq, Debug, Default)]
pub(crate) struct RootPriorResolution {
    pub acting: Option<Vec<f32>>,
    pub fallbacks: usize,
}

/// Gather and apply the ROOT node's priors for both seats off row 0 of a
/// one-row forward.
///
/// The action MAPS are the caller's to build — that is `LeafContext` work, and
/// the map itself is the part the campaign has already measured. Everything
/// downstream of the map is here: which head each seat reads, which side each
/// seat owns, and which stat vector each vector lands on. `opponent_map` is
/// `None` when the flag is off, when the evaluator has no opponent head, or
/// when the opponent has no real choice at the root.
#[cfg_attr(not(feature = "model"), allow(dead_code))]
pub(crate) fn resolve_root_priors(
    root: &mut DecisionNode,
    heads: &HeadPair<'_>,
    seat_source: &dyn SearchingSeat,
    acting_map: &[Option<usize>],
    opponent_map: Option<&[Option<usize>]>,
) -> RootPriorResolution {
    let self_side_one = seat_source.searching_side_one();
    let mut resolution = RootPriorResolution::default();
    for (seat, map) in [
        (PriorSeat::Acting, Some(acting_map)),
        (PriorSeat::Opponent, opponent_map),
    ] {
        let Some(map) = map else { continue };
        let side_one = seat.owning_side_one(self_side_one);
        let gathered = heads
            .row(seat, 0)
            .and_then(|priors_row| gather_self_priors(priors_row, map));
        match gathered {
            Some(priors) if apply_self_priors(root, side_one, &priors) => {
                if seat == PriorSeat::Acting {
                    resolution.acting = Some(priors);
                }
            }
            _ => resolution.fallbacks += 1,
        }
    }
    resolution
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tree::{ChanceBranch, ChanceNode, DecisionNode, Tree};
    use crate::MoveStats;
    use poke_engine::engine::state::MoveChoice;
    use std::collections::HashMap;

    /// A [`SearchingSeat`] that is just a side. Production's only implementor is
    /// `LeafContext`, which a unit test cannot build (it needs the ~475 KB
    /// encoder tables); this stands in for it and exercises the same code.
    struct Seat(bool);

    impl SearchingSeat for Seat {
        fn searching_side_one(&self) -> bool {
            self.0
        }
    }

    /// Named heads for `HeadPair::from_source`, mirroring `LeafBatchOutput`.
    struct Heads {
        acting: Vec<f32>,
        opponent: Vec<f32>,
        action_count: usize,
    }

    /// The digest is ORDER-sensitive, not just value-sensitive.
    ///
    /// This is the whole reason the digest exists rather than a count. M9
    /// (`branch: opponent_prefix() -> self_prefix()`) does not change WHICH
    /// vectors the opponent gathers or HOW MANY; it changes the ORDER, because
    /// a frozen request order permutes the map. A count cannot see that. A
    /// value-only hash could not either, if the same vectors are gathered.
    /// Section 2 of this module's header prescribes exactly this observable.
    #[test]
    fn the_opponent_digest_separates_a_reordering_from_the_same_vectors() {
        let a = [0.7f32, 0.2, 0.1];
        let b = [0.1f32, 0.2, 0.7];

        let in_order = fold_prior_digest(fold_prior_digest(0, 0, &a), 1, &b);
        let reversed = fold_prior_digest(fold_prior_digest(0, 0, &b), 1, &a);
        assert_ne!(
            in_order, reversed,
            "the same two vectors gathered in the other order must not collide"
        );

        // Same order, same values -> same digest. Without this the digest would
        // be useless as an equality oracle for a legitimately reshaped tree.
        let again = fold_prior_digest(fold_prior_digest(0, 0, &a), 1, &b);
        assert_eq!(in_order, again);

        // A single changed value moves it too.
        let perturbed = [0.7f32, 0.2, 0.100_001];
        assert_ne!(
            in_order,
            fold_prior_digest(fold_prior_digest(0, 0, &perturbed), 1, &b)
        );

        // "Nothing gathered" is zero and nothing gathered folds to it.
        assert_eq!(0u64, 0u64);
        assert_ne!(fold_prior_digest(0, 0, &a), 0);
    }

    /// Counts only, as a tuple in a fixed order:
    /// (applied, fallbacks, acting_applied, acting_fallbacks,
    ///  opponent_applied, opponent_fallbacks).
    ///
    /// Separated from the digest deliberately. The counts answer "did this
    /// seat's half run"; the digest answers "did it gather the same vectors in
    /// the same order". Asserting them together would make every count test
    /// brittle to a digest change that is not what the test is about.
    fn counts(r: PriorResolution) -> (usize, usize, usize, usize, usize, usize) {
        (
            r.applied,
            r.fallbacks,
            r.acting_applied,
            r.acting_fallbacks,
            r.opponent_applied,
            r.opponent_fallbacks,
        )
    }

    impl HeadSource for Heads {
        fn acting_head(&self) -> &[f32] {
            &self.acting
        }
        fn opponent_head(&self) -> &[f32] {
            &self.opponent
        }
        fn action_count(&self) -> usize {
            self.action_count
        }
    }

    /// A recognizable non-uniform, non-monotone row over 9 action slots
    /// (schema v1's action count). Every entry is distinct, so ANY wrong slot
    /// choice changes the gathered vector.
    const ROW: [f32; 9] = [0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.05, 0.03, 0.29];

    fn approx(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len(), "length: {actual:?}");
        for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (a - e).abs() < 1e-6,
                "slot {i}: got {a}, want {e} (full: {actual:?})"
            );
        }
    }

    // -----------------------------------------------------------------
    // gather: slot order and indexing
    // -----------------------------------------------------------------

    /// THE slot-order pin. The map is a permutation with a gap, so the
    /// gathered vector equals the row read THROUGH the map and nothing else:
    /// identity (`row[i]`), sorted-slot order, and any ±1 index shift all
    /// produce a different vector.
    #[test]
    fn gather_reads_the_row_through_the_map_in_option_order() {
        let map = vec![Some(5), Some(1), Some(8)];
        let gathered = gather_self_priors(&ROW, &map).expect("mapped mass is well above underflow");
        let sum = ROW[5] + ROW[1] + ROW[8];
        approx(&gathered, &[ROW[5] / sum, ROW[1] / sum, ROW[8] / sum]);

        // Explicitly NOT the identity read, and explicitly NOT the same
        // multiset in slot-sorted order: both are the mutants this pins.
        let identity_sum = ROW[0] + ROW[1] + ROW[2];
        assert!(
            (gathered[0] - ROW[0] / identity_sum).abs() > 1e-3,
            "gather must not ignore the map"
        );
        let sorted = [ROW[1] / sum, ROW[5] / sum, ROW[8] / sum];
        assert!(
            (gathered[0] - sorted[0]).abs() > 1e-3,
            "gather must preserve OPTION order, not action-slot order"
        );
    }

    /// An off-by-one on the action index is the cheapest way to silently
    /// mis-price every arm, and a uniform row would hide it. This row is
    /// strictly increasing over the mapped window, so a +1 or -1 shift is
    /// order-detectable, not just value-detectable.
    #[test]
    fn gather_is_off_by_one_sensitive() {
        let row = [0.1, 0.2, 0.3, 0.4];
        let map = vec![Some(0), Some(1)];
        let gathered = gather_self_priors(&row, &map).expect("mass present");
        approx(&gathered, &[1.0 / 3.0, 2.0 / 3.0]);
        // What the +1 mutant would produce, spelled out so the difference is
        // asserted rather than assumed.
        let shifted = gather_self_priors(&row, &vec![Some(1), Some(2)]).expect("mass present");
        approx(&shifted, &[0.4, 0.6]);
        assert!((gathered[0] - shifted[0]).abs() > 1e-3);
    }

    #[test]
    fn gather_renormalizes_over_the_mapped_subset_only() {
        let map = vec![Some(0), Some(2)];
        let gathered = gather_self_priors(&ROW, &map).expect("mass present");
        let total: f32 = gathered.iter().sum();
        assert!((total - 1.0).abs() < 1e-6, "gathered mass {total} != 1");
        // Renormalized over the MAPPED subset (0.01 + 0.04), not over the whole
        // row (which sums to 1.0 and would leave the raw values untouched).
        approx(&gathered, &[0.2, 0.8]);
    }

    /// Single mapped option: the renormalization must produce 1.0, not the raw
    /// row value. A no-op "renormalizer" leaves 0.16 here.
    #[test]
    fn gather_of_a_single_mapped_option_is_certainty() {
        let gathered = gather_self_priors(&ROW, &vec![Some(4)]).expect("mass present");
        approx(&gathered, &[1.0]);
    }

    /// Order of the map is the order of the output, including when the map is
    /// the reverse permutation. Pins that no sort sneaks in.
    #[test]
    fn gather_output_order_follows_the_map_exactly() {
        let forward = gather_self_priors(&ROW, &vec![Some(0), Some(5)]).expect("mass");
        let reverse = gather_self_priors(&ROW, &vec![Some(5), Some(0)]).expect("mass");
        approx(&forward, &[reverse[1], reverse[0]]);
        assert!(
            (forward[0] - reverse[0]).abs() > 1e-3,
            "reversing the map must reverse the output"
        );
    }

    // -----------------------------------------------------------------
    // gather: the fallback contract (whole node, never partial)
    // -----------------------------------------------------------------

    /// One unmapped option fails the WHOLE node. A per-option fallback (0.0 or
    /// uniform for the unmapped arm) would zero or invent an arm the model
    /// cannot see; the design says the node goes uniform instead.
    #[test]
    fn one_unmapped_option_falls_the_whole_node_back() {
        assert_eq!(
            gather_self_priors(&ROW, &vec![Some(0), None, Some(2)]),
            None
        );
        // ... including when the unmapped option is last, i.e. after real
        // values have already been accumulated.
        assert_eq!(
            gather_self_priors(&ROW, &vec![Some(0), Some(2), None]),
            None
        );
    }

    #[test]
    fn an_out_of_range_action_slot_falls_back_rather_than_panicking() {
        assert_eq!(gather_self_priors(&ROW, &vec![Some(0), Some(9)]), None);
        assert_eq!(gather_self_priors(&[], &vec![Some(0)]), None);
    }

    #[test]
    fn an_empty_option_list_falls_back() {
        assert_eq!(gather_self_priors(&ROW, &[]), None);
    }

    #[test]
    fn underflowing_mapped_mass_falls_back() {
        let row = [1e-12f32, 1e-12, 1.0];
        assert_eq!(gather_self_priors(&row, &vec![Some(0), Some(1)]), None);
        // The guard is on the MAPPED mass, so the same row with the big slot
        // mapped in must succeed — otherwise the test above would also pass
        // against a "never gather anything" mutant.
        assert!(gather_self_priors(&row, &vec![Some(0), Some(2)]).is_some());
    }

    #[test]
    fn non_finite_priors_fall_back_instead_of_poisoning_stats() {
        let nan = [f32::NAN, 0.5, 0.5];
        assert_eq!(gather_self_priors(&nan, &vec![Some(0), Some(1)]), None);
        let inf = [f32::INFINITY, 0.5, 0.5];
        assert_eq!(gather_self_priors(&inf, &vec![Some(0), Some(1)]), None);
        // A negative row that cancels to ~0 would divide by ~0; the finite +
        // underflow guard has to catch it before it reaches stat.prior.
        let cancelling = [1.0f32, -1.0, 0.5];
        assert_eq!(
            gather_self_priors(&cancelling, &vec![Some(0), Some(1)]),
            None
        );
    }

    // -----------------------------------------------------------------
    // prior_row: batch row indexing
    // -----------------------------------------------------------------

    #[test]
    fn prior_row_slices_the_requested_row_and_only_that_row() {
        let flat: Vec<f32> = (0..12).map(|v| v as f32).collect();
        assert_eq!(prior_row(&flat, 0, 4), Some(&[0.0, 1.0, 2.0, 3.0][..]));
        assert_eq!(prior_row(&flat, 1, 4), Some(&[4.0, 5.0, 6.0, 7.0][..]));
        assert_eq!(prior_row(&flat, 2, 4), Some(&[8.0, 9.0, 10.0, 11.0][..]));
        // Past the end -> fallback, not a panic and not a wrapped read.
        assert_eq!(prior_row(&flat, 3, 4), None);
        assert_eq!(prior_row(&flat, 0, 13), None);
        assert_eq!(prior_row(&flat, 0, 0), None);
        assert_eq!(prior_row(&flat, usize::MAX, 4), None);
    }

    // -----------------------------------------------------------------
    // resolve: seat routing, row routing, storage, telemetry
    // -----------------------------------------------------------------

    fn stats(n: usize) -> Vec<MoveStats> {
        (0..n)
            .map(|i| MoveStats {
                display: format!("arm{i}"),
                prior: 1.0 / n as f32,
                visits: 0,
                total_value: 0.0,
            })
            .collect()
    }

    fn decision(s1_arms: usize, s2_arms: usize) -> DecisionNode {
        DecisionNode {
            visits: 0,
            depth: 1,
            s1_options: vec![MoveChoice::None; s1_arms],
            s2_options: vec![MoveChoice::None; s2_arms],
            s1_stats: stats(s1_arms),
            s2_stats: stats(s2_arms),
            children: HashMap::new(),
        }
    }

    fn branch(child: Option<usize>) -> ChanceBranch {
        ChanceBranch {
            probability: 1.0,
            instructions: Vec::new(),
            value_sum: 0.5,
            visits: 1,
            terminal: None,
            no_expand: false,
            pending_row: None,
            child,
            child_self_priors: None,
            child_opponent_priors: None,
        }
    }

    /// A [`HeadPair`] whose OPPONENT head is `row` and whose acting head is a
    /// same-width block of a distinguishable constant, so a test that reads the
    /// wrong head gets visibly wrong numbers rather than the right ones.
    fn opponent_only_heads(row: &[f32]) -> HeadPair<'_> {
        // A distinguishable constant acting head of the same width, so a test
        // that reads the wrong head gets visibly wrong numbers.
        const ACTING: [f32; 9] = [1.0 / 9.0; 9];
        assert_eq!(
            row.len(),
            ACTING.len(),
            "helper is sized for the 9-slot row"
        );
        HeadPair::new(&ACTING, row, row.len(), true).expect("same width")
    }

    /// Root decision + one chance node whose single branch already has a child
    /// decision node with `s1_arms`/`s2_arms` arms.
    fn tree_with_child(s1_arms: usize, s2_arms: usize) -> Tree {
        Tree {
            decisions: vec![decision(2, 2), decision(s1_arms, s2_arms)],
            chances: vec![ChanceNode {
                branches: vec![branch(Some(1))],
            }],
        }
    }

    /// The one negation in the path, isolated. Everything else about seat
    /// routing is a consequence of this function.
    #[test]
    fn owning_side_one_is_the_only_seat_negation() {
        assert!(PriorSeat::Acting.owning_side_one(true));
        assert!(!PriorSeat::Acting.owning_side_one(false));
        assert!(!PriorSeat::Opponent.owning_side_one(true));
        assert!(PriorSeat::Opponent.owning_side_one(false));
    }

    // -----------------------------------------------------------------
    // HeadPair: the two heads are checked against each other ONCE
    // -----------------------------------------------------------------

    /// Regression pin for the defect independent review measured. `action_count`
    /// is the POLICY head's width; the opponent head is a separate tensor. When
    /// they disagree, striding one by the other yields windows that span two
    /// leaves' distributions and are almost all IN RANGE, so nothing fails:
    ///
    /// ```text
    /// opponent width 5, action_count 4, 3 rows, flat[r*5+s] = 100*r+s
    ///   row 0 -> [  0,   1,   2,   3]   slot 4 dropped
    ///   row 1 -> [  4, 100, 101, 102]   leaf 0 + leaf 1
    ///   row 2 -> [103, 104, 200, 201]   leaf 1 + leaf 2
    /// ```
    ///
    /// Two of three branches priced off a blend, reported clean. Refused now.
    #[test]
    fn a_mismatched_opponent_head_width_is_refused_not_silently_strided() {
        let acting: Vec<f32> = (0..12).map(|v| v as f32).collect(); // 3 x 4
        let opponent: Vec<f32> = (0..15).map(|v| v as f32).collect(); // 3 x 5
        let error = HeadPair::new(&acting, &opponent, 4, true)
            .expect_err("a width mismatch must refuse, not stride");
        assert!(
            error.contains("opponent action head width does not match the policy head"),
            "{error}"
        );
        assert!(error.contains("15") && error.contains("12"), "{error}");

        // The narrow direction is refused too. Pre-fix this one at least
        // PANICKED on the last row, which was the only signal a mismatch
        // existed; the wide direction above was silent all the way through.
        let narrow: Vec<f32> = (0..9).map(|v| v as f32).collect(); // 3 x 3
        assert!(HeadPair::new(&acting, &narrow, 4, true).is_err());
    }

    /// Flag OFF drops the opponent head unread, so a model with a mismatched
    /// opponent head still runs exactly the search it always ran. Flag-off
    /// equivalence is the campaign's anchor and must not gain a way to fail.
    #[test]
    fn a_mismatched_opponent_head_is_ignored_entirely_when_the_flag_is_off() {
        let acting: Vec<f32> = (0..12).map(|v| v as f32).collect();
        let opponent: Vec<f32> = (0..15).map(|v| v as f32).collect();
        let heads = HeadPair::new(&acting, &opponent, 4, false).expect("flag-off never refuses");
        assert_eq!(
            heads.row(PriorSeat::Acting, 1),
            Some(&[4.0, 5.0, 6.0, 7.0][..])
        );
        assert_eq!(
            heads.row(PriorSeat::Opponent, 0),
            None,
            "flag-off must not read the opponent head at all"
        );
    }

    /// An evaluator with no opponent head at all (the throughput-bench cores)
    /// is legal and stays uniform rather than erroring.
    #[test]
    fn an_absent_opponent_head_is_legal_and_yields_no_rows() {
        let acting = [0.25f32; 8];
        let heads = HeadPair::new(&acting, &[], 4, true).expect("no opponent head is legal");
        assert!(heads.row(PriorSeat::Acting, 1).is_some());
        assert_eq!(heads.row(PriorSeat::Opponent, 0), None);
    }

    #[test]
    fn a_ragged_policy_head_is_refused() {
        let acting = [0.1f32; 7]; // not a multiple of 4
        assert!(HeadPair::new(&acting, &[], 4, true).is_err());
        assert!(HeadPair::new(&acting, &[], 0, true).is_err());
    }

    /// The head is selected BY THE SEAT. A caller cannot hand the opponent seat
    /// the acting head, which is one of the four live-path mutations review
    /// found surviving.
    #[test]
    fn each_seat_reads_its_own_head() {
        let acting = [1.0f32, 2.0, 3.0, 4.0];
        let opponent = [10.0f32, 20.0, 30.0, 40.0];
        let heads = HeadPair::new(&acting, &opponent, 4, true).expect("same width");
        assert_eq!(heads.row(PriorSeat::Acting, 0), Some(&acting[..]));
        assert_eq!(heads.row(PriorSeat::Opponent, 0), Some(&opponent[..]));
    }

    /// `from_source` is the ONLY public constructor and the seam the module's
    /// structural claim rests on — every other test here reaches past it into
    /// the private `new`, so without this one it is production code with no
    /// coverage. Three measured survivors motivate each assertion:
    ///
    /// * transposing the two `source` reads swaps the heads on every flag-on
    ///   search, and the width check compares the same two lengths so it still
    ///   passes — the #937 orientation class, silent;
    /// * hard-coding the flag `true` makes flag-OFF run the opponent-width
    ///   check, so a mismatched artifact errors a search that must be
    ///   byte-for-byte unchanged;
    /// * hard-coding it `false` drops the opponent head with the flag ON. Every
    ///   opponent branch falls back to uniform, `prior_fallbacks` rises, nothing
    ///   gates on that counter, and cells B and E read "opponent priors do not
    ///   help" off a feature that never ran.
    #[test]
    fn from_source_names_each_head_and_honours_the_flag() {
        let source = Heads {
            acting: vec![0.7, 0.1, 0.1, 0.1],
            opponent: vec![0.1, 0.1, 0.1, 0.7],
            action_count: 4,
        };

        // Named, not positional: the acting seat gets the acting head.
        let heads = HeadPair::from_source(&source, true).expect("same width");
        assert_eq!(
            heads.row(PriorSeat::Acting, 0),
            Some(&source.acting[..]),
            "the acting seat must read the acting head, not the opponent's"
        );
        assert_eq!(
            heads.row(PriorSeat::Opponent, 0),
            Some(&source.opponent[..]),
            "and the opponent seat the opponent head"
        );

        // The flag is threaded, not assumed, in BOTH directions.
        let off = HeadPair::from_source(&source, false).expect("flag-off never refuses");
        assert_eq!(
            off.row(PriorSeat::Opponent, 0),
            None,
            "flag-off must drop the opponent head unread"
        );
        assert!(
            heads.row(PriorSeat::Opponent, 0).is_some(),
            "flag-on must NOT drop it, or the feature is inert and the cells \
             measure nothing"
        );

        // The width check follows the flag too.
        let mismatched = Heads {
            acting: vec![0.25; 8],
            opponent: vec![0.2; 10],
            action_count: 4,
        };
        assert!(
            HeadPair::from_source(&mismatched, true).is_err(),
            "flag-on must refuse a mismatched opponent head"
        );
        assert!(
            HeadPair::from_source(&mismatched, false).is_ok(),
            "flag-off must not: the head is never read, and flag-off \
             equivalence cannot acquire a new way to fail"
        );
    }

    /// The orientation pin. With the searching seat on side one the opponent is
    /// side two, so the opponent head lands on `s2_stats` and side one is left
    /// uniform. The side is DERIVED from `self_side_one` inside the module, so
    /// this pins the live coupling and not a value the caller computed.
    #[test]
    fn opponent_priors_land_on_the_other_seat_when_the_searcher_is_side_one() {
        let mut tree = tree_with_child(3, 2);
        let pending = vec![((0usize, 0usize), 0usize, vec![Some(5), Some(1)])];
        let heads = opponent_only_heads(&ROW);
        let resolution =
            resolve_pending_priors(&mut tree, &pending, &heads, true, PriorSeat::Opponent);
        assert_eq!(counts(resolution), (1, 0, 0, 0, 1, 0));
        // Gathered, so the opponent digest moved off its "nothing gathered" zero.
        assert_ne!(resolution.opponent_digest, 0);
        let sum = ROW[5] + ROW[1];
        let child = &tree.decisions[1];
        approx(
            &child.s2_stats.iter().map(|s| s.prior).collect::<Vec<_>>(),
            &[ROW[5] / sum, ROW[1] / sum],
        );
        // The acting seat is untouched: still the uniform 1/3 from make_stats.
        approx(
            &child.s1_stats.iter().map(|s| s.prior).collect::<Vec<_>>(),
            &[1.0 / 3.0; 3],
        );
    }

    /// The same call with the searching seat on side TWO. The opponent is then
    /// side one, so the identical vector must land on `s1_stats` instead. A
    /// hard-coded seat (always s2 for "opponent") passes the test above and
    /// fails this one.
    #[test]
    fn opponent_priors_land_on_the_other_seat_when_the_searcher_is_side_two() {
        let mut tree = tree_with_child(2, 3);
        let pending = vec![((0usize, 0usize), 0usize, vec![Some(5), Some(1)])];
        let heads = opponent_only_heads(&ROW);
        resolve_pending_priors(&mut tree, &pending, &heads, false, PriorSeat::Opponent);
        let sum = ROW[5] + ROW[1];
        let child = &tree.decisions[1];
        approx(
            &child.s1_stats.iter().map(|s| s.prior).collect::<Vec<_>>(),
            &[ROW[5] / sum, ROW[1] / sum],
        );
        approx(
            &child.s2_stats.iter().map(|s| s.prior).collect::<Vec<_>>(),
            &[1.0 / 3.0; 3],
        );
    }

    /// Both heads resolved against the SAME batch through the ONE call the
    /// search makes. The two seats must end up with their own head's
    /// distribution — a mutant that feeds one array to both seats leaves the
    /// two stat vectors equal.
    ///
    /// The two pending lists are DELIBERATELY different — different branch
    /// keys, different rows, different maps, different arities. Passing the
    /// same list twice (as this test first did) makes transposing the two
    /// arguments a no-op, and that transposition is a real live-path defect:
    /// `pending_opponent_maps` is gated on the flag AND on the opponent having
    /// a real choice, so in production it is a strict subset carrying different
    /// maps. Under the swap the acting seat is priced off the opponent's map
    /// and its own branches are never resolved.
    #[test]
    fn the_two_heads_stay_separated_when_resolved_off_one_batch() {
        // branch 0 -> child decision 1 (2 arms each seat)
        // branch 1 -> child decision 2 (3 arms on side one, 2 on side two)
        let mut tree = Tree {
            decisions: vec![decision(2, 2), decision(2, 2), decision(3, 2)],
            chances: vec![ChanceNode {
                branches: vec![branch(Some(1)), branch(Some(2))],
            }],
        };
        // Two batch rows; the acting head favours slot 0, the opponent head
        // slot 3, and rows 0/1 differ so a row mix-up is visible too.
        let self_rows = [0.7f32, 0.1, 0.1, 0.1, 0.1, 0.7, 0.1, 0.1];
        let opponent_rows = [0.1f32, 0.1, 0.1, 0.7, 0.7, 0.1, 0.1, 0.1];
        let heads = HeadPair::new(&self_rows, &opponent_rows, 4, true).expect("same width");
        // The acting seat has BOTH branches; the opponent only branch 0 (its
        // arity at branch 1 is 2 while the acting node there has 3 arms, so a
        // swapped call would also change the fallback count).
        let acting_pending = vec![
            ((0usize, 0usize), 0usize, vec![Some(0), Some(3)]),
            ((0usize, 1usize), 1usize, vec![Some(0), Some(1), Some(3)]),
        ];
        // A different map from the acting seat's, so a transposed call also
        // changes what lands on side one at branch 0.
        let opponent_pending = vec![((0usize, 0usize), 0usize, vec![Some(1), Some(3)])];
        let resolution = resolve_round_priors(
            &mut tree,
            &acting_pending,
            &opponent_pending,
            &heads,
            &Seat(true),
        );
        // The seats are attributed, not summed: two acting branches and one
        // opponent branch. A transposed call would move these across seats
        // while leaving `applied` at 3.
        assert_eq!(counts(resolution), (3, 0, 2, 0, 1, 0));
        assert_ne!(resolution.opponent_digest, 0);
        // Branch 1 is the acting seat's alone: its child's side-one arms carry
        // row 1 of the ACTING head, and its side-two arms stay uniform. A
        // transposed call leaves this node untouched entirely.
        approx(
            &priors_of_stats(&tree.decisions[2].s1_stats),
            &[0.1 / 0.9, 0.7 / 0.9, 0.1 / 0.9],
        );
        approx(&priors_of_stats(&tree.decisions[2].s2_stats), &[0.5, 0.5]);
        assert!(
            tree.chances[0].branches[1].child_opponent_priors.is_none(),
            "the opponent had no pending map for branch 1"
        );
        let child = &tree.decisions[1];
        approx(
            &child.s1_stats.iter().map(|s| s.prior).collect::<Vec<_>>(),
            &[0.875, 0.125],
        );
        approx(
            &child.s2_stats.iter().map(|s| s.prior).collect::<Vec<_>>(),
            &[0.125, 0.875],
        );
        // And each head is stored in its OWN branch slot, tagged with the seat
        // that owns the arms.
        let branch = &tree.chances[0].branches[0];
        assert_eq!(
            branch.child_self_priors.as_ref().map(|(s, _)| *s),
            Some(true)
        );
        assert_eq!(
            branch.child_opponent_priors.as_ref().map(|(s, _)| *s),
            Some(false)
        );
        approx(
            &branch.child_self_priors.as_ref().unwrap().1,
            &[0.875, 0.125],
        );
        approx(
            &branch.child_opponent_priors.as_ref().unwrap().1,
            &[0.125, 0.875],
        );
    }

    /// Each pending entry reads ITS OWN batch row. Two branches in one round,
    /// with rows whose gathers are distinguishable: a mutant that always reads
    /// row 0 (or `row` without the `* action_count` stride) gives both
    /// branches the same priors.
    #[test]
    fn each_branch_reads_its_own_batch_row() {
        let mut tree = Tree {
            decisions: vec![decision(2, 2), decision(2, 2), decision(2, 2)],
            chances: vec![ChanceNode {
                branches: vec![branch(Some(1)), branch(Some(2))],
            }],
        };
        // row 0 favours slot 0, row 1 favours slot 1.
        let flat = vec![0.9f32, 0.1, 0.0, 0.1, 0.9, 0.0];
        let heads = HeadPair::new(&flat, &flat, 3, true).expect("2 x 3");
        let map = vec![Some(0), Some(1)];
        let pending = vec![
            ((0usize, 0usize), 0usize, map.clone()),
            ((0usize, 1usize), 1usize, map),
        ];
        let resolution =
            resolve_pending_priors(&mut tree, &pending, &heads, true, PriorSeat::Opponent);
        assert_eq!(resolution.applied, 2);
        approx(
            &tree.decisions[1]
                .s2_stats
                .iter()
                .map(|s| s.prior)
                .collect::<Vec<_>>(),
            &[0.9, 0.1],
        );
        approx(
            &tree.decisions[2]
                .s2_stats
                .iter()
                .map(|s| s.prior)
                .collect::<Vec<_>>(),
            &[0.1, 0.9],
        );
    }

    /// A branch whose child does not exist yet is NOT a fallback: the priors
    /// are parked on the branch and consumed when the child is created.
    #[test]
    fn a_childless_branch_stores_its_priors_for_later_and_counts_as_applied() {
        let mut tree = Tree {
            decisions: vec![decision(2, 2)],
            chances: vec![ChanceNode {
                branches: vec![branch(None)],
            }],
        };
        let pending = vec![((0usize, 0usize), 0usize, vec![Some(0), Some(5)])];
        let heads = opponent_only_heads(&ROW);
        let resolution =
            resolve_pending_priors(&mut tree, &pending, &heads, true, PriorSeat::Opponent);
        assert_eq!(counts(resolution), (1, 0, 0, 0, 1, 0));
        assert_ne!(resolution.opponent_digest, 0);
        let stored = tree.chances[0].branches[0]
            .child_opponent_priors
            .as_ref()
            .expect("priors parked on the branch");
        assert_eq!(stored.0, false, "the stored flag is the OWNING seat");
        let sum = ROW[0] + ROW[5];
        approx(&stored.1, &[ROW[0] / sum, ROW[5] / sum]);
        assert!(
            tree.chances[0].branches[0].child_self_priors.is_none(),
            "the opponent head must not write the acting seat's slot"
        );
    }

    /// An arity disagreement between the map and the node's arms is counted as
    /// a fallback and leaves the node's uniform priors intact — and it must
    /// still be STORED, because the mismatch is with the child that exists
    /// now, not necessarily with a later one.
    #[test]
    fn an_arity_mismatch_falls_back_without_touching_the_stats() {
        let mut tree = tree_with_child(2, 3);
        // Two mapped options for a three-arm opponent node.
        let pending = vec![((0usize, 0usize), 0usize, vec![Some(0), Some(5)])];
        let heads = opponent_only_heads(&ROW);
        let resolution =
            resolve_pending_priors(&mut tree, &pending, &heads, true, PriorSeat::Opponent);
        // Gathered but REFUSED on arity: the digest still moved, because the
        // gather happened. Counts and digest answer different questions and
        // this is the case that separates them.
        assert_eq!(counts(resolution), (0, 1, 0, 0, 0, 1));
        assert_ne!(resolution.opponent_digest, 0);
        approx(
            &tree.decisions[1]
                .s2_stats
                .iter()
                .map(|s| s.prior)
                .collect::<Vec<_>>(),
            &[1.0 / 3.0; 3],
        );
        // The docstring's claim, asserted: parked despite the mismatch, so a
        // later child with the right arity still gets the model's priors.
        let stored = tree.chances[0].branches[0]
            .child_opponent_priors
            .as_ref()
            .expect("an arity mismatch with TODAY's child must still park the vector");
        assert_eq!(stored.0, false);
        let sum = ROW[0] + ROW[5];
        approx(&stored.1, &[ROW[0] / sum, ROW[5] / sum]);
    }

    /// An unmapped option makes the branch a fallback and writes NOTHING —
    /// neither the stats nor the branch slot, so a later child does not pick
    /// up a half-built vector.
    #[test]
    fn a_gather_fallback_writes_nothing_at_all() {
        let mut tree = tree_with_child(2, 2);
        let pending = vec![((0usize, 0usize), 0usize, vec![Some(0), None])];
        let heads = opponent_only_heads(&ROW);
        let resolution =
            resolve_pending_priors(&mut tree, &pending, &heads, true, PriorSeat::Opponent);
        // The gather itself failed, so nothing was folded and the digest stays
        // at its "nothing gathered" zero -- distinguishable from the
        // gathered-then-refused case above.
        assert_eq!(counts(resolution), (0, 1, 0, 0, 0, 1));
        assert_eq!(resolution.opponent_digest, 0);
        assert!(tree.chances[0].branches[0].child_opponent_priors.is_none());
        approx(
            &tree.decisions[1]
                .s2_stats
                .iter()
                .map(|s| s.prior)
                .collect::<Vec<_>>(),
            &[0.5, 0.5],
        );
    }

    /// The acting-seat path routes to the acting seat's slot and stats, so the
    /// seat-routing tests above cannot pass by writing everything to s2.
    #[test]
    fn the_acting_head_writes_the_acting_seat() {
        let mut tree = tree_with_child(2, 2);
        let pending = vec![((0usize, 0usize), 0usize, vec![Some(0), Some(5)])];
        let heads = HeadPair::new(&ROW, &[], ROW.len(), true).expect("no opponent head");
        resolve_pending_priors(&mut tree, &pending, &heads, true, PriorSeat::Acting);
        let sum = ROW[0] + ROW[5];
        approx(
            &tree.decisions[1]
                .s1_stats
                .iter()
                .map(|s| s.prior)
                .collect::<Vec<_>>(),
            &[ROW[0] / sum, ROW[5] / sum],
        );
        approx(
            &tree.decisions[1]
                .s2_stats
                .iter()
                .map(|s| s.prior)
                .collect::<Vec<_>>(),
            &[0.5, 0.5],
        );
        assert!(tree.chances[0].branches[0].child_opponent_priors.is_none());
    }

    // -----------------------------------------------------------------
    // root: option-list selection and both-seat resolution
    // -----------------------------------------------------------------

    /// Rattata (toxic/seismictoss) vs Chansey (splash only): the two seats have
    /// DIFFERENT option lists. That asymmetry is load-bearing — `minimal.state`
    /// gives both seats `[Move(0), Move(1)]`, so a swap there is invisible and
    /// the first version of this test passed against a mutant that dropped the
    /// swap entirely.
    const ASYMMETRIC: &str = include_str!("test_fixtures/analytic_toxic.state");

    /// Interior-node seat selection, against a REAL engine state. With the
    /// searching seat on side one the acting list is the engine's side-one
    /// options and the opponent list is side two's; on side two they swap.
    #[test]
    fn branch_seats_selects_each_seats_own_engine_options() {
        pyo3::Python::initialize();
        let state = crate::parse_state(ASYMMETRIC.trim()).expect("fixture parses");
        let (s1, s2) = state.get_all_options();
        assert!(
            !s1.is_empty() && !s2.is_empty(),
            "fixture offers both seats"
        );
        assert_ne!(
            s1, s2,
            "this fixture must distinguish the seats or the swap is untestable"
        );

        let seats = branch_seats(&state, &Seat(true));
        assert_eq!(seats.acting_options, s1);
        assert_eq!(seats.opponent_options, s2);

        let seats = branch_seats(&state, &Seat(false));
        assert_eq!(
            seats.acting_options, s2,
            "searching on side two makes side two's options the acting list"
        );
        assert_eq!(seats.opponent_options, s1);
    }

    /// The empty-side defence. A seat the engine offers nothing must still get
    /// one arm, because `new_decision_node` gives it one and a prior vector of
    /// length 0 against a stat vector of length 1 falls back on every branch.
    ///
    /// Driven through `seat_split` rather than `branch_seats` because the case
    /// is unreachable through a real `State` by construction — the
    /// terminal-options patch, cited on that function.
    #[test]
    fn a_seat_the_engine_offers_nothing_still_gets_one_arm() {
        let seats = seat_split(Vec::new(), vec![MoveChoice::None; 2], true);
        assert_eq!(
            seats.acting_options,
            vec![MoveChoice::None],
            "an empty acting side must be widened to one arm"
        );
        assert_eq!(seats.opponent_options.len(), 2);

        let seats = seat_split(Vec::new(), vec![MoveChoice::None; 2], false);
        assert_eq!(seats.acting_options.len(), 2);
        assert_eq!(
            seats.opponent_options,
            vec![MoveChoice::None],
            "an empty opponent side must be widened too"
        );

        let seats = seat_split(Vec::new(), Vec::new(), true);
        assert_eq!(seats.acting_options, vec![MoveChoice::None]);
        assert_eq!(seats.opponent_options, vec![MoveChoice::None]);
    }

    /// The widened arm must read as "no real choice", so the caller skips it
    /// rather than building a one-slot map for an absent seat.
    #[test]
    fn a_widened_empty_seat_reads_as_no_real_choice() {
        let seats = seat_split(Vec::new(), Vec::new(), true);
        assert!(is_single_none(&seats.acting_options));
        assert!(is_single_none(&seats.opponent_options));
    }

    #[test]
    fn is_single_none_is_the_no_real_choice_shape() {
        assert!(is_single_none(&[MoveChoice::None]));
        assert!(!is_single_none(&[]));
        assert!(!is_single_none(&[MoveChoice::None, MoveChoice::None]));
    }

    #[test]
    fn root_seats_selects_the_searching_seats_own_option_list() {
        let mut node = decision(3, 2);
        node.s1_options = vec![MoveChoice::None; 3];
        node.s2_options = vec![MoveChoice::None; 2];

        let seats = root_seats(&node, &Seat(true));
        assert_eq!(seats.acting_options.len(), 3, "side one is searching");
        assert_eq!(seats.opponent_options.len(), 2);

        let seats = root_seats(&node, &Seat(false));
        assert_eq!(seats.acting_options.len(), 2, "side two is searching");
        assert_eq!(seats.opponent_options.len(), 3);
    }

    /// Root seat routing, searching seat on side one: acting head to `s1`,
    /// opponent head to `s2`, and `acting` echoed back for the report's
    /// `root_priors`.
    #[test]
    fn root_resolution_routes_each_head_to_its_own_seat() {
        let mut node = decision(2, 2);
        let acting_row = [0.7f32, 0.1, 0.1, 0.1];
        let opponent_row = [0.1f32, 0.1, 0.1, 0.7];
        let heads = HeadPair::new(&acting_row, &opponent_row, 4, true).expect("same width");
        let map = vec![Some(0), Some(3)];
        let resolution = resolve_root_priors(&mut node, &heads, &Seat(true), &map, Some(&map));
        assert_eq!(resolution.fallbacks, 0);
        approx(
            &resolution
                .acting
                .expect("acting priors echoed for the report"),
            &[0.875, 0.125],
        );
        approx(&priors_of_stats(&node.s1_stats), &[0.875, 0.125]);
        approx(&priors_of_stats(&node.s2_stats), &[0.125, 0.875]);
    }

    /// Same call, searching seat on side TWO. Both destinations must swap.
    #[test]
    fn root_resolution_swaps_destinations_with_the_searching_seat() {
        let mut node = decision(2, 2);
        let acting_row = [0.7f32, 0.1, 0.1, 0.1];
        let opponent_row = [0.1f32, 0.1, 0.1, 0.7];
        let heads = HeadPair::new(&acting_row, &opponent_row, 4, true).expect("same width");
        let map = vec![Some(0), Some(3)];
        resolve_root_priors(&mut node, &heads, &Seat(false), &map, Some(&map));
        approx(&priors_of_stats(&node.s2_stats), &[0.875, 0.125]);
        approx(&priors_of_stats(&node.s1_stats), &[0.125, 0.875]);
    }

    /// No opponent map (flag off, no opponent head, or no real opponent
    /// choice): the opponent seat stays uniform and nothing is counted.
    #[test]
    fn root_resolution_without_an_opponent_map_leaves_that_seat_uniform() {
        let mut node = decision(2, 2);
        let acting_row = [0.7f32, 0.1, 0.1, 0.1];
        let heads = HeadPair::new(&acting_row, &[], 4, true).expect("no opponent head");
        let map = vec![Some(0), Some(3)];
        let resolution = resolve_root_priors(&mut node, &heads, &Seat(true), &map, None);
        assert_eq!(resolution.fallbacks, 0);
        approx(&priors_of_stats(&node.s1_stats), &[0.875, 0.125]);
        approx(&priors_of_stats(&node.s2_stats), &[0.5, 0.5]);
    }

    /// A root fallback on one seat does not suppress the other, and each is
    /// counted once.
    #[test]
    fn root_resolution_counts_each_seats_fallback_independently() {
        let mut node = decision(2, 2);
        let acting_row = [0.7f32, 0.1, 0.1, 0.1];
        let opponent_row = [0.1f32, 0.1, 0.1, 0.7];
        let heads = HeadPair::new(&acting_row, &opponent_row, 4, true).expect("same width");
        let good = vec![Some(0), Some(3)];
        let unmapped = vec![Some(0), None];
        let resolution =
            resolve_root_priors(&mut node, &heads, &Seat(true), &good, Some(&unmapped));
        assert_eq!(resolution.fallbacks, 1);
        assert!(resolution.acting.is_some());
        approx(&priors_of_stats(&node.s1_stats), &[0.875, 0.125]);
        approx(&priors_of_stats(&node.s2_stats), &[0.5, 0.5]);

        let mut node = decision(2, 2);
        let resolution =
            resolve_root_priors(&mut node, &heads, &Seat(true), &unmapped, Some(&good));
        assert_eq!(resolution.fallbacks, 1);
        assert!(
            resolution.acting.is_none(),
            "a fallen-back acting seat must not be echoed as root_priors"
        );
        approx(&priors_of_stats(&node.s1_stats), &[0.5, 0.5]);
        approx(&priors_of_stats(&node.s2_stats), &[0.125, 0.875]);
    }

    fn priors_of_stats(stats: &[MoveStats]) -> Vec<f32> {
        stats.iter().map(|s| s.prior).collect()
    }
}
