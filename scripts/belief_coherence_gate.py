#!/usr/bin/env python3
"""V1 — whole-game belief coherence harness (the plan's highest-value item).

No test anywhere asserted that the TRUE variant stays in the candidate set at any point of a real
game. Containment is the property whose violation is maximally harmful under the owner criterion: a
belief that has excluded the truth poisons ``CANDIDATE_SET_COUNT``, ``UNCERTAINTY``, every
``possible-*`` count and every sampled search world at once, and it fails SILENTLY -- the set still
narrows, just to the wrong thing.

The local env drives the simulator, so it holds BOTH true teams already (each seat's own opening
``|request|``); no new channel is needed. Per turn, per revealed opponent mon, from BOTH
perspectives, this asserts the plan's seven families:

1. **containment** -- the true variant (by ``variant_id``, cross-checked on ``variant_identity``)
   is in ``candidate_variants``;
2. **non-emptiness** -- the set is never empty for a recognized species;
3. **monotonicity** -- the set never grows except through the documented inconsistent-fallback,
   and every fallback is counted and attributed, not silently absorbed;
4. **zero pin conflicts** -- ``variant_pin_conflicts`` stays empty with narrowing off;
5. **ruled-out soundness** -- ``ruled_out_abilities`` never contains the true ability and
   ``ruled_out_items`` never the true item. Only the ITEM arm has a producer in this format; the
   ability arm is reported ``n/a`` and does NOT count toward reachability (see
   ``_ABILITY_ARM_STATUS``);
6. **derived-stat legality** -- every surviving variant's spread stays inside the generator's legal
   set (the C1 guard, at game scale);
7. **clone equivalence** -- sampled-world beliefs equal the parent's on the checked properties.

Exit criterion (plan §1): **zero violations of 1/2/4/5/6/7 over the full sweep**, and every
monotonicity fallback attributed to a known cause. Any violation is a launch-gating defect by
definition -- it is a wrong belief in live data shape.

REACHABILITY IS REPORTED AND GATED, not assumed. A containment sweep over games where no opponent
mon was ever recognized passes vacuously, which is the bug and not the fix -- so the verdict is
FAIL unless the run actually reached recognized mons, narrowed sets, and pinned sets.

Usage:
    uv run python scripts/belief_coherence_gate.py --games 200 --seed 7 \
        --showdown-root ~/workspace/pokerena/vendor/pokemon-showdown \
        --out runs/belief-coherence-2026-08-04
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from pokezero.belief import RevealedPokemonBelief, variant_identity  # noqa: E402
from pokezero.dex import load_showdown_dex_cached  # noqa: E402
from pokezero.local_showdown import LocalShowdownConfig, LocalShowdownEnv  # noqa: E402
from pokezero.randbat import (  # noqa: E402
    canonical_gen3_randbat_species_id,
    load_gen3_randbat_source_cached,
)
from dataclasses import replace  # noqa: E402

from pokezero.observation import DEFAULT_OBSERVATION_FEATURE_MASKS  # noqa: E402
from pokezero.showdown import MOVE_ACTION_COUNT, _variant_spread_stats  # noqa: E402
from pokezero.tier2 import variant_has_physical_attack  # noqa: E402
from tier2_gate import _first_requests, _team_truth  # noqa: E402

PLAYERS = ("p1", "p2")

# Family 5 has two arms and only ONE of them can fire in this format.
#
# ``ruled_out_abilities`` has exactly one producer in the engine -- the Intimidate non-trigger rule
# in ``_resolve_pending_switches_as_no_trigger`` (``belief.py``). Both its gates,
# ``_can_queue_intimidate_non_trigger`` and ``_can_rule_out_intimidate``, require the mon to have
# Intimidate AND at least one OTHER possible ability, because a mon whose only ability is Intimidate
# cannot have it eliminated. Measured against the gen3 randbats pool (220 species): all 11
# Intimidate carriers -- Arbok, Arcanine, Tauros, Gyarados, Granbull, Stantler, Hitmontop,
# Mightyena, Masquerain, Mawile, Salamence -- have Intimidate as their SOLE ability, and only 15
# pool species have more than one possible ability at all (none of them an Intimidate carrier). So
# the arm is structurally unfailable here, and the sweep confirms it: 0 mons with a non-empty
# ``ruled_out_abilities`` over 206,653 observations.
#
# The docstring used to justify this arm as "the TRAPPER_ALIVE correctness check ... Shadow Tag and
# Arena Trap are both pool-reachable". That is false as a justification. The SPECIES are reachable
# (Wobbuffet/Shadow Tag, Dugtrio/Arena Trap) but each is its species' only ability, so neither can
# ever land in ``ruled_out_abilities`` -- this sweep cannot check TRAPPER_ALIVE's soundness, and
# claiming it did is the vacuous-assertion failure the plan's §3 names.
_ABILITY_ARM_STATUS = "n/a (no producer in gen3 randbats)"


def _other(player: str) -> str:
    return "p2" if player == "p1" else "p1"


def _norm(value: Any) -> str:
    return "".join(ch for ch in str(value or "").lower() if ch.isalnum())


def _norm_moves(moves: Sequence[Any]) -> frozenset[str]:
    """Normalized move-id set.

    Request payloads suffix some move ids with their PP (``tier2.py``'s note), and the pool's own
    ids are bare, so the comparison has to be on normalized ids and as a SET -- the generator
    shuffles move order, so a tuple comparison would miss the true variant and report a phantom
    containment violation.
    """
    out: set[str] = set()
    for move in moves or ():
        token = _norm(move)
        if token:
            out.add(token)
    return frozenset(out)


class Violation(dict):
    """One assertion failure, JSON-shaped for the artifact."""


def _true_variant_for(
    source, *, species: str, truth_row: Mapping[str, Any]
) -> Optional[Mapping[str, Any]]:
    """The pool variant the generator actually produced for this mon, as a summary mapping.

    Matched on the normalized (moves, ability, item) triple against the species universe, then
    returned as the source's OWN ``to_summary()`` mapping so ``variant_id`` and
    ``variant_identity`` are directly comparable with what the belief engine hands out. Returning
    a hand-built mapping instead would compare two different constructions and could report
    containment violations that are really identity-construction mismatches.
    """
    universe = source.universe_for(species)
    if universe is None:
        return None
    want_moves = _norm_moves(truth_row.get("moves") or ())
    want_ability = _norm(truth_row.get("ability"))
    want_item = _norm(truth_row.get("item"))
    exact: list[Mapping[str, Any]] = []
    for variant in universe.variants:
        if _norm_moves(variant.moves) != want_moves:
            continue
        if want_ability and _norm(variant.ability) != want_ability:
            continue
        if want_item and _norm(variant.item) != want_item:
            continue
        exact.append(variant.to_summary())
    if len(exact) == 1:
        return exact[0]
    if exact:
        # Several pool variants are indistinguishable on the triple (same moves/ability/item under
        # different roles). Any of them IS the truth for containment purposes; take the first and
        # record that the match was ambiguous rather than pretending it was exact.
        first = dict(exact[0])
        first["_ambiguous_match"] = len(exact)
        first["_ambiguous_variant_ids"] = [str(v.get("variant_id")) for v in exact]
        return first
    return None


def _check_pp_ledger(
    *,
    env,
    dex,
    game: int,
    turn: int,
    counts: Counter,
    violations: dict[str, list[Violation]],
) -> None:
    """V3 -- the PP-ledger differential, riding this sweep (plan item V3).

    The omniscient channel: each seat's OWN request carries its active mon's true remaining PP per
    move (``active[0].moves[].pp``). The OPPONENT's belief independently derives the same number as
    ``max_pp - move_uses``, and ``showdown._encode_opponent_move_pp`` turns that into
    ``OPP_MOVE_PP_OFFSET``. Those are two independent derivations of one quantity, so comparing them
    is a real differential rather than a self-check.

    Exit criterion (plan V3): **100% agreement with true remaining PP**, or a settled owner decision
    that the contract is "observed uses". Step 1 of V3 -- reading the engine rules rather than
    recalling them -- is recorded in ``deployment/docs/v3-pp-ledger-engine-rules-20260804.md``; it
    settled that Pressure DOES double-charge in gen3 and that ``move_uses`` already accounts for it,
    so the plan's suspected defect does not hold. This is the measurement that was still owed.

    Only REVEALED moves are compared: the column is populated for revealed moves only, so an
    unrevealed one has no encoded value to be wrong about.
    """
    for owner in PLAYERS:
        request = env._latest_requests.get(owner) or {}
        active = request.get("active") or []
        if not active or not isinstance(active[0], Mapping):
            # Counted, not silently dropped: a differential that skips most of its opportunities
            # and reports "0 violations" is indistinguishable from one that checked everything.
            counts["pp_skipped_no_active_request"] += 1
            continue
        truth: dict[str, tuple[int, int]] = {}
        for entry in active[0].get("moves") or ():
            if not isinstance(entry, Mapping):
                continue
            move_id = _norm(entry.get("id"))
            pp, max_pp = entry.get("pp"), entry.get("maxpp")
            if move_id and isinstance(pp, int) and isinstance(max_pp, int):
                truth[move_id] = (pp, max_pp)
        if not truth:
            counts["pp_skipped_request_without_pp"] += 1
            continue
        # The opponent's belief about THIS mon.
        view = env._belief_engine.resolved_player_view(_other(owner))
        for belief in view.opponent_pokemon:
            if not belief.active:
                continue
            if belief.transformed:
                # While a mon is transformed the request's `active[0].moves` describes the COPIED
                # slots, not the mon's own set. `sim/pokemon.ts transformInto` (`:1306-1326`, not
                # overridden in the gen3 chain -- the five `data/mods/*/scripts.ts` that redefine
                # it are format mods) clears `moveSlots` and rebuilds them from the target:
                #
                #     pp    = Math.min(5, move.pp)
                #     maxpp = gen >= 5 ? pp : calculatePP(move, this.ppUps[i] || 0)
                #
                # The `|| 0` is the whole story, and TWO earlier versions of this comment got it
                # wrong -- both claimed the slots keep "the target's maxpp". They do not.
                # `this.ppUps` is built from the TRANSFORMER's own `set.moves`
                # (`sim/pokemon.ts:350-373`), so it is indexed by slot and sized to the
                # transformer's move count. A gen3 randbats Ditto has `movepool: ["transform"]`,
                # so `ppUps == [3]`: slot 0 gets 3 PP-ups and slots 1-3 fall through to 0.
                # With `calculatePP = pp * (5 + ppUps) / 5` (`sim/battle.ts:2390-2395`):
                #
                #     slot 0  psychic      base 10  @3 -> 16    <- the only boosted slot
                #     slot 1  calmmind     base 20  @0 -> 20
                #     slot 2  icepunch     base 15  @0 -> 15
                #     slot 3  thunderbolt  base 15  @0 -> 15
                #
                # Measured on seed 4711 game 90 turn 48, the transformed Ditto reports exactly
                # 16/20/15/15 -- while JIRACHI'S OWN request that turn reports 16/32/24/24. The
                # copied maxpp is not the target's; three of the four slots are the copied move's
                # unboosted base. The observed numbers were what disproved the earlier claim, and
                # they were sitting in the same paragraph that made it.
                #
                # So the two sides of this differential are not the same quantity during a
                # transform, and comparing them is an error in the HARNESS. It stayed invisible
                # until a Ditto transformed into the opposing DITTO: the copied slot is then
                # `transform` 5/16, which collides with the mon's own move id, and the gate read
                # 5 against belief's untouched 16 - 1 = 15. Five violations, one game, seed 4711.
                #
                # Skipped rather than modelled: belief's 15 is the right answer for Ditto's OWN
                # Transform PP, which is what survives the revert. What the encoded column does
                # NOT represent is that a transformed mon cannot reach its own moves at all --
                # that is an action-availability gap, not a PP disagreement, and it is recorded as
                # an open item in the status doc rather than papered over here.
                counts["pp_skipped_transformed_mon"] += 1
                continue
            uses = {_norm(k): int(v) for k, v in (belief.move_uses or ())}
            revealed = {_norm(m) for m in (belief.revealed_moves or ())}
            for move_id in sorted(revealed):
                observed = truth.get(move_id)
                info = dex.move_info(move_id)
                if observed is None:
                    # The belief has this move revealed but the owning seat's request does not
                    # list it. Fully attributed by measurement: this fired 163 times over 400
                    # games on seed 4711 BEFORE the transform skip above existed, and 0 times
                    # after -- every instance was a transformed mon whose slots had been replaced.
                    # Kept as a counter so a new cause cannot arrive silently.
                    counts["pp_skipped_move_absent_from_request"] += 1
                    continue
                if info is None or not info.max_pp:
                    counts["pp_skipped_no_dex_max_pp"] += 1
                    continue
                true_remaining, true_max = observed
                # The encoder's own arithmetic (showdown.py): remaining = max(0, max_pp - uses).
                believed = max(0, int(info.max_pp) - uses.get(move_id, 0))
                counts["pp_comparisons"] += 1
                if int(info.max_pp) != true_max:
                    violations["pp_max"].append(
                        Violation(
                            {
                                "game": game, "turn": turn, "owner": owner,
                                "species": belief.species, "move": move_id,
                                "detail": "dex max_pp disagrees with the request's maxpp",
                                "dex_max_pp": int(info.max_pp), "request_maxpp": true_max,
                            }
                        )
                    )
                if believed != true_remaining:
                    violations["pp_remaining"].append(
                        Violation(
                            {
                                "game": game, "turn": turn, "owner": owner,
                                "species": belief.species, "move": move_id,
                                "detail": "believed remaining PP != true remaining PP",
                                "believed": believed, "true": true_remaining,
                                "max_pp": int(info.max_pp),
                                "uses": uses.get(move_id, 0),
                                "revealed_moves": list(belief.revealed_moves),
                            }
                        )
                    )
                elif believed != int(info.max_pp):
                    # A comparison where PP has actually been spent. Counted separately because
                    # agreeing on an untouched full-PP move is nearly free and would let this
                    # family look exercised while never testing the ledger's arithmetic.
                    counts["pp_spent_comparisons"] += 1


def _record_pressure_reachability(*, env, counts: Counter) -> None:
    """Fold the belief engine's Pressure tallies into the run counts (see
    `PublicBattleBeliefEngine.pressure_charge_counts`). Reported per game because the engine is
    rebuilt per game; the counter names are prefixed so they read as reachability, not violations.
    """
    for key, value in (env._belief_engine.pressure_charge_counts or {}).items():
        counts[f"pressure_{key}"] += int(value)


def _check_mon(
    *,
    belief: RevealedPokemonBelief,
    true_variant: Mapping[str, Any],
    truth_row: Mapping[str, Any],
    dex,
    source,
    perspective: str,
    game: int,
    turn: int,
    seen_counts: dict[tuple[str, str], int],
    counts: Counter,
    violations: dict[str, list[Violation]],
    fallbacks: list[dict[str, Any]],
) -> None:
    key = (perspective, belief.key if hasattr(belief, "key") else belief.species)
    candidates = tuple(belief.candidate_variants or ())
    ctx = {
        "game": game,
        "turn": turn,
        "perspective": perspective,
        "species": belief.species,
        "revealed_moves": list(belief.revealed_moves),
        "candidate_count": len(candidates),
        "true_variant_id": true_variant.get("variant_id"),
    }

    # (2) non-emptiness -- a recognized species must never have an empty set.
    if not candidates:
        violations["non_empty"].append(Violation({**ctx, "detail": "empty candidate set"}))
        return
    counts["mon_observations"] += 1

    # (1) containment, by variant_id, cross-checked on variant_identity.
    ids = {str(v.get("variant_id")) for v in candidates}
    # ANY of the indistinguishable truth candidates counts. All 34 ambiguous groups in the pool
    # share one `variant_identity` (they differ only in role/source_set_id), so no filter can
    # separate them -- but relying on that unasserted invariant would make `exact[0]` a latent
    # false-violation generator once narrowing is on.
    truth_ids = {str(vid) for vid in (true_variant.get("_ambiguous_variant_ids") or ())}
    truth_ids.add(str(true_variant.get("variant_id")))
    contained_by_id = bool(truth_ids & ids)
    identities = {variant_identity(v) for v in candidates}
    contained_by_identity = variant_identity(true_variant) in identities
    if not contained_by_id:
        violations["containment"].append(
            Violation({**ctx, "detail": "true variant_id absent", "candidate_ids": sorted(ids)})
        )
    elif not contained_by_identity:
        # variant_id present but the identity tuple disagrees: the two comparison keys have
        # drifted apart, which would silently break narrow_candidate_variants' intersection.
        violations["identity_drift"].append(
            Violation({**ctx, "detail": "variant_id contained but variant_identity absent"})
        )
    else:
        counts["containment_ok"] += 1
    if len(candidates) == 1 and contained_by_id:
        counts["pinned_and_correct"] += 1

    # (3) monotonicity -- the set may shrink or hold, never grow, except via the documented
    # inconsistent-fallback (which widens to the FULL species pool at uncertainty 1.0).
    universe = source.universe_for(belief.species)
    pool_size = len(universe.variants) if universe is not None else 0
    previous = seen_counts.get(key)
    if previous is not None and len(candidates) > previous:
        looks_like_fallback = (
            pool_size and len(candidates) == pool_size and float(belief.uncertainty) >= 1.0
        )
        # SECOND legitimate growth cause, found by this sweep and attributed rather than absorbed:
        # an item mutation. A revealed item narrows the set; Knock Off / Trick then removes it, the
        # candidate filter stops constraining on it, and previously-excluded variants are
        # re-admitted. Captured live:
        #   |-enditem|p1a: Raichu|Leftovers|[from] move: Knock Off|[of] p2a: Lickitung
        #   revealed_item Leftovers, candidates 3 -> 4
        # It widens rather than narrows, so containment is preserved -- the SAFE direction -- but it
        # discards evidence the reveal had already established, so it is reported with its cause
        # rather than treated as clean. The plan's family-3 criterion is attribution, not zero
        # growth; silently absorbing this is what it forbids.
        item_mutation = bool(
            getattr(belief, "item_mutated", False) or getattr(belief, "item_removed", False)
        )
        if item_mutation and not looks_like_fallback:
            counts["item_mutation_regrowths"] += 1
            fallbacks.append(
                {
                    **ctx,
                    "previous": previous,
                    "pool_size": pool_size,
                    "cause": "item mutation re-widened the set (Knock Off / Trick)",
                    "revealed_item": belief.revealed_item,
                    "original_public_item": getattr(belief, "original_public_item", None),
                }
            )
        elif looks_like_fallback:
            counts["inconsistent_fallbacks"] += 1
            # ATTRIBUTED, not just counted: the plan requires every fallback tied to a cause, and
            # a bare integer cannot be checked against that by a reader.
            fallbacks.append(
                {**ctx, "previous": previous, "pool_size": pool_size, "cause": "full-pool widening"}
            )
        else:
            violations["monotonicity"].append(
                Violation(
                    {
                        **ctx,
                        "detail": "candidate set grew outside the inconsistent-fallback",
                        "previous": previous,
                        "pool_size": pool_size,
                        "uncertainty": float(belief.uncertainty),
                    }
                )
            )
    if previous is not None and len(candidates) < previous:
        counts["narrowing_steps"] += 1
    seen_counts[key] = len(candidates)

    # (5) ruled-out soundness -- never rule out the truth. The ITEM arm is live; the ABILITY arm
    # has NO PRODUCER in this format and is reported n/a (see ``_ABILITY_ARM_STATUS``). The
    # ability assertion is kept -- it costs one set-membership test and would bind the day a
    # producer appears -- but it must not be advertised as coverage, and it is not counted as
    # reached.
    true_ability = _norm(truth_row.get("ability"))
    true_item = _norm(truth_row.get("item"))
    if true_ability and true_ability in {_norm(a) for a in belief.ruled_out_abilities or ()}:
        violations["ruled_out_ability"].append(
            Violation({**ctx, "detail": f"true ability {true_ability!r} was ruled out"})
        )
    if true_item and true_item in {_norm(i) for i in belief.ruled_out_items or ()}:
        violations["ruled_out_item"].append(
            Violation(
                {
                    **ctx,
                    "detail": f"true item {true_item!r} was ruled out",
                    # Enough context to adjudicate without re-running: which rule fired, what the
                    # surviving candidates now claim, and whether the set fell back. A bare
                    # "was ruled out" cannot distinguish a real soundness break from a
                    # mis-specified assertion, and guessing between those is how a false defect
                    # gets reported.
                    "ruled_out_items": list(belief.ruled_out_items),
                    "uncertainty": float(belief.uncertainty),
                    "candidate_items": sorted(
                        {str(v.get("item") or "") for v in candidates}
                    ),
                    "item_mutated": bool(belief.item_mutated),
                    "revealed_item": belief.revealed_item,
                    "current_public_item": getattr(belief, "current_public_item", None),
                    "original_public_item": getattr(belief, "original_public_item", None),
                    "condition": belief.condition,
                    "rule_out_evidence": [
                        e.detail for e in (belief.evidence or ()) if e.kind == "ruled-out-item"
                    ],
                }
            )
        )
    if belief.ruled_out_abilities:
        counts["mons_with_ruled_out_abilities"] += 1
    if belief.ruled_out_items:
        counts["mons_with_ruled_out_items"] += 1
    # The ability arm's ONLY producer is the Intimidate non-trigger rule, and
    # ``_can_queue_intimidate_non_trigger``/``_can_rule_out_intimidate`` both require the mon to
    # have Intimidate AND at least one other possible ability. Counting that precondition turns
    # "the arm never fired" from an inference into a measurement: if this stays 0 while
    # ``mon_observations`` is large, the arm is unfailable in this format rather than merely lucky.
    ability_ids = {_norm(a) for a in belief.possible_abilities or ()}
    if "intimidate" in ability_ids and len(ability_ids) > 1:
        counts["intimidate_ruleout_preconditions"] += 1

    # (6) derived-stat legality -- every surviving variant's spread inside the generator's legal
    # set. _variant_spread_stats RAISES on an illegal spread by design, so a raise here is the
    # signal, not an error to swallow.
    info = dex.species_info(canonical_gen3_randbat_species_id(belief.species) or belief.species)
    if info is not None and info.base_stats:
        level = int(true_variant.get("level") or truth_row.get("level") or 100)
        for candidate in candidates:
            try:
                spread = _variant_spread_stats(
                    info.base_stats,
                    level,
                    candidate,
                    variant_has_physical_attack(candidate.get("moves") or (), dex),
                )
            except Exception as exc:  # noqa: BLE001 -- an illegal spread is the finding
                violations["stat_legality"].append(
                    Violation({**ctx, "detail": f"illegal spread: {exc}", "variant": dict(candidate)})
                )
                continue
            if spread is None:
                counts["unevaluable_candidate_spreads"] += 1
            else:
                counts["stat_legality_checks"] += 1


def run_sweep(
    *,
    showdown_root: Path,
    games: int = 200,
    seed: int = 7,
    max_steps: int = 400,
    move_bias: float = 0.75,
    clone_equivalence_every: int = 10,
    investment_belief_narrowing: bool = False,
    item_belief_narrowing: bool = False,
) -> dict[str, Any]:
    """Run the coherence sweep and return its summary dict.

    Split out from ``main`` so ``tests/test_belief_coherence.py`` can run a short sweep as a real
    integration test rather than re-implementing the assertions -- two copies of a coherence check
    drifting apart is the same defect class this harness exists to catch.
    """

    class _Args:
        pass

    args = _Args()
    args.showdown_root = showdown_root
    args.games = games
    args.seed = seed
    args.max_steps = max_steps
    args.move_bias = move_bias
    args.clone_equivalence_every = clone_equivalence_every

    dex = load_showdown_dex_cached(args.showdown_root)
    source = load_gen3_randbat_source_cached(args.showdown_root)
    rng = random.Random(args.seed)
    # PIN the candidate-set belief source ON rather than deferring to POKEZERO_BELIEF_SET_SOURCE,
    # which defaults to "0". Deferring made the first run of this harness report 1492 "empty
    # candidate set" violations from three games -- not a defect, just every candidate set switched
    # off. A belief-containment sweep with no candidate sets is the vacuous pass this file exists to
    # refuse, so the flag is pinned here and re-asserted as a precondition below.
    # Narrowing flags are plumbed so V7 step 2 ("re-run V1 with both narrowing flags on") is
    # runnable from this harness. With them OFF, family 4 cannot fire at all -- see the verdict.
    # The narrowing switches live on the FEATURE MASKS, not on the config directly, and the
    # investment arm additionally requires the tier2 channel (`investment_belief_narrowing_active`
    # is `tier2_residuals_active() and ...`). Enabling them by hand rather than through this is
    # how a "narrowing on" run silently stays off.
    masks = DEFAULT_OBSERVATION_FEATURE_MASKS
    if investment_belief_narrowing or item_belief_narrowing:
        masks = replace(
            masks,
            tier2_residuals=masks.tier2_residuals or investment_belief_narrowing,
            investment_belief_narrowing=investment_belief_narrowing,
            item_belief_narrowing=item_belief_narrowing,
        )
    env = LocalShowdownEnv(
        LocalShowdownConfig(
            showdown_root=str(args.showdown_root),
            set_belief_source=True,
            feature_masks=masks,
        )
    )
    if not env.config.belief_set_source_enabled():
        env.close()
        raise RuntimeError(
            "candidate-set belief source is disabled; the sweep would be vacuous"
        )
    env_narrowing_flags = {
        "investment": bool(env.investment_belief_narrowing_active()),
        "item": bool(env.item_belief_narrowing_active()),
    }

    # Seeded at zero so the artifact distinguishes "measured, never happened" from "never
    # measured". A Counter only carries keys it incremented, which makes a missing
    # ``inconsistent_fallbacks`` read as an absent measurement -- and the plan requires every
    # fallback be ATTRIBUTED, which a reader cannot check against a key that is not there.
    counts: Counter = Counter(
        {
            "games": 0,
            "mon_observations": 0,
            "containment_ok": 0,
            "pinned_and_correct": 0,
            "narrowing_steps": 0,
            "inconsistent_fallbacks": 0,
            "item_mutation_regrowths": 0,
            "stat_legality_checks": 0,
            "unevaluable_candidate_spreads": 0,
            "pin_conflict_checks": 0,
            "clone_equivalence_checks": 0,
            "mons_with_ruled_out_abilities": 0,
            "mons_with_ruled_out_items": 0,
            "intimidate_ruleout_preconditions": 0,
            "pins_observed": 0,
            "ambiguous_true_variant_matches": 0,
            "mons_without_resolvable_true_variant": 0,
            "belief_mons_without_truth": 0,
            "pp_comparisons": 0,
            "pp_spent_comparisons": 0,
            # The PP arm's non-comparisons and its Pressure-branch reachability, seeded for the
            # same reason as everything else here: a missing key and a measured zero must not look
            # alike in the artifact.
            "pp_skipped_no_active_request": 0,
            "pp_skipped_request_without_pp": 0,
            "pp_skipped_transformed_mon": 0,
            "pp_skipped_move_absent_from_request": 0,
            "pp_skipped_no_dex_max_pp": 0,
            "pressure_charges": 0,
            "pressure_vs_pressure": 0,
            "pressure_doubled": 0,
            "pressure_single_self_targeted_vs_pressure": 0,
            "pressure_doubled_despite_blank_or_self_wire_slot": 0,
            "truncated_games": 0,
            "games_without_both_requests": 0,
        }
    )
    violations: dict[str, list[Violation]] = {
        name: []
        for name in (
            "containment",
            "identity_drift",
            "non_empty",
            "monotonicity",
            "pin_conflicts",
            "ruled_out_ability",
            "ruled_out_item",
            "stat_legality",
            "clone_equivalence",
            "pp_remaining",
            "pp_max",
        )
    }
    species_seen: set[str] = set()
    fallbacks: list[dict[str, Any]] = []

    try:
        for game in range(args.games):
            # Per-GAME rng: a single shared Random() made game N depend on every prior game's
            # action draws, so a violation could not be replayed in isolation -- diagnosing the
            # Leftovers defect needed a separate probe purely because of this.
            rng = random.Random(args.seed * 1_000_003 + game)
            env.reset(seed=args.seed + game)
            first = _first_requests(env.protocol_lines)
            if "p1" not in first or "p2" not in first:
                counts["games_without_both_requests"] += 1
                continue
            truth = {slot: _team_truth(first[slot]) for slot in PLAYERS}
            # variant_id of the true set per (slot, species), resolved once per game.
            true_variants: dict[str, dict[str, Mapping[str, Any]]] = {}
            for slot in PLAYERS:
                resolved: dict[str, Mapping[str, Any]] = {}
                for species_key, row in truth[slot].items():
                    canonical = canonical_gen3_randbat_species_id(species_key) or species_key
                    variant = _true_variant_for(source, species=canonical, truth_row=row)
                    if variant is None:
                        counts["mons_without_resolvable_true_variant"] += 1
                        continue
                    if variant.get("_ambiguous_match"):
                        counts["ambiguous_true_variant_matches"] += 1
                    resolved[canonical] = variant
                true_variants[slot] = resolved
            counts["games"] += 1

            seen_counts: dict[tuple[str, str], int] = {}
            steps = 0
            turn = 0
            while steps < args.max_steps and env.terminal() is None:
                requested = env.requested_players()
                if not requested:
                    break
                turn += 1

                for perspective in PLAYERS:
                    view = env._belief_engine.resolved_player_view(perspective)
                    opponent_slot = _other(perspective)
                    for belief in view.opponent_pokemon:
                        canonical = (
                            canonical_gen3_randbat_species_id(belief.species) or belief.species
                        )
                        truth_row = truth[opponent_slot].get(canonical)
                        true_variant = true_variants[opponent_slot].get(canonical)
                        if truth_row is None or true_variant is None:
                            counts["belief_mons_without_truth"] += 1
                            continue
                        species_seen.add(canonical)
                        _check_mon(
                            belief=belief,
                            true_variant=true_variant,
                            truth_row=truth_row,
                            dex=dex,
                            source=source,
                            perspective=perspective,
                            game=game,
                            turn=turn,
                            seen_counts=seen_counts,
                            counts=counts,
                            violations=violations,
                            fallbacks=fallbacks,
                        )

                # (8) V3 -- PP ledger vs the omniscient channel.
                _check_pp_ledger(
                    env=env, dex=dex, game=game, turn=turn,
                    counts=counts, violations=violations,
                )

                # (4) zero pin conflicts with narrowing off.
                conflicts = dict(env._belief_engine.variant_pin_conflicts)
                if conflicts:
                    violations["pin_conflicts"].append(
                        Violation(
                            {
                                "game": game,
                                "turn": turn,
                                "detail": "variant_pin_conflicts non-empty with narrowing off",
                                "conflicts": conflicts,
                                "investment_narrowing": env.investment_belief_narrowing_active(),
                                "item_narrowing": env.item_belief_narrowing_active(),
                            }
                        )
                    )
                counts["pin_conflict_checks"] += 1
                # ...and whether any PIN was ever written, which is the precondition for a
                # conflict to be recordable at all. `pin_conflict_checks` only counts loop
                # iterations, so it cannot tell "no conflicts because the state is clean" from
                # "no conflicts because nothing ever pinned". Reporting the config flag instead of
                # observed pin activity has the same hole: the flag can be ON while the tier2 /
                # investment producers still never write a pin in a given sweep.
                if getattr(env._belief_engine, "_variant_pins", None):
                    counts["pins_observed"] += 1

                # (7) clone equivalence -- a sampled world's beliefs must equal the parent's on
                # the checked properties. Sampling a search world goes through clone(), so a
                # divergence here means the model's search prior disagrees with its own encode.
                if args.clone_equivalence_every and turn % args.clone_equivalence_every == 0:
                    twin = env._belief_engine.clone()
                    # Compare the ENGINE's own state, not two player views.
                    #
                    # The first version of this check compared
                    # `engine.resolved_player_view(p)` against `twin.resolved_player_view(p)` --
                    # but `resolved_player_view` CALLS clone() itself, so both sides went through
                    # the same copy and suffered any state loss identically. It could not fail:
                    # deleting `_variant_pins`, `_hp_after_actions` or `_leftovers_healed_this_turn`
                    # from clone() each left the sweep PASSing with the full count of checks. A
                    # guard with no possible kill is not coverage (plan §3).
                    mismatches = _engine_state_mismatches(env._belief_engine, twin)
                    if mismatches:
                        violations["clone_equivalence"].append(
                            Violation(
                                {
                                    "game": game,
                                    "turn": turn,
                                    "detail": "clone() did not reproduce the parent's state",
                                    "fields": mismatches,
                                }
                            )
                        )
                    # ...and the projected views too, which is what search actually consumes.
                    for perspective in PLAYERS:
                        parent = env._belief_engine.resolved_player_view(perspective)
                        child = twin.resolved_player_view(perspective)
                        if _view_fingerprint(parent) != _view_fingerprint(child):
                            violations["clone_equivalence"].append(
                                Violation(
                                    {
                                        "game": game,
                                        "turn": turn,
                                        "perspective": perspective,
                                        "detail": "cloned belief view differs from parent",
                                    }
                                )
                            )
                        counts["clone_equivalence_checks"] += 1

                actions = {}
                for player in requested:
                    mask = env.observe(player).legal_action_mask
                    legal = [index for index, allowed in enumerate(mask) if allowed]
                    if not legal:
                        counts["truncated_games"] += 1
                        break
                    moves = [index for index in legal if index < MOVE_ACTION_COUNT]
                    if moves and rng.random() < args.move_bias:
                        actions[player] = rng.choice(moves)
                    else:
                        actions[player] = rng.choice(legal)
                if len(actions) != len(requested):
                    counts["truncated_games"] += 1
                    break
                env.step(actions)
                steps += 1

            # Per game: the engine is rebuilt on reset(), so fold its Pressure tallies before the
            # next game discards them.
            _record_pressure_reachability(env=env, counts=counts)
    finally:
        env.close()

    total_violations = sum(len(rows) for rows in violations.values())
    # Reachability gate: the sweep must have actually exercised the properties. Each of these was
    # chosen because a run that fails it would make the corresponding assertion vacuous.
    reachability = {
        "mon_observations": counts["mon_observations"],
        "distinct_species": len(species_seen),
        "narrowing_steps": counts["narrowing_steps"],
        "pinned_and_correct": counts["pinned_and_correct"],
        "stat_legality_checks": counts["stat_legality_checks"],
        "clone_equivalence_checks": counts["clone_equivalence_checks"],
        # V3: agreeing on an untouched full-PP move is nearly free, so the SPENT count is the one
        # that binds -- without it this family could report itself exercised while never testing
        # the ledger's arithmetic at all.
        "pp_comparisons": counts["pp_comparisons"],
        "pp_spent_comparisons": counts["pp_spent_comparisons"],
        # V3 Pressure: `pp_spent_comparisons` binds the ledger's arithmetic in general, but the
        # Pressure branch is a minority of charges and a sweep can exercise the ledger thoroughly
        # while never doubling once. These two make the branch's own coverage explicit -- and the
        # second one is the defect class specifically: a double charged on a move whose wire
        # target slot the engine had blanked, which is exactly what the old proxy got wrong.
        "pressure_doubled": counts["pressure_doubled"],
        "pressure_doubled_despite_blank_or_self_wire_slot": counts[
            "pressure_doubled_despite_blank_or_self_wire_slot"
        ],
    }
    reached = all(value > 0 for value in reachability.values())

    # Family 4 (pin conflicts) is NOT in the reachability set above, because with narrowing off it
    # is structurally unfailable: `_variant_pin_conflicts` is only written from
    # `narrow_candidate_variants`/`_apply_variant_pin`, both of which require a non-empty
    # `_variant_pins`, and only the tier2/investment producers write those. Counting loop
    # iterations as "reached" reported a property as exercised when it could not fire.
    # It is reported as n/a instead, and the verdict says so.
    #
    # The status is derived from OBSERVED pin activity, not from the config flag. Reporting the
    # flag said "exercised" as soon as `--item-belief-narrowing` was passed, which is a claim about
    # the invocation and not about the run: the producers can be enabled and still never write a
    # pin (nothing pinned => nothing to conflict), and the reader had no way to tell.
    narrowing_on = env_narrowing_flags["investment"] or env_narrowing_flags["item"]
    if counts["pins_observed"]:
        pin_conflict_status = f"exercised ({counts['pins_observed']} turns with pins)"
    elif narrowing_on:
        pin_conflict_status = "n/a (narrowing on, but no pin was ever written)"
    else:
        pin_conflict_status = "n/a (narrowing off; no producer writes _variant_pins)"

    # Family 5's ABILITY arm has no producer in gen3 randbats -- see ``_ABILITY_ARM_STATUS``. The
    # measurement is carried in the summary so the claim is checkable from the artifact rather than
    # taken on the docstring's word.
    ability_arm_status = (
        f"exercised ({counts['mons_with_ruled_out_abilities']} mons with ruled-out abilities)"
        if counts["mons_with_ruled_out_abilities"]
        else (
            f"{_ABILITY_ARM_STATUS}; measured: 0 of {counts['mon_observations']} observations had a"
            f" non-empty ruled_out_abilities and"
            f" {counts['intimidate_ruleout_preconditions']} met the rule's precondition"
        )
    )

    # Mons the sweep SKIPPED must fail the run, not vanish into a counter. A regression in truth
    # resolution would otherwise drop most of the population and still print PASS: mutating
    # `_true_variant_for` to fail for half the species silently skipped 1106 of 2551 observations
    # -- including the very mon whose defect motivated this harness -- and the verdict stayed
    # green. Both counters are 0 on every clean run, so requiring 0 costs nothing and binds.
    #
    # ``games_without_both_requests`` belongs here too: it was counted but not gated, so a change
    # that stopped `_first_requests` from resolving either seat's opening request would `continue`
    # past EVERY game, leave `games` at 0, and -- because the reachability counters are all also 0
    # -- report FAIL only by accident of the reachability gate rather than by naming the cause. It
    # is 0 on every clean run.
    skipped = {
        "mons_without_resolvable_true_variant": counts["mons_without_resolvable_true_variant"],
        "belief_mons_without_truth": counts["belief_mons_without_truth"],
        "truncated_games": counts["truncated_games"],
        "games_without_both_requests": counts["games_without_both_requests"],
    }
    no_silent_skips = all(value == 0 for value in skipped.values())

    verdict = "PASS" if (total_violations == 0 and reached and no_silent_skips) else "FAIL"

    summary = {
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "verdict": verdict,
        "reached": reached,
        "reachability": reachability,
        "pin_conflict_family": pin_conflict_status,
        "ruled_out_ability_arm": ability_arm_status,
        "skipped": skipped,
        "inconsistent_fallback_details": fallbacks[:50],
        "no_silent_skips": no_silent_skips,
        "args": {
            "games": args.games,
            "seed": args.seed,
            "max_steps": args.max_steps,
            "move_bias": args.move_bias,
            "clone_equivalence_every": args.clone_equivalence_every,
            "showdown_root": str(args.showdown_root),
        },
        "counts": dict(counts),
        "violation_counts": {name: len(rows) for name, rows in violations.items()},
        "violations": {name: rows[:25] for name, rows in violations.items() if rows},
        "total_violations": total_violations,
    }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--move-bias", type=float, default=0.75)
    parser.add_argument("--showdown-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--investment-belief-narrowing", action="store_true")
    parser.add_argument("--item-belief-narrowing", action="store_true")
    parser.add_argument(
        "--clone-equivalence-every",
        type=int,
        default=10,
        help="check assertion 7 on every Nth turn (it clones the engine, so it is the costly one)",
    )
    args = parser.parse_args()

    summary = run_sweep(
        showdown_root=args.showdown_root,
        games=args.games,
        seed=args.seed,
        max_steps=args.max_steps,
        move_bias=args.move_bias,
        clone_equivalence_every=args.clone_equivalence_every,
        investment_belief_narrowing=args.investment_belief_narrowing,
        item_belief_narrowing=args.item_belief_narrowing,
    )
    counts = summary["counts"]
    violations = summary.get("violations", {})
    verdict = summary["verdict"]
    reached = summary["reached"]
    total_violations = summary["total_violations"]
    species_seen = summary["reachability"]["distinct_species"]

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "belief-coherence.json").write_text(json.dumps(summary, indent=2) + "\n")

    print(
        f"[belief-coherence] {verdict} games={counts.get('games', 0)} "
        f"mon_observations={counts.get('mon_observations', 0)} species={species_seen} "
        f"containment_ok={counts.get('containment_ok', 0)} pinned_correct={counts.get('pinned_and_correct', 0)} "
        f"narrowings={counts.get('narrowing_steps', 0)} fallbacks={counts.get('inconsistent_fallbacks', 0)} "
        f"stat_checks={counts.get('stat_legality_checks', 0)} "
        f"clone_checks={counts.get('clone_equivalence_checks', 0)} "
        f"pp={counts.get('pp_comparisons', 0)}/{counts.get('pp_spent_comparisons', 0)}-spent "
        f"violations={total_violations} reached={reached} "
        f"skips={sum(summary['skipped'].values())} pin_conflicts={summary['pin_conflict_family']}"
    )
    # The PP arm's own accounting, printed rather than left in the artifact: a differential that
    # declines most of its opportunities and reports "0 violations" reads identically to one that
    # checked everything. None of these are silent skips in the `skipped` sense (no mon is dropped)
    # -- they are turns where the two sides are not the same quantity -- but they belong in the
    # verdict line's field of view.
    print(
        "[belief-coherence] V3 PP arm: "
        f"compared={counts['pp_comparisons']} spent={counts['pp_spent_comparisons']} | "
        f"not-compared: no-active-request={counts['pp_skipped_no_active_request']} "
        f"request-without-pp={counts['pp_skipped_request_without_pp']} "
        f"transformed-mon={counts['pp_skipped_transformed_mon']} "
        f"move-absent-from-request={counts['pp_skipped_move_absent_from_request']} "
        f"no-dex-max-pp={counts['pp_skipped_no_dex_max_pp']}"
    )
    print(
        "[belief-coherence] V3 Pressure branch: "
        f"charges={counts['pressure_charges']} vs-pressure={counts['pressure_vs_pressure']} "
        f"doubled={counts['pressure_doubled']} "
        f"single-because-self-targeted={counts['pressure_single_self_targeted_vs_pressure']} "
        f"doubled-on-a-blanked-wire-slot="
        f"{counts['pressure_doubled_despite_blank_or_self_wire_slot']}"
    )
    print(f"[belief-coherence] family-5 ability arm: {summary['ruled_out_ability_arm']}")
    print(
        f"[belief-coherence] family-5 item arm: "
        f"{counts.get('mons_with_ruled_out_items', 0)} mons with ruled-out items"
    )
    for name, rows in violations.items():
        if rows:
            print(f"  [{name}] {len(rows)} violation(s); first: {rows[0]}")
    return 0 if verdict == "PASS" else 1


_CLONE_EXEMPT_FIELDS = ("set_source", "format_id", "item_belief_narrowing")


def _engine_state_mismatches(parent, twin) -> list[str]:
    """Field names where ``clone()`` failed to reproduce the parent's state.

    Enumerated from ``vars()`` rather than a hand-written list, so a per-turn attribute added to
    the engine LATER without a matching line in ``clone()`` is caught by this check rather than
    silently escaping it -- the failure shape is a sampled search world that has forgotten
    evidence the live game holds.

    ``set_source`` is excluded because clone() shares it deliberately (it is large and immutable);
    identity is asserted instead. The narrowing flags are constructor arguments, not state.

    Kill-confirmed by dropping ``twin._hp_after_actions`` from ``clone()``: 20 violations naming
    that field. Stated precisely, because the bound matters: this detects a dropped copy only for
    state that is actually POPULATED in the run. ``_variant_pins`` is empty with narrowing off, so
    dropping its copy is undetectable BY VALUE here -- see the identity check below, which does
    cover it.

    VALUE equality is not enough on its own. ``repr()`` is identical for a field the clone COPIED
    and a field the clone ALIASED, so an aliasing bug -- ``twin._x = self._x`` on a mutable
    container instead of ``dict(self._x)`` -- was completely invisible to this check, in either
    direction: the two engines then share one object and every subsequent mutation of the sampled
    world writes back into the live game's belief state. That is the same class of defect as a
    dropped copy and strictly harder to see, so mutable containers additionally get an IDENTITY
    check. Only ``dict``/``list``/``set`` are checked: a correct copy always constructs a new
    container, and immutable state (ints, strings, ``None``, frozen dataclasses like
    ``_PendingTrick``) is shared on purpose and would false-positive. Unlike the value check, the
    identity check binds on EMPTY containers too, which is why it also covers ``_variant_pins``
    with narrowing off.
    """
    mismatches: list[str] = []
    parent_state = vars(parent)
    twin_state = vars(twin)
    for name in sorted(set(parent_state) | set(twin_state)):
        if name in _CLONE_EXEMPT_FIELDS:
            continue
        if name not in twin_state:
            mismatches.append(f"{name}: missing on the clone")
            continue
        parent_value = parent_state[name]
        twin_value = twin_state[name]
        if repr(parent_value) != repr(twin_value):
            mismatches.append(name)
            continue
        if isinstance(parent_value, (dict, list, set)) and parent_value is twin_value:
            mismatches.append(f"{name}: clone ALIASED the parent's container instead of copying it")
    if parent.set_source is not twin.set_source:
        mismatches.append("set_source: clone must SHARE the source, not copy it")
    return mismatches


def _view_fingerprint(view) -> tuple:
    """The properties clone equivalence is checked on, per plan item 7."""
    return tuple(
        (
            belief.species,
            tuple(belief.revealed_moves),
            belief.candidate_set_count,
            round(float(belief.uncertainty), 12),
            tuple(str(v.get("variant_id")) for v in belief.candidate_variants or ()),
            tuple(belief.ruled_out_abilities or ()),
            tuple(belief.ruled_out_items or ()),
            tuple(belief.move_uses or ()),
        )
        for belief in view.opponent_pokemon
    )


if __name__ == "__main__":
    raise SystemExit(main())
