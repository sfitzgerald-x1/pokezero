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
   ``ruled_out_items`` never the true item (the ``TRAPPER_ALIVE`` correctness check rides this
   sweep: Shadow Tag and Arena Trap are both pool-reachable);
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
        if looks_like_fallback:
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

    # (5) ruled-out soundness -- never rule out the truth. This is also the TRAPPER_ALIVE
    # correctness check: that column is derived from ruled_out_abilities.
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
            "stat_legality_checks": 0,
            "unevaluable_candidate_spreads": 0,
            "pin_conflict_checks": 0,
            "clone_equivalence_checks": 0,
            "mons_with_ruled_out_abilities": 0,
            "mons_with_ruled_out_items": 0,
            "ambiguous_true_variant_matches": 0,
            "mons_without_resolvable_true_variant": 0,
            "belief_mons_without_truth": 0,
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
                    # deleting `_variant_pins`, `_hp_after_actions` or `_healed_to_full_this_turn`
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
    }
    reached = all(value > 0 for value in reachability.values())

    # Family 4 (pin conflicts) is NOT in the reachability set above, because with narrowing off it
    # is structurally unfailable: `_variant_pin_conflicts` is only written from
    # `narrow_candidate_variants`/`_apply_variant_pin`, both of which require a non-empty
    # `_variant_pins`, and only the tier2/investment producers write those. Counting loop
    # iterations as "reached" reported a property as exercised when it could not fire.
    # It is reported as n/a instead, and the verdict says so.
    narrowing_on = env_narrowing_flags["investment"] or env_narrowing_flags["item"]
    pin_conflict_status = "exercised" if narrowing_on else "n/a (narrowing off)"

    # Mons the sweep SKIPPED must fail the run, not vanish into a counter. A regression in truth
    # resolution would otherwise drop most of the population and still print PASS: mutating
    # `_true_variant_for` to fail for half the species silently skipped 1106 of 2551 observations
    # -- including the very mon whose defect motivated this harness -- and the verdict stayed
    # green. Both counters are 0 on every clean run, so requiring 0 costs nothing and binds.
    skipped = {
        "mons_without_resolvable_true_variant": counts["mons_without_resolvable_true_variant"],
        "belief_mons_without_truth": counts["belief_mons_without_truth"],
        "truncated_games": counts["truncated_games"],
    }
    no_silent_skips = all(value == 0 for value in skipped.values())

    verdict = "PASS" if (total_violations == 0 and reached and no_silent_skips) else "FAIL"

    summary = {
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "verdict": verdict,
        "reached": reached,
        "reachability": reachability,
        "pin_conflict_family": pin_conflict_status,
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
        f"violations={total_violations} reached={reached} "
        f"skips={sum(summary['skipped'].values())} pin_conflicts={summary['pin_conflict_family']}"
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
    dropping its copy is undetectable here -- not a hole in the check, but a reason the
    narrowing-on rerun is where that particular copy gets its coverage.
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
        if repr(parent_state[name]) != repr(twin_state[name]):
            mismatches.append(name)
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
