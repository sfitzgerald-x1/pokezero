"""Selected INITIAL roster fields, not current-state fidelity or provenance.

Callers must independently bind the consumed public completion and original
generator teams. Never feed original teams to the public selector. Team and
move order are ignored here; duplicate species are explicitly ambiguous.
Unpaired field comparisons keep the full truth-roster denominator as bounds.
"""
from collections import Counter
import hashlib

from ..dex import normalize_id
from ..engine_world import unpack_team
from ..randbat import canonical_gen3_randbat_species_id
from ..tier2 import canonical_move_id


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _team(packed, size):
    _require(type(packed) is str and packed, "initial packed team required")
    team = unpack_team(packed)
    _require(len(team) == size, "fixed initial roster size required")
    result = []
    for mon in team:
        species = canonical_gen3_randbat_species_id(mon.species)
        moves = sorted(canonical_move_id(move) for move in mon.moves)
        _require(species and 1 <= len(moves) <= 4 and all(moves)
                 and len(set(moves)) == len(moves), "unambiguous initial move set required")
        ability = normalize_id(mon.ability)
        # Empty/slot-coded abilities require a species ability resolver, not
        # string equality. Two unresolved values are not evidence of agreement.
        if ability in {"", "0", "1", "h", "s"}:
            ability = None
        result.append(dict(species=species, moves=moves, ability=ability,
                           item=normalize_id(mon.item)))
    return result


def _agreement(values):
    total = len(values)
    matches = sum(value is True for value in values)
    unknown = sum(value is None for value in values)
    return dict(denominator=total, matches=matches, mismatches=total - matches - unknown,
                unknown=unknown, agreement_bounds=[matches / total, (matches + unknown) / total])


def _moves_equal(actual, predicted):
    base = lambda moves: sorted("hiddenpower" if m.startswith("hiddenpower") else m for m in moves)
    if base(actual) != base(predicted):
        return False
    if "hiddenpower" in actual or "hiddenpower" in predicted:
        return None  # Its type needs IV-derived information outside this field comparison.
    return actual == predicted


def compare_initial_teams(sampled_packed, truth_packed, *, team_size=6):
    """Compare species coverage and selected set fields without greedy pairing.

    The truth roster supplies every denominator slot. Missing sampled species
    make the joint selected-field match false, but do not invent a corresponding
    member for separate move/item/ability comparisons. Duplicate normalized
    species leave those pairings unknown. These are partial-identification
    bounds, not complete-case percentages or calibrated confidence intervals.
    """
    _require(type(team_size) is int and 1 <= team_size <= 6, "fixed team size required")
    sampled, truth = _team(sampled_packed, team_size), _team(truth_packed, team_size)
    sc, tc = (Counter(row["species"] for row in team) for team in (sampled, truth))
    sampled_by_species = {row["species"]: (i, row) for i, row in enumerate(sampled)}
    rows = []
    for index, actual in enumerate(truth):
        species = actual["species"]
        fields = dict(moves=None, ability=None, item=None)
        if sc[species] == 0:
            status, sampled_index, joint = "MISSING_SPECIES", None, False
        elif tc[species] != 1 or sc[species] > 1:
            status, sampled_index, joint = "AMBIGUOUS_SPECIES_IDENTITY", None, None
        else:
            status = "UNIQUELY_PAIRED"
            sampled_index, predicted = sampled_by_species[species]
            fields = {name: None if name == "ability" and (
                actual[name] is None or predicted[name] is None) else actual[name] == predicted[name]
                for name in fields}
            fields["moves"] = _moves_equal(actual["moves"], predicted["moves"])
            joint = False if any(v is False for v in fields.values()) else (
                None if any(v is None for v in fields.values()) else True)
        rows.append(dict(truth_index=index, species=species, sampled_index=sampled_index,
                         identity_status=status, fields=fields, joint_selected_fields=joint))
    covered = sum((sc & tc).values())
    return dict(schema="pokezero.search-over-raw.initial-team-comparison.v1", team_size=team_size,
        sampled_packed_sha256=hashlib.sha256(sampled_packed.encode()).hexdigest(),
        truth_packed_sha256=hashlib.sha256(truth_packed.encode()).hexdigest(),
        species=dict(matched_multiset_members=covered, denominator=team_size,
                     coverage=covered / team_size,
                     missing=dict(tc - sc), extra=dict(sc - tc)),
        fields={name: _agreement([r["fields"][name] for r in rows])
                for name in ("moves", "ability", "item")},
        joint_selected_fields=_agreement([r["joint_selected_fields"] for r in rows]), rows=rows,
        field_scope=["species", "initial_move_set", "initial_ability", "initial_item"],
        excluded_fields=["level", "nature", "evs", "ivs", "gender", "current_battle_state"],
        move_slot_order_compared=False, current_materialization_compared=False,
        consumed_world_validated=False, source_truth_validated=False,
        calibration_certified=False, scientific_strength_claim=False)
