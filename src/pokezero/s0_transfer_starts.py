"""Shared physical constructor identities for the engine and native Showdown."""

import numpy as np

from . import v5_python_backend as original, v5_run
from .dex import normalize_id
from .s0_transfer_roster import digest


BIRTH_FIELDS = ("species", "level", "item", "ability", "moves", "pp", "stats", "happiness", "gender")
STATS = ("hp", "atk", "def", "spa", "spd", "spe")


def prepare_starts(prepared, clusters, *, gender_seed, gender_domain):
    slots = np.random.Generator(np.random.PCG64(gender_seed)).choice(
        len(prepared.arrays["species"]), size=(len(clusters), 2), replace=False,
    )
    games = tuple(v5_run.Game(c, f"transfer-{c}", c, tuple(map(int, ix)))
                  for c, ix in zip(clusters, slots, strict=True))
    batch = v5_run.Batch(games, 0, 0, gender_seed, gender_domain, tuple(clusters))
    gender = original._genders(prepared, batch)
    result = {}
    for game, ix, actual_gender in zip(games, slots, gender, strict=True):
        start = original.packed_start(prepared, game, actual_gender)
        engine = {name: prepared.arrays[name][ix].tolist()
                  for name in BIRTH_FIELDS if name not in ("happiness", "gender")}
        engine["gender"] = actual_gender.tolist()
        engine["happiness"] = (prepared.arrays["happiness"][ix].tolist() if "happiness" in prepared.arrays
                               else [[255] * 6, [255] * 6])
        indices = ix.tolist()
        result[str(game.stream_id)] = dict(
            team_indices=indices, packed=start.player_teams, gender=engine["gender"], engine_birth=engine,
            showdown_birth=_expected_showdown(prepared, indices, engine),
        )
    return result


def _expected_showdown(prepared, indices, birth):
    vocabulary = prepared.metadata["vocabulary"]
    result = []
    for side, index in enumerate(indices):
        team = []
        for mon in range(6):
            def label(field, axis):
                number = birth[field][side][mon]
                return normalize_id(vocabulary[axis][number]) if number else ""

            moves = [normalize_id(vocabulary["moves"][n]) for n in birth["moves"][side][mon] if n]
            team.append(dict(
                species=label("species", "species"), item=label("item", "items"), ability=label("ability", "abilities"),
                level=birth["level"][side][mon], moves=moves,
                move_slots=[["hiddenpower" if m.startswith("hiddenpower") else m, p]
                            for m, p in zip(moves, birth["pp"][side][mon])],
                stats=birth["stats"][side][mon], happiness=birth["happiness"][side][mon],
                gender=birth["gender"][side][mon],
                ivs=prepared.arrays["ivs"][index, mon].tolist(), evs=prepared.arrays["evs"][index, mon].tolist(),
                nature=normalize_id(vocabulary["natures"][int(prepared.arrays["nature"][index, mon])]),
            ))
        result.append(team)
    return result


def showdown_birth(snapshot, expected):
    sides = snapshot["battle"]["sides"]
    actual = []
    for side in sides:
        team = []
        for p in side["pokemon"]:
            set_ = p["set"]
            team.append(dict(
                species=normalize_id(set_["species"]), item=normalize_id(set_["item"]),
                ability=normalize_id(set_["ability"]), level=set_["level"],
                moves=[normalize_id(m) for m in set_["moves"]],
                move_slots=[[m["id"], m["pp"]] for m in p["moveSlots"]],
                stats=[p["maxhp"], *(p["storedStats"][key] for key in STATS[1:])],
                happiness=set_.get("happiness", 255), gender={"": 0, "N": 0, "M": 1, "F": 2}[p["gender"]],
                ivs=[set_["ivs"][key] for key in STATS], evs=[set_["evs"][key] for key in STATS],
                nature=normalize_id(set_.get("nature") or ""),
            ))
        actual.append(team)
    if actual != expected:
        raise ValueError(f"actual Showdown birth differs from declared start ({digest(actual)})")
    return actual
