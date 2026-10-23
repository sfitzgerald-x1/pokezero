"""Closed bridge diagnostics: source positions and declared public-effect labels."""

# Public parser vocabulary plus its separately encoded recharge flag. Never
# copy arbitrary simulator messages, IDs, names or an unrecognized input token.
PUBLIC_VOLATILE_REASON_TOKENS = frozenset({
    "confusion", "leechseed", "substitute", "taunt", "encore", "disable", "torment", "attract",
    "nightmare", "curse", "ingrain", "foresight", "lockon", "mindreader", "destinybond", "grudge",
    "focusenergy", "charge", "yawn", "stockpile", "bide", "uproar", "imprison", "magiccoat",
    "snatch", "mudsport", "watersport", "defensecurl", "minimize", "rage", "partiallytrapped",
    "perishsong", "perish0", "perish1", "perish2", "perish3", "flashfire", "solarbeam",
    "mustrecharge", "unrecognized",
})


def safe_bridge_error_reason(value):
    """Accept only the one authored reason and a closed public-effect token."""
    if (type(value) is not dict or set(value) != {"code", "volatile"}
            or type(value["code"]) is not str or value["code"] != "UNSUPPORTED_PUBLIC_VOLATILE"
            or type(value["volatile"]) is not str
            or value["volatile"] not in PUBLIC_VOLATILE_REASON_TOKENS):
        return None
    return dict(value)

BRIDGE_SOURCE_FILES = frozenset({
    "scripts/battle_bridge.mjs",
    *("showdown/dist/" + name for name in (
        "sim/battle.js", "sim/battle-actions.js", "sim/battle-queue.js",
        "sim/battle-stream.js", "sim/field.js", "sim/pokemon.js", "sim/side.js",
        "sim/state.js", "sim/prng.js", "sim/dex.js", "sim/dex-conditions.js",
        "sim/dex-moves.js", "lib/streams.js", "data/moves.js", "data/abilities.js",
        "data/conditions.js", "data/items.js", "data/mods/gen3/moves.js",
        "data/mods/gen3/abilities.js", "data/mods/gen3/conditions.js",
        "data/mods/gen3/items.js", "data/mods/gen3/scripts.js",
    )),
})


def safe_bridge_error_origin(value):
    """The emitting catch boundary, not an inferred operation or error cause."""
    return value if type(value) is str and value in {"command", "stream"} else None


def safe_bridge_error_sites(value):
    """Reject malformed metadata wholesale, copying only allowlisted locations."""
    if type(value) not in (list, tuple) or not 1 <= len(value) <= 8:
        return []
    result = []
    for row in value:
        if (type(row) is not dict or set(row) != {"file", "line", "column"}
                or type(row["file"]) is not str or row["file"] not in BRIDGE_SOURCE_FILES
                or any(type(row[key]) is not int or not 0 < row[key] <= 10**7
                       for key in ("line", "column"))):
            return []
        result.append(dict(row))
    return result
