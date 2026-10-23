"""Closed diagnostic schema for bridge source positions, never error content."""


def safe_bridge_error_sites(value):
    """Reject malformed metadata wholesale, copying only allowlisted locations."""
    if type(value) not in (list, tuple) or not 1 <= len(value) <= 8:
        return []
    result = []
    for row in value:
        if (type(row) is not dict or set(row) != {"file", "line", "column"}
                or type(row["file"]) is not str or row["file"] != "scripts/battle_bridge.mjs"
                or any(type(row[key]) is not int or not 0 < row[key] <= 10**7
                       for key in ("line", "column"))):
            return []
        result.append(dict(row))
    return result
