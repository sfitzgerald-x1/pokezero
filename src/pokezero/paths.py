"""Path helpers safe to import from anywhere: stdlib only, no pokezero imports.

A leaf on purpose. ``portable_path`` is needed by both ``local_showdown`` and ``randbat``, and
``local_showdown`` already imports ``randbat`` -- so defining it in either would be a cycle.
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["portable_path"]


def portable_path(path: "Path | str") -> str:
    """A path safe to record in a TRACKED artifact: the user's home rewritten to ``~``.

    Provenance blocks that record absolute module or checkout paths are how a username gets into
    a public repo without anyone deciding to put it there -- a tool writes the value, it is
    committed as an audit artifact or a golden-corpus row, and nobody reads it again. Twelve
    tracked artifacts had to be scrubbed on 2026-08-03 for exactly this, and a thirteenth could
    not be, because its rows are hash-sealed.

    Also collapses a home directory FLATTENED into a single path segment, e.g. a scratch
    directory named ``-Users-someone-workspace-...``. That form carries the username just as
    plainly but survives any rule that looks for a real ``/Users/<name>/`` prefix, and it is the
    shape that was still sitting in two tracked files after the first scrub.

    Returns the path unchanged when it is neither under home nor carries a flattened home: a
    system path like ``/usr/lib`` identifies nobody and is more useful recorded in full.
    """
    resolved = Path(path).resolve()
    text = str(resolved)
    try:
        return "~/" + str(resolved.relative_to(Path.home()))
    except ValueError:
        pass
    home_parts = Path.home().parts
    if len(home_parts) >= 3:
        # "/Users/name" -> "-Users-name", the shape a temp-dir namer produces.
        flattened = "-".join(part for part in home_parts if part != "/")
        if flattened and flattened in text:
            return text.replace(flattened, "<home>")
    return text
