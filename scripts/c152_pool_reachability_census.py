#!/usr/bin/env python
"""The gen3 randbats pool census behind C152's reachability verdicts.

WHY THIS EXISTS. `tests/test_branch_mass_reconstruction.py` discharges one of
c137 section 4's owed families -- the order-10.3 Rain Dish mirror step -- with a
reachability VERDICT rather than a fixture. A waiver is only as good as its
measurement, and review found the measurement living in prose: the docstring said
"machine-checked" while the only assertion checked that *some* verdict existed for
the family, not that Rain Dish is absent from the pool. Nothing re-derived 71/0.

WHAT THIS FIXES, AND WHAT IT DOES NOT. This script re-derives the census from the
Showdown checkout and commits it as `tests/data/c152_pool_reachability_census.json`;
the gate then asserts `POOL_UNREACHABLE`'s figures against that artifact. So the
numbers are a committed measurement with a named regeneration command instead of
prose, and they cannot drift silently.

They can still go STALE, and the gate says so rather than pretending otherwise:
CI builds no Showdown checkout (`tests/_showdown_root.py`, and the mass-gate step
forbids skips outright), so nothing in CI re-derives these counts against a live
pool. A Showdown bump that added Rain Dish to a gen3 set would leave the artifact,
the gate and the waiver all green and all wrong. That is why the artifact records
the commit it was taken at: the staleness is bounded and nameable, not invisible.
Regenerating it is a deliberate act that shows up in review.

Regenerate with::

    python scripts/c152_pool_reachability_census.py --write \\
        tests/data/c152_pool_reachability_census.json

from a machine with a pokemon-showdown checkout resolvable by
``pokezero.local_showdown.default_showdown_root`` (``POKEZERO_SHOWDOWN_ROOT``
wins). The script refuses to write without one, and refuses to write if the
checkout has no built ``dist/sim/dex`` -- move classification needs the real dex,
because `multihit` / `damage` / `ohko` / `volatileStatus` are dex fields and
reimplementing them here would make this a second copy of Showdown rather than a
measurement of it.

INSTRUMENTS, from `reports/c138_known_gaps_ledger.md` section 1.2, which is where
the choice of instrument is adjudicated and not re-litigated here:

  * is a MOVE reachable?    union of every set's `movepool` in
                            `data/random-battles/gen3/sets.json`
  * is an ABILITY reachable? union of every set's `abilities` in the same file
  * does it exist in gen3?  `Dex.mod('gen3')`

Items are deliberately NOT measured here: c138 section 1.2 records that `sets.json`
is the wrong instrument for them (a gen3 set has no item field at all), and no
verdict in C152 rests on an item.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pokezero.local_showdown import default_showdown_root  # noqa: E402

#: Move ids whose reachability C152 section 2 reports, beyond the class counts.
NAMED_MOVES = ("leechseed", "wish", "wrap", "seismictoss", "bonemerang")

#: Abilities the order-10.3 branch of ``residual_phase_final_hp`` can fire on.
#: DRYSKIN is included because the mirror's own comment names it as the other
#: HP-changing ability at 10.3, so a waiver that only measured Rain Dish would be
#: narrower than the step it waives. TRACE is included because it is the only
#: pool mechanism that could import an ability a set does not list -- and it
#: copies the OPPONENT's, so it cannot manufacture one absent from the whole pool.
NAMED_ABILITIES = ("Rain Dish", "Dry Skin", "Trace")

_NODE = r"""
const {Dex} = require(process.argv[1] + '/dist/sim/dex');
const fs = require('fs');
const sets = JSON.parse(fs.readFileSync(
    process.argv[1] + '/data/random-battles/gen3/sets.json', 'utf8'));
const dex = Dex.mod('gen3');
const moves = new Map();
const abilities = new Map();
let nsets = 0;
for (const species of Object.keys(sets)) {
  for (const set of sets[species].sets) {
    nsets++;
    for (const m of (set.movepool || [])) {
      if (!moves.has(m)) moves.set(m, new Set());
      moves.get(m).add(species);
    }
    for (const a of (set.abilities || [])) {
      if (!abilities.has(a)) abilities.set(a, new Set());
      abilities.get(a).add(species);
    }
  }
}
const classes = {multihit: [], fixed_damage: [], ohko: [], partiallytrapped: []};
for (const id of [...moves.keys()].sort()) {
  const mv = dex.moves.get(id);
  if (mv.multihit) classes.multihit.push(id);
  if (mv.damage) classes.fixed_damage.push(id);
  if (mv.ohko) classes.ohko.push(id);
  if (mv.volatileStatus === 'partiallytrapped') classes.partiallytrapped.push(id);
}
const named_moves = {};
for (const id of JSON.parse(process.argv[2])) {
  named_moves[id] = moves.has(id) ? moves.get(id).size : 0;
}
const named_abilities = {};
for (const name of JSON.parse(process.argv[3])) {
  named_abilities[name] = abilities.has(name) ? abilities.get(name).size : 0;
}
process.stdout.write(JSON.stringify({
  species: Object.keys(sets).length,
  sets: nsets,
  distinct_moves: moves.size,
  distinct_abilities: abilities.size,
  move_classes: classes,
  named_moves,
  named_abilities,
  raindish_exists_in_gen3: !!dex.abilities.get('raindish').exists,
  raindish_num: dex.abilities.get('raindish').num,
}));
"""


def _showdown_commit(root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    return result.stdout.strip()


def census(root: Path) -> dict:
    """Run the census against ``root``. Raises if the checkout cannot answer."""

    if not (root / "data" / "random-battles" / "gen3" / "sets.json").is_file():
        raise SystemExit(
            f"ERROR: no gen3 randbats sets.json under {root}. Set "
            "POKEZERO_SHOWDOWN_ROOT to a pokemon-showdown checkout."
        )
    if not (root / "dist" / "sim" / "dex.js").is_file():
        raise SystemExit(
            f"ERROR: {root} has no built dist/sim/dex.js. Move classification "
            "needs the real dex; run the checkout's build first."
        )
    result = subprocess.run(
        ["node", "-e", _NODE, "--",
         str(root), json.dumps(list(NAMED_MOVES)), json.dumps(list(NAMED_ABILITIES))],
        capture_output=True, text=True,
    )
    if result.returncode:
        raise SystemExit(f"ERROR: census failed\n{result.stdout}\n{result.stderr}")
    payload = json.loads(result.stdout)
    payload["showdown_commit"] = _showdown_commit(root)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", type=Path, default=None)
    args = parser.parse_args()

    payload = {
        "_README": (
            "Pool-reachability census behind C152's Rain Dish waiver and section 2's "
            "table. Instruments are c138 section 1.2's. Regenerate with "
            "scripts/c152_pool_reachability_census.py --write, from a checkout with "
            "a built pokemon-showdown; CI builds none, so nothing re-derives this "
            "against a live pool. NEVER edit it by hand to make a test pass: the "
            "test compares POOL_UNREACHABLE against it, so a hand edit silently "
            "moves the waiver."
        ),
        **census(default_showdown_root()),
    }
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.write is None:
        print(text, end="")
    else:
        args.write.parent.mkdir(parents=True, exist_ok=True)
        args.write.write_text(text, encoding="utf-8")
        print(f"wrote {args.write}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
