"""Scope gate for C116 Phase 2 — enumeration is a flag-gated ORACLE, not a shipped path.

The engine ships two roll paths from ONE build, selected by the runtime env flag
``POKEZERO_ENUMERATE_ROLLS`` (a ``OnceLock`` read on the first call into the engine):

* **collapsed** — the partition cascade that has always shipped. Each chance branch
  picks a representative roll and, where a threshold straddles the fan, splits it into
  a small number of arms. This is what search runs, what training runs, and what the
  transition differential runs. It is the shipping configuration, and every process in
  this repository must take it unless something explicitly asks for the other one.
* **enumerated** — one arm per distinct ``floor(max * r / 100)`` for ``r`` in 85..=100
  at mass 1/16, pre-merged on equal integers, with ``residual_lethality_threshold``
  not consulted at all. Exact where the collapsed path is approximate, and therefore
  usable as a REFERENCE ORACLE for the collapsed path's masses — by tests, on request,
  per process. Nothing that ships consumes it.

WHY THIS GATE IS A RUNTIME GATE, and why its previous form was not a gate at all.

The first version of this module scanned tracked files for the literal string
``POKEZERO_ENUMERATE_ROLLS`` and required the set of files mentioning it to equal an
allowlist. Review defeated that in one line: a file containing

    import engine_transition_differential

mentions no flag, appears in no allowlist, and — while that module had an
``os.environ.setdefault`` at module scope — enabled enumeration for its whole process
AND every child it spawned, with the gate still green. That is not a hypothetical; the
leak was demonstrated end to end into a spawned SEARCH child, because ``unittest``
imports every selected module before running any test, and
``tests/test_engine_search_no_panic.py`` runs ``python -m pokezero.engine_search``
through ``subproc_env()``.

So the load-bearing assertions here are BEHAVIOURAL. Each one spawns a child process,
imports a surface into it, and requires the engine's fan to come back COLLAPSED, pinned
by value at ``[112, 160]``. A textual scan is kept below as a change ledger, but it is
explicitly not what holds the line: the runtime probes would fail on the import-graph
leak the scan could not see.

EVERY behavioural check here runs in a CHILD process, with ``POKEZERO_ENUMERATE_ROLLS``
REMOVED from the child's environment first. Removed, deliberately: the subject is
"does importing this surface turn enumeration on", so the operator's ambient shell must
not be able to make the answer either yes or no. The flag is also read once per process
through a ``OnceLock``, so an in-process toggle would silently measure whichever path
happened to be latched first.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _subproc_env import subproc_env  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

ENV_FLAG = "POKEZERO_ENUMERATE_ROLLS"

# The collapsed fan, by VALUE. 112 is the collapsed non-crit representative -- the whole
# non-crit fan is 103..122, every roll of which the defender survives, so nothing
# straddles and the cascade emits one arm. 160 is the crit fan, also collapsed: min_crit
# 207 already exceeds 160 hp so every crit roll kills and the damage clamps at the HP
# there is. Two arms where enumeration produces twenty-six.
#
# Pinned by value, not by "fewer than enumeration": a cascade that started emitting some
# OTHER pair of representatives would still be fewer, and would still be wrong.
COLLAPSED_FAN = [112, 160]

# Surfaces that must never take the enumerated path. The Python entries are imported in
# a child and MEASURED; the Rust entry cannot be imported, so it keeps a textual check.
#
# scripts/bench_multiply_search.py is here because it is the bench that produces the
# throughput table this decision cites -- a bench that silently measured the enumerated
# path would report a number for a configuration nobody runs. Review found it missing.
_SEARCH_SURFACES = (
    "src/pokezero/engine_search.py",
    "src/pokezero/env.py",
    "src/pokezero/selfplay.py",
    "scripts/bench_crate_search.py",
    "scripts/bench_multiply_search.py",
)
_UNIMPORTABLE_SEARCH_SURFACES = ("rust/pokezero-search/src/lib.rs",)

# The differential. It USED to enable the flag at module scope, which made every one of
# its importers an enabler.
_DIFFERENTIAL = "scripts/engine_transition_differential.py"

# Floor on the size of the import-graph probe set, so the sweep cannot silently shrink
# to nothing and read green. Measured at 26 tracked modules referencing the differential
# when this gate was written; 21 of those actually load it (13 by ``import`` statement,
# 8 through ``importlib.util.spec_from_file_location``). The probe set is the wider
# reference-based one deliberately -- importing more than strictly necessary can only
# make the check stricter, and a module that grows an import tomorrow is already covered.
_MIN_DIFFERENTIAL_CONSUMERS = 24

# Tracked modules that are known not to import cleanly in isolation for reasons
# unrelated to this flag (heavy optional dependencies, ``__main__``-only wiring).
# Empty today; an entry here is a scope reduction and needs a reason.
_IMPORT_PROBE_EXCLUSIONS: frozenset[str] = frozenset()


# ---------------------------------------------------------------------------------------------
# Child-process probes. Run as ``python tests/test_roll_enumeration_scope.py --probe NAME``.
# ---------------------------------------------------------------------------------------------

_ACCURACY_ROCKSLIDE = 0.9
_ACCURACY_FIREBLAST = 0.85
_BURN_CHANCE = 0.10
_CRIT_RATE = 1.0 / 16.0


def _import_targets(relatives: list[str]) -> list[str]:
    """Import each tracked path as a module, the way a test runner would.

    ``sys.path`` gets ``src``, ``scripts`` and ``tests`` the same way the importers
    themselves do, so a module that does ``from engine_transition_differential import
    ...`` resolves exactly as it does in a real run. Import order is the caller's.
    """

    import importlib

    for extra in ("src", "scripts", "tests", ""):
        candidate = str(ROOT / extra) if extra else str(ROOT)
        if candidate not in sys.path:
            sys.path.insert(0, candidate)
    imported: list[str] = []
    for relative in relatives:
        parts = Path(relative).with_suffix("").parts
        if parts[0] == "src":
            # A package module: import it by its DOTTED name, the way production
            # does. Importing it by file location would execute it outside its
            # package and die on the first relative import -- and a module that
            # cannot even load is not a measurement of what it does when it loads.
            parts = parts[1:]
        else:
            # scripts/ and tests/ are flat sys.path roots for their own modules.
            parts = parts[-1:]
        if parts[-1] == "__init__":
            parts = parts[:-1]
        importlib.import_module(".".join(parts))
        imported.append(relative)
    return imported


def _fan_probe(imports: list[str] | None = None) -> dict[str, object]:
    """Distinct move-damage values the engine emits for one Rock Slide fan.

    Rock Slide into a defender at 160/244 with no residual: the non-crit fan can
    never kill and the crit fan straddles, so the COLLAPSED cascade has exactly one
    threshold to act on. Damage VALUES rather than branch counts, because Rock Slide
    flinches 30% and doubles every arm.

    ``imports`` names tracked files to import BEFORE the engine is touched. That is the
    runtime scope check: if importing them writes ``POKEZERO_ENUMERATE_ROLLS``, or
    triggers anything that does, the fan comes back enumerated and the caller sees it.

    Also returns an independent reconstruction of what enumeration must produce:
    ``floor(max * r / 100)`` for r in 85..=100, crit and non-crit, clamped at the
    defender's HP because a roll cannot take HP that is not there. Nothing here calls
    an engine helper for the enumeration; ``calculate_damage`` supplies two scalars.
    """

    imported = _import_targets(imports or [])

    import poke_engine as pe

    hp, maxhp = 160, 244

    def dummy():
        return pe.Pokemon(id="pikachu", level=1, hp=0)

    attacker = pe.Pokemon(
        id="gligar", level=81,
        types=("ground", "flying"), base_types=("ground", "flying"),
        hp=205, maxhp=205, ability="none", item="none",
        attack=170, defense=160, special_attack=120,
        special_defense=130, speed=250,
        moves=[pe.Move(id="rockslide", pp=16)],
    )
    defender = pe.Pokemon(
        id="fearow", level=81,
        types=("normal", "flying"), base_types=("normal", "flying"),
        hp=hp, maxhp=maxhp, ability="none", item="none",
        attack=170, defense=145, special_attack=110,
        special_defense=125, speed=100, status="none",
        moves=[pe.Move(id="splash", pp=16)],
    )
    state = pe.State(
        side_one=pe.Side(active_index="0", pokemon=[attacker] + [dummy()] * 5),
        side_two=pe.Side(active_index="0", pokemon=[defender] + [dummy()] * 5),
        weather="none", terrain="none", trick_room=False,
    )

    max_regular = pe.calculate_damage(state, "rockslide", "splash", False)[0][0]
    max_crit = pe.calculate_damage(state, "rockslide", "splash", True)[0][1]
    reconstruction = sorted(
        {min(raw * r // 100, hp) for raw in (max_regular, max_crit) for r in range(85, 101)}
    )

    branches = pe.generate_instructions(state, "rockslide", "splash")
    emitted: set[int] = set()
    total = 0.0
    for branch in branches:
        total += branch.percentage
        for instruction in branch.instruction_list:
            text = str(instruction)
            if text.startswith("Damage SideTwo"):
                emitted.add(int(text.split(": ")[1]))
                break

    return {
        "flag": os.environ.get(ENV_FLAG),
        "imported": imported,
        "emitted_damages": sorted(emitted),
        "enumeration_reconstruction": reconstruction,
        "mass_total": total,
    }


def _secondary_composition_probe() -> dict[str, object]:
    """C119's objection, answered by measurement: ``count/16`` DOES express a secondary.

    C119 held that a count-over-sixteen mass cannot represent a probabilistic
    secondary. It does not have to. Enumeration replaces the roll collapse *inside*
    each chance branch, and ``run_move`` then fans every one of those arms through
    ``get_instructions_from_secondaries``, so an arm's mass comes out as
    ``count/16 x branch probability`` by construction — with no recipe to hand-derive
    and nothing for this test to trust.

    Fire Blast is the witness: accuracy 85, a 10% burn, a 1/16 crit, into a Normal
    type (a Fire type could not be burned, which would delete the secondary and make
    the check vacuous) at FULL HP so no roll kills and no lethality merge can rescue
    a wrong mass. The reconstruction below shares no arithmetic with the engine's
    branch construction: two scalars from ``calculate_damage``, then pure Python.
    """

    import poke_engine as pe
    from collections import defaultdict

    hp = maxhp = 404

    def dummy():
        return pe.Pokemon(id="pikachu", level=1, hp=0)

    attacker = pe.Pokemon(
        id="charizard", level=81,
        types=("fire", "flying"), base_types=("fire", "flying"),
        hp=220, maxhp=220, ability="none", item="none",
        attack=140, defense=140, special_attack=190,
        special_defense=150, speed=250,
        moves=[pe.Move(id="fireblast", pp=8)],
    )
    defender = pe.Pokemon(
        id="fearow", level=81,
        types=("normal", "typeless"), base_types=("normal", "typeless"),
        hp=hp, maxhp=maxhp, ability="none", item="none",
        attack=150, defense=150, special_attack=110,
        special_defense=150, speed=100, status="none",
        moves=[pe.Move(id="splash", pp=16)],
    )
    state = pe.State(
        side_one=pe.Side(active_index="0", pokemon=[attacker] + [dummy()] * 5),
        side_two=pe.Side(active_index="0", pokemon=[defender] + [dummy()] * 5),
        weather="none", terrain="none", trick_room=False,
    )

    max_regular = pe.calculate_damage(state, "fireblast", "splash", False)[0][0]
    max_crit = pe.calculate_damage(state, "fireblast", "splash", True)[0][1]

    groups: dict[int, float] = defaultdict(float)
    for raw, chance in ((max_regular, 1.0 - _CRIT_RATE), (max_crit, _CRIT_RATE)):
        for r in range(85, 101):
            groups[min(raw * r // 100, hp)] += chance / 16.0
    expected: dict[str, float] = {}
    for damage, mass in groups.items():
        expected[f"{damage}|True"] = _ACCURACY_FIREBLAST * mass * _BURN_CHANCE * 100.0
        expected[f"{damage}|False"] = _ACCURACY_FIREBLAST * mass * (1 - _BURN_CHANCE) * 100.0

    actual: dict[str, float] = defaultdict(float)
    miss_mass = 0.0
    total = 0.0
    for branch in pe.generate_instructions(state, "fireblast", "splash"):
        total += branch.percentage
        texts = [str(i) for i in branch.instruction_list]
        damage = 0
        for text in texts:
            if text.startswith("Damage SideTwo"):
                damage = int(text.split(": ")[1])
                break
        if damage == 0:
            miss_mass += branch.percentage
            continue
        burned = any("-> BURN" in t and "SideTwo" in t for t in texts)
        actual[f"{damage}|{burned}"] += branch.percentage

    return {
        "flag": os.environ.get(ENV_FLAG),
        "expected": expected,
        "actual": dict(actual),
        "miss_mass": miss_mass,
        "mass_total": total,
    }


def _multihit_probe() -> dict[str, object]:
    """The multi-hit SEMANTIC CHANGE, measured instead of assumed.

    The collapsed cascade forms a TOTAL across the hits and converts it back to a
    per-hit amount. Enumeration never forms the total: it rolls once, per hit, and
    ``run_move`` applies that same per-hit amount to every hit. gen3 rolls a multi-hit
    move's damage once and reuses it, so that is the correct semantics — but "correct"
    was an argument in a comment and nothing pinned it, which is precisely the hole
    that would let a wrong oracle bless a wrong recipe.

    Bonemerang: two hits, fixed count, so the branch structure is not confounded by a
    hit-count distribution. Defender at FULL HP and out of reach of both hits, so no
    clamp and no lethality merge fold two distinct rolls into one arm. What is asserted:

      * the distinct per-hit damages ARE ``floor(max * r / 100)`` for r in 85..=100,
        reconstructed in pure Python from one ``calculate_damage`` scalar; and
      * within every branch the two ``Damage`` instructions are EQUAL -- one roll
        shared across the hits, not two independent ones and not a halved total.
    """

    import poke_engine as pe

    hp = maxhp = 404

    def dummy():
        return pe.Pokemon(id="pikachu", level=1, hp=0)

    attacker = pe.Pokemon(
        id="marowak", level=81,
        types=("ground", "typeless"), base_types=("ground", "typeless"),
        hp=220, maxhp=220, ability="none", item="none",
        attack=190, defense=160, special_attack=100,
        special_defense=130, speed=250,
        moves=[pe.Move(id="bonemerang", pp=16)],
    )
    defender = pe.Pokemon(
        id="fearow", level=81,
        types=("normal", "typeless"), base_types=("normal", "typeless"),
        hp=hp, maxhp=maxhp, ability="none", item="none",
        attack=150, defense=150, special_attack=110,
        special_defense=150, speed=100, status="none",
        moves=[pe.Move(id="splash", pp=16)],
    )
    state = pe.State(
        side_one=pe.Side(active_index="0", pokemon=[attacker] + [dummy()] * 5),
        side_two=pe.Side(active_index="0", pokemon=[defender] + [dummy()] * 5),
        weather="none", terrain="none", trick_room=False,
    )

    max_regular = pe.calculate_damage(state, "bonemerang", "splash", False)[0][0]
    max_crit = pe.calculate_damage(state, "bonemerang", "splash", True)[0][1]
    reconstruction = sorted(
        {raw * r // 100 for raw in (max_regular, max_crit) for r in range(85, 101)}
    )

    per_hit: set[int] = set()
    shared_roll = True
    hit_counts: set[int] = set()
    total = 0.0
    for branch in pe.generate_instructions(state, "bonemerang", "splash"):
        total += branch.percentage
        hits = [
            int(str(i).split(": ")[1])
            for i in branch.instruction_list
            if str(i).startswith("Damage SideTwo")
        ]
        if not hits:
            continue
        hit_counts.add(len(hits))
        per_hit.update(hits)
        if len(set(hits)) != 1:
            shared_roll = False

    return {
        "flag": os.environ.get(ENV_FLAG),
        "max_regular": max_regular,
        "max_crit": max_crit,
        "per_hit_damages": sorted(per_hit),
        "reconstruction": reconstruction,
        "one_roll_shared_across_hits": shared_roll,
        "hit_counts": sorted(hit_counts),
        "mass_total": total,
    }


_PROBES = {
    "fan": _fan_probe,
    "secondary_composition": _secondary_composition_probe,
    "multihit": _multihit_probe,
}


def _run_probe(
    name: str, *, flag: str | None, imports: list[str] | None = None
) -> dict[str, object]:
    """Run one probe in a fresh interpreter with the flag exactly as given.

    ``POKEZERO_ENUMERATE_ROLLS`` is REMOVED from the child's environment first and then
    set only if ``flag`` is not ``None``. So ``flag=None`` measures the build's default
    under an import list, and NOT the operator's shell -- an ambient value inherited
    from whoever ran the suite can neither hide a leak nor manufacture one. What an
    import-time write would do is exactly what these probes are for: it happens inside
    the child, after the scrub, and the fan comes back enumerated.
    """

    environment = subproc_env()
    environment.pop(ENV_FLAG, None)
    if flag is not None:
        environment[ENV_FLAG] = flag
    argv = [sys.executable, str(Path(__file__).resolve()), "--probe", name]
    if imports:
        argv += ["--import", *imports]
    result = subprocess.run(
        argv,
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise AssertionError(
            f"probe {name!r} (flag={flag!r}, imports={imports!r}) exited "
            f"{result.returncode}\n{result.stdout}\n{result.stderr}"
        )
    return json.loads(result.stdout)


def _tracked_files() -> list[str]:
    listed = subprocess.run(
        ["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True
    )
    return [name for name in listed.stdout.decode("utf-8").split("\0") if name]


def _differential_consumers() -> list[str]:
    """Every tracked Python module that imports the differential, discovered live.

    Discovered rather than listed, so a NEW consumer is covered the day it lands
    instead of the day someone remembers to add it. This is the exact edge the textual
    scan could not see: these files need never mention the flag.
    """

    consumers = []
    for name in _tracked_files():
        if not name.endswith(".py") or name == _DIFFERENTIAL:
            continue
        if name in _IMPORT_PROBE_EXCLUSIONS:
            continue
        path = ROOT / name
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if "engine_transition_differential" in text:
            consumers.append(name)
    return sorted(consumers)


class RollEnumerationRuntimeScope(unittest.TestCase):
    """The load-bearing half: what the ENGINE does, measured in child processes."""

    def _assert_collapsed(self, result: dict, what: str) -> None:
        self.assertIsNone(result["flag"], what)
        self.assertEqual(
            result["emitted_damages"],
            COLLAPSED_FAN,
            f"{what} left the engine on the ENUMERATED roll path. Enumeration is a "
            "reference oracle for tests, not a configuration anything may turn on for "
            "a whole process: it is ~3700x slower in search, and it makes the fidelity "
            "differential certify a path production never takes. Whatever this surface "
            "imports must not write POKEZERO_ENUMERATE_ROLLS at module scope.",
        )
        self.assertLess(
            len(result["emitted_damages"]), len(result["enumeration_reconstruction"])
        )
        self.assertAlmostEqual(result["mass_total"], 100.0, delta=1e-3)

    def test_default_build_collapses_the_fan(self) -> None:
        """With the variable unset the engine takes the shipped partition path.

        The floor everything else stands on. It fails if the patch's default flips or
        if a build-time feature starts forcing the path. It does NOT and cannot detect
        an ambient environment leak -- ``_run_probe`` scrubs the variable on purpose --
        which is what the import probes below are for.
        """
        self._assert_collapsed(_run_probe("fan", flag=None), "the default build")

    def test_no_search_surface_puts_the_engine_on_the_enumerated_path(self) -> None:
        """Import each search surface in a child, then MEASURE. One child each.

        One child per surface, not one for all of them: a single batch would go red on
        the first enabler and name a set rather than a file.
        """
        for relative in _SEARCH_SURFACES:
            self.assertTrue((ROOT / relative).is_file(), f"missing search surface {relative}")
            with self.subTest(surface=relative):
                self._assert_collapsed(
                    _run_probe("fan", flag=None, imports=[relative]),
                    f"importing {relative}",
                )

    def test_importing_the_differential_does_not_enable_enumeration(self) -> None:
        """The regression pin for the defect this rework exists to fix.

        ``scripts/engine_transition_differential.py`` used to run
        ``os.environ.setdefault("POKEZERO_ENUMERATE_ROLLS", "1")`` at module scope.
        ``os.environ`` writes go through ``putenv``, so that was process-global and
        inherited by every child -- and 18 tracked modules import this file. Under the
        old code this test goes red; under the new code the flag is only ever written
        by ``main`` after ``--enumerate-rolls`` is parsed.
        """
        result = _run_probe("fan", flag=None, imports=[_DIFFERENTIAL])
        self._assert_collapsed(result, f"importing {_DIFFERENTIAL}")

    def test_importing_every_differential_consumer_leaves_the_fan_collapsed(self) -> None:
        """The import graph, swept: one child, every tracked consumer, then measure.

        This is the review-proven leak path. ``unittest`` imports every selected module
        before running any test, so one enabling module poisons the whole process --
        including any search child it later spawns through ``subproc_env()``. Importing
        them all into one child reproduces that exactly.

        Discovered live, with a floor, so the sweep cannot shrink to nothing and read
        green.
        """
        consumers = _differential_consumers()
        self.assertGreaterEqual(
            len(consumers),
            _MIN_DIFFERENTIAL_CONSUMERS,
            f"only {len(consumers)} tracked modules import the differential; the probe "
            "set shrank, so this sweep is weaker than the one that was reviewed",
        )
        result = _run_probe("fan", flag=None, imports=consumers + [_DIFFERENTIAL])
        self.assertEqual(sorted(result["imported"]), sorted(consumers + [_DIFFERENTIAL]))
        self._assert_collapsed(
            result, f"importing all {len(consumers)} differential consumers"
        )

    def test_the_runtime_probe_can_actually_see_an_enabler(self) -> None:
        """Negative control. Without this, every assertion above could be vacuous.

        Forces the child ON and requires the same probe to report ENUMERATED. If the
        probe could not tell the two paths apart -- wrong fixture, engine not rebuilt,
        flag renamed in the patch -- every "collapsed" result above would be an artifact
        of the measurement rather than a fact about the code.
        """
        result = _run_probe("fan", flag="1", imports=[_DIFFERENTIAL])
        self.assertEqual(result["flag"], "1")
        self.assertNotEqual(result["emitted_damages"], COLLAPSED_FAN)
        self.assertEqual(result["emitted_damages"], result["enumeration_reconstruction"])


class RollEnumerationOracle(unittest.TestCase):
    """The oracle itself. If it is wrong it blesses wrong collapsed recipes."""

    def test_the_flag_enumerates_the_fan_exactly(self) -> None:
        """With the flag on, the emitted damages ARE the sixteen rolls, both fans."""
        result = _run_probe("fan", flag="1")
        self.assertEqual(result["flag"], "1")
        self.assertEqual(
            result["emitted_damages"],
            result["enumeration_reconstruction"],
            "enumerated arms must be exactly floor(max * r / 100) for r in 85..=100, "
            "crit and non-crit, clamped at the defender's HP",
        )
        self.assertAlmostEqual(result["mass_total"], 100.0, delta=1e-3)

    def test_enumerated_arms_compose_with_a_probabilistic_secondary(self) -> None:
        """The C119 answer, as a standing assertion rather than a spike transcript."""
        result = _run_probe("secondary_composition", flag="1")
        expected = result["expected"]
        actual = result["actual"]
        self.assertEqual(
            set(expected), set(actual),
            "the (damage x burn) cells the engine emits must be exactly the cells the "
            "reconstruction predicts",
        )
        for key in sorted(expected):
            with self.subTest(cell=key):
                self.assertAlmostEqual(actual[key], expected[key], delta=1e-6)
        self.assertAlmostEqual(
            result["miss_mass"], 100.0 * (1.0 - _ACCURACY_FIREBLAST), delta=1e-4
        )
        self.assertAlmostEqual(result["mass_total"], 100.0, delta=1e-3)

    def test_the_collapsed_path_cannot_express_that_partition(self) -> None:
        """The negative control for the check above: it is not vacuously satisfiable.

        Run the identical reconstruction against the COLLAPSED engine. If it agreed
        there too, the previous test would be measuring nothing about enumeration.
        """
        result = _run_probe("secondary_composition", flag=None)
        self.assertNotEqual(
            set(result["expected"]), set(result["actual"]),
            "the collapsed cascade reproduced a full sixteen-roll x burn partition; "
            "either the default flipped or this witness stopped discriminating",
        )

    def test_enumeration_shares_one_roll_across_a_multi_hit(self) -> None:
        """Pins the multi-hit SEMANTIC CHANGE, which nothing else measures.

        See ``_multihit_probe``. Two hits, one roll, per-hit values equal to
        ``floor(max * r / 100)``. If enumeration ever rolled per hit, or halved a
        total, this goes red -- and until this existed, a reviewer who wanted to check
        it had to read the patch.
        """
        result = _run_probe("multihit", flag="1")
        self.assertEqual(result["hit_counts"], [2], "Bonemerang must emit two hits")
        self.assertTrue(
            result["one_roll_shared_across_hits"],
            "a branch applied two DIFFERENT per-hit damages; enumeration is specified "
            "to roll once and reuse the roll for every hit",
        )
        self.assertEqual(
            result["per_hit_damages"],
            result["reconstruction"],
            "per-hit damages must be floor(max * r / 100) for r in 85..=100",
        )
        self.assertAlmostEqual(result["mass_total"], 100.0, delta=1e-3)

    def test_the_collapsed_path_does_not_produce_that_multi_hit_partition(self) -> None:
        """Negative control for the multi-hit pin."""
        result = _run_probe("multihit", flag=None)
        self.assertNotEqual(
            result["per_hit_damages"],
            result["reconstruction"],
            "the collapsed cascade reproduced the full sixteen-roll multi-hit fan; "
            "either the default flipped or this witness stopped discriminating",
        )


class RollEnumerationMentionLedger(unittest.TestCase):
    """A change ledger, NOT the scope gate. The runtime class above is the gate.

    Kept because it is cheap and it makes a deliberate scope change visible in review.
    It is explicitly insufficient on its own: review defeated exactly this check with a
    file that mentions nothing and enables everything.
    """

    _MENTION_ALLOWLIST = {
        # Defines the flag and the enumerate-then-merge arms; default false.
        "third_party/poke-engine-gen3-enumerate-damage-rolls.patch",
        # Registers the patch in the frozen stack and records the decision.
        "third_party/poke-engine-gen3-patches.txt",
        # Explains why it does NOT set the flag at import, and exposes
        # --enumerate-rolls for a one-off comparison run.
        "scripts/engine_transition_differential.py",
        # Forces the flag OFF so the standing mass gate always measures the
        # collapsed cascade, whatever the ambient environment says.
        "tests/test_branch_mass_reconstruction.py",
        # This gate.
        "tests/test_roll_enumeration_scope.py",
        # The measurement record.
        "reports/c134_enumerate_rolls_oracle.md",
    }

    def test_the_set_of_files_mentioning_the_flag_is_recorded(self) -> None:
        mentions = set()
        for name in _tracked_files():
            path = ROOT / name
            if not path.is_file():
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            if ENV_FLAG in text:
                mentions.add(name)
        self.assertEqual(
            mentions,
            self._MENTION_ALLOWLIST,
            "the set of files mentioning POKEZERO_ENUMERATE_ROLLS changed. That is a "
            "scope change and belongs in review with a reason. Note this check is a "
            "LEDGER, not the gate: a file can enable enumeration process-wide without "
            "mentioning the flag at all, which is what RollEnumerationRuntimeScope "
            "measures.",
        )

    def test_the_unimportable_search_surfaces_do_not_mention_it(self) -> None:
        """Textual, because a Rust source file cannot be imported and measured."""
        for relative in _UNIMPORTABLE_SEARCH_SURFACES:
            path = ROOT / relative
            if not path.is_file():
                continue
            with self.subTest(file=relative):
                self.assertNotIn(
                    ENV_FLAG,
                    path.read_text(encoding="utf-8"),
                    f"{relative} references the enumeration flag. Search must keep the "
                    "collapsed path.",
                )

    def test_no_tracked_module_writes_the_environment_at_import(self) -> None:
        """AST pin on the SHAPE of the defect, across every tracked Python module.

        The runtime probes prove the property for the surfaces they import. This
        generalises it statically to the whole tree and names the mechanism, so a
        reintroduction fails with "you wrote os.environ at module scope" rather than
        with a fan mismatch in some unrelated file.

        AST rather than substring, because the previous attempt at this check matched
        the prose in the module docstring that EXPLAINS the defect and went red on a
        clean file. What is walked is module-level executable code only: function and
        class bodies are skipped, since those run when something calls them, which is
        the explicit act this whole design is built around.
        """

        import ast

        offenders: list[str] = []
        for name in _tracked_files():
            if not name.endswith(".py"):
                continue
            path = ROOT / name
            if not path.is_file():
                continue
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=name)
            except (SyntaxError, UnicodeDecodeError, OSError):
                continue
            # Module-level ``NAME = "POKEZERO_ENUMERATE_ROLLS"`` aliases, resolved, so
            # the check cannot be sidestepped by naming the constant. The version of
            # this test that only matched string literals was mutation-tested and
            # stayed GREEN against ``os.environ.setdefault(_ENUMERATE_ROLLS_ENV, "1")``,
            # which is exactly the line the differential used to carry.
            aliases = {ENV_FLAG}
            for statement in tree.body:
                if (
                    isinstance(statement, ast.Assign)
                    and isinstance(statement.value, ast.Constant)
                    and statement.value.value == ENV_FLAG
                ):
                    aliases.update(
                        slot.id for slot in statement.targets if isinstance(slot, ast.Name)
                    )

            def names_the_flag(node: ast.expr) -> bool:
                if isinstance(node, ast.Constant):
                    return node.value == ENV_FLAG
                return isinstance(node, ast.Name) and node.id in aliases

            for statement in tree.body:
                if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    continue
                for node in ast.walk(statement):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        continue
                    if isinstance(node, ast.Call):
                        target = ast.unparse(node.func)
                        if target in {
                            "os.environ.setdefault",
                            "os.environ.__setitem__",
                            "os.environ.update",
                            "os.putenv",
                        } and any(names_the_flag(argument) for argument in node.args):
                            offenders.append(f"{name}:{node.lineno}")
                        continue
                    if not isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
                        continue
                    written = node.targets if isinstance(node, ast.Assign) else [node.target]
                    for slot in written:
                        if (
                            isinstance(slot, ast.Subscript)
                            and ast.unparse(slot.value) == "os.environ"
                            and names_the_flag(slot.slice)
                        ):
                            offenders.append(f"{name}:{node.lineno}")
        self.assertEqual(
            offenders,
            [],
            "these modules write POKEZERO_ENUMERATE_ROLLS at IMPORT time: "
            f"{offenders}. Every importer then becomes an enabler -- os.environ goes "
            "through putenv, so the value is process-global and inherited by every "
            "child. Turning enumeration on has to be an explicit per-run act.",
        )

    def test_the_differential_exposes_the_explicit_opt_in(self) -> None:
        source = (ROOT / _DIFFERENTIAL).read_text(encoding="utf-8")
        self.assertIn("--enumerate-rolls", source, "the explicit opt-in must exist")


def _main(argv: list[str]) -> int:
    if len(argv) >= 3 and argv[1] == "--probe":
        name = argv[2]
        imports: list[str] = []
        if "--import" in argv:
            imports = argv[argv.index("--import") + 1 :]
        probe = _PROBES[name]
        payload = probe(imports) if name == "fan" else probe()
        print(json.dumps(payload))
        return 0
    unittest.main(argv=[argv[0]] + argv[1:])
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main(sys.argv))
