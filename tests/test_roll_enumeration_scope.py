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

# Surfaces that must never take the enumerated path.
#
# DISCOVERED LIVE, with a floor -- see ``search_surfaces()``. The first version of this
# gate hardcoded a 5-tuple, and review defeated it by counting: 22 of 37 tracked modules
# that touch the engine or the search crate were outside every probe. The one that
# mattered was ``src/pokezero/engine_env.py``, the engine-as-environment self-play driver
# that calls ``pokezero_search.env_step``, reached from the ``pokezero-rollout`` console
# script through ``src/pokezero/rollout_cli.py``. With a module-scope enabler there, a
# real search process came back ENUMERATED and this module still reported
# ``Ran 14 tests ... OK``. Also outside: ``scripts/bench_leaf_search.py``,
# ``hc_depth_grid.py``, ``depth_tactics_probe.py``, ``run_mcts_depth_eval.py``,
# ``leaf_root_parity.py``, ``search_crate_branch_probe.py``, and
# ``scripts/engine_behavioral_probes.py`` -- a gate this PR itself cites.
#
# A hardcoded tuple cannot enforce a property about a growing tree. It states the surfaces
# someone remembered; the property is about the ones they did not.
_UNIMPORTABLE_SEARCH_SURFACES = ("rust/pokezero-search/src/lib.rs",)

# The named CORE. These get one child EACH, so a failure names a file instead of a set,
# and each is asserted to still be inside the discovered set -- a core entry that stops
# being discovered means the discovery predicate rotted.
_CORE_SEARCH_SURFACES = (
    # The search entry point.
    "src/pokezero/engine_search.py",
    # The engine-as-environment driver review found uncovered: pokezero_search.env_step
    # plus LeafEncoder, i.e. a real search process.
    "src/pokezero/engine_env.py",
    # The console script that reaches it (`pokezero-rollout`).
    "src/pokezero/rollout_cli.py",
    "scripts/bench_crate_search.py",
    # The bench behind the throughput table this decision cites: a bench that silently
    # measured the enumerated path would report a number for a configuration nobody runs.
    "scripts/bench_multiply_search.py",
    # A gate this PR cites for its own evidence.
    "scripts/engine_behavioral_probes.py",
)

# Carried over from the hardcoded tuple this gate replaced. They are probed per file like
# the core, but they are NOT required to be discovered: measured, they reference neither
# the engine nor the search crate, directly or by console-script entry, so no honest
# predicate reaches them. Keeping them probed costs one child each and keeps the coverage
# the old tuple claimed; pretending discovery finds them would mean bending the predicate
# to fit a list, which is the failure this gate exists to stop repeating.
_LEGACY_SEARCH_SURFACES = (
    "src/pokezero/env.py",
    "src/pokezero/selfplay.py",
)

# Floor on the discovered search-surface set. Measured at 59 when this gate was written;
# the floor sits below that so ordinary deletions do not trip it, but a predicate that
# collapses does.
_MIN_SEARCH_SURFACES = 45

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

# Tracked modules deliberately kept OUT of discovery. Empty today; an entry here is a
# scope reduction and needs a reason, because a module listed here is a module this gate
# does not cover.
_IMPORT_PROBE_EXCLUSIONS: frozenset[str] = frozenset()

# The ONLY tolerated reason a search surface may fail to import: an optional third-party
# dependency that is not installed in this environment. Anything else -- a SyntaxError, a
# missing FIRST-party module, an exception raised at import -- is a real failure and reds
# the gate, because a module that cannot load is a module this gate does not cover.
#
# Tolerated rather than pinned by equality, deliberately: the set is a property of the
# ENVIRONMENT, not of the tree, so pinning it would go red on any machine with a different
# extras install. The floor below is what stops that tolerance becoming a hole.
_OPTIONAL_IMPORT_DEPENDENCIES = frozenset({"numpy", "torch", "scipy", "matplotlib", "pandas"})

# Floor on how many search surfaces were actually IMPORTED, not merely discovered.
# Discovery can find 59 and still cover nothing if every import dies. Measured at 58 of
# 59 locally, the one gap being ``scripts/bench_leaf_search.py`` under a torch-less venv.
_MIN_IMPORTED_SEARCH_SURFACES = 40


# ---------------------------------------------------------------------------------------------
# Child-process probes. Run as ``python tests/test_roll_enumeration_scope.py --probe NAME``.
# ---------------------------------------------------------------------------------------------

_ACCURACY_ROCKSLIDE = 0.9
_ACCURACY_FIREBLAST = 0.85
_BURN_CHANCE = 0.10
_CRIT_RATE = 1.0 / 16.0


def _module_name_for(relative: str) -> str:
    """Dotted import name for a tracked path."""

    parts = Path(relative).with_suffix("").parts
    if parts[0] == "src":
        # A package module: import it by its DOTTED name, the way production does.
        # Importing it by file location would execute it outside its package and die
        # on the first relative import -- and a module that cannot even load is not a
        # measurement of what it does when it loads.
        parts = parts[1:]
    else:
        # scripts/ and tests/ are flat sys.path roots for their own modules.
        parts = parts[-1:]
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _import_targets(relatives: list[str]) -> tuple[list[str], dict[str, str]]:
    """Import each tracked path as a module, the way a test runner would.

    ``sys.path`` gets ``src``, ``scripts`` and ``tests`` the same way the importers
    themselves do, so a module that does ``from engine_transition_differential import
    ...`` resolves exactly as it does in a real run. Import order is the caller's.

    Returns ``(imported, failed)``. An import that RAISES is recorded rather than
    propagated, for two reasons. It must not abort the sweep -- one module with a
    missing optional dependency would otherwise take every module after it out of
    coverage silently. And a module that enabled the flag and *then* died still shows
    up, because the fan is measured after this returns.

    The caller asserts on ``failed``, so losing coverage is a visible, named event and
    not a quiet shrink.
    """

    import importlib

    for extra in ("src", "scripts", "tests", ""):
        candidate = str(ROOT / extra) if extra else str(ROOT)
        if candidate not in sys.path:
            sys.path.insert(0, candidate)
    imported: list[str] = []
    failed: dict[str, str] = {}
    for relative in relatives:
        try:
            importlib.import_module(_module_name_for(relative))
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as error:  # noqa: BLE001
            failed[relative] = f"{type(error).__name__}: {error}"
            continue
        imported.append(relative)
    return imported, failed


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

    imported, import_failures = _import_targets(imports or [])

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
        "import_failures": import_failures,
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


def module_scope_env_writes(source: str, filename: str = "<probe>") -> list[int]:
    """Line numbers where module-level code writes ``POKEZERO_ENUMERATE_ROLLS``.

    AST rather than substring, because a substring check matches the prose in the module
    docstring that EXPLAINS the defect and goes red on a clean file. Only module-level
    executable code is walked: function and class bodies run when something calls them,
    which is the explicit per-run act this whole design is built around.

    THREE alias families are resolved, because review broke the first version on two of
    them and only one of the two was adversarial:

    * the ``os`` MODULE -- ``import os``, ``import os as _os``;
    * ``os.environ`` ITSELF -- ``from os import environ``, ``from os import environ as
      E``, and a module-level ``environ = os.environ``. ``from os import environ`` is an
      ordinary idiom, not a dodge, so a check blind to it is wrong on real code and not
      merely gameable; and
    * the flag NAME -- a module-level ``NAME = "POKEZERO_ENUMERATE_ROLLS"``.

    The first version resolved only the third. ``import os as _os`` +
    ``_os.environ.setdefault(...)`` and ``from os import environ`` +
    ``environ.setdefault(...)`` both passed it, leaving nothing but the mention ledger --
    which this module's own docstring calls "explicitly not what holds the line".
    """

    import ast

    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError:
        return []

    os_modules: set[str] = set()
    environ_objects: set[str] = set()
    flag_names: set[str] = {ENV_FLAG}

    for statement in ast.walk(tree):
        if isinstance(statement, ast.Import):
            for alias in statement.names:
                if alias.asname:
                    if alias.name == "os":
                        os_modules.add(alias.asname)
                # ``import os`` and ``import os.path`` both bind the root name ``os``.
                elif alias.name == "os" or alias.name.startswith("os."):
                    os_modules.add("os")
        elif isinstance(statement, ast.ImportFrom) and statement.module == "os":
            for alias in statement.names:
                if alias.name == "environ":
                    environ_objects.add(alias.asname or "environ")
                elif alias.name == "putenv":
                    environ_objects.add(f"putenv:{alias.asname or 'putenv'}")

    def is_environ(node: ast.expr) -> bool:
        if isinstance(node, ast.Name):
            return node.id in environ_objects
        return (
            isinstance(node, ast.Attribute)
            and node.attr == "environ"
            and isinstance(node.value, ast.Name)
            and node.value.id in os_modules
        )

    # Module-level rebindings: ``environ = os.environ`` and ``NAME = "<flag>"``.
    for statement in tree.body:
        if not isinstance(statement, ast.Assign):
            continue
        if is_environ(statement.value):
            environ_objects.update(
                slot.id for slot in statement.targets if isinstance(slot, ast.Name)
            )
        elif isinstance(statement.value, ast.Constant) and statement.value.value == ENV_FLAG:
            flag_names.update(
                slot.id for slot in statement.targets if isinstance(slot, ast.Name)
            )

    def names_the_flag(node: ast.expr) -> bool:
        if isinstance(node, ast.Constant):
            return node.value == ENV_FLAG
        return isinstance(node, ast.Name) and node.id in flag_names

    def mapping_mentions_flag(node: ast.expr) -> bool:
        return isinstance(node, ast.Dict) and any(
            key is not None and names_the_flag(key) for key in node.keys
        )

    def is_putenv(func: ast.expr) -> bool:
        if isinstance(func, ast.Name):
            return f"putenv:{func.id}" in environ_objects
        return (
            isinstance(func, ast.Attribute)
            and func.attr == "putenv"
            and isinstance(func.value, ast.Name)
            and func.value.id in os_modules
        )

    _WRITE_METHODS = {"setdefault", "__setitem__", "update", "pop", "popitem", "clear"}
    offenders: list[int] = []
    for statement in tree.body:
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        for node in ast.walk(statement):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            if isinstance(node, ast.Call):
                func = node.func
                if is_putenv(func) and any(names_the_flag(a) for a in node.args):
                    offenders.append(node.lineno)
                elif (
                    isinstance(func, ast.Attribute)
                    and func.attr in _WRITE_METHODS
                    and is_environ(func.value)
                    and any(
                        names_the_flag(a) or mapping_mentions_flag(a) for a in node.args
                    )
                ):
                    offenders.append(node.lineno)
                continue
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                written = node.targets if isinstance(node, ast.Assign) else [node.target]
                for slot in written:
                    if (
                        isinstance(slot, ast.Subscript)
                        and is_environ(slot.value)
                        and names_the_flag(slot.slice)
                    ):
                        offenders.append(node.lineno)
            elif isinstance(node, ast.AugAssign):
                # ``os.environ |= {"<flag>": "1"}`` -- MutableMapping supports it.
                if is_environ(node.target) and mapping_mentions_flag(node.value):
                    offenders.append(node.lineno)
                elif (
                    isinstance(node.target, ast.Subscript)
                    and is_environ(node.target.value)
                    and names_the_flag(node.target.slice)
                ):
                    offenders.append(node.lineno)
    return sorted(set(offenders))


def _console_script_modules() -> list[str]:
    """Every ``[project.scripts]`` entry point, as a tracked path under ``src/``.

    Console scripts are the things that BECOME processes. ``src/pokezero/rollout_cli.py``
    references neither ``poke_engine`` nor ``pokezero_search`` by name, and reaches
    ``engine_env`` -- and therefore ``pokezero_search.env_step`` -- transitively, so a
    text predicate alone would miss the exact entry point review used to break the old
    gate.
    """

    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    modules: list[str] = []
    in_scripts = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("["):
            in_scripts = stripped == "[project.scripts]"
            continue
        if not in_scripts or "=" not in stripped or stripped.startswith("#"):
            continue
        target = stripped.split("=", 1)[1].strip().strip('"').strip("'")
        dotted = target.split(":", 1)[0]
        candidate = "src/" + dotted.replace(".", "/") + ".py"
        if (ROOT / candidate).is_file():
            modules.append(candidate)
    return modules


def search_surfaces() -> list[str]:
    """Every tracked module that could put a real process on a roll path, discovered.

    Two predicates, unioned:

    * any tracked ``.py`` under ``src/pokezero/`` or ``scripts/`` whose text references
      ``pokezero_search`` or ``poke_engine`` -- it touches the search crate or the
      engine directly; and
    * every ``[project.scripts]`` console script -- it becomes a process, whatever it
      references by name.

    Discovered rather than listed, because the property is about the modules nobody
    remembered. The hardcoded five-tuple this replaces left 22 of 37 engine-touching
    modules outside every probe, including a self-play driver that runs the search.
    """

    found: set[str] = set(_console_script_modules())
    for name in _tracked_files():
        if not name.endswith(".py"):
            continue
        if not (name.startswith("src/pokezero/") or name.startswith("scripts/")):
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
        if "pokezero_search" in text or "poke_engine" in text:
            found.add(name)
    # This module's own probes import the differential deliberately; it is measured by
    # its own test and is not a search surface.
    found.discard(_DIFFERENTIAL)
    return sorted(found)


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

    def test_the_core_search_surfaces_are_still_discovered(self) -> None:
        """The named core must remain INSIDE the discovered set.

        Without this, a discovery predicate that quietly stopped matching
        ``engine_env.py`` would shrink coverage back to roughly the hardcoded tuple
        review defeated, and the count floor alone would not notice: the set would still
        be large, just missing the module that matters.
        """
        discovered = set(search_surfaces())
        for relative in _CORE_SEARCH_SURFACES:
            with self.subTest(surface=relative):
                self.assertTrue((ROOT / relative).is_file(), f"missing {relative}")
                self.assertIn(
                    relative,
                    discovered,
                    f"{relative} is no longer discovered as a search surface; the "
                    "discovery predicate in search_surfaces() has rotted",
                )

    def test_no_core_search_surface_puts_the_engine_on_the_enumerated_path(self) -> None:
        """Import each CORE surface in its own child, then MEASURE.

        One child per surface here, not one for all of them: a batch goes red on the
        first enabler and names a set rather than a file. The exhaustive sweep over the
        discovered set is the next test.
        """
        for relative in _CORE_SEARCH_SURFACES + _LEGACY_SEARCH_SURFACES:
            with self.subTest(surface=relative):
                self.assertTrue((ROOT / relative).is_file(), f"missing {relative}")
                result = _run_probe("fan", flag=None, imports=[relative])
                self.assertEqual(
                    result["import_failures"], {}, f"{relative} did not import"
                )
                self._assert_collapsed(result, f"importing {relative}")

    def test_no_discovered_search_surface_enumerates(self) -> None:
        """The exhaustive sweep: every discovered surface, one child, then MEASURE.

        This is the assertion review's counter-example breaks. With a module-scope
        enabler in ``src/pokezero/engine_env.py`` the old gate reported
        ``Ran 14 tests ... OK``; this test imports that module along with every other
        engine-touching module in the tree and reads the fan afterwards, so the enabler
        shows up wherever it is.

        Modules that fail to IMPORT are reported, not skipped silently -- losing one is
        losing coverage, and the assertion below names which.
        """
        surfaces = search_surfaces()
        self.assertGreaterEqual(
            len(surfaces),
            _MIN_SEARCH_SURFACES,
            f"only {len(surfaces)} search surfaces discovered; the predicate in "
            "search_surfaces() collapsed, so this sweep is weaker than the one reviewed",
        )
        result = _run_probe("fan", flag=None, imports=surfaces)

        # Every failure must be a MISSING OPTIONAL DEPENDENCY and nothing else. A module
        # that fails for any other reason is a module this gate silently stopped covering.
        unexplained = {
            name: reason
            for name, reason in result["import_failures"].items()
            if not any(
                reason == f"ModuleNotFoundError: No module named '{dependency}'"
                for dependency in _OPTIONAL_IMPORT_DEPENDENCIES
            )
        }
        self.assertEqual(
            unexplained,
            {},
            "these search surfaces failed to import for a reason other than a missing "
            f"optional dependency: {unexplained}. Each one is outside this gate's "
            "coverage until it loads.",
        )
        self.assertGreaterEqual(
            len(result["imported"]),
            _MIN_IMPORTED_SEARCH_SURFACES,
            f"only {len(result['imported'])} of {len(surfaces)} search surfaces actually "
            f"imported (skipped: {sorted(result['import_failures'])}). Discovery finding "
            "them is not coverage; loading them is.",
        )
        self._assert_collapsed(
            result, f"importing all {len(result['imported'])} discovered search surfaces"
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
        self.assertEqual(result["import_failures"], {}, "a consumer did not import")
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
        """

        offenders: list[str] = []
        for name in _tracked_files():
            if not name.endswith(".py"):
                continue
            path = ROOT / name
            if not path.is_file():
                continue
            try:
                source = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            offenders.extend(
                f"{name}:{line}" for line in module_scope_env_writes(source, name)
            )
        self.assertEqual(
            offenders,
            [],
            "these modules write POKEZERO_ENUMERATE_ROLLS at IMPORT time: "
            f"{offenders}. Every importer then becomes an enabler -- os.environ goes "
            "through putenv, so the value is process-global and inherited by every "
            "child. Turning enumeration on has to be an explicit per-run act.",
        )

    def test_the_ast_check_sees_through_every_alias_form(self) -> None:
        """Pins the detector itself, on synthetic sources, form by form.

        Running the detector over a clean tree proves only that it found nothing. Review
        broke the first version by writing the two forms it could not see -- and one of
        them, ``from os import environ``, is an ordinary idiom rather than a dodge, so
        the check was wrong on real code and not merely gameable. Each row below is a
        way to write the same defect.
        """
        flag = ENV_FLAG
        positives = {
            "import os": f'import os\nos.environ.setdefault("{flag}", "1")\n',
            "import os as alias": f'import os as _os\n_os.environ.setdefault("{flag}", "1")\n',
            "from os import environ": f'from os import environ\nenviron.setdefault("{flag}", "1")\n',
            "from os import environ as alias": (
                f'from os import environ as E\nE.setdefault("{flag}", "1")\n'
            ),
            "module-level environ rebinding": (
                f'import os\nenv = os.environ\nenv["{flag}"] = "1"\n'
            ),
            "subscript assignment": f'import os\nos.environ["{flag}"] = "1"\n',
            "subscript via alias": f'import os as _os\n_os.environ["{flag}"] = "1"\n',
            "flag-name constant": (
                f'import os\nFLAG = "{flag}"\nos.environ.setdefault(FLAG, "1")\n'
            ),
            "both aliased at once": (
                f'from os import environ as E\nFLAG = "{flag}"\nE.setdefault(FLAG, "1")\n'
            ),
            "update with a dict": f'import os\nos.environ.update({{"{flag}": "1"}})\n',
            "putenv": f'import os\nos.putenv("{flag}", "1")\n',
            "putenv imported directly": f'from os import putenv\nputenv("{flag}", "1")\n',
            "augmented merge": f'import os\nos.environ |= {{"{flag}": "1"}}\n',
        }
        for label, source in positives.items():
            with self.subTest(form=label):
                self.assertTrue(
                    module_scope_env_writes(source),
                    f"the AST check cannot see {label!r}; that form would ship an "
                    "import-time enabler with this gate green",
                )

        negatives = {
            # The explicit per-run act. Inside a function, so it runs when called.
            "inside a function": (
                f'import os\ndef go():\n    os.environ["{flag}"] = "1"\n'
            ),
            # Reading is fine; only writing makes an importer an enabler.
            "module-level read": f'import os\nVALUE = os.environ.get("{flag}")\n',
            # A different variable entirely.
            "another variable": 'import os\nos.environ["PYTHONPATH"] = "x"\n',
            # Prose naming the flag, which is what a substring check tripped on.
            "docstring prose": f'"""We must never set {flag} at import."""\nimport os\n',
        }
        for label, source in negatives.items():
            with self.subTest(form=label):
                self.assertEqual(
                    module_scope_env_writes(source), [],
                    f"the AST check false-positives on {label!r}",
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
