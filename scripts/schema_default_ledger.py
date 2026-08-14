#!/usr/bin/env python3
"""Enumerate every site that reaches the global observation-schema default.

THE DENOMINATOR IS THE POINT. This script exists so no figure about the schema-default
conflation is ever quoted from memory or from a hand-picked file list again. It walks the AST
of every tracked .py file and reports each site with a stable kind, so the count is derived and
re-derivable rather than recalled.

A "site reaching the global default" is any of:

  bare-const           a read of `OBSERVATION_SCHEMA_VERSION` itself              (15)
  default-spec         a read of `DEFAULT_REPLAY_OBSERVATION_SPEC`                (39)
  implicit:<Surface>   a call to <Surface> leaving at least one default-bearing kwarg unnamed,
                       one kind per surface so a new one cannot join an existing bucket. All THREE
                       surfaces, and all three DERIVED from src/ -- the `*`-marked alternate-constructor
                       row is gone, because `compact_category` stopped being one when
                       TransformerPolicyConfig named its schema:

                         LinearPolicyModel                3    OnlineBattleAgent           2
                         observation_from_player_state    0

                       WAS EIGHT. `ObservationSpec` (24 rows here) and `PokeZeroObservationV0` (49)
                       left the model when their `schema_version` field defaults stopped naming the
                       global and named v2.2 instead: `derive_surfaces` identifies a surface BY that
                       default, so the class stops being one and all 73 of their call sites stop
                       being counted. Intended direction -- those sites now inherit a named version.
                       N went 322 -> 249 in the same change.

                       `observation_from_player_state` is DERIVED and carries ZERO rows: all 130 of
                       its call sites pass `spec=` explicitly. A modelled surface with no open sites
                       is the goal state, not an absence -- it is listed so that a caller which
                       later drops `spec=` lands in a named bucket instead of appearing to be a new
                       surface. An earlier version of this table said "all seven derived surfaces"
                       while listing compact_category (not derived) and omitting this one: the count
                       was right and the membership was not.

                       The row's `unclosed` field names which kwarg is still open. All FIVE counts
                       above are held by tests/test_schema_default_ledger.py, by two different
                       mechanisms: the surface rows are PARSED and compared to the derivation, and
                       the kind rows at the top of this table are GENERATED -- the test
                       asserts that region is byte-equal to `--render-kinds-table`. Regenerate it
                       with that flag rather than editing it by hand.

                       The kind rows are generated because parsing them could not be made sound.
                       Five review rounds narrowed a regex that tried to recognise a hand-written
                       row, and each round closed the escapes just named and left the next ones
                       open: an uppercase or capitalised name, an underscore, a `*` or `-` bullet
                       this table already uses, a trailing colon in the style of the `implicit:`
                       row below, `[16]`, `n=16`, `(16 rows)` in the style of the WAS EIGHT line
                       above, the count moved to the next line -- plus a duplicated row where the
                       WRONG count silently won. Any of those made a retired kind's stale count
                       invisible while the test stayed green. A byte comparison has no grammar to
                       evade, so a reformat is a diff instead of a disappearance.

                       This line itself once said "nine" while six were pinned and eight existed.

  Two retired names, `implicit-spec` and `implicit-cfg`, were listed here long after the code
  stopped emitting them -- the reader-facing vocabulary staled in the same change that derived the
  surfaces and recovered 187 sites. Worse, a commit message claimed this list had been updated when
  the edit had silently no-op'd: a `str.replace` whose target no longer existed, with no assertion
  that it applied. Every figure above is re-derivable by running this script.

Not counted, deliberately: reads of the per-version names (`..._V2_2`, `..._V4`) and of
`SUPPORTED_...`/`REPLAY_OBSERVATION_SPECS_BY_SCHEMA`. Those NAME a schema, which is the state
this migration is moving sites INTO; counting them would make the burndown never converge.

Usage:  .venv/bin/python scripts/schema_default_ledger.py [--json] [--by-file]

The script imports nothing from the package -- it only parses -- so any interpreter works. It
exits 2 in EVERY mode if any tracked file fails to parse or is missing, because an unmeasured
file silently shrinks the denominator. (Until #1239 one tracked file could not be parsed on 3.11
and this docstring told readers to avoid the venv; that is no longer true and the advice is
removed rather than left to mislead.)
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CONST = "OBSERVATION_SCHEMA_VERSION"
DEFAULT_SPEC = "DEFAULT_REPLAY_OBSERVATION_SPEC"
# DERIVED, not hardcoded. The first version listed three call names by hand and therefore
# undercounted by 187 sites -- `LocalShowdownConfig.observation_spec` alone has ~133 callers.
# That is this program's own error class committed inside the instrument built to retire it:
# a denominator chosen rather than enumerated. `derive_surfaces` scans src/ for every class
# attribute or parameter whose DEFAULT is one of GLOBALS, so a new surface is counted the day
# it is written.
GLOBALS = {CONST, DEFAULT_SPEC}
# Named once so the gate and its test cannot disagree. The test previously mirrored this as a
# literal and left `__post_init__` unpinned.
CONSTRUCTOR_NAMES = ("__init__", "__new__", "__post_init__")
# Alternate constructors do not re-declare the field, so they cannot be derived; they are
# listed against the type they build and asserted to exist.
# EMPTY now that TransformerPolicyConfig names its schema. `compact_category` was listed here as an
# alternate constructor of that type: it does not re-declare the field, so `derive_surfaces` cannot
# find it by scanning for a default that IS one of the globals. Once the type stops being a surface,
# the entry is stale -- and the guard below caught exactly that rather than letting the count drift.
EXTRA_CONSTRUCTORS: dict[str, list[str]] = {}


def dotted_segments(node: ast.AST) -> list[str]:
    """["a", "b", "c"] for `a.b.c`; [] if the chain is not a pure Name/Attribute path.

    Returned leftmost-first. Used to find a global ANYWHERE in an attribute chain: checking only
    the outermost `.attr` or only the leftmost Name both miss `mod.GLOBAL.attribute`, which is how
    the two live width defaults at neural_policy.py:245-246 would be spelled module-qualified.
    """
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return []
    parts.append(node.id)
    return list(reversed(parts))


def is_pokezero_submodule(node: ast.ImportFrom, name: str, *, importer: Path | None = None) -> bool:
    """Is `name` in `from <node.module> import <name>` a pokezero SUBMODULE, or just a name?

    Derived from the filesystem, not from the name's shape. `from pokezero import observation`
    binds a module; `from pokezero.observation import ObservationSpec` binds a class, and
    registering the latter as a module base turned `ObservationSpec.OBSERVATION_SCHEMA_VERSION`
    into a row. Guessing by capitalisation would be another enumeration from memory.

    A RELATIVE import resolves against the importing file's own package -- see the body. An earlier
    version of this docstring asserted the opposite ("resolve against src/pokezero/ ... the deepest
    node.level in the tree is 1, so treating any relative level as rooted at the package is exact
    here rather than approximate"). Every clause of that was false, and the correction was written
    as a comment eight lines below while the false text stayed. Deleted rather than annotated: two
    contradicting statements in one function is worse than the wrong one alone, because a reader
    who stops at the docstring never reaches the correction.
    """
    if node.level > 0:
        # Resolve against the IMPORTING file's package, walking up `node.level - 1` directories, and
        # then through `node.module` if there is one.
        #
        # The previous version hardcoded `src/pokezero` and discarded `node.module`, with a docstring
        # claiming "the deepest node.level in the tree is 1, so treating any relative level as rooted
        # at the package is exact here rather than approximate". The max level is **2**
        # (src/pokezero/mcts_eval/lattice.py:31), and it was approximate in BOTH directions:
        # `from .mcts_eval import lattice` scored 0 where it should score 1, and
        # `from .mcts_eval import observation` -- a NAME, not a module -- scored 1 where it should
        # score 0. That sentence was itself the enumeration-from-memory this file accuses itself of
        # three lines away.
        base = (importer.parent if importer is not None else REPO / "src" / "pokezero")
        for _ in range(node.level - 1):
            base = base.parent
        package = base / Path(*(node.module or "").split(".")) if node.module else base
    else:
        parts = (node.module or "").split(".")
        package = REPO / "src" / Path(*parts)
    return (package / f"{name}.py").is_file() or (package / name / "__init__.py").is_file()


def base_root(node: ast.AST) -> str | None:
    """Leftmost Name of a pure Name/Attribute chain: `a.b.c` -> "a", `f().b` -> None.

    Walking to the ROOT rather than testing `isinstance(node.value, ast.Name)` is what makes an
    arbitrarily deep base match. The one-level test could only see `O.GLOBAL`, so every dotted
    base -- `pokezero.observation.GLOBAL`, `pokezero.showdown.GLOBAL.numeric_feature_count` --
    read the default with the denominator unmoved and the authorship gate green.
    """
    while isinstance(node, ast.Attribute):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


def class_body_statements(node: ast.ClassDef):
    """Statements in a class body, descending through control flow but NOT into methods.

    A class attribute inside `if TYPE_CHECKING:` or `if sys.version_info >= ...:` is a nested
    statement, so scanning only `node.body` lost it. But plain `ast.walk(node)` goes too far the
    other way -- it descends into METHOD bodies, and `from_dict` contains
    `default_spec = REPLAY_OBSERVATION_SPECS_BY_SCHEMA.get(...)`, a LOCAL VARIABLE. Walking that
    derived a bogus `default_spec` kwarg for TransformerPolicyConfig/compact_category, which moved N
    from 390 to 398 and rewrote the `unclosed` field of 101 rows -- a false positive that would have
    read as "eight new sites found" in the diff.

    So: recurse through If/Try/With/For/While bodies, and stop at any nested scope.
    """
    stack = list(node.body)
    while stack:
        statement = stack.pop()
        if isinstance(
            statement,
            (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
        ):
            continue  # a nested scope: its assignments are not this class's fields
        yield statement
        for field in ("body", "orelse", "finalbody"):
            stack.extend(getattr(statement, field, []) or [])
        for handler in getattr(statement, "handlers", []) or []:
            stack.extend(handler.body)


def derive_surfaces() -> dict[str, set[str]]:
    """{callable name -> kwargs that silently default to the global default}.

    SPELLING COVERAGE MATTERS MORE HERE THAN IN `sites_in`. A missed read in `sites_in` loses one
    row; a missed DECLARATION here loses the surface, and with it every one of its call sites --
    `LocalShowdownConfig` alone is 133 of the 390. So an under-derived surface is the largest
    single way this denominator can be wrong, and for six rounds this function recognised exactly
    one spelling (`name: T = GLOBAL`) while `sites_in` accumulated four. The round-5 alias fix was
    applied to `sites_in` and never here.

    Spellings now handled, each pinned by SurfaceDerivationSeesEverySpellingTest:
      name: T = GLOBAL                          the original
      name = GLOBAL                             un-annotated (ast.Assign, not AnnAssign)
      name: T = ALIAS                           `from ... import GLOBAL as ALIAS`
      name: T = field(default=GLOBAL)           dataclasses -- 201 `field(default` uses in src/
      name: T = field(default_factory=lambda: GLOBAL)
      name: T = mod.GLOBAL / pkg.mod.GLOBAL     module-qualified, at any depth
      name: T = GLOBAL.attribute                the value side (2 live sites)
    """
    found: dict[str, set[str]] = {}
    # Aliases for the globals, resolved across all of src/ rather than per file: a surface is
    # declared in one module, and a scan that missed the alias silently de-derived it.
    alias_to_global: dict[str, str] = {}
    for path in (REPO / "src").rglob("*.py"):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                for a in node.names:
                    if a.name in GLOBALS and a.asname:
                        alias_to_global[a.asname] = a.name

    # Module roots visible anywhere in src/, for the same reason `sites_in` resolves them per file:
    # without this the Attribute arm below accepted ANY chain containing a global token, so
    # `numpy.OBSERVATION_SCHEMA_VERSION` as a class-attribute default derived a surface -- and a
    # spuriously derived surface costs EVERY call site of that class name, which is the more
    # expensive of the two over-match directions by this file's own 133-of-390 argument.
    src_module_roots: set[str] = {"pokezero"}
    for path in (REPO / "src").rglob("*.py"):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                for a in node.names:
                    if node.level > 0 or (node.module or "").startswith("pokezero"):
                        if is_pokezero_submodule(node, a.name, importer=path):
                            src_module_roots.add(a.asname or a.name)
            elif isinstance(node, ast.Import):
                for a in node.names:
                    if a.name == "pokezero" or a.name.startswith("pokezero."):
                        src_module_roots.add(a.asname or a.name.split(".")[0])

    def dflt_name(node):
        # `field(default=GLOBAL)` / `field(default_factory=lambda: GLOBAL)`. A dataclass field
        # wrapper is the single most likely spelling for a NEW surface in this codebase, and it
        # was invisible: the default sits inside a Call, so neither arm below ever saw it.
        if isinstance(node, ast.Call):
            fn = node.func
            fname = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", None)
            if fname == "field":
                for kw in node.keywords:
                    if kw.arg == "default":
                        return dflt_name(kw.value)
                    if kw.arg == "default_factory" and isinstance(kw.value, ast.Lambda):
                        return dflt_name(kw.value.body)
                return None
            # Any OTHER call: recurse into its arguments. `field(default_factory=lambda:
            # GLOBAL.copy())` and `lambda: replace(GLOBAL, ...)` are the realistic factory idioms
            # for a frozen spec, and returning None for a non-`field` Call lost the surface for all
            # of them.
            for arg in list(node.args) + [kw.value for kw in node.keywords]:
                found_in_arg = dflt_name(arg)
                if found_in_arg in GLOBALS:
                    return found_in_arg
            return None
        if isinstance(node, ast.Name):
            return alias_to_global.get(node.id, node.id)
        if isinstance(node, ast.Attribute):
            # EVERY segment of the chain, not just the two ends. `a.b.GLOBAL.attr` puts the
            # global in the middle, so neither `node.attr` (the outermost) nor `base_root`
            # (the leftmost) finds it: the first cut of this fix walked to the root and returned
            # `pokezero`. Collect the whole dotted path and check each name, which is the only
            # form that does not depend on where in the chain the global happens to sit.
            #
            # This replaced two hand-written arms, one per end of the chain. The outer arm --
            # matching `node.attr` -- shipped with a comment claiming its spelling was "an idiom
            # used 31 times in this repo" and that it was "pinned by test_schema_default_ledger.py".
            # Both were false. The count was 0, not 31: no dotted read of either global exists
            # anywhere in the tree (see docs/schema_default_ledger.md for the derivation and for
            # the grep/AST reconciliation). And deleting the arm outright left the whole gate GREEN,
            # so it was pinned by nothing -- a false claim about coverage, in the file whose thesis
            # is that unpinned prose stales.
            #
            # Both are now true of the code as written: the walk below is exercised by
            # SurfaceDerivationSeesEverySpellingTest, which probes the one-level, dotted-base and
            # value-side forms and was kill-confirmed against this line, and the prose no longer
            # quotes a prevalence figure. Kept because a surface DECLARED module-qualified would
            # otherwise be derive-blind, which costs every one of its call sites -- 133 for the
            # largest surface. It is a guard against a spelling that has not landed, and that is
            # now what it says it is.
            segments = dotted_segments(node)
            # The chain must be rooted in something that can actually BE a pokezero module, or a
            # lookalike on an unrelated object derives a surface.
            if segments and (segments[0] in src_module_roots or segments[0] in GLOBALS
                             or segments[0] in alias_to_global):
                for segment in segments:
                    resolved = alias_to_global.get(segment, segment)
                    if resolved in GLOBALS:
                        return resolved
        return None

    for path in (REPO / "src").rglob("*.py"):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for st in class_body_statements(node):
                    # AnnAssign AND Assign. An un-annotated class attribute is ordinary Python and
                    # declared a surface this function could not see.
                    if isinstance(st, ast.AnnAssign) and st.value is not None:
                        targets = [st.target]
                    elif isinstance(st, ast.Assign):
                        targets = st.targets
                    else:
                        continue
                    if dflt_name(st.value) in GLOBALS:
                        for target in targets:
                            if isinstance(target, ast.Name):
                                found.setdefault(node.name, set()).add(target.id)
                # A CONSTRUCTOR's parameter defaults belong to the CLASS, not to `__init__`.
                # Keying on `node.name` registered surface `__init__`, so every `Surf(...)` call site
                # was invisible -- the "loses the surface and with it every call site" failure this
                # file calls its worst -- and it planted a phantom `__init__` surface that would
                # score rows on any literal `x.__init__(...)`. So it mis-answers rather than staying
                # silent, which is the same shape as the posonly slice.
                #
                # 22 classes in src/ already define `__init__`/`__new__` with parameter defaults
                # (LocalShowdownEnv, EngineEnv, _ReplayParser, ...) against 0 positional-only
                # functions -- and round 8 treated the posonly case as blocking on the grounds that
                # "has not landed" is no defence. This one has landed 22 times.
                for statement in node.body:
                    if not isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        continue
                    if statement.name not in CONSTRUCTOR_NAMES:
                        continue
                    a = statement.args
                    positional = a.posonlyargs + a.args
                    ctor_pairs = (
                        list(zip(positional[-len(a.defaults):], a.defaults)) if a.defaults else []
                    )
                    ctor_pairs += [
                        (k, d) for k, d in zip(a.kwonlyargs, a.kw_defaults) if d is not None
                    ]
                    for arg, default in ctor_pairs:
                        if dflt_name(default) in GLOBALS:
                            found.setdefault(node.name, set()).add(arg.arg)
            # ast.Lambda is DELIBERATELY EXCLUDED. Round 8 added it, claiming the lambda
            # parameter-default spelling was "fixed and pinned"; both halves were false. `ast.Lambda`
            # has no `.name`, so `found.setdefault(node.name, ...)` raised AttributeError -- and
            # `SURFACES` is built at import, so ONE `f = lambda spec=GLOBAL: ...` anywhere in src/
            # killed the script in every mode and turned the gate into "ledger derivation failed".
            # There was no probe for it either. An anonymous callable has no call-site NAME for
            # `sites_in` to match, so there is nothing to derive; excluding it is the fix, not
            # including it. Pinned below by a negative that asserts the ledger still runs.
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Constructors are handled above, attributed to their CLASS. Without this skip they
                # ALSO register under their own name, leaving a phantom `__init__` surface that would
                # score a row on any literal `x.__init__(...)` -- so the first half of this fix
                # produced the right answer and kept the wrong one alongside it.
                if node.name in CONSTRUCTOR_NAMES:
                    continue
                a = node.args
                # `posonlyargs + args`, because `ast.arguments.defaults` covers BOTH, combined.
                # Slicing `a.args` alone misaligns the pairing, and this is the one escape in this
                # file that does not merely under-count -- it names the WRONG kwarg:
                #   def f(spec=GLOBAL, /, other=1)   ->  derived {'other'}
                # so a call site scores CLOSED the moment it passes `other=`, which closes nothing,
                # while `spec` is positional-only and can never be closed by keyword at all. The
                # other three shapes lose the surface outright. 0 posonly functions exist in src/
                # today (4208 scanned), but "has not landed" is not a defence for a matcher that
                # actively mis-answers rather than one that stays silent.
                positional = a.posonlyargs + a.args
                pairs = (
                    list(zip(positional[-len(a.defaults):], a.defaults)) if a.defaults else []
                )
                pairs += [(k, d) for k, d in zip(a.kwonlyargs, a.kw_defaults) if d is not None]
                for arg, d in pairs:
                    if dflt_name(d) in GLOBALS:
                        found.setdefault(node.name, set()).add(arg.arg)
    for owner, aliases in EXTRA_CONSTRUCTORS.items():
        if owner not in found:
            raise SystemExit(
                f"ledger: {owner} no longer defaults to the global default; its EXTRA_CONSTRUCTORS "
                "entry is stale and the count would silently drift."
            )
        for alias in aliases:
            found[alias] = found[owner]
    return found


SURFACES = derive_surfaces()
# The file that DEFINES the default necessarily reads it; definition sites are not conflation.
DEFINITION_SITES = {"src/pokezero/observation.py"}


def tracked_py() -> list[Path]:
    out = subprocess.run(
        ["git", "-C", str(REPO), "ls-files", "*.py"], capture_output=True, text=True, check=True
    ).stdout.split()
    return [REPO / p for p in out]


def enclosing(tree: ast.AST) -> dict[int, str]:
    """line -> INNERMOST enclosing def/class name, so a row is addressable after line drift.

    Innermost, not outermost. The first version used `ast.walk` (breadth-first) with
    `setdefault`, which locked in the OUTERMOST scope and contradicted this docstring: every
    method of a TestCase collapsed onto the class name, so
    one TestCase's single key covered 54 separate call sites as
    one key. 202 rows collapsed to 87 distinct keys -- 115 rows, 57%, invisible to any
    key-based comparison. Assigning unconditionally in depth order makes the innermost scope win.
    """
    owner: dict[int, str] = {}

    def visit(node: ast.AST) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            for ln in range(node.lineno, (node.end_lineno or node.lineno) + 1):
                # Unconditional assignment is enough: pre-order visits the shallowest scope
                # first, so a deeper one always overwrites it. A first cut carried a module-level
                # `_DEPTH` dict keyed on `(id(tree), lineno)` to compare depths -- dead weight
                # (the short-circuit meant the depth was never actually read), ~10 MB leaked per
                # scan, and keyed on the `id()` of trees that get freed and reused: 23 id-reuse
                # events across 524 files. Removing the guard clause it hid behind mis-owned 8
                # files, which is how it looked load-bearing.
                owner[ln] = node.name
        for child in ast.iter_child_nodes(node):
            visit(child)

    visit(tree)
    return owner


def sites_in(path: Path) -> list[dict]:
    rel = str(path.relative_to(REPO))
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError, FileNotFoundError, IsADirectoryError) as exc:
        # Loud, not skipped: an unparsed file is an unmeasured file, and an unmeasured file is
        # exactly how a denominator goes wrong.
        return [{"file": rel, "line": 0, "kind": "UNPARSED", "owner": type(exc).__name__}]
    owner = enclosing(tree)
    found: list[dict] = []

    # Per-file ALIAS map. `from pokezero.observation import OBSERVATION_SCHEMA_VERSION as SV`
    # made every `SV` read invisible, and `import pokezero.observation as O` made every
    # `O.OBSERVATION_SCHEMA_VERSION` invisible. Both were demonstrated to add default reads with
    # N unchanged and the gate green -- the same defect class as the any-of bug: a denominator
    # blind to a spelling. Resolved here so the kind reported is the GLOBAL, not the local name.
    alias_to_global: dict[str, str] = {}
    # Names bound in this file that a pokezero module (or the package) can be reached THROUGH.
    # Matched by ROOT segment, not by whole dotted path: `import pokezero` binds only `pokezero`,
    # yet `pokezero.observation.OBSERVATION_SCHEMA_VERSION` is then a legal read, so requiring the
    # full base to have been imported by name is itself a spelling hole.
    module_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for a in node.names:
                if a.name in (CONST, DEFAULT_SPEC) and a.asname:
                    alias_to_global[a.asname] = a.name
                elif node.level > 0 or (node.module and node.module.startswith("pokezero")):
                    # `from pokezero import observation [as O]`, and `from . import observation`.
                    #
                    # `node.level > 0` FIRST, because a relative import has `node.module is None`
                    # -- so `startswith("pokezero")` never fired and every read through a
                    # relatively-imported module was invisible. That was still an enumeration
                    # from memory, this time of ABSOLUTE import syntax, three lines below a
                    # comment disclaiming exactly that. And the idiom is live inside the package
                    # (src/pokezero/linear_policy.py:24, src/pokezero/selfplay.py:17), a stronger
                    # occurrence record than either spelling found in round 6.
                    #
                    # Any relative import counts: inside this package, `from . import X` and
                    # `from .. import X` can only resolve to pokezero modules.
                    #
                    # But ONLY if the imported name is actually a MODULE. `from pokezero.observation
                    # import ObservationSpec` binds a CLASS, and registering it made
                    # `ObservationSpec.OBSERVATION_SCHEMA_VERSION` score a row -- an OVER-match,
                    # which inflates the denominator and is the same failure as under-matching: the
                    # figure stops meaning what it says. Resolved against the filesystem rather
                    # than by guessing from the name's case or depth, so the answer is derived.
                    if is_pokezero_submodule(node, a.name, importer=path):
                        module_roots.add(a.asname or a.name)
        elif isinstance(node, ast.Import):
            for a in node.names:
                # `import pokezero` -- no dot -- binds the package, which re-exports
                # OBSERVATION_SCHEMA_VERSION via __all__. (An earlier version of this comment said
                # "both globals"; DEFAULT_REPLAY_OBSERVATION_SPEC is neither in __all__ nor an
                # attribute of the package -- `hasattr(pokezero, ...)` is False. The narrower claim
                # is the true one.) The previous `startswith("pokezero.")` test registered NOTHING
                # for the bare form, making the public spelling the one the ledger could not see.
                if a.name == "pokezero" or a.name.startswith("pokezero."):
                    module_roots.add(a.asname or a.name.split(".")[0])

    def add(node, kind, unclosed=None):
        row = {"file": rel, "line": node.lineno, "kind": kind,
               "owner": owner.get(node.lineno, "<module>")}
        if unclosed:
            # Which default-bearing kwarg is still unnamed. Without this a row says "this call
            # reaches a default" without saying through which of several routes.
            row["unclosed"] = unclosed
        found.append(row)

    for node in ast.walk(tree):
        # `ctx=Load` -- only a READ is a read. Without it the ASSIGNMENT TARGET at
        # `showdown.py:1143` (`DEFAULT_REPLAY_OBSERVATION_SPEC = ...`) scored a `default-spec` row,
        # so N was 391 where the truth is 390. That contradicted three of this file's own claims at
        # once: the docstring says `default-spec` is "a read of" the global; DEFINITION_SITES exists
        # because "the file that DEFINES the default necessarily reads it" but names only
        # observation.py, so the file defining the OTHER global got no exemption; and the doc
        # insists over-matching is the same failure as under-matching. Exactly 2 non-Load
        # occurrences of either global exist across all 524 tracked files -- observation.py:77
        # (excluded) and showdown.py:1143 (was counted) -- so the over-count was exactly 1.
        #
        # Note the SAME line also carries a legitimate `bare-const` row: the subscript reads
        # OBSERVATION_SCHEMA_VERSION. Only the Store target was spurious.
        if isinstance(node, ast.Name) and not isinstance(node.ctx, ast.Load):
            continue
        if isinstance(node, ast.Name) and rel not in DEFINITION_SITES and (
            node.id == CONST or alias_to_global.get(node.id) == CONST
        ):
            add(node, "bare-const")
        elif isinstance(node, ast.Name) and rel not in DEFINITION_SITES and (
            node.id == DEFAULT_SPEC or alias_to_global.get(node.id) == DEFAULT_SPEC
        ):
            add(node, "default-spec")
        elif (
            isinstance(node, ast.Attribute)
            and node.attr in (CONST, DEFAULT_SPEC)
            and base_root(node.value) in module_roots
            and rel not in DEFINITION_SITES
        ):
            add(node, "bare-const" if node.attr == CONST else "default-spec")
        elif isinstance(node, ast.Call):
            fn = node.func
            name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", None)
            kwargs = {k.arg for k in node.keywords if k.arg}
            # EACH-OF, not any-of. `SURFACES[name] & kwargs` scored a call safe if it passed ANY
            # of the surface's default-bearing kwargs, so a `compact_category(numeric_feature_
            # count=..., ...)` that never named a schema counted as migrated while still taking
            # the process-wide default. TWO figures, which are easy to conflate and were:
            #   43 sites were hidden by the any-of bug; 41 of those 43 pin a WIDTH and default the
            #      SCHEMA -- precisely the shape of #1227 (token_count) and #1228 (the widths).
            #   127 of ALL 390 rows have only a SCHEMA route open, under either kwarg name
            #      (41 compact_category + 49 PokeZeroObservationV0 + 34 ObservationSpec +
            #      3 LinearPolicyModel). Of those, 44 are open specifically on
            #      `observation_schema_version`. An earlier comment quoted the 44 while
            #      describing the 127's question -- a true number answering a narrower question
            #      than the one asked, which is the failure this whole ledger exists to retire.
            # A site is only safe once EVERY route to a global is closed.
            unclosed = SURFACES.get(name, frozenset()) - kwargs
            if name in SURFACES and unclosed:
                # One kind per surface so a new surface cannot quietly join an existing bucket.
                add(node, f"implicit:{name}", sorted(unclosed))
    return found


#: One line of prose per non-surface kind, owned HERE rather than in the docstring, so the docstring
#: can be a verbatim copy of this tool's output instead of a hand-maintained parallel table. Adding
#: a kind without adding its description fails loudly in `render_kinds_table` rather than rendering
#: a blank cell.
KIND_DESCRIPTIONS = {
    "bare-const": "a read of `OBSERVATION_SCHEMA_VERSION` itself",
    "default-spec": "a read of `DEFAULT_REPLAY_OBSERVATION_SPEC`",
}


def render_kinds_table(rows: list[dict]) -> str:
    """Render the non-surface kind rows exactly as the module docstring must contain them.

    THE DOCSTRING IS GENERATED, NOT PARSED. Five rounds of review were spent narrowing a regex that
    tried to recognise a hand-written table: each round closed the escapes the previous reviewer
    named and left the next ones open -- 2 reformats, then 17 more (uppercase or capitalised kind
    name, an underscore, a `*` or `-` bullet the table already uses elsewhere, a trailing colon in
    the house style of the very next row, `[16]`, `n=16`, `(16 rows)` which the docstring itself uses
    nine lines below, the count moved to the next line), plus a containment test that masked an
    unreadable row whenever its text was a substring of a readable one, plus last-write-wins on a
    duplicated row so a WRONG count inserted above the right one was silently discarded.

    A grammar cannot win that: every version is one character away from the next escape. So the tool
    emits the block and the test asserts the docstring region is BYTE-EQUAL to it. There is no
    grammar to evade, a reformat is a diff rather than a disappearance, a duplicate row is a diff,
    and a retired kind is a diff. Regenerate with:

        python scripts/schema_default_ledger.py --render-kinds-table
    """
    counts = Counter(r["kind"] for r in rows)
    kinds = sorted(k for k in counts if not k.startswith("implicit:") and k != "UNPARSED")
    missing = [k for k in kinds if k not in KIND_DESCRIPTIONS]
    if missing:
        raise SystemExit(
            f"schema_default_ledger: no KIND_DESCRIPTIONS entry for {missing}. A kind with no "
            "description would render a blank cell and the docstring would document a count "
            "without saying what it counts."
        )
    return "\n".join(
        f"  {k:<20} {KIND_DESCRIPTIONS[k]:<58} ({counts[k]})" for k in kinds
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--by-file", action="store_true")
    ap.add_argument(
        "--render-kinds-table",
        action="store_true",
        help="emit the docstring's kinds table; the docstring must contain this verbatim",
    )
    args = ap.parse_args()

    scanned = tracked_py()
    rows = [r for p in scanned for r in sites_in(p)]
    rows.sort(key=lambda r: (r["file"], r["line"]))

    kinds_all = {r["kind"] for r in rows}
    if args.render_kinds_table:
        print(render_kinds_table(rows))
        return 2 if "UNPARSED" in kinds_all else 0
    if args.json:
        print(json.dumps(rows, indent=2))
        # Exit 2 here too. The FIRST version returned 0 from this branch before the UNPARSED
        # check below, so the one output mode the CI gate actually consumes was the one mode
        # without the loud-failure guarantee -- and the gate duly reported the UNPARSED marker
        # row as a brand-new default reader. The discipline has to hold in every mode or it
        # holds in none.
        return 2 if "UNPARSED" in kinds_all else 0

    kinds: dict[str, int] = {}
    files: dict[str, int] = {}
    for r in rows:
        kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
        files[r["file"]] = files.get(r["file"], 0) + 1

    if args.by_file:
        for f, n in sorted(files.items(), key=lambda x: (-x[1], x[0])):
            print(f"{n:5d}  {f}")
        print()
    else:
        for r in rows:
            print(f"{r['file']}:{r['line']}\t{r['kind']}\t{r['owner']}")
        print()

    print(f"DENOMINATOR: {len(rows)} sites across {len(files)} files "
          f"(scanned {len(scanned)} tracked .py files)")
    for k, n in sorted(kinds.items(), key=lambda x: -x[1]):
        print(f"  {n:5d}  {k}")
    if kinds.get("UNPARSED"):
        print("\nWARNING: unparsed files present -- the denominator is INCOMPLETE.")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
