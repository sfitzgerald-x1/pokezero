"""Schema gates must ask a PROPERTY, not name a version.

`if schema_version == OBSERVATION_SCHEMA_VERSION_V4` is the schema-default conflation wearing a
different hat. It asks "is this the feature-pack schema" and answers by naming one version, so the
day a v5 lands with the same projection the gate is silently False and the caller takes a path
built for a different layout. Nothing fails; the numbers are merely wrong.

That is not hypothetical for this repo: the same shape is what let a v4 config carry v2.2's widths
(#1228) and v2.2's `token_count` (#1227) for two schema generations. Those were caught only when
the default finally moved. An identity gate has no such tripwire at all -- it fails quietly for a
schema that does not exist yet, which means the test has to be the tripwire.

Two things are pinned here:

1. **Equivalence today.** Every converted gate returns exactly what the identity form returned, for
   all five supported schemas. The conversion is a refactor, not a behaviour change, and that is a
   claim a reviewer should not have to take on trust.
2. **Behaviour under a schema that does not exist yet.** A synthetic schema is registered with a
   known property set, and the gates are required to route it by property. This is the half the
   identity form fails, and the only half that justifies the change.
"""
from __future__ import annotations

import re
import unittest

from pokezero import observation as obs
from pokezero.observation import (
    FEATURE_PACK_OBSERVATION_SCHEMA_VERSIONS,
    GROUPED_LAYOUT_OBSERVATION_SCHEMA_VERSIONS,
    OBSERVATION_SCHEMA_VERSION_V2,
    OBSERVATION_SCHEMA_VERSION_V2_1,
    OBSERVATION_SCHEMA_VERSION_V2_2,
    OBSERVATION_SCHEMA_VERSION_V3,
    OBSERVATION_SCHEMA_VERSION_V4,
    SUPPORTED_OBSERVATION_SCHEMA_VERSIONS,
    TURN_MERGED_OBSERVATION_SCHEMA_VERSIONS,
    V2_1_LINEAGE_OBSERVATION_SCHEMA_VERSIONS,
    V3_PROJECTION_OBSERVATION_SCHEMA_VERSIONS,
    ObservationSpec,
    PokeZeroObservationV0,
)


class PropertyTupleShapeTest(unittest.TestCase):
    """The tuples themselves, before anything that reads them."""

    TUPLES = {
        "TURN_MERGED": TURN_MERGED_OBSERVATION_SCHEMA_VERSIONS,
        "GROUPED_LAYOUT": GROUPED_LAYOUT_OBSERVATION_SCHEMA_VERSIONS,
        "FEATURE_PACK": FEATURE_PACK_OBSERVATION_SCHEMA_VERSIONS,
        "V2_1_LINEAGE": V2_1_LINEAGE_OBSERVATION_SCHEMA_VERSIONS,
        "V3_PROJECTION": V3_PROJECTION_OBSERVATION_SCHEMA_VERSIONS,
    }

    def test_every_property_tuple_holds_only_supported_schemas(self) -> None:
        for name, members in self.TUPLES.items():
            with self.subTest(tuple=name):
                unknown = [m for m in members if m not in SUPPORTED_OBSERVATION_SCHEMA_VERSIONS]
                self.assertEqual(
                    unknown, [],
                    f"{name} names {unknown}, which is not in "
                    "SUPPORTED_OBSERVATION_SCHEMA_VERSIONS. A property tuple carrying a retired "
                    "or misspelled version is a gate that silently never fires.",
                )

    def test_no_property_tuple_is_empty_or_holds_duplicates(self) -> None:
        for name, members in self.TUPLES.items():
            with self.subTest(tuple=name):
                self.assertGreater(len(members), 0, f"{name} is empty; every gate reading it is dead")
                self.assertEqual(
                    len(members), len(set(members)), f"{name} holds a duplicate: {members}"
                )

    def test_the_lineage_tuple_is_every_schema_except_v2(self) -> None:
        """Stated as its own set, so this asserts the membership rather than a derivation."""
        self.assertEqual(
            set(V2_1_LINEAGE_OBSERVATION_SCHEMA_VERSIONS),
            set(SUPPORTED_OBSERVATION_SCHEMA_VERSIONS) - {OBSERVATION_SCHEMA_VERSION_V2},
            "the v2.1 lineage is 'every current-state surface that survives a region trim', "
            "which is everything except v2. If a new schema drops those blocks this assertion "
            "SHOULD fail and be updated deliberately.",
        )

    def test_turn_merged_and_feature_pack_are_disjoint(self) -> None:
        """The two axes v4 exists to separate: grouped layout without a transition region."""
        self.assertEqual(
            set(TURN_MERGED_OBSERVATION_SCHEMA_VERSIONS)
            & set(FEATURE_PACK_OBSERVATION_SCHEMA_VERSIONS),
            set(),
            "turn-merged is a property of the TRANSITION REGION and the feature-pack schema has "
            "none, so no schema can be both. A schema in both means the encode gates contradict.",
        )


class GatesMatchTheIdentityFormTodayTest(unittest.TestCase):
    """The refactor half: identical answers for all five supported schemas.

    Deliberately reimplements the OLD expressions verbatim rather than importing anything, so this
    compares two independent statements of the same rule. Reading the new tuples to build the
    expectation would make it tautological.
    """

    def _old_gates(self, schema: str) -> dict[str, bool]:
        v4 = schema == OBSERVATION_SCHEMA_VERSION_V4
        v3 = v4 or schema == OBSERVATION_SCHEMA_VERSION_V3
        turn_merged = (not v4) and (v3 or schema == OBSERVATION_SCHEMA_VERSION_V2_2)
        return {
            "schema_v4": v4,
            "schema_v3": v3,
            "schema_turn_merged": turn_merged,
            "schema_v2_1": turn_merged or v4 or schema == OBSERVATION_SCHEMA_VERSION_V2_1,
            "attention_turn_merged": schema
            in (OBSERVATION_SCHEMA_VERSION_V2_2, OBSERVATION_SCHEMA_VERSION_V3),
            "numeric_index_v4": v4,
            "numeric_index_v3": schema == OBSERVATION_SCHEMA_VERSION_V3,
        }

    def _new_gates(self, schema: str) -> dict[str, bool]:
        return {
            "schema_v4": schema in FEATURE_PACK_OBSERVATION_SCHEMA_VERSIONS,
            "schema_v3": schema in GROUPED_LAYOUT_OBSERVATION_SCHEMA_VERSIONS,
            "schema_turn_merged": schema in TURN_MERGED_OBSERVATION_SCHEMA_VERSIONS,
            "schema_v2_1": schema in V2_1_LINEAGE_OBSERVATION_SCHEMA_VERSIONS,
            "attention_turn_merged": schema in TURN_MERGED_OBSERVATION_SCHEMA_VERSIONS,
            "numeric_index_v4": schema in FEATURE_PACK_OBSERVATION_SCHEMA_VERSIONS,
            "numeric_index_v3": schema in V3_PROJECTION_OBSERVATION_SCHEMA_VERSIONS,
        }

    def test_all_six_converted_gates_agree_with_the_identity_form(self) -> None:
        self.assertEqual(
            len(SUPPORTED_OBSERVATION_SCHEMA_VERSIONS), 5,
            "the denominator moved: this test claims to cover every supported schema, so a new "
            "one must be added to the tables above deliberately rather than silently uncovered.",
        )
        for schema in SUPPORTED_OBSERVATION_SCHEMA_VERSIONS:
            with self.subTest(schema=schema):
                self.assertEqual(
                    self._new_gates(schema), self._old_gates(schema),
                    f"the property form and the identity form disagree for {schema}; the "
                    "conversion is supposed to be behaviour-preserving on today's schema set.",
                )


class GatesRouteASchemaThatDoesNotExistYetTest(unittest.TestCase):
    """The half the identity form fails -- asserted by CALLING production, not by restating it.

    The first version of this class was inert, in two independent ways, and it is worth recording
    both because the second is subtle enough to catch anyone:

    1. It asserted tautologies. `SYNTHETIC in obs.FEATURE_PACK_...` right after appending SYNTHETIC
       to that tuple is true by construction, and `SYNTHETIC != OBSERVATION_SCHEMA_VERSION_V4` is
       true of any string. No production function was ever called.
    2. Even had it called one, the patch was invisible. `showdown.py` does
       `from .observation import (FEATURE_PACK_OBSERVATION_SCHEMA_VERSIONS, ...)`, which binds the
       tuple OBJECT at import. Rebinding `obs.FEATURE_PACK_...` therefore leaves
       `showdown.FEATURE_PACK_...` pointing at the original -- verified: after the patch the two
       names are not even the same object. The old class comment said "re-read the tuple the way
       production does, through the module object, so the patch is visible", which is exactly
       backwards: `obs` is the module production does NOT read.

    Kill-confirmed: reverting the ENTIRE encode-routing change (`git checkout origin/main --
    src/pokezero/showdown.py`) left the old class fully green; only the AST scanner noticed. A test
    whose docstring calls itself "the only reason to make the change" and which passes with the
    change reverted is worse than absent.

    So this version patches `showdown`'s OWN names, registers a spec so the production lookup
    resolves, and compares real routing output against v4's over every legacy index.
    """

    SYNTHETIC = "pokezero.observation.v5-membership-probe"

    def setUp(self) -> None:
        from pokezero import showdown

        self.showdown = showdown
        # Patch the names SHOWDOWN holds, not observation's. Saved and restored per test.
        self._saved = {
            name: getattr(showdown, name)
            for name in (
                "FEATURE_PACK_OBSERVATION_SCHEMA_VERSIONS",
                "V3_PROJECTION_OBSERVATION_SCHEMA_VERSIONS",
                "GROUPED_LAYOUT_OBSERVATION_SCHEMA_VERSIONS",
                "TURN_MERGED_OBSERVATION_SCHEMA_VERSIONS",
                "V2_1_LINEAGE_OBSERVATION_SCHEMA_VERSIONS",
                "REPLAY_OBSERVATION_SPECS_BY_SCHEMA",
            )
        }

    def tearDown(self) -> None:
        for name, value in self._saved.items():
            setattr(self.showdown, name, value)

    def _register(self, *, tuple_name: str, like: str) -> None:
        """Register SYNTHETIC in one property tuple, with `like`'s spec shape."""
        setattr(
            self.showdown, tuple_name, (*self._saved[tuple_name], self.SYNTHETIC)
        )
        self.showdown.REPLAY_OBSERVATION_SPECS_BY_SCHEMA = {
            **self._saved["REPLAY_OBSERVATION_SPECS_BY_SCHEMA"],
            self.SYNTHETIC: self._saved["REPLAY_OBSERVATION_SPECS_BY_SCHEMA"][like],
        }

    def test_a_new_feature_pack_schema_gets_v4s_numeric_projection(self) -> None:
        """The measured failure: under `== V4` a new feature-pack schema takes the LEGACY path.

        Compared over every legacy index rather than a sample, and against v4's own output rather
        than a transcribed table, so the assertion cannot drift from the projection it checks.
        """
        self._register(
            tuple_name="FEATURE_PACK_OBSERVATION_SCHEMA_VERSIONS",
            like=OBSERVATION_SCHEMA_VERSION_V4,
        )
        census = self.showdown.observation_spec_for_schema(
            OBSERVATION_SCHEMA_VERSION_V4
        ).numeric_feature_count
        compared = 0
        for legacy_index in range(-1, 260):
            expected = self._route(OBSERVATION_SCHEMA_VERSION_V4, legacy_index)
            actual = self._route(self.SYNTHETIC, legacy_index)
            self.assertEqual(
                actual, expected,
                f"legacy index {legacy_index}: the synthetic feature-pack schema routed to "
                f"{actual!r} where v4 gives {expected!r}. An identity gate sends it to the "
                "UNPROJECTED legacy path, which returns wrong-but-plausible indices.",
            )
            compared += 1
        self.assertGreater(compared, census, "fewer indices probed than the census; not exhaustive")

    def test_a_new_feature_pack_schema_drops_the_same_columns(self) -> None:
        """`numeric_index_if_present_for_schema` must return None for every v4-dropped column."""
        self._register(
            tuple_name="FEATURE_PACK_OBSERVATION_SCHEMA_VERSIONS",
            like=OBSERVATION_SCHEMA_VERSION_V4,
        )
        dropped = self.showdown.V4_DROPPED_LEGACY_NUMERIC_INDICES
        self.assertGreater(len(dropped), 0, "no dropped columns; this test would be vacuous")
        for legacy_index in sorted(dropped):
            with self.subTest(legacy_index=legacy_index):
                self.assertIsNone(
                    self.showdown.numeric_index_if_present_for_schema(
                        self.SYNTHETIC, legacy_index
                    ),
                    f"legacy index {legacy_index} is dropped at v4 but the synthetic feature-pack "
                    "schema returned an index for it. Under an identity gate the whole dropped "
                    "set returns indices where None is required.",
                )

    def test_a_new_v3_projection_schema_gets_v3s_numeric_projection(self) -> None:
        self._register(
            tuple_name="V3_PROJECTION_OBSERVATION_SCHEMA_VERSIONS",
            like=OBSERVATION_SCHEMA_VERSION_V3,
        )
        for legacy_index in range(-1, 260):
            self.assertEqual(
                self._route(self.SYNTHETIC, legacy_index),
                self._route(OBSERVATION_SCHEMA_VERSION_V3, legacy_index),
                f"legacy index {legacy_index}: the synthetic v3-projection schema diverged from v3",
            )

    def _route(self, schema: str, legacy_index: int):
        """Production's answer, or the exception type+message if it refuses. Never a bare bool."""
        try:
            return ("ok", self.showdown.numeric_index_for_schema(schema, legacy_index))
        except Exception as exc:  # noqa: BLE001 -- the refusal IS the observable
            # Message with the schema name stripped, so two schemas refusing for the same REASON
            # compare equal while a different reason does not.
            return (type(exc).__name__, str(exc).replace(schema, "<schema>"))


class EachGateSiteReadsItsOwnTupleTest(unittest.TestCase):
    """Pin the call-site -> tuple mapping, which nothing else binds.

    `GatesMatchTheIdentityFormTodayTest` restates the mapping inside the test, so it asserts the
    mapping against itself. Review demonstrated the gap: swapping `V3_PROJECTION_...` for
    `GROUPED_LAYOUT_...` at both `numeric_index_for_schema` sites left the new test file green, a
    371-test schema battery green, and the encode output byte-identical -- because the FEATURE_PACK
    branch returns first, so today's schema set makes the two coincide. Behaviour-identical, and yet
    the code would then say "v4 uses the v3 projection", which is the precise claim this change
    exists to deny. Gate reordering was a silent survivor for the same reason.

    Asserted over the AST, because the invariant is about which NAME each site reads -- a runtime
    check cannot distinguish two tuples that happen to agree on the current five schemas.
    """

    # (enclosing function, the tuple each membership test in it must read, how many such tests).
    EXPECTED = {
        "numeric_index_for_schema": {
            "FEATURE_PACK_OBSERVATION_SCHEMA_VERSIONS": 1,
            "V3_PROJECTION_OBSERVATION_SCHEMA_VERSIONS": 1,
        },
        "numeric_index_if_present_for_schema": {
            "FEATURE_PACK_OBSERVATION_SCHEMA_VERSIONS": 1,
            "V3_PROJECTION_OBSERVATION_SCHEMA_VERSIONS": 1,
        },
        "observation_from_player_state": {
            "FEATURE_PACK_OBSERVATION_SCHEMA_VERSIONS": 1,
            "GROUPED_LAYOUT_OBSERVATION_SCHEMA_VERSIONS": 1,
            "TURN_MERGED_OBSERVATION_SCHEMA_VERSIONS": 1,
            "V2_1_LINEAGE_OBSERVATION_SCHEMA_VERSIONS": 1,
        },
        "_attention_mask": {"TURN_MERGED_OBSERVATION_SCHEMA_VERSIONS": 1},
        # PRE-EXISTING on origin/main, not converted by this change -- and found by the
        # completeness test below rather than by me reading the file, which is the point of having
        # it. Worth recording: main already contained exactly one property-tuple gate, so this PR
        # follows an established pattern in its own codebase rather than inventing one.
        "_observation_metadata": {"FEATURE_PACK_OBSERVATION_SCHEMA_VERSIONS": 1},
    }

    def _membership_reads(self) -> dict:
        """{enclosing function -> {tuple name -> count}} for every `<x> in <TUPLE>` in showdown."""
        import ast
        from collections import Counter
        from pathlib import Path

        source = (Path(__file__).resolve().parents[1] / "src/pokezero/showdown.py").read_text(
            encoding="utf-8"
        )
        tree = ast.parse(source)
        owner: dict[int, str] = {}

        def label(node, name):
            for line in range(node.lineno, (node.end_lineno or node.lineno) + 1):
                owner[line] = name

        def walk(node):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                label(node, node.name)
            for child in ast.iter_child_nodes(node):
                walk(child)

        walk(tree)
        reads: dict[str, Counter] = {}
        for node in ast.walk(tree):
            if not isinstance(node, ast.Compare):
                continue
            for op, comparator in zip(node.ops, node.comparators):
                if isinstance(op, ast.In) and isinstance(comparator, ast.Name):
                    if comparator.id.endswith("_OBSERVATION_SCHEMA_VERSIONS"):
                        fn = owner.get(node.lineno, "<module>")
                        reads.setdefault(fn, Counter())[comparator.id] += 1
        return {fn: dict(counter) for fn, counter in reads.items()}

    # Gate VARIABLE -> the tuple it must be assigned from. Counting tuples per function is not
    # enough: swapping which of `schema_v4`/`schema_v3` reads which tuple leaves the per-function
    # counts identical, so that mutant survived the check below until this was added. The reordering
    # is behaviour-visible (55 battery failures) but the new test file must catch it too, or the
    # file is relying on the rest of the suite for its own subject.
    EXPECTED_ASSIGNMENTS = {
        "schema_v4": "FEATURE_PACK_OBSERVATION_SCHEMA_VERSIONS",
        "schema_v3": "GROUPED_LAYOUT_OBSERVATION_SCHEMA_VERSIONS",
        "schema_turn_merged": "TURN_MERGED_OBSERVATION_SCHEMA_VERSIONS",
        "schema_v2_1": "V2_1_LINEAGE_OBSERVATION_SCHEMA_VERSIONS",
    }

    def _gate_assignments(self) -> dict:
        """{variable -> tuple name} for `<var> = <x> in <TUPLE>` anywhere in showdown."""
        import ast
        from pathlib import Path

        source = (Path(__file__).resolve().parents[1] / "src/pokezero/showdown.py").read_text(
            encoding="utf-8"
        )
        found: dict[str, str] = {}
        for node in ast.walk(ast.parse(source)):
            if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Compare):
                continue
            compare = node.value
            for op, comparator in zip(compare.ops, compare.comparators):
                if not (isinstance(op, ast.In) and isinstance(comparator, ast.Name)):
                    continue
                if not comparator.id.endswith("_OBSERVATION_SCHEMA_VERSIONS"):
                    continue
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        found[target.id] = comparator.id
        return found

    def test_each_gate_variable_is_assigned_from_the_tuple_naming_its_property(self) -> None:
        assignments = self._gate_assignments()
        for variable, expected in self.EXPECTED_ASSIGNMENTS.items():
            with self.subTest(variable=variable):
                self.assertEqual(
                    assignments.get(variable), expected,
                    f"`{variable}` is assigned from {assignments.get(variable)!r}, not "
                    f"{expected!r}. Swapping two gates keeps the per-function tuple counts "
                    "identical, so only this assertion sees it.",
                )

    def test_each_gate_site_reads_the_tuple_naming_its_own_property(self) -> None:
        reads = self._membership_reads()
        for function, expected in self.EXPECTED.items():
            with self.subTest(function=function):
                self.assertIn(
                    function, reads,
                    f"{function} performs no property-tuple membership test at all. Either it was "
                    f"renamed, or a gate was reverted to naming a version. Sites found: "
                    f"{sorted(reads)}",
                )
                self.assertEqual(
                    reads[function], expected,
                    f"{function} reads the wrong property tuple(s).\n"
                    f"  expected: {expected}\n  found:    {reads[function]}\n"
                    "Two tuples can agree on today's five schemas and still say different things "
                    "-- pointing the v3-projection gate at GROUPED_LAYOUT is behaviour-identical "
                    "today and asserts that v4 uses v3's projection, which is what this change "
                    "exists to deny.",
                )

    # Which BODY each tuple guards. Counting tuples per function does not see a swap of the two
    # branch bodies with the tuples left in place -- review demonstrated exactly that:
    #     if schema_version in V3_PROJECTION_...:   return v4_numeric_index(...)
    #     if schema_version in FEATURE_PACK_...:    return v3_numeric_index(...)
    # 18 tests / 92 subtests still passed. Per-function counts are unchanged ({FEATURE_PACK: 1,
    # V3_PROJECTION: 1}), these functions assign no gate variables so EXPECTED_ASSIGNMENTS has no
    # purchase, and the synthetic-schema test compares against v4, which moved with it. The file
    # asserted THAT each site reads two named tuples and never WHICH branch each tuple guards --
    # which is the "v4 uses the v3 projection" claim this change exists to deny.
    EXPECTED_BODIES = {
        ("numeric_index_for_schema", "FEATURE_PACK_OBSERVATION_SCHEMA_VERSIONS"): {
            "v4_numeric_index"},
        ("numeric_index_for_schema", "V3_PROJECTION_OBSERVATION_SCHEMA_VERSIONS"): {
            "v3_numeric_index"},
        ("numeric_index_if_present_for_schema", "FEATURE_PACK_OBSERVATION_SCHEMA_VERSIONS"): {
            "V4_DROPPED_LEGACY_NUMERIC_INDICES"},
        ("numeric_index_if_present_for_schema", "V3_PROJECTION_OBSERVATION_SCHEMA_VERSIONS"): {
            "V3_DROPPED_LEGACY_NUMERIC_INDICES", "V4_ONLY_NUMERIC_INDICES"},
    }

    def test_each_tuple_guards_the_body_belonging_to_its_own_projection(self) -> None:
        import ast
        from pathlib import Path

        source = (Path(__file__).resolve().parents[1] / "src/pokezero/showdown.py").read_text(
            encoding="utf-8"
        )
        tree = ast.parse(source)
        found: dict[tuple[str, str], set[str]] = {}
        for function in ast.walk(tree):
            if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for node in ast.walk(function):
                if not isinstance(node, ast.If) or not isinstance(node.test, ast.Compare):
                    continue
                tuples = {
                    c.id
                    for op, c in zip(node.test.ops, node.test.comparators)
                    if isinstance(op, ast.In) and isinstance(c, ast.Name)
                    and c.id.endswith("_OBSERVATION_SCHEMA_VERSIONS")
                }
                if not tuples:
                    continue
                # Every NAME the guarded body mentions, so the assertion is about what the branch
                # reaches rather than about its exact statement shape.
                body_names = {
                    n.id for stmt in node.body for n in ast.walk(stmt) if isinstance(n, ast.Name)
                }
                for tuple_name in tuples:
                    key = (function.name, tuple_name)
                    found.setdefault(key, set()).update(body_names)

        for key, expected in self.EXPECTED_BODIES.items():
            with self.subTest(function=key[0], tuple=key[1]):
                self.assertIn(key, found, f"{key[0]} no longer gates on {key[1]}")
                missing = expected - found[key]
                self.assertEqual(
                    missing, set(),
                    f"the branch guarded by {key[1]} in {key[0]} does not reach {sorted(missing)}. "
                    f"It reaches {sorted(found[key])}. Swapping two branch bodies leaves every "
                    "count and every tuple name unchanged, so only this assertion sees it -- and "
                    "the result asserts that one projection's schemas use the other's indices.",
                )
                # And it must NOT reach the sibling projection's body.
                for other_key, other_expected in self.EXPECTED_BODIES.items():
                    if other_key[0] != key[0] or other_key == key:
                        continue
                    crossed = other_expected & found[key]
                    self.assertEqual(
                        crossed, set(),
                        f"the branch guarded by {key[1]} also reaches {sorted(crossed)}, which "
                        f"belongs to {other_key[1]}. The two projections have been crossed.",
                    )

    def test_no_other_function_gates_on_a_property_tuple_unnoticed(self) -> None:
        """A new gate site must be added to EXPECTED deliberately, not appear silently."""
        unexpected = sorted(set(self._membership_reads()) - set(self.EXPECTED))
        self.assertEqual(
            unexpected, [],
            f"these functions gate on a property tuple but are not pinned above: {unexpected}. "
            "Add them to EXPECTED with the tuple each one should read, so the mapping stays "
            "reviewable rather than growing unchecked.",
        )


class HandBuiltFixturesAreInternallyCoherentTest(unittest.TestCase):
    """The two dataclass stamp defaults.

    `ObservationSpec` computes `token_count` from the v2-family module constants and defaults
    `transition_token_count` to the v2-family 128, so a hand-built spec has a v2.2 SHAPE whatever
    its `schema_version` says. Stamping it from the global default therefore produces a spec that
    claims one schema and carries another's shape -- and `validate()` accepts it, so the
    incoherence is silent. These pin the stamp to the shape rather than to the default.
    """

    def test_a_hand_built_spec_stamps_the_schema_whose_shape_it_has(self) -> None:
        spec = ObservationSpec(categorical_feature_count=1, numeric_feature_count=1)
        self.assertEqual(
            spec.schema_version, OBSERVATION_SCHEMA_VERSION_V2_2,
            "a hand-built spec must stamp v2.2, the schema whose shape its other defaults "
            "produce, not whichever schema currently holds the default slot.",
        )
        self.assertEqual(
            spec.transition_token_count, obs.TRANSITION_TOKEN_COUNT,
            "the shape and the stamp have to name the same schema, or the coherence this "
            "default exists to provide is accidental. `TRANSITION_TOKEN_COUNT` is the v2-family "
            "region size (128), which is v2.2's -- so the stamp above and this shape agree.",
        )
        self.assertEqual(
            obs.TRANSITION_TOKEN_COUNT, 128,
            "v2.2's region size moved, so 'the stamp matches the shape' no longer follows from "
            "the assertion above and this test has stopped meaning what it says.",
        )

    def test_the_stamped_version_is_still_supported(self) -> None:
        """Naming v2.2 by hand takes on a dependency the old form could not have.

        `OBSERVATION_SCHEMA_VERSION` is by definition supported; a hand-written constant is not.
        If v2.2 ever leaves `SUPPORTED_OBSERVATION_SCHEMA_VERSIONS`, every hand-built fixture in the
        repo fails `require_current_observation_schema` at `validate()` -- and the failure would
        appear in whatever test happened to build a spec, not here. This makes it appear here.
        """
        self.assertIn(
            OBSERVATION_SCHEMA_VERSION_V2_2, SUPPORTED_OBSERVATION_SCHEMA_VERSIONS,
            "the two dataclass stamp defaults name v2.2, which is no longer supported. Every "
            "hand-built ObservationSpec/PokeZeroObservationV0 now fails validate(). Re-point both "
            "defaults at the oldest schema whose SHAPE the other defaults produce -- the stamp must "
            "match the shape, which is why it is not simply the newest supported version.",
        )

    def test_the_observation_stamp_matches_the_spec_stamp(self) -> None:
        """They are one decision: `validate()` refuses a pair that disagrees."""
        spec = ObservationSpec(categorical_feature_count=1, numeric_feature_count=1)
        observation = PokeZeroObservationV0(
            categorical_ids=(),
            numeric_features=(),
            token_type_ids=(),
            attention_mask=(),
            legal_action_mask=(),
        )
        self.assertEqual(
            observation.schema_version, spec.schema_version,
            "the two dataclass stamp defaults drifted apart. `validate()` refuses a mismatched "
            "pair, so moving one without the other converts a silent incoherence into a loud "
            "one -- better, but still wrong. They must move together.",
        )

    def test_neither_stamp_default_is_written_as_the_process_default(self) -> None:
        """Read the SOURCE, because no runtime check can see this.

        The first version of this guard re-pointed `obs.OBSERVATION_SCHEMA_VERSION` and asserted a
        freshly built spec did not follow. It passed with the fix AND with the fix reverted, so it
        was decoration: a dataclass field default is evaluated once at class-definition time, so a
        spec built after the patch still carries the value frozen at import. Proven by reverting
        the default to the global and re-running -- the stamp still read v2.2.

        The invariant is therefore genuinely a property of the source text: these two defaults must
        name a VERSION, not the mutable default. Asserted over the AST rather than by grepping, so
        a comment mentioning the constant cannot satisfy or break it.
        """
        import ast
        from pathlib import Path

        source = Path(obs.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        found: dict[str, str] = {}
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            for statement in node.body:
                if (
                    isinstance(statement, ast.AnnAssign)
                    and isinstance(statement.target, ast.Name)
                    and statement.target.id == "schema_version"
                    and statement.value is not None
                ):
                    found[node.name] = ast.unparse(statement.value)

        self.assertEqual(
            sorted(found), ["ObservationSpec", "PokeZeroObservationV0"],
            f"expected exactly the two hand-buildable dataclasses to default `schema_version`; "
            f"found {sorted(found)}. A third one is either a new fixture surface that needs the "
            "same treatment, or one of these two stopped defaulting it.",
        )
        for owner, expression in sorted(found.items()):
            with self.subTest(dataclass=owner):
                self.assertEqual(
                    expression, "OBSERVATION_SCHEMA_VERSION_V2_2",
                    f"{owner}.schema_version defaults to `{expression}`. It must name the version "
                    "whose SHAPE the other defaults produce. Writing "
                    "`OBSERVATION_SCHEMA_VERSION` here stamps whatever schema currently holds the "
                    "default slot onto a v2.2-shaped object, and validate() accepts it -- so the "
                    "incoherence is silent until something reads the widths.",
                )


class NoNewIdentityGateTest(unittest.TestCase):
    """Converting six gates is a cleanup; keeping them converted is the point.

    Without this, the next routing decision gets written as `== OBSERVATION_SCHEMA_VERSION_V4`
    again and the class is back. Scanned over the AST of the encode module, not by grep, so a
    comment discussing the identity form (this file is full of them, and so is showdown.py) cannot
    trip it and a reformatted comparison cannot hide from it.
    """

    # Where the routing decisions live. Deliberately a short, named list rather than all of src/:
    # a wide scan would sweep in the definition site, the spec tables, and the version-validation
    # helpers, all of which compare a schema to a version LEGITIMATELY. Widening this is a
    # deliberate act, and an unlisted file is stated as a gap below rather than implied to be safe.
    SCANNED = ("src/pokezero/showdown.py",)

    VERSION_CONSTANTS = {
        "OBSERVATION_SCHEMA_VERSION_V2",
        "OBSERVATION_SCHEMA_VERSION_V2_1",
        "OBSERVATION_SCHEMA_VERSION_V2_2",
        "OBSERVATION_SCHEMA_VERSION_V3",
        "OBSERVATION_SCHEMA_VERSION_V4",
        # The MUTABLE DEFAULT belongs in this set, and its absence was the worst gap. This whole
        # file's thesis is that an identity gate is "the schema-default conflation wearing a
        # different hat" -- and `spec.schema_version == OBSERVATION_SCHEMA_VERSION` is that
        # conflation in its purest form, gating on whatever schema currently holds the default
        # slot. The first version of this scanner did not list it.
        "OBSERVATION_SCHEMA_VERSION",
    }

    # The schema namespace. Any string literal containing it is naming a schema, however it is
    # assembled -- which is what catches a concatenation that matches no complete version string.
    VERSION_NAMESPACE = "pokezero.observation."

    # The version STRINGS. A gate can name a schema without naming a constant, and the scanner saw
    # only `ast.Name`, so `== "pokezero.observation.v4"` was invisible.
    VERSION_LITERALS = {
        "pokezero.observation.v2",
        "pokezero.observation.v2.1",
        "pokezero.observation.v2.2",
        "pokezero.observation.v3",
        "pokezero.observation.v4",
    }

    def _identity_comparisons(self, source: str) -> list[tuple[int, str]]:
        """`x == V4`, `x != V3`, and `x in (V2_2, V3)` -- all three spell an identity gate.

        The `in (...)` form matters and is easy to miss: an adversarial review of the rotation
        drill found a real one at `_attention_mask` precisely because that drill's guard matched
        only `== V<n>`. A guard blind to one spelling of the thing it forbids is the same defect
        this whole effort is about.
        """
        import ast

        def names_in(node) -> set[str]:
            """Every version-naming token this operand carries.

            Handles four spellings the first version missed, each of which review demonstrated as
            a NEW identity gate inserted into showdown.py that the scanner did not see:
              `== "pokezero.observation.v4"`     a string literal, not a Name
              `V4 == spec.schema_version`        operands reversed, so it sat in `node.left`
              `== OBSERVATION_SCHEMA_VERSION`    the mutable default, absent from the set
              `== _obs.OBSERVATION_SCHEMA_VERSION_V4`   an Attribute, not a Name
            """
            found: set[str] = set()
            if isinstance(node, ast.Name):
                found.add(node.id)
            elif isinstance(node, ast.Attribute):
                # `mod.OBSERVATION_SCHEMA_VERSION_V4` -- the attr is the version token.
                found.add(node.attr)
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                if node.value in self.VERSION_LITERALS:
                    found.add(node.value)
            elif isinstance(node, (ast.Tuple, ast.List, ast.Set)):
                for element in node.elts:
                    found |= names_in(element)
            return found

        tree = ast.parse(source)

        # Local ALIASES of a version constant. `_V4 = OBSERVATION_SCHEMA_VERSION_V4` then `== _V4`
        # is an identity gate whose token is not in VERSION_CONSTANTS. Collected first so the
        # sweep below can resolve them.
        aliases: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                if isinstance(target, ast.Name) and names_in(node.value) & (
                    self.VERSION_CONSTANTS | self.VERSION_LITERALS
                ):
                    aliases.add(target.id)

        def offends(node) -> bool:
            """Any version token ANYWHERE in this operand, however it is spelled.

            Falls back to `ast.unparse` rather than enumerating node types. The type-by-type version
            missed a subscript (`SUPPORTED[-1]`), a call (`frozenset({V4})`), a concatenation
            (`"pokezero.observation." + "v4"`), and a local alias -- four more spellings after four
            had already been added, which is the point at which enumerating node types stops being
            the right approach.
            """
            text = ast.unparse(node)
            if any(re.search(rf"\b{re.escape(t)}\b", text)
                   for t in self.VERSION_CONSTANTS | aliases):
                return True
            if any(lit in text for lit in self.VERSION_LITERALS):
                return True
            # The literal PREFIX, not only whole version strings. `"pokezero.observation." + "v4"`
            # unparses to two fragments and matches no complete literal, so a concatenation slipped
            # through. Any string mentioning the schema namespace is naming a schema.
            if self.VERSION_NAMESPACE in text:
                return True
            # INDEXING a schema tuple yields one specific version, so it is an identity gate even
            # though the tuple name is the approved form. `in SUPPORTED_...` (no subscript) is
            # legitimate -- "is this a supported schema" -- and must stay unflagged, which is why
            # this tests for a Subscript rather than for the name.
            return any(
                isinstance(inner, ast.Subscript)
                and isinstance(inner.value, ast.Name)
                and inner.value.id.endswith("_OBSERVATION_SCHEMA_VERSIONS")
                for inner in ast.walk(node)
            )

        hits: list[tuple[int, str]] = []
        for node in ast.walk(tree):
            # `is` / `is not` -- ONE CHARACTER from `==`, and because both operands reference the
            # same module constant it is True at runtime, so it is a working identity gate. It was
            # absent from the operator filter entirely.
            if isinstance(node, ast.Compare):
                for op, comparator in zip(node.ops, node.comparators):
                    if not isinstance(
                        op, (ast.Eq, ast.NotEq, ast.Is, ast.IsNot, ast.In, ast.NotIn)
                    ):
                        continue
                    if isinstance(op, (ast.Eq, ast.NotEq, ast.Is, ast.IsNot)):
                        # BOTH operands: `node.left` was never inspected, so writing the comparison
                        # the other way round defeated the scan.
                        if offends(comparator) or offends(node.left):
                            hits.append((node.lineno, ast.unparse(node)))
                    elif offends(comparator):
                        hits.append((node.lineno, ast.unparse(node)))

            # `match schema_version: case "pokezero.observation.v4":` -- no Compare node at all.
            elif isinstance(node, ast.Match):
                for case in node.cases:
                    if offends(case.pattern):
                        hits.append((case.pattern.lineno, f"case {ast.unparse(case.pattern)}"))

            # `.startswith("pokezero.observation.v4")` / `.endswith(...)` -- not a comparison, and
            # a prefix test on a version string is an identity gate with extra steps.
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr in ("startswith", "endswith") and any(
                    offends(arg) for arg in node.args
                ):
                    hits.append((node.lineno, ast.unparse(node)))
        return hits

    def test_the_encode_module_routes_by_property_not_by_version(self) -> None:
        from pathlib import Path

        repo = Path(__file__).resolve().parents[1]
        for relative in self.SCANNED:
            with self.subTest(file=relative):
                path = repo / relative
                self.assertTrue(path.is_file(), f"{relative} is gone; this scan measures nothing")
                hits = self._identity_comparisons(path.read_text(encoding="utf-8"))
                self.assertEqual(
                    hits, [],
                    f"{relative} compares a schema to a version constant:\n  "
                    + "\n  ".join(f"line {line}: {code}" for line, code in hits)
                    + "\n\nRoute on a PROPERTY tuple instead (TURN_MERGED_..., "
                    "GROUPED_LAYOUT_..., FEATURE_PACK_..., V2_1_LINEAGE_..., V3_PROJECTION_...), "
                    "or add a new tuple naming the property this decision actually depends on. "
                    "An identity gate is silently False for every future schema that has the "
                    "property, so the wrong branch is taken with no error.",
                )

    def test_the_scanner_detects_every_spelling_it_claims_to(self) -> None:
        """Kill-confirm the scanner itself, or a green scan proves nothing about the module.

        Renamed from "all three spellings": it covered five, and review then found four MORE that
        it did not cover at all -- each demonstrated as a real new identity gate in showdown.py
        that passed the scan, the 371-test schema battery, and a byte-for-byte encode digest.
        """
        cases = {
            "==": "x = spec.schema_version == OBSERVATION_SCHEMA_VERSION_V4",
            "!=": "x = spec.schema_version != OBSERVATION_SCHEMA_VERSION_V3",
            "in tuple": "x = spec.schema_version in (OBSERVATION_SCHEMA_VERSION_V2_2, "
                        "OBSERVATION_SCHEMA_VERSION_V3)",
            "in list": "x = spec.schema_version in [OBSERVATION_SCHEMA_VERSION_V2]",
            "not in": "x = spec.schema_version not in (OBSERVATION_SCHEMA_VERSION_V4,)",
            "string literal": 'x = spec.schema_version == "pokezero.observation.v4"',
            "operands reversed": "x = OBSERVATION_SCHEMA_VERSION_V4 == spec.schema_version",
            "the MUTABLE DEFAULT": "x = spec.schema_version == OBSERVATION_SCHEMA_VERSION",
            "attribute form": "x = spec.schema_version == _obs.OBSERVATION_SCHEMA_VERSION_V4",
            "literal in a tuple": 'x = spec.schema_version in ("pokezero.observation.v3",)',
            # Round-2 review found these ELEVEN more, four of them demonstrated as live gates
            # injected into showdown.py that left this file at 18 passed / 92 subtests.
            "`is` (True at runtime; one char from ==)":
                "x = spec.schema_version is OBSERVATION_SCHEMA_VERSION_V4",
            "`is not`": "x = spec.schema_version is not OBSERVATION_SCHEMA_VERSION_V3",
            "a local alias of a version constant":
                "_V4 = OBSERVATION_SCHEMA_VERSION_V4\nx = spec.schema_version == _V4",
            "a subscript of the supported tuple":
                "x = spec.schema_version == SUPPORTED_OBSERVATION_SCHEMA_VERSIONS[-1]",
            "a call as the comparator":
                "x = spec.schema_version in frozenset({OBSERVATION_SCHEMA_VERSION_V4})",
            "startswith on a version literal":
                'x = spec.schema_version.startswith("pokezero.observation.v4")',
            "endswith on a version literal":
                'x = spec.schema_version.endswith("pokezero.observation.v4")',
            "match/case on a version literal":
                'match spec.schema_version:\n    case "pokezero.observation.v4":\n        x = 1',
            "a concatenated literal":
                'x = spec.schema_version == "pokezero.observation." + "v4"',
            "reversed with `is`":
                "x = OBSERVATION_SCHEMA_VERSION_V4 is spec.schema_version",
            "the mutable default via `is`":
                "x = spec.schema_version is OBSERVATION_SCHEMA_VERSION",
        }
        for label, source in cases.items():
            with self.subTest(spelling=label):
                self.assertEqual(
                    len(self._identity_comparisons(source)), 1,
                    f"the scanner missed the {label} spelling, so a gate written that way would "
                    f"pass the scan above: {source}",
                )

    def test_the_scanner_does_not_flag_legitimate_comparisons(self) -> None:
        """It must not fire on the property tuples, or it forces the pattern it is enforcing."""
        clean = {
            "property membership": "x = spec.schema_version in TURN_MERGED_OBSERVATION_SCHEMA_VERSIONS",
            "a comment": "# spec.schema_version == OBSERVATION_SCHEMA_VERSION_V4 would be a gate",
            "a string": "x = 'OBSERVATION_SCHEMA_VERSION_V4'",
            "an unrelated equality": "x = spec.schema_version == payload_schema",
            "the supported tuple": "x = spec.schema_version in SUPPORTED_OBSERVATION_SCHEMA_VERSIONS",
        }
        for label, source in clean.items():
            with self.subTest(case=label):
                self.assertEqual(
                    self._identity_comparisons(source), [],
                    f"the scanner flagged a legitimate construct ({label}): {source}",
                )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
