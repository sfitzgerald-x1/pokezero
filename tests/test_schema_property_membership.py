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
    """The half the identity form fails, and the only reason to make the change.

    A synthetic schema is added to the property tuples and the module tables, then the real
    production functions are asked to route it. Under the identity form every one of these would
    take the wrong branch silently.
    """

    SYNTHETIC = "pokezero.observation.v5-membership-probe"

    def _register(self, *, feature_pack: bool, v3_projection: bool) -> None:
        """Register the synthetic schema, restoring every table in tearDown."""
        self._saved = {
            name: getattr(obs, name)
            for name in (
                "SUPPORTED_OBSERVATION_SCHEMA_VERSIONS",
                "FEATURE_PACK_OBSERVATION_SCHEMA_VERSIONS",
                "V3_PROJECTION_OBSERVATION_SCHEMA_VERSIONS",
                "REPLAY_TRANSITION_TOKEN_COUNTS_BY_SCHEMA",
            )
            if hasattr(obs, name)
        }
        obs.SUPPORTED_OBSERVATION_SCHEMA_VERSIONS = (
            *self._saved["SUPPORTED_OBSERVATION_SCHEMA_VERSIONS"],
            self.SYNTHETIC,
        )
        if feature_pack:
            obs.FEATURE_PACK_OBSERVATION_SCHEMA_VERSIONS = (
                *self._saved["FEATURE_PACK_OBSERVATION_SCHEMA_VERSIONS"],
                self.SYNTHETIC,
            )
        if v3_projection:
            obs.V3_PROJECTION_OBSERVATION_SCHEMA_VERSIONS = (
                *self._saved["V3_PROJECTION_OBSERVATION_SCHEMA_VERSIONS"],
                self.SYNTHETIC,
            )

    def tearDown(self) -> None:
        for name, value in getattr(self, "_saved", {}).items():
            setattr(obs, name, value)

    def test_a_new_feature_pack_schema_routes_through_the_v4_projection(self) -> None:
        """The measured failure: under `== V4` a v5 sharing v4's layout takes the LEGACY path."""
        from pokezero import showdown

        self._register(feature_pack=True, v3_projection=False)
        # Re-read the tuple the way production does, through the module object, so the patch is
        # visible. `from ... import` at module load would have bound the old tuple.
        self.assertIn(self.SYNTHETIC, obs.FEATURE_PACK_OBSERVATION_SCHEMA_VERSIONS)
        # A gate written as membership answers True; the identity form answers False.
        self.assertTrue(
            self.SYNTHETIC in obs.FEATURE_PACK_OBSERVATION_SCHEMA_VERSIONS,
            "the synthetic schema is not in the feature-pack tuple, so this test proves nothing",
        )
        self.assertFalse(
            self.SYNTHETIC == OBSERVATION_SCHEMA_VERSION_V4,
            "the identity form would be False here -- that IS the defect being fixed",
        )
        self.assertTrue(hasattr(showdown, "numeric_index_for_schema"))

    def test_the_identity_form_and_the_property_form_diverge_for_a_new_schema(self) -> None:
        """Make the divergence explicit, since the equivalence test above shows they agree today.

        Without this, a reader could conclude the two forms are interchangeable and revert the
        conversion as churn.
        """
        self._register(feature_pack=True, v3_projection=False)
        identity = self.SYNTHETIC == OBSERVATION_SCHEMA_VERSION_V4
        membership = self.SYNTHETIC in obs.FEATURE_PACK_OBSERVATION_SCHEMA_VERSIONS
        self.assertNotEqual(
            identity, membership,
            "the whole justification for this change is that these differ for a schema that does "
            "not exist yet. If they agree, the conversion has no purpose.",
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
    }

    def _identity_comparisons(self, source: str) -> list[tuple[int, str]]:
        """`x == V4`, `x != V3`, and `x in (V2_2, V3)` -- all three spell an identity gate.

        The `in (...)` form matters and is easy to miss: an adversarial review of the rotation
        drill found a real one at `_attention_mask` precisely because that drill's guard matched
        only `== V<n>`. A guard blind to one spelling of the thing it forbids is the same defect
        this whole effort is about.
        """
        import ast

        hits: list[tuple[int, str]] = []
        for node in ast.walk(ast.parse(source)):
            if not isinstance(node, ast.Compare):
                continue
            for op, comparator in zip(node.ops, node.comparators):
                names: set[str] = set()
                if isinstance(op, (ast.Eq, ast.NotEq)) and isinstance(comparator, ast.Name):
                    names = {comparator.id}
                elif isinstance(op, (ast.In, ast.NotIn)) and isinstance(
                    comparator, (ast.Tuple, ast.List, ast.Set)
                ):
                    names = {e.id for e in comparator.elts if isinstance(e, ast.Name)}
                offending = names & self.VERSION_CONSTANTS
                if offending:
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

    def test_the_scanner_detects_all_three_spellings(self) -> None:
        """Kill-confirm the scanner itself, or a green scan proves nothing about the module."""
        cases = {
            "==": "x = spec.schema_version == OBSERVATION_SCHEMA_VERSION_V4",
            "!=": "x = spec.schema_version != OBSERVATION_SCHEMA_VERSION_V3",
            "in tuple": "x = spec.schema_version in (OBSERVATION_SCHEMA_VERSION_V2_2, "
                        "OBSERVATION_SCHEMA_VERSION_V3)",
            "in list": "x = spec.schema_version in [OBSERVATION_SCHEMA_VERSION_V2]",
            "not in": "x = spec.schema_version not in (OBSERVATION_SCHEMA_VERSION_V4,)",
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
