"""V4 review — CI coverage for ``scripts/v4_boundary_encode_differential.py``.

Three of the four review findings against that gate were about the INSTRUMENT rather than the
encoder, and an instrument with no test is how they survived review in the first place. Each
test here is the kill-confirm for one guard, run in CI so the guard cannot silently rot back:

- ``test_short_run_cannot_pass_the_exit_criterion`` (finding B2): the previous gate printed
  ``PASS states=4`` and exited 0 for ``--games 1 --seed 41 --max-steps 2``, because its only
  precondition was ``accumulator_states_reached > 0`` over an OR of nine columns. The verdict
  now enforces the plan's ``>=200 games / ~20k states`` minimums and PER-COLUMN reachability.
- ``test_ledger_aggregates_over_the_whole_run_not_a_head_slice`` (finding B1, first half): the
  previous report was ``mismatches[:40]``, a FIFO head cap, so every reported row came from the
  first offending game and the run-level picture was whatever game 0 happened to do.
- ``test_ledger_attributes_categorical_columns_by_name`` (finding B1, second half):
  ``detail["columns"]`` was computed only for ``numeric_features``, so a categorical divergence
  was reported as bare "categorical_ids differ". Guessing what that meant is what produced the
  PR's wrong claim that "the species categorical differs" -- the diverging column is
  CATEGORY_TYPE_1.
- ``test_native_build_identity_distinguishes_binaries`` (finding B4): the previous artifact read
  ``pokezero_search.ENGINE_BUILD_FINGERPRINT``, which does not exist, so the recorded value was
  always ``null`` and the recorded module was a constant venv path -- identical for a fresh
  wheel and a stale one, which is the exact failure plan §3 exists to prevent.

The gate itself needs a built Showdown checkout plus the native wheel, so the full-sweep test
skips without them (the ``scripts/expected_stat_gate.py`` pattern from #1073). The ledger and
build-identity tests need neither and always run.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import v4_boundary_encode_differential as gate  # noqa: E402

from ._showdown_root import showdown_root  # noqa: E402


def _ledger() -> gate._MismatchLedger:
    return gate._MismatchLedger(
        numeric_columns={"NUMERIC_BASE_HP": 0, "NUMERIC_EXPECTED_HP": 1},
        categorical_columns={"CATEGORY_SPECIES": 0, "CATEGORY_TYPE_1": 1},
    )


def _pair(shape, index, left_value, right_value, dtype=numpy.float64):
    left = numpy.zeros(shape, dtype=dtype)
    right = numpy.zeros(shape, dtype=dtype)
    left[index] = left_value
    right[index] = right_value
    return left, right


class MismatchLedgerTest(unittest.TestCase):
    def test_ledger_aggregates_over_the_whole_run_not_a_head_slice(self) -> None:
        """Every game contributes to the counts, and late-game modes still get an example.

        The head-cap defect is only visible with MORE offending states than the report cap, so
        the fixture deliberately exceeds it: 200 states in game 0 with one signature, then one
        state in game 199 with a different one. A ``[:40]`` report shows the first signature 40
        times and never mentions the second.
        """
        ledger = _ledger()
        for step in range(200):
            left, right = _pair((3, 2), (0, 0), 0.24, 0.0)
            ledger.record(
                game=0, step=step, player="p1", requested_seat=True,
                differing=[("numeric_features", left, right)],
            )
        left, right = _pair((3, 2), (1, 1), 0.5, 0.25)
        ledger.record(
            game=199, step=7, player="p2", requested_seat=False,
            differing=[("numeric_features", left, right)],
        )
        payload = ledger.payload()

        self.assertEqual(payload["mismatched_states"], 201)
        self.assertEqual(
            payload["column_mismatch_states"],
            {"numeric_features:NUMERIC_BASE_HP": 200, "numeric_features:NUMERIC_EXPECTED_HP": 1},
        )
        self.assertEqual(payload["mismatch_games"], [0, 199])
        self.assertEqual(
            payload["mismatch_states_by_seat_kind"], {"non_requested": 1, "requested": 200}
        )
        # The late-game signature survives, which a FIFO head slice would have dropped.
        self.assertEqual(payload["distinct_signatures"], 2)
        self.assertIn(
            199, [example["game"] for example in payload["signature_examples"]]
        )

    def test_ledger_attributes_categorical_columns_by_name(self) -> None:
        """A categorical divergence resolves to a column NAME, not "categorical_ids differ"."""
        left, right = _pair((3, 2), (0, 1), 41, 0, dtype=numpy.int32)
        ledger = _ledger()
        ledger.record(
            game=2, step=1, player="p2", requested_seat=True,
            differing=[("categorical_ids", left, right)],
        )
        payload = ledger.payload()
        self.assertEqual(
            payload["column_mismatch_states"], {"categorical_ids:CATEGORY_TYPE_1": 1}
        )
        self.assertNotIn("categorical_ids:CATEGORY_SPECIES", payload["column_mismatch_states"])
        example = payload["signature_examples"][0]
        self.assertEqual(example["first"]["python"], 41)
        self.assertEqual(example["first"]["rust"], 0)

    def test_ledger_attributes_one_dimensional_arrays_by_index(self) -> None:
        """The mask arrays have no column names, so the index is the attribution."""
        left = numpy.zeros(5, dtype=numpy.bool_)
        right = numpy.zeros(5, dtype=numpy.bool_)
        left[3] = True
        ledger = _ledger()
        ledger.record(
            game=0, step=0, player="p1", requested_seat=True,
            differing=[("legal_action_mask", left, right)],
        )
        self.assertEqual(
            ledger.payload()["column_mismatch_states"], {"legal_action_mask:INDEX_3": 1}
        )


class NativeBuildIdentityTest(unittest.TestCase):
    def test_binary_identity_distinguishes_two_binaries(self) -> None:
        """The claim in the name, on two ACTUAL binaries.

        Finding F6: the previous test of this name never compared two artifacts -- it inspected
        the one installed extension and checked the digest was 64 hex characters, which would
        hold for any constant. The property that matters is that the SAME bytes hash the same
        (reinstalling one wheel must not look like a new build) and DIFFERENT bytes do not.
        """
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "one.so"
            second = root / "two.so"
            copy = root / "one-reinstalled.so"
            first.write_bytes(b"\x7fELF fake extension A")
            second.write_bytes(b"\x7fELF fake extension B")
            copy.write_bytes(first.read_bytes())

            a = gate.binary_identity(first)
            b = gate.binary_identity(second)
            again = gate.binary_identity(copy)

            self.assertNotEqual(
                a["extension_sha256"], b["extension_sha256"], "two different binaries must differ"
            )
            self.assertEqual(
                a["extension_sha256"],
                again["extension_sha256"],
                "the same bytes at a different path must hash the same",
            )
            self.assertEqual(a["extension_bytes"], len(b"\x7fELF fake extension A"))

    def test_binary_identity_of_a_missing_artifact_is_null(self) -> None:
        identity = gate.binary_identity(Path("/nonexistent/pokezero_search.so"))
        self.assertIsNone(identity["extension"])
        self.assertIsNone(identity["extension_sha256"])

    def test_native_build_identity_is_derived_from_the_artifact(self) -> None:
        identity = gate.native_build_identity()
        # The compiled extension, not the 143-byte re-export stub: only the .so moves when the
        # crate is rebuilt. ``lib.rs`` compiles in nothing that does.
        self.assertTrue(
            str(identity.get("extension", "")).endswith((".so", ".pyd")),
            f"build identity must point at the compiled extension, got {identity.get('extension')!r}",
        )
        digest = identity.get("extension_sha256")
        self.assertIsInstance(digest, str)
        self.assertEqual(len(str(digest)), 64, f"expected a sha256 hex digest, got {digest!r}")
        self.assertGreater(int(identity["extension_bytes"]), 0)
        # And the crate SOURCE fingerprint, so "source moved but the .so did not" is detectable
        # rather than requiring the reader to remember whether they rebuilt.
        self.assertNotIn(
            "unavailable", str(identity["crate_source_fingerprint"]),
            "the crate source fingerprint must resolve in a source checkout",
        )


class ExitCriterionTest(unittest.TestCase):
    def test_short_run_cannot_pass_the_exit_criterion(self) -> None:
        if not (showdown_root() / "dist" / "sim" / "index.js").exists():
            self.skipTest("needs a BUILT Showdown checkout (dist/sim/index.js) and node")
        try:
            import pokezero_search  # noqa: F401
        except Exception as error:  # pragma: no cover
            self.skipTest(f"needs the native wheel: {error}")
        # The EXACT invocation that the previous gate reported as PASS / exit 0.
        summary = gate.run_gate(showdown_root=showdown_root(), games=1, seed=41, max_steps=2)
        self.assertEqual(summary["verdict"], "FAIL", summary["failures"])
        reasons = " | ".join(summary["failures"])
        self.assertIn("games 1 < required 200", reasons)
        self.assertIn("< required 20000", reasons)
        self.assertIn("never reached", reasons)
        # Finding F3: the liveness failure must name EVERY dead tracker, not fire on "none of
        # them moved". Finding F14: liveness is keyed per (tracker, SEAT), so a two-step run has
        # all eight dead on both seats -- sixteen entries, each naming its seat.
        constant = summary["reachability"]["accumulator_scalars_constant"]
        self.assertEqual(
            len(constant), len(gate.ACCUMULATOR_METADATA_KEYS) * len(gate.PLAYERS)
        )
        self.assertTrue(
            all(entry.endswith("[p1]") or entry.endswith("[p2]") for entry in constant),
            f"liveness entries must name their seat, got {constant}",
        )
        self.assertIn("published accumulator scalars never moved", reasons)
        # ... and it fails for the RIGHT reason: the four states really are byte-identical, so
        # this is the vacuity guard firing and not a masked encoder mismatch.
        self.assertEqual(summary["divergence"]["mismatched_states"], 0)
        self.assertEqual(summary["counts"]["states"], 4)

    def test_a_real_sweep_reaches_the_pack_and_moves_the_accumulators(self) -> None:
        """Reachability and accumulator MOVEMENT are measured, not assumed.

        Kept short enough for CI. It deliberately does not assert PASS: the gate's own exit
        criterion is 200 games, and a 5-game run asserting PASS would be the vacuity this
        review was about. What it asserts is that the instrument's reachability and
        accumulator-variation channels are live -- a frozen tracker would still be byte-parity
        clean on both sides, so movement is the only accumulation property in reach here.
        """
        if not (showdown_root() / "dist" / "sim" / "index.js").exists():
            self.skipTest("needs a BUILT Showdown checkout (dist/sim/index.js) and node")
        try:
            import pokezero_search  # noqa: F401
        except Exception as error:  # pragma: no cover
            self.skipTest(f"needs the native wheel: {error}")
        summary = gate.run_gate(
            showdown_root=showdown_root(), games=5, seed=3, min_games=5, min_states=100
        )
        reach = summary["reachability"]["v4_pack_states_reached"]
        for column in (
            "NUMERIC_SELF_HAZARD_CREDIT",
            # The column an earlier revision excluded from the required set on an unmeasured
            # "0/40850 states" claim (finding F2). Five games reach it; 200 reach it in 1315.
            "NUMERIC_OPP_HAZARD_EXPECTED",
            "NUMERIC_MON_STAYED_VS_ACTIVE",
            "NUMERIC_LAST_DAMAGE_DEALT",
            "NUMERIC_LAST_DAMAGE_TAKEN",
        ):
            self.assertGreater(reach[column], 0, f"5-game sweep never reached {column}")
        # PER-SEAT liveness. This 5-game sweep CANNOT reach 16/16 and asserting it would be false
        # precision: measured at seed 3, four (tracker, seat) pairs stay constant at 5 and 8 games,
        # two at 12. 16/16 is only reached at certification scale -- 220 games at seed 4711,
        # 26,220 states. (An earlier version of this comment cited the deploy status doc as
        # recording that; the doc said 8/8, the POOLED denominator, and it is untracked, so the
        # pointer was wrong in both directions. The number is stated here instead, where it can
        # be checked by re-running the gate.)
        #
        # So this asserts the PROPERTY that distinguishes per-seat keying from pooled, rather than
        # the instance: at least one tracker must have exactly ONE seat constant and the other
        # varied. Pooling cannot produce that shape -- it merges the two into a single set, so a
        # constant seat beside a varying one reads as "varied" and vanishes. Asserting the property
        # survives a change in WHICH tracker happens to be asymmetric; at seed 3 there are two
        # independent witnesses (self_hazard_damage_suffered and opponent_items_removed on p1,
        # mirrored on p2), where naming one would have depended on one sweep's outcome.
        #
        # Worth recording why the old assertion went: it was `constant == []`, and it passed on main
        # only BECAUSE of the pooling bug -- four of sixteen pairs were dead and pooling reported
        # zero. It was satisfied by the defect, not despite it.
        constant = summary["reachability"]["accumulator_scalars_constant"]
        varied = summary["reachability"]["accumulator_scalars_varied"]

        def _by_tracker(entries):
            out: dict[str, set[str]] = {}
            for entry in entries:
                name, _, seat = entry.rpartition("[")
                out.setdefault(name, set()).add(seat.rstrip("]"))
            return out

        constant_seats = _by_tracker(constant)
        varied_seats = _by_tracker(varied)
        asymmetric = sorted(
            name
            for name, seats in constant_seats.items()
            if len(seats) == 1 and len(varied_seats.get(name, set()) - seats) == 1
        )
        self.assertTrue(
            asymmetric,
            "no tracker has exactly one seat constant and the other varied, so this sweep cannot "
            "distinguish per-seat liveness from pooled -- either the sweep changed or liveness "
            f"has gone back to pooling. constant={constant} varied={varied}",
        )
        # And every entry is seat-qualified, so a future pooled regression fails loudly.
        seat_suffixes = tuple(f"[{player}]" for player in gate.PLAYERS)
        for entry in constant + varied:
            self.assertTrue(
                entry.endswith(seat_suffixes),
                f"not seat-qualified with one of {seat_suffixes}: {entry}",
            )
        self.assertGreater(
            summary["reachability"]["v4_pack_categorical_states_reached"][
                "CATEGORY_LAST_USED_MOVE"
            ],
            0,
        )
        # The vacuity channel must be honest about the history surface it cannot reach.
        self.assertGreater(summary["vacuity"]["numeric_columns_never_nonzero_count"], 0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
