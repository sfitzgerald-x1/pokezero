"""A transformed request must not overwrite a Pokemon's own retained move state.

This is the dominant half of `self_moveset_mismatch`: 365 killed decisions in era 59,
44.8% of the construction channel and the largest class there after row 2 closed. All
10,368 world failures were seat p1, zero p2.

THE MECHANISM. `actor_move_states_from_request_history` keeps the most recent
request-known move state per own Pokemon, so direct search can restore PP for a Pokemon
that was active earlier and is now benched. It reads `request["active"][0]["moves"]`,
which is the USABLE moveset -- and while Ditto is transformed that is the COPIED one.

Retaining it was permanent. Gen 3 reverts Transform on switch-out, so a benched Ditto
never appears active again with its own set and no later request refreshes the entry. The
stale row reached `engine_world._move_specs` as Ditto's request-known moveset, was
compared against the root snapshot's `[transform]`, and refused every world with
`request-known move 'bodyslam' is absent from the sampled moveset`.

Failing closed there is CORRECT -- the input really is wrong -- so the fix had to be where
the wrong moveset is minted.

THREE WAYS A FIX HERE GOES WRONG. Each has a test below, and each was reached by
measurement rather than foresight:

1. Comparing against the SAME request's `side.pokemon[].moves` is a no-op. That assumes
   Showdown reports base moves there. It does not on this path: instrumenting the sweep
   showed the copied set in BOTH places. The battle-start request is the only trustworthy
   reference, because it necessarily predates any Transform.
2. ERASING the identity strands the Pokemon with no PP, and `engine_world` then refuses it
   as `self_pp_unknown` -- 7 refusals became 8 on the golden-corpus sweep. Keeping the
   pre-transform snapshot is also right on the merits: the copied moves spend the COPIED
   PP, so a transformed Ditto using Body Slam does not decrement Ditto's own Transform PP.
3. Comparing raw ids calls Return and Hidden Power carriers transformed, because
   `side.pokemon[].moves` carries `return102` and `hiddenpowerice` while
   `active[0].moves[].id` carries the base ids `return` and `hiddenpower`. That regression
   shows up in era counts as fewer retained snapshots -- indistinguishable from the fix
   working. NOTE the gen: gen 3 emits `hiddenpowerice` with NO power suffix (Showdown
   appends the BP only for gen >= 6), so `hiddenpowerice70` is a gen 6+ spelling and would
   be a fiction fixture here. `return102` is the only digit-bearing spelling gen 3 emits.
"""

import unittest

from pokezero.local_showdown import actor_move_states_from_request_history

# Ditto's real set, as the battle-start request reports it.
DITTO_TEAM = (
    ("p1: Ditto", True, ["transform"]),
    ("p1: Swampert", False, ["earthquake", "surf", "icebeam", "protect"]),
)
# The same team as a LATER request reports it while Ditto is transformed: Showdown gives
# the copied set in `side.pokemon[].moves` too, which is why a within-request check fails.
DITTO_TEAM_WHILE_TRANSFORMED = (
    ("p1: Ditto", True, ["bodyslam", "curse", "rest", "shadowball"]),
    ("p1: Swampert", False, ["earthquake", "surf", "icebeam", "protect"]),
)


def _request(active_moves, team_rows):
    """A request in the shape `_public_materialization_state` parses."""

    return {
        "active": [
            {"moves": [{"id": mid, "pp": pp, "maxpp": pp} for mid, pp in active_moves]}
        ],
        "side": {
            "pokemon": [
                {
                    "ident": ident,
                    "details": ident.split(": ", 1)[1],
                    "active": active,
                    "moves": list(moves),
                }
                for ident, active, moves in team_rows
            ]
        },
    }


def _ids(states):
    return {identity: [row["id"] for row in rows] for identity, rows in states.items()}


class TransformedMoveStateRetentionTests(unittest.TestCase):
    def test_a_transformed_request_does_not_overwrite_the_clean_snapshot(self):
        """The era-59 case, and the whole point of the change."""

        clean = _request([("transform", 10)], DITTO_TEAM)
        transformed = _request(
            [("bodyslam", 30), ("curse", 10), ("rest", 10), ("shadowball", 15)],
            DITTO_TEAM_WHILE_TRANSFORMED,
        )

        states = actor_move_states_from_request_history([clean, transformed])

        self.assertEqual(
            _ids(states),
            {"ditto": ["transform"]},
            "Ditto must keep its own pre-transform moveset, not the copied one -- the copied "
            "set reverts on switch-out and engine_world would refuse it forever",
        )

    def test_the_identity_is_kept_not_erased(self):
        """Failure mode 2. Erasing removes the mismatch and creates `self_pp_unknown`.

        Measured on the golden-corpus sweep: erasing turned 7 refusals into 8. So the
        assertion that matters is that PP SURVIVES, not merely that the copied moves are
        gone -- an empty dict would satisfy the test above and still not be a fix.
        """

        clean = _request([("transform", 7)], DITTO_TEAM)
        transformed = _request(
            [("bodyslam", 30), ("curse", 10)], DITTO_TEAM_WHILE_TRANSFORMED
        )

        states = actor_move_states_from_request_history([clean, transformed])

        self.assertIn("ditto", states, "erasing the identity strands it with no PP at all")
        self.assertEqual(states["ditto"][0]["pp"], 7, "the clean PP must survive")

    def test_a_transformed_request_with_no_earlier_clean_one_retains_nothing(self):
        """Nothing is invented when no clean snapshot exists.

        The battle-start request is still the reference, so a copied moveset is recognised
        even on the first request that carries one. Retaining it would be the original bug.
        """

        transformed = _request(
            [("bodyslam", 30), ("curse", 10)], DITTO_TEAM_WHILE_TRANSFORMED
        )
        # Battle start, Swampert active: establishes Ditto's real set as `[transform]`.
        opener = _request(
            [("earthquake", 10), ("surf", 15), ("icebeam", 10), ("protect", 10)],
            (
                ("p1: Ditto", False, ["transform"]),
                ("p1: Swampert", True, ["earthquake", "surf", "icebeam", "protect"]),
            ),
        )

        states = actor_move_states_from_request_history([opener, transformed])

        self.assertNotIn("ditto", states)
        self.assertIn("swampert", states, "the unaffected Pokemon must be unchanged")

    def test_resolved_power_spellings_are_not_read_as_a_copy(self):
        """Failure mode 3. `hiddenpowerice` vs `hiddenpower`, `return102` vs `return`."""

        states = actor_move_states_from_request_history(
            [
                _request(
                    [
                        ("hiddenpower", 24),
                        ("return", 32),
                        ("thunderbolt", 15),
                        ("roar", 20),
                    ],
                    (
                        (
                            "p1: Zapdos",
                            True,
                            # GEN 3 spellings: `hiddenpowerice` with no BP suffix, and
                            # `return102` with one. Using `hiddenpowerice70` here would
                            # test a shape gen 3 never emits.
                            ["hiddenpowerice", "return102", "thunderbolt", "roar"],
                        ),
                    ),
                )
            ]
        )

        self.assertEqual(
            _ids(states), {"zapdos": ["hiddenpower", "return", "thunderbolt", "roar"]}
        )

    def test_subset_not_intersection_so_mew_is_not_reopened(self):
        """A partially-overlapping own set is what separates SUBSET from INTERSECTION.

        Ditto's own set is the single move `transform`, which no opponent carries, so for
        Ditto "not a subset" and "no overlap" coincide -- every other test here is blind to
        the difference. Review demonstrated that an `usable & own` mutant passes the whole
        suite AND the live gate while reopening the bug for the entire Mew population: Mew
        has 7 gen 3 randbats sets whose four own moves overlap what it copies.

        `mew-1-variant-8` is (flamethrower, psychic, softboiled, transform); transformed
        onto a Blissey the copied set shares `softboiled`, so an intersection test sees
        overlap and retains the copy.
        """

        mew_team = (
            (
                "p1: Mew",
                True,
                ["flamethrower", "psychic", "softboiled", "transform"],
            ),
        )
        clean = _request(
            [("flamethrower", 15), ("psychic", 16), ("softboiled", 16), ("transform", 16)],
            mew_team,
        )
        # Transformed onto Blissey: `softboiled` overlaps Mew's own set, the rest does not.
        transformed = _request(
            [("softboiled", 16), ("aromatherapy", 8), ("seismictoss", 32), ("toxic", 16)],
            (
                (
                    "p1: Mew",
                    True,
                    ["softboiled", "aromatherapy", "seismictoss", "toxic"],
                ),
            ),
        )

        states = actor_move_states_from_request_history([clean, transformed])

        self.assertEqual(
            _ids(states),
            {"mew": ["flamethrower", "psychic", "softboiled", "transform"]},
            "an intersection test would see `softboiled` overlap and keep the copied set",
        )

    def test_a_request_without_pp_bearing_active_moves_does_not_clobber(self):
        """The `not moves` guard is load-bearing, and nothing pinned it.

        Review measured 173 no-active-block requests and 6 pp-less `recharge` requests
        across the scenario sweep, and showed that dropping `or not moves` passes the whole
        suite. It must not: a faint force-switch would then write an EMPTY tuple, and
        `_has_self_benched_move_history` reads `set(state.self_move_states)`, so the Pokemon
        counts as PP-known. `engine_world._move_specs` then skips `self_pp_unknown` and falls
        through to CATALOG FULL PP for our own side -- exactly the silent wrongness that
        guard exists to refuse.
        """

        team = (("p1: Snorlax", True, ["bodyslam", "curse", "rest", "shadowball"]),)
        clean = _request([("bodyslam", 21), ("curse", 8), ("rest", 6), ("shadowball", 12)], team)
        # A faint force-switch: no `active` block at all.
        force_switch = {
            "forceSwitch": [True],
            "side": {
                "pokemon": [
                    {
                        "ident": "p1: Snorlax",
                        "details": "Snorlax",
                        "active": True,
                        "condition": "0 fnt",
                        "moves": ["bodyslam", "curse", "rest", "shadowball"],
                    }
                ]
            },
        }

        states = actor_move_states_from_request_history([clean, force_switch])

        self.assertEqual(states["snorlax"][0]["pp"], 21, "the real PP must survive")
        self.assertEqual(len(states["snorlax"]), 4)

    def test_an_unreadable_first_request_still_retains(self):
        """Fail OPEN here, deliberately, and only here.

        This gates retention of player-known information. With no usable battle-start
        reference we cannot tell a copy from a real set, and discarding real PP would be its
        own regression. The fail-CLOSED decision stays with `engine_world`, which still
        refuses any world whose request-known moves are absent from the sampled set.
        """

        no_moves_listed = {
            "active": [{"moves": [{"id": "bodyslam", "pp": 30, "maxpp": 30}]}],
            "side": {"pokemon": [{"ident": "p1: Snorlax", "details": "Snorlax", "active": True}]},
        }

        self.assertEqual(
            _ids(actor_move_states_from_request_history([no_moves_listed])),
            {"snorlax": ["bodyslam"]},
        )

    def test_a_truncated_history_does_not_invert_the_fix(self):
        """The reference must be the PASSED battle-start request, not `requests[0]`.

        This is the failure review called the biggest risk, and it is not a no-op -- it is an
        INVERSION. With a history that begins mid-battle while the Pokemon is transformed,
        deriving the reference from `requests[0]` makes the COPIED set the definition of
        "its own moves", so the copied set is retained and the real one is rejected:

            [transformed, clean] -> {'ditto': ['bodyslam','curse','rest','shadowball']}

        Pre-fix that ordering returned the clean set (last-wins), so getting this wrong is
        strictly worse than not having the fix at all.

        Two real surfaces produce non-battle-start histories: `materialize_scenario_state`
        seeds `_request_history` with a single mid-battle request, and
        `restore_public_materialization` leaves it empty after `_reset`, so any stepped
        search env starts mid-battle.
        """

        battle_start = _request([("transform", 16)], DITTO_TEAM)
        transformed = _request(
            [("bodyslam", 30), ("curse", 10), ("rest", 10), ("shadowball", 15)],
            DITTO_TEAM_WHILE_TRANSFORMED,
        )
        clean_later = _request([("transform", 15)], DITTO_TEAM)

        # History TRUNCATED to start mid-battle, transformed first.
        states = actor_move_states_from_request_history(
            [transformed, clean_later], initial_request=battle_start
        )

        self.assertEqual(
            _ids(states),
            {"ditto": ["transform"]},
            "with the battle-start request supplied, a truncated history must still keep "
            "Ditto's own moveset -- deriving the reference from requests[0] keeps the copy",
        )
        self.assertEqual(states["ditto"][0]["pp"], 15, "the later clean PP must win")

    def test_the_normal_path_is_unchanged(self):
        """The reason this function exists: PP for a Pokemon that is now benched."""

        team = (
            ("p1: Snorlax", True, ["bodyslam", "curse", "rest", "shadowball"]),
            ("p1: Swampert", False, ["earthquake", "surf", "icebeam", "protect"]),
        )
        first = _request(
            [("bodyslam", 30), ("curse", 10), ("rest", 10), ("shadowball", 15)], team
        )
        second = _request(
            [("earthquake", 9), ("surf", 15), ("icebeam", 10), ("protect", 10)],
            (
                ("p1: Snorlax", False, ["bodyslam", "curse", "rest", "shadowball"]),
                ("p1: Swampert", True, ["earthquake", "surf", "icebeam", "protect"]),
            ),
        )

        states = actor_move_states_from_request_history([first, second])

        self.assertEqual(sorted(states), ["snorlax", "swampert"])
        self.assertEqual(
            _ids(states)["snorlax"], ["bodyslam", "curse", "rest", "shadowball"]
        )

    def test_the_most_recent_clean_state_still_wins(self):
        """Two clean requests for the same Pokemon: the later PP must survive."""

        # FULL move lists on both sides. `active[0].moves` is always the complete current
        # moveSlots list; review measured 0 strict-subset pp-bearing requests in 2,219 real
        # ones, so a 1-of-2 fixture would test a shape Showdown does not emit.
        team = (("p1: Snorlax", True, ["bodyslam", "curse"]),)
        early = _request([("bodyslam", 30), ("curse", 10)], team)
        later = _request([("bodyslam", 28), ("curse", 7)], team)

        states = actor_move_states_from_request_history([early, later])

        self.assertEqual([row["pp"] for row in states["snorlax"]], [28, 7])


if __name__ == "__main__":
    unittest.main()
