"""Gates for the oracle-belief arm (value-gap plan §4a / H5).

The arm searches the TRUE hidden state instead of a sampled one, so the plan can
split "belief quality is the frontier" from "search/value is the limitation". Its
entire validity rests on one factual claim about the bridge, which the plan
ASSERTS and this file MEASURES:

    the harness drives both Showdown streams to run FoulPlay at all, so it holds
    both seats' opening `|request|` lines, and the opponent's true six are
    therefore in scope at decision time.

If that claim were false the arm would be unrunnable, and the failure shape would
not be an error -- `TruthWorldBuilder` would refuse, and a fallback to sampled
belief would report a truth arm that never searched the truth. So the pins here
are, in order of what they protect:

* the true opponent's six reach the packer, from bridge state, for BOTH seats;
* the injection happens on EVERY decision (one policy is reused for every seed,
  so a hook left over from the previous battle searches the wrong teams);
* a truth that cannot be built RAISES rather than falling back;
* the flag is refused where it would reach nothing at all.

What is NOT pinned here: that the packed truth is byte-identical to the hidden
state Showdown generated. That equality is the truth-injection census's own
measurement (233/233 decisions on the self half; see `TruthWorldBuilder`), and it
is measured against a live env rather than a synthetic request.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _showdown_root import requires_showdown, showdown_root  # noqa: E402

from pokezero.foulplay_bridge import (  # noqa: E402
    ControlledFoulPlayConfig,
    _ControlledBattleState,
    _OpeningRequestsView,
    _handle_decision_boundary,
    _install_oracle_belief_override,
)


def _request_line(slot: str, *, lead: str, turn_marker: str = "a") -> str:
    """A minimally realistic `|request|` line naming all six of a seat's team.

    Shaped after what `showdown._self_team_from_request` reads: `side.pokemon[]`
    rows with ident/details/condition/moves/ability/item/stats. `turn_marker`
    varies the content so a test can tell the FIRST request for a seat from a
    later one.
    """

    species = [lead, "Blissey", "Skarmory", "Tyranitar", "Claydol", "Gengar"]
    rows = [
        {
            "ident": f"{slot}: {name}",
            "details": f"{name}, L100",
            "condition": "100/100",
            "active": index == 0,
            "moves": [f"move{turn_marker}{index}"],
            "baseAbility": "pressure",
            "item": "leftovers",
            "stats": {"atk": 200, "def": 200, "spa": 200, "spd": 200, "spe": 200},
        }
        for index, name in enumerate(species)
    ]
    return (
        '|request|{"active":[{"moves":[{"id":"tackle","move":"Tackle","pp":35,'
        '"maxpp":35,"target":"normal","disabled":false}]}],'
        '"side":{"name":"x","id":"' + slot + '","pokemon":'
        + __import__("json").dumps(rows)
        + '},"rqid":1}'
    )


def _state_with_both_seats() -> _ControlledBattleState:
    state = _ControlledBattleState(battle_id="b-1", seed=7, format_id="gen3randombattle")
    # Interleaved and with a LATER request per seat, which is the real arrival
    # shape: the view must take the opening one, not the newest.
    state.request_history.extend([
        ("p1", _request_line("p1", lead="Zapdos", turn_marker="a")),
        ("p2", _request_line("p2", lead="Suicune", turn_marker="a")),
        ("p1", _request_line("p1", lead="Blissey", turn_marker="b")),
        ("p2", _request_line("p2", lead="Skarmory", turn_marker="b")),
    ])
    return state


class _StubBuilder:
    """`TruthWorldBuilder`'s two-value contract, without a catalog."""

    def __init__(self, override, failure=None) -> None:
        self.override = override
        self.failure = failure
        self.calls: list[object] = []

    def override_for(self, context):
        self.calls.append(context)
        return self.override, self.failure


class _StubPolicy:
    def __init__(self) -> None:
        self._fixed_override = None
        self.stats = SimpleNamespace(oracle_belief_decisions=0)


def _context(round_index: int = 0) -> SimpleNamespace:
    return SimpleNamespace(
        battle_id="b-1", format_id="gen3randombattle", decision_round_index=round_index
    )


class OpeningRequestsViewTest(unittest.TestCase):
    def test_it_carries_both_seats(self) -> None:
        # THE claim the plan makes. p2 is FoulPlay's seat; its request is the only
        # place the true opponent team exists in this process.
        view = _OpeningRequestsView(_state_with_both_seats())
        self.assertEqual(sorted(view._first_requests), ["p1", "p2"])

    def test_it_takes_the_opening_request_not_the_newest(self) -> None:
        # Showdown reorders side.pokemon[] active-first as the battle runs, so a
        # later snapshot packs the wrong lead into slot zero. Distinguished by
        # content, not by position, so a reversed scan cannot pass.
        view = _OpeningRequestsView(_state_with_both_seats())
        for slot, lead in (("p1", "Zapdos"), ("p2", "Suicune")):
            rows = view._first_requests[slot]["side"]["pokemon"]
            self.assertEqual(rows[0]["ident"], f"{slot}: {lead}")

    def test_non_request_lines_are_ignored(self) -> None:
        state = _state_with_both_seats()
        state.request_history.insert(0, ("p1", "|turn|1"))
        view = _OpeningRequestsView(state)
        self.assertEqual(sorted(view._first_requests), ["p1", "p2"])
        self.assertEqual(
            view._first_requests["p1"]["side"]["pokemon"][0]["ident"], "p1: Zapdos"
        )

    def test_a_seat_that_never_requested_is_simply_absent(self) -> None:
        # Not an exception here: TruthWorldBuilder turns a missing seat into its
        # own "opening request missing or short" failure, which the installer
        # then raises on. One refusal path, not two.
        state = _ControlledBattleState(
            battle_id="b-1", seed=7, format_id="gen3randombattle"
        )
        state.request_history.append(("p1", _request_line("p1", lead="Zapdos")))
        self.assertEqual(sorted(_OpeningRequestsView(state)._first_requests), ["p1"])


class TruthReachesThePackerTest(unittest.TestCase):
    """The adapter is driven by the REAL builder, not asserted to fit it.

    `_OpeningRequestsView` exists only to satisfy `TruthWorldBuilder`'s `env`
    duck-type, and that seam is a `getattr` -- so nothing would fail loudly if the
    shape were wrong. These call the builder's own methods against the view.
    """

    def test_the_real_builder_reads_six_rows_per_seat_from_bridge_state(self) -> None:
        from pokezero.truth_differential import TruthWorldBuilder

        builder = TruthWorldBuilder(
            _OpeningRequestsView(_state_with_both_seats()), set_source=object()
        )
        for slot, lead in (("p1", "Zapdos"), ("p2", "Suicune")):
            rows = builder._opening_rows(slot)
            self.assertIsNotNone(rows, f"{slot} produced no metadata rows")
            self.assertEqual(len(rows), 6, f"{slot} must yield a full team")
            self.assertEqual(rows[0]["species"], lead)
            self.assertEqual(rows[0]["item"], "leftovers")
            self.assertTrue(rows[0]["moves"], "the set's moves must survive")

    def test_the_opponents_true_six_reach_the_packer(self) -> None:
        # One layer past the rows: `packed_teams` hands each seat's rows to
        # `_self_team_from_metadata_result`. Patched to a recorder so the catalog
        # is not needed -- what is under test is that p2's SIX arrive, which is
        # the whole premise of the arm.
        import pokezero.truth_differential as td

        seen: dict[str, int] = {}

        def recorder(rows, *, team_size, set_source):
            del team_size, set_source
            seen[str(rows[0]["showdown_slot"])] = len(rows)
            return (("set",), None)

        with patch.object(td, "_self_team_from_metadata_result", recorder), \
                patch.object(td, "pack_team", lambda team: "packed"):
            packed, failure = td.TruthWorldBuilder(
                _OpeningRequestsView(_state_with_both_seats()), set_source=object()
            ).packed_teams("b-1")
        self.assertIsNone(failure)
        self.assertEqual(packed, {"p1": "packed", "p2": "packed"})
        self.assertEqual(seen, {"p1": 6, "p2": 6})

    def test_a_short_team_is_a_named_failure_not_a_silent_five(self) -> None:
        # The control for the reading above. A builder that returned four rows and
        # packed them would produce a plausible "truth" world missing two mons.
        import pokezero.truth_differential as td

        state = _state_with_both_seats()
        trimmed = __import__("json").loads(
            state.request_history[1][1][len("|request|") :]
        )
        trimmed["side"]["pokemon"] = trimmed["side"]["pokemon"][:4]
        state.request_history[1] = (
            "p2", "|request|" + __import__("json").dumps(trimmed)
        )
        with patch.object(
            td, "_self_team_from_metadata_result", lambda rows, **kw: (("set",), None)
        ), patch.object(td, "pack_team", lambda team: "packed"):
            packed, failure = td.TruthWorldBuilder(
                _OpeningRequestsView(state), set_source=object()
            ).packed_teams("b-1")
        self.assertIsNone(packed)
        self.assertIn("p2", str(failure))


class DynamicBudgetConfigTest(unittest.TestCase):
    """`--engine-early-stop` at the config boundary (dynamic-search-budget plan).

    The stop rule is pre-existing and covered crate-side; these pin the two things
    that were missing and that fail QUIETLY rather than loudly:

    * the flag is refused where it would reach nothing, and
    * a min-sims floor cannot be set without the feature that reads it -- an inert
      knob that looks set is how a cell gets mislabelled as a budget policy it
      never ran.
    """

    def _config(self, **overrides):
        base = dict(
            checkpoint=Path("/tmp/ckpt.pt"),
            showdown_root=Path("/tmp/showdown"),
            policy_mode="engine-mcts",
            engine_model_path=Path("/tmp/model_ts.pt"),
            engine_tables_path=Path("/tmp/tables.json"),
        )
        base.update(overrides)
        return ControlledFoulPlayConfig(**base)

    def test_the_default_is_off(self) -> None:
        cfg = self._config()
        self.assertFalse(cfg.engine_early_stop)
        self.assertIsNone(cfg.engine_early_stop_min_sims)

    def test_engine_mcts_accepts_it(self) -> None:
        cfg = self._config(engine_early_stop=True)
        self.assertTrue(cfg.engine_early_stop)

    def test_it_is_refused_where_it_would_reach_nothing(self) -> None:
        # The stop rule lives in the native search; under raw the flag would reach
        # nothing and the shard would claim a dynamic budget it never had.
        for mode in ("raw", "root-puct"):
            with self.subTest(mode=mode):
                with self.assertRaises(ValueError) as caught:
                    self._config(policy_mode=mode, engine_early_stop=True)
                self.assertIn("engine_early_stop", str(caught.exception))

    def test_a_floor_without_the_feature_is_refused(self) -> None:
        with self.assertRaises(ValueError) as caught:
            self._config(engine_early_stop_min_sims=128)
        self.assertIn("requires engine_early_stop", str(caught.exception))

    def test_a_floor_above_the_budget_is_refused_at_the_cli_boundary(self) -> None:
        # Mirrors EngineMctsConfig's own validator so the refusal lands before the
        # pod claims its GPUs rather than after.
        with self.assertRaises(ValueError):
            self._config(engine_early_stop=True, engine_sims=64,
                         engine_early_stop_min_sims=65)
        with self.assertRaises(ValueError):
            self._config(engine_early_stop=True, engine_early_stop_min_sims=0)

    def test_a_floor_inside_the_budget_is_accepted(self) -> None:
        cfg = self._config(engine_early_stop=True, engine_sims=1024,
                           engine_early_stop_min_sims=1024)
        self.assertEqual(cfg.engine_early_stop_min_sims, 1024)


@requires_showdown("the truth source is only verifiable against real randbat teams")
class TruthSourceAgainstRealBattlesTest(unittest.TestCase):
    """The plan's asserted premise, measured on real battles rather than fixtures.

    Everything above runs on a synthetic request, which proves the plumbing and
    nothing about the DATA. This starts real gen3randombattles on the local
    Showdown checkout and packs the truth two ways:

      A) `TruthWorldBuilder` over the live `LocalShowdownEnv` -- the truth-injection
         census's own input, the one whose self half was measured byte-identical to
         production's over 233 decisions;
      B) `TruthWorldBuilder` over `_OpeningRequestsView(_ControlledBattleState)` --
         the oracle arm's input, built from `|request|` LINES exactly as the bridge
         journals them.

    A == B is the whole verdict: the bridge adapter reproduces the census's truth
    source, so the oracle arm searches the same TRUE world the census does. Not
    self-play-only, because what is compared is the request payload, and the
    bridge journals both seats' payloads for the same reason this env holds them --
    it owns both Showdown streams.
    """

    SEEDS = (11, 12, 13)

    @classmethod
    def setUpClass(cls) -> None:
        from pokezero.local_showdown import LocalShowdownConfig, LocalShowdownEnv
        from pokezero.randbat import load_gen3_randbat_source_cached

        root = showdown_root()
        cls.env = LocalShowdownEnv(
            LocalShowdownConfig(showdown_root=root, set_belief_source=True)
        )
        cls.set_source = load_gen3_randbat_source_cached(root)

    @classmethod
    def tearDownClass(cls) -> None:
        # Explicit: the env's own __del__ closes pipes at interpreter shutdown,
        # which races the finalizer and prints a fatal-looking traceback.
        with __import__("contextlib").suppress(Exception):
            cls.env.close()

    def test_the_bridge_journal_packs_the_same_truth_as_the_live_env(self) -> None:
        from pokezero.truth_differential import TruthWorldBuilder

        for seed in self.SEEDS:
            with self.subTest(seed=seed):
                self.env.reset(seed=seed)
                live = getattr(self.env, "_first_requests")
                self.assertEqual(
                    sorted(live), ["p1", "p2"],
                    "the env holds both seats' opening requests -- the premise",
                )
                state = _ControlledBattleState(
                    battle_id=f"b-{seed}", seed=seed, format_id="gen3randombattle"
                )
                for slot in ("p1", "p2"):
                    state.request_history.append(
                        (slot, "|request|" + json.dumps(live[slot], separators=(",", ":")))
                    )
                from_env, env_failure = TruthWorldBuilder(
                    self.env, set_source=self.set_source
                ).packed_teams(f"live-{seed}")
                from_bridge, bridge_failure = TruthWorldBuilder(
                    _OpeningRequestsView(state), set_source=self.set_source
                ).packed_teams(f"view-{seed}")
                self.assertIsNone(env_failure, "the census's own source refused")
                self.assertIsNone(bridge_failure, "the bridge's journal refused")
                self.assertEqual(
                    from_bridge, from_env,
                    "the bridge adapter is not the census's truth source",
                )
                self.assertEqual(sorted(from_bridge), ["p1", "p2"])
                for slot in ("p1", "p2"):
                    # Showdown's packed format joins six sets with five ']'.
                    self.assertEqual(
                        from_bridge[slot].count("]"), 5,
                        f"{slot} packed fewer than six sets",
                    )

    def test_the_opponent_half_is_not_the_self_half(self) -> None:
        # The control for the equality above: if the builder somehow packed one
        # seat twice, every assertion there would still pass.
        from pokezero.truth_differential import TruthWorldBuilder

        self.env.reset(seed=self.SEEDS[0])
        live = getattr(self.env, "_first_requests")
        state = _ControlledBattleState(
            battle_id="b-x", seed=self.SEEDS[0], format_id="gen3randombattle"
        )
        for slot in ("p1", "p2"):
            state.request_history.append(
                (slot, "|request|" + json.dumps(live[slot], separators=(",", ":")))
            )
        packed, failure = TruthWorldBuilder(
            _OpeningRequestsView(state), set_source=self.set_source
        ).packed_teams("b-x")
        self.assertIsNone(failure)
        self.assertNotEqual(
            packed["p1"], packed["p2"], "both seats packed the same team"
        )


class InstallOracleOverrideTest(unittest.TestCase):
    def _config(self, **overrides) -> ControlledFoulPlayConfig:
        base = dict(
            checkpoint="/tmp/c.pt",
            showdown_root="/tmp/showdown",
            policy_mode="engine-mcts",
            engine_model_path="/tmp/m.pt",
            engine_tables_path="/tmp/t.json",
            engine_oracle_belief=True,
        )
        base.update(overrides)
        return ControlledFoulPlayConfig(**base)

    def test_the_hook_is_set_on_every_decision(self) -> None:
        # The bridge builds ONE policy and reuses it for every seed. Setting the
        # hook once per battle would leave the previous battle's teams installed
        # for the first decisions of the next one -- a confident paired delta
        # computed against the wrong hidden state, reported as a clean run.
        state = _state_with_both_seats()
        policy = _StubPolicy()
        first, second = object(), object()
        state.oracle_truth_builder = _StubBuilder(first)
        _install_oracle_belief_override(
            state, policy, _context(0), config=self._config()
        )
        self.assertIs(policy._fixed_override, first)
        state.oracle_truth_builder = _StubBuilder(second)
        _install_oracle_belief_override(
            state, policy, _context(1), config=self._config()
        )
        self.assertIs(policy._fixed_override, second)

    def test_the_cached_builder_is_reused_rather_than_rebuilt(self) -> None:
        state = _state_with_both_seats()
        builder = _StubBuilder(object())
        state.oracle_truth_builder = builder
        for round_index in range(3):
            _install_oracle_belief_override(
                state, _StubPolicy(), _context(round_index), config=self._config()
            )
        self.assertIs(state.oracle_truth_builder, builder)
        self.assertEqual(len(builder.calls), 3)

    def test_the_applied_counter_moves_once_per_decision(self) -> None:
        # The requested-vs-applied distinction. `engine.oracle_belief` in the
        # summary only says the flag was set; `oracle_belief_decisions` says the
        # truth was installed. Opponent priors is this campaign's own precedent
        # for why one is not the other.
        from pokezero.engine_search import EngineMctsStats

        self.assertEqual(EngineMctsStats().oracle_belief_decisions, 0)
        self.assertIn("oracle_belief_decisions", EngineMctsStats().to_dict())
        state = _state_with_both_seats()
        state.oracle_truth_builder = _StubBuilder(object())
        policy = _StubPolicy()
        for round_index in range(3):
            _install_oracle_belief_override(
                state, policy, _context(round_index), config=self._config()
            )
        self.assertEqual(policy.stats.oracle_belief_decisions, 3)

    def test_a_refusal_does_not_move_the_applied_counter(self) -> None:
        state = _state_with_both_seats()
        state.oracle_truth_builder = _StubBuilder(None, failure="short")
        policy = _StubPolicy()
        with self.assertRaises(RuntimeError):
            _install_oracle_belief_override(
                state, policy, _context(0), config=self._config()
            )
        self.assertEqual(policy.stats.oracle_belief_decisions, 0)

    def test_a_truth_that_cannot_be_built_raises(self) -> None:
        # NOT a fallback. A sampled world here is a sampled-belief decision
        # wearing the oracle arm's config_id, and §4a is read entirely off the
        # difference between those two arms.
        state = _state_with_both_seats()
        state.oracle_truth_builder = _StubBuilder(None, failure="opening request short for p2")
        policy = _StubPolicy()
        with self.assertRaises(RuntimeError) as caught:
            _install_oracle_belief_override(
                state, policy, _context(4), config=self._config()
            )
        self.assertIn("opening request short for p2", str(caught.exception))
        self.assertIn("b-1", str(caught.exception))
        self.assertIsNone(policy._fixed_override, "no partial injection on refusal")

    def test_a_policy_without_the_hook_raises(self) -> None:
        state = _state_with_both_seats()
        state.oracle_truth_builder = _StubBuilder(object())
        with self.assertRaises(RuntimeError) as caught:
            _install_oracle_belief_override(
                state, object(), _context(0), config=self._config()
            )
        self.assertIn("fixed_override", str(caught.exception))


class OracleBeliefConfigTest(unittest.TestCase):
    def test_the_default_is_off(self) -> None:
        import dataclasses

        field = {f.name: f for f in dataclasses.fields(ControlledFoulPlayConfig)}[
            "engine_oracle_belief"
        ]
        self.assertIs(field.default, False)

    def test_it_is_refused_where_it_would_reach_nothing(self) -> None:
        # The hook belongs to EngineMctsPolicy. Under 'raw' or 'root-puct' the
        # flag would be accepted, recorded in the shard, and search sampled
        # worlds -- the exact "flag that reaches nothing" defect this campaign
        # keeps rediscovering after the GPU-hours are spent.
        for mode in ("raw", "root-puct"):
            with self.subTest(policy_mode=mode):
                with self.assertRaises(ValueError) as caught:
                    ControlledFoulPlayConfig(
                        checkpoint="/tmp/c.pt",
                        showdown_root="/tmp/showdown",
                        policy_mode=mode,
                        engine_oracle_belief=True,
                    )
                self.assertIn("engine-mcts", str(caught.exception))

    def test_engine_mcts_accepts_it(self) -> None:
        config = ControlledFoulPlayConfig(
            checkpoint="/tmp/c.pt",
            showdown_root="/tmp/showdown",
            policy_mode="engine-mcts",
            engine_model_path="/tmp/m.pt",
            engine_tables_path="/tmp/t.json",
            engine_oracle_belief=True,
        )
        self.assertTrue(config.engine_oracle_belief)


class OracleBeliefIsReachedFromTheDecisionPathTest(unittest.TestCase):
    """A helper nothing calls is the same defect as a flag nothing forwards.

    Read off the decision boundary's own source: the function is async and takes
    a live BattleStream, a FoulPlay process and a websocket server, so driving it
    here would test the doubles rather than the wiring.
    """

    def test_the_decision_boundary_installs_the_override_under_the_flag(self) -> None:
        source = inspect.getsource(_handle_decision_boundary)
        self.assertIn("if config.engine_oracle_belief:", source)
        self.assertIn("_install_oracle_belief_override(", source)
        # Before the wall-clock boundary: the packing is this arm's instrument,
        # not the search it measures, so charging it to policy_elapsed_seconds
        # would make the oracle arm read slower than its sampled twin by the cost
        # of the instrument -- and §4a compares the two.
        self.assertLess(
            source.index("_install_oracle_belief_override("),
            source.index("pokezero_choice_wall_start = time.perf_counter()"),
        )

    def test_the_shard_summary_witnesses_the_arm(self) -> None:
        # Standing rule: arm identity witnessed from shard telemetry, not job
        # labels. The engine block is built inside a method on a result dataclass,
        # so the key is checked at the source.
        import pokezero.foulplay_bridge as bridge

        self.assertIn(
            '"oracle_belief": self.config.engine_oracle_belief',
            inspect.getsource(bridge),
        )


if __name__ == "__main__":
    unittest.main()
