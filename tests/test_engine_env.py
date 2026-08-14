"""Engine-as-environment driver: contract, determinism and anti-leakage.

These tests play REAL gen3 randbats games, so they need the native wheel
(``pokezero_search`` with the env-stepping exports), the patched ``poke_engine``
wheel, and a built Showdown checkout for team generation. Each requirement is
skipped individually so a partial dev environment reports the missing piece
instead of a spurious failure.

What is actually asserted, in the order the value of the env depends on it:

1. The env satisfies the :class:`~pokezero.env.PokeZeroEnv` protocol, plays
   legal actions, and terminates.
2. Seeded determinism — the whole point of the seeded chance sampler.
3. Anti-leakage — an opponent Pokemon that has not appeared in the public
   transcript is absent from the observation.
4. k=0 — the transition region is present in the tensor and entirely masked.
5. The legal mask and the action the env will accept come from one source.
"""

from __future__ import annotations

import json
import random
import subprocess
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

try:
    import pokezero_search
except ModuleNotFoundError:  # pragma: no cover
    pokezero_search = None

try:
    import poke_engine
except ModuleNotFoundError:  # pragma: no cover
    poke_engine = None

from pokezero.env import PokeZeroEnv  # noqa: E402
from pokezero.observation import ObservationFeatureMasks  # noqa: E402
from pokezero.showdown import V2_2_REPLAY_OBSERVATION_SPEC  # noqa: E402

_ENV_EXPORTS = ("env_step", "env_options", "env_battle_over")
_TRANSITION_TOKEN_OFFSET = 23


def _missing() -> str | None:
    if pokezero_search is None:
        return "pokezero_search wheel is not installed"
    for name in _ENV_EXPORTS:
        if not hasattr(pokezero_search, name):
            return f"pokezero_search wheel predates {name} (rebuild the crate)"
    if not hasattr(getattr(pokezero_search, "LeafEncoder", None), "rebased"):
        return "pokezero_search wheel predates LeafEncoder.rebased (rebuild the crate)"
    if poke_engine is None:
        return "poke_engine wheel is not installed"
    try:
        from pokezero.local_showdown import LocalShowdownConfig

        LocalShowdownConfig().resolved_showdown_root()
    except Exception as exc:  # noqa: BLE001
        return f"Showdown checkout unavailable: {exc}"
    return None


SKIP_REASON = _missing()


def _play(env, seed, *, rng_seed=0, max_rounds=400):
    """Play one game with a seeded random-legal policy; return the trajectory."""
    rng = random.Random(rng_seed)
    env.reset(seed=seed)
    trace = []
    rounds = 0
    while env.terminal() is None and rounds < max_rounds:
        requested = env.requested_players()
        if not requested:
            break
        actions = {}
        for player in requested:
            mask = env.legal_actions(player)
            legal = [index for index, ok in enumerate(mask) if ok]
            assert legal, f"{player} has no legal action at round {rounds}"
            actions[player] = rng.choice(legal)
        trace.append((requested, tuple(sorted(actions.items()))))
        env.step(actions)
        rounds += 1
    return trace, env.terminal()


@unittest.skipIf(SKIP_REASON is not None, SKIP_REASON or "")
class EngineEnvTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from pokezero.engine_env import EngineEnv, EngineEnvConfig

        cls.EngineEnv = EngineEnv
        # NAMES v2.2 rather than following the process default. This class is about a transition
        # region that EXISTS and is masked -- `test_k0_leaves_the_transition_region_present_but_masked`
        # asserts `spec.transition_token_count > 0` and slices at `_TRANSITION_TOKEN_OFFSET = 23`,
        # both facts about the v2-lineage 151-token layout. Left unnamed, the config resolves through
        # `engine_env._default_observation_spec()`, which reads DEFAULT_REPLAY_OBSERVATION_SPEC; under
        # a v4 default that gives a schema with NO transition region at all, and the test failed
        # `0 not greater than 0` -- inapplicable rather than wrong. Naming the schema makes the
        # premise explicit and is a no-op while v2.2 is the default.
        cls.config = EngineEnvConfig(
            feature_masks=ObservationFeatureMasks(transition_token_budget=0),
            observation_spec=V2_2_REPLAY_OBSERVATION_SPEC,
        )
        # One env for the read-only assertions; tests that need isolation make
        # their own. Team generation spawns a Node process, so sharing matters.
        cls.env = EngineEnv(cls.config)

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def test_satisfies_the_env_protocol(self):
        self.assertIsInstance(self.env, PokeZeroEnv)

    def test_plays_a_complete_legal_terminating_game(self):
        trace, terminal = _play(self.env, seed=4242)
        self.assertIsNotNone(terminal, "game did not terminate within the round cap")
        self.assertFalse(terminal.capped, "game hit the ply cap instead of a real result")
        self.assertIn(terminal.winner, {"p1", "p2", None})
        self.assertGreater(terminal.turn_count, 1)
        self.assertGreater(len(trace), 5)
        # Every recorded decision had at least one requested seat.
        for requested, _ in trace:
            self.assertTrue(requested)

    def test_same_seed_gives_an_identical_trajectory(self):
        first, first_terminal = _play(self.env, seed=99, rng_seed=7)
        second, second_terminal = _play(self.env, seed=99, rng_seed=7)
        self.assertEqual(first, second, "same seed produced a different trajectory")
        self.assertEqual(first_terminal, second_terminal)

    def test_different_seeds_give_different_trajectories(self):
        # Guards the determinism test above: if the env ignored its seed, that
        # test would pass trivially.
        first, _ = _play(self.env, seed=1001, rng_seed=7)
        second, _ = _play(self.env, seed=1002, rng_seed=7)
        self.assertNotEqual(first, second)

    def test_observation_validates_against_the_spec(self):
        self.env.reset(seed=5)
        spec = self.env._observation_spec()
        for player in ("p1", "p2"):
            observation = self.env.observe(player)
            observation.validate(spec)  # raises on any shape/version drift
            self.assertEqual(observation.perspective.player_id, player)
            self.assertEqual(observation.perspective.showdown_slot, player)
            self.assertEqual(len(observation.legal_action_mask), 9)

    def test_k0_leaves_the_transition_region_present_but_masked(self):
        self.env.reset(seed=6)
        observation = self.env.observe("p1")
        spec = self.env._observation_spec()
        # The region still exists — k=0 is a mask, not a narrower spec.
        self.assertEqual(len(observation.attention_mask), spec.token_count)
        self.assertGreater(spec.transition_token_count, 0)
        transition_attention = observation.attention_mask[_TRANSITION_TOKEN_OFFSET:]
        self.assertTrue(
            not any(transition_attention),
            "transition tokens are attended under transition_token_budget=0",
        )
        # ...and the boundary tokens ARE attended, so the mask is not just all-zero.
        self.assertTrue(any(observation.attention_mask[:_TRANSITION_TOKEN_OFFSET]))

    def test_legal_mask_matches_the_actions_the_env_accepts(self):
        """The mask an agent reads and the option surface the engine enforces
        must be the same set — a widened mask means an unplayable action."""
        env = self.EngineEnv(self.config)
        try:
            rng = random.Random(3)
            env.reset(seed=777)
            checked = 0
            while env.terminal() is None and checked < 400:
                actions = {}
                for player in env.requested_players():
                    mask = {index for index, ok in enumerate(env.legal_actions(player)) if ok}
                    mapped = {
                        index
                        for _, index in env._encoder(player).self_action_map(
                            env._state_str, env._lines or None, True, True
                        )
                        if index is not None
                    }
                    self.assertEqual(
                        mask, mapped, f"{player}: observation mask != engine option surface"
                    )
                    # Every legal action must translate to an engine option.
                    for index in sorted(mask):
                        env._engine_choice(player, index)
                    actions[player] = rng.choice(sorted(mask))
                    checked += 1
                env.step(actions)
        finally:
            env.close()
        self.assertGreater(checked, 10)

    def test_unrevealed_opponent_pokemon_never_reach_the_observation(self):
        """Anti-leakage: the env knows both teams; the observation must not."""
        env = self.EngineEnv(self.config)
        try:
            rng = random.Random(11)
            env.reset(seed=2024)
            true_opponent = {mon.species_key for mon in env._party["p2"]}
            self.assertEqual(len(true_opponent), 6)

            saw_hidden = False
            rounds = 0
            while env.terminal() is None and rounds < 400:
                metadata = env.observe("p1").metadata
                encoded = {
                    _normalize(entry.get("species"))
                    for entry in (metadata.get("opponent_team") or ())
                }
                revealed = set(env._ledger.revealed_species("p2"))
                # Never more than the public transcript has shown...
                self.assertTrue(
                    encoded <= revealed,
                    f"observation leaked unrevealed opponent mons: {sorted(encoded - revealed)}",
                )
                # ...and every encoded mon is genuinely on the opponent's team.
                self.assertTrue(encoded <= true_opponent)
                belief = metadata.get("belief_view") or {}
                belief_species = {
                    _normalize(entry.get("species"))
                    for entry in (belief.get("opponent_pokemon") or ())
                }
                self.assertTrue(
                    belief_species <= revealed,
                    f"belief view leaked: {sorted(belief_species - revealed)}",
                )
                if revealed < true_opponent:
                    saw_hidden = True

                actions = {}
                for player in env.requested_players():
                    legal = [i for i, ok in enumerate(env.legal_actions(player)) if ok]
                    actions[player] = rng.choice(legal)
                env.step(actions)
                rounds += 1

            # The assertions above are vacuous if the whole team was public
            # from turn 1; prove there was something to hide.
            self.assertTrue(saw_hidden, "opponent team was fully revealed immediately")
        finally:
            env.close()

    def test_revealed_moves_accumulate_but_never_exceed_the_truth(self):
        """A revealed opponent move must be one the mon actually has.

        Hidden Power is compared in its PUBLIC spelling: the protocol line is
        ``|move|p2a: X|Hidden Power|`` with no type, so an observer learns the
        move slot was used but not which type it is. Collapsing the truth set
        the same way is the correct comparison, not a loosened one — and
        `test_hidden_power_type_is_not_publicly_revealed` pins that the type
        genuinely does not leak.
        """
        env = self.EngineEnv(self.config)
        try:
            rng = random.Random(5)
            env.reset(seed=31337)
            truth = {
                mon.species_key: {_public_move_id(move) for move in mon.moves}
                for mon in env._party["p2"]
            }
            seen_any = False
            rounds = 0
            while env.terminal() is None and rounds < 400:
                for entry in env.observe("p1").metadata.get("belief_view", {}).get(
                    "opponent_pokemon", ()
                ):
                    species = _normalize(entry.get("species"))
                    moves = {_normalize(move) for move in (entry.get("revealed_moves") or ())}
                    expected = truth.get(species, set())
                    self.assertTrue(
                        moves <= expected,
                        f"{species}: invented revealed moves {sorted(moves - expected)}",
                    )
                    seen_any = seen_any or bool(moves)
                actions = {}
                for player in env.requested_players():
                    legal = [i for i, ok in enumerate(env.legal_actions(player)) if ok]
                    actions[player] = rng.choice(legal)
                env.step(actions)
                rounds += 1
            self.assertTrue(seen_any, "no opponent move was ever revealed across a full game")
        finally:
            env.close()

    def test_hidden_power_type_is_not_publicly_revealed(self):
        """The engine keys Hidden Power by type+BP; the transcript must not."""
        from pokezero.engine_env import _PublicLedger

        ledger = _PublicLedger({"p1": ["swampert"], "p2": ["magneton"]})
        ledger.ingest(["|move|p2a: Magneton|Hidden Power|p1a: Swampert"])
        self.assertEqual(ledger.facts("p2", "magneton").moves, ["hiddenpower"])

    def test_hidden_power_reaches_the_engine_typed_and_stays_playable(self):
        """Regression: the Showdown spelling produces an EMPTY engine move slot.

        A Hidden Power set handed to the engine untyped silently loses the
        move, and a mon whose only move is Hidden Power (gen3 randbats Unown)
        then has no legal action at all.
        """
        from pokezero.engine_env import _generated_mon

        dex = self.env._dex_cached()
        row = {
            "species": "Magneton",
            "moves": ["hiddenpowerice", "thunderbolt"],
            "ability": "magnetpull",
            "item": "leftovers",
            "level": 74,
            "nature": "Serious",
            "gender": "",
            "evs": {stat: 85 for stat in ("hp", "atk", "def", "spa", "spd", "spe")},
            "ivs": _hidden_power_ivs("ice"),
        }
        mon = _generated_mon(row, dex)
        self.assertEqual(mon.moves, ("hiddenpowerice", "thunderbolt"))
        self.assertEqual(len(mon.engine_moves), 2)
        self.assertTrue(
            mon.engine_moves[0].startswith("hiddenpowerice")
            and mon.engine_moves[0] != "hiddenpowerice",
            f"engine id must carry the base power, got {mon.engine_moves[0]!r}",
        )
        self.assertEqual(mon.engine_moves[1], "thunderbolt")

    def test_cosmetic_formes_collapse_only_for_the_engine(self):
        from pokezero.engine_env import _generated_mon

        dex = self.env._dex_cached()
        row = {
            "species": "Unown-K",
            "moves": ["hiddenpowerfighting"],
            "ability": "levitate",
            "item": "",
            "level": 100,
            "nature": "Serious",
            "gender": "",
            "evs": {stat: 85 for stat in ("hp", "atk", "def", "spa", "spd", "spe")},
            "ivs": _hidden_power_ivs("fighting"),
        }
        mon = _generated_mon(row, dex)
        # Display and matching keys keep the forme; only the engine id collapses.
        self.assertEqual(mon.species, "Unown-K")
        self.assertEqual(mon.species_key, "unownk")
        self.assertEqual(mon.engine_id, "unown")

    def test_metadata_hp_fields_track_the_battle(self):
        """`hp_fraction` / `fainted` feed the dataset's potential-shaping terms.

        The native encoder rewrites `condition` but not these derived fields —
        left stale they read "everyone at full HP, nobody fainted" all game and
        silently zero `--hp-delta-return-weight` / `--faint-delta-return-weight`
        instead of failing.
        """
        env = self.EngineEnv(self.config)
        try:
            rng = random.Random(1)
            env.reset(seed=4242)
            saw_damage = False
            saw_faint = False
            rounds = 0
            while env.terminal() is None and rounds < 400:
                for entry in env.observe("p1").metadata["self_team"]:
                    fraction = entry.get("hp_fraction")
                    self.assertIsNotNone(fraction)
                    self.assertGreaterEqual(fraction, 0.0)
                    self.assertLessEqual(fraction, 1.0)
                    # The two fields must agree with each other and with the
                    # condition string the encoder maintains.
                    fainted = bool(entry.get("fainted"))
                    self.assertEqual(fainted, "fnt" in str(entry.get("condition", "")))
                    if fainted:
                        self.assertEqual(fraction, 0.0)
                        saw_faint = True
                    elif fraction < 1.0:
                        saw_damage = True
                actions = {}
                for player in env.requested_players():
                    legal = [i for i, ok in enumerate(env.legal_actions(player)) if ok]
                    actions[player] = rng.choice(legal)
                env.step(actions)
                rounds += 1
            self.assertTrue(saw_damage, "no chip damage ever showed up in hp_fraction")
            self.assertTrue(saw_faint, "no faint ever showed up in the fainted flag")
        finally:
            env.close()

    def test_condition_parsing(self):
        from pokezero.engine_env import _condition_hp

        self.assertEqual(_condition_hp("202/232"), (202 / 232, False))
        self.assertEqual(_condition_hp("150/300 brn"), (0.5, False))
        self.assertEqual(_condition_hp("0 fnt"), (0.0, True))
        self.assertEqual(_condition_hp(""), (None, False))
        self.assertEqual(_condition_hp(None), (None, False))
        self.assertEqual(_condition_hp("garbage"), (None, False))

    def test_step_rejects_missing_and_out_of_range_actions(self):
        from pokezero.engine_env import EngineEnvError

        env = self.EngineEnv(self.config)
        try:
            env.reset(seed=8)
            requested = env.requested_players()
            self.assertEqual(len(requested), 2)
            with self.assertRaises(EngineEnvError):
                env.step({requested[0]: 0})  # missing the other seat
            with self.assertRaises(EngineEnvError):
                env.step({player: 99 for player in requested})
        finally:
            env.close()

    def test_engine_mcts_policies_are_rejected_rather_than_downgraded(self):
        from pokezero.engine_env import EngineEnvUnsupportedError

        env = self.EngineEnv(self.config)
        try:
            with self.assertRaises(EngineEnvUnsupportedError):
                env.public_materialization_state("p1")
        finally:
            env.close()

    def test_rewards_are_zero_until_terminal_then_plus_minus_one(self):
        env = self.EngineEnv(self.config)
        try:
            rng = random.Random(2)
            env.reset(seed=6060)
            result = None
            rounds = 0
            while env.terminal() is None and rounds < 400:
                actions = {}
                for player in env.requested_players():
                    legal = [i for i, ok in enumerate(env.legal_actions(player)) if ok]
                    actions[player] = rng.choice(legal)
                result = env.step(actions)
                if result.terminal is None:
                    self.assertEqual(dict(result.rewards), {"p1": 0.0, "p2": 0.0})
                rounds += 1
            self.assertIsNotNone(result)
            terminal = env.terminal()
            self.assertIsNotNone(terminal)
            if terminal.winner is not None:
                rewards = dict(result.rewards)
                self.assertEqual(rewards[terminal.winner], 1.0)
                loser = "p2" if terminal.winner == "p1" else "p1"
                self.assertEqual(rewards[loser], -1.0)
                self.assertEqual(result.observations, {})
        finally:
            env.close()


def _normalize(value) -> str:
    return "".join(ch for ch in str(value or "").lower() if ch.isalnum())


def _hidden_power_ivs(hp_type: str) -> dict:
    """The generator's IV spread for a Hidden Power type.

    Taken from the same table the randbats spread replication uses, so the
    fixture cannot drift from the sets the bridge actually emits — and so the
    env's fail-closed IV/type consistency check sees a legitimate spread.
    """
    from pokezero.gen3_damage import HIDDEN_POWER_IVS

    ivs = {stat: 31 for stat in ("hp", "atk", "def", "spa", "spd", "spe")}
    ivs.update(HIDDEN_POWER_IVS[hp_type])
    return ivs


def _public_move_id(move_id: str) -> str:
    """A move id as the PUBLIC transcript spells it.

    Hidden Power's type is private: Showdown's protocol emits a bare
    ``Hidden Power``, and the crate's event renderer matches that
    (``events.rs`` collapses ``hiddenpower<type><bp>`` to ``hiddenpower``).
    """
    return "hiddenpower" if move_id.startswith("hiddenpower") else move_id


@unittest.skipIf(pokezero_search is None, "pokezero_search wheel is not installed")
class PublicLedgerTest(unittest.TestCase):
    """The ledger is the anti-leakage gate, so it is tested without a battle."""

    def _ledger(self):
        from pokezero.engine_env import _PublicLedger

        return _PublicLedger({"p1": ["swampert", "zapdos"], "p2": ["blissey", "starmie"]})

    def test_nothing_is_revealed_until_a_line_says_so(self):
        ledger = self._ledger()
        self.assertEqual(ledger.revealed_species("p2"), ())
        self.assertFalse(ledger.is_revealed("p2", "Blissey"))

    def test_switch_reveals_exactly_that_pokemon(self):
        ledger = self._ledger()
        ledger.ingest(["|switch|p2a: Blissey|Blissey, L80, F|100/100"])
        self.assertEqual(ledger.revealed_species("p2"), ("blissey",))
        self.assertFalse(ledger.is_revealed("p2", "Starmie"))

    def test_move_reveals_the_user_and_the_move(self):
        ledger = self._ledger()
        ledger.ingest(["|move|p2a: Starmie|Ice Beam|p1a: Swampert"])
        facts = ledger.facts("p2", "starmie")
        self.assertTrue(facts.revealed)
        self.assertEqual(facts.moves, ["icebeam"])

    def test_called_and_locked_moves_do_not_reveal_a_slot(self):
        """`[from]` executions are not the user's own move slot — the same rule
        the encoder's PP replay follows."""
        ledger = self._ledger()
        ledger.ingest(
            [
                "|move|p2a: Starmie|Surf|p1a: Swampert|[from] Sleep Talk",
                "|move|p2a: Starmie|Thrash|p1a: Swampert|[from]lockedmove",
            ]
        )
        facts = ledger.facts("p2", "starmie")
        self.assertTrue(facts.revealed, "the user is still publicly identified")
        self.assertEqual(facts.moves, [])

    def test_struggle_is_never_recorded_as_a_move(self):
        ledger = self._ledger()
        ledger.ingest(["|move|p2a: Starmie|Struggle|p1a: Swampert"])
        self.assertEqual(ledger.facts("p2", "starmie").moves, [])

    def test_ability_and_item_lines_reveal_traits(self):
        ledger = self._ledger()
        ledger.ingest(
            [
                "|-ability|p2a: Blissey|Natural Cure",
                "|-enditem|p2a: Blissey|Leftovers",
            ]
        )
        facts = ledger.facts("p2", "blissey")
        self.assertEqual(facts.ability, "naturalcure")
        self.assertEqual(facts.item, "leftovers")

    def test_version_advances_only_on_new_information(self):
        ledger = self._ledger()
        start = ledger.version
        ledger.ingest(["|switch|p2a: Blissey|Blissey, L80, F|100/100"])
        after_switch = ledger.version
        self.assertGreater(after_switch, start)
        # Re-ingesting the same disclosure must not churn the encoder cache.
        ledger.ingest(["|switch|p2a: Blissey|Blissey, L80, F|100/100"])
        self.assertEqual(ledger.version, after_switch)

    def test_unknown_and_malformed_lines_are_ignored(self):
        ledger = self._ledger()
        ledger.ingest(["", "|", "|turn|5", "|upkeep", "|move|", "|switch|nonsense"])
        self.assertEqual(ledger.revealed_species("p1"), ())
        self.assertEqual(ledger.revealed_species("p2"), ())


@unittest.skipIf(SKIP_REASON is not None, SKIP_REASON or "")
class EngineStepExportsTest(unittest.TestCase):
    """The native stepping surface, exercised from a real battle state."""

    @classmethod
    def setUpClass(cls):
        from pokezero.engine_env import EngineEnv, EngineEnvConfig

        cls.env = EngineEnv(
            EngineEnvConfig(feature_masks=ObservationFeatureMasks(transition_token_budget=0))
        )
        cls.env.reset(seed=17)

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def test_env_options_reports_both_seats_and_a_live_battle(self):
        import json

        report = json.loads(pokezero_search.env_options(self.env._state_str, True))
        self.assertTrue(report["p1_requested"])
        self.assertTrue(report["p2_requested"])
        self.assertEqual(report["battle_over"], 0.0)
        self.assertTrue(report["p1"])
        self.assertTrue(report["p2"])

    def test_env_battle_over_agrees_with_env_options(self):
        import json

        report = json.loads(pokezero_search.env_options(self.env._state_str, True))
        self.assertEqual(
            pokezero_search.env_battle_over(self.env._state_str), report["battle_over"]
        )

    def test_env_step_is_a_pure_function_of_its_seed(self):
        import json

        report = json.loads(pokezero_search.env_options(self.env._state_str, True))
        args = (self.env._state_str, report["p1"][0], report["p2"][0], self.env._ctx_json(1))
        first = pokezero_search.env_step(*args, 12345, True)
        second = pokezero_search.env_step(*args, 12345, True)
        self.assertEqual(first, second)
        payload = json.loads(first)
        self.assertTrue(payload["events"])
        self.assertGreaterEqual(payload["branch_count"], 1)
        # The post-state must be consumable again — that is what makes the
        # env's stepping loop chainable.
        pokezero_search.env_options(payload["post_state"], True)

    def test_env_step_rejects_a_malformed_state(self):
        with self.assertRaises(ValueError):
            pokezero_search.env_step(
                "not a state", "tackle", "tackle", self.env._ctx_json(1), 1, True
            )




class EncoderTableSchemaSelectionTest(unittest.TestCase):
    """`_load_encoder_tables` must pick the layout matching the observation schema.

    The mapping was `"v3" if schema_version.endswith(".v3") else "v2.2"`, written when v2.2 and v3
    were "the two current layouts". v4 fell through the else and loaded V2.2 TABLES — a wrong
    answer rather than an error. The layouts disagree on numeric width (132 vs 155) and on which
    columns exist, so the failure surfaced deep inside the encoder as a missing column, on the
    first `encode_leaf`, rather than here as an unsupported schema.

    Needs no Showdown checkout: `resolved_showdown_root()` only resolves a path, it does not stat
    one, and the schema decision happens before any artifact is read or built.

    These tests DO touch the filesystem -- several write a tempfile artifact and pass its path, one
    drives a fake repo root through the exporter lane. Two earlier versions of this docstring said
    otherwise ("touch no filesystem", "no test here passes a path"); both were false when written
    and got falser as tests were added. A round-3 commit message claimed the wording had been fixed
    while the diff did not touch it.
    """

    def _selected_schema(self, schema_version: str) -> str:
        """Calls PRODUCTION. The first version of this helper re-derived the mapping in the test
        body, which would have passed with `_load_encoder_tables` still broken."""
        from pokezero.engine_env import encoder_tables_schema

        return encoder_tables_schema(schema_version)

    def test_every_exportable_schema_selects_its_own_layout(self) -> None:
        from pokezero import engine_env

        for schema in sorted(engine_env._EXPORTABLE_TABLE_SCHEMAS):
            with self.subTest(schema=schema):
                self.assertEqual(
                    self._selected_schema(f"pokezero.observation.{schema}"), schema
                )

    def test_v4_does_not_silently_resolve_to_v2_2(self) -> None:
        """The regression, named: this returned "v2.2" for a v4 row."""
        self.assertEqual(self._selected_schema("pokezero.observation.v4"), "v4")

    def test_the_exportable_set_matches_the_exporter_cli(self) -> None:
        """If the exporter grows a schema and this set does not, the new one fails as unsupported.

        Read from the script rather than restated, so the two cannot drift silently.
        """
        import re
        from pathlib import Path as _Path

        from pokezero import engine_env

        script = (
            _Path(__file__).resolve().parents[1] / "scripts" / "export_encoder_tables.py"
        ).read_text(encoding="utf-8")
        # Anchored to the --observation-schema argument, not the first `choices=(` in the file: a
        # future argument with a `("v...")` choices tuple earlier in the script would otherwise
        # silently redirect this comparison.
        block = re.search(
            r'"--observation-schema".*?choices=\((.*?)\)', script, re.S
        )
        self.assertIsNotNone(block, "could not locate the --observation-schema argument")
        match = re.search(r"(\s*\"v[^)]*)", block.group(1))
        self.assertIsNotNone(match, "could not find the exporter's --observation-schema choices")
        advertised = set(re.findall(r'"([^"]+)"', match.group(1)))
        self.assertEqual(
            advertised,
            set(engine_env._EXPORTABLE_TABLE_SCHEMAS),
            "engine_env._EXPORTABLE_TABLE_SCHEMAS has drifted from the exporter's CLI choices",
        )

    def test_the_loader_itself_rejects_an_unbuildable_schema(self) -> None:
        """Exercises `_load_encoder_tables`, not just the helper it calls.

        The first version of these tests only covered `encoder_tables_schema`, so restoring the old
        inline expression at the actual defect site left every one of them green -- the extraction
        that was supposed to cure vacuity moved the tested surface off the changed line. Found in
        review. Hermetic: the schema check runs before any artifact is read or built.
        """
        from pokezero.engine_env import _load_encoder_tables

        with self.assertRaises(ValueError) as caught:
            _load_encoder_tables(None, None, "pokezero.observation.v9")
        self.assertIn("v9", str(caught.exception))

    def test_the_loader_rejects_a_supported_schema_the_exporter_cannot_build(self) -> None:
        """v2.1 and v2 are real observation schemas with no exporter layout.

        These are the interesting unknowns, not `v9`: before the fix they silently resolved to
        v2.2 tables, and they are the inputs whose behaviour this change actually alters.
        """
        from pokezero.engine_env import _load_encoder_tables

        for schema_version in ("pokezero.observation.v2.1", "pokezero.observation.v2"):
            with self.subTest(schema=schema_version):
                with self.assertRaises(ValueError):
                    _load_encoder_tables(None, None, schema_version)

    def test_an_explicitly_passed_artifact_must_match_the_env_schema(self) -> None:
        """The lane production uses (`--engine-encoder-tables`) trusted the file unchecked.

        The layouts differ in numeric width (155 vs 132), categorical count (51 vs 41) and vocab
        size (1217 vs 899 rows), so accepting a mismatched artifact means reading a different table
        through the wrong indices.
        """
        import json
        import tempfile
        from pathlib import Path as _Path

        from pokezero.engine_env import _load_encoder_tables

        payload = json.dumps({"layout": {"schema_version": "pokezero.observation.v2.2"}})
        with tempfile.TemporaryDirectory() as tmp:
            artifact = _Path(tmp) / "encoder_tables_v2.2.json"
            artifact.write_text(payload, encoding="utf-8")
            # Matching schema: accepted, returned verbatim.
            self.assertEqual(
                _load_encoder_tables(artifact, None, "pokezero.observation.v2.2"), payload
            )
            # Mismatched: refused, naming both schemas.
            with self.assertRaises(ValueError) as caught:
                _load_encoder_tables(artifact, None, "pokezero.observation.v4")
            message = str(caught.exception)
            self.assertIn("v2.2", message)
            self.assertIn("v4", message)

    def test_a_mismatch_reports_the_mismatch_even_for_an_unbuildable_env_schema(self) -> None:
        """The error message must not raise while explaining itself.

        The rebuild hint called `encoder_tables_schema(schema_version)` unguarded, which RAISES for
        any schema the exporter cannot build -- so this lane, whose whole purpose is accepting
        tables for such a schema, reported "no encoder-tables layout for v2.1" instead of the
        mismatch: the wrong problem, with an impossible remedy. v2.1 is an advertised
        `rollout_cli.py --observation-schema` choice, so it was reachable. Found in review.
        """
        import json
        import tempfile
        from pathlib import Path as _Path

        from pokezero.engine_env import _load_encoder_tables

        payload = json.dumps({"layout": {"schema_version": "pokezero.observation.v2.2"}})
        with tempfile.TemporaryDirectory() as tmp:
            artifact = _Path(tmp) / "tables.json"
            artifact.write_text(payload, encoding="utf-8")
            # assertRaisesRegex on the mismatch phrasing: the buggy message also contained both
            # "v2.1" and "v2.2" (the latter via "supports ['v2.2', 'v3', 'v4']"), so the two
            # assertIn checks alone passed on the bug and only the assertNotIn discriminated.
            with self.assertRaisesRegex(ValueError, "are for observation schema") as caught:
                _load_encoder_tables(artifact, None, "pokezero.observation.v2.1")
        message = str(caught.exception)
        self.assertIn("v2.2", message)
        self.assertIn("v2.1", message)
        self.assertNotIn("no encoder-tables layout", message)

    def test_a_non_object_artifact_reports_a_clean_error(self) -> None:
        """A JSON list/string reached `.get` and raised AttributeError, escaping the ValueError
        this function installs for exactly that case."""
        import tempfile
        from pathlib import Path as _Path

        from pokezero.engine_env import _load_encoder_tables

        with tempfile.TemporaryDirectory() as tmp:
            for body in ("[1, 2, 3]", '"hello"', "42"):
                artifact = _Path(tmp) / "tables.json"
                artifact.write_text(body, encoding="utf-8")
                with self.subTest(body=body):
                    with self.assertRaises(ValueError) as caught:
                        _load_encoder_tables(artifact, None, "pokezero.observation.v4")
                    self.assertIn("not a JSON object", str(caught.exception))

    def test_both_derived_cache_reads_validate_against_a_cache_this_test_owns(self) -> None:
        """Binds BOTH cache-lane guards without depending on gitignored build output.

        An earlier version of this coverage read the repo's own `corpus/encoder_tables_v2.2.json`,
        so it SKIPPED on any clean checkout -- `corpus/` is gitignored -- leaving both cache guards
        unverified in CI. It also only ever reached the cache-HIT read: deleting the post-build
        guard left the whole suite green, which is the "guards with no guard" state this change's
        own earlier message complained about, at the neighbouring site.

        Here the test owns the repo root, so both branches are reachable and neither depends on
        build output:
          - post-build: `subprocess.run` is patched to write a MISMATCHED artifact.
          - cache-hit: the artifact is present up front, and the exporter must not be invoked.
        """
        import tempfile
        from pathlib import Path as _Path
        from unittest import mock

        from pokezero import engine_env

        wrong = json.dumps({"layout": {"schema_version": "pokezero.observation.v2.2"}})

        for lane in ("post-build", "cache-hit"):
            with self.subTest(lane=lane), tempfile.TemporaryDirectory() as tmp:
                root = _Path(tmp)
                # `_load_encoder_tables` derives the repo root from the module's own __file__.
                (root / "src" / "pokezero").mkdir(parents=True)
                fake_module = root / "src" / "pokezero" / "engine_env.py"
                fake_module.write_text("", encoding="utf-8")
                # The loader refuses to build when the exporter is absent, so the fake root needs
                # one. Empty is enough -- `subprocess.run` is patched and never executes it.
                (root / "scripts").mkdir()
                (root / "scripts" / "export_encoder_tables.py").write_text("", encoding="utf-8")
                cache = root / "corpus" / "encoder_tables_v4.json"
                cache.parent.mkdir(parents=True)

                if lane == "cache-hit":
                    cache.write_text(wrong, encoding="utf-8")

                    def _run(*args, **kwargs):  # pragma: no cover - must not be reached
                        raise AssertionError("cache hit should not invoke the exporter")
                else:
                    def _run(*args, **kwargs):
                        cache.write_text(wrong, encoding="utf-8")
                        return subprocess.CompletedProcess(args, 0, "", "")

                # `subprocess` is imported INSIDE `_load_encoder_tables` (a deliberate lazy
                # import), so `engine_env.subprocess` does not exist as a module attribute --
                # patch the real target instead.
                with mock.patch.object(
                    engine_env, "__file__", str(fake_module)
                ), mock.patch("subprocess.run", _run):
                    # assertRaisesRegex on the MISMATCH phrasing, not a bare ValueError. Dropping
                    # "v4" from _EXPORTABLE_TABLE_SCHEMAS makes the loader raise the
                    # unbuildable-schema error instead, whose text also contains "v2.2" (via
                    # "supports ['v2.2', 'v3']") and "v4" -- so assertRaises plus two assertIns
                    # passed without either cache guard ever being reached. Same soft spot this
                    # change fixed one site over; taking the same fix here rather than shipping the
                    # neighbouring instance again.
                    with self.assertRaisesRegex(
                        ValueError, "are for observation schema"
                    ) as caught:
                        engine_env._load_encoder_tables(
                            None, None, "pokezero.observation.v4"
                        )
                message = str(caught.exception)
                self.assertIn("v2.2", message)
                self.assertIn("v4", message)

    def test_the_validator_accepts_an_artifact_it_cannot_introspect(self) -> None:
        """Absent/blank `layout.schema_version` is accepted DELIBERATELY, and pinned so the
        permissiveness is documented rather than an untested hole in the lane being hardened."""
        from pokezero.engine_env import _assert_tables_match_schema

        for body in ('{}', '{"layout": {}}', '{"layout": {"schema_version": null}}',
                     '{"layout": {"schema_version": ""}}', '{"layout": [1, 2]}'):
            with self.subTest(body=body):
                _assert_tables_match_schema(
                    body, "pokezero.observation.v4", source="<probe>"
                )

    def test_an_unknown_schema_fails_here_rather_than_in_the_encoder(self) -> None:
        with self.assertRaises(ValueError):
            self._selected_schema("pokezero.observation.v9")


if __name__ == "__main__":  # pragma: no cover
    # At the END. My own EncoderTableSchemaSelectionTest was appended BELOW this block, so
    # direct execution never defined it -- the same defect PR #1112 exists to fix, committed
    # into a concurrent branch while writing that fix. #1112's repo-wide guard caught it.
    unittest.main()
