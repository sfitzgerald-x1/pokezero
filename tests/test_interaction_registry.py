"""Validated-interactions registry regression suite (docs/validated_interactions.md).

Drives each registry scenario through the production ``LocalShowdownEnv`` and asserts
the ENCODED observation. The two in-battle-retype ENCODER BUGS (Castform Forecast,
Kecleon Color Change) are wired as ``expectedFailure`` asserting the CORRECT type — the
day the encoder is fixed they flip to xpass and this suite tells you.

The well-formedness tests always run; the encoding tests are gated on a built local
Showdown checkout (same gate as the golden-corpus scenario sweep).
"""

from __future__ import annotations

import os
import random
import shutil
import sys
import unittest
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from pokezero.golden_corpus_scenarios import (  # noqa: E402
    ScenarioSpec,
    ScriptedPreferencePolicy,
    interaction_registry_specs,
)


def _showdown_root() -> str:
    return os.environ.get("POKEZERO_SHOWDOWN_ROOT") or "/Users/scott/workspace/pokerena/vendor/pokemon-showdown"


def _live_showdown_available() -> bool:
    root = Path(_showdown_root())
    return (root / "dist" / "sim" / "index.js").exists() and shutil.which("node") is not None


class RegistryWellFormedTests(unittest.TestCase):
    def test_specs_present_and_unique(self) -> None:
        specs = interaction_registry_specs()
        names = [s.name for s in specs]
        self.assertGreaterEqual(len(specs), 12)
        self.assertEqual(len(names), len(set(names)), "duplicate registry scenario names")
        for spec in specs:
            self.assertTrue(spec.p1_team and spec.p2_team, spec.name)

    def test_flagged_bug_scenarios_present(self) -> None:
        names = {s.name for s in interaction_registry_specs()}
        # The two retype bugs (Castform + Color Change) must stay in the registry.
        self.assertIn("castform_forecast_formechange", names)
        self.assertIn("colorchange_kecleon", names)


@unittest.skipIf(not _live_showdown_available(), "requires a built local Showdown checkout")
class RegistryEncodingTests(unittest.TestCase):
    """Drive each registry scenario and assert the encoded observation."""

    @classmethod
    def setUpClass(cls) -> None:
        from pokezero.local_showdown import LocalShowdownEnv, LocalShowdownConfig
        from pokezero.randbat_vocab import gen3_category_vocabulary
        import pokezero.showdown as S

        cls.S = S
        cls.env = LocalShowdownEnv(LocalShowdownConfig(showdown_root=_showdown_root()))
        cls.vocab = gen3_category_vocabulary(_showdown_root(), include_turn_merged=True)
        cls.specs = {s.name: s for s in interaction_registry_specs()}

    @classmethod
    def tearDownClass(cls) -> None:
        cls.env.close()

    # --- driver -----------------------------------------------------------
    def _drive(self, name: str):
        """Run the scenario; return (per-round observations per player, env, states)."""
        from pokezero.env import BattleStartOverride
        from pokezero.showdown_fixture import pack_team

        spec: ScenarioSpec = self.specs[name]
        override = BattleStartOverride(player_teams={
            "p1": pack_team(tuple(spec.p1_team)),
            "p2": pack_team(tuple(spec.p2_team)),
        })
        self.env.reset_with_start_override(seed=spec.seed, start_override=override)
        pols = {"p1": ScriptedPreferencePolicy(spec.p1_prefs), "p2": ScriptedPreferencePolicy(spec.p2_prefs)}
        obs_by_round = {"p1": [], "p2": []}
        rng = random.Random(0)
        for _ in range(spec.max_decision_rounds):
            requested = self.env.requested_players()
            if not requested:
                break
            actions = {}
            for pl in requested:
                obs = self.env.observe(pl)
                obs_by_round[pl].append(obs)
                actions[pl] = pols[pl].select_action(obs, rng=rng).action_index
            result = self.env.step(actions)
            if getattr(result, "terminal", None) is not None:
                break
        for pl in ("p1", "p2"):
            try:
                obs_by_round[pl].append(self.env.observe(pl))
            except Exception:
                pass
        return obs_by_round

    def _active_token(self, player: str, side: str) -> int:
        """Token index of the active mon on ``side`` ('self'|'opp') for ``player``."""
        S = self.S
        state = self.env._state_for_player(player)
        team = state.self_team if side == "self" else state.opponent_team
        offset = S.SELF_POKEMON_TOKEN_OFFSET if side == "self" else S.OPPONENT_POKEMON_TOKEN_OFFSET
        for idx, mon in enumerate(team):
            if getattr(mon, "active", False):
                return offset + idx
        return offset  # fall back to slot 0

    def _type1(self, obs, token) -> int:
        return obs.categorical_ids[token][self.S.CATEGORY_TYPE_1]

    # --- VERIFIED-CORRECT assertions --------------------------------------
    def test_deoxys_forme_distinct_stats(self) -> None:
        obs = self._drive("deoxys_forme_swap")["p1"][0]
        tok = self.S.SELF_POKEMON_TOKEN_OFFSET  # Deoxys-Attack lead, slot 0
        atk = obs.numeric_features[tok][self.S.NUMERIC_BASE_ATK]
        dfn = obs.numeric_features[tok][self.S.NUMERIC_BASE_DEF]
        # Deoxys-Attack is a glass cannon (base atk 180 -> 0.9, def 20 -> 0.1).
        self.assertGreater(atk, 0.8, "Deoxys-Attack base atk should resolve high")
        self.assertLess(dfn, 0.2, "Deoxys-Attack base def should resolve low")

    def test_intimidate_drops_lead_attack(self) -> None:
        # p2 Salamence Intimidate fires on switch-in -> p1 lead atk -1 at round 0.
        obs = self._drive("intimidate_switchin")["p1"][0]
        tok = self.S.SELF_POKEMON_TOKEN_OFFSET
        self.assertAlmostEqual(obs.numeric_features[tok][self.S.NUMERIC_BOOST_ATK], -1.0 / 6.0, places=4)

    def test_bellydrum_maxes_attack(self) -> None:
        rounds = self._drive("bellydrum_snorlax")["p1"]
        tok = self.S.SELF_POKEMON_TOKEN_OFFSET
        # After Belly Drum resolves, atk is +6 -> 1.0 on a later observation.
        self.assertTrue(
            any(o.numeric_features[tok][self.S.NUMERIC_BOOST_ATK] >= 0.999 for o in rounds),
            "Belly Drum should set atk boost to +6 (1.0)",
        )

    def test_spikes_stack_to_three_layers(self) -> None:
        rounds = self._drive("spikes_stack")["p1"]
        haz = max(o.numeric_features[self.S.FIELD_TOKEN_OFFSET][self.S.NUMERIC_SELF_HAZARDS] for o in rounds)
        self.assertGreaterEqual(haz, 0.999, "3 Spikes layers should read hazards feat 1.0")

    def test_substitute_volatile_encoded(self) -> None:
        rounds = self._drive("substitute_focuspunch")["p1"]
        sub = self.vocab.encode("volatile:substitute")
        tok = self.S.SELF_POKEMON_TOKEN_OFFSET
        seen = any(
            sub in [o.categorical_ids[tok][self.S.CATEGORY_VOLATILE_OFFSET + i]
                    for i in range(self.S.VOLATILE_BUCKET_COUNT)]
            for o in rounds
        )
        self.assertTrue(seen, "Substitute should appear as volatile:substitute")

    def test_sand_stream_weather_permanent(self) -> None:
        rounds = self._drive("sand_stream_permanence")["p1"]
        perm = max(o.numeric_features[self.S.FIELD_TOKEN_OFFSET][self.S.NUMERIC_WEATHER_PERMANENT] for o in rounds)
        self.assertEqual(perm, 1.0, "Sand Stream weather must be permanent in gen3")

    def test_roar_drag_resets_boosts(self) -> None:
        self._drive("roar_drag_reset")
        # After Roar drags the boosted Raikou out for Snorlax, the opponent active
        # mon carries reset (zero) boosts.
        state = self.env._state_for_player("p1")
        self.assertEqual(state.opponent_active_boosts.get("spa", 0), 0)

    def test_future_sight_pending_on_opponent_side(self) -> None:
        self._drive("future_sight_pending")
        state = self.env._state_for_player("p1")
        # Future Sight cast by p1 schedules a strike on the OPPONENT (p2) side.
        self.assertGreater(state.opponent_future_sight_turns, 0)

    def test_perish_song_volatile_countdown(self) -> None:
        rounds = self._drive("perish_song")["p1"]
        tok = self.S.SELF_POKEMON_TOKEN_OFFSET
        perish_ids = {self.vocab.encode(f"volatile:perish{n}") for n in range(4)}
        seen = any(
            perish_ids & set(o.categorical_ids[tok][self.S.CATEGORY_VOLATILE_OFFSET + i]
                             for i in range(self.S.VOLATILE_BUCKET_COUNT))
            for o in rounds
        )
        self.assertTrue(seen, "Perish Song should encode a perishN volatile")

    def test_counter_mirrorcoat_drives_clean(self) -> None:
        # Fixed-damage (Counter/Mirror Coat) + Mirror-Coat-vs-Dark immunity must
        # encode without crashing (smoke regression on callback-damage moves).
        rounds = self._drive("counter_mirrorcoat")["p1"]
        self.assertGreaterEqual(len(rounds), 1)

    # --- FIXED: in-battle retype now reflected in the type slots (see PR: in-battle-retype). --
    def test_castform_forecast_retype(self) -> None:
        # p1 sets sun -> p2 Castform Forecast-changes to Castform-Sunny (Fire).
        # Observe from p1 (Castform = opponent-active) AFTER the sun is up.
        rounds = self._drive("castform_forecast_formechange")["p1"]
        obs = rounds[-1]
        tok = self._active_token("p1", "opp")
        fire = self.vocab.encode("type:Fire")
        self.assertEqual(
            self._type1(obs, tok), fire,
            "-formechange retype must encode the forme's type (Castform-Sunny -> Fire). "
            "See docs/validated_interactions.md.",
        )

    def test_colorchange_kecleon_retype(self) -> None:
        # p2 Alakazam Psychic hits p1 Kecleon -> Color Change to Psychic type.
        rounds = self._drive("colorchange_kecleon")["p1"]
        obs = rounds[-1]
        tok = self._active_token("p1", "self")
        psychic = self.vocab.encode("type:Psychic")
        self.assertEqual(
            self._type1(obs, tok), psychic,
            "typechange retype must encode the payload type (Color Change -> Psychic). "
            "See docs/validated_interactions.md.",
        )


if __name__ == "__main__":
    unittest.main()
