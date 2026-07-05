"""Live-env integration for the defender-side investment channel (v2.1 batch 2).

End-to-end through the real BattleStream env: the investment tracker only exists under
the double mask (tier2_residuals AND tier2_investment), populated token codes reach the
reserved column, and the live accrual is knowledge-monotone-consistent with the batch
inference (live boundaries can carry MORE belief evidence, which can only remove
candidate variants — so live may pin where batch's next-action cutoff still sees two
families, or stand a pin down to vacuity, but a value concluded by BOTH must be equal).
All tests need a built Showdown checkout + node and skip cleanly without them.
"""

import unittest

from pokezero.investment import conclusion_column_code
from pokezero.observation import ObservationFeatureMasks
from pokezero.showdown import (
    NUMERIC_TIER2_INVESTMENT_PINNED,
    NUMERIC_TT_INVESTMENT_BIT,
    OPPONENT_POKEMON_TOKEN_OFFSET,
    TRANSITION_TOKEN_OFFSET,
    _normalize_identifier,
    parse_showdown_replay,
)
from tests.test_tier2_live_env import _integration_root, _play


@unittest.skipUnless(_integration_root() is not None, "requires built Showdown checkout and node")
class LiveInvestmentPopulationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = _integration_root()

    def _env(self, masks: ObservationFeatureMasks):
        from pokezero.local_showdown import LocalShowdownConfig, LocalShowdownEnv

        return LocalShowdownEnv(
            LocalShowdownConfig(
                showdown_root=self.root, set_belief_source=True, feature_masks=masks
            )
        )

    def test_default_masks_build_no_tracker(self) -> None:
        env = self._env(ObservationFeatureMasks())
        try:
            _play(env, 303, max_steps=12)
            self.assertEqual(env._investment_trackers, {})
            self.assertFalse(env.investment_active())
        finally:
            env.close()

    def test_live_codes_are_knowledge_monotone_vs_batch_and_encode(self) -> None:
        from pokezero.dex import load_showdown_dex_cached
        from pokezero.investment import infer_investment
        from pokezero.randbat import load_gen3_randbat_source_cached
        from pokezero.tier2 import own_team_from_request

        dex = load_showdown_dex_cached(self.root)
        source = load_gen3_randbat_source_cached(self.root)

        conclusions_checked = 0
        codes_seen = 0
        pin_strikes = 0
        pinned_checked = 0
        for seed in range(401, 409):
            env = self._env(ObservationFeatureMasks(tier2_investment=True))
            try:
                observations = _play(env, seed)
                self.assertTrue(env.investment_active())
                replay = parse_showdown_replay(env.protocol_lines)
                budget = env.config.feature_masks.transition_token_budget
                for player in ("p1", "p2"):
                    tracker = env._investment_trackers.get(player)
                    if tracker is None:
                        continue
                    batch = infer_investment(
                        replay,
                        perspective_slot=player,
                        own_team=own_team_from_request(env._first_requests[player]),
                        dex=dex,
                        set_source=source,
                    )
                    live = tracker.conclusions
                    for key, conclusion in live.items():
                        batch_conclusion = batch.conclusions.get(key)
                        if batch_conclusion is None:
                            continue
                        if (
                            conclusion.hp_value is not None
                            and batch_conclusion.hp_value is not None
                        ):
                            conclusions_checked += 1
                            self.assertEqual(
                                conclusion.hp_value, batch_conclusion.hp_value,
                                f"seed {seed} {player} {key}: live and batch pin different max HP",
                            )
                        for stat, value in conclusion.defense_values.items():
                            batch_value = batch_conclusion.defense_values.get(stat)
                            if batch_value is not None:
                                conclusions_checked += 1
                                self.assertEqual(
                                    value, batch_value,
                                    f"seed {seed} {player} {key}: defense pins differ",
                                )
                    codes_seen += sum(1 for code in tracker.token_codes.values() if code)
                    pin_strikes += sum(
                        1
                        for strike in tracker._state.strikes
                        if strike.hp_pin is not None or strike.defense_pin is not None
                    )

                # Encode alignment: the final observation of each player carries the
                # tracker's codes in the tt history column (mask on), and the per-mon
                # PINNED column (139) carries each concluded opponent mon's
                # conclusion_column_code — the authoritative current-state surface must
                # agree exactly with the tracker's per-mon view, and stay 0.0 for
                # unconcluded mons.
                for player in ("p1", "p2"):
                    state = env._state_for_player(player)
                    observation = env.observe(player)
                    encoded = state.transition_tokens[-budget:]
                    for offset, token in enumerate(encoded):
                        row = observation.numeric_features[TRANSITION_TOKEN_OFFSET + offset]
                        self.assertEqual(row[NUMERIC_TT_INVESTMENT_BIT], token.investment)
                    tracker = env._investment_trackers.get(player)
                    expected_by_species: dict[str, float] = {}
                    if tracker is not None:
                        for key, conclusion in tracker.conclusions.items():
                            code = conclusion_column_code(conclusion)
                            if code:
                                expected_by_species[key.split(":", 1)[1]] = code
                    for index, mon in enumerate(state.opponent_team):
                        row = observation.numeric_features[OPPONENT_POKEMON_TOKEN_OFFSET + index]
                        self.assertEqual(
                            row[NUMERIC_TIER2_INVESTMENT_PINNED],
                            expected_by_species.get(_normalize_identifier(mon.species), 0.0),
                            f"seed {seed} {player} {mon.species}: pinned column drifts "
                            "from the tracker conclusion",
                        )
                        pinned_checked += 1
            finally:
                env.close()
        # The sweep must actually exercise the live pin path. CONCLUSIONS are rare by
        # design (two-strike rule; the gate saw 22 over 240 perspectives), so the
        # floor is on single-strike pin events, which the gate saw at ~0.5 per
        # perspective under move-biased play.
        self.assertGreater(pin_strikes, 0)
        # Both perspectives of all eight seeds must have exercised the pinned-column
        # comparison across the full opponent bench.
        self.assertGreaterEqual(pinned_checked, 8 * 2 * 6)


if __name__ == "__main__":
    unittest.main()
