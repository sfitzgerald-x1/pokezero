"""Live-env integration for the Tier-2 residual channel (#505 follow-up).

End-to-end through the real BattleStream env: collection populates the reserved
residual/validity slots where the protocol says it should, the incremental live
tracker is evidence-monotone-consistent with the batch inference (equal values
wherever both are valid; divergence only in the live-sees-more-evidence direction),
mask-off encodes are byte-identical to a population-free pipeline, and the collect
CLI records the masks in cache metadata for the trainer's cross-check. All tests need
a built Showdown checkout + node and skip cleanly without them.
"""

import json
import os
import random
import shutil
import tempfile
import unittest
from pathlib import Path

from pokezero.observation import ObservationFeatureMasks
from pokezero.showdown import (
    NUMERIC_TT_CB_BIT,
    NUMERIC_TT_INVESTMENT_BIT,
    NUMERIC_TT_RESIDUAL,
    NUMERIC_TT_RESIDUAL_VALID,
    TRANSITION_TOKEN_OFFSET,
    parse_showdown_replay,
)

_SEED = 303  # deterministic ~59-turn game with populated residuals on both sides


def _integration_root() -> Path | None:
    from pokezero.local_showdown import DEFAULT_SHOWDOWN_ROOT

    root = Path(os.environ.get("POKEZERO_SHOWDOWN_ROOT") or DEFAULT_SHOWDOWN_ROOT)
    if not (root / "dist" / "sim" / "index.js").exists():
        return None
    if shutil.which("node") is None:
        return None
    return root


def _play(env, seed: int, max_steps: int = 400):
    """Seeded random-legal self-play; observations collected per step per player."""
    rng = random.Random(seed)
    env.reset(seed=seed)
    observations = []
    steps = 0
    while steps < max_steps and env.terminal() is None:
        requested = env.requested_players()
        if not requested:
            break
        actions = {}
        for player in requested:
            observation = env.observe(player)
            observations.append((steps, player, observation))
            legal = [index for index, ok in enumerate(observation.legal_action_mask) if ok]
            actions[player] = rng.choice(legal)
        env.step(actions)
        steps += 1
    return observations


@unittest.skipUnless(_integration_root() is not None, "requires built Showdown checkout and node")
class LiveResidualPopulationTest(unittest.TestCase):
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

    def test_populated_slots_match_batch_inference_and_encode_alignment(self) -> None:
        from pokezero.dex import load_showdown_dex_cached
        from pokezero.randbat import load_gen3_randbat_source_cached
        from pokezero.tier2 import cb_whitelist_for_source, infer_tier2, own_team_from_request

        env = self._env(ObservationFeatureMasks())
        try:
            observations = _play(env, _SEED)
            self.assertEqual(sorted(env._tier2_trackers), ["p1", "p2"])
            dex = load_showdown_dex_cached(self.root)
            source = load_gen3_randbat_source_cached(self.root)
            whitelist = cb_whitelist_for_source(source, dex)
            replay = parse_showdown_replay(env.protocol_lines)
            budget = env.config.feature_masks.transition_token_budget

            populated_total = 0
            for player in ("p1", "p2"):
                state = env._state_for_player(player)
                live = [(t.residual, t.residual_valid) for t in state.transition_tokens]
                batch = infer_tier2(
                    replay,
                    perspective_slot=player,
                    own_team=own_team_from_request(env._first_requests[player]),
                    dex=dex,
                    set_source=source,
                    whitelist=whitelist,
                )
                # At this pinned seed live and batch agree outright (empirically
                # verified); the GENERAL invariant is only evidence-monotone
                # consistency — see MonotoneConsistencySweepTest below.
                self.assertEqual(live, [(t.residual, t.residual_valid) for t in batch.tokens])
                self.assertEqual(env._tier2_trackers[player].cb_bits, dict(batch.cb_bits))
                valid_count = sum(1 for _, valid in live if valid)
                self.assertGreater(valid_count, 0, f"{player}: no populated residuals")
                populated_total += valid_count

                # Encode alignment: valid tokens write clamped residual + validity 1.0;
                # everything else (disqualified strikes, switches, cants) stays 0/0.
                observation = env.observe(player)
                encoded = state.transition_tokens[-budget:]
                for offset, token in enumerate(encoded):
                    row = observation.numeric_features[TRANSITION_TOKEN_OFFSET + offset]
                    if token.residual_valid and token.residual is not None:
                        self.assertEqual(row[NUMERIC_TT_RESIDUAL_VALID], 1.0)
                        self.assertAlmostEqual(
                            row[NUMERIC_TT_RESIDUAL], max(-1.0, min(1.0, token.residual)), places=5
                        )
                    else:
                        self.assertEqual(row[NUMERIC_TT_RESIDUAL], 0.0)
                        self.assertEqual(row[NUMERIC_TT_RESIDUAL_VALID], 0.0)
                    self.assertEqual(row[NUMERIC_TT_CB_BIT], 1.0 if token.cb_bit else 0.0)
                    # The investment column stays zero under default masks
                    # (tier2_investment defaults off; see test_investment_live_env
                    # for the mask-on path).
                    self.assertEqual(row[NUMERIC_TT_INVESTMENT_BIT], 0.0)
            self.assertGreater(populated_total, 5)
        finally:
            env.close()

    def test_mask_off_is_byte_identical_except_residual_columns(self) -> None:
        env_on = self._env(ObservationFeatureMasks())
        env_off = self._env(ObservationFeatureMasks(tier2_residuals=False))
        try:
            obs_on = _play(env_on, _SEED)
            obs_off = _play(env_off, _SEED)
            # Mask-off builds no trackers at all: zero hot-path cost, and the encodes
            # must be byte-identical to the tier2-on run outside the two reserved
            # residual columns (proving the wiring perturbs nothing else).
            self.assertEqual(env_off._tier2_trackers, {})
            self.assertEqual(len(obs_on), len(obs_off))
            saw_residual_difference = False
            for (step_a, player_a, a), (step_b, player_b, b) in zip(obs_on, obs_off):
                self.assertEqual((step_a, player_a), (step_b, player_b))
                self.assertEqual(a.categorical_ids, b.categorical_ids)
                self.assertEqual(a.attention_mask, b.attention_mask)
                self.assertEqual(a.token_type_ids, b.token_type_ids)
                for row_a, row_b in zip(a.numeric_features, b.numeric_features):
                    stripped_a = list(row_a)
                    stripped_b = list(row_b)
                    if (
                        stripped_a[NUMERIC_TT_RESIDUAL] != stripped_b[NUMERIC_TT_RESIDUAL]
                        or stripped_a[NUMERIC_TT_RESIDUAL_VALID] != stripped_b[NUMERIC_TT_RESIDUAL_VALID]
                        or stripped_a[NUMERIC_TT_CB_BIT] != stripped_b[NUMERIC_TT_CB_BIT]
                    ):
                        saw_residual_difference = True
                    stripped_a[NUMERIC_TT_RESIDUAL] = stripped_b[NUMERIC_TT_RESIDUAL] = 0.0
                    stripped_a[NUMERIC_TT_RESIDUAL_VALID] = stripped_b[NUMERIC_TT_RESIDUAL_VALID] = 0.0
                    stripped_a[NUMERIC_TT_CB_BIT] = stripped_b[NUMERIC_TT_CB_BIT] = 0.0
                    self.assertEqual(stripped_a, stripped_b)
                # The mask-off run's tier2 columns are all zero; the investment
                # column is zero under BOTH configs (tier2_investment defaults off).
                for row in b.numeric_features:
                    self.assertEqual(row[NUMERIC_TT_RESIDUAL], 0.0)
                    self.assertEqual(row[NUMERIC_TT_RESIDUAL_VALID], 0.0)
                    self.assertEqual(row[NUMERIC_TT_CB_BIT], 0.0)
                    self.assertEqual(row[NUMERIC_TT_INVESTMENT_BIT], 0.0)
                for row in a.numeric_features:
                    self.assertEqual(row[NUMERIC_TT_INVESTMENT_BIT], 0.0)
            self.assertTrue(saw_residual_difference, "seed produced no populated residuals")
        finally:
            env_on.close()
            env_off.close()


@unittest.skipUnless(_integration_root() is not None, "requires built Showdown checkout and node")
class MonotoneConsistencySweepTest(unittest.TestCase):
    """Live-vs-batch over a multi-game sweep: evidence-monotone consistency.

    The live tracker assesses each strike at the first observation boundary after it,
    which can carry strictly MORE belief evidence than the batch inference's
    next-action cutoff (end-of-turn non-proc pruning lands in the same window). The
    invariant (per the #507 review's 60-game sweep, which found 4 such divergences,
    all in the more-evidence direction): divergences are ONLY live-stands-down (CB) or
    live-invalidates (residual), and residual values are identical wherever both sides
    are valid.
    """

    def test_ten_game_sweep_only_monotone_divergences(self) -> None:
        from pokezero.dex import load_showdown_dex_cached
        from pokezero.local_showdown import LocalShowdownConfig, LocalShowdownEnv
        from pokezero.randbat import load_gen3_randbat_source_cached
        from pokezero.tier2 import cb_whitelist_for_source, infer_tier2, own_team_from_request

        root = _integration_root()
        dex = load_showdown_dex_cached(root)
        source = load_gen3_randbat_source_cached(root)
        whitelist = cb_whitelist_for_source(source, dex)

        jointly_valid = 0
        live_invalidations = 0
        cb_stand_downs = 0
        for seed in range(401, 411):
            env = LocalShowdownEnv(
                LocalShowdownConfig(showdown_root=root, set_belief_source=True)
            )
            try:
                _play(env, seed)
                replay = parse_showdown_replay(env.protocol_lines)
                for player in ("p1", "p2"):
                    state = env._state_for_player(player)
                    batch = infer_tier2(
                        replay,
                        perspective_slot=player,
                        own_team=own_team_from_request(env._first_requests[player]),
                        dex=dex,
                        set_source=source,
                        whitelist=whitelist,
                    )
                    self.assertEqual(len(state.transition_tokens), len(batch.tokens))
                    for live_token, batch_token in zip(state.transition_tokens, batch.tokens):
                        if live_token.residual_valid and batch_token.residual_valid:
                            jointly_valid += 1
                            self.assertAlmostEqual(
                                live_token.residual, batch_token.residual, places=12,
                                msg=f"seed {seed} {player}: jointly-valid residuals differ",
                            )
                        elif batch_token.residual_valid and not live_token.residual_valid:
                            live_invalidations += 1  # allowed: live saw more evidence
                        elif live_token.residual_valid and not batch_token.residual_valid:
                            self.fail(
                                f"seed {seed} {player}: live claims a residual the batch "
                                f"cutoff masks ({live_token.action}) — not evidence-monotone"
                            )
                        if live_token.cb_bit and not batch_token.cb_bit:
                            self.fail(
                                f"seed {seed} {player}: live sets the as-of-strike CB bit "
                                f"where the batch cutoff does not — not evidence-monotone"
                            )
                    live_true = {k for k, v in env._tier2_trackers[player].cb_bits.items() if v}
                    batch_true = {k for k, v in batch.cb_bits.items() if v}
                    self.assertLessEqual(
                        live_true, batch_true,
                        f"seed {seed} {player}: live CB bit set where batch stands down",
                    )
                    cb_stand_downs += len(batch_true - live_true)
            finally:
                env.close()
        # The sweep must actually exercise the value-equality assertion.
        self.assertGreater(jointly_valid, 100)


@unittest.skipUnless(_integration_root() is not None, "requires built Showdown checkout and node")
class CbBitFixtureGameTest(unittest.TestCase):
    """The CB bit fires on a REAL gate-corpus game with a Choice Band reveal-by-damage.

    Fixture: seed-11 game 7 of the #505 gate corpus — p2's Pidgeot holds Choice Band
    (ground truth from its own opening request, recorded in the fixture header) and the
    p1-perspective inference concluded it from damage exceedance on turns 21-25.
    """

    def test_cb_bit_fires_and_is_monotone_on_the_fixture_game(self) -> None:
        from pokezero.dex import load_showdown_dex_cached
        from pokezero.randbat import load_gen3_randbat_source_cached
        from pokezero.tier2 import cb_whitelist_for_source, infer_tier2, own_team_from_request

        root = _integration_root()
        fixture = Path(__file__).parent / "fixtures" / "showdown" / "tier2-cb-pidgeot-game.log"
        lines = [
            line
            for line in fixture.read_text(encoding="utf-8").splitlines()
            if line and not line.startswith("#")
        ]
        replay = parse_showdown_replay(lines)
        first_request = next(
            json.loads(line[len("|request|"):])
            for line in lines
            if line.startswith("|request|") and '"id":"p1"' in line.replace(" ", "")
        )
        dex = load_showdown_dex_cached(root)
        source = load_gen3_randbat_source_cached(root)
        inference = infer_tier2(
            replay,
            perspective_slot="p1",
            own_team=own_team_from_request(first_request),
            dex=dex,
            set_source=source,
            whitelist=cb_whitelist_for_source(source, dex),
        )
        self.assertTrue(inference.cb_bits.get("p2:pidgeot"), "CB conclusion did not fire")
        self.assertGreaterEqual(len(inference.cb_strike_turns["p2:pidgeot"]), 2)
        # As-of-strike monotonicity on the token stream: no bit before the concluding
        # strike, bit on every assessed Pidgeot strike from it onward.
        flags = [
            (token.turn, token.cb_bit)
            for token in inference.tokens
            if token.kind == "move" and token.actor_slot == "p2" and token.actor_species == "Pidgeot"
        ]
        bits = [bit for _, bit in flags]
        self.assertIn(True, bits)
        first_true = bits.index(True)
        self.assertGreater(first_true, 0)  # two strikes are needed before it can fire
        self.assertTrue(all(bits[first_true:]))
        concluding_turn = flags[first_true][0]
        self.assertEqual(concluding_turn, inference.cb_strike_turns["p2:pidgeot"][1])

    def test_cb_pinned_bit_fires_at_the_concluding_turn_and_persists(self) -> None:
        """The v2.1 per-mon PINNED surface on the CB-Pidgeot fixture: dark on a replay
        prefix ending before the concluding strike, 1.0 on Pidgeot's opp-mon row from the
        conclusion onward — including at game end, after later switches (per-mon fact,
        truncation-independent) — and only on Pidgeot's row."""
        from dataclasses import replace as dc_replace

        from pokezero.dex import load_showdown_dex_cached
        from pokezero.randbat import load_gen3_randbat_source_cached
        from pokezero.randbat_vocab import gen3_category_vocabulary
        from pokezero.showdown import (
            NUMERIC_TIER2_CB_PINNED,
            NUMERIC_TIER2_INVESTMENT_PINNED,
            OPPONENT_POKEMON_TOKEN_OFFSET,
            V2_1_REPLAY_OBSERVATION_SPEC,
            _normalize_identifier,
            normalize_for_player,
            observation_from_player_state,
        )
        from pokezero.tier2 import (
            apply_residuals,
            cb_whitelist_for_source,
            infer_tier2,
            own_team_from_request,
        )

        root = _integration_root()
        fixture = Path(__file__).parent / "fixtures" / "showdown" / "tier2-cb-pidgeot-game.log"
        lines = [
            line
            for line in fixture.read_text(encoding="utf-8").splitlines()
            if line and not line.startswith("#")
        ]
        first_request = next(
            json.loads(line[len("|request|"):])
            for line in lines
            if line.startswith("|request|") and '"id":"p1"' in line.replace(" ", "")
        )
        dex = load_showdown_dex_cached(root)
        source = load_gen3_randbat_source_cached(root)
        vocab = gen3_category_vocabulary(root)
        whitelist = cb_whitelist_for_source(source, dex)
        own_team = own_team_from_request(first_request)

        def pinned_rows(prefix_lines):
            replay = parse_showdown_replay(prefix_lines)
            inference = infer_tier2(
                replay,
                perspective_slot="p1",
                own_team=own_team,
                dex=dex,
                set_source=source,
                whitelist=whitelist,
            )
            state = normalize_for_player(
                replay,
                player_id="p1",
                configured_showdown_slot="p1",
                format_id="gen3randombattle",
                set_source=source,
            )
            wired = dc_replace(
                state, transition_tokens=apply_residuals(state.transition_tokens, inference)
            )
            observation = observation_from_player_state(
                wired, category_vocab=vocab, spec=V2_1_REPLAY_OBSERVATION_SPEC, dex=dex
            )
            return inference, state, {
                _normalize_identifier(mon.species): observation.numeric_features[
                    OPPONENT_POKEMON_TOKEN_OFFSET + index
                ]
                for index, mon in enumerate(state.opponent_team)
            }

        full_inference, _, _ = pinned_rows(lines)
        concluding_turn = full_inference.cb_strike_turns["p2:pidgeot"][1]

        # Prefix ending as the concluding turn OPENS: the second exceedance has not
        # landed yet, so the conclusion must not stand anywhere.
        prefix = lines[: lines.index(f"|turn|{concluding_turn}") + 1]
        _, _, before = pinned_rows(prefix)
        for species, row in before.items():
            self.assertEqual(row[NUMERIC_TIER2_CB_PINNED], 0.0, species)

        # Full game: Pidgeot is long since off the field (and the game moved on), yet
        # its row carries the pinned conclusion; every other row stays dark, and the
        # investment twin stays a reserve everywhere.
        _, final_state, after = pinned_rows(lines)
        self.assertIn("pidgeot", after)
        pidgeot_active = next(
            mon.active for mon in final_state.opponent_team
            if _normalize_identifier(mon.species) == "pidgeot"
        )
        self.assertFalse(pidgeot_active, "fixture drift: Pidgeot should be benched/fainted at end")
        for species, row in after.items():
            self.assertEqual(
                row[NUMERIC_TIER2_CB_PINNED], 1.0 if species == "pidgeot" else 0.0, species
            )
            self.assertEqual(row[NUMERIC_TIER2_INVESTMENT_PINNED], 0.0, species)


@unittest.skipUnless(_integration_root() is not None, "requires built Showdown checkout and node")
class CollectCacheMaskMetadataTest(unittest.TestCase):
    def test_checkpointless_collect_records_masks_and_train_cross_checks(self) -> None:
        from types import SimpleNamespace

        from pokezero.neural_cli import _require_cache_masks_match_model_config
        from pokezero.rollout_cli import main as rollout_main

        root = _integration_root()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "cache"
            exit_code = rollout_main(
                [
                    "collect-selfplay-training-cache",
                    "--games", "1",
                    "--out", str(out),
                    "--seed-start", "17",
                    "--showdown-root", str(root),
                    "--transition-token-budget", "32",
                ]
            )
            self.assertEqual(exit_code, 0)
            metadata = json.loads((out / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(
                metadata["feature_masks"],
                {
                    "stats_block": True,
                    "exact_state": True,
                    "transition_token_budget": 32,
                    "tier2_residuals": True,
                    "tier2_investment": False,
                },
            )
            matching_model = SimpleNamespace(
                stats_block_enabled=True,
                exact_state_enabled=True,
                transition_token_budget=32,
                tier2_residuals=True,
                tier2_investment=False,
            )
            _require_cache_masks_match_model_config([out], matching_model)  # no raise
            mismatched_model = SimpleNamespace(
                stats_block_enabled=True,
                exact_state_enabled=True,
                transition_token_budget=128,
                tier2_residuals=True,
                tier2_investment=False,
            )
            with self.assertRaisesRegex(ValueError, "mask-mismatched"):
                _require_cache_masks_match_model_config([out], mismatched_model)


if __name__ == "__main__":
    unittest.main()
