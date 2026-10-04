"""Opt-in policy sampling in the incumbent model/value native traversal."""

import json
import hashlib
import os
from pathlib import Path
import time
from types import SimpleNamespace
import unittest

from test_model_priors_search import _EncodedSearchFixture, _crate_ready, pokezero_search


@unittest.skipUnless(_crate_ready, "requires native model feature")
class NativePolicyOpponentSearchTest(_EncodedSearchFixture, unittest.TestCase):
    def run_search(self, callback=None, position=None, **options):
        position = self.position if position is None else position
        ctx = json.loads(position["ctx"])
        opponent_slot = "p2" if position["self_side"] == "side_one" else "p1"
        side_index = 1 if opponent_slot == "p2" else 0
        order = list(ctx[opponent_slot])
        active = position["actives"][side_index]
        order[0], order[active] = order[active], order[0]
        kwargs = dict(max_depth=3, seed=5, model_priors=True, use_opponent_priors=False, arm_priors=True)
        if callback is not None:
            kwargs.update(policy_opponent_callback=callback, policy_opponent_seed=101,
                policy_opponent_request_order=order)
        kwargs.update(options)
        return json.loads(self.native.search_batched_multi_encoded(position["state_str"], 64, 1,
            self.tables_json, position["row_inputs"], position["ctx"],
            pokezero_search.FoldState.from_payload(position["fold_state"]), **kwargs))

    def test_provider_reaches_root_and_children_with_private_safe_prefixes(self):
        payloads = []
        def provide(raw):
            payload = json.loads(raw)
            payloads.append(payload)
            indices = payload["native_request_bundle"]["native_action_indices"]
            return [1.0 if index == indices[0] else 0.0 for index in indices]

        report = self.run_search(provide)
        self.assertEqual(report["iterations"], 64)
        self.assertEqual(report["policy_opponent_mode"], "own_policy_callback")
        self.assertEqual(report["policy_opponent_provider_calls"], len(payloads))
        self.assertEqual(report["policy_opponent_evals"], sum(
            len(p["native_request_bundle"]["native_action_indices"]) > 1 for p in payloads))
        self.assertGreater(report["policy_opponent_samples"], len(payloads))
        self.assertEqual(payloads[0]["public_branch_lines"], [])
        self.assertTrue(any(p["public_branch_lines"] for p in payloads[1:]), "child policy was never inferred")
        for payload in payloads:
            self.assertEqual(set(payload), {"native_request_bundle", "public_branch_lines", "opponent_slot"})
            bundle = payload["native_request_bundle"]
            self.assertEqual(bundle["request"]["side"]["id"], payload["opponent_slot"])
            for line in payload["public_branch_lines"]:
                parts = line.split("|")
                if parts[1] in {"switch", "drag", "replace", "-damage", "-heal", "-sethp"}:
                    field = 4 if parts[1] in {"switch", "drag", "replace"} else 3
                    self.assertTrue(parts[field] == "0 fnt" or parts[field].split()[0].endswith("/100"), line)

    def test_seeded_trace_and_distribution_are_reproducible(self):
        def collect():
            trace = []
            def provide(raw):
                trace.append(json.loads(raw))
                return [1.0] * len(trace[-1]["native_request_bundle"]["native_action_indices"])
            return self.run_search(provide), trace
        first, trace = collect()
        second, duplicate = collect()
        self.assertEqual(trace, duplicate)
        for field in ("side_one", "side_two", "iterations", "policy_opponent_samples", "policy_opponent_evals", "policy_opponent_provider_calls"):
            self.assertEqual(first[field], second[field])

    def test_switches_evolve_request_order(self):
        payloads = []
        def provide(raw):
            payload = json.loads(raw)
            payloads.append(payload)
            indices = payload["native_request_bundle"]["native_action_indices"]
            # Force a switch at every genuine choice boundary. The reached
            # request's first party row must follow accumulated switch swaps.
            chosen = next((index for index in indices if index >= 4), indices[0])
            return [1.0 if index == chosen else 0.0 for index in indices]
        self.run_search(provide)
        root_ident = payloads[0]["native_request_bundle"]["request"]["side"]["pokemon"][0]["ident"]
        self.assertTrue(any(p["native_request_bundle"]["request"]["side"]["pokemon"][0]["ident"] != root_ident
                            for p in payloads[1:]), "opponent never switched")

    def test_provider_errors_and_bad_probabilities_abort_without_fallback(self):
        def failure(raw):
            raise ValueError("test opponent provider failed")
        for callback in (failure, lambda raw: [], lambda raw: [float("nan")] * len(json.loads(raw)["native_request_bundle"]["native_action_indices"]),
                         lambda raw: [0.0] * len(json.loads(raw)["native_request_bundle"]["native_action_indices"])):
            with self.subTest(callback=callback), self.assertRaises(ValueError):
                self.run_search(callback)
        # Hidden trapping can collapse the native legal surface to one move.
        # It must still refuse at the side-only constructor, not bypass the
        # provider just because there is only one native arm.
        trapped = dict(self.position)
        subject, opponent = trapped["state_str"].split("/", 1)
        subject_fields = subject.split(",")
        subject_fields[8] = subject_fields[9] = "SHADOWTAG"
        opponent_fields = opponent.split(",")
        for index in (23, 24, 25):
            opponent_fields[index] = "NONE;true;0"
        trapped["state_str"] = ",".join(subject_fields) + "/" + ",".join(opponent_fields)
        with self.assertRaisesRegex(ValueError, "native and private-knowledge legal surfaces differ"):
            self.run_search(failure, position=trapped, max_depth=1)
        # A provider exception cannot poison subsequent searches or the model.
        self.assertEqual(self.run_search()["iterations"], 64)

    def test_disabled_mode_keeps_report_schema_and_requires_complete_opt_in(self):
        report = self.run_search()
        self.assertFalse(any(key.startswith("policy_opponent_") for key in report))
        for options in (dict(policy_opponent_seed=9), dict(policy_opponent_request_order=["unknown"]),
                        dict(policy_opponent_callback=123, policy_opponent_seed=9, policy_opponent_request_order=["unknown"])):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.run_search(**options)
        for options in (dict(use_opponent_priors=True), dict(model_priors=False), dict(rollout_leaf_mode="model_value")):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.run_search(lambda raw: [], **options)

    def test_subject_root_priors_are_unchanged_by_opponent_sampling(self):
        incumbent = self.run_search()
        sampled = self.run_search(lambda raw: [1.] * len(
            json.loads(raw)["native_request_bundle"]["native_action_indices"]))
        self.assertEqual(incumbent["root_priors"], sampled["root_priors"])
        self.assertEqual(incumbent["model_priors"], sampled["model_priors"])
        self.assertEqual(sampled["root_prior_fallbacks"], 0)

    def test_opponent_inference_is_charged_to_native_deadline(self):
        def costly_provider(raw):
            time.sleep(.15)
            return [1.] * len(json.loads(raw)["native_request_bundle"]["native_action_indices"])
        report = self.run_search(costly_provider, time_budget_ms=100)
        self.assertGreaterEqual(report["policy_opponent_evals"], 1)
        self.assertGreaterEqual(report["policy_opponent_s"], .15)
        self.assertGreaterEqual(report["time_budget_elapsed_ms"], 150)
        self.assertTrue(report["time_budget_exhausted"])
        self.assertLess(report["iterations"], 64)
        self.assertGreater(report["time_budget_batch_overshoot_ms"], 0)


@unittest.skipUnless(_crate_ready, "requires native model feature")
class NativeCanonicalOwnPolicyTest(NativePolicyOpponentSearchTest):
    """The same fixture weights serve subject value and opponent OWN head.

    V4 avoids claiming support for the not-yet-integrated legacy residual
    observer. This does not change the champion's settings or certify it.
    """
    observation_schema_version = "pokezero.observation.v4"
    transition_token_budget = 0

    def test_full_policy_dispatches_canonical_own_head_in_both_seats(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        import random
        from threading import Lock
        from pokezero.engine_search import (EngineMctsConfig, EngineMctsPolicy,
            EngineMctsStats, OpponentRequestOrderResolution)
        from pokezero.dex import load_showdown_dex_cached
        from pokezero.local_showdown import DEFAULT_SHOWDOWN_ROOT
        from pokezero.showdown import parse_showdown_replay
        from test_showdown import FakeSetSource
        from golden_encoder_backends import _legal_mask_from_metadata

        for side, position in self.perspective_positions.items():
            with self.subTest(side=side):
                inputs, ctx = json.loads(position["row_inputs"]), json.loads(position["ctx"])
                md = inputs["observation_metadata"]
                subject = md["showdown_slot"]
                opponent = "p2" if subject == "p1" else "p1"
                prefix = tuple(line.replace("|selfa:", f"|{subject}a:").replace("|opponenta:", f"|{opponent}a:")
                               for line in md["recent_public_events"])
                replay = parse_showdown_replay(prefix, battle_id=inputs["battle_id"],
                                               hp_visibility={"p1": "exact", "p2": "exact"})
                context = SimpleNamespace(player_id=subject, battle_id=inputs["battle_id"],
                    seed=inputs["battle_seed"], format_id=inputs["format_id"], decision_round_index=0,
                    observation=SimpleNamespace(metadata=md, legal_action_mask=_legal_mask_from_metadata(md)),
                    public_materialization_state=SimpleNamespace(replay=replay))
                policy = object.__new__(EngineMctsPolicy)
                policy.policy_id = "own-policy-test"
                policy._config = EngineMctsConfig(worlds=1, leaf_eval="model", model_path="fixture.pt",
                    checkpoint_path="fixture.ckpt", tables_path="fixture.json", strict_fallbacks=True,
                    policy_opponent=True, policy_opponent_seed=103, search_sims=64, search_batch=1,
                    search_depth=3)
                policy.stats = EngineMctsStats()
                policy._world_failures_before = {}
                policy._tables_json = self.tables_json
                policy._policy_opponent_model = self.python_model
                policy._policy_opponent_inference_lock = Lock()
                policy._policy_opponent_result = getattr(self, "champion_result", None) or SimpleNamespace(
                    model_config=self.config, belief_set_source_hash=None)
                policy._set_source = getattr(self, "champion_source", None) or FakeSetSource()
                policy._dex = load_showdown_dex_cached(DEFAULT_SHOWDOWN_ROOT)
                order = list(ctx[opponent])
                active = position["actives"][0 if opponent == "p1" else 1]
                order[0], order[active] = order[active], order[0]
                world = SimpleNamespace(party_species={slot: ctx[slot] for slot in ("p1", "p2")},
                                        slot_sides={subject: side})
                state = SimpleNamespace(to_string=lambda: position["state_str"])
                with (patch.object(policy, "_validate_model_root_observation"),
                      patch.object(policy, "_root_inputs_json", return_value=position["row_inputs"]),
                      patch.object(policy, "_native", return_value=self.native),
                      patch.object(policy, "_search", side_effect=lambda context, rng: policy._search_model(
                          context, [(world, state)],
                          SimpleNamespace(to_payload=lambda: position["fold_state"]), rng)),
                      patch("pokezero.engine_search.opponent_request_order_resolution",
                            return_value=OpponentRequestOrderResolution(tuple(order), "resolved"))):
                    decision = policy.select_action_with_context(context, rng=random.Random(7))
                witness = decision.metadata["engine_mcts"]["policy_opponent"]
                self.assertEqual(len(witness["native_invocations"]), 1)
                self.assertGreater(witness["native_invocations"][0]["evaluations"], 1)
                self.assertEqual(policy.stats.total_iterations, 64)
                self.assertEqual(policy.stats.worlds_searched, 1)
                self.assertEqual(policy.stats.to_dict()["policy_opponent"]["evaluations"],
                                 witness["native_invocations"][0]["evaluations"])
                self.assertGreaterEqual(decision.metadata["policy_opponent_decision_elapsed_ms"],
                                        policy.stats.policy_opponent_wall_seconds * 1000)

    def test_canonical_own_head_runs_at_native_root_and_reached_children(self):
        self.assertEqual(set(self.perspective_positions), {"side_one", "side_two"})
        for side, position in self.perspective_positions.items():
            with self.subTest(subject_side=side):
                self._check_canonical_own_head(position)

    def _check_canonical_own_head(self, position):
        from pokezero.neural_policy import category_vocab_from_model_config
        from pokezero.dex import load_showdown_dex_cached
        from pokezero.local_showdown import DEFAULT_SHOWDOWN_ROOT
        from pokezero.policy_opponent import make_policy_opponent_callback
        from test_showdown import FakeSetSource

        inputs = json.loads(position["row_inputs"])
        md = inputs["observation_metadata"]
        subject = md["showdown_slot"]
        opponent = "p2" if subject == "p1" else "p1"
        # The fixture's first corpus boundary contains the complete public
        # battle-start prefix; translate only its explicit relative seat IDs.
        prefix = tuple(line.replace("|selfa:", f"|{subject}a:").replace("|opponenta:", f"|{opponent}a:")
                       for line in md["recent_public_events"])
        self.assertIn("|start", prefix)
        self.assertEqual(md["turn_number"], 1)
        vocab = category_vocab_from_model_config(self.config, DEFAULT_SHOWDOWN_ROOT)
        callback = make_policy_opponent_callback(
            public_lines=prefix, hp_visibility={"p1": "exact", "p2": "exact"}, opponent_slot=opponent,
            battle_id=inputs["battle_id"], battle_seed=inputs["battle_seed"], format_id=inputs["format_id"],
            set_source=getattr(self, "champion_source", None) or FakeSetSource(), model=self.python_model,
            result=getattr(self, "champion_result", None) or SimpleNamespace(model_config=self.config, belief_set_source_hash=None),
            category_vocab=vocab,
            dex=load_showdown_dex_cached(DEFAULT_SHOWDOWN_ROOT), device="cpu",
        )
        seen = []
        def recorded(raw):
            probabilities = callback(raw)
            self.assertAlmostEqual(sum(probabilities), 1.0, places=6)
            seen.append(json.loads(raw))
            return probabilities
        report = self.run_search(recorded, position=position)
        self.assertGreater(report["policy_opponent_evals"], 1)
        self.assertTrue(any(payload["public_branch_lines"] for payload in seen))
        self.assertFalse(vocab.observed_oov_tokens)


@unittest.skipUnless(_crate_ready and os.environ.get("POKEZERO_POLICY_OPPONENT_CHECKPOINT")
                     and os.environ.get("POKEZERO_POLICY_OPPONENT_BELIEF_CACHE"),
                     "requires explicit champion checkpoint and registered belief cache")
class ChampionCanonicalOwnPolicyTest(NativeCanonicalOwnPolicyTest):
    """Opt-in real-weight gate; never silently substitute fixture weights.

    The paths are provided explicitly so ordinary unit tests stay portable.
    This proves the local checkpoint/cache bytes and independently rebuilt
    runtime source binding, not an immutable image publication or playing strength.
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        from pokezero.engine_search import _fence_calibration_seam, _latch_encoder_tables_to_model_config
        from pokezero.neural_policy import (FreshValueHeadWarning, load_transformer_checkpoint,
            load_transformer_checkpoint_payload, observation_spec_from_model_config,
            feature_masks_from_model_config)
        from pokezero.local_showdown import DEFAULT_SHOWDOWN_ROOT
        from pokezero.randbat import Gen3RandbatSource
        from test_model_priors_search import _load_export_module
        from export_encoder_tables import build_tables
        import warnings

        checkpoint = Path(os.environ["POKEZERO_POLICY_OPPONENT_CHECKPOINT"])
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        expected = "0fd095923b4ac7e05d6e2b3ccab9c1e6869dff4893c2dae456caff10dce690be"
        if digest != expected:
            raise ValueError("champion gate requires the registered iteration-9375 checkpoint hash")
        payload = load_transformer_checkpoint_payload(checkpoint)
        _fence_calibration_seam(payload, "registered champion")
        with warnings.catch_warnings():
            warnings.simplefilter("error", FreshValueHeadWarning)
            model, result = load_transformer_checkpoint(checkpoint, map_location="cpu")
        model.eval()
        source_path = Path(os.environ["POKEZERO_POLICY_OPPONENT_BELIEF_CACHE"])
        source_bytes = source_path.read_bytes()
        source = Gen3RandbatSource.from_payload(json.loads(source_bytes))
        if source.metadata.source_hash != result.belief_set_source_hash:
            raise ValueError("champion gate belief source does not match checkpoint")
        # Re-enumerate from the configured built runtime, without adopting its
        # cached payload. A matching imported cache alone cannot certify the
        # Showdown bytes that the live environment is actually using.
        runtime_source = Gen3RandbatSource.from_showdown_root(DEFAULT_SHOWDOWN_ROOT, use_cache=False)
        if runtime_source.metadata.source_hash != result.belief_set_source_hash:
            raise ValueError("champion gate runtime source differs from checkpoint; use the clean pinned Showdown checkout")
        registered_payload = source.to_payload()
        runtime_payload = runtime_source.to_payload()
        registered_payload.pop("metadata")
        runtime_payload.pop("metadata")
        if runtime_payload != registered_payload:
            raise ValueError("champion gate rebuilt runtime universe differs from registered cache")
        cls.champion_source = runtime_source
        cls.champion_result = result
        cls.python_model = model
        cls.config = result.model_config
        cls.tables_json = _latch_encoder_tables_to_model_config(json.dumps(build_tables(
            str(DEFAULT_SHOWDOWN_ROOT), observation_schema_version=cls.config.observation_schema_version,
            spec=observation_spec_from_model_config(cls.config),
            masks=feature_masks_from_model_config(cls.config), trained_tokens=cls.config.category_vocab,
        )), cls.config)
        export = _load_export_module()
        artifact = Path(cls.tmpdir.name) / "champion-policy.pt"
        export.export_torchscript(export.build_exportable_module(model),
            export.make_random_inputs(cls.config, export.TRACE_BATCH, seed=7), artifact)
        cls.champion_artifact = artifact
        cls.champion_checkpoint = checkpoint
        cls.native = pokezero_search.NativeLeafModel(str(artifact), device="cpu",
            window=cls.config.window_size, tokens=cls.config.token_count,
            categorical_features=cls.config.categorical_feature_count,
            numeric_features=cls.config.numeric_feature_count)
        print("[champion gate] checkpoint_sha256=" + digest
              + " belief_source_hash=" + source.metadata.source_hash
              + " belief_cache_sha256=" + hashlib.sha256(source_bytes).hexdigest())

    def test_live_context_builds_same_worlds_without_root_or_order_stubs(self):
        """Qualify full dispatch on a fixed historical opening, not a strength panel.

        No cache loader, observation, fold, world constructor, request-order
        resolver, root encoder, or native search is stubbed. The configured
        runtime's independently rebuilt set universe must bind the champion.
        This is still not an immutable image receipt. Ground-truth teams create the test
        battle; sampled-mode policies never receive the opponent's packed team.
        """
        import random
        from pokezero.env import BattleStartOverride
        from pokezero.engine_search import EngineMctsConfig, EngineMctsPolicy
        from pokezero.golden_corpus import load_golden_corpus
        from pokezero.local_showdown import (DEFAULT_SHOWDOWN_ROOT,
            LocalShowdownConfig, LocalShowdownEnv, env_config_from_checkpoint_provenance)
        from pokezero.neural_policy import (category_vocab_from_model_config,
            observation_spec_from_model_config, feature_masks_from_model_config)
        from pokezero.dex import load_showdown_dex_cached
        from pokezero.policy import PolicyContext
        from pokezero.trajectory import BattleTrajectory
        from test_model_priors_search import COMMITTED_SAMPLE_DIR

        game = load_golden_corpus(COMMITTED_SAMPLE_DIR).games[0].record
        override = BattleStartOverride(player_teams={
            slot: game.true_teams[slot]["packed"] for slot in ("p1", "p2")
        }, observation_format_id="gen3randombattle")
        vocab = category_vocab_from_model_config(self.config, DEFAULT_SHOWDOWN_ROOT)
        env_config = env_config_from_checkpoint_provenance(
            LocalShowdownConfig(set_belief_source=True),
            feature_masks_from_model_config(self.config), context="champion live gate",
            required_specs=observation_spec_from_model_config(self.config), required_vocabs=vocab,
        )
        tables_path = Path(self.tmpdir.name) / "champion-tables.json"
        tables_path.write_text(self.tables_json)
        with LocalShowdownEnv(env_config) as env:
            self.assertEqual(env.belief_set_source_hash, self.champion_result.belief_set_source_hash)
            env.reset_with_start_override(seed=game.battle_seed, start_override=override)
            for player in ("p1", "p2"):
                observation = env.observe(player)
                public = env.public_materialization_state(player)
                self.assertEqual(public.replay.requests, {})
                self.assertEqual(public.self_request["side"]["id"], player)
                context = PolicyContext(player_id=player, decision_round_index=0,
                    battle_id=public.replay.battle_id, seed=game.battle_seed, format_id="gen3randombattle",
                    observation=observation, requested_players=env.requested_players(),
                    requested_observations={player: observation},
                    requested_legal_action_masks={player: tuple(observation.legal_action_mask)},
                    public_materialization_state=public,
                    trajectory=BattleTrajectory(battle_id=public.replay.battle_id,
                        format_id="gen3randombattle", seed=game.battle_seed, steps=[], terminal=None, metadata={}))
                for fixed in (True, False):
                    compared_worlds = []
                    for enabled in (False, True):
                        with self.subTest(player=player, fixed=fixed, enabled=enabled):
                            worlds = []
                            policy = EngineMctsPolicy(
                                dex=load_showdown_dex_cached(DEFAULT_SHOWDOWN_ROOT),
                                set_source=self.champion_source, fixed_override=override if fixed else None,
                                world_observer=lambda context, world, state: worlds.append(state.to_string()),
                                config=EngineMctsConfig(worlds=1, search_sims=16, search_batch=1,
                                    search_depth=2, leaf_eval="model", strict_fallbacks=True,
                                    model_path=str(self.champion_artifact),
                                    checkpoint_path=str(self.champion_checkpoint), tables_path=str(tables_path),
                                    policy_opponent=enabled, policy_opponent_seed=103 if enabled else None))
                            decision = policy.select_action_with_context(context, rng=random.Random(7))
                            self.assertTrue(observation.legal_action_mask[decision.action_index])
                            self.assertEqual(policy.stats.worlds_attempted, 1)
                            self.assertEqual(policy.stats.worlds_searched, 1)
                            self.assertEqual(policy.stats.total_iterations, 16)
                            self.assertEqual(len(worlds), 1)
                            compared_worlds.append(worlds)
                            metadata = decision.metadata["engine_mcts"]
                            if enabled:
                                invocation = metadata["policy_opponent"]["native_invocations"][0]
                                self.assertGreater(invocation["evaluations"], 0)
                                self.assertEqual(policy.stats.to_dict()["policy_opponent"]["evaluations"],
                                                 invocation["evaluations"])
                            else:
                                self.assertNotIn("policy_opponent", metadata)
                                self.assertNotIn("policy_opponent", policy.stats.to_dict())
                    self.assertEqual(compared_worlds[0], compared_worlds[1])

    def test_raw_profile_replays_live_champion_root_and_uses_one_forward(self):
        """Actual replay/observation/own-head path, not persisted input tensors."""
        import random
        from pokezero.local_showdown import DEFAULT_SHOWDOWN_ROOT
        from pokezero.mcts_eval.manifest import SearchConfig
        from pokezero.mcts_eval.policy_opponent_profile import (
            _LiveRawPolicyTimingDecider, validate_selection,
        )
        from pokezero.mcts_eval.resolver import resolve_checkpoint_contract
        from pokezero.neural_policy import load_transformer_policy
        from pokezero.public_decision_corpus import PublicDecisionRecord, PublicObservation

        contract = resolve_checkpoint_contract(str(self.champion_checkpoint),
            expected_sha256="0fd095923b4ac7e05d6e2b3ccab9c1e6869dff4893c2dae456caff10dce690be",
            model_device="cpu", showdown_root=str(DEFAULT_SHOWDOWN_ROOT))
        decider = _LiveRawPolicyTimingDecider(contract, str(DEFAULT_SHOWDOWN_ROOT))
        try:
            seed = 2026100100
            config = SearchConfig(depth=2, sims=16, batch=1, worlds=1)
            for player in ("p1", "p2"):
                decider._env.reset(seed=seed)
                observation = decider._env.observe(player)
                canonical = load_transformer_policy(str(self.champion_checkpoint), device="cpu",
                    deterministic=True, exploration_epsilon=0., sampling_temperature=1., family_gated_selection=False)
                canonical.record_policy_distribution = True
                expected = canonical.select_action(observation, rng=random.Random(7))
                record = PublicDecisionRecord(
                    decision_id=hashlib.sha256(f"raw-profile-live-gate:{seed}:{player}".encode()).hexdigest(),
                    battle_id=f"mcts-h2h-{seed}-{player}", seed=seed, format_id="gen3randombattle",
                    acting_player=player, turn_index=0, recorded_action_index=expected.action_index,
                    observation=PublicObservation.from_observation(observation), history=(),
                    current_legal_action_mask=tuple(observation.legal_action_mask),
                    public_resolved_action_rounds=(), public_belief_view=dict(observation.metadata["belief_view"]))
                prepared = decider.prepare_public_decision(record, config, public_action_rounds=(),
                    decision_rng_seed=7, source_requested_players=("p1", "p2"))
                model = decider._policies[config.config_id].policy.model
                forwards = []
                hook = model.register_forward_hook(lambda module, args, output: forwards.append(True))
                try:
                    telemetry = prepared()
                finally:
                    hook.remove()
                self.assertEqual(forwards, [True])
                self.assertEqual(telemetry["raw_policy"]["action_index"], expected.action_index)
                self.assertEqual(telemetry["raw_policy"]["policy_distribution"],
                                 list(expected.metadata["policy_distribution"]))
                validate_selection(telemetry, arm="raw_policy", mode="fixed_work", config=config,
                    mask=record.current_legal_action_mask, opponent_seed=71,
                    deadline_ms=1000, native_batch_guard_ms=64)
        finally:
            decider.close()


if __name__ == "__main__":
    unittest.main()
