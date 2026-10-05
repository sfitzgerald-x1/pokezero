"""Real public-root materialization and ownership, not strength qualification."""

from dataclasses import replace
import random
import pickle
import unittest

from _showdown_root import requires_showdown, showdown_root
from pokezero.local_showdown import LocalShowdownConfig, LocalShowdownEnv
from pokezero.mcts_eval.paper_reference import Evaluation, ReferenceRefusal, Terminal
from pokezero.mcts_eval.paper_reference_factory import PublicRootWorldFactory
from pokezero.mcts_eval.paper_reference_showdown import decision_state
from pokezero.mcts_eval.paper_reference_runtime import PublicRootRequest
from pokezero.randbat import load_gen3_randbat_source_cached


def evaluate(observation):
    indices = tuple(i for i, legal in enumerate(observation.legal_action_mask) if legal)
    return indices, Evaluation((1 / len(indices),) * len(indices), 0.)


@requires_showdown()
class PublicRootFactoryTests(unittest.TestCase):
    def setUp(self):
        config = LocalShowdownConfig(showdown_root=showdown_root(), set_belief_source=True)
        self.live, self.sampled = LocalShowdownEnv(config), LocalShowdownEnv(config)
        self.addCleanup(self.live.close)
        self.addCleanup(self.sampled.close)
        self.live.reset(seed=2026100408)
        self.observation = self.live.observe("p1")
        self.public = self.live.public_materialization_state("p1")
        self.source = load_gen3_randbat_source_cached(showdown_root())

    def factory(self, **kwargs):
        arguments = dict(env=self.sampled, state=self.public, observation=self.observation,
            evaluator=evaluate, set_source=self.source)
        return PublicRootWorldFactory(**{**arguments, **kwargs})

    def test_new_world_each_call_preserves_player_known_root_and_releases_ownership(self):
        factory = self.factory()
        rng = random.Random(1)
        root = decision_state(self.observation, player="p1")
        for _ in range(3):
            world = factory(rng)
            self.assertEqual(world.frame().subject, root)
            with self.assertRaisesRegex(ReferenceRefusal, "still owned"):
                factory(rng)
            world.close()
            world.close()
            self.assertFalse(factory.active)
        self.assertEqual(len({r["packed_team_sha256"] for r in factory.receipts}), 3)
        self.assertTrue(all(r["status"] == "ROOT_VALIDATED" and r["released"] for r in factory.receipts))
        # No choices were ever submitted to the source battle.
        self.assertEqual(decision_state(self.live.observe("p1"), player="p1"), root)

    def test_reseed_message_after_materialization_cannot_fake_a_completed_step(self):
        factory = self.factory()
        world = factory(random.Random(1))
        try:
            frame = world.frame()
            outcome = world.advance(frame.subject.actions[0], frame.opponent_actions[0], random.Random(3))
            # Before the bridge fix, a stale "ready" arrived before choices;
            # no fresh requests existed and advance refused at frame().
            if not isinstance(outcome, Terminal):
                self.assertTrue(self.sampled.requested_players())
                self.assertTrue(outcome.subject is not None or outcome.opponent_actions)
            self.assertTrue(any(line.startswith("|move|") for line in self.sampled.protocol_lines))
        finally:
            world.close()

    def test_private_replay_requests_source_drift_and_missing_actor_opening_refuse(self):
        with self.assertRaisesRegex(ReferenceRefusal, "strip replay"):
            self.factory(state=replace(self.public, replay=replace(self.public.replay, requests={"p2": {}})))
        self.sampled._belief_set_source = None
        with self.assertRaisesRegex(ReferenceRefusal, "source binding"):
            self.factory()
        self.sampled._belief_set_source = self.source
        with self.assertRaisesRegex(ReferenceRefusal, "opening party"):
            self.factory(state=replace(self.public, self_initial_request={}))

    def test_worker_transport_strips_catalog_and_nonpublic_metadata_not_public_evidence(self):
        observation = replace(self.observation, metadata={**self.observation.metadata,
            "private_canary": "must-not-cross-processes"})
        request = PublicRootRequest.capture(self.public, observation)
        self.assertIs(self.public.belief_engine.set_source, self.source)
        self.assertNotIn("private_canary", request.observation.metadata)
        self.assertIsNone(request.state.belief_engine.set_source)
        wire = pickle.dumps(request)
        restored = pickle.loads(wire)
        self.assertEqual(restored.set_source_hash, self.source.metadata.source_hash)
        self.assertEqual(decision_state(restored.observation, player="p1"),
                         decision_state(self.observation, player="p1"))
        self.assertEqual(restored.state.belief_engine.snapshot(), self.public.belief_engine.snapshot())
        restored.state.belief_engine.set_source = self.source
        factory = self.factory(state=restored.state, observation=restored.observation)
        world = factory(random.Random(1))
        world.close()
        self.assertEqual(factory.receipts[0]["status"], "ROOT_VALIDATED")


if __name__ == "__main__":
    unittest.main()
