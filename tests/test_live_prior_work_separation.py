"""Model-only Python orchestration fixtures; NOT native gate qualification.

Exercise real live-fold advancement, world-loop retries, duplicate collapse,
chance seeds, dispatch and telemetry with an authored sampler/constructor and
fake native reports. No model, native engine or Showdown is loaded. In
particular these fixtures cannot establish budget-independent native priors
or qualify the scientific branch's model-tree HP override, absent from main.
"""
from copy import deepcopy
import json
import random
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tests import test_engine_search as fixtures

from pokezero.engine_search import EngineMctsConfig, EngineMctsPolicy
from pokezero.engine_world import EngineWorldUnsupported


class LivePriorWorkSeparationTests(unittest.TestCase):
    def run_fixture(self, work, pattern=("A", "B", "C", "D"), *,
                    unavailable=False, singleton=False, strict=True):
        policy = fixtures._policy()
        policy._config = EngineMctsConfig(
            worlds=4, leaf_eval="model", model_path="authored-model",
            checkpoint_path="authored-checkpoint", tables_path="authored-tables",
            search_sims=work, search_batch=16, search_depth=6,
            strict_fallbacks=strict, override_telemetry=True,
            root_prior_exact_aggregate=True,
        )
        policy._tables_json = "{}"
        draws, constructions, calls, decisions = [], [], [], []
        draw_index = 0
        boundary = 0

        def sample(*, context, rng, **kwargs):
            nonlocal draw_index
            label = pattern[draw_index % len(pattern)]
            draw_index += 1
            draws.append((context.decision_round_index, rng.getrandbits(32), label))
            return (None, "authored_unavailable") if label is None else (label, None)

        def construct(state, override, *, rng, **kwargs):
            constructions.append((boundary, override, rng.getrandbits(32)))
            if override == "refused":
                raise EngineWorldUnsupported("authored_refusal", "fixture construction unavailable")
            return SimpleNamespace(
                spec=override, party_species={"p1": ("rattata",), "p2": ("chansey",)},
                slot_sides={"p1": "side_one"},
            )

        def native(*args, **kwargs):
            calls.append((boundary, deepcopy(args), deepcopy(kwargs)))
            # Authored invariant priors are an INPUT to this fixture, not a
            # finding about native inference. Visits deliberately depend on work.
            p = {"A": .2, "B": .6, "C": .75, "D": .1}[args[0]]
            priors = [1.] if singleton else [p, 1-p]
            visits = [args[1]] if singleton else (
                [0, args[1]] if args[1] >= 1024 else [args[1], 0])
            arms = [(name, count, .5, None if unavailable else prior)
                    for name, count, prior in zip(("alpha", "beta"), visits, priors)]
            if not singleton and args[1] >= 1024:
                arms.reverse()  # Native reports sort by visits.
            return json.dumps(fixtures.RootDecisionTelemetryTest._report(
                arms, root_priors=None if unavailable else priors))

        fake_module = SimpleNamespace(FoldState=SimpleNamespace(
            from_payload=lambda payload: deepcopy(payload)))
        error = None
        lead = fixtures.LiveFoldAdvanceTests.LEAD
        round2 = fixtures.LiveFoldAdvanceTests.ROUND2
        with (
            patch.dict("sys.modules", {"pokezero_search": fake_module}),
            patch("pokezero.engine_search._gen3_randbat_belief_start_override_result", sample),
            patch("pokezero.engine_search.world_battle_spec", construct),
            patch("pokezero.engine_search.build_poke_engine_state",
                  side_effect=lambda spec, **kw: SimpleNamespace(to_string=lambda: spec)),
            patch.object(EngineMctsPolicy, "_native", return_value=SimpleNamespace(
                search_batched_multi_encoded=native)),
            patch.object(EngineMctsPolicy, "_validate_model_root_observation"),
            patch.object(EngineMctsPolicy, "_root_inputs_json", return_value="{}"),
        ):
            for boundary, lines in enumerate((lead, lead + round2)):
                context = fixtures.EarlyStopPolicyIntegrationTests._context()
                context.decision_round_index = boundary
                context.public_materialization_state = fixtures._FakePublicState(lines)
                if singleton:
                    context.observation.legal_action_mask = (True,) + (False,) * 8
                    context.observation.metadata["action_candidates"] = [
                        {"action_index": 0, "kind": "move", "legal": True, "move_id": "alpha"}]
                try:
                    decisions.append(policy.select_action_with_context(
                        context, rng=random.Random(83)))
                except fixtures.EngineSearchFallbackError as exc:
                    error = str(exc)
                    break
        return SimpleNamespace(policy=policy, draws=draws, constructions=constructions,
                               calls=calls, decisions=decisions, error=error)

    def compare_inputs(self, full, small):
        self.assertEqual(full.draws, small.draws)
        self.assertEqual(full.constructions, small.constructions)
        self.assertEqual(len(full.calls), len(small.calls))
        self.assertGreater(len(full.calls), 0)
        for (fb, fa, fk), (sb, sa, sk) in zip(full.calls, small.calls):
            self.assertEqual(fb, sb)
            self.assertEqual(fk, sk)
            self.assertEqual({i for i, (a, b) in enumerate(zip(fa, sa)) if a != b}, {1})
            self.assertEqual(len(fa), len(sa))
            self.assertEqual(fa[1], sa[1] * 64)
        self.assertEqual(full.policy._fold_consumed, small.policy._fold_consumed)

    @staticmethod
    def allocation(decision):
        return decision.metadata["engine_mcts"]["override"]["root_allocation"]

    def test_only_work_changes_at_native_seam_and_opening_fold_is_retained(self):
        full, small = self.run_fixture(1024), self.run_fixture(16)
        self.compare_inputs(full, small)
        self.assertEqual(len(full.calls), 8)
        self.assertEqual([d.action_index for d in full.decisions], [1, 1])
        self.assertEqual([d.action_index for d in small.decisions], [0, 0])
        for left, right in zip(full.decisions, small.decisions):
            lp = {a["action_index"]: a["model_prior_pre_display"]
                  for a in self.allocation(left)["arms"]}
            rp = {a["action_index"]: a["model_prior_pre_display"]
                  for a in self.allocation(right)["arms"]}
            self.assertEqual(lp, rp)
        self.assertEqual(full.policy.stats.fold_advanced_lines, 7)
        self.assertNotEqual(full.calls[0][1][6], full.calls[4][1][6])

    def test_duplicate_multiplicity_and_chance_seeds_survive_work_reduction(self):
        full = self.run_fixture(1024, ("A", "A", "A", "B"))
        small = self.run_fixture(16, ("A", "A", "A", "B"))
        self.compare_inputs(full, small)
        self.assertEqual([call[1][1] for call in small.calls], [48, 16, 48, 16])
        self.assertEqual(full.policy.stats.worlds_collapsed, 4)
        for result in (full, small):
            prior = {arm["action_index"]: arm["model_prior_pre_display"]
                     for arm in self.allocation(result.decisions[0])["arms"]}
            self.assertAlmostEqual(prior[0], .3)  # 3:1, not two unique worlds at 1:1.

    def test_sampler_and_constructor_refusals_keep_the_same_retry_law(self):
        pattern = (None, "refused", "A", "B", "C", "D")
        full, small = self.run_fixture(1024, pattern), self.run_fixture(16, pattern)
        self.compare_inputs(full, small)
        self.assertEqual(len(full.draws), 12)
        self.assertEqual(full.policy.stats.world_failure_reasons,
                         small.policy.stats.world_failure_reasons)
        self.assertEqual(sum(full.policy.stats.world_failure_reasons.values()), 4)

    def test_exhausted_constructor_refuses_without_a_fake_prior_or_native_call(self):
        full = self.run_fixture(1024, ("refused",))
        small = self.run_fixture(16, ("refused",))
        self.assertEqual(full.draws, small.draws)
        self.assertEqual(full.constructions, small.constructions)
        self.assertEqual(len(full.draws), 16)
        self.assertTrue(full.error)
        self.assertEqual(full.error, small.error)
        self.assertEqual(full.calls, [])
        self.assertEqual(small.calls, [])
        self.assertEqual(full.decisions, [])

    def test_missing_report_priors_remain_unmeasurable_at_both_work_amounts(self):
        for work in (1024, 16):
            result = self.run_fixture(work, unavailable=True)
            self.assertIsNone(result.error)
            self.assertEqual(len(result.decisions), 2)
            for decision in result.decisions:
                allocation = self.allocation(decision)
                self.assertFalse(allocation["prior_authority"])
                self.assertIsNotNone(allocation["prior_cause"])
                self.assertTrue(allocation["arms"])
                self.assertTrue(all(a["model_prior"] is None for a in allocation["arms"]))
                self.assertTrue(all(a["model_prior_pre_display"] is None
                                    for a in allocation["arms"]))

    def test_single_legal_action_retains_exact_prior(self):
        full = self.run_fixture(1024, singleton=True)
        small = self.run_fixture(16, singleton=True)
        self.compare_inputs(full, small)
        for result in (full, small):
            self.assertEqual([d.action_index for d in result.decisions], [0, 0])
            self.assertEqual(self.allocation(result.decisions[0])["arms"][0]
                             ["model_prior_pre_display"], 1.)


if __name__ == "__main__":
    unittest.main()
