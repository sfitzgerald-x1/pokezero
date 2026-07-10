import contextlib
import io
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from pokezero.actions import ACTION_COUNT
from pokezero.belief import PlayerBeliefView, RevealedPokemonBelief
from pokezero.foulplay_capture import build_capture_arg_parser
from pokezero.neural_cli import main as neural_main
from pokezero.observation import PokeZeroObservationV0
from pokezero.prior_belief_profile import (
    MINIMUM_PROFILE_DECISIONS,
    PriorBeliefProfileConfig,
    WorldScenarioEvaluation,
    initial_candidate_value_top_two_margin,
    phase_for_turn,
    profile_public_corpus,
    profile_public_decisions,
    public_belief_sampling_profile,
)
from pokezero.public_decision_corpus import (
    PublicDecisionCorpus,
    PublicDecisionCorpusWriter,
    PublicDecisionRecord,
    PublicObservation,
    load_public_decision_corpus,
    public_corpus_manifest,
    public_decision_id,
    public_decision_records_from_trajectory,
)
from pokezero.trajectory import BattleTrajectory, TrajectoryStep


def _mask(*legal_actions: int) -> tuple[bool, ...]:
    return tuple(index in set(legal_actions) for index in range(ACTION_COUNT))


def _belief_view(*, variants: int = 1) -> dict:
    return PlayerBeliefView(
        self_slot="p1",
        opponent_slot="p2",
        self_pokemon=(),
        opponent_pokemon=(
            RevealedPokemonBelief(
                showdown_slot="p2",
                species="Charizard",
                active=True,
                candidate_variants=tuple({"variant_id": f"charizard-{index}"} for index in range(variants)),
            ),
        ),
    ).to_overlay_payload()


def _observation(*legal_actions: int, variants: int = 1, metadata: dict | None = None) -> PokeZeroObservationV0:
    return PokeZeroObservationV0(
        categorical_ids=(),
        numeric_features=(),
        token_type_ids=(),
        attention_mask=(),
        legal_action_mask=_mask(*legal_actions),
        metadata={"belief_view": _belief_view(variants=variants), **(metadata or {})},
        schema_version="v2.2",
    )


def _record(*, turn_index: int = 1, variants: int = 1) -> PublicDecisionRecord:
    observation = PublicObservation.from_observation(_observation(0, 1, 2, variants=variants))
    prototype = PublicDecisionRecord(
        decision_id="pending",
        battle_id="public-profile-test",
        seed=7,
        format_id="gen3randombattle",
        acting_player="p1",
        turn_index=turn_index,
        recorded_action_index=0,
        observation=observation,
        history=(),
        current_legal_action_mask=_mask(0, 1, 2),
        public_resolved_action_rounds=(),
        public_belief_view=_belief_view(variants=variants),
    )
    return replace(prototype, decision_id=public_decision_id(prototype))


class PublicCorpusTest(unittest.TestCase):
    def test_public_roundtrip_and_private_opponent_leakage_invariance(self) -> None:
        p1_observation = _observation(0, 1, metadata={"self_team": []})
        private_p2_observation = _observation(
            0,
            metadata={
                "request": {"private": "first"},
                "opponent_legal_mask": [True] * ACTION_COUNT,
                "private_token": "first",
            },
        )
        trajectory = BattleTrajectory(battle_id="controlled-7", format_id="gen3randombattle", seed=7)
        trajectory.append(TrajectoryStep("p1", 0, p1_observation, _mask(0, 1), 0))
        trajectory.append(TrajectoryStep("p2", 0, private_p2_observation, _mask(0), 0))
        trajectory.append(TrajectoryStep("p1", 1, p1_observation, _mask(0, 1), 1))

        changed_private = replace(
            private_p2_observation,
            metadata={
                "request": {"private": "second", "payload": [1, 2, 3]},
                "opponent_legal_mask": [False] * ACTION_COUNT,
                "private_token": "second",
            },
        )
        changed = BattleTrajectory(battle_id="controlled-7", format_id="gen3randombattle", seed=7)
        changed.append(TrajectoryStep("p1", 0, p1_observation, _mask(0, 1), 0))
        changed.append(TrajectoryStep("p2", 0, changed_private, _mask(0), 0))
        changed.append(TrajectoryStep("p1", 1, p1_observation, _mask(0, 1), 1))

        records = public_decision_records_from_trajectory(trajectory)
        changed_records = public_decision_records_from_trajectory(changed)
        self.assertEqual(records, changed_records)
        encoded = records[1].to_dict()
        self.assertNotIn("request", str(encoded))
        self.assertNotIn("opponent_legal", str(encoded))
        self.assertEqual(records[1].public_resolved_action_rounds[0].actions, {"p1": 0, "p2": 0})
        profile_kwargs = {
            "prior_evaluator": lambda _history: (0.6, 0.4) + (0.0,) * (ACTION_COUNT - 2),
            "candidate_value_evaluator": lambda _record: (
                WorldScenarioEvaluation(0, 0, "hidden-prior:p2:0", 1.0, {0: 0.2, 1: 0.1}),
            ),
        }
        self.assertEqual(
            profile_public_decisions(records, **profile_kwargs)["profile_sha256"],
            profile_public_decisions(changed_records, **profile_kwargs)["profile_sha256"],
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "public.jsonl"
            manifest = public_corpus_manifest(
                checkpoint_sha256="checkpoint-hash",
                belief_set_source_hash=None,
                capture_config={"opponent_legal_mask_mode": "hidden", "root_dirichlet_alpha": None},
            )
            with PublicDecisionCorpusWriter(path, manifest=manifest) as writer:
                for record in records:
                    writer.append(record)
            loaded = load_public_decision_corpus(path)
        self.assertEqual(loaded.decisions, records)

    def test_reader_rejects_private_payload_field(self) -> None:
        record = _record().to_dict()
        record["opponent_observation"] = {"secret": 1}
        with self.assertRaisesRegex(ValueError, "forbidden private field"):
            PublicDecisionRecord.from_dict(record)

    def test_writer_appends_controlled_seed_bands_without_duplicate_decisions(self) -> None:
        record = _record()
        manifest = public_corpus_manifest(
            checkpoint_sha256="checkpoint-hash",
            belief_set_source_hash=None,
            capture_config={"opponent_legal_mask_mode": "hidden", "root_dirichlet_alpha": None},
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "public.jsonl"
            with PublicDecisionCorpusWriter(path, manifest=manifest) as writer:
                self.assertEqual(writer.append(record), 1)
            with PublicDecisionCorpusWriter(path, manifest=manifest, append=True) as writer:
                self.assertEqual(writer.append(record), 0)
            self.assertEqual(len(load_public_decision_corpus(path).decisions), 1)

    def test_foulplay_capture_parser_exposes_public_corpus_sidecar(self) -> None:
        args = build_capture_arg_parser().parse_args(
            ["--checkpoint", "checkpoint.pt", "--out", "rollouts.jsonl", "--public-decision-corpus-out", "public.jsonl"]
        )
        self.assertEqual(args.public_decision_corpus_out, Path("public.jsonl"))
        self.assertFalse(args.append_public_decision_corpus)


class PriorBeliefProfileTest(unittest.TestCase):
    def test_metrics_and_selection_context_rows_use_raw_priors(self) -> None:
        record = _record(variants=2)
        report = profile_public_decisions(
            [record],
            prior_evaluator=lambda _history: (0.8, 0.15, 0.05) + (0.0,) * (ACTION_COUNT - 3),
            candidate_value_evaluator=lambda _record: (
                WorldScenarioEvaluation(
                    world_index=0,
                    scenario_index=0,
                    scenario_label="hidden-prior:p2:0",
                    scenario_weight=1.0,
                    candidate_values={0: 0.8, 1: 0.5, 2: 0.1},
                ),
            ),
            config=PriorBeliefProfileConfig(
                entropy_thresholds=(0.5,),
                margin_thresholds=(0.3,),
                world_sample_cap=4,
            ),
        )
        row = report["decision_rows"][0]
        self.assertEqual(row["top1_action_index"], 0)
        self.assertEqual(row["top2_action_index"], 1)
        self.assertAlmostEqual(row["top2_prior_mass"], 0.95)
        self.assertGreater(row["normalized_policy_entropy"], 0.0)
        self.assertLess(row["normalized_policy_entropy"], 1.0)
        self.assertAlmostEqual(row["initial_candidate_value_top_two_margin"], 0.3)
        self.assertEqual(row["belief_combination_count"], 2)
        self.assertAlmostEqual(row["belief_uncertainty_bits"], 1.0)
        self.assertEqual(row["resolved_dynamic_k"], 2)
        context = report["selection_context_rows"][0]
        self.assertEqual(context["selection_gate_inputs"]["opponent_legal_mask_mode"], "hidden")
        self.assertFalse(context["selection_gate_inputs"]["root_noise_enabled"])
        self.assertAlmostEqual(context["initial_candidate_value_top_two_margin"], 0.3)
        self.assertIn("profile_sha256", report)

    def test_phase_boundaries_and_candidate_margin_tie_breaking(self) -> None:
        self.assertEqual(phase_for_turn(0), "early")
        self.assertEqual(phase_for_turn(5), "early")
        self.assertEqual(phase_for_turn(6), "mid")
        self.assertEqual(phase_for_turn(15), "mid")
        self.assertEqual(phase_for_turn(16), "late")
        # The margin uses the same descending-value, ascending-action tie order
        # as ValueBranchSearchResult.best_candidate/root candidate selection.
        margin, top1, top2 = initial_candidate_value_top_two_margin({3: 0.5, 1: 0.5, 2: 0.4})
        self.assertAlmostEqual(top1, 0.5)
        self.assertAlmostEqual(top2, 0.5)
        self.assertEqual(margin, 0.0)

    def test_dynamic_belief_profile_uses_public_variant_count_only(self) -> None:
        profile = public_belief_sampling_profile(_record(variants=3), sample_cap=2, set_source=None)
        self.assertEqual(profile.combination_count, 3)
        self.assertEqual(profile.sample_count, 2)
        self.assertAlmostEqual(profile.uncertainty_bits, 1.584962500721156)
        self.assertEqual(profile.uncertain_slot_count, 1)

    def test_corpus_floor_rejects_less_than_two_thousand_decisions(self) -> None:
        record = _record()
        corpus = PublicDecisionCorpus(
            manifest={"schema_version": "pokezero.public-decision-corpus.v1"},
            decisions=(record,),
        )
        with self.assertRaisesRegex(ValueError, str(MINIMUM_PROFILE_DECISIONS)):
            profile_public_corpus(
                corpus,
                prior_evaluator=lambda _history: (1.0,) + (0.0,) * (ACTION_COUNT - 1),
                candidate_value_evaluator=lambda _record: (
                    WorldScenarioEvaluation(0, 0, "single", 1.0, {0: 0.0}),
                ),
            )

    def test_cli_rejects_privileged_mask_mode_before_opening_inputs(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()):
            exit_code = neural_main(
                [
                    "prior-belief-profile",
                    "--corpus",
                    "missing.jsonl",
                    "--checkpoint",
                    "missing.pt",
                    "--showdown-root",
                    "missing-showdown",
                    "--opponent-legal-mask-mode",
                    "privileged",
                ]
            )
        self.assertEqual(exit_code, 1)


if __name__ == "__main__":
    unittest.main()
