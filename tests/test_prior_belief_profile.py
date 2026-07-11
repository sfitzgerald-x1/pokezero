import contextlib
import hashlib
import io
import json
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from pokezero.actions import ACTION_COUNT
from pokezero.belief import PlayerBeliefView, RevealedPokemonBelief
from pokezero.foulplay_capture import build_capture_arg_parser
from pokezero.neural_cli import gen3_category_vocabulary, main as neural_main
from pokezero.observation import PokeZeroObservationV0
from pokezero.prior_belief_profile import (
    CandidateValueEvaluation,
    MINIMUM_PROFILE_DECISIONS,
    PriorBeliefProfileConfig,
    WorldScenarioEvaluation,
    initial_candidate_value_top_two_margin,
    phase_for_turn,
    profile_public_corpus,
    profile_public_decisions,
    public_belief_sampling_profile,
    public_policy_context,
)
from pokezero.public_decision_corpus import (
    PublicActionIdentifier,
    PublicDecisionCorpus,
    PublicDecisionCorpusWriter,
    PublicDecisionRecord,
    PublicObservation,
    PublicResolvedActionRound,
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
    def test_prior_belief_cli_imports_the_category_vocabulary_loader(self) -> None:
        # The runtime profile must select the vocabulary family from checkpoint schema provenance.
        self.assertTrue(callable(gen3_category_vocabulary))

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
        round_actions = records[1].public_resolved_action_rounds[0].actions
        self.assertEqual(round_actions["p1"].event_id, "unresolved-public-action")
        self.assertEqual(round_actions["p2"].event_id, "unresolved-public-action")
        self.assertNotIn("action_index", str(records[1].to_dict()["public_resolved_action_rounds"]))
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

    def test_reader_rejects_request_local_public_action_fields(self) -> None:
        record = _record().to_dict()
        record["public_resolved_action_rounds"] = [
            {
                "turn_index": 0,
                "actions": {"p2": {"kind": "move", "move_id": "tackle", "action_index": 0}},
            }
        ]
        record["turn_index"] = 1
        with self.assertRaisesRegex(ValueError, "request-local"):
            PublicDecisionRecord.from_dict(record)

    def test_history_keeps_source_turns_across_asymmetric_request_gap(self) -> None:
        first = _observation(0, 1, metadata={"turn_number": 1})
        p2_private = _observation(0, metadata={"request": {"private": "do-not-read"}})
        third = _observation(0, 1, metadata={"turn_number": 3})
        current = _observation(0, 1, metadata={"turn_number": 4})
        trajectory = BattleTrajectory(battle_id="asymmetric-gap", format_id="gen3randombattle", seed=8)
        trajectory.append(TrajectoryStep("p1", 0, first, _mask(0, 1), 0))
        trajectory.append(TrajectoryStep("p2", 1, p2_private, _mask(0), 0))
        trajectory.append(TrajectoryStep("p1", 2, third, _mask(0, 1), 0))
        trajectory.append(TrajectoryStep("p1", 3, current, _mask(0, 1), 0))
        record = public_decision_records_from_trajectory(trajectory)[-1]

        context = public_policy_context(record)

        self.assertEqual([step.turn_index for step in context.trajectory.steps], [0, 2])
        self.assertEqual([step.observation.metadata["turn_number"] for step in context.trajectory.steps], [1, 3])
        self.assertEqual(context.observation.metadata["turn_number"], 4)

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

    def test_reader_can_select_a_deterministic_prefix_without_hashing_the_full_file(self) -> None:
        manifest = public_corpus_manifest(
            checkpoint_sha256="checkpoint-hash",
            belief_set_source_hash=None,
            capture_config={"opponent_legal_mask_mode": "hidden", "root_dirichlet_alpha": None},
        )
        records = (_record(turn_index=1), _record(turn_index=2), _record(turn_index=3))
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "public.jsonl"
            with PublicDecisionCorpusWriter(path, manifest=manifest) as writer:
                for record in records:
                    writer.append(record)
            with path.open("a", encoding="utf-8") as handle:
                handle.write("{\n")
            first = load_public_decision_corpus(path, max_decisions=2)
            second = load_public_decision_corpus(path, max_decisions=2)
            with self.assertRaisesRegex(ValueError, "invalid public corpus JSON"):
                load_public_decision_corpus(path)

        self.assertEqual(first.decisions, records[:2])
        self.assertEqual(first.selected_decision_limit, 2)
        expected = hashlib.sha256()
        for payload in (manifest, *(record.to_dict() for record in records[:2])):
            expected.update((json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode())
        self.assertEqual(first.selected_content_sha256, expected.hexdigest())
        self.assertEqual(first.selected_content_sha256, second.selected_content_sha256)
        self.assertIsNone(first.source_file_sha256)

    def test_uncapped_reader_keeps_the_raw_source_file_hash(self) -> None:
        manifest = public_corpus_manifest(
            checkpoint_sha256="checkpoint-hash",
            belief_set_source_hash=None,
            capture_config={"opponent_legal_mask_mode": "hidden", "root_dirichlet_alpha": None},
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "public.jsonl"
            with PublicDecisionCorpusWriter(path, manifest=manifest) as writer:
                writer.append(_record())
            expected = hashlib.sha256(path.read_bytes()).hexdigest()
            loaded = load_public_decision_corpus(path)
            source_hash = loaded.source_file_sha256
            compatibility_hash = loaded.corpus_sha256

        self.assertEqual(source_hash, expected)
        self.assertEqual(compatibility_hash, expected)

    def test_reader_rejects_non_positive_decision_prefix_limit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "public.jsonl"
            with self.assertRaisesRegex(ValueError, "max_decisions"):
                load_public_decision_corpus(path, max_decisions=0)

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

    def test_empty_replay_contexts_are_skipped_and_forced_margin_is_unavailable(self) -> None:
        record = _record()
        skipped = profile_public_decisions(
            [record],
            prior_evaluator=lambda _history: (1.0,) + (0.0,) * (ACTION_COUNT - 1),
            candidate_value_evaluator=lambda _record: (),
        )
        self.assertEqual(skipped["decision_count"], 0)
        self.assertEqual(skipped["skipped_decision_count"], 1)
        self.assertEqual(skipped["skipped_decision_rows"][0]["reason"], "no_public_replay_contexts")

        forced = profile_public_decisions(
            [record],
            prior_evaluator=lambda _history: (1.0,) + (0.0,) * (ACTION_COUNT - 1),
            candidate_value_evaluator=lambda _record: (
                WorldScenarioEvaluation(0, 0, "forced", 1.0, {0: 0.2}),
            ),
            config=PriorBeliefProfileConfig(entropy_thresholds=(0.0,), margin_thresholds=(0.3,)),
        )
        self.assertIsNone(forced["decision_rows"][0]["initial_candidate_value_top_two_margin"])
        margin_sweep = next(row for row in forced["threshold_sweeps"] if row["gate"] == "margin")
        self.assertEqual(margin_sweep["margin_eligible_selection_context_count"], 0)
        self.assertEqual(margin_sweep["forced_or_insufficient_context_count"], 1)
        self.assertEqual(margin_sweep["contested_count"], 0)
        self.assertIsNone(margin_sweep["contested_fraction"])
        decision_margin_sweep = next(
            row for row in forced["decision_normalized_threshold_sweeps"] if row["gate"] == "margin"
        )
        self.assertEqual(decision_margin_sweep["margin_eligible_decision_count"], 0)
        self.assertEqual(decision_margin_sweep["forced_or_insufficient_decision_count"], 1)
        self.assertIsNone(decision_margin_sweep["contested_fraction"])

    def test_unsupported_event_skip_is_named_and_counted_by_phase(self) -> None:
        record = _record()
        event_record = replace(
            record,
            turn_index=2,
            public_resolved_action_rounds=(
                PublicResolvedActionRound(
                    turn_index=0,
                    actions={"p2": PublicActionIdentifier(kind="event", event_id="unresolved-public-event")},
                ),
                PublicResolvedActionRound(
                    turn_index=1,
                    actions={"p1": PublicActionIdentifier(kind="move", move_id="tackle")},
                ),
            ),
        )
        report = profile_public_decisions(
            [event_record],
            prior_evaluator=lambda _history: (1.0,) + (0.0,) * (ACTION_COUNT - 1),
            candidate_value_evaluator=lambda _record: CandidateValueEvaluation(
                contexts=(),
                skip_reason="unsupported_public_event:unresolved-public-event",
                failure_reasons={"unsupported_public_event:unresolved-public-event": 1},
            ),
        )

        self.assertEqual(report["skipped_decision_rows"][0]["reason"], "unsupported_public_event:unresolved-public-event")
        early = next(row for row in report["representativeness"]["by_phase"] if row["phase"] == "early")
        self.assertEqual(early["event_bearing_prefix_count"], 1)
        self.assertEqual(early["unsupported_event_prefix_count"], 1)
        self.assertEqual(
            early["skip_reason_counts"],
            {"unsupported_public_event:unresolved-public-event": 1},
        )

    def test_decision_normalized_sweeps_do_not_multiply_multi_world_contexts(self) -> None:
        report = profile_public_decisions(
            [_record()],
            prior_evaluator=lambda _history: (0.6, 0.4) + (0.0,) * (ACTION_COUNT - 2),
            candidate_value_evaluator=lambda _record: (
                WorldScenarioEvaluation(0, 0, "world-0", 0.25, {0: 0.5, 1: 0.4}),
                WorldScenarioEvaluation(1, 0, "world-1", 0.75, {0: 0.9, 1: 0.2}),
            ),
            config=PriorBeliefProfileConfig(entropy_thresholds=(9.0,), margin_thresholds=(0.2,)),
        )

        context_margin = next(row for row in report["threshold_sweeps"] if row["gate"] == "margin")
        decision_margin = next(
            row for row in report["decision_normalized_threshold_sweeps"] if row["gate"] == "margin"
        )
        self.assertEqual(context_margin["rate_unit"], "per_selection_context")
        self.assertEqual(context_margin["selection_context_count"], 2)
        self.assertEqual(context_margin["contested_count"], 1)
        self.assertAlmostEqual(context_margin["contested_fraction"], 0.5)
        self.assertEqual(decision_margin["rate_unit"], "per_decision_normalized")
        self.assertEqual(decision_margin["decision_count"], 1)
        self.assertEqual(decision_margin["contested_count"], 0)
        self.assertEqual(decision_margin["contested_fraction"], 0.0)
        self.assertIn("scenario-weighted", decision_margin["decision_aggregation"])

    def test_malformed_candidate_context_skips_without_aborting_two_thousand_valid_decisions(self) -> None:
        valid = _record()
        malformed = replace(valid, battle_id="malformed-candidate")
        corpus = PublicDecisionCorpus(
            manifest={"schema_version": "pokezero.public-decision-corpus.v1"},
            decisions=(valid,) * MINIMUM_PROFILE_DECISIONS + (malformed,),
        )
        report = profile_public_corpus(
            corpus,
            prior_evaluator=lambda _history: (1.0,) + (0.0,) * (ACTION_COUNT - 1),
            candidate_value_evaluator=lambda record: (
                WorldScenarioEvaluation(0, 0, "malformed", 1.0, {3: 0.1})
                if record.battle_id == "malformed-candidate"
                else WorldScenarioEvaluation(0, 0, "valid", 1.0, {0: 0.1}),
            ),
        )

        self.assertEqual(report["decision_count"], MINIMUM_PROFILE_DECISIONS)
        self.assertEqual(report["skipped_decision_count"], 1)
        self.assertEqual(report["skipped_decision_rows"][0]["reason"], "candidate_legality_or_contract_failure")

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

    def test_corpus_floor_counts_successfully_profiled_decisions_not_rows(self) -> None:
        record = _record()
        corpus = PublicDecisionCorpus(
            manifest={"schema_version": "pokezero.public-decision-corpus.v1"},
            decisions=(record,) * MINIMUM_PROFILE_DECISIONS,
        )
        with self.assertRaisesRegex(ValueError, "successfully profiled"):
            profile_public_corpus(
                corpus,
                prior_evaluator=lambda _history: (1.0,) + (0.0,) * (ACTION_COUNT - 1),
                candidate_value_evaluator=lambda _record: (),
            )

    def test_profile_distinguishes_capped_selection_from_source_file_hash(self) -> None:
        record = _record()
        corpus = PublicDecisionCorpus(
            manifest={"schema_version": "pokezero.public-decision-corpus.v1"},
            decisions=(record,) * MINIMUM_PROFILE_DECISIONS,
            selected_content_sha256="selected-content-hash",
            selected_decision_limit=MINIMUM_PROFILE_DECISIONS,
        )
        report = profile_public_corpus(
            corpus,
            prior_evaluator=lambda _history: (1.0,) + (0.0,) * (ACTION_COUNT - 1),
            candidate_value_evaluator=lambda _record: (
                WorldScenarioEvaluation(0, 0, "selected", 1.0, {0: 0.1}),
            ),
        )

        self.assertIsNone(report["corpus_sha256"])
        self.assertEqual(report["selected_content_sha256"], "selected-content-hash")
        self.assertEqual(
            report["corpus_selection"],
            {"max_decisions": MINIMUM_PROFILE_DECISIONS, "selected_decision_count": MINIMUM_PROFILE_DECISIONS},
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
