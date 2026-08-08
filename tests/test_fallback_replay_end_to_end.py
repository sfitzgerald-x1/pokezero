"""End-to-end: play battles, read the addresses back, replay one, read the refusal.

The rest of the replay suite is unit-level, and unit-level is where this
codebase has repeatedly shipped a green suite over wrong output. So this test
runs the whole chain against a real Showdown simulator and a real search:

    play N battles  ->  EngineMctsStats.to_dict()  (the real producer)
      ->  fallback_addresses          (the real reader)
      ->  fallback_replay_spec        (the real resolver)
      ->  replay_fallback             (the real driver)
      ->  assert the recorded address REPRODUCES, with state attached.

Nothing here is a fixture: the addresses are whatever the engine actually
refused on, and the assertion is that replaying one of them lands on the same
decision. If the recorder, the id grammar, the resolver, the seat-parity rule,
the opponent policy or the belief-sampling stream is wrong, this fails; a mock
cannot make it pass.

**It does NOT cover the search parameters, and an earlier version of this
docstring claimed it did.** The refusals this band produces are construction-side
(`no_worlds_constructed`), and `engine_search.py:1049-1101` reads only `worlds`,
`sample_retry_factor` and the four `approximate_*` flags before refusing -- never
`search_sims`, `search_depth`, `c_puct`, `deep_ko_split` or `leaf_eval`. Deleting
the override splat from `engine_config_for` left this suite green, twice,
including with non-default recorded values. The search config is therefore
asserted directly, off `BattleRun.engine_config`, rather than inferred from a
reproduction.

It also pins the determinism claim the whole driver rests on for this harness,
by replaying twice and requiring the same refusal both times.

Skipped without a Showdown checkout, a patched ``poke_engine`` and the search
crate. The skip names what is missing, because a silent skip here would leave
the suite green with the only real check switched off.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import pytest

from _showdown_root import requires_showdown, showdown_root_str

pytest.importorskip("poke_engine", reason="end-to-end replay needs the patched engine")
pytest.importorskip("pokezero_search", reason="end-to-end replay needs the search crate")

from pokezero.fallback_replay import (  # noqa: E402
    ReplayOutcome,
    attach_refusal_recorder,
    replay_fallback,
    rollout_runner,
)
from pokezero.fallback_replay_spec import (  # noqa: E402
    FIDELITY_EXACT,
    ReplaySpec,
    resolve_corpus,
)

# Small on purpose: this is a correctness check on the chain, not a search-quality
# measurement. d2/s32/w2 keeps a 40-battle sweep to a few seconds while still
# producing real refusals -- the trapped class fires in this band.
# NON-DEFAULT on purpose, all three. `EngineMctsConfig` defaults c_puct=1.4 and
# deep_ko_split=True, and `rollout_runner` defaults max_decision_rounds=250 --
# so a fixture that used those values left the whole spec->config WIRING
# unexercised: deleting `**overrides` from the EngineMctsConfig call, or
# reverting RolloutConfig to a hardcoded literal, kept the suite green even
# WITH a Showdown checkout. The helpers were unit-tested; the wiring was not.
_CELL, _DEPTH, _SIMS, _WORLDS, _CPUCT = "hc-d2", 2, 32, 2, 1.1
_DEEP_KO_SPLIT = False
# `hc_depth_grid` exposes --max-decision-rounds but does NOT record it, so the
# spec cannot pin it and the runner must default -- and must SAY it defaulted.
# Asserting that note is how the RolloutConfig wiring gets covered without
# committing a shard shape no producer emits.
_MAX_ROUNDS = 250
_RAW_SPEC = "simple-legal"
_SEED_START, _GAMES = 600000, 40


def _play_and_write_shard(out_dir: Path) -> dict:
    """Run the battles and emit a document shaped exactly like hc_depth_grid's.

    Shaped from `scripts/hc_depth_grid.py:282-300`, and the `engine_stats` block
    is the producer's own `to_dict()` rather than a hand-written dict -- the
    address store under test is the real one.
    """
    from pokezero.collection import policy_from_spec, run_rollout_record_on_env
    from pokezero.dex import load_showdown_dex_cached
    from pokezero.engine_search import EngineMctsConfig, EngineMctsPolicy
    from pokezero.local_showdown import LocalShowdownConfig, LocalShowdownEnv
    from pokezero.randbat import load_gen3_randbat_source_cached
    from pokezero.rollout import RolloutConfig

    root = showdown_root_str()
    candidate = EngineMctsPolicy(
        dex=load_showdown_dex_cached(root),
        set_source=load_gen3_randbat_source_cached(root),
        config=EngineMctsConfig(
            leaf_eval="hp_fraction_crate",
            worlds=_WORLDS,
            search_sims=_SIMS,
            search_depth=_DEPTH,
            c_puct=_CPUCT,
            deep_ko_split=_DEEP_KO_SPLIT,
        ),
        policy_id=f"engine-mcts-hc-d{_DEPTH}-s{_SIMS}",
    )
    opponent = policy_from_spec(_RAW_SPEC)
    env = LocalShowdownEnv(LocalShowdownConfig(showdown_root=Path(root)))
    rollout_config = RolloutConfig(
        max_decision_rounds=_MAX_ROUNDS, format_id="gen3randombattle"
    )
    for offset in range(_GAMES):
        seed = _SEED_START + offset
        seat = "p1" if seed % 2 == 0 else "p2"
        other = "p2" if seat == "p1" else "p1"
        run_rollout_record_on_env(
            env=env,
            policies={seat: candidate, other: opponent},
            rollout_config=rollout_config,
            seed=seed,
            battle_id=f"hcgrid-{_CELL}-{seed}",
        )
    payload = {
        "cell": _CELL,
        "checkpoint": _RAW_SPEC,
        "raw_spec": _RAW_SPEC,
        "depth": _DEPTH,
        "sims": _SIMS,
        "worlds": _WORLDS,
        "c_puct": _CPUCT,
        "deep_ko_split": _DEEP_KO_SPLIT,
        "seed_start": _SEED_START,
        "games": _GAMES,
        "engine_stats": candidate.stats.to_dict(),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{_CELL}.json").write_text(json.dumps(payload, indent=2))
    return payload


@requires_showdown("end-to-end fallback replay needs a Showdown checkout")
class TestReplayChainAgainstRealBattles(unittest.TestCase):
    shard: dict
    specs: list[ReplaySpec]

    @classmethod
    def setUpClass(cls) -> None:
        import tempfile

        cls._tmp = tempfile.TemporaryDirectory()
        cls.shard = _play_and_write_shard(Path(cls._tmp.name))
        resolutions = resolve_corpus([Path(cls._tmp.name)])
        cls.specs = [r for r in resolutions if isinstance(r, ReplaySpec)]
        cls.unresolved = [r for r in resolutions if not isinstance(r, ReplaySpec)]

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def test_the_run_actually_refused_something(self):
        # The guard against a vacuous suite: with zero fallbacks every assertion
        # below is trivially satisfiable, and the whole file would be a no-op
        # that reads as coverage.
        stats = self.shard["engine_stats"]
        self.assertGreater(stats["decisions"], 100, "battles did not run")
        self.assertGreater(
            stats["fallback_decisions"],
            0,
            "this seed band produced no refusals, so the chain was never exercised; "
            "widen _GAMES or lower the search budget rather than deleting the test",
        )

    def test_every_address_resolves(self):
        self.assertEqual(self.unresolved, [])
        self.assertGreater(len(self.specs), 0)
        for spec in self.specs:
            self.assertEqual(spec.harness, "rollout-hc-grid")
            self.assertEqual(spec.fidelity, FIDELITY_EXACT)
            # Read back off the document, not defaulted.
            self.assertEqual((spec.engine_depth, spec.engine_sims), (_DEPTH, _SIMS))
            self.assertEqual(spec.engine_worlds, _WORLDS)
            self.assertEqual(spec.engine_c_puct, _CPUCT)
            self.assertEqual(spec.deep_ko_split, _DEEP_KO_SPLIT)
            self.assertIsNone(spec.max_decision_rounds)
            self.assertIn("max_decision_rounds", spec.missing)

    def test_the_recorded_address_replays(self):
        runner = rollout_runner(showdown_root=showdown_root_str())
        spec = self.specs[0]
        result = replay_fallback(spec, runner)
        self.assertIs(
            result.outcome,
            ReplayOutcome.REPRODUCED,
            f"replaying {spec.locator} gave {result.outcome}",
        )
        self.assertIsNone(result.fidelity_caveat)
        self.assertEqual(result.instrument_errors, ())
        self.assertTrue(result.trustworthy)
        # The runner consulted `rollout_settings` rather than a hardcoded
        # literal, and reported what it had to assume. Reverting RolloutConfig
        # to a literal drops this note.
        self.assertTrue(
            any("max_decision_rounds not recorded" in n for n in result.runner_notes),
            result.runner_notes,
        )

        record = result.record
        self.assertIsNotNone(record)
        # The point of the whole exercise: state, not a boolean. A refusal that
        # constructed no worlds must name the classes that closed them.
        self.assertEqual(record.reason, spec.reason)
        if record.reason == "no_worlds_constructed":
            self.assertEqual(record.worlds_constructed, 0)
            self.assertGreater(record.worlds_attempted, 0)
            self.assertGreater(len(record.world_failures), 0)
        self.assertEqual(
            record.decision_rng_seed, f"{spec.seed}:{spec.seat}:{spec.round}"
        )

    def test_replay_is_deterministic(self):
        # Half the justification for the only `exact` verdict in the codebase.
        #
        # The record comparison alone is NOT search-sensitive: these refusals
        # are construction-side, so `worlds_searched` is 0 and `engine_choices`
        # is empty in every record, and a nondeterministic search would not move
        # them. The sweep comparison in `_play_and_write_shard` is the
        # search-sensitive half (battle outcomes depend on the search), and the
        # explicit `total_iterations` assertion below makes this half
        # search-sensitive too, so the verdict does not rest on a comparison of
        # zeros.
        runner = rollout_runner(showdown_root=showdown_root_str())
        spec = self.specs[0]
        first = replay_fallback(spec, runner)
        second = replay_fallback(spec, runner)
        self.assertIs(first.outcome, second.outcome)
        self.assertEqual(
            [r.to_dict() for r in first.all_records],
            [r.to_dict() for r in second.all_records],
        )
        self.assertEqual(first.instrument_errors, second.instrument_errors)

    def test_the_search_itself_is_deterministic(self):
        # A search-sensitive quantity, measured twice. `total_iterations` is
        # charged per searched world, so it moves the moment the search does --
        # unlike everything on a construction-side refusal record.
        from pokezero.collection import policy_from_spec, run_rollout_record_on_env
        from pokezero.dex import load_showdown_dex_cached
        from pokezero.engine_search import EngineMctsPolicy
        from pokezero.fallback_replay import engine_config_for
        from pokezero.local_showdown import LocalShowdownConfig, LocalShowdownEnv
        from pokezero.randbat import load_gen3_randbat_source_cached
        from pokezero.rollout import RolloutConfig

        root = showdown_root_str()
        dex = load_showdown_dex_cached(root)
        set_source = load_gen3_randbat_source_cached(root)
        spec = self.specs[0]

        def play() -> dict:
            policy = EngineMctsPolicy(
                dex=dex, set_source=set_source, config=engine_config_for(spec)
            )
            seat = "p1" if spec.seed % 2 == 0 else "p2"
            other = "p2" if seat == "p1" else "p1"
            env = LocalShowdownEnv(LocalShowdownConfig(showdown_root=Path(root)))
            try:
                record = run_rollout_record_on_env(
                    env=env,
                    policies={seat: policy, other: policy_from_spec(_RAW_SPEC)},
                    rollout_config=RolloutConfig(
                        max_decision_rounds=_MAX_ROUNDS,
                        format_id="gen3randombattle",
                    ),
                    seed=spec.seed,
                    battle_id=spec.battle_id,
                )
            finally:
                env.close()
            stats = policy.stats.to_dict()
            return {
                "total_iterations": stats["total_iterations"],
                "worlds_searched": stats["worlds_searched"],
                "searched_decisions": stats["searched_decisions"],
                "winner": record.terminal.winner,
            }

        first, second = play(), play()
        self.assertGreater(
            first["total_iterations"],
            0,
            "no world was searched, so this measures nothing",
        )
        self.assertEqual(first, second)

    def test_recording_does_not_perturb_the_run(self):
        # An instrument that changes the run cannot be used to diagnose it. The
        # recorder consumes no RNG draw and touches no counter, so a recorded
        # run and an unrecorded one must refuse identically.
        from pokezero.collection import policy_from_spec, run_rollout_record_on_env
        from pokezero.dex import load_showdown_dex_cached
        from pokezero.engine_search import EngineMctsConfig, EngineMctsPolicy
        from pokezero.local_showdown import LocalShowdownConfig, LocalShowdownEnv
        from pokezero.randbat import load_gen3_randbat_source_cached
        from pokezero.rollout import RolloutConfig

        spec = self.specs[0]
        root = showdown_root_str()
        dex = load_showdown_dex_cached(root)
        set_source = load_gen3_randbat_source_cached(root)

        def play(record: bool) -> dict:
            policy = EngineMctsPolicy(
                dex=dex,
                set_source=set_source,
                config=EngineMctsConfig(
                    leaf_eval="hp_fraction_crate",
                    worlds=_WORLDS,
                    search_sims=_SIMS,
                    search_depth=_DEPTH,
                    c_puct=_CPUCT,
                    deep_ko_split=_DEEP_KO_SPLIT,
                ),
            )
            recorder = attach_refusal_recorder(policy) if record else None
            seat = "p1" if spec.seed % 2 == 0 else "p2"
            other = "p2" if seat == "p1" else "p1"
            run_rollout_record_on_env(
                env=LocalShowdownEnv(LocalShowdownConfig(showdown_root=Path(root))),
                policies={seat: policy, other: policy_from_spec(_RAW_SPEC)},
                rollout_config=RolloutConfig(
                    max_decision_rounds=_MAX_ROUNDS, format_id="gen3randombattle"
                ),
                seed=spec.seed,
                battle_id=spec.battle_id,
            )
            if recorder is not None:
                recorder.detach()
            return policy.stats.to_dict()

        recorded = play(True)
        plain = play(False)
        for key in (
            "decisions",
            "fallback_decisions",
            "fallback_reasons",
            "world_failure_reasons",
            "fallback_samples",
            "worlds_attempted",
            "worlds_constructed",
        ):
            self.assertEqual(recorded[key], plain[key], f"recorder perturbed {key}")

    def test_the_runner_searches_under_the_recorded_config(self):
        # The assertion the reproduction cannot make. `BattleRun.engine_config`
        # is what the runner actually handed the policy, so an inline config
        # that drops the recorded overrides fails here even though the address
        # still reproduces.
        from pokezero.fallback_replay import rollout_runner

        runner = rollout_runner(showdown_root=showdown_root_str())
        run = runner(self.specs[0])
        self.assertIsNotNone(run.engine_config)
        self.assertEqual(run.engine_config.c_puct, _CPUCT)
        self.assertEqual(run.engine_config.deep_ko_split, _DEEP_KO_SPLIT)
        self.assertEqual(run.engine_config.search_sims, _SIMS)
        self.assertEqual(run.engine_config.search_depth, _DEPTH)
        self.assertEqual(run.engine_config.worlds, _WORLDS)
        self.assertEqual(run.engine_config.leaf_eval, "hp_fraction_crate")
