"""Tests for address -> replay-spec resolution.

Every test here is written against the null-world question: *would this fail if
the implementation were replaced by something trivially wrong?* Several of them
name the trivially-wrong implementation they exist to kill, because in this
module the plausible wrong answer is usually a one-liner that passes a weaker
test -- ``rsplit("-", 1)`` for the seed, ``seed`` for the opponent seed, a
constant for the fidelity verdict.

The fixtures are SYNTHETIC but shaped from real documents: field names, nesting
and value types were read off a real era-64 sidecar and the committed
``docs/audit_artifacts/hc-depth-grid-20260729/hc-d4.json``. Paths and checkpoint
names here are placeholders on purpose -- the real ones name a cluster.
"""

from __future__ import annotations

import json

import pytest

from pokezero.fallback_addresses import FallbackAddress
from pokezero.fallback_replay_spec import (
    FIDELITY_EXACT,
    FIDELITY_OPPONENT_UNPINNED,
    HARNESS_FOULPLAY_BRIDGE,
    HARNESS_ROLLOUT_ACCEPTANCE,
    HARNESS_ROLLOUT_HC_GRID,
    HARNESS_ROLLOUT_K0_GRID,
    ReplaySpec,
    UnresolvedAddress,
    battle_id_grammar,
    main,
    parse_battle_id,
    resolve_address,
    resolve_corpus,
)

_SLEEP_TALK_KEY = (
    "crate_search: attribution-unsafe renderer branch rejected before tree/model fold: "
    "sleeptalk_called_unidentified:ambiguous_unrenderable:heal_zero_marker"
)


def _address(
    battle_id: str = "battle-gen3randombattle-controlled-8220001",
    *,
    round_index: int = 103,
    seat: str = "p1",
    reason: str = "crate_search_failed",
    key: str = _SLEEP_TALK_KEY,
    source: str = "fp-c-probe-00-p1.json",
) -> FallbackAddress:
    return FallbackAddress(
        battle_id=battle_id,
        round=round_index,
        seat=seat,
        reason=reason,
        key=key,
        source=source,
    )


def _foulplay_sidecar(
    *,
    seed_start: int = 8220000,
    games: int = 8,
    opponent_seeds: list[int] | None = None,
    seat: str = "p1",
    address: FallbackAddress | None = None,
) -> dict:
    """A bridge summary, shaped like ControlledFoulPlayBenchmarkResult.to_dict()."""
    address = address or _address()
    seeds = opponent_seeds if opponent_seeds is not None else list(
        range(seed_start, seed_start + games)
    )
    return {
        "schema_version": "pokezero.controlled-foulplay-benchmark.v1",
        "checkpoint": "checkpoints/trimmed.pt",
        "checkpoint_sha256": "0" * 64,
        "format_id": "gen3randombattle",
        "policy_id": "foundation-midscale-iter-3219+engine-mcts-d4-s1024",
        "policy_mode": "engine-mcts",
        "opponent_policy_id": "foul-play",
        "pokezero_player": seat,
        "foulplay_player": "p2" if seat == "p1" else "p1",
        "games": games,
        "seed_start": seed_start,
        "foulplay_random_seed": seeds[0],
        "max_decision_rounds": 250,
        "belief_set_source": False,
        "capture_driver": "checkpoint",
        "engine_mcts": {
            "decisions": 417,
            "fallback_decisions": 1,
            "depth": 4,
            "sims": 1024,
            "batch": 64,
            "worlds": 8,
            "opponent_priors": False,
            "policy_stats": {
                "fallback_samples": {
                    address.key: [
                        {
                            "battle_id": address.battle_id,
                            "round": address.round,
                            "seat": address.seat,
                            "reason": address.reason,
                        }
                    ]
                },
                "fallback_sample_addresses_dropped": 0,
            },
        },
        "root_puct": {"foulplay_search_time_ms": 1000},
        "foulplay_random_seed_schedule": {
            "count": len(seeds),
            "first_seed": seeds[0],
            "last_seed": seeds[-1],
            "mode": "per_game_incrementing",
            "seeds": seeds,
        },
    }


def _hc_grid_shard(*, cell: str = "hc-d4", seed: int = 600000) -> dict:
    return {
        "cell": cell,
        "checkpoint": "checkpoints/raw.pt",
        "raw_spec": "neural:checkpoints/raw.pt?deterministic=true&device=cpu",
        "depth": 4,
        "sims": 1024,
        "worlds": 4,
        "c_puct": 1.4,
        "deep_ko_split": True,
        "seed_start": seed,
        "games": 1,
        "engine_stats": {
            "fallback_samples": {
                "fallback:no_worlds_constructed": [
                    {
                        "battle_id": f"hcgrid-{cell}-{seed}",
                        "round": 12,
                        "seat": "p1" if seed % 2 == 0 else "p2",
                        "reason": "no_worlds_constructed",
                    }
                ]
            }
        },
    }


# --- battle-id grammar ------------------------------------------------------


class TestBattleIdGrammar:
    """The naive parse is `rsplit("-", 1)[-1]`. These say why it is wrong."""

    @pytest.mark.parametrize(
        ("battle_id", "harness", "seed"),
        [
            (
                "battle-gen3randombattle-controlled-8220001",
                HARNESS_FOULPLAY_BRIDGE,
                8220001,
            ),
            # THE null-world case: the seat trails the seed here, so a
            # last-field parse returns "p1" and a seed-typed parse returns None.
            ("accept-search-600004-p1", HARNESS_ROLLOUT_ACCEPTANCE, 600004),
            ("accept-control-raw-v-raw-600005-p2", HARNESS_ROLLOUT_ACCEPTANCE, 600005),
            # The cell embeds hyphens, so the seed is not field 2 either.
            ("hcgrid-hc-d4-600000", HARNESS_ROLLOUT_HC_GRID, 600000),
            ("k0grid-600003", HARNESS_ROLLOUT_K0_GRID, 600003),
        ],
    )
    def test_parses_seed_and_harness(self, battle_id, harness, seed):
        assert parse_battle_id(battle_id) == (harness, seed)

    def test_acceptance_seed_is_not_the_last_field(self):
        # Stated separately from the table so the property survives a table edit.
        battle_id = "accept-search-600004-p1"
        assert battle_id.rsplit("-", 1)[-1] == "p1"
        assert parse_battle_id(battle_id) == (HARNESS_ROLLOUT_ACCEPTANCE, 600004)

    @pytest.mark.parametrize(
        "battle_id",
        [
            "battle-gen3randombattle-controlled-notaseed",
            "battle-gen3randombattle-controlled-",
            "battle-gen3randombattle-controlled-+7",
            # A full-width digit string: `str.isdigit()` is True but this is not
            # a battle seed, and `int()` would happily accept it.
            "battle-gen3randombattle-controlled-７",
            "unheard-of-prefix-7",
            "accept-600004",  # no seat field, so field -2 is "accept"
        ],
    )
    def test_refuses_ids_it_cannot_read(self, battle_id):
        assert parse_battle_id(battle_id) is None

    def test_longest_prefix_wins(self):
        grammar = battle_id_grammar("battle-gen3randombattle-controlled-1")
        assert grammar is not None
        assert grammar.harness == HARNESS_FOULPLAY_BRIDGE


# --- foul-play sidecar resolution ------------------------------------------


class TestFoulplaySidecar:
    def test_resolves_the_full_config(self):
        spec = resolve_address(_address(), _foulplay_sidecar())
        assert isinstance(spec, ReplaySpec)
        assert spec.harness == HARNESS_FOULPLAY_BRIDGE
        assert spec.seed == 8220001
        assert (spec.engine_depth, spec.engine_sims) == (4, 1024)
        assert (spec.engine_batch, spec.engine_worlds) == (64, 8)
        assert spec.leaf_eval == "model"
        assert spec.format_id == "gen3randombattle"
        assert spec.max_decision_rounds == 250
        assert spec.belief_set_source is False
        assert spec.opponent_policy_id == "foul-play"
        assert spec.opponent_search_time_ms == 1000

    def test_decision_rng_seed_is_the_literal_bridge_expression(self):
        # foulplay_bridge.py:3541 -- reproduced verbatim, because a replay that
        # builds this string differently gets a different world sample and a
        # different refusal.
        spec = resolve_address(_address(), _foulplay_sidecar())
        assert isinstance(spec, ReplaySpec)
        assert spec.decision_rng_seed == "8220001:p1:103"
        assert spec.rng_regime == "per-decision"

    def test_opponent_seed_is_read_from_the_schedule_not_recomputed(self):
        # The trivially-wrong implementation is `opponent_seed = seed`. It passes
        # for the default schedule, so the fixture DECOUPLES the two bands: this
        # run set foulplay_random_seed=990000 against seed_start=8220000, which
        # foulplay_bridge.py:2375 says makes battle 8220001's opponent 990001.
        document = _foulplay_sidecar(
            opponent_seeds=list(range(990000, 990008)),
        )
        spec = resolve_address(_address(), document)
        assert isinstance(spec, ReplaySpec)
        assert spec.seed == 8220001
        assert spec.opponent_random_seed == 990001

    def test_seed_outside_the_shards_band_is_unresolved(self):
        # The address names a battle this shard did not play. Resolving it would
        # attach the wrong config -- the exact collision the locator's `source`
        # component exists to prevent.
        spec = resolve_address(
            _address("battle-gen3randombattle-controlled-9999999"),
            _foulplay_sidecar(),
        )
        assert isinstance(spec, UnresolvedAddress)
        assert "outside this shard's band" in spec.problem

    def test_wrong_seat_sidecar_is_unresolved(self):
        # A sidecar is one --pokezero-player invocation. A p2 address read
        # against the p1 sidecar would inherit p1's seat-specific config.
        spec = resolve_address(_address(seat="p2"), _foulplay_sidecar(seat="p1"))
        assert isinstance(spec, UnresolvedAddress)
        assert "sidecar's seat" in spec.problem

    def test_fidelity_is_opponent_unpinned_with_evidence(self):
        spec = resolve_address(_address(), _foulplay_sidecar())
        assert isinstance(spec, ReplaySpec)
        assert spec.fidelity == FIDELITY_OPPONENT_UNPINNED
        assert not spec.replayable_exactly
        joined = " ".join(spec.fidelity_notes)
        assert "wall-clock" in joined
        assert "random.choices" in joined

    def test_device_is_reported_missing_not_defaulted(self):
        spec = resolve_address(_address(), _foulplay_sidecar())
        assert isinstance(spec, ReplaySpec)
        assert spec.device is None
        assert "device" in spec.missing

    def test_paired_shard_defers_to_the_sidecar(self):
        # The merged paired shard carries the addresses under per_seat but keeps
        # depth/sims/batch/worlds only inside a config_id string. Guessing them
        # out of it is a reading dressed as a measurement.
        address = _address(source="fp-c-probe-00.json")
        paired = {
            "schema_version": "pokezero.foulplay-paired-shard.v1",
            "config_id": "d4-s1024-b64-w8@k0",
            "seed_start": 8220000,
            "pairs": 8,
            "per_seat": {
                "p1": {
                    "seat": "p1",
                    "policy_stats": {
                        "fallback_samples": {
                            address.key: [
                                {
                                    "battle_id": address.battle_id,
                                    "round": address.round,
                                    "seat": address.seat,
                                    "reason": address.reason,
                                }
                            ]
                        }
                    },
                }
            },
        }
        spec = resolve_address(address, paired)
        assert isinstance(spec, UnresolvedAddress)
        assert "sidecar" in spec.problem
        assert "fp-c-probe-00-p1.json" in spec.problem


# --- self-play writers ------------------------------------------------------


class TestSelfPlayWriters:
    def test_hc_grid_resolves_and_is_exact(self):
        document = _hc_grid_shard()
        address = _address(
            "hcgrid-hc-d4-600000",
            round_index=12,
            seat="p1",
            reason="no_worlds_constructed",
            key="fallback:no_worlds_constructed",
            source="hc-d4.json",
        )
        spec = resolve_address(address, document)
        assert isinstance(spec, ReplaySpec)
        assert spec.harness == HARNESS_ROLLOUT_HC_GRID
        assert spec.seed == 600000
        assert spec.leaf_eval == "hp_fraction_crate"
        assert spec.engine_c_puct == 1.4
        assert spec.device == "cpu"  # parsed out of raw_spec
        assert spec.fidelity == FIDELITY_EXACT
        assert spec.replayable_exactly

    def test_hc_grid_decision_rng_is_not_addressable(self):
        # The distinction that matters: the BATTLE is exact, the per-decision RNG
        # is not addressable, because RolloutDriver advances one stream per seat
        # (rollout.py:414-426). A spec that offered a decision_rng_seed here
        # would be inviting a wrong replay.
        spec = resolve_address(
            _address(
                "hcgrid-hc-d4-600000",
                round_index=12,
                key="fallback:no_worlds_constructed",
            ),
            _hc_grid_shard(),
        )
        assert isinstance(spec, ReplaySpec)
        assert spec.rng_regime == "per-battle-stream"
        assert spec.decision_rng_seed is None
        # ...and it is not reported as a shard omission, because no producer
        # change could supply it.
        assert "decision_rng_seed" not in spec.missing

    def test_hc_grid_seat_parity_violation_is_unresolved(self):
        # scripts/hc_depth_grid.py:235 makes seat a function of seed. An address
        # that disagrees was filed against a different run.
        document = _hc_grid_shard(seed=600000)
        spec = resolve_address(
            _address("hcgrid-hc-d4-600000", seat="p2", round_index=12),
            document,
        )
        assert isinstance(spec, UnresolvedAddress)
        assert "seed-parity" in spec.problem

    def test_acceptance_parses_config_id(self):
        document = {
            "schema_version": "pokezero.mcts-acceptance-shard.v1",
            "arm": "search",
            "config_id": "d4-s2048-b64-w8",
            "checkpoint": "checkpoints/k0.pt",
            "pair_start": 600000,
            "pairs": 8,
            "policy_stats": {"fallback_samples": {}},
        }
        spec = resolve_address(
            _address("accept-search-600004-p1", round_index=7), document
        )
        assert isinstance(spec, ReplaySpec)
        assert spec.seed == 600004
        assert (spec.engine_depth, spec.engine_sims) == (4, 2048)
        assert (spec.engine_batch, spec.engine_worlds) == (64, 8)
        assert spec.fidelity == FIDELITY_EXACT

    def test_acceptance_unparseable_config_id_names_the_gap(self):
        # The control arm's config_id is "control-raw-v-raw" and carries no
        # numbers. Inventing defaults would replay a different search.
        document = {
            "schema_version": "pokezero.mcts-acceptance-shard.v1",
            "config_id": "control-raw-v-raw",
            "checkpoint": "checkpoints/k0.pt",
            "policy_stats": {"fallback_samples": {}},
        }
        spec = resolve_address(
            _address("accept-control-raw-v-raw-600005-p2", round_index=3), document
        )
        assert isinstance(spec, ReplaySpec)
        assert spec.engine_sims is None
        assert {"engine_depth", "engine_sims", "engine_batch", "engine_worlds"} <= set(
            spec.missing
        )

    def test_k0_grid_layout_resolves(self):
        document = {
            "arm": "search",
            "config": "d4-s1024-b64-w4",
            "per_game": [{"seed": 600002, "search_seat": "p1"}],
            "checkpoint": "checkpoints/k0.pt",
            "checkpoint_sha256": "abc123",
            "depth": 4,
            "sims": 1024,
            "worlds": 4,
            "seed_start": 600000,
            "games": 8,
        }
        spec = resolve_address(
            _address("k0grid-600002", round_index=5, seat="p1"), document
        )
        assert isinstance(spec, ReplaySpec)
        assert spec.harness == HARNESS_ROLLOUT_K0_GRID
        assert spec.engine_batch == 64  # only available inside `config`
        assert spec.checkpoint_sha256 == "abc123"


# --- cross-checks between the two independent identifications ---------------


class TestHarnessDisagreement:
    def test_id_and_document_must_agree(self):
        # A foul-play battle id inside an hc-grid document. Either the shard was
        # misfiled or the walk reached a nested foreign block; both must refuse,
        # because the field layouts are different and neither reading is safe.
        spec = resolve_address(_address(), _hc_grid_shard())
        assert isinstance(spec, UnresolvedAddress)
        assert "battle id names harness" in spec.problem

    def test_unidentifiable_document_is_unresolved(self):
        spec = resolve_address(_address(), {"some": "unrelated json"})
        assert isinstance(spec, UnresolvedAddress)
        assert "does not identify its writer" in spec.problem


# --- corpus resolution and CLI ---------------------------------------------


class TestCorpus:
    def test_resolves_addresses_found_in_files(self, tmp_path):
        (tmp_path / "fp-c-probe-00-p1.json").write_text(
            json.dumps(_foulplay_sidecar())
        )
        (tmp_path / "hc-d4.json").write_text(json.dumps(_hc_grid_shard()))
        resolutions = resolve_corpus([tmp_path])
        specs = [r for r in resolutions if isinstance(r, ReplaySpec)]
        assert len(resolutions) == 2
        assert {spec.harness for spec in specs} == {
            HARNESS_FOULPLAY_BRIDGE,
            HARNESS_ROLLOUT_HC_GRID,
        }

    def test_overlapping_arguments_do_not_double_count(self):
        # The reader's `_iter_shard_paths` de-duplicates by resolved path; this
        # pins that we inherit that property rather than re-walking naively.
        import tempfile
        from pathlib import Path as _Path

        with tempfile.TemporaryDirectory() as raw:
            root = _Path(raw)
            shard = root / "fp-c-probe-00-p1.json"
            shard.write_text(json.dumps(_foulplay_sidecar()))
            assert len(resolve_corpus([root, shard])) == 1

    def test_cli_reports_unresolved_separately(self, tmp_path, capsys):
        # An unresolvable address must not read as an empty corpus, and must not
        # read as a resolved one either.
        document = _hc_grid_shard()
        document["engine_stats"]["fallback_samples"]["fallback:crate_search_failed"] = [
            {
                "battle_id": "battle-gen3randombattle-controlled-8220001",
                "round": 103,
                "seat": "p1",
                "reason": "crate_search_failed",
            }
        ]
        (tmp_path / "hc-d4.json").write_text(json.dumps(document))
        assert main([str(tmp_path)]) == 0
        out = capsys.readouterr().out
        assert "resolved: 1" in out
        assert "unresolved: 1" in out
        assert "UNRESOLVED (1)" in out

    def test_cli_json_out_round_trips(self, tmp_path):
        (tmp_path / "fp-c-probe-00-p1.json").write_text(
            json.dumps(_foulplay_sidecar())
        )
        out = tmp_path / "specs.json"
        assert main([str(tmp_path), "--json-out", str(out)]) == 0
        payload = json.loads(out.read_text())
        assert len(payload["specs"]) == 1
        assert payload["specs"][0]["decision_rng_seed"] == "8220001:p1:103"
        assert payload["specs"][0]["fidelity"] == FIDELITY_OPPONENT_UNPINNED

    def test_cli_rejects_a_missing_path(self, tmp_path, capsys):
        assert main([str(tmp_path / "nope")]) == 2
        assert "does not exist" in capsys.readouterr().out

    def test_cli_reports_no_addresses(self, tmp_path, capsys):
        (tmp_path / "empty.json").write_text(json.dumps({"schema_version": "x"}))
        assert main([str(tmp_path)]) == 1
        assert "no fallback addresses" in capsys.readouterr().out
