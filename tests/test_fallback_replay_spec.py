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
from pathlib import Path

import pytest

from pokezero.fallback_addresses import FallbackAddress
from pokezero.fallback_replay_spec import (
    FIDELITY_EXACT,
    FIDELITY_OPPONENT_UNPINNED,
    FIDELITY_UNDERSPECIFIED,
    FIDELITY_UNVERIFIED,
    _PINNABLE_FIELDS,
    HARNESS_FOULPLAY_BRIDGE,
    HARNESS_ROLLOUT_ACCEPTANCE,
    HARNESS_ROLLOUT_HC_GRID,
    HARNESS_ROLLOUT_K0_GRID,
    ReplaySpec,
    UnresolvedAddress,
    BattleIdGrammar,
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
        # The earlier version asserted only that the bridge prefix resolves to
        # the bridge, which no registered grammar contests -- it passed under
        # `matches[0]`, under `min(...)`, and under any ordering. Contest it:
        # register two grammars where one prefix extends the other and the
        # SHORTER one is registered first, so first-match and shortest-match
        # both give the wrong answer.
        import pokezero.fallback_replay_spec as module

        short = BattleIdGrammar("short-harness", "grid-")
        long_ = BattleIdGrammar("long-harness", "grid-hc-")
        original = module.GRAMMARS
        module.GRAMMARS = (short, long_)
        try:
            grammar = battle_id_grammar("grid-hc-600000")
            assert grammar is not None
            assert grammar.harness == "long-harness"
        finally:
            module.GRAMMARS = original

    def test_grammar_lookup_is_by_prefix_not_by_membership(self):
        assert battle_id_grammar("battle-gen3randombattle-controlled-1").harness == (
            HARNESS_FOULPLAY_BRIDGE
        )
        assert battle_id_grammar("prefixed-accept-600004-p1") is None


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

    def test_fidelity_notes_name_the_mechanism_that_actually_holds(self):
        # These notes are surfaced verbatim as `ReplayResult.fidelity_caveat`,
        # so a wrong mechanism here is a wrong mechanism in the artifact. The
        # earlier revision blamed the wall-clock budget, which is refutable:
        # `monte_carlo_tree_search(iterations=N)` exists, and switching to it
        # would NOT make the opponent reproducible. Measured: five runs at
        # iterations=4000 on one captured state, five distinct visit
        # distributions. The cause is the unseeded chance sampler.
        spec = resolve_address(_address(), _foulplay_sidecar())
        assert isinstance(spec, ReplaySpec)
        assert spec.fidelity == FIDELITY_OPPONENT_UNPINNED
        assert not spec.replayable_exactly
        joined = " ".join(spec.fidelity_notes)
        assert "FIXED ITERATION COUNT" in joined
        assert "sample_node" in joined
        # ...and it must say the obvious remedy does not work, because a reader
        # who reaches for `iterations=` will otherwise believe it does.
        assert "does NOT fix this" in joined
        # The refuted claim must be gone, not merely supplemented: `random.choices`
        # consumes exactly one draw whatever the weights, so the stream is not
        # what desynchronises.
        assert "does not desynchronise" in joined
        assert "wall-clock budget" not in joined

    def test_device_is_reported_missing_not_defaulted(self):
        spec = resolve_address(_address(), _foulplay_sidecar())
        assert isinstance(spec, ReplaySpec)
        assert spec.device is None
        assert "device" in spec.missing

    def test_missing_names_every_unpinned_field(self):
        # The contract on `ReplaySpec.missing` is "a driver must not discover an
        # omission by getting None". The earlier hand-maintained per-reader
        # lists broke it silently. Derived now, so this is checkable in general.
        spec = resolve_address(_address(), _foulplay_sidecar())
        assert isinstance(spec, ReplaySpec)
        unpinned = {
            name
            for name in _PINNABLE_FIELDS
            if getattr(spec, name) is None
        }
        assert unpinned, "fixture pins everything; this test would be vacuous"
        assert unpinned <= set(spec.missing)
        # ...and nothing is named missing that is in fact pinned.
        assert all(getattr(spec, name) is None for name in spec.missing)

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
            _address("accept-search-600004-p1", round_index=7, seat="p1"), document
        )
        assert isinstance(spec, ReplaySpec)
        assert spec.seed == 600004
        assert (spec.engine_depth, spec.engine_sims) == (4, 2048)
        assert (spec.engine_batch, spec.engine_worlds) == (64, 8)
        # NOT `exact`: this harness runs `leaf_eval="model"`, whose byte identity
        # has never been measured here (unlike `hp_fraction_crate`) and which
        # runs a TorchScript forward. An earlier revision called it exact on the
        # strength of a miscited seeding line.
        assert spec.fidelity == FIDELITY_UNVERIFIED

    def test_hc_grid_reads_deep_ko_split(self):
        # A real producer setting (`hc_depth_grid.py:107` BooleanOptionalAction,
        # written :288, consumed `engine_search.py:1203`). Dropping it let a
        # `false` shard resolve `exact` and replay under the dataclass default
        # `true` -- a different search reported as the recorded one.
        document = _hc_grid_shard()
        document["deep_ko_split"] = False
        spec = resolve_address(
            _address("hcgrid-hc-d4-600000", round_index=12), document
        )
        assert isinstance(spec, ReplaySpec)
        assert spec.deep_ko_split is False
        assert "deep_ko_split" not in spec.missing

    def test_hc_grid_cell_must_match_the_document(self):
        # The cell selects the depth (`hc_depth_grid.py:196`). An hc-d8 address
        # read against an hc-d4 shard previously resolved with depth=4 and
        # fidelity=exact -- the collision `ReplaySpec.source` exists to name.
        document = _hc_grid_shard(cell="hc-d4", seed=600000)
        spec = resolve_address(
            _address("hcgrid-hc-d8-600000", round_index=12), document
        )
        assert isinstance(spec, UnresolvedAddress)
        assert "names cell 'hc-d8'" in spec.problem

    def test_hc_grid_seed_outside_the_band_is_unresolved(self):
        document = _hc_grid_shard(seed=600000)  # games: 1
        spec = resolve_address(
            _address("hcgrid-hc-d4-600400", round_index=12), document
        )
        assert isinstance(spec, UnresolvedAddress)
        assert "outside this shard's band" in spec.problem

    def test_acceptance_arm_must_match_the_document(self):
        document = {
            "schema_version": "pokezero.mcts-acceptance-shard.v1",
            "arm": "search",
            "config_id": "d4-s2048-b64-w8",
            "checkpoint": "checkpoints/k0.pt",
            "policy_stats": {"fallback_samples": {}},
        }
        spec = resolve_address(
            _address("accept-control-600004-p1", round_index=7, seat="p1"), document
        )
        assert isinstance(spec, UnresolvedAddress)
        assert "names arm 'control'" in spec.problem

    def test_acceptance_seat_must_match_the_battle_id(self):
        # The seat is IN the id, and both grid readers check theirs. This one did
        # not, so `accept-search-600004-p1` filed under seat p2 resolved `exact`.
        # The arm slice cannot catch it: it slices by the LENGTH of
        # `-{seed}-{seat}`, and "p1" and "p2" are the same length.
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
            _address("accept-search-600004-p1", round_index=7, seat="p2"), document
        )
        assert isinstance(spec, UnresolvedAddress)
        assert "names seat 'p1'" in spec.problem

    def test_acceptance_seed_outside_the_band_is_unresolved(self):
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
            _address("accept-search-700004-p1", round_index=7, seat="p1"), document
        )
        assert isinstance(spec, UnresolvedAddress)
        assert "outside this shard's band" in spec.problem

    def test_k0_grid_seed_outside_the_band_is_unresolved(self):
        document = {
            "arm": "search",
            "config": "d4-s1024-b64-w4",
            "per_game": [{"seed": 600002}],
            "checkpoint": "checkpoints/k0.pt",
            "depth": 4,
            "sims": 1024,
            "worlds": 4,
            "seed_start": 600000,
            "games": 8,
        }
        spec = resolve_address(_address("k0grid-700000", round_index=5), document)
        assert isinstance(spec, UnresolvedAddress)
        assert "outside this shard's band" in spec.problem

    def test_acceptance_unparseable_config_id_names_the_gap(self):
        # The control arm's config_id is "control-raw-v-raw" and carries no
        # numbers. Inventing defaults would replay a different search.
        document = {
            "schema_version": "pokezero.mcts-acceptance-shard.v1",
            "arm": "control-raw-v-raw",
            "config_id": "control-raw-v-raw",
            "checkpoint": "checkpoints/k0.pt",
            "pair_start": 600000,
            "pairs": 8,
            "policy_stats": {"fallback_samples": {}},
        }
        spec = resolve_address(
            _address("accept-control-raw-v-raw-600005-p2", round_index=3, seat="p2"),
            document,
        )
        assert isinstance(spec, ReplaySpec)
        assert spec.engine_sims is None
        assert {"engine_depth", "engine_sims", "engine_batch", "engine_worlds"} <= set(
            spec.missing
        )
        # AND it must not claim to be exactly replayable. The earlier `_fidelity`
        # read only the harness, so this spec -- which pins no search whatsoever
        # -- reported fidelity='exact' and replayable_exactly=True.
        assert spec.fidelity == FIDELITY_UNDERSPECIFIED
        assert not spec.replayable_exactly
        assert "engine_sims" in " ".join(spec.fidelity_notes)

    def test_unseeded_leaf_eval_is_not_exact(self):
        # `leaf_eval="hp_fraction"` calls poke-engine's own MCTS, measured
        # nondeterministic at a fixed iteration count. A pokezero-only harness
        # is therefore not sufficient for exactness -- the search engine decides.
        import pokezero.fallback_replay_spec as module

        fidelity, notes = module._fidelity(
            module.HARNESS_ROLLOUT_HC_GRID,
            {
                "leaf_eval": "hp_fraction",
                "engine_depth": 4,
                "engine_sims": 1024,
                "engine_worlds": 4,
            },
        )
        assert fidelity == FIDELITY_OPPONENT_UNPINNED
        assert "sample_node" in " ".join(notes)

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

    def test_int_fields_reject_booleans(self):
        # `isinstance(True, int)` is True in Python, so a shard with
        # `"sims": true` would otherwise resolve to sims=1 and replay a
        # one-simulation search reported as the recorded one. The guard exists
        # in `_as_int`; nothing asserted it.
        document = _hc_grid_shard()
        document["sims"] = True
        spec = resolve_address(
            _address("hcgrid-hc-d4-600000", round_index=12), document
        )
        assert isinstance(spec, ReplaySpec)
        assert spec.engine_sims is None
        assert "engine_sims" in spec.missing
        assert spec.fidelity == FIDELITY_UNDERSPECIFIED

    def test_cli_rejects_a_missing_path(self, tmp_path, capsys):
        assert main([str(tmp_path / "nope")]) == 2
        assert "does not exist" in capsys.readouterr().out

    def test_committed_fixtures_cover_three_real_grammars_plus_one_hypothetical(self):
        # Until this existed, no artifact IN THE REPO exercised the resolver:
        # `docs/audit_artifacts/hc-depth-grid-20260729/` predates #1178's
        # producer and carries `fallback_samples: {}`, so every number quoted
        # for this module rested on an uncommitted volume and CI checked none
        # of it. One small shard per grammar, shaped from the real layouts.
        fixtures = Path(__file__).parent / "fixtures" / "fallback_replay"
        resolutions = resolve_corpus([fixtures])
        specs = [r for r in resolutions if isinstance(r, ReplaySpec)]
        assert len(specs) == len(resolutions), [
            r.problem for r in resolutions if isinstance(r, UnresolvedAddress)
        ]
        assert {spec.harness for spec in specs} == {
            HARNESS_FOULPLAY_BRIDGE,
            HARNESS_ROLLOUT_ACCEPTANCE,
            HARNESS_ROLLOUT_HC_GRID,
            HARNESS_ROLLOUT_K0_GRID,
        }
        by_harness = {spec.harness: spec for spec in specs}
        # The two verdicts, both reachable from committed data.
        assert by_harness[HARNESS_FOULPLAY_BRIDGE].fidelity == (
            FIDELITY_OPPONENT_UNPINNED
        )
        assert by_harness[HARNESS_ROLLOUT_HC_GRID].fidelity == FIDELITY_EXACT
        # THREE of these shapes are emitted by a producer today. The k0 one is
        # not: `k0_grid_h2h.py:236-246` never calls `to_dict()`, so it writes no
        # `fallback_samples` and no such shard exists. Naming that here keeps the
        # fixture from reading as evidence that it does.
        k0 = json.loads((fixtures / "k0grid-search.json").read_text())
        assert "HYPOTHETICAL SHAPE" in k0["_fixture_note"]

    def test_cli_reports_no_addresses(self, tmp_path, capsys):
        (tmp_path / "empty.json").write_text(json.dumps({"schema_version": "x"}))
        assert main([str(tmp_path)]) == 1
        assert "no fallback addresses" in capsys.readouterr().out


class TestPropertiesAreHeldNotAsserted:
    """Three round-2 defects were comments claiming things no test enforced."""

    def test_pinnable_fields_is_derived_from_the_dataclass(self):
        # The round-2 comment said "adding a field to the spec cannot forget to
        # add it here" while the list was hand-written and the missing-fields
        # test iterated that same list -- self-referential, blind to drift.
        # Adding `engine_threads` to ReplaySpec left 84 tests green.
        import dataclasses

        import pokezero.fallback_replay_spec as module

        declared = {f.name for f in dataclasses.fields(ReplaySpec)}
        # An exact partition: every field is either config or explicitly not.
        assert set(_PINNABLE_FIELDS) | module._NON_CONFIG_FIELDS == declared
        assert not (set(_PINNABLE_FIELDS) & module._NON_CONFIG_FIELDS)
        # And the exclusion list may not name a field that no longer exists.
        assert module._NON_CONFIG_FIELDS <= declared

    def test_a_new_spec_field_is_pinnable_without_editing_a_list(self):
        # The direct form of the reviewer's mutant, run rather than asserted.
        import dataclasses

        import pokezero.fallback_replay_spec as module

        extended = dataclasses.make_dataclass(
            "ExtendedSpec",
            [("engine_threads", "int | None", dataclasses.field(default=None))],
            bases=(ReplaySpec,),
            frozen=True,
        )
        original = module.ReplaySpec
        module.ReplaySpec = extended
        try:
            assert "engine_threads" in module._derive_pinnable_fields()
        finally:
            module.ReplaySpec = original

    def test_an_uncheckable_seed_band_costs_the_exact_verdict(self):
        # Round 2's `_seed_band_problem` returned None when the band fields were
        # absent, with a comment claiming `missing` reported the gap. It cannot:
        # `seed_start` and `games` are not spec fields. So a shard with a start
        # and no count resolved `exact` with the band never checked.
        document = _hc_grid_shard(seed=600000)
        del document["games"]
        spec = resolve_address(
            _address("hcgrid-hc-d4-999998", round_index=12), document
        )
        assert isinstance(spec, ReplaySpec)
        assert spec.fidelity == FIDELITY_UNDERSPECIFIED
        assert "could not be checked" in " ".join(spec.fidelity_notes)
        assert "games" in " ".join(spec.fidelity_notes)

    def test_leaf_eval_is_read_from_the_shard_not_asserted(self):
        # Round 2 hardcoded it in all three readers, so the leaf-eval fidelity
        # branch was unreachable through `resolve_address` -- a latent wrong
        # answer the day `hc_depth_grid` grows a `--leaf-eval` flag.
        document = _hc_grid_shard()
        document["leaf_eval"] = "hp_fraction"
        spec = resolve_address(
            _address("hcgrid-hc-d4-600000", round_index=12), document
        )
        assert isinstance(spec, ReplaySpec)
        assert spec.leaf_eval == "hp_fraction"
        assert spec.fidelity == FIDELITY_OPPONENT_UNPINNED
        assert "sample_node" in " ".join(spec.fidelity_notes)

    def test_the_shipped_hc_grid_default_still_resolves_exact(self):
        # ...and reading it from the shard must not break the shard shape that
        # actually exists today, which records no `leaf_eval`.
        spec = resolve_address(
            _address("hcgrid-hc-d4-600000", round_index=12), _hc_grid_shard()
        )
        assert isinstance(spec, ReplaySpec)
        assert spec.leaf_eval == "hp_fraction_crate"
        assert spec.fidelity == FIDELITY_EXACT


class TestRoundFourRegressions:
    def test_the_band_caveat_survives_every_fidelity_verdict(self):
        # The round-3 fix checked `caveats` only after the `seeded-unverified`
        # and unmeasured-leaf branches had already returned, so for acceptance
        # and k0 -- two of the three self-play harnesses -- the caveat it exists
        # to carry was silently dropped. `missing` cannot cover: `pair_start`,
        # `pairs` and `games` are not spec fields.
        document = {
            "schema_version": "pokezero.mcts-acceptance-shard.v1",
            "arm": "search",
            "config_id": "d4-s2048-b64-w8",
            "checkpoint": "checkpoints/k0.pt",
            # no pair_start / pairs -> the band cannot be checked
            "policy_stats": {"fallback_samples": {}},
        }
        spec = resolve_address(
            _address("accept-search-600004-p1", round_index=7, seat="p1"), document
        )
        assert isinstance(spec, ReplaySpec)
        assert spec.fidelity == FIDELITY_UNVERIFIED  # not `exact`, so not the
        # branch the round-3 test happened to exercise
        assert "could not be checked" in " ".join(spec.fidelity_notes)

    def test_the_band_caveat_survives_the_unmeasured_leaf_verdict(self):
        document = _hc_grid_shard()
        document["leaf_eval"] = "hp_fraction"
        del document["games"]
        spec = resolve_address(
            _address("hcgrid-hc-d4-600000", round_index=12), document
        )
        assert isinstance(spec, ReplaySpec)
        assert spec.fidelity == FIDELITY_OPPONENT_UNPINNED
        assert "could not be checked" in " ".join(spec.fidelity_notes)

    def test_an_empty_leaf_eval_is_not_the_healthy_default(self):
        # `_as_str(...) or "hp_fraction_crate"` reads "" as the compiled-in
        # value -- the truthiness pattern this module condemns elsewhere. A
        # malformed recording must not resolve as a healthy one.
        document = _hc_grid_shard()
        document["leaf_eval"] = ""
        spec = resolve_address(
            _address("hcgrid-hc-d4-600000", round_index=12), document
        )
        assert isinstance(spec, ReplaySpec)
        assert spec.leaf_eval == ""
        assert spec.fidelity != FIDELITY_EXACT

    @pytest.mark.parametrize(
        ("config_id", "worlds"),
        [
            ("d4-s1024-b64-w8", 8),
            # M17: the checkpoint tag must be stripped before the digit scan, or
            # "w8@k1" is not a digit run, engine_worlds reads None, the spec goes
            # underspecified and the runner refuses a resolvable address.
            ("d4-s1024-b64-w8@k1", 8),
            ("d4-s1024-b64-w8@transformer-policy", 8),
        ],
    )
    def test_config_id_parses_with_and_without_a_checkpoint_tag(
        self, config_id, worlds
    ):
        document = {
            "schema_version": "pokezero.mcts-acceptance-shard.v1",
            "arm": "search",
            "config_id": config_id,
            "checkpoint": "checkpoints/k0.pt",
            "pair_start": 600000,
            "pairs": 8,
            "policy_stats": {"fallback_samples": {}},
        }
        spec = resolve_address(
            _address("accept-search-600004-p1", round_index=7, seat="p1"), document
        )
        assert isinstance(spec, ReplaySpec)
        assert spec.engine_worlds == worlds
        assert "engine_worlds" not in spec.missing


class TestFinalRoundPins:
    def test_the_underspecified_branch_keeps_its_caveats(self):
        # M67: the `*caveats` on the FIDELITY_UNDERSPECIFIED return was the last
        # unpinned one -- dropping it stayed green, because no existing input
        # produced an underspecified verdict AND a band caveat at the same time.
        # This one does: no `games` (band uncheckable) and no `sims` (a required
        # field unpinned).
        document = _hc_grid_shard(seed=600000)
        del document["games"]
        del document["sims"]
        spec = resolve_address(
            _address("hcgrid-hc-d4-600000", round_index=12), document
        )
        assert isinstance(spec, ReplaySpec)
        notes = " ".join(spec.fidelity_notes)
        assert spec.fidelity == FIDELITY_UNDERSPECIFIED
        assert "engine_sims" in notes          # the underspecified reason
        assert "could not be checked" in notes  # AND the caveat, not instead of

    def test_missing_never_names_something_that_is_not_a_spec_field(self):
        # `ReplaySpec.missing` is documented as naming spec FIELDS, and a driver
        # is told to decide what to do about each. A reader-supplied entry broke
        # that: a foul-play sidecar with no schedule put
        # "foulplay_random_seed_schedule.seeds" into `missing`, and any consumer
        # following the contract got AttributeError. That gap is a caveat now,
        # which is also what makes the foul-play `*caveats` splat reachable
        # instead of dead.
        document = _foulplay_sidecar()
        del document["foulplay_random_seed_schedule"]
        spec = resolve_address(_address(), document)
        assert isinstance(spec, ReplaySpec)
        assert set(spec.missing) <= set(_PINNABLE_FIELDS)
        for name in spec.missing:
            assert getattr(spec, name) is None  # would raise before the fix
        assert any("per-game seed" in note for note in spec.fidelity_notes)
