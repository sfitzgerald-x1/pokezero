"""Tests for the fallback address-store reader."""

from __future__ import annotations

import json

import pytest

from pokezero.fallback_addresses import (
    FallbackAddress,
    canonical_key,
    group_by_canonical_key,
    iter_shard_addresses,
    load_addresses,
    main,
    scan_corpus,
)
from pokezero.fallback_addresses import CorpusScan, _scan_document

# The six keys era 64 actually emitted for what is ONE class: the request says
# `trapped`, the sampled world does not trap. The foe's ability is a bystander --
# it names who was on the field, not what failed. Recorded verbatim so a change to
# the canonicalisation rule has to face the real strings.
_TRAPPED_KEYS = [
    "self_request_state_unsupported: self active request flags ['trapped'] constrain "
    f"legality beyond this construction (sampled world does not trap: foe ability '{ability}')"
    for ability in (
        "swarm",
        "chlorophyll",
        "stickyhold",
        "sandveil",
        "insomnia",
        "waterabsorb",
    )
]

_SLEEP_TALK_KEY = (
    "crate_search: attribution-unsafe renderer branch rejected before tree/model fold: "
    "sleeptalk_called_unidentified:ambiguous_unrenderable:heal_zero_marker"
)


def _entry(battle: str, round_index: int, seat: str, reason: str = "crate_search_failed"):
    return {"battle_id": battle, "round": round_index, "seat": seat, "reason": reason}


def _shard(samples):
    return {"engine_mcts": {"policy_stats": {"fallback_samples": samples}}}


class TestCanonicalKey:
    def test_collapses_the_six_trapped_keys_to_one(self):
        canonical = {canonical_key(key) for key in _TRAPPED_KEYS}
        assert len(canonical) == 1, canonical

    def test_retains_site_and_predicate(self):
        collapsed = canonical_key(_TRAPPED_KEYS[0])
        assert collapsed.startswith("self_request_state_unsupported:")
        assert "constrain legality beyond this construction" in collapsed
        assert "['trapped']" in collapsed  # the predicate is retained
        assert "foe ability" in collapsed
        # ...but not the bystander value itself.
        assert "swarm" not in collapsed

    def test_key_without_payload_is_unchanged(self):
        assert canonical_key(_SLEEP_TALK_KEY) == _SLEEP_TALK_KEY

    @pytest.mark.parametrize(
        "key",
        [
            "a: p+q, b: r",
            "materialization_blocker: baton-pass:substitute",
            "crate_search: x+y+z",
            "volatile_unsupported: side 'x': ['perish0', 'flashfire']",
        ],
    )
    def test_does_not_disturb_the_separator_grammar(self, key):
        # `,` joins independent slugs; `+` joins predicates within a slug where only
        # the first carries the family prefix. Rewriting either strips prefixes and
        # invents phantom bare rows. Canonicalisation must leave both alone.
        assert canonical_key(key) == key

    @pytest.mark.parametrize(
        ("left", "right"),
        [
            # Real recorded keys whose quoted operand IS the actionable content --
            # the volatile/condition/boost-key/weather that names the fix. Blanket
            # literal-stripping merged each of these pairs. This is the test that
            # must go red if canonicalisation ever over-merges again.
            (
                "volatile_unsupported: side 'p1': ['perish0']",
                "volatile_unsupported: side 'p1': ['flashfire']",
            ),
            (
                "boost_unsupported: side 'p1' boost key 'evasion'",
                "boost_unsupported: side 'p1' boost key 'accuracy'",
            ),
            (
                "side_condition_unsupported: side 'p1' condition 'spikes'",
                "side_condition_unsupported: side 'p1' condition 'lightscreen'",
            ),
            (
                "weather_unsupported: weather 'hail' has no Gen 3 engine mapping",
                "weather_unsupported: weather 'sandstorm' has no Gen 3 engine mapping",
            ),
            (
                "self_moveset_mismatch: p1: request-known move 'shadowball' is absent",
                "self_moveset_mismatch: p1: request-known move 'hiddenpower' is absent",
            ),
        ],
    )
    def test_actionable_operands_are_never_merged(self, left, right):
        assert canonical_key(left) != canonical_key(right)

    def test_distinct_families_do_not_collapse_together(self):
        assert canonical_key(_TRAPPED_KEYS[0]) != canonical_key(_SLEEP_TALK_KEY)

    def test_seat_spellings_are_unified_without_erasing_the_side(self):
        # The two producer spellings must agree in FORM, but the side must survive:
        # the slot in the key is the side that FAILED TO BUILD, which is independent
        # of the acting seat (`engine_world.py:484` builds both sides every decision).
        # hc-d4.json records 16 `side 'p1'` and 16 `side 'p2'` in one stats block.
        quoted = ("encore_move_unknown: side '%s' is encored but the locked move "
                  "cannot be determined")
        unquoted = "self_moveset_mismatch: %s: request-known move 'toxic' is absent"
        assert canonical_key(quoted % "p1") != canonical_key(quoted % "p2")
        assert canonical_key(unquoted % "p1") != canonical_key(unquoted % "p2")
        # ...and the unquoted spelling is rewritten onto the quoted one.
        assert canonical_key(unquoted % "p1").startswith(
            "self_moveset_mismatch: side 'p1':"
        )

    def test_real_trapping_abilities_are_not_bystanders(self):
        # shadowtag/arenatrap/magnetpull mean the foe DOES trap and the exemption
        # test declined it -- a different bug from "no trapper was sampled at all".
        trapped = ("self_request_state_unsupported: self active request flags "
                   "['trapped'] constrain legality beyond this construction "
                   "(sampled world does not trap: foe ability '%s')")
        for trapper in ("shadowtag", "arenatrap", "magnetpull"):
            assert canonical_key(trapped % trapper) != canonical_key(trapped % "swarm")
            assert trapper in canonical_key(trapped % trapper)
        # non-trappers still collapse
        assert canonical_key(trapped % "swarm") == canonical_key(trapped % "chlorophyll")

    def test_apostrophe_inside_an_interpolated_error_is_left_alone(self):
        # `crate_search: {reason}` interpolates arbitrary native error text. A naive
        # `'[^']*'` matches "'t materialize '", destroying the predicate and keeping
        # the payload.
        key = "crate_search: can't materialize 'Zapdos'"
        assert canonical_key(key) == key
        assert canonical_key(key) != canonical_key("crate_search: can't refuse 'Zapdos'")


class TestIterShardAddresses:
    def test_reads_battle_round_seat_and_reason(self):
        shard = _shard({_SLEEP_TALK_KEY: [_entry("battle-x-8220001", 103, "p1")]})
        (address,) = list(iter_shard_addresses(shard))
        assert address.battle_id == "battle-x-8220001"
        assert address.round == 103
        assert address.seat == "p1"
        assert address.reason == "crate_search_failed"
        assert address.key == _SLEEP_TALK_KEY
        assert address.locator == ("", "battle-x-8220001", 103, "p1")

    def test_per_seat_mirroring_is_not_double_counted(self):
        # The known 2x trap: the same stats block reachable by two paths.
        samples = {_SLEEP_TALK_KEY: [_entry("battle-x-8220001", 103, "p1")]}
        shard = {
            "engine_mcts": {"policy_stats": {"fallback_samples": samples}},
            "per_seat": {"p1": {"fallback_samples": samples, "policy_stats": {"fallback_samples": samples}}},
        }
        assert len(list(iter_shard_addresses(shard))) == 1

    def test_one_decision_recorded_under_several_keys_yields_several_addresses(self):
        # A decision is filed under its reason key AND every world-failure class in
        # its delta. Those are distinct classes, so they are distinct addresses --
        # but they share one replay locator.
        entry = _entry("battle-x-8220001", 7, "p2")
        shard = _shard({"fallback:crate_search_failed": [entry], _SLEEP_TALK_KEY: [entry]})
        addresses = list(iter_shard_addresses(shard))
        assert len(addresses) == 2
        assert len({a.locator for a in addresses}) == 1

    def test_skips_malformed_entries_without_failing(self):
        shard = _shard(
            {
                _SLEEP_TALK_KEY: [
                    _entry("battle-x-1", 1, "p1"),
                    {"battle_id": "battle-x-2", "seat": "p1"},  # no round
                    {"round": 3, "seat": "p1"},  # no battle id
                    {"battle_id": "battle-x-4", "round": True, "seat": "p1"},  # bool
                    "not-a-mapping",
                ],
                "empty": [],
                "not-a-list": {"battle_id": "b", "round": 1, "seat": "p1"},
            }
        )
        addresses = list(iter_shard_addresses(shard))
        assert [a.battle_id for a in addresses] == ["battle-x-1"]

    def test_missing_reason_is_empty_not_fatal(self):
        shard = _shard({_SLEEP_TALK_KEY: [{"battle_id": "b", "round": 2, "seat": "p1"}]})
        (address,) = list(iter_shard_addresses(shard))
        assert address.reason == ""

    def test_document_without_samples_yields_nothing(self):
        assert list(iter_shard_addresses({"engine_mcts": {"policy_stats": {}}})) == []


class TestLoadAddresses:
    def test_reads_a_directory_recursively_and_skips_non_shards(self, tmp_path):
        (tmp_path / "nested").mkdir()
        (tmp_path / "a-p1.json").write_text(
            json.dumps(_shard({_SLEEP_TALK_KEY: [_entry("battle-x-1", 1, "p1")]}))
        )
        (tmp_path / "nested" / "b-p2.json").write_text(
            json.dumps(_shard({_TRAPPED_KEYS[0]: [_entry("battle-x-2", 2, "p2")]}))
        )
        (tmp_path / "summary.json").write_text(json.dumps({"score": 1}))
        (tmp_path / "broken.json").write_text("{not json")
        (tmp_path / "notes.txt").write_text("ignored")

        addresses = load_addresses([tmp_path])
        assert sorted(a.battle_id for a in addresses) == ["battle-x-1", "battle-x-2"]
        # Relative to the argument root, not the basename: era directories carry
        # duplicate basenames that differ only by parent.
        assert {a.source for a in addresses} == {"a-p1.json", "nested/b-p2.json"}

    def test_accepts_explicit_file_paths(self, tmp_path):
        shard = tmp_path / "a-p1.json"
        shard.write_text(json.dumps(_shard({_SLEEP_TALK_KEY: [_entry("b", 1, "p1")]})))
        assert len(load_addresses([shard])) == 1


class TestGrouping:
    def test_groups_the_trapped_family_into_a_single_bucket(self):
        addresses = [
            FallbackAddress(battle_id=f"b{i}", round=i, seat="p1", reason="r", key=key)
            for i, key in enumerate(_TRAPPED_KEYS)
        ]
        grouped = group_by_canonical_key(addresses)
        assert len(grouped) == 1
        assert len(next(iter(grouped.values()))) == len(_TRAPPED_KEYS)


class TestCli:
    def test_reports_canonical_collapse_and_writes_json(self, tmp_path, capsys):
        samples = {key: [_entry(f"battle-{i}", i, "p1")] for i, key in enumerate(_TRAPPED_KEYS)}
        (tmp_path / "a-p1.json").write_text(json.dumps(_shard(samples)))
        out = tmp_path / "corpus.json"

        assert main([str(tmp_path), "--json-out", str(out)]) == 0

        captured = capsys.readouterr().out
        assert "addresses: 6" in captured
        assert "distinct canonical keys: 1" in captured

        payload = json.loads(out.read_text())
        assert len(payload) == 6
        assert len({row["canonical_key"] for row in payload}) == 1
        assert len({row["key"] for row in payload}) == 6

    def test_raw_keys_flag_shows_the_shattering(self, tmp_path, capsys):
        samples = {key: [_entry(f"battle-{i}", i, "p1")] for i, key in enumerate(_TRAPPED_KEYS)}
        (tmp_path / "a-p1.json").write_text(json.dumps(_shard(samples)))

        assert main([str(tmp_path), "--raw-keys"]) == 0
        assert "distinct raw keys: 6" in capsys.readouterr().out

    def test_empty_corpus_is_a_nonzero_exit(self, tmp_path, capsys):
        (tmp_path / "summary.json").write_text(json.dumps({"score": 1}))
        assert main([str(tmp_path)]) == 1
        assert "no fallback addresses found" in capsys.readouterr().out


class TestCorpusCompletenessAndFrequency:
    def test_dropped_addresses_are_surfaced_not_swallowed(self, tmp_path, capsys):
        # `fallback_sample_addresses_dropped` non-zero means occurrences exist with
        # no replayable address. Silent truncation reads as "covered everything".
        (tmp_path / "a-p1.json").write_text(
            json.dumps(
                {
                    "engine_mcts": {
                        "policy_stats": {
                            "fallback_samples": {_SLEEP_TALK_KEY: [_entry("b", 1, "p1")]},
                            "fallback_sample_addresses_dropped": 1001,
                        }
                    }
                }
            )
        )
        scan = scan_corpus([tmp_path])
        assert scan.addresses_dropped == 1001
        assert scan.complete is False

        assert main([str(tmp_path)]) == 0
        assert "INCOMPLETE CORPUS: 1001" in capsys.readouterr().out

    def test_complete_corpus_says_nothing_about_dropping(self, tmp_path, capsys):
        (tmp_path / "a-p1.json").write_text(
            json.dumps(_shard({_SLEEP_TALK_KEY: [_entry("b", 1, "p1")]}))
        )
        scan = scan_corpus([tmp_path])
        assert scan.complete is True
        assert main([str(tmp_path)]) == 0
        assert "INCOMPLETE" not in capsys.readouterr().out

    def test_ranking_uses_true_occurrences_not_capped_address_counts(self, tmp_path, capsys):
        # The inversion this guards: a class capped at few addresses but with a huge
        # true count must outrank a class with many raw variants and a small count.
        rare_but_huge = "materialization_blocker: baton-pass:substitute"
        common_but_small = "materialization_blocker: toxic-stage-unknown"
        (tmp_path / "a-p1.json").write_text(
            json.dumps(
                {
                    "engine_mcts": {
                        "policy_stats": {
                            "fallback_samples": {
                                rare_but_huge: [_entry("b", 1, "p1")],
                                common_but_small: [
                                    _entry("b", 2, "p1"),
                                    _entry("b", 3, "p1"),
                                    _entry("b", 4, "p1"),
                                ],
                            },
                            "world_failure_reasons": {
                                rare_but_huge: 1472,
                                common_but_small: 80,
                            },
                        }
                    }
                }
            )
        )
        scan = scan_corpus([tmp_path])
        assert scan.world_counts[rare_but_huge] == 1472
        assert scan.count_for(rare_but_huge) == (1472, "worlds")

        assert main([str(tmp_path)]) == 0
        out = capsys.readouterr().out
        # 1 address / 1472 occurrences must be printed ABOVE 3 addresses / 80.
        assert out.index(rare_but_huge) < out.index(common_but_small)

    def test_fallback_reasons_are_namespaced_into_true_counts(self, tmp_path):
        (tmp_path / "a-p1.json").write_text(
            json.dumps(
                {
                    "engine_mcts": {
                        "policy_stats": {
                            "fallback_samples": {
                                "fallback:choices_unmapped": [_entry("b", 1, "p1")]
                            },
                            "fallback_reasons": {"choices_unmapped": 63},
                        }
                    }
                }
            )
        )
        scan = scan_corpus([tmp_path])
        assert scan.decision_counts["fallback:choices_unmapped"] == 63
        assert scan.count_for("fallback:choices_unmapped") == (63, "decisions")

    def test_worlds_and_decisions_are_never_co_ranked(self, tmp_path, capsys):
        # A world count of 10000 must not be printed above a decision count of 5 as
        # though they were the same quantity.
        (tmp_path / "a-p1.json").write_text(
            json.dumps(
                {
                    "engine_mcts": {
                        "policy_stats": {
                            "fallback_samples": {
                                "big-world-class": [_entry("b", 1, "p1")],
                                "fallback:choices_unmapped": [_entry("b", 2, "p1")],
                            },
                            "world_failure_reasons": {"big-world-class": 10000},
                            "fallback_reasons": {"choices_unmapped": 5},
                        }
                    }
                }
            )
        )
        assert main([str(tmp_path)]) == 0
        out = capsys.readouterr().out
        assert "ordered by true decisions" in out
        assert "ordered by true worlds" in out
        # Decisions section comes first and the world class is not inside it.
        decisions_block = out.split("ordered by true decisions")[1].split("ordered by true worlds")[0]
        assert "big-world-class" not in decisions_block


class TestPathHandling:
    def test_overlapping_arguments_do_not_double_count(self, tmp_path):
        shard = tmp_path / "a-p1.json"
        shard.write_text(json.dumps(_shard({_SLEEP_TALK_KEY: [_entry("b", 1, "p1")]})))
        assert len(load_addresses([tmp_path])) == 1
        assert len(load_addresses([tmp_path, shard])) == 1
        assert len(load_addresses([shard, shard])) == 1

    def test_locator_separates_identical_seeds_in_different_shards(self, tmp_path):
        # A depth/arm grid reuses one seed_start, so battle_id collides across
        # shards that are genuinely different search configurations.
        for name in ("arm-a-d1.json", "arm-c-d2.json"):
            (tmp_path / name).write_text(
                json.dumps(_shard({_SLEEP_TALK_KEY: [_entry("battle-600000", 5, "p1")]}))
            )
        addresses = load_addresses([tmp_path])
        assert len(addresses) == 2
        assert len({a.locator for a in addresses}) == 2

    def test_a_missing_path_is_not_an_empty_corpus(self, tmp_path, capsys):
        assert main([str(tmp_path / "typo.json")]) == 2
        assert "path does not exist" in capsys.readouterr().out

    def test_json_out_creates_missing_parent(self, tmp_path):
        (tmp_path / "a-p1.json").write_text(
            json.dumps(_shard({_SLEEP_TALK_KEY: [_entry("b", 1, "p1")]}))
        )
        out = tmp_path / "deep" / "nested" / "corpus.json"
        assert main([str(tmp_path), "--json-out", str(out)]) == 0
        assert len(json.loads(out.read_text())) == 1


class TestStatsBlockDiscovery:
    """Completeness and occurrence totals must be found wherever addresses are."""

    _SAMPLES = {"materialization_blocker: baton-pass:substitute": [
        {"battle_id": "b", "round": 1, "seat": "p1", "reason": "no_worlds_constructed"}
    ]}
    _STATS = {
        "fallback_samples": _SAMPLES,
        "fallback_sample_addresses_dropped": 1001,
        "world_failure_reasons": {"materialization_blocker: baton-pass:substitute": 1472},
        "fallback_reasons": {"no_worlds_constructed": 7},
    }

    @pytest.mark.parametrize(
        ("layout", "document"),
        [
            # foulplay_bridge: nested under engine_mcts
            ("bridge", {"engine_mcts": {"policy_stats": _STATS}}),
            # scripts/foulplay_paired_eval.py: per_seat, no top-level wrapper
            ("paired_eval", {"per_seat": {"p1": {"policy_stats": _STATS}}}),
            # scripts/mcts_acceptance_h2h.py: policy_stats at document root
            ("acceptance_h2h", {"policy_stats": _STATS}),
        ],
    )
    def test_every_shard_layout_yields_completeness_and_counts(
        self, tmp_path, layout, document
    ):
        (tmp_path / f"{layout}.json").write_text(json.dumps(document))
        scan = scan_corpus([tmp_path])
        assert len(scan.addresses) == 1, layout
        assert scan.addresses_dropped == 1001, layout
        assert scan.complete is False, layout
        assert scan.world_counts[
            "materialization_blocker: baton-pass:substitute"
        ] == 1472, layout
        assert scan.decision_counts["fallback:no_worlds_constructed"] == 7, layout

    def test_a_mirrored_stats_dict_is_not_counted_twice(self, tmp_path):
        stats = dict(self._STATS)
        (tmp_path / "a.json").write_text(
            json.dumps({"per_seat": {"p1": {"policy_stats": stats, "extra": stats}}})
        )
        assert scan_corpus([tmp_path]).addresses_dropped == 1001

    def test_the_same_field_at_two_depths_is_counted_once(self):
        # A real shard carries `fallback_reasons` at BOTH `engine_mcts` and
        # `engine_mcts.policy_stats`. Counting both doubles every decision total --
        # measured 1835 -> 3670 on the four-era corpus before this guard.
        stats = {
            "fallback_samples": {"fallback:crate_search_failed": [
                {"battle_id": "b", "round": 1, "seat": "p1", "reason": "crate_search_failed"}
            ]},
            "fallback_reasons": {"crate_search_failed": 1835},
        }
        document = {
            "engine_mcts": {
                "fallback_reasons": {"crate_search_failed": 1835},
                "policy_stats": stats,
            }
        }
        scan = CorpusScan()
        _scan_document(document, scan, source="a.json")
        assert scan.decision_counts["fallback:crate_search_failed"] == 1835

    def test_two_independent_seat_scopes_both_count(self):
        # paired_eval runs each seat as a SEPARATE policy instance over the same seed
        # band. Both are real cumulative scopes and must sum.
        def seat(n):
            return {"policy_stats": {
                "fallback_samples": {"fallback:crate_search_failed": [
                    {"battle_id": f"b{n}", "round": n, "seat": f"p{n}",
                     "reason": "crate_search_failed"}]},
                "fallback_reasons": {"crate_search_failed": n},
            }}
        scan = CorpusScan()
        _scan_document({"per_seat": {"p1": seat(10), "p2": seat(25)}}, scan, source="a")
        assert scan.decision_counts["fallback:crate_search_failed"] == 35

    def test_equal_dropped_counts_in_two_scopes_are_not_collapsed(self):
        # The scalar fingerprint of `fallback_sample_addresses_dropped` is a bare
        # integer, so per-FIELD content dedup halved the total whenever two
        # independent scopes happened to share one equal nonzero value. No dict
        # collision was required for the loss.
        def seat(n):
            return {"policy_stats": {
                "fallback_samples": {"k": [
                    {"battle_id": f"b{n}", "round": n, "seat": "p1", "reason": "r"}]},
                "fallback_sample_addresses_dropped": 1001,
                "world_failure_reasons": {f"class-{n}": n},
            }}
        scan = CorpusScan()
        _scan_document({"per_seat": {"p1": seat(1), "p2": seat(2)}}, scan, source="a")
        assert scan.addresses_dropped == 2002

    def test_per_game_delta_blocks_are_not_summed_with_the_cumulative(self):
        # engine_search.py:2645-2654 appends a per-game DELTA block per game, and
        # :2683-2684 writes them beside the cumulative totals. The deltas sum exactly
        # to the cumulative, so accepting them doubles every count. Deltas carry no
        # `fallback_samples`, which is the marker that separates them.
        document = {
            "games": [
                {"seed": 600000, "world_failure_reasons": {"K": 10},
                 "fallback_reasons": {"R": 4}},
                {"seed": 600001, "world_failure_reasons": {"K": 6},
                 "fallback_reasons": {"R": 3}},
            ],
            "engine_mcts": {
                "fallback_samples": {"K": [
                    {"battle_id": "b", "round": 1, "seat": "p1", "reason": "R"}]},
                "world_failure_reasons": {"K": 16},
                "fallback_reasons": {"R": 7},
                "fallback_sample_addresses_dropped": 0,
            },
        }
        scan = CorpusScan()
        _scan_document(document, scan, source="a.json")
        assert scan.world_counts["K"] == 16
        assert scan.decision_counts["fallback:R"] == 7

    def test_identical_deltas_do_not_produce_a_fractional_multiple(self):
        # The nastier shape: equal deltas plus content dedup gave an unpredictable
        # 1.5x rather than a clean 2x.
        delta = {"world_failure_reasons": {"K": 8}, "fallback_reasons": {"R": 4}}
        document = {
            "games": [dict(delta), dict(delta)],
            "engine_mcts": {
                "fallback_samples": {"K": [
                    {"battle_id": "b", "round": 1, "seat": "p1", "reason": "R"}]},
                "world_failure_reasons": {"K": 16},
                "fallback_reasons": {"R": 8},
            },
        }
        scan = CorpusScan()
        _scan_document(document, scan, source="a.json")
        assert scan.world_counts["K"] == 16
        assert scan.decision_counts["fallback:R"] == 8


class TestLiteralPattern:
    def test_the_boundary_guard_is_not_the_naive_pattern(self):
        # Inert today -- every registered position anchors on "foe ability " -- but
        # the deferred species registration is unanchored, and the naive pattern
        # would then destroy the predicate of any interpolated native error.
        import re as _re

        from pokezero.fallback_addresses import _LITERAL

        text = "can't materialize 'Zapdos'"
        assert _re.search(_LITERAL, text).group() == "'Zapdos'"
        assert _re.compile(r"'[^']*'").search(text).group() == "'t materialize '"


class TestRawKeysMode:
    def test_raw_keys_still_resolves_occurrence_counts(self, tmp_path, capsys):
        # Display keys are raw while the occurrence tables are keyed canonical; a
        # missing lookup mislabelled every class "not in these shards".
        raw = ("self_request_state_unsupported: self active request flags ['trapped'] "
               "constrain legality beyond this construction (sampled world does not "
               "trap: foe ability 'swarm')")
        (tmp_path / "a-p1.json").write_text(
            json.dumps(
                {"engine_mcts": {"policy_stats": {
                    "fallback_samples": {raw: [_entry("b", 1, "p1")]},
                    "world_failure_reasons": {raw: 5216},
                }}}
            )
        )
        assert main([str(tmp_path), "--raw-keys"]) == 0
        out = capsys.readouterr().out
        assert "5216" in out
        assert "not in these shards" not in out


class TestMarkerIsPresenceNotTruthiness:
    def test_empty_fallback_samples_still_owns_its_counts(self):
        # world_failure_reasons is incremented per failed world regardless of
        # whether the decision falls back, so a healthy run has `fallback_samples:
        # {}` alongside real counts. Selecting on truthiness drops all of them.
        document = {
            "engine_mcts": {
                "policy_stats": {
                    "fallback_samples": {},
                    "world_failure_reasons": {"K": 16},
                    "fallback_reasons": {"R": 3},
                    "fallback_sample_addresses_dropped": 7,
                }
            }
        }
        scan = CorpusScan()
        _scan_document(document, scan, source="a.json")
        assert scan.addresses == []
        assert scan.world_counts["K"] == 16
        assert scan.decision_counts["fallback:R"] == 3
        assert scan.addresses_dropped == 7
