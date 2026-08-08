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

    def test_seat_is_normalised_in_both_spellings(self):
        # `{slot}:` unquoted and `side {slot!r}` quoted must agree, or the canonical
        # key space is partitioned by an f-string accident.
        quoted = ("encore_move_unknown: side '%s' is encored but the locked move "
                  "cannot be determined")
        unquoted = "self_moveset_mismatch: %s: request-known move 'toxic' is absent"
        assert canonical_key(quoted % "p1") == canonical_key(quoted % "p2")
        assert canonical_key(unquoted % "p1") == canonical_key(unquoted % "p2")

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
