"""Public-prefix parsing parity on authored transcripts, no scientific roots."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pokezero.belief import CandidateSetSummary
from pokezero.category_vocab import build_category_vocabulary
from pokezero.observation import ObservationFeatureMasks
from pokezero.policy_opponent import make_policy_opponent_callback
from pokezero.policy_opponent_view import (
    PolicyOpponentViewError, _PublicPolicyPrefix,
    build_policy_opponent_view_from_native_bundle, public_policy_lines,
)
from pokezero.showdown import _ReplayParser, V4_REPLAY_OBSERVATION_SPEC
from test_engine_world import _dex
from test_policy_opponent import config
from test_policy_opponent_request import arguments, bundle
from test_policy_opponent_view import LINES, VOCAB as BASE_VOCAB, tensor_fields
from test_showdown import FakeSetSource


PERCENTAGE = {"p1": "percentage", "p2": "percentage"}
FORWARD = "pokezero.neural_policy.evaluate_transformer_action_priors"
# OOV recording is mutable even though token identity is frozen. This authored
# mechanics vocabulary must not poison another module's closed-vocabulary test.
VOCAB = deepcopy(BASE_VOCAB)


def projected(lines):
    return public_policy_lines(lines, hp_visibility={"p1": "exact", "p2": "exact"})


def mirror(lines):
    return tuple(line.replace("p1", "TMP").replace("p2", "p1").replace("TMP", "p2")
        for line in lines)


def history():
    """Authored chronology includes mid-move, residual and replacement cuts."""
    return (*LINES,
        "|move|p1a: Swampert|Surf|p2a: Snorlax",
        "|-damage|p2a: Snorlax|300/400",
        "|move|p2a: Snorlax|Body Slam|p1a: Swampert",
        "|-damage|p1a: Swampert|150/300",
        "|-status|p1a: Swampert|tox",
        "|-heal|p2a: Snorlax|325/400|[from] item: Leftovers",
        "|upkeep", "|turn|2",
        "|move|p1a: Swampert|Protect|p1a: Swampert",
        "|-singleturn|p1a: Swampert|Protect",
        "|move|p2a: Snorlax|Body Slam|p1a: Swampert",
        "|-activate|p1a: Swampert|move: Protect",
        "|-damage|p1a: Swampert|130/300 tox|[from] psn",
        "|upkeep", "|turn|3",
        "|switch|p1a: Starmie|Starmie, L79|250/250",
        "|-ability|p1a: Starmie|Natural Cure",
        "|-weather|RainDance|[from] ability: Drizzle|[of] p2a: Snorlax",
        "|-mustrecharge|p2a: Snorlax", "|upkeep", "|turn|4",
        "|cant|p2a: Snorlax|recharge", "|-sideend|p1: Subject|Reflect",
        "|upkeep", "|turn|5")


class CountingSource(FakeSetSource):
    def __init__(self):
        self.calls = []

    def summarize(self, **kwargs):
        self.calls.append(deepcopy(kwargs))
        return CandidateSetSummary(species=kwargs["species"], candidate_count=len(self.calls),
            uncertainty=.25, possible_moves=kwargs["revealed_moves"])


class PublicPrefixParityTests(unittest.TestCase):
    def make(self, lines, supplied=None, prefix=None, **overrides):
        args = arguments(lines)
        args.update(hp_visibility=PERCENTAGE, **overrides)
        if prefix is not None:
            args["_public_prefix"] = prefix
        return build_policy_opponent_view_from_native_bundle(
            native_request_bundle=bundle() if supplied is None else supplied, **args)

    def assert_views(self, reference, actual, *, category_vocab=VOCAB):
        self.assertEqual(reference.state, actual.state)
        self.assertEqual(reference.public_lines, actual.public_lines)
        self.assertEqual(reference.native_action_indices, actual.native_action_indices)
        self.assertEqual(reference.materialization.replay, actual.materialization.replay)
        self.assertEqual(reference.materialization.self_request, actual.materialization.self_request)
        self.assertEqual(reference.materialization.self_move_states, actual.materialization.self_move_states)
        self.assertEqual(vars(reference.materialization.belief_engine),
            vars(actual.materialization.belief_engine))
        self.assertEqual(reference.row_inputs(dex=_dex()), actual.row_inputs(dex=_dex()))
        one = reference.observation(category_vocab=category_vocab, dex=_dex())
        two = actual.observation(category_vocab=category_vocab, dex=_dex())
        self.assertEqual(tensor_fields(one), tensor_fields(two))
        self.assertEqual(one.metadata, two.metadata)
        two.validate(actual.spec)

    def test_each_authored_prefix_cut_matches_full_replay_in_both_seats(self):
        source = FakeSetSource()
        for mirrored in (False, True):
            lines = projected(mirror(history()) if mirrored else history())
            supplied = bundle()
            if mirrored:
                supplied = json.loads(json.dumps(supplied).replace("p2:", "p1:").replace('"p2"', '"p1"'))
            for cut in range(3, len(lines) + 1):
                with self.subTest(mirrored=mirrored, cut=cut):
                    prefix = _PublicPolicyPrefix(lines[:cut], battle_id="test")
                    options = dict(opponent_slot="p1" if mirrored else "p2", set_source=source)
                    self.assert_views(self.make(lines, supplied, **options),
                        self.make(lines, supplied, prefix, **options))

    def test_pending_damage_window_is_preserved_not_snapshot_hydrated(self):
        lines = projected((*LINES, "|move|p1a: Swampert|Surf|p2a: Snorlax"))
        prefix = _PublicPolicyPrefix(lines, battle_id="test")
        complete = (*lines, "|-damage|p2a: Snorlax|75/100")
        replay = prefix.parse(complete, battle_id="test")
        self.assertEqual(replay.current_damage_dealt["p1"], .25)
        parser = _ReplayParser("test", complete_prefix=True, hp_visibility=PERCENTAGE)
        parser.feed(complete)
        self.assertEqual(replay, parser.snapshot())

    def test_root_is_never_snapshotted_and_all_internal_fields_are_cloned(self):
        lines = projected(history())
        prefix = _PublicPolicyPrefix(lines[:10], battle_id="test")
        snapshots = []
        original = _ReplayParser.snapshot
        def snapshot(parser):
            snapshots.append(parser)
            return original(parser)
        with patch.object(_ReplayParser, "snapshot", snapshot):
            prefix.parse(lines, battle_id="test")
            before = deepcopy(vars(prefix._parser))
            prefix.parse(lines[:10], battle_id="test")
            self.assertEqual(before, vars(prefix._parser))
            self.assertFalse(any(parser is prefix._parser for parser in snapshots))
            self.assertEqual(len(snapshots), 2)

    def test_siblings_parent_and_returned_mutable_payloads_are_isolated(self):
        lines = projected(LINES)
        prefix = _PublicPolicyPrefix(lines, battle_id="test")
        source = FakeSetSource()
        first = self.make((*lines, "|-damage|p1a: Swampert|50/100"), prefix=prefix, set_source=source)
        before = deepcopy(vars(prefix._parser))
        first.state.request["side"]["pokemon"][0]["stats"]["atk"] = 9999
        first.materialization.replay.boosts["p1"]["atk"] = 6
        first.materialization.replay.players["p1"] = "mutated"
        second = self.make(lines, prefix=prefix, set_source=source)
        self.assert_views(self.make(lines, set_source=source), second)
        self.assertEqual(before, vars(prefix._parser))
        self.assertEqual(second.materialization.replay.public_active["p1"].condition, "67/100")

    def test_parallel_invocations_match_independent_full_rebuilds(self):
        lines = projected(LINES)
        prefix = _PublicPolicyPrefix(lines, battle_id="test")
        source = FakeSetSource()
        cases = [(*lines, f"|-damage|p1a: Swampert|{percent}/100") for percent in range(20, 40)]
        with ThreadPoolExecutor(max_workers=4) as pool:
            actual = list(pool.map(lambda branch: self.make(branch, prefix=prefix, set_source=source), cases))
        for branch, result in zip(cases, actual):
            self.assert_views(self.make(branch, set_source=source), result)

    def test_stateful_sources_are_replayed_fully_with_identical_call_order(self):
        lines = projected(history())
        prefix = _PublicPolicyPrefix(lines[:12], battle_id="test")
        reference_source, cached_source = CountingSource(), CountingSource()
        for branch in (lines, lines[:12], lines[:20], lines):
            reference = self.make(branch, set_source=reference_source)
            actual = self.make(branch, prefix=prefix, set_source=cached_source)
            # Identity of the separate stateful source is intentionally different.
            self.assertEqual(reference.state, actual.state)
            self.assertEqual(reference.row_inputs(dex=_dex()), actual.row_inputs(dex=_dex()))
            self.assertEqual(reference_source.calls, cached_source.calls)

    def test_new_sampled_request_and_full_bench_pp_are_not_cached(self):
        lines = projected(LINES)
        prefix = _PublicPolicyPrefix(lines, battle_id="test")
        source = FakeSetSource()
        first = self.make(lines, prefix=prefix, set_source=source)
        supplied = bundle()
        supplied["request"]["side"]["pokemon"][0]["stats"]["atk"] = 211
        supplied["request"]["active"][0]["moves"][0]["pp"] = 17
        supplied["self_move_states"]["snorlax"][0]["pp"] = 17
        supplied["self_move_states"]["starmie"][0]["pp"] = 3
        second = self.make(lines, supplied, prefix, set_source=source)
        self.assert_views(self.make(lines, supplied, set_source=source), second)
        self.assertEqual(first.materialization.self_move_states["starmie"][0]["pp"], 7)
        self.assertEqual(second.materialization.self_move_states["starmie"][0]["pp"], 3)

    def test_wait_and_forced_replacement_keep_exact_action_surface(self):
        lines = projected(LINES)
        prefix = _PublicPolicyPrefix(lines, battle_id="test")
        source = FakeSetSource()
        for waiting in (True, False):
            supplied = bundle()
            supplied["request"].pop("active")
            if waiting:
                supplied["request"]["wait"] = True
                supplied["native_action_indices"] = [None]
                branch = lines
            else:
                supplied["request"]["forceSwitch"] = [True]
                supplied["request"]["side"]["pokemon"][0]["condition"] = "0 fnt"
                supplied["native_action_indices"] = [4]
                branch = (*lines, "|faint|p2a: Snorlax")
            self.assert_views(self.make(branch, supplied, set_source=source),
                self.make(branch, supplied, prefix, set_source=source))

    def test_private_split_alternatives_are_not_retained(self):
        a = (*LINES[:3], "|split|p1", LINES[3], "|switch|p1a: Swampert|Swampert, L84|67/100", *LINES[4:])
        b = (*LINES[:3], "|split|p1", "|switch|p1a: SECRET|SECRET|999/999", a[5], *LINES[4:])
        self.assertEqual(projected(a), projected(b))
        prefix = _PublicPolicyPrefix(projected(b), battle_id="test")
        result = self.make(projected(a), prefix=prefix)
        self.assertNotIn("SECRET", repr(vars(prefix._parser)))
        self.assertNotIn("SECRET", repr(result.row_inputs(dex=_dex())))

    def test_noncanonical_upkeep_spelling_and_toxic_provenance_match(self):
        source = FakeSetSource()
        for marker in ("|upkeep", "|upkeep ", "|upkeep\r", "|upkeep|payload"):
            lines = projected((*LINES, "|-status|p1a: Swampert|tox", marker))
            prefix = _PublicPolicyPrefix(lines, battle_id="test")
            branch = (*lines, "|turn|2", "|switch|p1a: Starmie|Starmie, L79|100/100")
            self.assert_views(self.make(branch, set_source=source), self.make(branch, prefix=prefix, set_source=source))

    def test_terminal_and_private_suffix_fail_identically_without_poisoning_root(self):
        lines = projected(LINES)
        prefix = _PublicPolicyPrefix(lines, battle_id="test")
        for suffix in ("|win|Opponent", "|tie", '|request|{"wait":true}', "private", "|split|p1"):
            errors = []
            for cached in (None, prefix):
                with self.assertRaises(Exception) as raised:
                    self.make((*lines, suffix), prefix=cached)
                errors.append((type(raised.exception), str(raised.exception)))
            self.assertEqual(*errors)
        source = FakeSetSource()
        self.assert_views(self.make(lines, set_source=source), self.make(lines, prefix=prefix, set_source=source))

    def test_bad_request_pp_or_action_map_refuses_without_lazy_preparation(self):
        lines = projected(LINES)
        for mutate in (lambda b: b["request"]["side"].update(id="p1"),
                lambda b: b["self_move_states"]["starmie"][0].update(pp=-1),
                lambda b: b.update(native_action_indices=[0, 0])):
            prefix = _PublicPolicyPrefix(lines, battle_id="test")
            supplied = bundle()
            mutate(supplied)
            with self.assertRaises(PolicyOpponentViewError):
                self.make(lines, supplied, prefix)
            self.assertIsNone(prefix._parser)

    def test_mismatched_prefix_identity_and_subclasses_fail_closed(self):
        lines = projected(LINES)
        class Impostor(_PublicPolicyPrefix):
            pass
        for prefix in (_PublicPolicyPrefix(lines, battle_id="other"),
                _PublicPolicyPrefix((*lines, "|turn|999"), battle_id="test"),
                Impostor(lines, battle_id="test")):
            with self.assertRaisesRegex(PolicyOpponentViewError, "prefix"):
                self.make(lines, prefix=prefix)
            self.assertIsNone(prefix._parser)

    def test_failed_prefix_parse_does_not_publish_partial_state(self):
        lines = projected(LINES)
        prefix = _PublicPolicyPrefix(lines, battle_id="test")
        original = _ReplayParser.feed
        error = RuntimeError("authored parser failure")
        def fail(parser, supplied):
            original(parser, supplied[:3])
            raise error
        with patch.object(_ReplayParser, "feed", fail), self.assertRaises(RuntimeError) as raised:
            self.make(lines, prefix=prefix)
        self.assertIs(raised.exception, error)
        self.assertIsNone(prefix._parser)

    def test_legacy_turn_merged_and_feature_masks_retain_full_products(self):
        from pokezero.showdown import V2_2_REPLAY_OBSERVATION_SPEC
        lines = projected(history())
        prefix = _PublicPolicyPrefix(lines[:10], battle_id="test")
        source = FakeSetSource()
        for spec, masks in ((V4_REPLAY_OBSERVATION_SPEC, ObservationFeatureMasks(transition_token_budget=0,
                item_belief_narrowing=True, tier2_residuals=True, tier2_investment=True)),
                (V2_2_REPLAY_OBSERVATION_SPEC, ObservationFeatureMasks(
                    tier2_residuals=False, tier2_investment=False))):
            # Synthetic vocabulary only; this checks exact reference parity,
            # not champion coverage or a closed randbat vocabulary universe.
            vocab = build_category_vocabulary((*VOCAB.tokens, "tt_phase:turn"))
            self.assert_views(self.make(lines, spec=spec, feature_masks=masks, set_source=source),
                self.make(lines, prefix=prefix, spec=spec, feature_masks=masks, set_source=source),
                category_vocab=vocab)

    def test_callback_still_forwards_each_choice_and_parses_root_only_once(self):
        cfg = config()
        source = FakeSetSource()
        callback = make_policy_opponent_callback(public_lines=LINES,
            hp_visibility={"p1": "exact", "p2": "exact"}, opponent_slot="p2",
            battle_id="test", battle_seed=10, format_id="gen3randombattle", set_source=source,
            model=SimpleNamespace(config=cfg), result=SimpleNamespace(model_config=cfg, belief_set_source_hash=None),
            category_vocab=VOCAB, dex=_dex(), raw_argmax=True)
        payload = json.dumps(dict(opponent_slot="p2", public_branch_lines=[], native_request_bundle=bundle()))
        feeds = []
        original = _ReplayParser.feed
        def record(parser, lines):
            feeds.append(tuple(lines))
            return original(parser, lines)
        with patch.object(_ReplayParser, "feed", record), patch(FORWARD, return_value=(.8,.1,0,0,.1,0,0,0,0)) as forward:
            self.assertEqual(callback(payload), (1., 0., 0.))
            self.assertEqual(callback(payload), (1., 0., 0.))
            self.assertEqual(forward.call_count, 2)
        self.assertEqual(feeds, [projected(LINES), (), ()])

    def test_oversized_templates_use_unchanged_full_rebuild_not_truncation(self):
        for tail in (("|message|padding",) * _PublicPolicyPrefix._MAX_LINES,
                ("|message|" + "x" * _PublicPolicyPrefix._MAX_CHARACTERS,),
                ("|message|" + "\ud800" * _PublicPolicyPrefix._MAX_CHARACTERS,)):
            lines = (*projected(LINES), *tail)
            prefix = _PublicPolicyPrefix(lines, battle_id="test")
            source = FakeSetSource()
            self.assert_views(self.make(lines, set_source=source), self.make(lines, prefix=prefix, set_source=source))
            self.assertIsNone(prefix._parser)

    def test_clone_failure_preserves_original_exception_and_unchanged_root(self):
        lines = projected(LINES)
        prefix = _PublicPolicyPrefix(lines, battle_id="test")
        prefix.parse(lines, battle_id="test")
        before = deepcopy(vars(prefix._parser))
        error = RuntimeError("authored clone failure")
        with patch("pokezero.policy_opponent_view.deepcopy", side_effect=error), self.assertRaises(RuntimeError) as raised:
            self.make(lines, prefix=prefix)
        self.assertIs(raised.exception, error)
        self.assertEqual(before, vars(prefix._parser))

    def test_custom_battle_id_is_retained_without_deepcopy_or_len_hooks(self):
        class Identifier(str):
            def __deepcopy__(self, memo):
                raise AssertionError("custom identity must not be deep-copied")
            def __len__(self):
                raise AssertionError("custom identity must not be measured")
        lines = projected(LINES)
        identifier = Identifier("test")
        prefix = _PublicPolicyPrefix(lines, battle_id=identifier)
        source = FakeSetSource()
        reference = self.make(lines, set_source=source, battle_id=identifier)
        actual = self.make(lines, prefix=prefix, set_source=source, battle_id=identifier)
        self.assertIs(reference.state.battle_id, identifier)
        self.assertIs(actual.state.battle_id, identifier)
        self.assertIsNone(prefix._parser)
        self.assert_views(reference, actual)

    def test_custom_prefix_lines_and_containers_add_no_measurement_hooks(self):
        class Line(str):
            def __len__(self):
                raise AssertionError("custom line must not be measured")
            def __deepcopy__(self, memo):
                raise AssertionError("custom line must not be deep-copied")
        class Lines(tuple):
            def __len__(self):
                raise AssertionError("custom container must not be measured")
            def __iter__(self):
                raise AssertionError("custom container must not be iterated")
        lines = projected(LINES)
        source = FakeSetSource()
        for template in ((Line(lines[0]), *lines[1:]), Lines(lines)):
            prefix = _PublicPolicyPrefix(template, battle_id="test")
            self.assert_views(self.make(lines, set_source=source),
                self.make(lines, prefix=prefix, set_source=source))
            self.assertIsNone(prefix._parser)

    def test_complete_start_and_observer_refusals_precede_lazy_parser(self):
        for lines, options in ((projected(LINES[3:]), {}), (projected(LINES), dict(
                feature_masks=ObservationFeatureMasks(investment_belief_narrowing=True)))):
            prefix = _PublicPolicyPrefix(lines, battle_id="test")
            with self.assertRaises(PolicyOpponentViewError):
                self.make(lines, prefix=prefix, **options)
            self.assertIsNone(prefix._parser)

    def test_all_parser_transients_match_at_every_mechanics_cut_before_snapshot(self):
        lines = (*projected(LINES),
            "|-ability|p2a: Snorlax|Truant",
            "|move|p1a: Swampert|Leech Seed|p2a: Snorlax",
            "|-start|p2a: Snorlax|move: Leech Seed",
            "|move|p2a: Snorlax|Rest|p2a: Snorlax",
            "|-heal|p2a: Snorlax|100/100", "|-status|p2a: Snorlax|slp|[from] move: Rest",
            "|-damage|p2a: Snorlax|88/100|[from] Leech Seed|[of] p1a: Swampert",
            "|-heal|p1a: Swampert|80/100|[from] Leech Seed|[of] p2a: Snorlax",
            "|upkeep", "|turn|2", "|cant|p2a: Snorlax|slp",
            "|move|p2a: Snorlax|Sleep Talk|p2a: Snorlax",
            "|move|p2a: Snorlax|Body Slam|p1a: Swampert|[from] move: Sleep Talk",
            "|-damage|p1a: Swampert|60/100", "|-start|p1a: Swampert|confusion",
            "|-start|p1a: Swampert|Encore", "|upkeep", "|turn|3",
            "|cant|p2a: Snorlax|ability: Truant",
            "|move|p1a: Swampert|Baton Pass|p1a: Swampert",
            "|switch|p1a: Starmie|Starmie, L79|100/100|[from] Baton Pass",
            "|-end|p1a: Starmie|confusion", "|-end|p1a: Starmie|Encore", "|upkeep", "|turn|4")
        full = _ReplayParser("test", complete_prefix=True, hp_visibility=PERCENTAGE)
        full.feed(lines)
        expected = deepcopy(vars(full))
        original = _ReplayParser.snapshot
        for cut in range(3, len(lines) + 1):
            with self.subTest(cut=cut):
                prefix = _PublicPolicyPrefix(lines[:cut], battle_id="test")
                before_snapshots = []
                def capture(parser):
                    before_snapshots.append(deepcopy(vars(parser)))
                    return original(parser)
                with patch.object(_ReplayParser, "snapshot", capture):
                    prefix.parse(lines, battle_id="test")
                self.assertEqual(before_snapshots, [expected])


if __name__ == "__main__":
    unittest.main()
