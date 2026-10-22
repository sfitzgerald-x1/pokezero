"""Information-state tests; no model, engine wheel, or live cluster required."""

from copy import deepcopy
from dataclasses import replace
import unittest

from pokezero.observation import ObservationFeatureMasks
from pokezero.policy_opponent_view import (
    PolicyOpponentViewError, advance_policy_opponent_view,
    build_policy_opponent_view, public_policy_lines,
)
from pokezero.showdown import V4_REPLAY_OBSERVATION_SPEC
from test_engine_world import _dex
from test_showdown import FakeSetSource
from pokezero.category_vocab import build_category_vocabulary


VOCAB = build_category_vocabulary((
    "last_used_move:switch", "ability:trace", "species:swampert", "species:snorlax",
    "species:starmie", "move:bodyslam", "move:shadowball", "move:surf",
    "move:earthquake", "request_kind:move", "request_kind:force_switch",
    "field", "stats", "status:none", "pokemon:self", "pokemon:opponent",
    "action:move", "action:switch", "action", "lastmove:switch", "move_priority:0",
    "type:normal", "type:water", "type:psychic", "type:ground", "move_category:physical",
    "belief:possible_ability:immunity", "belief:possible_ability:naturalcure",
    "belief:possible_ability:intimidate", "belief:possible_item:leftovers", "belief:possible_item:lumberry",
) + tuple(f"{prefix}:{slot}" for prefix in ("move_slot", "move:slot") for slot in range(1, 5))
  + tuple(f"{prefix}:{slot}" for prefix in ("switch_slot", "species:slot") for slot in range(1, 6)))


LINES = (
    "|player|p1|Subject", "|player|p2|Opponent", "|start",
    "|switch|p1a: Swampert|Swampert, L84|201/300",
    "|switch|p2a: Snorlax|Snorlax, L80|400/400", "|turn|1",
)


def request(slot="p2"):
    return {
        "side": {"id": slot, "pokemon": [
            {"ident": f"{slot}: Snorlax", "details": "Snorlax, L80", "active": True,
             "condition": "400/400", "moves": ["bodyslam", "shadowball"],
             "ability": "Immunity", "item": "Leftovers",
             "stats": {"atk": 200, "def": 150, "spa": 100, "spd": 200, "spe": 100}},
            {"ident": f"{slot}: Starmie", "details": "Starmie, L79", "active": False,
             "condition": "250/250", "moves": ["surf"], "ability": "Natural Cure",
             "item": "Leftovers", "stats": {"atk": 100, "def": 100, "spa": 200, "spd": 100, "spe": 200}},
        ]},
        "active": [{"moves": [
            {"id": "bodyslam", "move": "Body Slam", "pp": 20, "maxpp": 24, "disabled": False},
            {"id": "shadowball", "move": "Shadow Ball", "pp": 10, "maxpp": 24, "disabled": False},
        ]}],
    }


def view(lines=LINES, own_request=None, **kwargs):
    arguments = dict(
        public_lines=lines, hp_visibility={"p1": "exact", "p2": "exact"},
        sampled_self_request=request() if own_request is None else own_request,
        opponent_slot="p2", battle_id="test", battle_seed=10, format_id="gen3randombattle",
        set_source=FakeSetSource(), spec=V4_REPLAY_OBSERVATION_SPEC,
        feature_masks=ObservationFeatureMasks(transition_token_budget=0),
    )
    arguments.update(kwargs)
    return build_policy_opponent_view(**arguments)


def tensor_fields(observation):
    return tuple(getattr(observation, field) for field in (
        "categorical_ids", "numeric_features", "token_type_ids", "attention_mask", "legal_action_mask",
    ))


class PolicyOpponentViewTest(unittest.TestCase):
    def test_own_sampled_party_is_known_subject_is_public_only(self):
        row = view().row_inputs(dex=_dex())
        md = row["observation_metadata"]
        self.assertEqual(md["showdown_slot"], "p2")
        self.assertEqual(len(md["self_team"]), 2)
        self.assertEqual(md["self_team"][0]["stats"]["atk"], 200)
        self.assertEqual(md["self_team"][1]["moves"], ["surf"])
        self.assertEqual(len(md["opponent_team"]), 1)
        subject = md["opponent_team"][0]
        self.assertEqual(subject["condition"], "67/100")
        self.assertIsNone(subject["stats"])
        self.assertEqual(subject["moves"], [])
        self.assertIsNone(subject["ability"])
        self.assertIsNone(subject["item"])
        self.assertEqual(md["belief_view"]["opponent_pokemon"][0]["revealed_moves"], [])
        self.assertEqual(view().materialization.replay.requests, {})

    def test_exact_subject_maxhp_and_live_private_split_branch_are_invisible(self):
        # Different exact maxima and numerators with the same shared HP.
        altered = list(LINES)
        altered[3] = "|switch|p1a: Swampert|Swampert, L84|268/400"
        self.assertEqual(view().row_inputs(dex=_dex()), view(altered).row_inputs(dex=_dex()))
        split_a = ("|split|p1", LINES[3], "|switch|p1a: Swampert|Swampert, L84|67/100")
        split_b = ("|split|p1", "|switch|p1a: SECRET|SECRET|999/999", split_a[2])
        a = (*LINES[:3], *split_a, *LINES[4:])
        b = (*LINES[:3], *split_b, *LINES[4:])
        self.assertEqual(view(a).row_inputs(dex=_dex()), view(b).row_inputs(dex=_dex()))

    def test_branch_reveals_evolve_and_unrevealed_moves_remain_hidden(self):
        branch = (*LINES, "|move|p1a: Swampert|Earthquake|p2a: Snorlax",
                  "|-damage|p2a: Snorlax|300/400", "|turn|2",
                  "|switch|p1a: Starmie|Starmie, L79|250/250")
        updated = request()
        updated["side"]["pokemon"][0]["condition"] = "300/400"
        md = view(branch, updated).row_inputs(dex=_dex())["observation_metadata"]
        self.assertEqual(md["turn_number"], 2)
        self.assertEqual({m["species"] for m in md["opponent_team"]}, {"Swampert", "Starmie"})
        swampert = next(m for m in md["opponent_team"] if m["species"] == "Swampert")
        self.assertEqual(swampert["moves"], ["earthquake"])
        self.assertEqual(md["self_team"][0]["condition"], "300/400")

    def test_request_is_cloned_and_no_live_other_request_is_accepted(self):
        own = request()
        result = view(own_request=own)
        before = result.row_inputs(dex=_dex())
        own["side"]["pokemon"][0]["stats"]["atk"] = 9999
        self.assertEqual(before, result.row_inputs(dex=_dex()))
        with self.assertRaises(PolicyOpponentViewError):
            view(own_request=request("p1"))
        with self.assertRaises(PolicyOpponentViewError):
            view((*LINES, '|request|{"side":{"id":"p1"}}'))

    def test_prefix_provenance_hp_and_party_fail_closed(self):
        with self.assertRaises(PolicyOpponentViewError):
            view(LINES[3:])
        with self.assertRaises(PolicyOpponentViewError):
            public_policy_lines(LINES, hp_visibility={})
        with self.assertRaises(PolicyOpponentViewError):
            public_policy_lines(("|split|p1", LINES[3]), hp_visibility={})
        with self.assertRaises(PolicyOpponentViewError):
            public_policy_lines(("|-damage|p1a: A|42/48",), hp_visibility={"p1": "percentage"})
        for malformed in (
            "|-sethp|p1a: A|10/20|p2a: B",
            "|-sethp|p1a: A|10/20|p3a: B|10/20",
            "|-sethp|p1a: A|10/20|[from] move: Pain Split|10/20",
        ):
            with self.assertRaises(PolicyOpponentViewError):
                public_policy_lines((malformed,), hp_visibility={"p1": "exact", "p2": "exact"})
        broken = deepcopy(request())
        broken["side"]["pokemon"][1]["ident"] = "p1: SECRET"
        with self.assertRaises(PolicyOpponentViewError):
            view(own_request=broken)

    def test_near_full_hp_does_not_disclose_full_health_and_sethp_handles_both(self):
        actual = public_policy_lines(
            ("|-heal|p1a: A|999/1000", "|-sethp|p1a: A|5/10|p2a: B|10/20",
             "|-sethp|p2a: Wigglytuff|128/407|[from] move: Pain Split|[silent]",
             "|-sethp|p1a: Dusclops|128/209|[from] move: Pain Split",
             "|-sethp|p1a: A|5/10|p2a: B|10/20|[from] move: Pain Split"),
            hp_visibility={"p1": "exact", "p2": "exact"},
        )
        self.assertEqual(actual, ("|-heal|p1a: A|99/100", "|-sethp|p1a: A|50/100|p2a: B|50/100",
            "|-sethp|p2a: Wigglytuff|32/100|[from] move: Pain Split|[silent]",
            "|-sethp|p1a: Dusclops|62/100|[from] move: Pain Split",
            "|-sethp|p1a: A|50/100|p2a: B|50/100|[from] move: Pain Split"))

    def test_public_projection_never_canonicalizes_upkeep_evidence(self):
        markers = ("|upkeep ", "|upkeep\r", "|upkeep|payload")
        self.assertEqual(public_policy_lines(markers, hp_visibility={}), markers)

    def test_canonical_encoded_tensors_are_private_truth_invariant(self):
        before = view().observation(category_vocab=VOCAB, dex=_dex())
        before.validate(V4_REPLAY_OBSERVATION_SPEC)
        changed = list(LINES)
        changed[3] = "|switch|p1a: Swampert|Swampert, L84|268/400"
        after = view(changed).observation(category_vocab=VOCAB, dex=_dex())
        self.assertEqual(tensor_fields(before), tensor_fields(after))
        private_changes = (
            "|split|p1", "|switch|p1a: HIDDEN|HIDDEN, L100|888/999",
            "|switch|p1a: Swampert|Swampert, L84|67/100",
        )
        split = view((*LINES[:3], *private_changes, *LINES[4:])).observation(category_vocab=VOCAB, dex=_dex())
        self.assertEqual(tensor_fields(before), tensor_fields(split))
        self.assertEqual(VOCAB.observed_oov_tokens, frozenset())

    def test_mirrored_seats_encode_identical_relative_tensors(self):
        # Exchange the real seats, not the relative observation labels.
        mirrored_lines = tuple(line.replace("p1", "TEMP").replace("p2", "p1").replace("TEMP", "p2") for line in LINES)
        normal = view().observation(category_vocab=VOCAB, dex=_dex())
        mirrored = view(mirrored_lines, request("p1"), opponent_slot="p1").observation(category_vocab=VOCAB, dex=_dex())
        self.assertEqual(tensor_fields(normal), tensor_fields(mirrored))

    def test_child_exact_hp_is_projected_once_and_parent_siblings_are_immutable(self):
        parent = view()
        before = parent.row_inputs(dex=_dex())
        suffix = ("|move|p1a: Swampert|Earthquake|p2a: Snorlax", "|-damage|p2a: Snorlax|300/400", "|turn|2")
        own = request()
        own["side"]["pokemon"][0]["condition"] = "300/400"
        child = advance_policy_opponent_view(parent, public_branch_lines=suffix,
            branch_hp_visibility={"p1": "exact", "p2": "exact"},
            sampled_self_request=own, set_source=FakeSetSource())
        self.assertIn("|-damage|p2a: Snorlax|75/100", child.public_lines)
        self.assertIn("|switch|p1a: Swampert|Swampert, L84|67/100", child.public_lines)
        self.assertEqual(child.row_inputs(dex=_dex())["observation_metadata"]["self_team"][0]["condition"], "300/400")
        self.assertEqual(before, parent.row_inputs(dex=_dex()))
        sibling = advance_policy_opponent_view(parent, public_branch_lines=("|turn|2",),
            branch_hp_visibility={}, sampled_self_request=request(), set_source=FakeSetSource())
        self.assertNotIn("|-damage|p2a: Snorlax|75/100", sibling.public_lines)

    def test_forced_replacement_and_sparse_move_slots_preserve_action_mask(self):
        sparse = request()
        sparse["active"][0]["moves"][0]["disabled"] = True
        mask = view(own_request=sparse).observation(category_vocab=VOCAB, dex=_dex()).legal_action_mask
        self.assertEqual(tuple(i for i, enabled in enumerate(mask) if enabled), (1, 4))
        forced = request()
        forced.pop("active")
        forced["forceSwitch"] = [True]
        forced["side"]["pokemon"][0]["condition"] = "0 fnt"
        result = view((*LINES, "|faint|p2a: Snorlax"), forced)
        mask = result.observation(category_vocab=VOCAB, dex=_dex()).legal_action_mask
        self.assertEqual(tuple(i for i, enabled in enumerate(mask) if enabled), (4,))
        self.assertEqual(result.row_inputs(dex=_dex())["observation_metadata"]["request_kind"], "force_switch")

    def test_public_recharge_lock_is_relative_and_clears_after_forced_turn(self):
        parent = view((*LINES, "|-mustrecharge|p1a: Swampert", "|turn|2"))
        md = parent.row_inputs(dex=_dex())["observation_metadata"]
        self.assertTrue(md["opponent_must_recharge"])
        self.assertFalse(md["self_must_recharge"])
        child = advance_policy_opponent_view(parent,
            public_branch_lines=("|cant|p1a: Swampert|recharge", "|turn|3"),
            branch_hp_visibility={}, sampled_self_request=request(), set_source=FakeSetSource())
        self.assertFalse(child.row_inputs(dex=_dex())["observation_metadata"]["opponent_must_recharge"])

    def test_item_narrowing_matches_mask_and_missing_observers_refuse(self):
        masks = ObservationFeatureMasks(transition_token_budget=0, item_belief_narrowing=True)
        self.assertTrue(view(feature_masks=masks).materialization.belief_engine.item_belief_narrowing)
        for flag in ("investment_belief_narrowing",):
            with self.assertRaisesRegex(PolicyOpponentViewError, "observer"):
                view(feature_masks=replace(masks, **{flag: True}))

    def test_v4_retired_annotations_do_not_change_encoded_champion_surface(self):
        # The champion retains tier2 provenance flags but V4 dropped their
        # input columns. Test nonzero annotations, not merely empty history.
        lines = (*LINES, "|move|p2a: Snorlax|Body Slam|p1a: Swampert",
                 "|-damage|p1a: Swampert|150/300", "|turn|2")
        masks = ObservationFeatureMasks(transition_token_budget=0, tier2_residuals=True,
            tier2_investment=True, investment_belief_narrowing=False)
        original = view(lines, feature_masks=masks)
        self.assertTrue(original.state.transition_tokens)
        poisoned = replace(original, state=replace(original.state, transition_tokens=tuple(
            replace(token, residual=.9, residual_valid=True, cb_bit=True, investment=1.)
            for token in original.state.transition_tokens)))
        inactive = view(lines, feature_masks=replace(masks, tier2_residuals=False, tier2_investment=False))
        baseline = tensor_fields(original.observation(category_vocab=VOCAB, dex=_dex()))
        self.assertEqual(baseline, tensor_fields(poisoned.observation(category_vocab=VOCAB, dex=_dex())))
        self.assertEqual(baseline, tensor_fields(inactive.observation(category_vocab=VOCAB, dex=_dex())))
        self.assertEqual(original.feature_masks, masks)

    def test_legacy_observer_columns_remain_strictly_refused(self):
        from pokezero.showdown import V3_REPLAY_OBSERVATION_SPEC
        for masks in (
            ObservationFeatureMasks(transition_token_budget=0, tier2_residuals=False, tier2_investment=True),
            ObservationFeatureMasks(transition_token_budget=0, tier2_residuals=True, tier2_investment=False),
        ):
            with self.subTest(masks=masks), self.assertRaisesRegex(PolicyOpponentViewError, "observer"):
                view(spec=V3_REPLAY_OBSERVATION_SPEC, feature_masks=masks)

    def test_terminal_and_non_json_requests_fail_without_fallback(self):
        for terminal in ("|win|Opponent", "|tie"):
            with self.assertRaisesRegex(PolicyOpponentViewError, "terminal"):
                view((*LINES, terminal))
        for broken in ([], None, {"x": float("nan")}):
            with self.assertRaises(PolicyOpponentViewError):
                build_policy_opponent_view(public_lines=LINES,
                    hp_visibility={"p1": "exact", "p2": "exact"}, sampled_self_request=broken,
                    opponent_slot="p2", battle_id="test", battle_seed=10, format_id="gen3randombattle",
                    set_source=FakeSetSource(), spec=V4_REPLAY_OBSERVATION_SPEC,
                    feature_masks=ObservationFeatureMasks(transition_token_budget=0))


class ReplaySummaryReuseTest(unittest.TestCase):
    @staticmethod
    def source():
        from pokezero.randbat import (Gen3RandbatSource, Gen3RandbatVariant,
            Gen3RandbatSpeciesUniverse, RandbatSourceMetadata)
        universes = {}
        for species, moves, ability in (("Snorlax", ("bodyslam", "shadowball"), "Immunity"),
                ("Starmie", ("surf",), "Natural Cure"),
                ("Swampert", ("surf", "earthquake"), "Torrent")):
            variant = Gen3RandbatVariant(species, "fixture", species, "fixture", 80,
                moves, ability, "Leftovers")
            universes[species.casefold()] = Gen3RandbatSpeciesUniverse(species, 80, (variant,))
        return Gen3RandbatSource(RandbatSourceMetadata("gen3randombattle", 3,
            None, None, None, "fixture"), universes)

    @staticmethod
    def query(**overrides):
        arguments = dict(format_id="gen3randombattle", species="Snorlax",
            revealed_moves=(), revealed_ability=None, revealed_item=None,
            ruled_out_abilities=(), ruled_out_items=())
        arguments.update(overrides)
        return arguments

    def test_repeated_queries_are_reused_with_exact_keys_and_none(self):
        from unittest.mock import patch
        from pokezero.policy_opponent_view import _ReplaySummarySource
        source = self.source()
        cached = _ReplaySummarySource(source)
        changes = ({}, {"format_id": None}, {"format_id": ""}, {"species": "missing"},
            {"species": "snorlax"}, {"revealed_moves": ("bodyslam", "shadowball")},
            {"revealed_moves": ("shadowball", "bodyslam")}, {"revealed_ability": "Immunity"},
            {"revealed_ability": ""}, {"revealed_item": "Leftovers"}, {"revealed_item": ""},
            {"ruled_out_abilities": ("Immunity",)}, {"ruled_out_items": ("Leftovers",)})
        with patch.object(type(source), "summarize", wraps=source.summarize) as summarize:
            for change in changes:
                query = self.query(**change)
                expected = source.summarize(**query)
                before = summarize.call_count
                self.assertEqual(cached.summarize(**query), expected)
                self.assertEqual(cached.summarize(**query), expected)
                self.assertEqual(summarize.call_count, before + 1)
        self.assertEqual(len(cached.cache), len(changes))

    def test_cache_values_and_all_returned_nested_data_are_isolated(self):
        from pokezero.policy_opponent_view import _ReplaySummarySource
        source = self.source()
        cached = _ReplaySummarySource(source)
        first = cached.summarize(**self.query())
        expected = source.summarize(**self.query())
        first.candidate_variants[0]["moves"].append("secret")
        first.source_metadata["source_hash"] = "changed"
        second = cached.summarize(**self.query())
        self.assertEqual(second, expected)
        second.candidate_variants[0]["moves"].clear()
        second.source_metadata.clear()
        self.assertEqual(cached.summarize(**self.query()), expected)
        self.assertEqual(_ReplaySummarySource(source).summarize(**self.query()), expected)

    def test_exceptions_are_not_cached_and_non_tuple_queries_keep_source_behavior(self):
        from unittest.mock import patch
        from pokezero.policy_opponent_view import _ReplaySummarySource
        source = self.source()
        cached = _ReplaySummarySource(source)
        with patch.object(type(source), "summarize", side_effect=TypeError("original failure")) as call:
            for _ in range(2):
                with self.assertRaisesRegex(TypeError, "original failure"):
                    cached.summarize(**self.query())
            self.assertEqual(call.call_count, 2)
        self.assertEqual(cached.cache, {})
        query = self.query(revealed_moves=["bodyslam"])
        with patch.object(type(source), "summarize", wraps=source.summarize) as call:
            self.assertEqual(cached.summarize(**query), source.summarize(**query))
            self.assertEqual(cached.summarize(**query), source.summarize(**query))
            self.assertEqual(call.call_count, 4)
        self.assertEqual(cached.cache, {})

    def test_full_views_equal_canonical_replay_with_item_narrowing_and_off_script_reveals(self):
        from unittest.mock import patch
        from pokezero.belief import PublicBattleBeliefEngine
        source = self.source()
        canonical = PublicBattleBeliefEngine.from_events
        vocab = build_category_vocabulary((*VOCAB.tokens, "belief:possible_ability:torrent",
            "belief:possible_move:earthquake", "belief:possible_move:surf",
            "belief:possible_move:thunderbolt", "move:thunderbolt"))
        histories = (LINES, (*LINES, "|-item|p1a: Swampert|Leftovers"),
            (*LINES, "|move|p1a: Swampert|Thunderbolt|p2a: Snorlax",
             "|-damage|p2a: Snorlax|90/100", "|upkeep", "|turn|2"))
        for lines in histories:
            for narrow in (False, True):
                with self.subTest(lines=lines, item_narrowing=narrow):
                    masks = ObservationFeatureMasks(transition_token_budget=0,
                        item_belief_narrowing=narrow)
                    cached = view(lines, set_source=source, feature_masks=masks)
                    with patch("pokezero.policy_opponent_view._rebuild_policy_belief", side_effect=canonical):
                        baseline = view(lines, set_source=source, feature_masks=masks)
                    self.assertEqual(cached.row_inputs(dex=_dex()), baseline.row_inputs(dex=_dex()))
                    self.assertEqual(tensor_fields(cached.observation(category_vocab=vocab, dex=_dex())),
                        tensor_fields(baseline.observation(category_vocab=vocab, dex=_dex())))
                    self.assertIs(cached.materialization.belief_engine.set_source, source)

    def test_replays_do_not_share_queries_and_custom_sources_or_subclasses_are_not_cached(self):
        from unittest.mock import patch
        from pokezero.policy_opponent_view import _ReplaySummarySource
        source = self.source()
        with patch.object(type(source), "summarize", wraps=source.summarize) as call:
            view(set_source=source)
            first = call.call_count
            view(set_source=source)
            self.assertEqual(call.call_count, 2 * first)
        subclass = type("SourceSubclass", (type(source),), {})
        child = subclass(source.metadata, source.universes)
        for other in (FakeSetSource(), child):
            with patch("pokezero.policy_opponent_view._ReplaySummarySource", wraps=_ReplaySummarySource) as wrapper:
                result = view(set_source=other)
                wrapper.assert_not_called()
                self.assertIs(result.materialization.belief_engine.set_source, other)

    def test_original_source_is_restored_even_when_ingestion_raises(self):
        from unittest.mock import patch
        from pokezero.belief import PublicBattleBeliefEngine
        from pokezero.policy_opponent_view import _rebuild_policy_belief, _ReplaySummarySource
        source = self.source()
        observed = []
        def fail(engine, event):
            observed.append(engine)
            self.assertIsInstance(engine.set_source, _ReplaySummarySource)
            raise ValueError("authored ingestion failure")
        with patch.object(PublicBattleBeliefEngine, "ingest_event", fail):
            with self.assertRaisesRegex(ValueError, "authored ingestion failure"):
                _rebuild_policy_belief([object()], format_id="gen3randombattle",
                    set_source=source, item_belief_narrowing=False)
        self.assertIs(observed[0].set_source, source)


if __name__ == "__main__":
    unittest.main()
