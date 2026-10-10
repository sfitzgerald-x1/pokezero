"""Structural copy preserves mutable payload isolation, including nested aliases."""
import copy
from dataclasses import replace
import unittest

from pokezero.belief import (
    BeliefEvidence, PublicBattleBeliefEngine, RevealedPokemonBelief,
    _copy_revealed_belief,
)


class BeliefClonePayloadTest(unittest.TestCase):
    def engine(self):
        nested = {"moves": ["surf", "rest"], "extra": {"range": [1, 2]}}
        record = RevealedPokemonBelief(
            showdown_slot="p2", species="Slowbro", active=True,
            candidate_variants=(nested,), source_metadata={"shared": nested},
            evidence=(BeliefEvidence("move", "observed surf", "|move|"),),
            revealed_moves=("surf",), move_uses=(("surf", 1),),
        )
        engine = PublicBattleBeliefEngine()
        engine._sides["p2"] = [record, replace(record, species="Slowking")]
        engine._pending_mudshot = {"target_key": record.key, "saw_damage": False}
        return engine

    def test_clone_equals_old_deepcopy_values(self):
        engine = self.engine()
        want = copy.deepcopy(engine._sides)
        twin = engine.clone()
        self.assertEqual(twin._sides, want)
        self.assertIsNot(twin._sides, engine._sides)
        self.assertIsNot(twin._sides["p2"], engine._sides["p2"])
        self.assertIsNot(twin._sides["p2"][0], engine._sides["p2"][0])

    def test_nested_payloads_isolated_between_parent_and_siblings(self):
        engine = self.engine()
        a, b = engine.clone(), engine.clone()
        a._sides["p2"][0].candidate_variants[0]["extra"]["range"].append(3)
        a._sides["p2"][0].source_metadata["shared"]["moves"].append("psychic")
        a._pending_mudshot["saw_damage"] = True
        self.assertEqual(engine._sides, b._sides)
        self.assertEqual(engine._pending_mudshot, b._pending_mudshot)
        self.assertEqual(engine._sides["p2"][0].candidate_variants[0]["moves"], ["surf", "rest"])

    def test_internal_aliases_preserved_but_not_shared_with_parent(self):
        engine = self.engine()
        a = engine.clone()._sides["p2"]
        self.assertIs(a[0].candidate_variants[0], a[0].source_metadata["shared"])
        self.assertIs(a[0].candidate_variants[0], a[1].candidate_variants[0])
        self.assertIsNot(a[0].candidate_variants[0], engine._sides["p2"][0].candidate_variants[0])

    def test_resolved_view_payload_mutation_does_not_change_engine(self):
        engine = self.engine()
        original = copy.deepcopy(engine._sides)
        view = engine.resolved_player_view("p1")
        view.opponent_pokemon[0].source_metadata["shared"]["moves"].append("psychic")
        self.assertEqual(engine._sides, original)

    def test_new_untyped_or_out_of_contract_mutable_values_copied(self):
        record = self.engine()._sides["p2"][0]
        # Frozen dataclasses do not enforce annotated tuple/string types. Do not
        # assume that annotations or future fields make their values immutable.
        object.__setattr__(record, "revealed_moves", ["surf"])
        object.__setattr__(record, "new_field", {"mutable": [1]})
        a = _copy_revealed_belief(record, {})
        a.revealed_moves.append("rest")
        a.new_field["mutable"].append(2)
        self.assertEqual(record.revealed_moves, ["surf"])
        self.assertEqual(record.new_field, {"mutable": [1]})

    def test_record_cycles_and_repeated_references_use_deepcopy_memo(self):
        record = self.engine()._sides["p2"][0]
        record.source_metadata["record"] = record
        a = _copy_revealed_belief(record, {})
        self.assertIs(a.source_metadata["record"], a)

    def test_event_updates_one_sibling_only(self):
        engine = self.engine()
        a, b = engine.clone(), engine.clone()
        a.ingest_event({"event_type": "turn", "raw_line": "|turn|7"})
        self.assertEqual(engine._turn_number, 0)
        self.assertEqual(b._turn_number, 0)
        self.assertEqual(a._turn_number, 1)

    def test_payload_backreferences_to_side_containers_preserved(self):
        engine = self.engine()
        record = engine._sides["p2"][0]
        record.source_metadata["sides"] = engine._sides
        record.source_metadata["side"] = engine._sides["p2"]
        a = engine.clone()
        self.assertIs(a._sides["p2"][0].source_metadata["sides"], a._sides)
        self.assertIs(a._sides["p2"][0].source_metadata["side"], a._sides["p2"])


if __name__ == "__main__":
    unittest.main()
