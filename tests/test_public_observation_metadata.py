"""Public model-state metadata survives transport without private expansion."""
from copy import deepcopy
import unittest

from pokezero.actions import ACTION_COUNT
from pokezero.observation import PokeZeroObservationV0
from pokezero.public_decision_corpus import PublicObservation


class PublicObservationMetadataTests(unittest.TestCase):
    def observation(self, metadata):
        return PokeZeroObservationV0(categorical_ids=((1,2),),numeric_features=((.25,.5),),
            token_type_ids=(1,),attention_mask=(True,),legal_action_mask=(True,)+(False,)*(ACTION_COUNT-1),
            metadata=metadata,schema_version="pokezero.observation.v4")

    def test_existing_public_v3_v4_inputs_survive_two_roundtrips(self):
        values={
            "self_sleep_clause_blocks":True,"opponent_sleep_clause_blocks":False,
            "self_wish_turns":1,"opponent_wish_turns":2,
            "self_stall_counter":3,"opponent_stall_counter":4,
            "self_confusion_elapsed":1,"opponent_confusion_elapsed":2,
            "self_encore_elapsed":2,"opponent_encore_elapsed":3,
            "self_wrap_trap_elapsed":3,"opponent_wrap_trap_elapsed":4,
            "self_meanlook_trap":False,"opponent_meanlook_trap":True,
            "self_must_recharge":False,"opponent_must_recharge":True,
            "self_truant_loaf":True,"opponent_truant_loaf":False,
            "self_last_used_move":"surf","opponent_last_used_move":"switch",
            "self_traced_ability":"levitate","opponent_traced_ability":None,
            "self_last_damage_dealt":.2,"self_last_damage_taken":.4,
            "opponent_last_damage_dealt":.4,"opponent_last_damage_taken":.2,
            "self_hazard_damage_suffered":.5,"opponent_hazard_damage_suffered":.75,
            "self_items_removed":1,"opponent_items_removed":2,
            "self_arrived_by_baton_pass":False,"opponent_arrived_by_baton_pass":True,
            "self_choice_locked":True,"opponent_choice_locked":False,
            "self_item_swapped":False,"opponent_item_swapped":True,
            "self_hazard_credit":.1,"opponent_hazard_credit":.2,
            "self_hazard_expected":.3,"opponent_hazard_expected":.4,
            "self_items_removed_credit":.1,"opponent_items_removed_credit":.2,
            "opponent_matchup_switch_evidence":{"snorlax":[2,3]},
        }
        original=self.observation(deepcopy(values));public=PublicObservation.from_observation(original)
        self.assertEqual(public.acting_player_state,values)
        restored=PublicObservation.from_dict(public.to_dict()).to_observation(belief_view={})
        again=PublicObservation.from_observation(restored)
        self.assertEqual(again,public)
        self.assertEqual(restored.categorical_ids,original.categorical_ids)
        self.assertEqual(restored.numeric_features,original.numeric_features)
        original.metadata["opponent_matchup_switch_evidence"]["snorlax"][0]=99
        self.assertEqual(public.acting_player_state["opponent_matchup_switch_evidence"]["snorlax"],[2,3])

    def test_absent_old_fields_stay_absent_and_opponent_private_fields_stay_removed(self):
        public=PublicObservation.from_observation(self.observation(dict(
            opponent_active=dict(species="snorlax",moves=["secret"],ability="secret",item="secret",stats={"atk":999}),
            opponent_observation="secret",raw_choice="secret")))
        self.assertEqual(public.acting_player_state,{"opponent_active":{"species":"snorlax"}})
        restored=PublicObservation.from_dict(public.to_dict())
        self.assertNotIn("self_last_used_move",restored.acting_player_state)
        with self.assertRaises(ValueError):
            PublicObservation.from_observation(self.observation(dict(
                opponent_matchup_switch_evidence={"private_request":"secret"})))


if __name__ == "__main__":unittest.main()
