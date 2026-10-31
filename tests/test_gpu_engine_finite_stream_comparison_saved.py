"""Saved-packet decoding only; native collector and all real inputs replaced."""
import copy
import unittest
from unittest.mock import patch

import numpy as np

from gpu_engine import finite_stream_comparison_saved as saved


class SavedConstructorComparisonTests(unittest.TestCase):
    def test_actual_checkpoint_mask_names_build_exact_public_encoder_masks(self):
        from pokezero.recorded_mirror_roots import CHAMPION_CONTRACT
        masks = CHAMPION_CONTRACT['feature_masks']
        actual = saved.champion_feature_masks(masks)
        for name, value in masks.items():
            self.assertEqual(getattr(actual,
                'opponent_tendency_stats_block' if name == 'stats_block' else name), value)

    def packet(self):
        actual={name:[[0],[0]] for name in saved.BattleState._fields}
        actual['hp']=[[[341]*6,[287]*6],[[193]*6,[211]*6]]
        observations=[]
        for side in (0,1):
            value={name:np.zeros((2,*shape),np.float32 if name=='numeric_features' else
                bool if name.endswith('mask') else np.int32).tolist()
                for name,shape in saved.comparison.SHAPES.items()}
            value['admitted']=[side==0,side==1]
            observations.append(value)
        return dict(original_denominator=2,cohort=[dict(stream_id=101,team_indices=[0,1]),
            dict(stream_id=102,team_indices=[2,3])],initial_engine_state=actual,
            initial_frame_input_witness=dict(owner_full_words=np.zeros((2,2,1,64,16),np.int32).tolist(),
                owner_full_valid=np.zeros((2,2,1,64),bool).tolist(),owner_full_more=np.zeros((2,2,1),bool).tolist()),
            initial_owner_observations=observations)

    def test_missing_actual_constructor_packets_stay_unknown_without_native_call(self):
        packet=self.packet();packet.pop('initial_engine_state')
        with patch.object(saved.comparison,'collect_constructors') as collect:
            result=saved.compare_saved(packet,None,None,showdown_root='/not-read',node='not-run',native_seed_start=700)
            collect.assert_not_called()
        self.assertEqual(result['summary']['original_owner_views'],4)
        self.assertEqual(result['summary']['state_counts'],{'UNKNOWN':2})
        self.assertTrue(all(row['error']['type']=='MISSING_ACTUAL_CONSTRUCTOR_CAPTURE' for row in result['rows']))

    def test_direct_state_decode_and_per_owner_admission_preserve_fixed_seed_ordinal(self):
        packet=self.packet();original=copy.deepcopy(packet)
        calls=[]
        def collect(pool,teams,state,captured,catalog,**kwargs):
            index=len(calls);calls.append((teams,state,captured,kwargs))
            self.assertEqual(kwargs['native_seed_start'],700+index)
            self.assertEqual(kwargs['root_ids'],[101+index])
            self.assertEqual(teams,[packet['cohort'][index]['team_indices']])
            np.testing.assert_array_equal(state.hp,np.asarray(packet['initial_engine_state']['hp'])[index:index+1])
            owners=kwargs['engine_observations']
            self.assertIsNone(owners[1-index])
            self.assertEqual(owners[index]['numeric_features'].dtype,np.float32)
            self.assertEqual(owners[index]['categorical_ids'].dtype,np.int32)
            self.assertEqual(owners[index]['attention_mask'].dtype,np.bool_)
            self.assertEqual(owners[index]['numeric_features'].shape,(1,23,132))
            return dict(rows=[dict(root_id=101+index,state_status='UNKNOWN',owners=[])])
        with patch.object(saved.comparison,'collect_constructors',side_effect=collect):
            result=saved.compare_saved(packet,None,None,showdown_root='/not-read',node='not-run',native_seed_start=700)
        self.assertEqual(len(calls),2)
        self.assertEqual(result['summary']['original_roots'],2)
        self.assertEqual(packet,original)

    def test_duplicate_or_dropped_original_roster_rejected_before_any_native_work(self):
        packet=self.packet();packet['original_denominator']=3
        with patch.object(saved.comparison,'collect_constructors') as collect:
            with self.assertRaisesRegex(ValueError,'original finite cohort'):
                saved.compare_saved(packet,None,None,showdown_root='/not-read',node='not-run',native_seed_start=700)
            collect.assert_not_called()


if __name__=='__main__':
    unittest.main()
