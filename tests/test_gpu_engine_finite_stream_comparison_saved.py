"""Saved-packet decoding only; native collector and all real inputs replaced."""
import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
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

    def test_reference_config_explicitly_uses_champion_v4_layout_and_original_vocabulary(self):
        from pokezero.recorded_mirror_roots import CHAMPION_CONTRACT
        vocabulary=object()
        config=saved.champion_env_config(showdown_root='/not-opened',node='not-run',vocabulary=vocabulary,
            masks=CHAMPION_CONTRACT['feature_masks'])
        spec=config.observation_spec
        self.assertEqual((spec.schema_version,spec.token_count,spec.numeric_feature_count,spec.categorical_feature_count),
            ('pokezero.observation.v4',23,132,41))
        self.assertEqual(spec.transition_token_count,0)
        self.assertIs(config.category_vocab,vocabulary)
        self.assertEqual(config.feature_masks.transition_token_budget,0)

    def original_pool(self, path, *, corrupt=None):
        from gpu_engine import source_provenance
        arrays={name:np.ones((2,6),np.int32) for name in source_provenance._POOL_STATE_FIELDS}
        arrays.update(level=np.tile([[73],[81]],(1,6)), moves=np.tile([1,2,0,0],(2,6,1)),
            pp=np.ones((2,6,4),np.int32),types=np.ones((2,6,2),np.int32),
            stats=np.ones((2,6,6),np.int32),nature=np.tile([1,2,1,2,1,2],(2,1)),
            evs=np.tile([4,8,12,16,20,24],(2,6,1)),ivs=np.arange(72).reshape(2,6,6)%32)
        if corrupt is not None:arrays['ivs']=corrupt
        vocabulary=dict(species=['_none','mew'],moves=['_none','tackle','hiddenpowerfire'],
            items=['_none','leftovers'],abilities=['_none','synchronize'],natures=['_none','jolly','serious'])
        arrays['metadata_utf8']=np.frombuffer(json.dumps(dict(vocabulary=vocabulary)).encode(),np.uint8)
        np.savez_compressed(path,**arrays)
        return source_provenance.load_constructor_pool(path),hashlib.sha256(path.read_bytes()).hexdigest()

    def test_real_production_loader_and_original_IV_snapshot_supply_exact_packed_parties(self):
        comparison=saved.comparison
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'original.npz';pool,sha=self.original_pool(path)
            self.assertNotIn('ivs',pool.arrays)  # Exercise the actual production omission.
            identity=(pool.receipt_json,pool.receipt_sha256,set(pool.arrays))
            sidecar=comparison.load_original_ivs(path,pool,expected_pool_sha256=sha)
            genders=np.array([[2,1,0,2,1,0],[1,0,2,1,0,2]],np.int32)
            happiness=np.arange(12,dtype=np.int32).reshape(2,6)+117
            parties=comparison.pool_sets(pool,[1,0],genders,happiness,original_ivs=sidecar)
            stats=('hp','atk','def','spa','spd','spe')
            expected=[]
            for side,team in enumerate((1,0)):
                expected.append([dict(species='mew',level=(73,81)[team],moves=['tackle','hiddenpowerfire'],
                    item='leftovers',ability='synchronize',nature=('jolly','serious')[slot%2],
                    ivs=dict(zip(stats,((team*36+slot*6+column)%32 for column in range(6)))),
                    evs=dict(zip(stats,(4,8,12,16,20,24))),
                    gender={0:'N',1:'M',2:'F'}[int(genders[side,slot])],happiness=117+side*6+slot)
                    for slot in range(6)])
            self.assertEqual(parties,expected)
            self.assertEqual(list(parties[0][0]['ivs'].values()),[4,5,6,7,8,9])
            with patch.object(comparison.subprocess,'check_output',return_value='["party1","party2"]') as pack:
                comparison.pack_sets(parties,showdown_root='/not-opened',node='not-run')
            self.assertEqual(json.loads(pack.call_args.kwargs['input']),expected)
            self.assertIn('Teams.pack',pack.call_args.args[0][2])
            self.assertNotIn('Teams.generate',pack.call_args.args[0][2])
            self.assertEqual((pool.receipt_json,pool.receipt_sha256,set(pool.arrays)),identity)
            with self.assertRaises(ValueError):sidecar.values(pool).setflags(write=True)
            with self.assertRaisesRegex(ValueError,'SNAPSHOT_REQUIRED'):
                comparison.pool_sets(pool,[1,0],genders,happiness)
            with self.assertRaisesRegex(ValueError,'BINDING_CHANGED'):
                replace(sidecar,archive_bytes=sidecar.archive_bytes+b'corrupt').values(pool)
            other,other_sha=self.original_pool(Path(tmp)/'foreign.npz',corrupt=np.full((2,6,6),7))
            with self.assertRaisesRegex(ValueError,'ARCHIVE_CHANGED'):
                comparison.load_original_ivs(Path(tmp)/'foreign.npz',other,expected_pool_sha256=sha)
            with self.assertRaisesRegex(ValueError,'BINDING_CHANGED'):
                comparison.load_original_ivs(Path(tmp)/'foreign.npz',pool,expected_pool_sha256=other_sha)
            with self.assertRaisesRegex(ValueError,'original pool binding differs'):
                saved.compare_saved(dict(pool_receipt=dict(pool_sha256=other_sha)),pool,None,
                    showdown_root='/not-opened',node='not-run',native_seed_start=700,original_ivs=sidecar)

    def test_corrupt_original_IV_dtype_shape_or_range_never_defaults_to31(self):
        with tempfile.TemporaryDirectory() as tmp:
            for index,ivs in enumerate((np.full((2,6,6),32),np.full((2,6,6),-1),
                    np.full((2,6,6),30.5),np.full((2,6,5),30))):
                path=Path(tmp)/f'bad-{index}.npz';pool,sha=self.original_pool(path,corrupt=ivs)
                with self.subTest(index=index),self.assertRaisesRegex(ValueError,'LITERAL_ORIGINAL_POOL_IVS'):
                    saved.comparison.load_original_ivs(path,pool,expected_pool_sha256=sha)

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
