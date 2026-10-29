"""Device Toxic state compared with the actual incremental Python parser."""
import importlib.util
import json
import unittest


@unittest.skipUnless(importlib.util.find_spec('jax'),'requires isolated JAX')
class PublicToxicTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import test_gpu_engine_public_team as fixtures
        fixtures.PublicTeamTests.setUpClass.__func__(cls)
        from gpu_engine import public_toxic as t
        cls.t=t;cls.tables=t.compile_toxic(cls.catalog,cls.vocab)
        cls.ids={'p1a: Snorlax':1,'p1a: Gardevoir':2,'p1a: Absol':3,
                 'p2a: Absol':1,'p2a: Gardevoir':2,'p2a: Snorlax':3}
        cls.prefix=('|switch|p1a: Snorlax|Snorlax, L80|100/100',
                    '|switch|p2a: Absol|Absol, L80|100/100')

    def initial(self,batch=1,complete=True,visibility=(0,0)):
        return self.t.initialize(batch,self.tables,species_binding=self.catalog.device.binding,
            complete_prefix=complete,hp_visibility=visibility)

    def compare(self,state,parser):
        self.assertEqual(int(state.turn[0]),parser.turn_number)
        self.assertEqual(bool(state.post_upkeep[0]),parser._post_upkeep_window)
        maps={'stage':'toxic_stage','known':'toxic_stage_known','zero':'toxic_stage_zero_after_upkeep',
              'deadline':'toxic_stage_zero_after_upkeep_expires_after_turn','pending':'toxic_faint_replacement_pending',
              'invalid':'toxic_faint_replacement_invalid'}
        for side in (0,1):
            seat=f'p{side+1}'
            for name,source in maps.items():
                value=getattr(parser,source)[seat]
                expected=0 if value is None else value
                actual=getattr(state,name)[0,side].item()
                self.assertEqual(actual,expected,f'{name} {seat}')
            for name,source in [('action_ident','toxic_stage_reset_ident'),('zero_ident','toxic_stage_zero_after_upkeep_ident'),
                                ('expected','toxic_faint_replacement_expected_ident')]:
                self.assertEqual(int(getattr(state,name)[0,side]),self.ids.get(getattr(parser,source)[seat],0),name)
            self.assertEqual(bool(state.baton[0,side]),seat in parser.pending_baton_pass)
            active=parser.public_active.get(seat)
            self.assertEqual(int(state.active[0,side]),self.ids.get(active.ident,0) if active else 0)
            hp,maximum=self.reference._hp_numerator_denominator(active.condition if active else None)
            self.assertEqual(int(state.hp[0,side]),-1 if hp is None else hp,'hp')
            self.assertEqual(int(state.maximum[0,side]),maximum or 0,'maximum')
            self.assertEqual(bool(state.condition_tox[0,side]),self.reference._condition_has_status(active.condition if active else None,'tox'))
            patch=self.t.project_active(state,self.tables,expected_vocabulary=self.tables.vocabulary_binding)
            self.assertEqual(patch.admitted.tolist(),[True])
            if active:
                nums=[[0.]*200 for _ in range(6)];cats=[['']*60 for _ in range(6)]
                self.reference._encode_pokemon_tokens(cats,nums,0,parser.public_revealed[seat],role='opponent',limit=6,
                    dex=self.dex,schema_v3=True,schema_v4=True,active_toxic_stage=parser.toxic_stage[seat])
                for slot,candidate in enumerate(parser.public_revealed[seat]):
                    actual=float(patch.numeric[0,side,self.t.COLUMN]) if candidate.active else 0.
                    self.assertAlmostEqual(actual,nums[slot][self.reference.NUMERIC_TOXIC_STAGE],places=6)

    def prefixes(self,lines,*,complete=True,visibility=(0,0)):
        state=self.initial(complete=complete,visibility=visibility)
        modes={1:'exact',2:'percentage'}
        parser=self.reference._ReplayParser('toxic-device-fixture',complete_prefix=complete,
            hp_visibility={f'p{s+1}':modes[v] for s,v in enumerate(visibility) if v})
        for line in lines:
            fact=self.t.decode_line(line,identity_ids=self.ids)
            state,ok=self.t.apply(state,self.tables,fact,producer_binding=self.tables.binding)
            self.assertTrue(bool(ok.all()),line)
            parser._feed_line(line)
            self.compare(state,parser)
        return state,parser

    def test_status_ramp_sentinel_rest_and_active_vs_bench_cures(self):
        self.prefixes([*self.prefix,'|turn|1','|-status|p1a: Snorlax|tox',
            *[f'|turn|{n}' for n in range(2,21)],'|-status|p1a: Snorlax|slp|[from] move: Rest',
            '|-status|p1a: Snorlax|t.o.x','|-curestatus|p1: Snorlax|tox|[silent]',
            '|-curestatus|p1a: Snorlax|tox','|-status|p2a: Absol|tox','|-cureteam|p2a: Absol'])

    def test_residual_floor_exact_percentage_unknown_and_exact100(self):
        for visibility in ((0,0),(1,1),(2,2)):
            self.prefixes(['|switch|p1a: Snorlax|Snorlax, L80|100/100 tox',
                '|-damage|p1a: Snorlax|94/100 tox|[from] psn','|turn|1',
                '|-heal|p1a: Snorlax|100/100 tox','|-damage|p1a: Snorlax|95/100 tox|[from] psn',
                '|-damage|p1a: Snorlax|0 fnt|[from] psn','|faint|p1a: Snorlax'],visibility=visibility)
        for maximum,after in ((307,288),(14,13),(320,280)):
            self.prefixes([f'|switch|p1a: Snorlax|Snorlax, L80|{maximum}/{maximum} tox',
                f'|-damage|p1a: Snorlax|{after}/{maximum} tox|[from] psn','|turn|1',
                f'|-damage|p1a: Snorlax|{after-1}/{maximum} tox|[from] psn'])

    def test_request_presence_resolves_only_representation_provenance(self):
        requests=['|request|'+json.dumps({'side':{'id':f'p{n}'}}) for n in (1,2)]
        for middle in ([],requests[:1],requests):
            self.prefixes([*self.prefix,*middle,'|-status|p1a: Snorlax|tox','|-status|p2a: Absol|tox',
                '|switch|p1a: Gardevoir|Gardevoir, L80|100/100 tox',
                '|switch|p2a: Gardevoir|Gardevoir, L80|100/100 tox',
                '|-damage|p1a: Gardevoir|94/100 tox|[from] psn','|-damage|p2a: Gardevoir|95/100 tox|[from] psn'])

    def test_bounded_faint_replacement_zero_and_first_tick(self):
        for middle in ([],['|t:|123'],['|'],['|upkeep'],['|-activate|p1a: Snorlax|test']):
            self.prefixes([*self.prefix,'|turn|1','|faint|p1a: Snorlax',*middle,'|upkeep',
                '|switch|p1a: Gardevoir|Gardevoir, L80|100/100 tox','|turn|2',
                '|-damage|p1a: Gardevoir|94/100 tox|[from] psn','|upkeep'],visibility=(2,2))
        self.prefixes([*self.prefix,'|turn|1','|faint|p1a: Snorlax','|upkeep',
            '|switch|p1a: Gardevoir|Gardevoir, L80|100/100 tox','|turn|2','|upkeep','|turn|3'])

    def test_malformed_boundaries_and_wrong_occupants_retire_before_gates(self):
        for marker in ('|upkeep\r','|upkeep|payload','|turn','|turn|03','|turn|1','|faint','|faint|p1a: Wrong',
                       '|switch|p1a: Snorlax','|tie','|deinit'):
            self.prefixes([*self.prefix,'|turn|1','|switch|p1a: Gardevoir|Gardevoir, L80|100/100 tox',
                marker,'|turn|2','|-damage|p1a: Gardevoir|94/100 tox|[from] psn'])
        self.prefixes(['|turn|8',*self.prefix,'|-status|p1a: Unknown|tox','|-damage|p1a: Unknown|90/100 tox|[from] psn',
            '|-status|p1a: Unknown','|-damage|p1a: Unknown','|-curestatus|p1a: Unknown'])

    def test_baton_pass_transform_blank_and_partial_prefix(self):
        self.prefixes(['|switch|p1a: Snorlax|Snorlax, L80|307/307 tox','|turn|1',
            '|move|p1a: Snorlax|Baton Pass|p1a: Snorlax','|switch|p1a: Gardevoir|Gardevoir, L80|307/307 tox',
            '|-transform|p1a: Gardevoir|p2a: Absol','|','|upkeep','|turn|2',
            '|-damage|p1a: Gardevoir|288/307 tox|[from] psn'])
        state,_=self.prefixes(['','  ','>ignored','|t:|123','|turn|7'],complete=False)
        self.assertFalse(bool(state.known.any()))

    def test_jit_namespace_payload_overflow_and_shape_refusals_are_atomic(self):
        state,_=self.prefixes([*self.prefix,'|-status|p1a: Snorlax|tox'])
        state=self.jax.tree.map(lambda x:self.j.concatenate([x,x]),state)
        fact=self.t.decode_line('|turn|1',identity_ids=self.ids)
        fact=self.jax.tree.map(lambda x:self.j.broadcast_to(x,(2,)),fact)
        fn=self.jax.jit(lambda s,f:self.t.apply(s,self.tables,f,producer_binding=self.tables.binding))
        broken=state._replace(binding=state.binding.at[0,0].add(1))
        got,ok=fn(broken,fact);self.assertEqual(ok.tolist(),[False,True])
        for old,new in zip(self.jax.tree.leaves(broken),self.jax.tree.leaves(got)):self.assertTrue(bool(self.j.array_equal(old[0],new[0])))
        _,ok=fn(state,fact._replace(side=self.j.array([2,-1],self.j.int32)));self.assertEqual(ok.tolist(),[False,True])
        with self.assertRaises(ValueError):self.t.apply(state._replace(known=state.known.astype(self.j.int32)),self.tables,fact,producer_binding=self.tables.binding)
        for raw in ('|turn|2147483648','|-damage|p1a: Snorlax|2147483648/2147483648 tox'):
            with self.assertRaises(ValueError):self.t.decode_line(raw,identity_ids=self.ids)

    def test_initial_turn_handler_rechecks_after_latch_invalidation(self):
        state,parser=self.prefixes(['|switch|p1a: Snorlax|Snorlax, L80|100/100',
            '|faint|p1a: Snorlax','|upkeep','|switch|p1a: Gardevoir|Gardevoir, L80|100/100 tox','|turn|3'])
        self.assertEqual(int(state.turn[0]),3)
        self.assertTrue(bool(state.invalid[0,0]));self.assertFalse(bool(state.zero[0,0]))

    def test_positionless_alias_does_not_change_public_active_condition(self):
        ids=dict(self.ids,**{'p1: Snorlax':1})
        parser=self.reference._ReplayParser(complete_prefix=True)
        state=self.initial()
        for line in ['|switch|p1a: Snorlax|Snorlax, L80|100/100 tox','|-curestatus|p1: Snorlax|tox|[silent]',
                     '|-damage|p1: Snorlax|94/100 tox|[from] psn','|faint|p1: Snorlax',
                     '|turn|1','|-damage|p1a: Snorlax|94/100 tox|[from] psn']:
            state,ok=self.t.apply(state,self.tables,self.t.decode_line(line,identity_ids=ids),producer_binding=self.tables.binding)
            self.assertTrue(bool(ok.all()));parser._feed_line(line);self.compare(state,parser)
        for raw in ['|switch|p1a: Snorlax|Snorlax, L80|tox','|-damage|p1a: Snorlax|tox','|-sethp|p1a: Snorlax|']:
            with self.assertRaises(ValueError):self.t.decode_line(raw,identity_ids=ids)
        with self.assertRaises(ValueError):self.t.decode_line('|',identity_ids=dict(ids,**{'p1a: Alias':1}))


if __name__=='__main__':unittest.main()
