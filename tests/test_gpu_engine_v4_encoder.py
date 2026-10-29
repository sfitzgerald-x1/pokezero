"""Coherent public/request composition against the shipped Python encoder."""
import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec('jax'),'requires isolated JAX')
class V4EncoderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import jax
        import jax.numpy as j
        import test_gpu_engine_public_refresh_context as public
        import test_gpu_engine_private_actions as private
        from gpu_engine import v4_encoder as encoder,public_active
        from pokezero.category_vocab import CategoryVocabulary
        from pokezero.dex import ShowdownDex,resolve_move_effect,normalize_id
        from dataclasses import replace
        public.PublicRefreshContextTests.setUpClass()
        private.PrivateActionTests.setUpClass()
        cls.jax,cls.j,cls.e=jax,j,encoder
        cls.public=public.PublicRefreshContextTests
        cls.source=cls.public.host
        dex=ShowdownDex(dict(private.PrivateActionTests.dex.moves,**cls.public.dex.moves),
            dict({k:replace(v,name=k.title()) for k,v in private.PrivateActionTests.dex.species.items()},**cls.public.dex.species),cls.public.dex.type_chart)
        labels=set(cls.public.vocab.tokens)|set(private.PrivateActionTests.vocab.tokens)|{'field','stats','pokemon:opponent'}
        labels|={'request_kind:'+name for name in ('none','wait','team_preview','force_switch','move','unknown')}
        labels|={'weather:'+name for name in ('raindance','sunnyday','sandstorm','hail')}
        labels|={'volatile:'+name for name in public_active.NAMES if name}
        labels|={'type:'+name for name in ('Normal','Fighting','Flying','Poison','Ground','Rock','Bug','Ghost','Steel',
            'Fire','Water','Grass','Electric','Psychic','Ice','Dragon','Dark','???')}
        labels|={'status:'+name for name in ('none','slp','par','brn','frz','psn','tox','fnt')}
        for ability in ('naturalcure','earlybird','shedskin','trace','pressure','truant','levitate','immunity','arenatrap','shadowtag','magnetpull','shielddust'):
            labels|={'ability:'+ability,'belief:possible_ability:'+ability}
        for name,info in dex.moves.items():
            labels|={'move:'+name,'move_category:'+info.gen3_category,'move_priority:'+str(info.priority)}
            for types in ((),('Ghost',)):
                effect,_,_=resolve_move_effect(info,types)
                if effect:labels.add('move_effect:'+effect)
        labels|={'species:'+normalize_id(name) for name in cls.source.species_names if name}
        labels|={'species:'+normalize_id(name) for name in dex.species}
        for variants in cls.source.variants:
            for variant in variants:
                for kind,names in (('ability',(variant.ability,)),('item',(variant.item,)),('move',variant.moves)):
                    labels|={'belief:possible_'+kind+':'+normalize_id(name) for name in names if name}
        cls.inputdex=dex
        cls.vocab=CategoryVocabulary(tuple(sorted(labels)))
        cls.masks=dict(exact_state=True,feature_pack_last_move=True,investment_belief_narrowing=False,
            item_belief_narrowing=False,stats_block=True,tier2_investment=True,tier2_residuals=True,transition_token_budget=0)
        cls.catalog=encoder.compile_encoder(cls.source,dex,cls.vocab,checkpoint_sha256='a'*64,feature_masks=cls.masks)

    def test_compile_and_own_request_namespace_contract(self):
        self.assertEqual(self.catalog.masks.exact_state,True)
        self.assertEqual(self.catalog.binding.shape,(8,))
        original=self.inputdex.species['snorlax']
        from dataclasses import replace
        try:
            self.inputdex.species['snorlax']=replace(original,base_stats=dict(original.base_stats,hp=999))
            self.assertNotEqual(self.catalog.actions.dex.species['snorlax'].base_stats['hp'],999)
            with self.assertRaises(TypeError):self.catalog.dex.species['snorlax'].base_stats['hp']=999
        finally:self.inputdex.species['snorlax']=original
        with self.assertRaisesRegex(ValueError,'without defaults'):
            self.e.Masks.from_checkpoint(dict(exact_state=True))

    def request(self,owner=0):
        names=('Gardevoir','Snorlax') if owner==0 else ('Gardevoir','Absol')
        return {'rqid':3,'side':{'id':f'p{owner+1}','pokemon':[
            {'ident':f'p{owner+1}: {name}','details':f'{name}, L80','condition':'120/240',
             'active':i==1,'ability':'Immunity' if owner==0 else 'Pressure','item':'Leftovers',
             'stats':{'atk':151,'def':160,'spa':170,'spd':180,'spe':190}}
            for i,name in enumerate(names)]},'active':[{'moves':[{'id':'bodyslam','pp':10,'maxpp':24},
                {'id':'surf','pp':12,'maxpp':24}]}]}

    def projections(self,lines):
        from gpu_engine import public_wire
        return public_wire.project_lines(public_wire.WireState(),lines,**self.e.wire_kwargs(self.catalog))[1]

    def chunks(self,projections,batch=1):
        import numpy as np
        from gpu_engine import public_inputs
        p=projections[0]
        bindings={name:np.asarray(getattr(p,name),np.uint32) for name in (
            'species_binding','ability_binding','move_binding','item_binding','context_binding','field_binding',
            'active_binding','damage_binding','type_binding','sleep_binding','toxic_binding','tendency_binding')}
        events=[event for projection in projections for event in projection.events]
        return list(public_inputs.pack_chunks(np.arange(1,batch+1,dtype=np.int32),[events]*batch,
            source_binding=self.catalog.source.device.binding,**bindings))

    def ingest(self,lines):
        state=self.e.initialize(self.catalog,stream_ids=self.j.array([1],self.j.int32))
        step=self.jax.jit(lambda s,c:self.e.fold_chunk(s,c,self.catalog))
        for chunk in self.chunks(self.projections(lines)):
            state,error=step(state,chunk);self.assertEqual(error.tolist(),[0])
        return state

    def test_full_installed_writers_use_one_owner_packet_and_source_parity(self):
        lines=[*self.public.prefix,'|turn|1','|-boost|p1a: Snorlax|atk|2',
            '|move|p1a: Snorlax|Body Slam|p2a: Absol','|-damage|p2a: Absol|75/100']
        for sub in (False,True):
            self.compare_source(lines+(['|-start|p1a: Snorlax|Substitute','|-start|p2a: Absol|Substitute'] if sub else []))

    def compare_source(self,lines,*,exact_ability_case=False):
        import json
        from dataclasses import replace
        from pokezero import showdown as source
        state=self.ingest(lines)
        encode=self.jax.jit(lambda s,o,p:self.e.encode(s,o,self.catalog,observer_sides=p))
        for owner,flip in ((0,False),(1,False),(0,True),(1,True)):
            raw=self.request(owner)
            raw['side']['pokemon'][1]['condition']='120/247'
            if exact_ability_case:
                raw['side']['pokemon'][0]['ability']='Arena Trap'
                raw['side']['pokemon'][1]['condition']='120/247 slp'
            if flip:
                for row in raw['side']['pokemon']:row['active']=not row['active']
            own=self.e.parse_own_requests(self.catalog,[raw],owner_sides=[owner])
            out=encode(state,own,self.j.array([owner],self.j.int32));actual=out.observation
            self.assertEqual(actual.admitted.tolist(),[True]);self.assertEqual(actual.numeric_features.shape,(1,23,132))
            replay=source.parse_showdown_replay([*lines,'|request|'+json.dumps(raw)],battle_id='encoder-fixture')
            normalized=source.normalize_for_player(replay,player_id=f'p{owner+1}',configured_showdown_slot=f'p{owner+1}',
                format_id='gen3randombattle',set_source=self.public.source)
            masks=replace(source.DEFAULT_OBSERVATION_FEATURE_MASKS,exact_state=True,opponent_tendency_stats_block=True,
                feature_pack_last_move=True,transition_token_budget=0)
            expected=source.observation_from_player_state(normalized,category_vocab=self.vocab,
                spec=source.V4_REPLAY_OBSERVATION_SPEC,dex=self.catalog.dex,feature_masks=masks)
            import numpy as np
            numbers=np.asarray(actual.numeric_features[0]);want=np.asarray(expected.numeric_features)
            cats=np.asarray(actual.categorical_ids[0]);expected_cats=np.asarray(expected.categorical_ids)
            # Report exact physical mismatches, never hide missing fields by a
            # union of partial masks. No numeric column remains uninstalled.
            self.assertTrue(np.allclose(numbers,want,atol=1e-6,rtol=0),str(np.argwhere(abs(numbers-want)>1e-6).tolist()))
            self.assertTrue(np.array_equal(cats,expected_cats),str(np.argwhere(cats!=expected_cats).tolist()))
            self.assertEqual(actual.attention_mask[0].tolist(),list(expected.attention_mask))
            self.assertEqual(actual.legal_action_mask[0].tolist(),list(expected.legal_action_mask))
            self.assertEqual(out.pending_numeric_columns.tolist(),[])
            self.assertFalse(bool(out.full_encoder_admitted[0]))

    def test_exact_wake_and_private_bench_trapper_registration(self):
        lines=[*self.public.prefix,'|-status|p1a: Snorlax|slp|[from] move: Rest',
            '|-status|p2a: Absol|slp|[from] move: Rest']
        self.compare_source(lines,exact_ability_case=True)

    def test_contract_rollback_at_batch8_preserves_global_namespaces(self):
        import numpy as np
        from gpu_engine import public_inputs
        state=self.e.initialize(self.catalog,stream_ids=self.j.arange(1,9,dtype=self.j.int32))
        state=state._replace(binding=state.binding.at[0,0].set(0))
        before=state
        step=self.jax.jit(lambda s,c:self.e.fold_chunk(s,c,self.catalog))
        for chunk in self.chunks(self.projections(self.public.prefix),batch=8):
            state,error=step(state,chunk)
            self.assertEqual(error.tolist(),[int(public_inputs.Error.SOURCE)]+[0]*7)
        for old,new in zip(self.jax.tree.leaves(before),self.jax.tree.leaves(state)):
            if old.shape and old.shape[0]==8 and old.ndim>1:
                np.testing.assert_array_equal(old[0],new[0])
        self.assertTrue(bool((state.mons.encounter[1:]!=before.mons.encounter[1:]).any()))
        visible=state.context.refresh.controller.visible
        self.assertEqual(visible.source_binding.shape,(8,))
        np.testing.assert_array_equal(visible.source_binding,self.catalog.source.device.binding)
        self.assertEqual(visible.schema_binding.shape,(8,))
        np.testing.assert_array_equal(visible.schema_binding,before.context.refresh.controller.visible.schema_binding)

    def test_encode_owner_namespace_pending_frame_and_mask_refusals(self):
        state=self.ingest(self.public.prefix)
        own=self.e.parse_own_requests(self.catalog,[self.request()],owner_sides=[0])
        encode=self.jax.jit(lambda s,o,p:self.e.encode(s,o,self.catalog,observer_sides=p))
        for candidate,packet,owner in ((state,own,1),
                (state._replace(binding=state.binding.at[0,0].set(0)),own,0),
                (state,own._replace(binding=own.binding.at[0,0].set(0)),0),
                (state._replace(cursor=state.cursor._replace(needs_more=self.j.array([True]))),own,0)):
            out=encode(candidate,packet,self.j.array([owner],self.j.int32)).observation
            self.assertFalse(bool(out.admitted[0]));self.assertFalse(bool(out.numeric_features.any()))
            self.assertFalse(bool(out.categorical_ids.any()));self.assertFalse(bool(out.attention_mask.any()))
            self.assertFalse(bool(out.legal_action_mask.any()));self.assertTrue(bool((out.switch_targets==-1).all()))
        for candidate,packet in ((state._replace(binding=state.binding.astype(self.j.float32)),own),
                                 (state,own._replace(binding=own.binding.astype(self.j.float32)))):
            with self.assertRaisesRegex(ValueError,'uint32'):
                encode(candidate,packet,self.j.array([0],self.j.int32))
        for values in (dict(self.masks,transition_token_budget=1),dict(self.masks,investment_belief_narrowing=True),
                       dict(self.masks,exact_state=1)):
            with self.assertRaises(ValueError):self.e.Masks.from_checkpoint(values)

    def test_public_adapter_rejects_private_requests_and_unported_raw_family(self):
        import json
        with self.assertRaisesRegex(ValueError,'ordered evidence consumer: request'):
            self.projections(['|request|'+json.dumps(self.request())])
        with self.assertRaisesRegex(ValueError,'ordered evidence consumer'):
            self.projections([*self.public.prefix,'|unimplemented|wire'])


if __name__=='__main__':unittest.main()
