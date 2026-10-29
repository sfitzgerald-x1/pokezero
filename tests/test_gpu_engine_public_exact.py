"""Exact wake/trapper cells against the source, before ability-bucket truncation."""
import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec('jax'),'requires isolated JAX')
class PublicExactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from dataclasses import replace
        from types import SimpleNamespace
        from test_gpu_engine_v4_encoder import V4EncoderTests
        from pokezero.category_vocab import CategoryVocabulary
        from gpu_engine import public_exact,public_belief
        V4EncoderTests.setUpClass();cls.f=V4EncoderTests()
        raw=cls.f.public.source;universes=dict(raw.universes);old=universes['snorlax']
        cls.names=('Arena Trap','Cacophony','Early Bird','Shadow Tag','Pressure','Magnet Pull','Immunity')
        universes['snorlax']=replace(old,variants=tuple(replace(old.variants[0],variant_id='exact-'+str(i),ability=name)
            for i,name in enumerate(cls.names)))
        raw=replace(raw,universes=universes);source=public_belief.compile_source(raw)
        labels=set(cls.f.vocab.tokens)
        from pokezero.dex import normalize_id
        for name in cls.names:
            labels|={'ability:'+normalize_id(name),'belief:possible_ability:'+normalize_id(name)}
        vocab=CategoryVocabulary(tuple(sorted(labels)))
        catalog=cls.f.e.compile_encoder(source,cls.f.inputdex,vocab,checkpoint_sha256='a'*64,feature_masks=cls.f.masks)
        cls.f.source,cls.f.vocab,cls.f.catalog=source,vocab,catalog
        cls.f.public=SimpleNamespace(source=raw,prefix=cls.f.public.prefix)
        cls.j,cls.jax,cls.x,cls.catalog=cls.f.j,cls.f.jax,public_exact,catalog
        cls.exact=public_exact.compile_exact(catalog.abilities,catalog.features,catalog.sleep,vocab)
        cls.step=cls.jax.jit(lambda s,c:cls.f.e.fold_chunk(s,c,cls.catalog))
        cls.projectors={}

    def ingest(self,lines):
        state=self.f.e.initialize(self.catalog,stream_ids=self.j.array([1],self.j.int32))
        for chunk in self.f.chunks(self.f.projections(lines)):
            state,error=type(self).step(state,chunk);self.assertEqual(error.tolist(),[0])
        return state

    def project(self,state,owner=0,request=None,excluded=(),exact=True):
        request=request or self.f.request(owner)
        own=self.f.e.parse_own_requests(self.catalog,[request],owner_sides=[owner])
        ability=self.x.parse_own_abilities(request,owner_side=owner,catalog=self.exact,vocabulary=self.f.vocab)
        ability=self.jax.tree.map(lambda x:x[None],ability)
        from gpu_engine import public_belief_snapshot
        summary,admitted=public_belief_snapshot.read(state.context.refresh.controller.visible,state.beliefs,self.catalog.source.device)
        self.assertTrue(bool(admitted.all()))
        ruled=self.j.zeros((1,2,6,len(self.catalog.abilities.names)),bool)
        for name in excluded:ruled=ruled.at[0,1-owner,0,self.catalog.abilities.ability_id(name)].set(True)
        args=(ability,own.actions.team,own.status,state.mons,state.context,state.beliefs,summary,ruled)
        def apply(a,t,s,m,c,b,p,r):
            return self.x.project(a,t,s,m,c,b,p,self.catalog.features,self.catalog.sleep,self.exact.device,
                observer_side=self.j.array([owner],self.j.int32),expected_binding=self.exact.device.binding,
                ruled_out_abilities=r,exact_state=exact)
        key=(owner,exact)
        if key not in self.projectors:self.projectors[key]=self.jax.jit(apply)
        return self.projectors[key],args

    def compare(self,lines,*,owner=0,request=None,excluded=(),exact=True):
        from dataclasses import replace
        from pokezero import showdown as source
        from pokezero.belief import PublicBattleBeliefEngine
        import numpy as np
        state=self.ingest(lines);request=request or self.f.request(owner)
        run,args=self.project(state,owner,request,excluded,exact);got=run(*args)
        self.assertEqual(got.own.admitted.tolist(),[True]);self.assertEqual(got.opponent.admitted.tolist(),[True])
        engine=PublicBattleBeliefEngine(format_id='gen3randombattle',set_source=self.f.public.source)
        for line in lines:engine.ingest_event(source._public_event_from_line(line))
        replay=source.parse_showdown_replay(lines,battle_id='exact-fixture')
        for role,team,patch,side in (
                ('self',source._self_team_from_request(request,f'p{owner+1}'),got.own,owner),
                ('opponent',replay.public_revealed[f'p{2-owner}'],got.opponent,1-owner)):
            exacts={source._normalize_identifier(x.species):x for x in engine.snapshot().side(f'p{side+1}')}
            want=np.zeros((6,2),np.float32)
            for i,candidate in enumerate(team):
                record=exacts.get(source._normalize_identifier(candidate.species))
                if excluded and role=='opponent' and i==0 and record is not None:
                    record=replace(record,ruled_out_abilities=tuple(excluded))
                condition=source._condition_features(candidate.condition);nums=[0.]*200
                if exact:source._encode_mon_exact_state(nums,candidate,record,role=role,status=condition.status,fainted=condition.fainted)
                want[i]=[nums[source.NUMERIC_WAKE_KNOWN],nums[source.NUMERIC_TRAPPER_ALIVE]]
            np.testing.assert_allclose(np.asarray(patch.numeric[0])[:,list(self.x.COLUMNS)],want,atol=1e-6,rtol=0)
        return got,args,run

    def test_real_source_sleep_reveals_switches_private_bench_and_masks(self):
        prefix=list(self.f.public.prefix)
        cases=([],['|-status|p1a: Snorlax|slp|[from] move: Rest'],
            ['|-status|p1a: Snorlax|slp'],
            ['|-status|p1a: Snorlax|slp|[from] move: Rest','|-ability|p1a: Snorlax|Early Bird'],
            ['|-status|p1a: Snorlax|slp|[from] move: Rest','|-ability|p1a: Snorlax|Shadow Tag',
             '|switch|p1a: Gardevoir|Gardevoir, L80|100/100'],
            ['|-status|p1a: Snorlax|slp|[from] move: Rest','|-curestatus|p1a: Snorlax|slp'])
        for suffix in cases:
            for owner in (0,1):
                request=self.f.request(owner)
                request['side']['pokemon'][0]['ability']='Arena Trap'
                if owner==0 and suffix and suffix[0].startswith('|-status'):
                    request['side']['pokemon'][1]['condition']='120/240 slp'
                for exact in (False,True):self.compare(prefix+suffix,owner=owner,request=request,exact=exact)
        # Private active placement, unknown ability and a faint bench all have
        # separate source meanings even when no public encounter exists there.
        for ability,condition,active in (('Magnet Pull','120/240',False),('Shadow Tag','0 fnt',False),
                ('','120/240',False),('Arena Trap','120/240',True)):
            request=self.f.request();row=request['side']['pokemon'][0]
            row.update(ability=ability,condition=condition,active=active)
            request['side']['pokemon'][1]['active']=not active
            self.compare(prefix,request=request)

    def test_full_live_set_explicit_exclusions_and_original_reveal_priority(self):
        lines=[*self.f.public.prefix,'|-status|p1a: Snorlax|slp|[from] move: Rest',
            '|switch|p1a: Gardevoir|Gardevoir, L80|100/100']
        for excluded in ((),('Early Bird',),tuple(n for n in self.names if n!='Shadow Tag'),self.names):
            self.compare(lines,owner=1,excluded=excluded)
        # A public original reveal has priority even over an explicit negative
        # ledger and while the running ability is borrowed elsewhere.
        self.compare([*lines,'|-ability|p1a: Gardevoir|Trace'],owner=1,excluded=('Early Bird',))
        reveal=[*self.f.public.prefix,'|-ability|p1a: Snorlax|Shadow Tag',
            '|switch|p1a: Gardevoir|Gardevoir, L80|100/100']
        self.compare(reveal,owner=1,excluded=('Shadow Tag',))

    def test_bad_owner_namespace_exclusions_and_batch_isolation(self):
        import numpy as np
        state=self.ingest(self.f.public.prefix);run,args=self.project(state)
        good=run(*args);self.assertEqual(good.own.admitted.tolist(),[True])
        a,t,s,m,c,b,p,r=args
        for bad in (a._replace(owner_side=self.j.array([1],self.j.int32)),
                a._replace(ability_binding=a.ability_binding.at[0,0].set(a.ability_binding[0,0]^self.j.uint32(1))),
                a._replace(ability=a.ability.at[0,0].set(len(self.catalog.abilities.names)))):
            out=run(bad,t,s,m,c,b,p,r);self.assertEqual(out.own.admitted.tolist(),[False])
            self.assertFalse(bool(out.own.numeric.any()));self.assertFalse(bool(out.opponent.numeric.any()))
        for bad in (r.at[0,0,0,0].set(True),r.at[0,0,5,1].set(True)):
            out=run(a,t,s,m,c,b,p,bad);self.assertEqual(out.own.admitted.tolist(),[False])
        with self.assertRaisesRegex(ValueError,'shape/dtype'):run(a,t,s,m,c,b,p,r.astype(self.j.int32))
        repeat=lambda x:self.j.repeat(x,8,axis=0)
        # The visible cache namespace is global [8], not batch-owned.
        def widen(x):return self.jax.tree.map(repeat,x)
        cc=widen(c._replace(refresh=None))
        refresh=widen(c.refresh._replace(controller=c.refresh.controller._replace(visible=None)))
        visible=widen(c.refresh.controller.visible._replace(source_binding=self.j.zeros((1,0),self.j.uint32)))
        visible=visible._replace(source_binding=c.refresh.controller.visible.source_binding)
        refresh=refresh._replace(controller=refresh.controller._replace(visible=visible))
        cc=cc._replace(refresh=refresh)
        aa,tt,ss,mm,bb,pp,rr=map(widen,(a,t,s,m,b,p,r))
        aa=aa._replace(owner_side=aa.owner_side.at[0].set(1))
        out=self.jax.jit(lambda a:self.x.project(a,tt,ss,mm,cc,bb,pp,self.catalog.features,self.catalog.sleep,self.exact.device,
            observer_side=self.j.zeros(8,self.j.int32),expected_binding=self.exact.device.binding,ruled_out_abilities=rr))(aa)
        self.assertEqual(out.own.admitted.tolist(),[False]+[True]*7)
        self.assertFalse(bool(out.own.numeric[0].any()))
        np.testing.assert_array_equal(np.asarray(out.own.numeric[1:]),np.repeat(np.asarray(good.own.numeric),7,axis=0))


if __name__=='__main__':unittest.main()
