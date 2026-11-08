"""Constructor worlds into DepthNPricer on the fake engine: the worlds= seam, the adapter, refusals, budgets, bindings, searched mode."""
from dataclasses import replace
import math
from types import MappingProxyType,SimpleNamespace as NS
from typing import NamedTuple
import unittest
from unittest.mock import patch

import jax
import jax.numpy as j
import numpy as np

from pokezero import v5_gumbel as original
from pokezero.live_protocol_actor import MASK_NAMES
from pokezero.v5_contract import DecisionAddress
from gpu_engine import device_depthn_pricer as dn,device_public_teacher as serial,device_public_teacher_batch as batch
from gpu_engine import device_public_world_state as mapping,device_public_worlds as plans,depthn_backups as backups
from gpu_engine import champion_evaluator as evaluators,constructor_worlds as cw
from gpu_engine.phase_d_live_worlds import Refusal,WorldSet
from gpu_engine.phase_d_protocol_constructor import ProtocolConstructor
from tests.test_gpu_engine_device_public_teacher import facts
from tests import test_gpu_engine_depth2_pricer as T,test_gpu_engine_depthn_pricer as N

PARAMS=T.PARAMS
METADATA=None  # set in setUpClass from the fixture source's namespaces
CYCLE=(0,27,21,0,20,22,27,0,21,20,22)   # modes that price completely at depth 2 and 3


class CtorPublic(NamedTuple):
    """A constructor-style carry: the fake's two fields plus extra keys and a None member (from_b2 adds four keys, leaves two None)."""
    clock:object
    turn:object
    features:object
    tendencies:object


def ctor_public(public):
    z=j.zeros((1,2,6),j.int32)
    return CtorPublic(public.clock,public.turn,dict(choice_locked=z,wrap_trap_turns=z),None)


def carry_executor():
    """deep_executor stepping either carry: it reads the fake's two fields and keeps the carry's own type and extra members."""
    base=N.deep_executor()
    plain=lambda public:T.Public(public.clock,public.turn)
    def observe(state,public,owner,mask):return base.observe(state,plain(public),owner,mask)
    def advance(state,public,actions,chance):
        after=base.advance(state,plain(public),actions,chance)
        return after._replace(trackers=public._replace(turn=after.trackers.turn))
    return serial.Executor(base.legal,jax.jit(observe),base.dispatch,base.sample,jax.jit(advance))


def catalog(source_hash,checkpoint='c'*64):
    return NS(checkpoint_sha256=checkpoint,vocabulary=NS(tokens=('field','pokemon:self'),size=2),
        masks=NS(**{k:(0 if k=='transition_token_budget'else True)for k in MASK_NAMES}),source=NS(source_hash=source_hash))


class Evaluator:
    """A champion-like evaluator: the fake head under its own identity (program keys gain it)."""
    identity=('champion','ab'*32,'bfloat16')
    def __call__(self,params,inputs):return T.forward(params,inputs)


class FakeChampion(evaluators.ChampionEvaluator):
    """A real ChampionEvaluator (its view, identity and binding checks) whose forward is the fake head."""
    def __call__(self,params,inputs):return T.forward(params,inputs)


class ConstructorWorldsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _,cls.root,_,cls.tables,cls.recipe=facts()
        cls.source=cls.tables.source
        cls.metadata=dict(vocabulary={k:list(v)for k,v in cls.source.vocabulary.items()},moves=[None,*[dict(pp=10)]*4])
        cls.catalog=catalog(cls.source.source_hash)
        cls.binding=cw.binding(cls.source,cls.metadata,cls.catalog)

    # fixtures ----------------------------------------------------------------------------------------------------------------
    def constructor(self,count,**kw):
        cat=kw.pop('catalog',self.catalog)
        return ProtocolConstructor(catalog=cat,source=kw.pop('source',self.source),metadata=kw.pop('metadata',self.metadata),owning_config=object(),
            count=count,functions=NS(catalog=cat),**kw)

    def made(self,i,owner,modes,*,refused=(),carry='ctor',values=None,b2=None):
        """One root's prepare() output: W fake worlds (prefix-stable seeds), refused slots without a state, B2's legal mask
        (and `b2` as the root's B2 encoding, view['_b2_root_inputs'], when given)."""
        W=len(modes);detached=replace(self.root,address=DecisionAddress(100+i,4+i,owner))
        if b2 is not None:detached=replace(detached,view=MappingProxyType(dict(detached.view,_b2_root_inputs=b2)))
        seeds=tuple(1000*(i+1)+17*w+3 for w in range(W));worlds=[]
        for w,mode in enumerate(modes):
            if w in refused:worlds.append(mapping.World(w,None,None,('named-refusal',),seeds[w],False));continue
            state,public=T.world_data(detached,NS(seed=seeds[w]),mode)
            if values and w in values:state=state._replace(value=j.asarray([values[w]],j.float32))
            worlds.append(mapping.World(w,state,ctor_public(public)if carry=='ctor'else public,(),seeds[w],False))
        return detached,WorldSet(detached,tuple(worlds),seeds,W),T.ROOT_LEGAL

    def modes(self,i,W):
        return tuple(CYCLE[(5*i+w)%len(CYCLE)]for w in range(W))

    def prepared(self,roots,count,**kw):
        """prepare_roots through the real ProtocolConstructor with prepare() stubbed: `roots` maps request ids to made() tuples or
        to a Refusal to raise."""
        constructor=self.constructor(count,**kw);calls=[]
        def prepare(request,index):
            calls.append((request['request_id'],index));out=roots[request['request_id']]
            if isinstance(out,Exception):raise out
            return out
        with patch.object(ProtocolConstructor,'prepare',side_effect=prepare):
            out=cw.prepare_roots(constructor,[dict(request_id=k)for k in roots])
        return constructor,out,calls

    def pricer(self,ex,**kw):
        kw.setdefault('capacity',8);kw.setdefault('block_lanes',40);kw.setdefault('actor_capacity',3)
        return dn.DepthNPricer(ex,object(),**kw)

    def forward_patch(self):return patch.object(batch.v5_model,'forward',side_effect=T.forward)

    def assert_frames_equal(self,a,b,msg=''):
        A=dict(dn.arrays(a));B=dict(dn.arrays(b));self.assertEqual(set(A),set(B))
        for name in A:
            self.assertEqual((A[name].shape,A[name].dtype),(B[name].shape,B[name].dtype),name)
            np.testing.assert_array_equal(A[name],B[name],err_msg=f'{name} {msg}')

    def rows_equal(self,a,ra,b,rb):
        for (name,x),(_,y)in zip(dn.arrays(a),dn.arrays(b)):np.testing.assert_array_equal(x[ra],y[rb],err_msg=name)

    def row_specs(self,batch_,modes,owners):
        """check_oracle specs per request row (modes of the row's two worlds, owner, actors 0/0) and its missing slots."""
        specs=[];missing=set()
        for root in batch_.roots:
            if root['status']!='PRICED':continue
            for b,row in enumerate(root['rows']):
                specs.append(((modes[root['index']][2*b],modes[root['index']][2*b+1]),owners[root['index']],0,0))
                for wi in range(2):
                    if 2*b+wi in root['refused_worlds']:missing.add((row,wi))
        return tuple(specs),missing

    # (1) the seam ------------------------------------------------------------------------------------------------------------
    def test_seam_is_transparent_and_d3_and_prebuilt_programs_never_share(self):
        specs=(((0,27),0,0,1),((27,21),1,1,0),((22,0),0,2,2));missing=((1,1),)
        ex=carry_executor();engine=self.pricer(ex,depth=3,k_own=(2,3),k_opp=(3,2),blocks_per_call=4)
        requests=T.Depth2PricerTests.requests(self,specs,ex);materialize=T.Depth2PricerTests.materialize(specs,missing)
        with patch.object(mapping,'materialize_host',side_effect=materialize),self.forward_patch():
            d3=engine.price(requests,self.tables,self.recipe,PARAMS)
        d3_keys=set(engine.programs)
        self.assertTrue(d3_keys and all(not any(isinstance(x,tuple)and x[:1]==('worlds',)for x in key)for key in d3_keys))
        self.assertIn(('expand',6,4,2,3),d3_keys);self.assertIn(('leaf',6,4,3,2),d3_keys);self.assertNotIn('world_source',d3.timing)
        self.assertTrue(all('world_source'not in w for pair in d3.world_reports for w in pair))   # D3's reports are unchanged
        pairs=tuple(tuple(materialize(r.plans.root,s,self.tables,self.recipe)for s in r.plans.starts)for r in requests)
        ctor_pairs=tuple(tuple(replace(w,public=None if w.public is None else ctor_public(w.public))for w in pair)for pair in pairs)
        frames={};sigs={}
        for tag,given in(('same-carry',pairs),(cw.SOURCE,ctor_pairs)):
            before=set(engine.programs)
            with self.forward_patch():frames[tag]=engine.price(requests,self.binding,self.recipe,PARAMS,worlds=dn.PrebuiltWorlds(tag,self.binding,given))
            new=set(engine.programs)-before;sigs[tag]=frames[tag].timing['world_carry']
            # the same shapes as D3's call, yet every program compiled anew under its own suffix
            self.assertEqual({k[:-1]for k in new},d3_keys);self.assertTrue(all(k[-1]==('worlds',tag,sigs[tag])for k in new))
            self.assertTrue(engine.compiles);self.assertEqual(frames[tag].timing['world_source'],tag)
            self.assert_frames_equal(d3,frames[tag],tag)   # the seam is transparent: D3's own worlds, given pre-built, price bit-equal
            self.assertEqual(frames[tag].world_reports[1][1]['reason'],'physical-materialization:inert missing')
            self.assertTrue(all(w['world_source']==tag for pair in frames[tag].world_reports for w in pair))
            self.assertEqual([(w['status'],w['reason'])for pair in frames[tag].world_reports for w in pair],[(w['status'],w['reason'])for pair in d3.world_reports for w in pair])
        self.assertNotEqual(sigs['same-carry'],sigs[cw.SOURCE])   # two carry structures, two signatures
        N.DepthNPricerTests.check_oracle(self,frames[cw.SOURCE],requests,ex,specs,((2,3),(3,2)),missing=missing)
        with patch.object(mapping,'materialize_host',side_effect=materialize),self.forward_patch():
            again=engine.price(requests,self.tables,self.recipe,PARAMS)
        self.assertEqual(engine.compiles,());self.assert_frames_equal(d3,again)   # D3 reuses its own programs, untouched
        # A table shared between pricers (the latency tool's) keeps the partition: a sibling reuses the constructor programs.
        sibling=self.pricer(ex,depth=3,k_own=(2,3),k_opp=(3,2),blocks_per_call=4);sibling.bind(PARAMS);sibling.programs=engine.programs
        with self.forward_patch():shared=sibling.price(requests,self.binding,self.recipe,PARAMS,worlds=dn.PrebuiltWorlds(cw.SOURCE,self.binding,ctor_pairs))
        self.assertEqual(sibling.compiles,());self.assert_frames_equal(frames[cw.SOURCE],shared)
        # One frame holds one world source: mixed carries refuse before any program runs.
        mixed=(ctor_pairs[0],)+pairs[1:]
        with self.assertRaisesRegex(ValueError,'one world source per frame'):engine.price(requests,self.binding,self.recipe,PARAMS,worlds=dn.PrebuiltWorlds(cw.SOURCE,self.binding,mixed))

    # (2) the adapter end to end ----------------------------------------------------------------------------------------------
    def test_adapter_prices_constructor_worlds_end_to_end_against_the_oracle(self):
        W=8;owners=(0,1,0);modes={i:self.modes(i,W)for i in range(3)}
        roots={'r0':self.made(0,0,modes[0]),'r1':Refusal('SEARCH_PREFIX_OR_REQUEST_MISMATCH'),'r2':self.made(2,0,modes[2])}
        constructor,prepared,calls=self.prepared(roots,W)
        self.assertEqual(calls,[('r0',0),('r1',1),('r2',2)]);self.assertEqual([type(p).__name__ for p in prepared],['Prepared','RootRefusal','Prepared'])
        self.assertTrue(all(p.seconds>=0 for p in prepared))
        ex=carry_executor();engine=self.pricer(ex,depth=3,k_own=(2,2),k_opp=(2,2),capacity=8,block_lanes=16)
        with self.forward_patch():
            b=cw.build(constructor,prepared,engine,PARAMS,binding=self.binding,recipe=self.recipe)
            frame,raw=cw.price(engine,b,self.binding,self.recipe,PARAMS)
        self.assertIsNone(raw);self.assertEqual((b.count,b.blocks,len(b.requests)),(8,4,8))
        self.assertEqual([r['status']for r in b.roots],['PRICED','REFUSED','PRICED'])
        self.assertEqual(b.roots[1]['reason'],'constructor-root:SEARCH_PREFIX_OR_REQUEST_MISMATCH')
        self.assertEqual((b.roots[0]['rows'],b.roots[2]['rows']),((0,1,2,3),(4,5,6,7)))
        for r,(root,rows)in enumerate(((0,(0,1,2,3)),(2,(4,5,6,7)))):
            for k,row in enumerate(rows):
                req=b.requests[row];self.assertEqual(tuple(s.world for s in req.plans.starts),(2*k,2*k+1))
                self.assertIs(req.plans.root,roots[f'r{root}'][0]);self.assertTrue(req.plans.selected)
                self.assertEqual(tuple(s.seed for s in req.plans.starts),roots[f'r{root}'][1].seeds[2*k:2*k+2])
                self.assertEqual([w['world']for w in frame.world_reports[row]],[2*k,2*k+1])
        specs,missing=self.row_specs(b,modes,owners)
        N.DepthNPricerTests.check_oracle(self,frame,b.requests,ex,specs,((2,2),(2,2)),missing=missing,params=PARAMS)
        results=cw.reduce(frame,b)
        self.assertEqual([x['status']for x in results],['LABELED','REFUSED','LABELED'])
        for x,root in zip(results,b.roots):
            if root['status']!='PRICED':self.assertEqual(x['cause'],root['reason']);continue
            # the composition by hand: reduce_completed_q per block, the blocks' completed-Q averaged (every world 1/W)
            qs=[backups.learner_max_root(frame,row,root['root_logits'],root['root_legal']).improvement.completed_q for row in root['rows']]
            q=[math.fsum(float(z[a])for z in qs)/4 if root['root_legal'][a]else None for a in range(9)]
            self.assertEqual(x['completed_q'],tuple(q));self.assertEqual(x['block_causes'],(None,)*4)
            target=original._probabilities(tuple(l+(q[a]or 0.)for a,l in enumerate(root['root_logits'])),root['root_legal'])
            self.assertEqual(x['policy_target'],target);legal=[a for a in range(9)if root['root_legal'][a]]
            self.assertEqual(x['action_policy_target'],max(legal,key=lambda a:(target[a],-a)));self.assertEqual(x['action_mean_q'],max(legal,key=lambda a:(q[a],-a)))
            self.assertIsNone(root['b2_view']);self.assertEqual(root['refused_world_share'],0.)   # the fixture root carries no B2 encoding
            self.assertTrue(all(math.isnan(b.requests[row].plans.source_fixture_unknown_mass)for row in root['rows']))

    # (3) budgets -------------------------------------------------------------------------------------------------------------
    def test_world_budgets_2_8_16_32_are_prefixes_and_compose_by_the_mean(self):
        ex=carry_executor();engine=self.pricer(ex,depth=2,k_own=(2,),k_opp=(2,),capacity=32,block_lanes=16);frames={};batches={};results={}
        for W in(2,8,16,32):
            modes={i:self.modes(i,W)for i in range(2)}
            constructor,prepared,_=self.prepared({'a':self.made(0,0,modes[0]),'b':self.made(1,1,modes[1])},W)
            with self.forward_patch():
                b=batches[W]=cw.build(constructor,prepared,engine,PARAMS,binding=self.binding,recipe=self.recipe)
                frames[W],_=cw.price(engine,b,self.binding,self.recipe,PARAMS)
            K=W//2;self.assertEqual((b.blocks,len(b.requests),frames[W].pre.shape),(K,2*K,(2*K,2)))
            self.assertEqual([r['rows']for r in b.roots],[tuple(range(K)),tuple(range(K,2*K))])
            results[W]=cw.reduce(frames[W],b);self.assertTrue(all(x['status']=='LABELED'and x['blocks']==K and x['worlds']==W for x in results[W]))
            for x,root in zip(results[W],b.roots):
                single=[backups.learner_max_root(frames[W],row,root['root_logits'],root['root_legal'])for row in root['rows']]
                if K==1:self.assertEqual(x['completed_q'],tuple(single[0].improvement.completed_q));self.assertEqual(x['root_value'],single[0].improvement.root_value)
                else:self.assertEqual(x['completed_q'],tuple(math.fsum(float(s.improvement.completed_q[a])for s in single)/K if root['root_legal'][a]else None for a in range(9)))
        for W in(2,8,16):   # the same draws: a smaller budget's blocks are the first blocks of W=32, value for value
            for i in range(2):
                for k in range(W//2):self.rows_equal(frames[W],batches[W].roots[i]['rows'][k],frames[32],batches[32].roots[i]['rows'][k])
        self.assertEqual(sum(key[0]=='P1'for key in engine.programs),4)
        self.assertTrue(all(key[-1][:2]==('worlds',cw.SOURCE)for key in engine.programs))

    # (4) refusals ------------------------------------------------------------------------------------------------------------
    def test_a_world_refusal_is_pre_minus_one_and_refuses_the_root(self):
        W=8;owners=(0,1,0,1);modes={i:self.modes(i,W)for i in range(4)}
        roots={'one-refused':self.made(0,0,modes[0],refused=(3,)),'complete':self.made(1,1,modes[1]),
            'world0-refused':self.made(2,0,modes[2],refused=(0,)),'all-refused':self.made(3,1,modes[3],refused=tuple(range(W)))}
        constructor,prepared,_=self.prepared(roots,W)
        ex=carry_executor();engine=self.pricer(ex,depth=2,k_own=(2,),k_opp=(2,),capacity=16,block_lanes=16)
        with self.forward_patch():
            b=cw.build(constructor,prepared,engine,PARAMS,binding=self.binding,recipe=self.recipe)
            frame,_=cw.price(engine,b,self.binding,self.recipe,PARAMS)
        self.assertEqual([r['status']for r in b.roots],['PRICED','PRICED','PRICED','REFUSED'])
        self.assertEqual(b.roots[3]['reason'],'WORLD_CONSTRUCTOR_REFUSED:named-refusal');self.assertEqual(b.roots[3]['rows'],())
        self.assertEqual((b.roots[0]['refused_worlds'],b.roots[0]['constructed'],b.roots[2]['root_view_world']),((3,),7,1))
        row=b.roots[0]['rows'][1]
        self.assertEqual(int(frame.pre[row,1]),dn.UNAVAILABLE);self.assertEqual(int(frame.pre[row,0]),0)
        self.assertEqual(frame.world_reports[row][1]['reason'],'physical-materialization:named-refusal')
        self.assertEqual(frame.world_reports[row][1]['world'],3);self.assertFalse(frame.t1_status[row,1].any())
        self.assertEqual(int(frame.pre[b.roots[2]['rows'][0],0]),dn.UNAVAILABLE)
        results=cw.reduce(frame,b)
        self.assertEqual([(x['status'],x['cause'])for x in results],[('REFUSED','materialization'),('LABELED',None),('REFUSED','materialization'),
            ('REFUSED','WORLD_CONSTRUCTOR_REFUSED:named-refusal')])
        self.assertEqual(results[0]['block_causes'],(None,'materialization',None,None))
        specs,missing=self.row_specs(b,modes,owners);self.assertEqual(missing,{(1,1),(8,0)})
        N.DepthNPricerTests.check_oracle(self,frame,b.requests,ex,specs,((2,2),),missing=missing)
        # only_complete: refused before pricing, no rows, no device work for them
        with self.forward_patch():strict=cw.build(constructor,prepared,engine,PARAMS,binding=self.binding,recipe=self.recipe,only_complete=True)
        self.assertEqual([r['status']for r in strict.roots],['REFUSED','PRICED','REFUSED','REFUSED']);self.assertEqual(len(strict.requests),4)

    def test_an_unnamed_world_refusal_stays_the_roots_and_an_all_refused_batch_prices_nothing(self):
        W=2;detached,made,legal=self.made(0,0,self.modes(0,W))
        made=replace(made,worlds=(made.worlds[0],replace(made.worlds[1],state=None,public=None)))   # no state, no named refusal
        constructor,prepared,_=self.prepared({'r':(detached,made,legal),'b':self.made(1,1,self.modes(1,W))},W)
        ex=carry_executor();engine=self.pricer(ex,depth=2,k_own=(2,),k_opp=(2,),capacity=4,block_lanes=16)
        with self.forward_patch():
            b=cw.build(constructor,prepared,engine,PARAMS,binding=self.binding,recipe=self.recipe)
            frame,_=cw.price(engine,b,self.binding,self.recipe,PARAMS)
        self.assertEqual(frame.world_reports[0][1]['reason'],'physical-materialization:'+cw.UNNAMED)
        self.assertEqual([(x['status'],x['cause'])for x in cw.reduce(frame,b)],[('REFUSED','materialization'),('LABELED',None)])
        with self.assertRaisesRegex(TypeError,'prepare must return'):self.prepared({'r':(detached,made,legal,'a fourth member')},W)   # never a silent refusal
        constructor,prepared,_=self.prepared({'x':Refusal('MALFORMED_SEARCH_REQUEST'),'y':ValueError('plain')},W)
        deep=self.pricer(ex,depth=3,k_own=(2,2),k_opp=(2,2),selection='searched');full=self.pricer(ex,depth=2,k_own=9,k_opp=9)
        digest=self.pricer(ex,depth=3,k_own=(2,2),k_opp=(2,2),digest=True)   # its empty frame would raise KeyError 'digest_own' in DepthN's _frame
        for pricer,pass_pricer in((engine,None),(deep,full),(digest,None)):
            before=pricer.last_timing,None if pass_pricer is None else pass_pricer.last_timing
            with self.forward_patch():
                b=cw.build(constructor,prepared,pricer,PARAMS,binding=self.binding,recipe=self.recipe)
                frame,raw=cw.price(pricer,b,self.binding,self.recipe,PARAMS,pass_pricer=pass_pricer)
            self.assertEqual((len(b.requests),frame,raw),(0,None,None))   # no pricer is called
            self.assertIs(pricer.last_timing,before[0]);self.assertIs(None if pass_pricer is None else pass_pricer.last_timing,before[1])
            self.assertEqual([x['cause']for x in cw.reduce(frame,b)],['constructor-root:MALFORMED_SEARCH_REQUEST','constructor-root:ValueError:plain'])
        with self.assertRaises(ValueError):cw.price(deep,b,self.binding,self.recipe,PARAMS)   # the mode's pricers are still checked
        with self.assertRaises(ValueError):cw.reduce(None,cw.Batch((object(),),b.worlds,(),2,1))

    def test_root_inputs_are_world0s_own_view_and_hold_every_world_to_it(self):
        W=8;modes=self.modes(0,W)
        roots={'r':self.made(0,0,modes,values={5:.5}),'legal':self.made(1,0,modes)}
        roots['legal']=roots['legal'][:2]+((True,True,False,True)+(False,)*5,)   # B2's mask differs from every world's own encoder mask
        constructor,prepared,_=self.prepared(roots,W)
        ex=carry_executor();engine=self.pricer(ex,depth=2,k_own=(2,),k_opp=(2,),capacity=8,block_lanes=16)
        with self.forward_patch():
            b=cw.build(constructor,prepared,engine,PARAMS,binding=self.binding,recipe=self.recipe)
            frame,_=cw.price(engine,b,self.binding,self.recipe,PARAMS)
        world0=roots['r'][1].worlds[0];masks=ex.legal(world0.state)
        own=ex.observe(world0.state,world0.public,j.asarray([0],j.int32),masks[:,0]).inputs
        for row in b.roots[0]['rows']:
            req=b.requests[row];self.assertEqual(set(req.root_inputs),set(own))
            for k in own:np.testing.assert_array_equal(req.root_inputs[k],np.asarray(own[k]))
        expected=T.forward(PARAMS[0],own);self.assertEqual(b.roots[0]['root_logits'],tuple(map(float,np.asarray(expected.policy_logits).reshape(-1))))
        pre=frame.pre[list(b.roots[0]['rows'])]
        self.assertEqual(int(pre[2,1]),3);self.assertEqual(int(np.count_nonzero(pre)),1)   # world 5's own view differs: numeric_features, code 3
        self.assertTrue((frame.pre[list(b.roots[1]['rows'])]==2).all())   # every world's encoder mask differs from B2's: code 2
        self.assertEqual([(x['status'],x['cause'])for x in cw.reduce(frame,b)],[('REFUSED','pre'),('REFUSED','pre')])

    # (5) bindings ------------------------------------------------------------------------------------------------------------
    def test_binding_mismatches_fail_closed(self):
        W=2;roots={'r':self.made(0,0,self.modes(0,W))}
        constructor,prepared,_=self.prepared(roots,W)
        ex=carry_executor();engine=self.pricer(ex,depth=2,k_own=(2,),k_opp=(2,),capacity=4,block_lanes=16)
        with self.forward_patch():
            b=cw.build(constructor,prepared,engine,PARAMS,binding=self.binding,recipe=self.recipe)
            control=engine.price(b.requests,self.binding,self.recipe,PARAMS,worlds=b.worlds)   # every case below differs from this priced call by one field
        self.assertEqual(control.pre.tolist(),[[0,0]])
        metadata=dict(self.metadata,moves=[None,*[dict(pp=11)]*4])
        masks=NS(**dict(vars(self.catalog),masks=NS(**dict(vars(self.catalog.masks),stats_block=False))))
        vocab=NS(**dict(vars(self.catalog),vocabulary=NS(tokens=('field','pokemon:other'),size=2)))
        others={('metadata_sha256',None):cw.binding(self.source,metadata,self.catalog),
            ('encoder','checkpoint_sha256'):cw.binding(self.source,self.metadata,catalog(self.source.source_hash,checkpoint='d'*64)),
            ('encoder','feature_masks'):cw.binding(self.source,self.metadata,masks),
            ('encoder','category_vocab_sha256'):cw.binding(self.source,self.metadata,vocab),
            ('source_binding_sha256',None):cw.binding(replace(self.source,binding_sha256='0'*64),self.metadata,self.catalog)}
        for (field,part),other in others.items():
            with self.subTest(field=field,part=part):
                if part is not None:   # exactly that encoder entry differs
                    self.assertEqual([k for k,v in dict(other.encoder).items()if dict(self.binding.encoder)[k]!=v],[part])
                with self.forward_patch():
                    with self.assertRaisesRegex(ValueError,"built under another binding than the pricer's: "+field+'$'):
                        cw.build(constructor,prepared,engine,PARAMS,binding=other,recipe=self.recipe)
                    with self.assertRaisesRegex(ValueError,"built under another binding than the pricer's: "+field+'$'):
                        engine.price(b.requests,other,self.recipe,PARAMS,worlds=b.worlds)
                    with self.assertRaisesRegex(ValueError,"built under another binding than the pricer's: "+field+'$'):
                        engine.price(b.requests,self.binding,self.recipe,PARAMS,worlds=replace(b.worlds,binding=other))
        # The constructor's own inputs, not the pricer's: a constructor on other metadata refuses to build for this pricer,
        # and its prepared roots refuse to build under a constructor that does match (they carry their maker's stamp).
        foreign,prepared2,_=self.prepared(roots,W,metadata=metadata)
        self.assertEqual(prepared2[0].binding,cw.constructor_binding(foreign));self.assertEqual(prepared[0].binding,self.binding)
        with self.forward_patch():
            with self.assertRaisesRegex(ValueError,"constructor worlds are built under another binding than the pricer's: metadata_sha256$"):
                cw.build(foreign,prepared2,engine,PARAMS,binding=self.binding,recipe=self.recipe)
            with self.assertRaisesRegex(ValueError,"a prepared root comes from a constructor with another binding than this one's: metadata_sha256$"):
                cw.build(constructor,prepared2,engine,PARAMS,binding=self.binding,recipe=self.recipe)
            refused=cw.prepare_roots(self.constructor(W,metadata=metadata),[dict(request_id='x')])   # a RootRefusal is stamped too
            with self.assertRaisesRegex(ValueError,"a prepared root comes from a constructor"):
                cw.build(constructor,prepared+refused,engine,PARAMS,binding=self.binding,recipe=self.recipe)
        req=b.requests[0];pair=b.worlds.pairs[0]
        bad={'plan source binding':(replace(req,plans=replace(req.plans,root=replace(req.plans.root,source_binding_sha256='0'*64))),'root source binding differs'),
            'plan source hash':(replace(req,plans=replace(req.plans,source_hash='other')),'plan public source differs'),
            'unselected':(replace(req,plans=replace(req.plans,selected=False)),'selected plans required'),
            'world index':(replace(req,plans=replace(req.plans,starts=req.plans.starts[::-1])),'plan starts must name the pair'),
            'actor':(replace(req,learner_actor=3),'verified original actor index'),
            'input keys':(replace(req,root_inputs=dict(req.root_inputs,extra=np.zeros(1))),'original encoder input keys')}
        for name,(r,message)in bad.items():
            with self.subTest(request=name),self.forward_patch(),self.assertRaisesRegex(ValueError,message):
                engine.price((r,),self.binding,self.recipe,PARAMS,worlds=b.worlds)
        twice=replace(req,plans=replace(req.plans,starts=(req.plans.starts[0],)*2))
        for name,requests,pairs,message in(('count',b.requests,(),'one pre-built world pair per request'),
                ('unnamed refusal',b.requests,((replace(pair[0],state=None,public=None),pair[1]),),'naming its refusals'),
                ('three',b.requests,((pair[0],)+pair,),'two device_public_world_state.World per request'),
                ('not a World',b.requests,((NS(**vars(pair[0])),pair[1]),),'two device_public_world_state.World per request'),
                ('one world twice',(twice,),((pair[0],pair[0]),),'two distinct worlds per request'),
                ('one world twice, starts in order',b.requests,((pair[0],pair[0]),),'two distinct worlds per request')):
            with self.subTest(pairs=name),self.forward_patch(),self.assertRaisesRegex(ValueError,message):
                engine.price(requests,self.binding,self.recipe,PARAMS,worlds=replace(b.worlds,pairs=pairs))
        with self.forward_patch():
            for materializer,worlds,recipe in((self.tables,b.worlds,self.recipe),(self.binding,b.worlds,None),(self.binding,b.worlds.pairs,self.recipe)):
                with self.assertRaisesRegex(ValueError,'pre-built worlds need the pricer-side WorldBinding'):engine.price(b.requests,materializer,recipe,PARAMS,worlds=worlds)
            with self.assertRaisesRegex(ValueError,'original materializer and random recipe required'):engine.price(b.requests,self.binding,self.recipe,PARAMS)
            with self.assertRaisesRegex(ValueError,'a world-source tag required'):engine.price(b.requests,self.binding,self.recipe,PARAMS,worlds=replace(b.worlds,source=''))
            empty=engine.price((),self.binding,self.recipe,PARAMS,worlds=replace(b.worlds,pairs=()))
        self.assertEqual(empty.pre.shape,(0,2))

    # (6) searched mode -------------------------------------------------------------------------------------------------------
    def test_searched_mode_prices_both_passes_on_the_same_worlds(self):
        specs=(((0,27),0,0,1),((27,0),1,1,0));ex=carry_executor()
        requests=T.Depth2PricerTests.requests(self,specs,ex);materialize=T.Depth2PricerTests.materialize(specs)
        pass_pricer=self.pricer(ex,depth=2,k_own=9,k_opp=9);deep=self.pricer(ex,depth=3,k_own=(2,2),k_opp=(2,2),selection='searched')
        with patch.object(mapping,'materialize_host',side_effect=materialize),self.forward_patch():
            raw=pass_pricer.price(requests,self.tables,self.recipe,PARAMS)
            own,opp,_=dn.searched_maps(dn.as_depth2_frame(raw),2,2)
            d3=deep.price(requests,self.tables,self.recipe,PARAMS,selection_maps={2:(own,opp)})
        pairs=tuple(tuple(replace(w,public=ctor_public(w.public))for w in(materialize(r.plans.root,s,self.tables,self.recipe)for s in r.plans.starts))for r in requests)
        given=cw.Batch(requests,dn.PrebuiltWorlds(cw.SOURCE,self.binding,pairs),(),2,1)
        with self.forward_patch():frame,raw2=cw.price(deep,given,self.binding,self.recipe,PARAMS,pass_pricer=pass_pricer)
        self.assert_frames_equal(raw,raw2,'pass');self.assert_frames_equal(d3,frame,'searched')
        self.assertEqual(frame.t1_own_actions.shape[-1],dn.searched_widths(2,2)[0])
        # Through the adapter: prepare once per root, both passes on its worlds.
        W=8;modes={i:self.modes(i,W)for i in range(2)}
        constructor,prepared,calls=self.prepared({'a':self.made(0,0,modes[0]),'b':self.made(1,1,modes[1])},W)
        with self.forward_patch():
            b=cw.build(constructor,prepared,deep,PARAMS,binding=self.binding,recipe=self.recipe)
            frame,raw=cw.price(deep,b,self.binding,self.recipe,PARAMS,pass_pricer=pass_pricer)
        self.assertEqual(len(calls),2);self.assertEqual(frame.timing['world_source'],cw.SOURCE);self.assertEqual(raw.timing['world_carry'],frame.timing['world_carry'])
        self.assertTrue(all(x['status']=='LABELED'for x in cw.reduce(frame,b)))
        for bad in(None,self.pricer(ex,depth=2,k_own=2,k_opp=2),self.pricer(carry_executor(),depth=2,k_own=9,k_opp=9),self.pricer(ex,depth=2,k_own=9,k_opp=9,forward=Evaluator())):
            with self.assertRaises(ValueError):cw.price(deep,b,self.binding,self.recipe,PARAMS,pass_pricer=bad)
        with self.assertRaises(ValueError):cw.price(pass_pricer,b,self.binding,self.recipe,PARAMS,pass_pricer=pass_pricer)

    # (7) the champion seam -----------------------------------------------------------------------------------------------------
    def test_an_evaluator_prices_root_and_leaves_and_its_keys_carry_both_suffixes(self):
        W=2;constructor,prepared,_=self.prepared({'r':self.made(0,0,self.modes(0,W))},W)
        ex=carry_executor();evaluator=Evaluator()
        champion=self.pricer(ex,depth=2,k_own=(2,),k_opp=(2,),capacity=2,block_lanes=16,forward=evaluator)
        b=cw.build(constructor,prepared,champion,PARAMS[:1],binding=self.binding,recipe=self.recipe)   # no V2 patch: the evaluator alone
        frame,_=cw.price(champion,b,self.binding,self.recipe,PARAMS[:1])
        world0=prepared[0].made.worlds[0];own=ex.observe(world0.state,world0.public,j.asarray([0],j.int32),ex.legal(world0.state)[:,0]).inputs
        self.assertEqual(b.roots[0]['root_logits'],tuple(map(float,np.asarray(evaluator(PARAMS[0],own).policy_logits).reshape(-1))))
        self.assertTrue(all(key[-2:]==(('evaluator',)+Evaluator.identity,('worlds',cw.SOURCE,frame.timing['world_carry']))for key in champion.programs))
        self.assertEqual(cw.reduce(frame,b)[0]['status'],'LABELED')
        v2=self.pricer(ex,depth=2,k_own=(2,),k_opp=(2,),capacity=2,block_lanes=16);v2.bind(PARAMS[:1]);v2.programs=champion.programs
        with self.forward_patch():
            b2=cw.build(constructor,prepared,v2,PARAMS[:1],binding=self.binding,recipe=self.recipe);cw.price(v2,b2,self.binding,self.recipe,PARAMS[:1])
        self.assertTrue(v2.compiles and all(c[0][-1][0]=='worlds'and not any(isinstance(x,tuple)and x[:1]==('evaluator',)for x in c[0])
            for c in v2.compiles))   # V2 compiled its own programs: never a champion program from the shared table


    # (8) review follow-ups -----------------------------------------------------------------------------------------------------
    def test_a_champion_evaluator_must_be_bound_to_the_constructors_catalog(self):
        W=2;constructor,prepared,_=self.prepared({'r':self.made(0,0,self.modes(0,W))},W);ex=carry_executor()
        champion=FakeChampion(params=None,config=NS(tokens=23),meta=dict(weights_archive_sha256='ab'*32,checkpoint_sha256='cd'*32))
        pricer=self.pricer(ex,depth=2,k_own=(2,),k_opp=(2,),capacity=2,block_lanes=16,forward=champion)
        with self.assertRaisesRegex(ValueError,'9375 champion checkpoint'):   # the real check: not the real checkpoint
            cw.build(constructor,prepared,pricer,PARAMS[:1],binding=self.binding,recipe=self.recipe)
        with patch.object(cw.evaluators,'check_binding',side_effect=ValueError('catalog not the champion\'s'))as check:
            with self.assertRaisesRegex(ValueError,'catalog not the champion'):cw.build(constructor,prepared,pricer,PARAMS[:1],binding=self.binding,recipe=self.recipe)
        check.assert_called_once_with(champion,constructor.catalog)
        with patch.object(cw.evaluators,'check_binding',return_value=True)as check:
            b=cw.build(constructor,prepared,pricer,PARAMS[:1],binding=self.binding,recipe=self.recipe)
        check.assert_called_once_with(champion,constructor.catalog);self.assertEqual(b.roots[0]['status'],'PRICED')
        with patch.object(cw.evaluators,'check_binding',side_effect=AssertionError('not for a non-champion forward')):   # V2 and plain evaluators
            for forward in(None,Evaluator()):
                with self.forward_patch():cw.build(constructor,prepared,self.pricer(ex,depth=2,k_own=(2,),k_opp=(2,),capacity=2,forward=forward),
                    PARAMS[:1],binding=self.binding,recipe=self.recipe)

    def test_b2_view_diagnostic_compares_the_evaluator_view_with_b2s_encoding(self):
        rng=np.random.default_rng(7)
        v2=dict(numeric_features=rng.standard_normal((1,1,29,132)).astype(np.float32),categorical_ids=rng.integers(0,40,(1,1,29,41)).astype(np.int32),
            token_type_ids=rng.integers(0,9,(1,1,29)).astype(np.int32),attention_mask=rng.integers(0,2,(1,1,29)).astype(bool),history_mask=np.ones((1,1),bool))
        b2={k:v2[k][0,0,:23]for k in cw.B2_TOKEN_KEYS};b2['legal_action_mask']=np.ones(9,bool)
        champion=FakeChampion(params=None,config=NS(tokens=23),meta=dict(weights_archive_sha256='ab'*32,checkpoint_sha256='cd'*32))
        root=lambda b:NS(view=MappingProxyType(dict(species=0,**({}if b is None else{'_b2_root_inputs':MappingProxyType(b)}))))
        for forward in(None,Evaluator(),champion):   # the 23-token prefix, or the evaluator's own view
            self.assertEqual(cw.b2_view_check(forward,v2,root(b2)),dict(equal=True,keys={k:True for k in cw.B2_TOKEN_KEYS},numeric_max_abs_diff=0.))
            self.assertIsNone(cw.b2_view_check(forward,v2,root(None)))
        off=dict(b2,numeric_features=b2['numeric_features'].copy());off['numeric_features'][3,5]+=.25
        out=cw.b2_view_check(champion,v2,root(off))
        self.assertEqual((out['equal'],out['keys']['numeric_features'],out['keys']['categorical_ids'],out['numeric_max_abs_diff']),(False,False,True,.25))
        tail=dict(v2,numeric_features=v2['numeric_features'].copy());tail['numeric_features'][0,0,25]+=1   # belief tokens 23..28 are not B2's
        self.assertTrue(cw.b2_view_check(None,tail,root(b2))['equal'])
        self.assertEqual(cw.b2_view_check(champion,dict(v2,numeric_features=v2['numeric_features'][...,:5,:]),root(b2))['equal'],None)   # an error, not a gate
        # Through build: reported per root, never a gate (the fake encoder's inputs cannot be compared: the root still prices).
        W=2;constructor,prepared,_=self.prepared({'r':self.made(0,0,self.modes(0,W),b2=b2)},W)
        with self.forward_patch():b=cw.build(constructor,prepared,self.pricer(carry_executor(),depth=2,k_own=(2,),k_opp=(2,),capacity=2),
            PARAMS,binding=self.binding,recipe=self.recipe)
        self.assertEqual(b.roots[0]['status'],'PRICED');self.assertIsNone(b.roots[0]['b2_view']['equal']);self.assertIn('KeyError',b.roots[0]['b2_view']['error'])

    def test_a_reduction_error_refuses_its_root_only(self):
        """reduce_completed_q's range check raises when a block's completed-Q rounds outside [-1, 1]: here action 0 wins against
        every reply in both worlds of a block whose reply prior sums to 1 + ulp. That root alone refuses (cause 'reduction',
        the error kept); the batch's other root still labels."""
        legal=(True,True)+(False,)*7;opp=(True,)*3+(False,)*6;logits=(1.746,-2.561,-1.426)+(0.,)*6
        world=lambda v0,v1:original.WorldPrices(True,opp,logits,tuple(tuple(((v0,v1)[a]if a<2 and b<3 else None)for b in range(9))for a in range(9)))
        p=original._probabilities(logits,opp);self.assertEqual(math.fsum([x/2 for x in p[:3]]*2),1+2.**-52)   # the all-win block's completed-Q
        wins=world(1.,0.);plain=world(.1,.2);blocks={0:(wins,wins),1:(plain,plain),2:(plain,plain),3:(plain,plain)}
        roots=tuple(dict(index=i,request_id=f'r{i}',status='PRICED',reason=None,rows=rows,root_logits=(0.,)*9,root_legal=legal)for i,rows in enumerate(((0,1),(2,3))))
        with patch.object(cw.backups,'backup',side_effect=lambda frame,row,rule:blocks[row]),\
                patch.object(cw.backups,'refusal_summary',return_value=dict(roots=[dict(cause=None)]*4)):
            results=cw.reduce(object(),cw.Batch((object(),)*4,None,roots,4,2))
        self.assertEqual([(x['status'],x['cause'])for x in results],[('REFUSED','reduction'),('LABELED',None)])
        self.assertEqual([x['reduction_error']for x in results],['ValueError: finite in-range numerical value required',None])
        self.assertEqual((results[0]['block_causes'],results[0]['completed_q'],results[0]['action_mean_q']),((None,None),None,None))
        self.assertEqual(results[1]['action_mean_q'],1);self.assertAlmostEqual(results[1]['completed_q'][1],.2,places=15)
        # A block that refused (turn2) keeps the root's cause even when another block's reduction raises; the error is still kept.
        blocks[0]=(None,plain);causes=[dict(cause='turn2')]+[dict(cause=None)]*3;blocks[1]=(wins,wins)
        with patch.object(cw.backups,'backup',side_effect=lambda frame,row,rule:blocks[row]),patch.object(cw.backups,'refusal_summary',return_value=dict(roots=causes)):
            results=cw.reduce(object(),cw.Batch((object(),)*4,None,roots,4,2))
        self.assertEqual([(x['status'],x['cause'],x['block_causes'])for x in results],[('REFUSED','turn2',('turn2',None)),('LABELED',None,(None,None))])
        self.assertEqual(results[0]['reduction_error'],'ValueError: finite in-range numerical value required')
        # Any other ValueError from the reduction is malformed input, not the rounding edge: it raises.
        from gpu_engine import d3_search_latency as latency
        with patch.object(cw.backups,'backup',side_effect=lambda frame,row,rule:blocks[row]),patch.object(cw.backups,'refusal_summary',return_value=dict(roots=causes)),\
                patch.object(latency,'reduce_block',side_effect=ValueError('both declared posterior worlds required')),self.assertRaisesRegex(ValueError,'both declared'):
            cw.reduce(object(),cw.Batch((object(),)*4,None,roots,4,2))

    def test_both_root_moves_are_reported_and_can_differ(self):
        from gpu_engine.d3_search_latency import compose
        legal=(True,True)+(False,)*7;logits=(3.,0.)+(0.,)*7
        composed=compose([NS(completed_q=(.1,.6)+(None,)*7)]*2,logits,legal,original)   # the reviewer's P7: logits favour 0, Q favours 1
        self.assertEqual(cw.root_actions(composed['completed_q'],composed['policy_target'],legal),(1,0))
        self.assertEqual(cw.root_actions((.5,.5,.9)+(0.,)*6,(.5,.5,.9)+(0.,)*6,legal),(0,0))   # ties: the lower action; illegal never chosen


if __name__=='__main__':unittest.main()
