"""Literal Sleep Powder source components, fake scores; no scientific roots."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import subprocess
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch
import jax
import jax.numpy as j
import numpy as np
from pokezero.category_vocab import CategoryVocabulary
from pokezero.dex import ShowdownDex
from pokezero.randbat import Gen3RandbatSource,Gen3RandbatSpeciesUniverse,Gen3RandbatVariant,RandbatSourceMetadata
from gpu_engine import engine_retained_encoder_driver as driver,engine_request_source as request,engine_own_facts as own
from gpu_engine import engine_public_events as public,engine_leftovers_callbacks as leftovers,engine_sleep_powder_callbacks as powder
from gpu_engine import engine_execution_completeness as accounting,engine_kernel_frame_risk as risks,engine_toxic_callbacks as primary
from gpu_engine import engine_lifecycle_callbacks as life,engine_status_callbacks as status
from gpu_engine import public_belief,source_provenance,transition,v4_encoder,mechanics as m
import test_gpu_engine_engine_weather_public as fixture
import gpu_engine_eager_witness as eager


class SleepPowderRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture.WeatherPublicTests.setUpClass();cls.base=fixture.WeatherPublicTests
        cls.metadata=deepcopy(cls.base.metadata)
        cls.move_id=cls.metadata['vocabulary']['moves'].index('sleeppowder')
        cls.metadata['moves'][cls.move_id]=dict(id='sleeppowder',base_power=0,type=cls.metadata['vocabulary']['types'].index('grass'),
            source_type='Grass',category='Status',pp=15,accuracy=75,priority=0,flags={'protect':1,'reflectable':1})
        moves=dict(cls.base.catalog.dex.moves)
        moves['sleeppowder']=replace(moves['surf'],id='sleeppowder',name='Sleep Powder',type='Grass',category='Status',gen3_category='Status',
            base_power=0,pp=15,target='normal',status='slp',accuracy=75.,priority=0,effect_label='slp',effect_chance=100,
            self_hp_cost=0.,recoil=False,drain=False,heal=False,boosts={})
        dex=ShowdownDex(moves,cls.base.catalog.dex.species,cls.base.catalog.dex.type_chart)
        cls.names=('sleeppowder','raindance','sunnyday','hail')
        variants={n:Gen3RandbatSpeciesUniverse(n,80,(Gen3RandbatVariant('slp-'+n,'authored',n,'authored',80,
            cls.names,'Pressure','Leftovers'),)) for n in cls.metadata['vocabulary']['species'][1:]}
        src=Gen3RandbatSource(RandbatSourceMetadata('gen3randombattle',3,None,None,None,'literal-sleep-powder-fixture'),variants)
        labels=set(cls.base.catalog.vocabulary.tokens)|{'move:sleeppowder','belief:possible_move:sleeppowder','move_effect:slp'}
        cls.catalog=v4_encoder.compile_encoder(public_belief.compile_source(src),dex,CategoryVocabulary(tuple(sorted(labels))),
            checkpoint_sha256='a'*64,feature_masks=cls.base.catalog.masks.__dict__)
        cls.table=transition.tables_from_metadata(cls.metadata)
        cls.producer=public.compile_catalog(cls.metadata,cls.table,public_moves=cls.catalog.moves,dex=dex,
            exposure=public.HealthExposure(public.SHOWDOWN,'gen3randombattle',True,False))
        original=cls.base.f.f;rules=request.load_source_rules(original.root,cls.metadata,node=original.node)
        cls.request=request.compile_request_source(cls.metadata,vocabulary=cls.catalog.vocabulary,dex=dex,
            checkpoint_sha256='a'*64,rules=rules,lifecycle_catalog=cls.producer)
        cls.facts=own.compile_own_facts(cls.request,cls.catalog)
        cls.helper=powder.compile_sleep_powder(engine_catalog=cls.producer,metadata=cls.metadata)
        cls.prepared=cls.prepare(enabled=True);cls.runtime=driver._checked(cls.prepared)

    @classmethod
    def prepare(cls,enabled):
        return driver.prepare(cls.request,cls.facts,public_mode='retained-full',
            leftovers_compiled=leftovers.compile_leftovers(cls.metadata,engine_catalog=cls.producer),leftovers_mode='retained',
            account_source_sites=True,track_kernel_risks=True,frame_input_mode='source-witness',
            sleep_powder_compiled=cls.helper if enabled else None,sleep_powder_mode='kernel-event' if enabled else 'closed',
            fake_forward=lambda _:driver.FakeScores(j.zeros(9),j.array(0,j.int32)))

    def same(self,a,b):
        for x,y in zip(jax.tree.leaves(a),jax.tree.leaves(b)):np.testing.assert_array_equal(x,y)

    def construct(self,*,prepared=None):
        arrays=self.base().arrays('pressure','pressure')
        arrays['moves'][:]=np.array([self.move_id]+[self.metadata['vocabulary']['moves'].index(n) for n in self.names[1:]],np.int32)
        arrays['pp'][:]=np.array([24,8,8,16],np.int32)
        with TemporaryDirectory() as tmp:
            p=Path(tmp)/'literal.npz';np.savez_compressed(p,**dict(arrays,metadata_utf8=np.frombuffer(json.dumps(self.metadata).encode(),np.uint8)))
            pool=source_provenance.load_constructor_pool(p)
            session=driver.construct(pool,np.array([[0,1],[0,1]],np.int64),prepared=prepared or self.prepared,
                stream_ids=j.array([9801,9802],j.int32),seed=31)
        self.assertEqual(session.state.core.admitted.tolist(),[True,True]);return session

    def test_source_defaults_and_strict_wire_scope(self):
        self.assertEqual(int(self.runtime['kernel_risk_compiled'].device.selected_risk[self.move_id]),0)
        plain=risks.compile_risks(self.producer,self.runtime['execution_compiled'])
        self.assertEqual(int(plain.device.selected_risk[self.move_id])&(risks.NON_WEATHER|risks.FIELD_VOLATILE),9)
        # transition.py is certified against CURRENT_TRANSITION (full AST plus mutations) by terminal_closeout and seismic_toss_components; engine_toxic_callbacks.py: byte-exact against reviewed SOURCE_PINS in transition_observer_wiring's source-binding subtests.
        self.assertEqual(Path('gpu_engine/public_sleep.py').read_bytes(),subprocess.check_output(['git','show','b8092a85:gpu_engine/public_sleep.py']))
        side=j.zeros(1,j.int32);zero=j.zeros(1,j.int32);yes=j.ones(1,bool);no=j.zeros(1,bool)
        attempt=powder.Attempt(yes,yes,no,no,no,yes,yes,no,no,no,yes,zero,j.full(1,m.SLP,j.int32),zero,j.full_like(zero,2))
        raw=primary.words_device(side,side,side,j.full(1,self.move_id,j.int32),attempt)
        self.assertEqual(powder.supported_raw(raw).tolist(),[True]);self.assertEqual(primary.supported_raw(raw).tolist(),[False])
        for field,value in [('applied',no),('accuracy_hit',no),('hit',no),('type_immune',yes),('ability_immune',yes),('substitute_block',yes),('protected',yes),('absorbed',yes),('damage_immune',yes),('old_status',j.full(1,m.SLP,j.int32)),('new_status',j.full(1,m.TOX,j.int32)),('old_counter',j.ones(1,j.int32))]:
            changed=attempt._replace(**{field:value});self.assertFalse(bool(powder.supported_raw(primary.words_device(side,side,side,j.full(1,self.move_id,j.int32),changed)).any()),field)
        traces=[]
        with patch.object(risks,'_captured_toxic_matches',return_value=yes),patch.object(risks,'_captured_sleep_powder_matches',return_value=yes),patch.object(risks,'_captured_thunder_wave_matches',return_value=yes),patch.object(risks,'_observe',side_effect=lambda *a,**k:traces.append(a) or a[0]):
            risks.observe_primary_status_device(NS(stream_id=zero),NS(count=zero),yes,object(),active=yes,captured=object(),captured_sleep_powder=object())
        self.assertEqual(len(traces),1)

    def test_current_source_span_b8_and_lum_scope_stay_closed(self):
        b=8;mask=j.array([True]*6+[False]*2);side=j.zeros(b,j.int32);physical=j.full(b,3,j.int32);target=j.full(b,5,j.int32);move=j.full(b,self.move_id,j.int32)
        cursor=life.register(j.arange(9901,9901+b,dtype=j.int32),catalog=self.producer,expected_binding=self.producer.binding,expected_engine_binding=np.asarray(self.producer.engine_binding,np.uint32))
        mapping=j.full((b,2,6),-1,j.int32).at[:,0,3].set(0).at[:,1,5].set(0)
        cursor=cursor._replace(generation=j.ones((b,2),j.int32),turn=j.ones(b,j.int32),initialized=j.ones(b,bool),physical_active=j.stack((physical,target),axis=1),
            public_slot=j.zeros((b,2),j.int32),encounter=j.ones((b,2),j.int32),established=j.ones((b,2),bool),physical_to_public=mapping,next_slot=j.ones((b,2),j.int32))
        tape=life._tape(cursor,self.producer,64);account=self.runtime['execution_compiled'];r=self.runtime['kernel_risk_compiled']
        tape=accounting.begin_on_tape(tape,accounting.initialize(cursor.stream_id,compiled=account),account.device,active=j.ones(b,bool))
        carry=risks.begin_frame_device(risks.initialize_device(tape.stream_id,r.device),tape,r.device,active=j.ones(b,bool))
        tape=tape._replace(execution=tape.execution._replace(risks=risks.Context(carry,r.device)))
        tape=public.move(tape,side,physical,target,move,eligible=mask);tape,ticket=accounting.open_attempt(tape,accounting.Site.PRIMARY_SLEEP,1-side,active=mask)
        no=j.zeros(b,bool);yes=j.ones(b,bool);zero=j.zeros(b,j.int32)
        attempt=powder.Attempt(yes,yes,no,no,no,yes,yes,no,no,no,yes,zero,j.full(b,m.SLP,j.int32),zero,j.full_like(zero,2))
        tape=status.changed(tape,1-side,target,attempt.old_status,attempt.new_status,status.PRIMARY,eligible=mask)
        context=self.runtime['sleep_powder_context']
        good=powder.record_device(tape,ticket,side,physical,target,move,attempt,context,requested=mask)
        self.assertEqual(good.count.tolist(),[3]*6+[0,0])
        self.assertEqual(powder.status_pair_device(good,j.array(1,j.int32),j.array(self.move_id,j.int32)).tolist(),mask.tolist())
        self.assertEqual(powder.span_matches(good,ticket,side,physical,target,move,attempt,context).tolist(),mask.tolist())
        capture=lambda t,c=context:risks.primary_status_on_tape(t,mask,active=mask,
            captured_sleep_powder=risks.CapturedSleepPowder(c,ticket,side,physical,target,move,attempt))
        self.assertEqual(capture(good).execution.risks.carry.risk.tolist(),[0]*8)
        for label,bad in (
            ('cursor_generation',good._replace(cursor=good.cursor._replace(generation=good.cursor.generation.at[0,1].add(1)))),
            ('prefix_turn',good._replace(event_turn=good.event_turn.at[0,0].add(1))),
            ('status_target',good._replace(words=good.words.at[0,1,4].set(4))),
            ('counter',good._replace(words=good.words.at[0,2,9].set(16))),
            ('phase',good._replace(cursor=good.cursor._replace(phase=good.cursor.phase.at[0].set(1)))),
            ('claim',good._replace(execution=good.execution._replace(ledger=good.execution.ledger._replace(claimed=good.execution.ledger.claimed.at[0,2].set(0)))))):
            with self.subTest(label=label):
                self.assertTrue(int(capture(bad).execution.risks.carry.risk[0])&risks.FIELD_VOLATILE)
                self.assertEqual(capture(bad).execution.risks.carry.risk[1:].tolist(),[0]*7)
        foreign=context._replace(data=context.data._replace(binding=context.data.binding.at[0].add(j.uint32(1))))
        self.assertTrue(bool(((capture(good,foreign).execution.risks.carry.risk[:6]&risks.FIELD_VOLATILE)!=0).all()))
        prior=good._replace(refusal=good.refusal.at[0].set(65),execution=good.execution._replace(risks=good.execution.risks._replace(carry=good.execution.risks.carry._replace(risk=good.execution.risks.carry.risk.at[0].set(64)))))
        self.assertTrue(int(capture(prior).execution.risks.carry.risk[0])&64);self.assertEqual(int(capture(prior).refusal[0])&65,65)
        for before,after in zip(jax.tree.leaves(tape),jax.tree.leaves(good),strict=True):
            if getattr(before,'ndim',0) and before.shape[0]==8:np.testing.assert_array_equal(before[6:],after[6:])
        for name in ('engine_binding','source_binding','move_binding','producer_binding'):np.testing.assert_array_equal(getattr(tape,name),getattr(good,name))
        lum=NS(lum=j.array([False,True]));items=j.array([1,1,0,1,1,1,1,1],j.int32)
        applied=yes.at[3].set(False)
        excluded=powder.exclude_cure_device(good,mask,applied,items,lum)
        self.assertEqual((excluded.refusal&status.UNKNOWN_STATUS).tolist(),[128,128,0,0,128,128,0,0])
        self.assertEqual(excluded.count.tolist(),good.count.tolist())

    def test_all_tape_global_namespaces_require_exact_words_before_comparison(self):
        cursor=life.register(j.arange(11901,11909,dtype=j.int32),catalog=self.producer,
            expected_binding=self.producer.binding,expected_engine_binding=np.asarray(self.producer.engine_binding,np.uint32))
        tape=life._tape(cursor,self.producer,64);context=self.runtime['sleep_powder_context']
        for name in ('engine_binding','source_binding','move_binding','producer_binding'):
            original=getattr(tape,name)
            for bad in (original[None,:],j.broadcast_to(original,(8,8)),original.astype(j.int32)):
                with self.subTest(name=name,shape=bad.shape,dtype=str(bad.dtype)):
                    with self.assertRaises(ValueError):powder._bound(tape._replace(**{name:bad}),context)

    def test_exact_original_sleep_lexical_template_and_host_capability(self):
        from gpu_engine import public_wire,public_inputs,engine_full_v4_event_device as full
        catalog=self.catalog;name=catalog.species.names[1];ident=('p1a: '+name,'p2a: '+name)
        kwargs=v4_encoder.wire_kwargs(catalog)
        prior,_=public_wire.project_lines(public_wire.WireState(),[
            f'|switch|{ident[0]}|{name}, L80, M|100/100',f'|switch|{ident[1]}|{name}, L80, M|100/100','|turn|1'],**kwargs)
        _,projection=public_wire.project_line(prior,f'|-status|{ident[0]}|slp|[from] move: Sleep Powder',**kwargs)
        expected=np.array([e.words()for e in projection.events],np.int32)
        tables=self.runtime['public'].device;key=int(tables.sleep_powder_status)
        self.assertGreater(key,0);self.assertEqual(int(tables.length[key]),len(expected))
        np.testing.assert_array_equal(np.asarray(tables.templates[key])[:len(expected)],expected)
        # The whole original FULL writer owns the source attributes; the
        # private duration continuation contributes no public word itself.
        self.assertTrue(any(int(e.words()[0])==int(public_inputs.Kind.STATUS)for e in projection.events))
        with self.assertRaises(ValueError):powder.check_compiled(powder.Compiled(self.helper.device))
        with self.assertRaises(ValueError):jax.jit(lambda:powder.check_compiled(self.helper).binding)()
        z=j.zeros(1,j.int32);yes=j.ones(1,bool);no=j.zeros(1,bool)
        attempt=powder.Attempt(yes,yes,no,no,no,yes,yes,no,no,no,yes,z,j.full(1,m.SLP,j.int32),z,j.full(1,2,j.int32))
        for counter in range(7):
            raw=primary.words_device(z,z,z,j.full(1,self.move_id,j.int32),attempt._replace(new_counter=j.full(1,counter,j.int32)))
            self.assertEqual(bool(powder.supported_raw(raw)[0]),2<=counter<=5)
        import os,shutil
        root=Path(os.environ['POKEZERO_SHOWDOWN_ROOT'])
        self.assertEqual(subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip(),public.SHOWDOWN)
        script=r"""
const root=process.argv[1];const {Dex}=require(root+'/dist/sim/dex.js');const dex=Dex.mod('gen3');const move=dex.moves.get('sleeppowder'),effect=dex.conditions.get('slp');const lines=[],ranges=[],effectState={};
effect.onStart.call({effectState,add:(...xs)=>lines.push(xs.map(String).join('|')),random:(lo,hi)=>{ranges.push([lo,hi]);return 3}},
 {toString:()=> 'p2a: Target',removeVolatile:()=>false},{toString:()=> 'p1a: Actor'},move);
process.stdout.write(JSON.stringify({lines,ranges,effectState}));
"""
        result=json.loads(subprocess.check_output([shutil.which('node'),'-e',script,str(root)],text=True))
        self.assertEqual(result,dict(lines=['-status|p2a: Target|slp|[from] move: Sleep Powder'],ranges=[[2,6]],effectState=dict(time=3,skippedTime=0)))

    def test_actual_success_both_full_source_drains_and_single_primary_count(self):
        self.enterContext(eager.eager_source_witness(self.prepared))
        session=self.construct();before=session.state;frames=[];actual=self.runtime['frame']
        def capture(*args):value=actual(*args);frames.append(value);return value
        draws=j.zeros((2,2,12),j.int32).at[:,:,4].set(1)
        chance=transition.ChanceInputs(draws,*[j.zeros((2,2),j.int32) for _ in range(4)],j.zeros(2,j.int32))
        acts=j.array([[0,7],[0,7]],j.int32);self.runtime['frame']=capture
        try:result=driver.advance(session,acts,chance_inputs=chance)
        finally:self.runtime['frame']=actual
        self.same(result.state.core.engine,transition.step_with_chance(before.core.engine,acts,self.table,chance).state)
        frame=frames[0];diagnostic=dict(risk=result.state.core.risks.risk.tolist(),refusal=frame.callbacks.refusal.tolist(),
            accounting=result.state.core.execution.refusal.tolist(),witness=result.state.core.frame_input_report.refusal.tolist(),owners=[o.reason.tolist() for o in result.owners])
        self.assertEqual(result.state.core.admitted.tolist(),[True,True],json.dumps(diagnostic))
        raw=frame.callbacks.words;present=j.arange(raw.shape[1])[None,:]<frame.callbacks.count[:,None]
        for kind in (9,26):self.assertEqual(j.sum(present&(raw[:,:,0]==kind),axis=1).tolist(),[1,1])
        at=j.argmax(present&(raw[:,:,0]==26),axis=1);packet=raw[j.arange(2),at]
        self.assertEqual(packet[:,7:10].tolist(),[[0,m.SLP,32]]*2);self.assertEqual(packet[:,4].tolist(),[4,4])
        ledger=result.state.core.execution;selected=(ledger.site==int(accounting.Site.PRIMARY_SLEEP))&(ledger.serial>0)
        self.assertEqual(selected.sum(axis=1).tolist(),[1,1]);self.assertTrue(bool(j.all(j.where(selected,ledger.stage==3,True))))
        self.assertEqual(result.state.core.risks.counts[:,risks.Branch.PRIMARY_STATUS-1].tolist(),[2,2])
        self.assertEqual(result.state.core.risks.risk.tolist(),[0,0])
        for owner in result.owners:
            self.assertEqual(owner.full_line_consumers_complete.tolist(),[True,True]);self.assertEqual(owner.encoded.observation.admitted.tolist(),[True,True]);self.assertFalse(bool(owner.B4_qualified.any()))


def tearDownModule():
    eager.clear()   # drop the consume executables this module compiled
