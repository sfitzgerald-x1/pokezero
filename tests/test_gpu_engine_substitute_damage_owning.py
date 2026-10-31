"""Prospective literal component fixture only; no original scientific roots."""
import ast
from copy import deepcopy
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import unittest

import jax
import jax.numpy as j
import numpy as np

from pokezero.category_vocab import CategoryVocabulary
from pokezero.dex import ShowdownDex,load_showdown_dex_cached
from pokezero.randbat import Gen3RandbatSource,Gen3RandbatSpeciesUniverse,Gen3RandbatVariant,RandbatSourceMetadata
from gpu_engine import engine_substitute_damage_callbacks as sd,engine_substitute_callbacks as sub
from gpu_engine import engine_direct_hit_controls as direct,engine_request_source as request
from gpu_engine import engine_public_events as public,engine_own_facts as own,engine_retained_encoder_driver as driver
from gpu_engine import engine_execution_completeness as ex,engine_kernel_frame_risk as risk
from gpu_engine import engine_full_v4_event_device as full,engine_lifecycle_callbacks as life,engine_leftovers_callbacks as leftovers
from gpu_engine import public_belief,v4_encoder,public_wire,source_provenance,transition
from gpu_engine_default_observer_ast import dump_default_program,assert_hp_rng_mutations_detected
from _showdown_root import showdown_root
import test_gpu_engine_no_damage_owning as fixture


class SubstituteDamageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture.NoDamageOwningTests.setUpClass();cls.base=fixture.NoDamageOwningTests
        cls.metadata=deepcopy(cls.base.metadata);native=load_showdown_dex_cached(showdown_root())
        cls.metadata['vocabulary'].setdefault('natures',['_none'])
        if 'hardy' not in cls.metadata['vocabulary']['natures']:cls.metadata['vocabulary']['natures'].append('hardy')
        moves=dict(cls.base.dex.moves);species=dict(cls.base.dex.species)
        cls.names=('substitute','surf','tackle','protect')
        for name in cls.names:
            info=native.move_info(name);moves[name]=info
            if name not in cls.metadata['vocabulary']['moves']:
                cls.metadata['vocabulary']['moves'].append(name);cls.metadata['moves'].append(None)
            i=cls.metadata['vocabulary']['moves'].index(name)
            cls.metadata['moves'][i]=dict(id=name,base_power=info.base_power,type=cls.metadata['vocabulary']['types'].index(info.type.lower()),
                source_type=info.type,category=info.category,pp=info.pp,accuracy=True if info.accuracy is True else int(info.accuracy),priority=info.priority,flags={'protect':1} if name in ('surf','tackle') else {})
        for name in ('aerodactyl','mew'):
            if name not in cls.metadata['vocabulary']['species']:cls.metadata['vocabulary']['species'].append(name)
            species[name]=native.species_info(name)
        cls.dex=ShowdownDex(moves,species,native.type_chart)
        variants={name:Gen3RandbatSpeciesUniverse(name,80,(Gen3RandbatVariant('literal-sub-hit-'+name,'authored',name,'authored',80,
            cls.names,'Pressure' if name=='aerodactyl' else 'Synchronize','Leftovers'),)) for name in cls.metadata['vocabulary']['species'][1:]}
        source=Gen3RandbatSource(RandbatSourceMetadata('gen3randombattle',3,None,None,None,'literal-sub-damage-source'),variants)
        labels=set(cls.base.catalog.vocabulary.tokens)
        for name in cls.names:labels|={'move:'+name,'belief:possible_move:'+name}
        for name in ('aerodactyl','mew'):labels.add('species:'+name)
        labels|={'ability:synchronize','belief:possible_ability:synchronize','move_effect:substitute'}
        cls.catalog=v4_encoder.compile_encoder(public_belief.compile_source(source),cls.dex,CategoryVocabulary(tuple(sorted(labels))),
            checkpoint_sha256='a'*64,feature_masks=cls.base.catalog.masks.__dict__)
        cls.table=transition.tables_from_metadata(cls.metadata)
        cls.producer=public.compile_catalog(cls.metadata,cls.table,public_moves=cls.catalog.moves,dex=cls.dex,
            exposure=public.HealthExposure(public.SHOWDOWN,'gen3randombattle',True,False))
        cls.rules=request.load_source_rules(showdown_root(),cls.metadata)
        cls.req=request.compile_request_source(cls.metadata,vocabulary=cls.catalog.vocabulary,dex=cls.dex,
            checkpoint_sha256='a'*64,rules=cls.rules,lifecycle_catalog=cls.producer)
        cls.facts=own.compile_own_facts(cls.req,cls.catalog)
        cls.helper=direct.compile_direct_hit_controls(engine_catalog=cls.producer,metadata=cls.metadata,source_rules=cls.rules)
        cls.sub=sub.compile_substitute(engine_catalog=cls.producer,metadata=cls.metadata)
        cls.prepared=driver.prepare(cls.req,cls.facts,public_mode='retained-full',account_source_sites=True,
            leftovers_compiled=leftovers.compile_leftovers(cls.metadata,engine_catalog=cls.producer),leftovers_mode='retained',
            track_kernel_risks=True,frame_input_mode='source-witness',direct_hit_compiled=cls.helper,direct_hit_mode='kernel-event',
            substitute_compiled=cls.sub,substitute_mode='kernel-event',fake_forward=lambda _:driver.FakeScores(j.zeros(9),j.array(0,j.int32)))
        cls.runtime=driver._checked(cls.prepared)

    def same(self,a,b):
        for x,y in zip(jax.tree.leaves(a),jax.tree.leaves(b),strict=True):np.testing.assert_array_equal(x,y)

    def arrays(self):
        arrays=self.base.f.base().arrays('pressure','pressure')
        for key,value in arrays.items():
            if isinstance(value,np.ndarray) and value.ndim>=2:value[:]=0
        for side,name in enumerate(('aerodactyl','mew')):
            info=self.dex.species_info(name);base=np.array([info.base_stats[x]for x in ('hp','atk','def','spa','spd','spe')])
            stats=((2*base+31+85//4)*80)//100+5;stats[0]+=85
            arrays['species'][side,0]=self.metadata['vocabulary']['species'].index(name)
            arrays['stats'][side,0]=stats
            if 'hp' in arrays:arrays['hp'][side,0]=stats[0]
            if 'max_hp' in arrays:arrays['max_hp'][side,0]=stats[0]
            arrays['level'][side,0]=80;arrays['gender'][side,0]=0 if side==0 else 2
            arrays['ability'][side,0]=self.metadata['vocabulary']['abilities'].index('pressure'if side==0 else'synchronize')
            types=[self.metadata['vocabulary']['types'].index(t.lower())for t in info.types]
            arrays['types'][side,0]=types+[0]*(2-len(types))
        arrays['evs'][:,0]=85;arrays['ivs']=np.zeros((2,6,6),np.int32);arrays['ivs'][:,0]=31
        arrays['nature'][:,0]=self.metadata['vocabulary']['natures'].index('hardy')
        arrays['hidden_power_type'][:,0]=self.metadata['vocabulary']['types'].index('dark')
        arrays['hidden_power_power'][:,0]=70
        arrays['item'][:]=0
        arrays['moves'][:,0]=np.array([self.metadata['vocabulary']['moves'].index(n)for n in self.names],np.int32)
        arrays['pp'][:,0]=np.array([16,24,56,16],np.int32)
        return arrays

    def construct(self):
        arrays=self.arrays()
        with TemporaryDirectory()as directory:
            path=Path(directory)/'literal.npz';np.savez_compressed(path,**dict(arrays,metadata_utf8=np.frombuffer(json.dumps(self.metadata).encode(),np.uint8)))
            pool=source_provenance.load_constructor_pool(path)
            return driver.construct(pool,np.array([[0,1],[0,1]],np.int64),prepared=self.prepared,stream_ids=j.array([27101,27102],j.int32),seed=41)

    def test_source_mask_defaults_and_physical_program(self):
        data=self.helper.device.substitute_damage;sd.shape(data)
        for name in ('surf','thunderbolt','flamethrower','tackle'):
            self.assertTrue(bool(data.candidates[self.metadata['vocabulary']['moves'].index(name)]),name)
        for name in ('substitute','protect'):
            self.assertFalse(bool(data.candidates[self.metadata['vocabulary']['moves'].index(name)]),name)
        self.assertIsNone(direct.compile_direct_hit_controls(engine_catalog=self.producer).device.substitute_damage)
        with self.assertRaises(ValueError):sd.compile_data(self.producer,self.metadata,request.SourceRules(self.rules.device))
        before=ast.parse(subprocess.check_output(['git','show','0d4b8830:gpu_engine/transition.py'],text=True))
        after=ast.parse(Path('gpu_engine/transition.py').read_text());baseline=dump_default_program(before)
        self.assertEqual(dump_default_program(after),baseline);assert_hp_rng_mutations_detected(self,after,baseline)

    def test_b8_templates_use_actual_nonzero_target_encounter(self):
        b=8;rows=j.arange(b);actor=rows%2;target=1-actor;physical=j.full(b,3,j.int32);victim=j.full(b,5,j.int32)
        streams=j.arange(28001,28001+b,dtype=j.int32)
        cursor=life.register(streams,catalog=self.producer,expected_binding=self.producer.binding,
            expected_engine_binding=np.asarray(self.producer.engine_binding,np.uint32))
        active=j.zeros((b,2),j.int32).at[rows,actor].set(3).at[rows,target].set(5)
        mapping=j.full((b,2,6),-1,j.int32).at[:,:,0].set(0).at[:,:,1].set(1).at[rows,actor,physical].set(2).at[rows,target,victim].set(2)
        cursor=cursor._replace(turn=j.ones(b,j.int32),initialized=j.ones(b,bool),physical_active=active,
            physical_to_public=mapping,public_slot=j.full((b,2),2,j.int32),encounter=j.full((b,2),3,j.int32),
            next_slot=j.full((b,2),3,j.int32),revealed_species_id=j.ones((b,2),j.int32),established=j.ones((b,2),bool),generation=j.full((b,2),2,j.int32))
        tape=life._tape(cursor,self.producer,64);account=self.runtime['execution_compiled']
        tape=ex.begin_on_tape(tape,ex.initialize(streams,compiled=account),account.device,active=j.ones(b,bool))
        move=j.full(b,self.metadata['vocabulary']['moves'].index('tackle'),j.int32)
        tape=public.move(tape,actor,physical,victim,move,eligible=j.ones(b,bool))
        tape,ticket=ex.open_attempt(tape,ex.Site.SUBSTITUTE_DAMAGE,target,active=j.ones(b,bool))
        context=direct.RegisteredContext(self.helper.device,account.device)
        pending=sd.Pending(context,ticket,actor,physical,victim,move,j.ones(b,bool))
        broken=rows%2==1;old=j.full(b,60,j.int32);new=j.where(broken,0,40);dealt=old-new
        effect=(rows%3)-1;num=j.where(effect==1,2,1);den=j.where(effect==-1,2,1)
        critical=rows%4==3
        write=sd.Write(j.full(b,35,j.int32),j.ones(b,bool),*[j.zeros(b,bool)for _ in range(5)],critical,
            num,den,old,new,dealt,j.where(broken,100,20),j.full(b,100,j.int32),j.full(b,100,j.int32))
        unwritten=tape
        tape=sd.record(tape,pending,write)
        self.assertEqual(sd.span_matches(tape,pending,write).tolist(),[True]*b)
        stale=tape._replace(cursor=tape.cursor._replace(generation=tape.cursor.generation.at[0,0].add(1)))
        self.assertEqual(sd.span_matches(stale,pending,write).tolist(),[False]+[True]*7)
        wrong_move_stamp=tape._replace(event_turn=tape.event_turn.at[1,0].add(1))
        self.assertFalse(bool(sd.span_matches(wrong_move_stamp,pending,write)[1]))
        wrong_target=tape._replace(words=tape.words.at[2,1,4].set(4))
        self.assertFalse(bool(sd.span_matches(wrong_target,pending,write)[2]))
        dirty=unwritten._replace(refusal=unwritten.refusal|64,execution=unwritten.execution._replace(
            ledger=unwritten.execution.ledger._replace(refusal=unwritten.execution.ledger.refusal|ex.UNKNOWN_SITE)))
        retained=sd.record(dirty,pending,write)
        self.assertTrue(bool(j.all((retained.refusal&64)!=0)))
        self.assertTrue(bool(j.all((retained.execution.ledger.refusal&ex.UNKNOWN_SITE)!=0)))
        self.assertEqual(tape.execution.ledger.refusal.tolist(),[0]*b)
        adapter=self.runtime['public'];state=full.register(streams,j.zeros(b,j.int32),compiled=adapter)
        causal=state.causal._replace(cursor=cursor)
        species=state.species.at[:,:,2].set(3)
        state=state._replace(causal=causal,live=j.ones((b,2),bool),species=species)
        projected=jax.jit(lambda s,t:full.project_device(s,t,life.snapshot(cursor),life.snapshot(cursor),adapter.device))(state,tape)
        self.assertEqual(projected.error.tolist(),[0]*b)
        names=self.catalog.species.names[1:4];self.assertEqual(len(names),3)
        wire=public_wire.WireState();kw=v4_encoder.wire_kwargs(self.catalog)
        lines=[f'|switch|p{side+1}a: {name}|{name}, L80, M|100/100'for name in names for side in (0,1)]
        wire,_=public_wire.project_lines(wire,lines+['|turn|1'],**kw)
        for row in range(b):
            ident=f'p{int(target[row])+1}a: {names[-1]}'
            lines=[]
            if int(effect[row]):lines.append(f'|-{"supereffective"if int(effect[row])>0 else"resisted"}|{ident}')
            if bool(critical[row]):lines.append(f'|-crit|{ident}')
            lines.append(f'|-end|{ident}|Substitute'if bool(broken[row])else f'|-activate|{ident}|move: Substitute|[damage]')
            _,parts=public_wire.project_lines(wire,lines,**kw)
            expected=np.array([e.words()for p in parts for e in p.events]);expected[:,1]=0
            actual=np.asarray(projected.frames.words)[:,row][np.asarray(projected.frames.valid)[:,row]][-len(expected):].copy();actual[:,1]=0
            np.testing.assert_array_equal(actual,expected,err_msg=f'row{row}')

    def test_actual_create_then_hit_survive_and_break_both_consumers(self):
        session=self.construct();self.assertEqual(session.state.core.admitted.tolist(),[True,True])
        for observer in session.result.owners:self.assertEqual(observer.encoded.observation.admitted.tolist(),[True,True],str(observer.reason.tolist()))
        before=session.state;frames=[];actual=self.runtime['frame']
        def capture(*args):
            frame=actual(*args);frames.append(frame);return frame
        chance=transition.ChanceInputs(j.zeros((2,2,12),j.int32).at[:,:,4].set(1),
            *[j.zeros((2,2),j.int32)for _ in range(4)],j.zeros(2,j.int32))
        actions=j.array([[0,2],[0,1]],j.int32)
        self.runtime['frame']=capture
        try:result=driver.advance(session,actions,chance_inputs=chance)
        finally:self.runtime['frame']=actual
        self.assertEqual(len(frames),1);frame=frames[0]
        plain=transition.step_with_chance(before.core.engine,actions,self.table,chance)
        self.same(result.state.core.engine,plain.state)
        diagnostic=dict(core=result.state.core.admitted.tolist(),risk=result.state.core.risks.risk.tolist(),
            refusal=frame.callbacks.refusal.tolist(),account=result.state.core.execution.refusal.tolist(),
            witness=result.state.core.frame_input_report.refusal.tolist(),owner=[o.reason.tolist()for o in result.owners])
        self.assertEqual(diagnostic['core'],[True,True],json.dumps(diagnostic))
        self.assertEqual(diagnostic['risk'],[0,0],json.dumps(diagnostic))
        raw=np.asarray(frame.callbacks.words);records=[]
        for row in range(2):
            selected=raw[row,:,0]==sd.SUBSTITUTE_DAMAGE;self.assertEqual(int(selected.sum()),1)
            records.append(raw[row,selected][0])
        self.assertGreater(int(records[0][7]),0);self.assertEqual(int(records[1][7]),0)
        self.assertEqual([int(x[9])for x in records],[-1,1])
        ledger=result.state.core.execution;at=(ledger.site==int(ex.Site.SUBSTITUTE_DAMAGE))&(ledger.serial>0)
        self.assertEqual(at.sum(axis=1).tolist(),[1,1]);self.assertTrue(bool(j.all(j.where(at,ledger.stage==3,True))))
        for observer in result.owners:
            self.assertEqual(observer.full_line_consumers_complete.tolist(),[True,True])
            self.assertEqual(observer.encoded.observation.admitted.tolist(),[True,True],str(observer.reason.tolist()))
            self.assertFalse(bool(observer.B4_qualified.any()))


if __name__=='__main__':unittest.main()
