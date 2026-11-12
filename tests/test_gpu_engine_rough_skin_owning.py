"""Prospective literal component fixture only; no original scientific roots."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import jax
import jax.numpy as j
import numpy as np

from pokezero.category_vocab import CategoryVocabulary
from pokezero.dex import ShowdownDex,load_showdown_dex_cached
from pokezero.randbat import Gen3RandbatSource,Gen3RandbatSpeciesUniverse,Gen3RandbatVariant,RandbatSourceMetadata
from gpu_engine import engine_rough_skin_callbacks as rs,engine_substitute_callbacks as sub
from gpu_engine import engine_direct_hit_controls as direct,engine_request_source as request
from gpu_engine import engine_public_events as public,engine_own_facts as own,engine_retained_encoder_driver as driver
from gpu_engine import engine_execution_completeness as ex,engine_kernel_frame_risk as risk
from gpu_engine import engine_full_v4_event_device as full,engine_lifecycle_callbacks as life,engine_leftovers_callbacks as leftovers
from gpu_engine import public_belief,v4_encoder,public_wire,source_provenance,transition
from _showdown_root import showdown_root
import test_gpu_engine_no_damage_owning as fixture
import gpu_engine_eager_witness as eager


class RoughSkinTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture.NoDamageOwningTests.setUpClass();cls.base=fixture.NoDamageOwningTests
        cls.metadata=deepcopy(cls.base.metadata);native=load_showdown_dex_cached(showdown_root())
        cls.metadata['vocabulary'].setdefault('natures',['_none'])
        if 'hardy' not in cls.metadata['vocabulary']['natures']:cls.metadata['vocabulary']['natures'].append('hardy')
        moves=dict(cls.base.dex.moves);species=dict(cls.base.dex.species)
        if 'roughskin' not in cls.metadata['vocabulary']['abilities']:cls.metadata['vocabulary']['abilities'].append('roughskin')
        cls.names=('firepunch','surf','tackle','protect')
        for name in cls.names:
            info=native.move_info(name);moves[name]=info
            if name not in cls.metadata['vocabulary']['moves']:
                cls.metadata['vocabulary']['moves'].append(name);cls.metadata['moves'].append(None)
            i=cls.metadata['vocabulary']['moves'].index(name)
            cls.metadata['moves'][i]=dict(id=name,base_power=info.base_power,type=cls.metadata['vocabulary']['types'].index(info.type.lower()),
                source_type=info.type,category=info.category,pp=info.pp,accuracy=True if info.accuracy is True else int(info.accuracy),priority=info.priority,flags=dict(protect=1,**({'contact':1}if name in ('firepunch','tackle')else{})) if name!='protect' else {})
        for name in ('aerodactyl','sharpedo'):
            if name not in cls.metadata['vocabulary']['species']:cls.metadata['vocabulary']['species'].append(name)
            species[name]=native.species_info(name)
        cls.dex=ShowdownDex(moves,species,native.type_chart)
        variants={name:Gen3RandbatSpeciesUniverse(name,80,(Gen3RandbatVariant('literal-sub-hit-'+name,'authored',name,'authored',80,
            cls.names,'Pressure' if name=='aerodactyl' else 'Rough Skin','Leftovers'),)) for name in cls.metadata['vocabulary']['species'][1:]}
        source=Gen3RandbatSource(RandbatSourceMetadata('gen3randombattle',3,None,None,None,'literal-sub-damage-source'),variants)
        labels=set(cls.base.catalog.vocabulary.tokens)
        for name in cls.names:labels|={'move:'+name,'belief:possible_move:'+name}
        for name in ('aerodactyl','sharpedo'):labels.add('species:'+name)
        labels|={'ability:roughskin','belief:possible_ability:roughskin'}
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
        cls.prepared=driver.prepare(cls.req,cls.facts,public_mode='retained-full',account_source_sites=True,
            leftovers_compiled=leftovers.compile_leftovers(cls.metadata,engine_catalog=cls.producer),leftovers_mode='retained',
            track_kernel_risks=True,frame_input_mode='source-witness',direct_hit_compiled=cls.helper,direct_hit_mode='kernel-event',
            fake_forward=lambda _:driver.FakeScores(j.zeros(9),j.array(0,j.int32)))
        cls.runtime=driver._checked(cls.prepared)

    def same(self,a,b):
        for x,y in zip(jax.tree.leaves(a),jax.tree.leaves(b),strict=True):np.testing.assert_array_equal(x,y)

    def arrays(self):
        arrays=self.base.f.base().arrays('pressure','pressure')
        for key,value in arrays.items():
            if isinstance(value,np.ndarray) and value.ndim>=2:value[:]=0
        for side,name in enumerate(('aerodactyl','sharpedo')):
            info=self.dex.species_info(name);base=np.array([info.base_stats[x]for x in ('hp','atk','def','spa','spd','spe')])
            stats=((2*base+31+85//4)*80)//100+5;stats[0]+=85
            arrays['species'][side,0]=self.metadata['vocabulary']['species'].index(name)
            arrays['stats'][side,0]=stats
            if 'hp' in arrays:arrays['hp'][side,0]=stats[0]
            if 'max_hp' in arrays:arrays['max_hp'][side,0]=stats[0]
            arrays['level'][side,0]=80;arrays['gender'][side,0]=0 if side==0 else 2
            arrays['ability'][side,0]=self.metadata['vocabulary']['abilities'].index('pressure'if side==0 else'roughskin')
            types=[self.metadata['vocabulary']['types'].index(t.lower())for t in info.types]
            arrays['types'][side,0]=types+[0]*(2-len(types))
        arrays['evs'][:,0]=85;arrays['ivs']=np.zeros((2,6,6),np.int32);arrays['ivs'][:,0]=31
        arrays['nature'][:,0]=self.metadata['vocabulary']['natures'].index('hardy')
        arrays['hidden_power_type'][:,0]=self.metadata['vocabulary']['types'].index('dark')
        arrays['hidden_power_power'][:,0]=70
        arrays['item'][:]=0
        arrays['moves'][:,0]=np.array([self.metadata['vocabulary']['moves'].index(n)for n in self.names],np.int32)
        arrays['pp'][:,0]=np.array([self.dex.move_info(n).max_pp for n in self.names],np.int32)
        return arrays

    def construct(self):
        arrays=self.arrays()
        with TemporaryDirectory()as directory:
            path=Path(directory)/'literal.npz';np.savez_compressed(path,**dict(arrays,metadata_utf8=np.frombuffer(json.dumps(self.metadata).encode(),np.uint8)))
            pool=source_provenance.load_constructor_pool(path)
            return driver.construct(pool,np.array([[0,1],[0,1]],np.int64),prepared=self.prepared,stream_ids=j.array([27101,27102],j.int32),seed=41)

    # The former test_original_transition_and_other_secondary_programs_unchanged pinned both files to b8092a85.
    # transition.py is certified against CURRENT_TRANSITION (full AST plus mutations) by terminal_closeout and seismic_toss_components; secondary.py: byte-exact against reviewed SOURCE_PINS in transition_observer_wiring's source-binding subtests.

    def test_legacy_physical_contact_damage_and_exclusions(self):
        from gpu_engine import secondary
        from gpu_engine.state import empty_state
        b=8;state=empty_state(b,seed=116)
        ability=self.metadata['vocabulary']['abilities'].index('roughskin')
        state=state._replace(hp=j.full_like(state.hp,201),max_hp=j.full_like(state.hp,201),
            ability=state.ability.at[:,1,0].set(ability),status=state.status.at[:,0,0].set(1))
        state=state._replace(hp=state.hp.at[1,0,0].set(12).at[2,0,0].set(0).at[3,1,0].set(0))
        move=j.full(b,self.metadata['vocabulary']['moves'].index('firepunch'),j.int32)
        move=move.at[5].set(self.metadata['vocabulary']['moves'].index('surf'))
        eligible=j.ones(b,bool).at[4].set(False)
        run=jax.jit(lambda s,r:secondary.apply_contact(s,j.zeros(b,j.int32),move,r,self.table.secondary,eligible=eligible))
        out=run(state,j.zeros(b,j.int32));other=run(state,j.full(b,59,j.int32))
        self.assertEqual(out.hp[:,0,0].tolist(),[189,0,0,189,201,201,189,189])
        self.same(out,other);np.testing.assert_array_equal(state.keys,out.keys)
        # Neither prior status, target KO nor RNG threshold changes the contact
        # damage. Caller eligibility carries miss/Substitute/Protect/zero hit.
        self.assertEqual(out.status[:,0,0].tolist(),[1]*b)
        data=self.helper.device.rough_skin;rs.shape(data)
        self.assertTrue(bool(data.candidates[self.metadata['vocabulary']['moves'].index('firepunch')]))
        self.assertFalse(bool(data.candidates[self.metadata['vocabulary']['moves'].index('surf')]))
        self.assertIsNone(direct.compile_direct_hit_controls(engine_catalog=self.producer).device.rough_skin)
        with self.assertRaises(ValueError):rs.compile_data(self.producer,self.metadata,request.SourceRules(self.rules.device),self.helper.device.substitute_damage)

    def test_b8_current_ticket_and_both_owner_full_templates(self):
        from gpu_engine import secondary
        from gpu_engine.state import empty_state
        b=8;rows=j.arange(b);actor=rows%2;foe=1-actor;z=j.zeros(b,j.int32);yes=j.ones(b,bool);no=j.zeros(b,bool)
        physical=z+3;target=z+5;streams=rows+29001
        cursor=life.register(streams,catalog=self.producer,expected_binding=self.producer.binding,
            expected_engine_binding=np.asarray(self.producer.engine_binding,np.uint32))
        active=j.zeros((b,2),j.int32).at[rows,actor].set(3).at[rows,foe].set(5)
        mapping=j.full((b,2,6),-1,j.int32).at[:,:,0].set(0).at[:,:,1].set(1).at[rows,actor,physical].set(2).at[rows,foe,target].set(2)
        cursor=cursor._replace(turn=z+1,initialized=yes,physical_active=active,physical_to_public=mapping,
            public_slot=j.full((b,2),2,j.int32),encounter=j.full((b,2),3,j.int32),next_slot=j.full((b,2),3,j.int32),
            revealed_species_id=j.ones((b,2),j.int32),established=j.ones((b,2),bool),generation=j.full((b,2),2,j.int32))
        account=self.runtime['execution_compiled'];context=direct.RegisteredContext(self.helper.device,account.device)
        tape=life._tape(cursor,self.producer,64)
        tape=ex.begin_on_tape(tape,ex.initialize(streams,compiled=account),account.device,active=yes)
        tape=tape._replace(execution=tape.execution._replace(secondary_status=context))
        move=z+self.metadata['vocabulary']['moves'].index('firepunch')
        tape=public.move(tape,actor,physical,target,move,eligible=yes)
        tape,ticket=ex.open_attempt(tape,ex.Site.DIRECT_HIT,foe,active=yes)
        write=direct.HitWrite(z+75,no,yes,no,z+1,z+1,no,no,no,z+100,z+80,z+100)
        tape=direct.record_registered_direct_hit_controls_device(tape,actor,physical,target,move,write,context,eligible=yes)
        tape=public.condition(tape,foe,target,z+80,z+100,z,eligible=yes)
        tape=ex.capture_direct_hit_write(tape,ticket,actor,physical,target,move,write,requested=yes,data=context.data)
        captured=risk.CapturedDirectHit(context,ticket,actor,physical,target,move,write)
        state=empty_state(b,seed=117)._replace(active=active)
        state=state._replace(hp=state.hp.at[rows,actor,physical].set(201).at[rows,foe,target].set(80),
            max_hp=state.max_hp.at[rows,actor,physical].set(201).at[rows,foe,target].set(100),
            ability=state.ability.at[rows,foe,target].set(self.metadata['vocabulary']['abilities'].index('roughskin')))
        _,out=secondary.apply_contact(state,actor,move,z,self.table.secondary,eligible=yes,_status_trace=tape,
            _rough_skin_context=context,_rough_skin_capture=captured)
        self.assertEqual(out.execution.ledger.refusal.tolist(),[0]*b)
        self.assertEqual(out.count.tolist(),[3]*b)
        self.assertEqual(rs.owned(out,2,cursor).tolist(),[True]*b)
        stale=out._replace(cursor=cursor._replace(generation=cursor.generation.at[0,0].add(1)))
        self.assertFalse(bool(rs.owned(stale,2,stale.cursor)[0]))
        forged=out._replace(words=out.words.at[1,2,2].set(4))
        self.assertFalse(bool(rs.owned(forged,2,cursor)[1]))
        old=tape._replace(refusal=tape.refusal|64)
        _,dirty=secondary.apply_contact(state,actor,move,z,self.table.secondary,eligible=yes,_status_trace=old,
            _rough_skin_context=context,_rough_skin_capture=captured)
        self.assertTrue(bool(j.all((dirty.refusal&64)!=0)))
        self.assertTrue(bool(j.all((dirty.execution.ledger.refusal&ex.UNKNOWN_SITE)!=0)))
        adapter=self.runtime['public'];kw=v4_encoder.wire_kwargs(self.catalog)
        species_ids=[self.catalog.species.ids[name]for name in ('aerodactyl','sharpedo')]
        for owner in (0,1):
            initial=full.register(streams,z+owner,compiled=adapter)
            species=initial.species.at[rows,actor,2].set(species_ids[0]).at[rows,foe,2].set(species_ids[1])
            visible_hp=j.where(owner==actor,201,100);visible_max=visible_hp
            initial=initial._replace(causal=initial.causal._replace(cursor=cursor),live=j.ones((b,2),bool),species=species,
                hp=initial.hp.at[rows,actor,2].set(visible_hp).at[rows,foe,2].set(100),
                maximum=initial.maximum.at[rows,actor,2].set(visible_max).at[rows,foe,2].set(100))
            projected=jax.jit(lambda a,t:full.project_device(a,t,life.snapshot(cursor),life.snapshot(cursor),adapter.device))(initial,out)
            self.assertEqual(projected.error.tolist(),[0]*b)
            for row in range(b):
                sides=[int(actor[row])+1,int(foe[row])+1];wire=public_wire.WireState()
                names=list(self.catalog.species.names[1:3])+['aerodactyl']
                lines=[]
                # Two genuine earlier distinct public encounters establish slot2.
                for name in [n for n in self.catalog.species.names[1:]if n.lower() not in ('aerodactyl','sharpedo')][:2]:
                    lines += [f'|switch|p{q}a: {name}|{name}, L80|100/100'for q in (1,2)]
                lines += [f'|switch|p{sides[0]}a: Aerodactyl|Aerodactyl, L80|100/100',
                    f'|switch|p{sides[1]}a: Sharpedo|Sharpedo, L80|100/100','|turn|1']
                wire,_=public_wire.project_lines(wire,lines,**kw)
                hp,maximum=(189,201)if owner==int(actor[row])else(95,100)
                wire=replace(wire,next_sequence=1)
                component=[f'|move|p{sides[0]}a: Aerodactyl|Fire Punch|p{sides[1]}a: Sharpedo',
                    f'|-damage|p{sides[1]}a: Sharpedo|80/100',
                    f'|-damage|p{sides[0]}a: Aerodactyl|{hp}/{maximum}|[from] ability: Rough Skin|[of] p{sides[1]}a: Sharpedo']
                _,parts=public_wire.project_lines(wire,component,**kw)
                expected=np.array([e.words()for p in parts for e in p.events])
                actual=np.asarray(projected.frames.words)[:,row][np.asarray(projected.frames.valid)[:,row]]
                np.testing.assert_array_equal(actual,expected,err_msg=f'owner{owner} row{row}')

    def test_actual_kernel_contact_write_both_full_and_legacy_physics(self):
        self.enterContext(eager.eager_source_witness(self.prepared))
        session=self.construct();self.assertEqual(session.state.core.admitted.tolist(),[True,True])
        before=session.state;frames=[];original=self.runtime['frame']
        def capture(*args):
            result=original(*args);frames.append(result);return result
        draws=j.zeros((2,2,12),j.int32).at[:,:,4].set(1)
        chance=transition.ChanceInputs(draws,j.zeros((2,2),j.int32),j.full((2,2),99,j.int32),
            j.zeros((2,2),j.int32),j.zeros((2,2),j.int32),j.zeros(2,j.int32))
        actions=j.array([[0,2],[0,2]],j.int32)
        self.runtime['frame']=capture
        try:result=driver.advance(session,actions,chance_inputs=chance)
        finally:self.runtime['frame']=original
        plain=transition.step_with_chance(before.core.engine,actions,self.table,chance)
        self.same(result.state.core.engine,plain.state)
        frame=frames[0];raw=np.asarray(frame.callbacks.words)
        diag=dict(core=result.state.core.admitted.tolist(),risk=result.state.core.risks.risk.tolist(),
            refusal=frame.callbacks.refusal.tolist(),account=result.state.core.execution.refusal.tolist(),
            witness=result.state.core.frame_input_report.refusal.tolist(),owner=[o.reason.tolist()for o in result.owners])
        self.assertEqual(diag['core'],[True,True],json.dumps(diag))
        records=[]
        for row in range(2):
            selected=raw[row,:,0]==rs.ROUGH_SKIN_HP;self.assertEqual(int(selected.sum()),1,json.dumps(diag))
            receipt=raw[row,selected][0];records.append(receipt)
            self.assertEqual(int(receipt[6]-receipt[7]),max(1,int(receipt[8])//16))
            ledger=result.state.core.execution
            slots=np.where(np.asarray(ledger.site)[row]==int(ex.Site.ROUGH_SKIN_HP))[0]
            self.assertEqual(len(slots),1);self.assertEqual(int(ledger.stage[row,slots[0]]),3)
        for observer in result.owners:
            self.assertEqual(observer.full_line_consumers_complete.tolist(),[True,True],json.dumps(diag))
            self.assertEqual(observer.encoded.observation.admitted.tolist(),[True,True],json.dumps(diag))
            self.assertFalse(bool(observer.B4_qualified.any()))
        report=result.state.core.frame_input_report
        np.savez_compressed('/tmp/pokezero-rough-skin-unit-capture-r2.npz',raw=raw,
            full_words=np.asarray(report.owner_full_words),valid=np.asarray(report.owner_full_valid),
            count=np.asarray(frame.callbacks.count),event_turn=np.asarray(frame.callbacks.event_turn),
            event_phase=np.asarray(frame.callbacks.event_phase),event_generation=np.asarray(frame.callbacks.event_generation))
        for owner in (0,1):
            for row,receipt in enumerate(records):
                # Literal source line at the actual write, before residuals.
                state=public_wire.WireState();kw=v4_encoder.wire_kwargs(self.catalog)
                lines=['|switch|p1a: Aerodactyl|Aerodactyl, L80, M|100/100',
                    '|switch|p2a: Sharpedo|Sharpedo, L80, F|100/100','|turn|1']
                state,_=public_wire.project_lines(state,lines,**kw)
                hp,maximum=(int(receipt[7]),int(receipt[8])) if owner==0 else ((100*int(receipt[7])+int(receipt[8])-1)//int(receipt[8]),100)
                if owner==1 and hp==100 and receipt[7]<receipt[8]:hp=99
                _,part=public_wire.project_line(state,f'|-damage|p1a: Aerodactyl|{hp}/{maximum}|[from] ability: Rough Skin|[of] p2a: Sharpedo',**kw)
                expected=np.array([e.words()for e in part.events]);expected[:,1]=0
                words=np.asarray(report.owner_full_words)[row,owner][np.asarray(report.owner_full_valid)[row,owner]].copy();words[:,1]=0
                self.assertTrue(any(np.array_equal(words[i:i+len(expected)],expected)for i in range(len(words)-len(expected)+1)),
                    f'owner{owner} row{row} missing literal original Rough Skin controls')

def tearDownModule():
    eager.clear()   # drop the consume executables this module compiled


if __name__=='__main__':unittest.main()
