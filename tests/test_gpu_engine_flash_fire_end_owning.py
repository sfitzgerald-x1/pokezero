"""Prospective literal source components, original FULL writer; no models/games."""
import ast
from dataclasses import replace
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import unittest
import jax
import jax.numpy as j
import numpy as np
from gpu_engine import engine_flash_fire_end_callbacks as cb,engine_execution_completeness as ex
from gpu_engine import engine_lifecycle_callbacks as life,engine_direct_hit_controls as hit
from gpu_engine import engine_retained_encoder_driver as driver,engine_source_frame_witness as witness
from gpu_engine import entry,transition,source_provenance,public_wire,v4_encoder,absorption
from test_gpu_engine_gameplay_stream_comparison import state as literal_state
from _showdown_root import showdown_root
import test_gpu_engine_no_damage_owning as fixtures
import gpu_engine_eager_witness as eager


class FlashFireEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.NoDamageOwningTests.setUpClass();cls.f=fixtures.NoDamageOwningTests

    def test_original_callback_full_template_and_default_physics(self):
        script="""const d=require(process.argv[1]+'/dist/sim/dex.js').Dex.mod('gen3');
const a=d.abilities.get('flashfire'),lines=[],target={toString:()=> 'p1a: Actor',
removeVolatile(id){if(id!=='flashfire')throw Error(id);a.condition.onEnd.call({add:(...v)=>lines.push(v.map(String).join('|'))},this)}};
a.onEnd.call({},target);process.stdout.write(JSON.stringify(lines));"""
        lines=json.loads(subprocess.check_output(['node','-e',script,str(showdown_root())],text=True))
        self.assertEqual(lines,['-end|p1a: Actor|ability: Flash Fire|[silent]'])
        # transition.py is certified against CURRENT_TRANSITION (full AST plus mutations) by terminal_closeout and seismic_toss_components; entry.py: byte-exact against reviewed SOURCE_PINS in transition_observer_wiring's source-binding subtests.
        tables=self.f.runtime['public'].device;key=int(tables.flash_fire_end)
        self.assertGreater(key,0);self.assertEqual(int(tables.length[key]),7)

    def test_actual_end_formatter_uses_current_public_encounter_and_keeps_neutral_controls(self):
        import gpu_engine.engine_full_v4_event_device as full
        tree=ast.parse(Path(full.__file__).read_text())
        assign=next(n for n in ast.walk(tree)if isinstance(n,ast.Assign)
            and any(isinstance(t,ast.Name)and t.id=='toxic_enc'for t in n.targets))
        expression=compile(ast.Expression(assign.value),'<actual FULL target>','eval')
        zero=j.zeros(8,bool);slots=j.arange(8,dtype=j.int32)%6
        semantic=j.zeros((8,16),j.int32).at[:,8].set(5)
        values=dict(jnp=j,direct_control=zero,toxic_attempt=zero,no_damage_header=zero,
            sub_damage=zero,protect_block=zero,color_change=zero,pinch=zero,self_move=zero,
            self_heal=zero,heal_bell=zero,focus=zero,new_slot=slots,semantic=semantic)
        np.testing.assert_array_equal(eval(expression,{},dict(values,flash_ended=j.ones(8,bool))),slots+1)
        np.testing.assert_array_equal(eval(expression,{},dict(values,flash_ended=zero)),semantic[:,8])

    def component(self):
        f=self.f;b=8;rows=j.arange(b);yes=j.ones(b,bool);side=j.zeros(b,j.int32)
        state=jax.tree.map(j.asarray,literal_state(b));abi=f.metadata['vocabulary']['abilities'].index('flashfire')
        ability=state.ability.at[:,0,0].set(abi).at[6,0,0].set(f.metadata['vocabulary']['abilities'].index('pressure'))
        volatile=state.volatile.at[:,0,0].set(j.array([2,2,0,3,2,2,2,2],j.uint32))
        state=state._replace(ability=ability,base_ability=ability,volatile=volatile,hp=state.hp.at[5,0,0].set(0))
        cursor=life.register(rows+29001,**f.kw);tape=life._tape(cursor,f.producer,64)
        for s in (0,1):
            tape=life._reveal(tape,state,j.full(b,s,j.int32),j.full(b,-1,j.int32),side,
                eligible=yes,replacement=~yes,baton=~yes,initial=True)
        tape=tape._replace(cursor=tape.cursor._replace(initialized=yes));tape=life._turn(tape,yes)
        tape=life._tape(tape.cursor,f.producer,64)
        account=f.runtime['execution_compiled'];tape=ex.begin_on_tape(tape,ex.initialize(tape.stream_id,compiled=account),account.device,active=yes)
        tape=tape._replace(refusal=j.array([257,1025,1,1,1,1,1,1],j.int32))
        eligible=j.array([True,False,True,True,True,True,True,True]);baton=j.array([False]*4+[True]+[False]*3)
        tape,pending=cb.begin(tape,state,side,eligible=eligible,baton=baton,context=f.runtime['direct_hit_context'])
        state=entry.depart(state,side,f.table.entry,eligible=eligible);tape=cb.finish(tape,pending,state)
        state=state._replace(active=state.active.at[:,0].set(j.where(eligible,1,0)))
        tape=life.installed(tape,state,side,side,j.ones(b,j.int32),eligible=eligible,replacement=~yes,baton=~yes)
        return tape

    def test_mixed_exact_clear_and_sticky_unsupported_namespace_stamps(self):
        tape=self.component()
        self.assertEqual(cb.owned(tape,0,self.f.runtime['public'].device.causal.no_damage).tolist(),[True,False,False,False,False,False,False,True])
        self.assertEqual(tape.execution.ledger.count.tolist(),[1,0,0,1,1,1,1,1])
        self.assertEqual(int(tape.refusal[0]),257);self.assertEqual(int(tape.refusal[1]),1025)
        self.assertTrue(bool(j.all((tape.refusal[3:7]&512)!=0)))
        data=self.f.runtime['public'].device.causal.no_damage
        for bad in (tape._replace(event_turn=tape.event_turn.at[7,1].add(1)),
                    tape._replace(words=tape.words.at[7,1,2].set(2)),
                    tape._replace(words=tape.words.at[7,0,6].set(3))):
            self.assertFalse(bool(cb.owned(bad,0,data)[7]))
        for name in ('engine_binding','source_binding','move_binding','producer_binding'):
            bad=tape._replace(**{name:getattr(tape,name).at[0].add(j.uint32(1))})
            self.assertFalse(bool(cb.owned(bad,0,data).any()))

    def test_actual_absorb_then_switch_both_full_witness_encoder_and_physics(self):
        self.enterContext(eager.eager_source_witness(self.f.prepared))
        f=self.f;arrays=f.f.base().arrays('pressure','pressure')
        arrays['moves'][:]=np.array([f.metadata['vocabulary']['moves'].index(n)for n in f.names]);arrays['pp'][:]=[16,8,24,24];arrays['item'][:]=0
        arrays['stats'][:,:,0]=300;arrays['stats'][:,:,3]=50;arrays['stats'][:,:,4]=200
        arrays['stats'][0,:,5]=150;arrays['stats'][1,:,5]=220
        for side,name in enumerate(('pikachu','houndoom')):
            arrays['species'][side,0]=f.metadata['vocabulary']['species'].index(name)
            types=f.dex.species_info(name).types
            arrays['types'][side,0]=[f.metadata['vocabulary']['types'].index(t.lower())for t in types]+[0]*(2-len(types))
        arrays['ability'][1,0]=f.metadata['vocabulary']['abilities'].index('flashfire')
        with TemporaryDirectory()as tmp:
            p=Path(tmp)/'literal.npz';np.savez_compressed(p,**dict(arrays,metadata_utf8=np.frombuffer(json.dumps(f.metadata).encode(),np.uint8)))
            pool=source_provenance.load_constructor_pool(p)
            session=driver.construct(pool,np.tile([[0,1]],(2,1)).astype(np.int64),prepared=f.prepared,
                stream_ids=j.array([30001,30002],j.int32),seed=31)
        self.assertEqual(session.state.core.admitted.tolist(),[True,True])
        draws=j.zeros((2,2,12),j.int32).at[:,:,4].set(1)
        chance=transition.ChanceInputs(draws,j.zeros((2,2),j.int32),j.full((2,2),99,j.int32),j.zeros((2,2),j.int32),j.zeros((2,2),j.int32),j.zeros(2,j.int32))
        plain=jax.jit(lambda state,actions:transition.step_with_chance(state,actions,f.table,chance).state)
        frames=[];original=f.runtime['frame']
        def capture(*args,**kwargs):
            frame=original(*args,**kwargs);frames.append(frame);return frame
        f.runtime['frame']=capture
        try:
            for step,actions in enumerate((j.array([[2,1]]*2,j.int32),j.array([[1,4]]*2,j.int32))):
                before=session.state
                result=driver.advance(session,actions,chance_inputs=chance)
                frame=frames[-1]
                for a,z in zip(jax.tree.leaves(frame.state),jax.tree.leaves(plain(before.core.engine,actions)),strict=True):np.testing.assert_array_equal(a,z)
                self.assertEqual(result.state.core.admitted.tolist(),[True,True],msg=json.dumps(dict(step=step,
                    source=frame.callbacks.refusal.tolist(),risk=result.state.core.risks.risk.tolist(),
                    ledger=result.state.core.execution.refusal.tolist(),witness=result.state.core.frame_input_report.refusal.tolist(),
                    kinds=frame.callbacks.words[0,:int(frame.callbacks.count[0]),0].tolist(),owners=[o.reason.tolist()for o in result.owners])))
                for owner in result.owners:
                    self.assertEqual(owner.full_line_consumers_complete.tolist(),[True,True])
                    self.assertEqual(owner.encoded.observation.admitted.tolist(),[True,True]);self.assertFalse(bool(owner.B4_qualified.any()))
                if step==0:self.assertEqual((frame.state.volatile[:,1,0]&absorption.FLASH_FIRE).tolist(),[2,2])
        finally:f.runtime['frame']=original
        tape=frames[-1].callbacks;self.assertEqual(tape.words[:,:3,0].tolist(),[[35,37,6]]*2)
        self.assertEqual(frames[-1].state.volatile[:,1,0].tolist(),[0,0]);self.assertEqual(frames[-1].state.active[:,1].tolist(),[1,1])
        ledger=result.state.core.execution;tickets=ledger.site==int(ex.Site.FLASH_FIRE_END)
        self.assertEqual(j.sum(tickets,axis=1).tolist(),[1,1]);self.assertTrue(bool(j.all(j.where(tickets,ledger.stage==3,True))))
        # ORIGINAL complete writer, exact current sequence and lexical identity;
        # all twelve scheduler controls and seven end-line controls are compared.
        for owner in (0,1):
            lines=[]
            for side in (0,1):
                slot=int(before.core.engine.active[0,side]);sp=int(before.core.engine.species[0,side,slot])
                name=f.catalog.species.names[int(f.runtime['public'].device.causal.species[sp])]
                lines.append(f'|switch|p{side+1}a: {name}|{name}, L80, M|100/100')
                if side==1:outgoing=name
            lines.append(f'|turn|{int(tape.event_turn[0,0])}')
            wire,_=public_wire.project_lines(public_wire.WireState(),lines,**v4_encoder.wire_kwargs(f.catalog))
            wire=replace(wire,next_sequence=int(before.observers[owner].projection.sequence[0])+1)
            _,packets=public_wire.project_lines(wire,['|','|t:|1',f'|-end|p2a: {outgoing}|ability: Flash Fire|[silent]'],**v4_encoder.wire_kwargs(f.catalog))
            wanted=np.asarray([e.words()for packet in packets for e in packet.events],np.int32)
            report=result.state.core.frame_input_report;actual=np.asarray(report.owner_full_words[0,owner])[np.asarray(report.owner_full_valid[0,owner])]
            np.testing.assert_array_equal(actual[:len(wanted)],wanted)


def tearDownModule():
    eager.clear()   # drop the consume executables this module compiled


if __name__=='__main__':unittest.main()
