"""Literal primary-SLP accuracy-miss input; no games or sampled chance."""
from pathlib import Path
import subprocess
import unittest
import jax
import jax.numpy as j
import numpy as np
from gpu_engine import engine_sleep_powder_callbacks as sleep,engine_toxic_callbacks as primary
from gpu_engine import engine_lifecycle_callbacks as life,engine_public_events as events
from gpu_engine import engine_kernel_frame_risk as risk,engine_execution_completeness as ex
from gpu_engine import engine_full_v4_event_device as full,engine_source_frame_witness as witness
from gpu_engine import public_wire,public_inputs as inputs,v4_encoder,mechanics as m
import test_gpu_engine_primary_sleep_runtime as fixture


class PrimarySleepMissTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture.PrimarySleepRuntimeTests.setUpClass();cls.f=fixture.PrimarySleepRuntimeTests

    def component(self,b=8,active=None):
        f=self.f;side=j.arange(b,dtype=j.int32)%2;physical=j.full(b,3,j.int32);target=j.full(b,5,j.int32)
        move=j.full(b,f.move_id,j.int32);yes=j.ones(b,bool);no=j.zeros(b,bool);z=j.zeros(b,j.int32)
        active=yes if active is None else active
        cursor=life.register(j.arange(14901,14901+b,dtype=j.int32),catalog=f.producer,expected_binding=f.producer.binding,
            expected_engine_binding=np.asarray(f.producer.engine_binding,np.uint32))
        rows=j.arange(b);phys=j.zeros((b,2),j.int32).at[rows,side].set(physical).at[rows,1-side].set(target)
        mapping=j.full((b,2,6),-1,j.int32).at[rows,side,physical].set(0).at[rows,1-side,target].set(0)
        cursor=cursor._replace(generation=j.ones((b,2),j.int32),turn=j.ones(b,j.int32),initialized=yes,physical_active=phys,
            public_slot=j.zeros((b,2),j.int32),encounter=j.ones((b,2),j.int32),established=j.ones((b,2),bool),
            physical_to_public=mapping,next_slot=j.ones((b,2),j.int32),revealed_species_id=j.broadcast_to(j.array([1,2],j.int32),(b,2)))
        tape=life._tape(cursor,f.producer,64);account=f.runtime['execution_compiled'];r=f.runtime['kernel_risk_compiled']
        tape=ex.begin_on_tape(tape,ex.initialize(cursor.stream_id,compiled=account),account.device,active=yes)
        carry=risk.begin_frame_device(risk.initialize_device(tape.stream_id,r.device),tape,r.device,active=yes)
        tape=tape._replace(execution=tape.execution._replace(risks=risk.Context(carry,r.device)))
        tape=events.move(tape,side,physical,target,move,eligible=active)
        tape,ticket=ex.open_attempt(tape,ex.Site.PRIMARY_SLEEP,1-side,active=active)
        # Odd rows preserve the literal unused damaging-immunity bit.
        attempt=sleep.Attempt(no,no,no,no,no,yes,no,no,no,side==1,no,z,z,z,z)
        return tape,ticket,side,physical,target,move,attempt,active

    def record(self,parts,context=None):
        tape,ticket,side,physical,target,move,attempt,active=parts
        return sleep.record_device(tape,ticket,side,physical,target,move,attempt,context or self.f.runtime['sleep_powder_context'],requested=active)

    def test_explicit_original_mask_legacy_closed_and_default_sources_exact(self):
        f=self.f;data=f.helper.device;ids=f.metadata['vocabulary']['moves']
        np.testing.assert_array_equal(data.move_mask,data.miss_mask)
        np.testing.assert_array_equal(data.miss_mask,f.runtime['kernel_risk_compiled'].device.primary_sleep_miss_candidates)
        self.assertFalse(bool(f.old.helper.device.miss_mask.any()))
        self.assertTrue(bool(data.primary_sleep))
        for name in ('transition.py','mechanics.py','secondary.py','state.py','entry.py','engine_retained_encoder_driver.py'):
            self.assertEqual(Path('gpu_engine',name).read_bytes(),subprocess.check_output(['git','show','b8092a85:gpu_engine/'+name]))
        for name in ('hypnosis','spore','sleeppowder','sing'):
            row=ids.index(name);self.assertTrue(bool(data.miss_mask[row]));self.assertTrue(bool(f.runtime['public'].device.no_damage_miss[row]))
        legacy=data._replace(primary_sleep=j.array(False),miss_mask=j.zeros_like(data.miss_mask))
        self.assertFalse(bool(sleep.selected_miss_device(legacy,j.array([f.move_id],j.int32)).any()))

    def test_literal_mixed_span_binding_guards_and_sticky_flags(self):
        f=self.f;active=j.array([True]*6+[False]*2);parts=self.component(active=active);before=parts[0];good=self.record(parts)
        self.assertEqual(good.count.tolist(),[2]*6+[0,0]);self.assertEqual(sleep.owned_miss_device(good,j.array(1,j.int32)).tolist(),active.tolist())
        self.assertEqual(sleep.span_matches(good,*parts[1:7],f.runtime['sleep_powder_context']).tolist(),active.tolist())
        captured=lambda t,context=f.runtime['sleep_powder_context']:risk.primary_status_on_tape(t,active,active=active,
            captured_sleep_powder=risk.CapturedSleepPowder(context,*parts[1:7]))
        self.assertEqual(captured(good).execution.risks.carry.risk.tolist(),[0]*8)
        self.assertEqual(witness._raw_complete(good,f.runtime['frame_witness_compiled'].device).tolist(),[True]*8)
        raw=good.words[:,1]
        for field,val in (('accuracy_hit',j.ones(8,bool)),('hit',j.ones(8,bool)),('type_immune',j.ones(8,bool)),
                ('ability_immune',j.ones(8,bool)),('substitute_block',j.ones(8,bool)),('protected',j.ones(8,bool)),
                ('absorbed',j.ones(8,bool)),('applied',j.ones(8,bool)),('old_status',j.full(8,m.PAR,j.int32)),('old_counter',j.ones(8,j.int32))):
            with self.subTest(field=field):
                changed=parts[6]._replace(**{field:val});words=primary.words_device(parts[2],parts[3],parts[4],parts[5],changed)
                self.assertFalse(bool(sleep.supported_miss_raw(words).any()))
        for label,bad in (
            ('prefix',good._replace(words=good.words.at[0,0,5].set(0))),
            ('clock',good._replace(event_turn=good.event_turn.at[0,1].add(1))),
            ('snapshot',good._replace(execution=good.execution._replace(ledger=good.execution.ledger._replace(words=good.execution.ledger.words.at[0,1,6].add(1))))),
            ('claim',good._replace(execution=good.execution._replace(ledger=good.execution.ledger._replace(claimed=good.execution.ledger.claimed.at[0,1].set(0)))))):
            with self.subTest(label=label):self.assertEqual(sleep.owned_miss_device(bad,j.array(1,j.int32)).tolist(),[False]+active.tolist()[1:])
        moved=good._replace(cursor=good.cursor._replace(turn=good.cursor.turn+1,phase=j.ones(8,j.int32)))
        self.assertEqual(sleep.owned_miss_device(moved,j.array(1,j.int32)).tolist(),active.tolist())
        for name in ('engine_binding','source_binding','move_binding','producer_binding'):
            with self.subTest(name=name):
                self.assertFalse(bool(sleep.owned_miss_device(good._replace(**{name:getattr(good,name).at[0].add(j.uint32(1))}),j.array(1,j.int32)).any()))
                with self.assertRaises(ValueError):sleep.owned_miss_device(good._replace(**{name:j.broadcast_to(getattr(good,name),(8,8))}),j.array(1,j.int32))
        prior=good._replace(refusal=good.refusal.at[0].set(65),execution=good.execution._replace(risks=good.execution.risks._replace(carry=good.execution.risks.carry._replace(risk=good.execution.risks.carry.risk.at[0].set(64)))))
        kept=captured(prior);self.assertEqual(int(kept.refusal[0])&65,65);self.assertEqual(int(kept.execution.risks.carry.risk[0])&64,64)
        for a,b in zip(jax.tree.leaves(before),jax.tree.leaves(good),strict=True):
            if getattr(a,'ndim',0) and a.shape[0]==8:np.testing.assert_array_equal(a[6:],b[6:])

    def test_both_original_full_consumers_exact_miss_words_and_continuation(self):
        f=self.f;parts=self.component(b=2);tape=self.record(parts);cursor=tape.cursor;kw=v4_encoder.wire_kwargs(f.catalog)
        names=[f.catalog.species.names[int(f.runtime['public'].device.causal.species[i])]for i in (1,2)]
        ident=[f'p{i+1}a: '+names[i]for i in (0,1)]
        prefix=[f'|switch|{ident[0]}|{names[0]}, L80, M|100/100',f'|switch|{ident[1]}|{names[1]}, L80, M|100/100','|turn|1']
        wire,packets=public_wire.project_lines(public_wire.WireState(),prefix,**kw);initial=[e for packet in packets for e in packet.events]
        name=f.catalog.dex.move_info('hypnosis').name
        expected=[]
        for side in (0,1):
            _,packets=public_wire.project_lines(wire,[f'|move|{ident[side]}|{name}|{ident[1-side]}|[miss]',f'|-miss|{ident[side]}|{ident[1-side]}'],**kw)
            expected.append(np.array([e.words()for p in packets for e in p.events],np.int32))
        ns={k:np.asarray(getattr(f.runtime['public'].device.namespaces,k))for k in full._NAMES}
        fold=jax.jit(lambda stream,frame,p:full.project_and_fold(stream,frame,f.runtime['public'].device,p,f.catalog))
        for owner in (0,1):
            p=v4_encoder.initialize(f.catalog,stream_ids=tape.stream_id,owner_sides=j.full(2,owner,j.int32))
            for chunk in inputs.pack_chunks(np.asarray(tape.stream_id),[initial]*2,**ns):
                p,error=v4_encoder.fold_chunk(p,chunk,f.catalog);self.assertFalse(bool(error.any()))
            stream=full.register(tape.stream_id,j.full(2,owner,j.int32),compiled=f.runtime['public'])
            stream=stream._replace(causal=stream.causal._replace(cursor=cursor,sequence=j.full(2,3,j.int32)),sequence=p.cursor.sequence,
                live=j.ones((2,2),bool),species=p.mons.species,status=p.mons.status,hp=p.mons.hp_numerator,maximum=p.mons.hp_denominator)
            frame=life.LifecycleResult(None,cursor,tape,life.snapshot(cursor),life.snapshot(cursor),j.zeros(2,bool),j.zeros(2,j.int32))
            got=fold(stream,frame,p);self.assertEqual(got.projection.error.tolist(),[0,0])
            for row in (0,1):
                actual=np.asarray(got.projection.frames.words).transpose(1,0,2,3)[row][np.asarray(got.projection.frames.valid).transpose(1,0,2)[row]]
                np.testing.assert_array_equal(actual,expected[row])
            self.assertFalse(bool(got.public.cursor.needs_more.any()))
