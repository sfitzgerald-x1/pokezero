"""Literal Leftovers receipt -> ORIGINAL full writer/consumers, tiny fixtures.

These are trusted lower-seam fixtures, not actual owning-origin or complete
source coverage claims. Actual write/caller tests are separate.
"""
import unittest
from unittest.mock import patch
import jax
import jax.numpy as j
import numpy as np
from gpu_engine import engine_full_v4_event_device as full, engine_lifecycle_callbacks as life
from gpu_engine import engine_leftovers_callbacks as leftovers, engine_public_events as events
from gpu_engine import engine_status_callbacks as statuses
from gpu_engine import engine_weather_speed_profile as speed, engine_request_source as source
from gpu_engine import public_wire, public_inputs as inputs, v4_encoder as v4
import test_gpu_engine_engine_weather_public as fixture


class LeftoversPublicTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture.WeatherPublicTests.setUpClass();cls.f=fixture.WeatherPublicTests
        cls.rules=source.load_source_rules(cls.f.f.f.root,cls.f.metadata,node=cls.f.f.f.node)
        cls.profile=speed.compile_profile(cls.f.metadata,catalog=cls.f.catalog,
            engine_catalog=cls.f.producer,rules=cls.rules)
        cls.heal=leftovers.compile_leftovers(cls.f.metadata,engine_catalog=cls.f.producer)
        cls.fast=staticmethod(jax.jit(lambda s,f,p,r,data:full.project_and_fold(
            s,f,cls.f.compiled.device,p,cls.f.catalog,weather_hp=(r,cls.profile.device),leftovers=data)))

    def prepared(self,status=0,physical=0):
        f=self.f();original=f.initial('hail',arrays=f.arrays('pressure','pressure'))
        if physical:
            original=f.step(original,[[8,8]]*2)
        # An explicitly synthetic lower-seam pre-state. No Session origin claim.
        state=original.state._replace(turn=j.zeros(2,j.int32),active=j.zeros((2,2),j.int32),
            entry_pending=j.ones((2,2),bool),status=j.zeros_like(original.state.status),
            hp=original.state.max_hp)
        cursor=life.register(original.cursor.stream_id,**f.kwargs())
        initial=life.initialize_with_lifecycle(state,self.f.table,cursor=cursor,
            tie_coin=j.zeros(2,j.int32),**f.kwargs())
        record=speed.qualify_birth(state,initial.cursor.stream_id,profile=self.profile)
        stream,public=f.start(initial,j.array([0,1],j.int32))
        before,wire=f.compare(stream,initial,public,[public_wire.WireState()]*2)
        if physical:
            changed=f.step(initial,[[8,8]]*2)
            before,wire=f.compare(before.projection.state,changed,before.public,wire)
            initial=changed
        if status:
            tape=life._tape(initial.cursor,self.f.producer,64)
            for side in (0,1):
                tape=statuses.changed(tape,j.full(2,side,j.int32),initial.cursor.physical_active[:,side],
                    j.zeros(2,j.int32),j.full(2,status,j.int32),statuses.PRIMARY,eligible=j.ones(2,bool))
            changed=initial._replace(state=initial.state._replace(status=j.full_like(initial.state.status,status)),
                callbacks=tape,before_registry=life.snapshot(initial.cursor),after_registry=life.snapshot(initial.cursor))
            before,wire=f.compare(before.projection.state,changed,before.public,wire)
            initial=changed
        return before,wire,initial,record

    def frame(self,initial):
        cursor=initial.cursor;state=initial.state;rows=j.arange(2)
        tape=life._tape(cursor,self.f.producer,64)
        # Test-only literal actual-style damage write followed by residual heal.
        damage=[];heals=[]
        for side in (0,1):
            physical=cursor.physical_active[:,side]
            maximum=state.max_hp[rows,side,physical];old=maximum-7
            status=state.status[rows,side,physical]
            tape=events.condition(tape,j.full(2,side,j.int32),physical,old,maximum,status,eligible=j.ones(2,bool))
            damage.append((old,maximum,status))
        tape=life.action_end(tape,j.ones(2,bool))
        for side in (0,1):
            physical=cursor.physical_active[:,side];old,maximum,status=damage[side]
            new=maximum-1
            # Future source registration defers only its own UNKNOWN inside
            # the emitter. This isolated formatter fixture models that seam;
            # it never erases an aggregate refusal bit after emission.
            with patch.object(leftovers,'_unknown',side_effect=lambda tape,eligible:tape):
                tape=leftovers.record_leftovers_heal_device(tape,side,physical,
                    self.heal.device.leftovers_item,old,new,maximum,status,self.heal.device,eligible=True)
            heals.append((new,maximum,status))
        end=state._replace(turn=state.turn+1)
        tape=life.finish(tape,end,live=j.ones(2,bool),replacement=j.zeros(2,bool),
            pending_before=j.zeros(2,bool),residual_ready=j.ones(2,bool),
            defer_residual=j.zeros(2,bool),turn_context=j.ones(2,bool))
        return life.LifecycleResult(end,tape.cursor,tape,life.snapshot(cursor),life.snapshot(tape.cursor),
            j.zeros(2,bool),j.zeros(2,j.int32)),damage,heals

    def reference(self,before,wire,frame,damage,heals):
        rows=[];tokens=('', 'slp','par','brn','frz','psn','tox');status_map=(0,2,3,5,6,1,4)
        for row in range(2):
            active=dict(wire[row].active_idents());lines=[]
            def line(side,values,heal):
                hp,maximum,status=map(lambda a:int(a[row]),values)
                owner=side==row
                hp=hp if owner else min(99,(100*hp+maximum-1)//maximum)
                maximum=maximum if owner else 100
                token=tokens[status_map[status]]
                return f'|-{"heal" if heal else "damage"}|{active[side]}|{hp}/{maximum}'+(
                    ' '+token if token else '')+('|[from] item: Leftovers' if heal else '')
            lines.extend(line(side,damage[side],False) for side in (0,1));lines.append('|')
            lines.extend(line(side,heals[side],True) for side in (0,1))
            lines.extend(('|upkeep',f'|turn|{int(frame.cursor.turn[row])}'))
            _,ps=public_wire.project_lines(wire[row],lines,**v4.wire_kwargs(self.f.catalog))
            rows.append([e for p in ps for e in p.events])
        ns={k:np.array(getattr(self.f.compiled.device.namespaces,k)) for k in full._NAMES if k!='source_binding'}
        chunks=list(inputs.pack_chunks(np.array(frame.cursor.stream_id),rows,
            source_binding=np.array(self.f.catalog.source.device.binding),**ns))
        expected=before.public
        for chunk in chunks:
            expected,error=v4.fold_chunk(expected,chunk,self.f.catalog)
            self.assertEqual(error.tolist(),[0,0])
        return chunks,expected

    def test_exact_original_full_words_item_fact_and_retained_leaves_all_statuses_both_owners(self):
        for status in range(7):
            before,wire,initial,record=self.prepared(status)
            frame,damage,heals=self.frame(initial)
            chunks,expected=self.reference(before,wire,frame,damage,heals)
            got=self.fast(before.projection.state,frame,before.public,record,self.heal.device)
            self.assertEqual(got.projection.error.tolist(),[0,0],msg=str(status))
            self.f().same(got.public,expected)
            for i,chunk in enumerate(chunks):
                self.f().same(got.projection.frames.words[i],chunk.words)
                self.f().same(got.projection.frames.valid[i],chunk.valid)
            item=self.f.catalog.items.ids['leftovers']
            self.assertEqual(got.public.context.items.revealed[:,:,0].tolist(),[[item,item]]*2)
            self.assertFalse(bool(got.projection.full_encoder_admitted.any()))
            self.assertFalse(bool(got.projection.B4_qualified.any()))

    def test_nonphysical_active_addresses_default_closed_and_malformed_atomicity(self):
        before,wire,initial,record=self.prepared(physical=5)
        frame,damage,heals=self.frame(initial)
        chunks,expected=self.reference(before,wire,frame,damage,heals)
        baseline=self.fast(before.projection.state,frame,before.public,record,self.heal.device)
        self.assertEqual(baseline.projection.error.tolist(),[0,0])
        self.f().same(baseline.public,expected)
        closed=full.project_frame(before.projection.state,frame,self.f.compiled.device,
            weather_hp=(record,self.profile.device))
        self.assertTrue(bool((closed.error!=0).all()))
        positions=np.flatnonzero(np.array(frame.callbacks.words[0,:,0])==17);at=int(positions[0])
        blank=int(np.flatnonzero(np.array(frame.callbacks.words[0,:,0])==life.ACTION_END)[0])
        for label in ('old_hp','status','item','physical','before_blank','duplicate','foreign_data','birth','prior_refusal'):
            tape=frame.callbacks;r=record;data=self.heal.device
            if label=='old_hp':tape=tape._replace(words=tape.words.at[0,at,5].add(-1))
            elif label=='status':tape=tape._replace(words=tape.words.at[0,at,8].set(1))
            elif label=='item':tape=tape._replace(words=tape.words.at[0,at,9].set(0))
            elif label=='physical':tape=tape._replace(words=tape.words.at[0,at,4].set(0))
            elif label=='before_blank':tape=tape._replace(words=tape.words.at[0,blank].set(tape.words[0,at]).at[0,at].set(tape.words[0,blank]))
            elif label=='duplicate':tape=tape._replace(words=tape.words.at[0,int(positions[1])].set(tape.words[0,at]))
            elif label=='foreign_data':data=data._replace(source_binding=data.source_binding.at[0].add(j.uint32(1)))
            elif label=='birth':r=r._replace(qualified_birth=r.qualified_birth.at[0].set(False))
            else:tape=tape._replace(refusal=tape.refusal.at[0].set(257))
            got=self.fast(before.projection.state,frame._replace(callbacks=tape),before.public,r,data)
            self.assertNotEqual(int(got.projection.error[0]),0,msg=label)
            if label!='foreign_data':self.assertEqual(int(got.projection.error[1]),0,msg=label)
            self.assertFalse(bool(got.projection.frames.valid[:,0].any()),msg=label)
            self.assertEqual(int(got.public.cursor.sequence[0]),int(before.public.cursor.sequence[0]),msg=label)


if __name__=='__main__':unittest.main()
