"""Trusted format-seam controls, NOT owning hook/history/source qualification.

The diagnostic emitter's own UNKNOWN is suppressed only by an explicit fixture
patch. No aggregate prior bit is erased, and the actual owning registered path
must remain closed until its separate hook/history proof is implemented.
"""
import unittest
from unittest.mock import patch
import jax
import jax.numpy as j
import numpy as np
from gpu_engine import engine_full_v4_event_device as full,engine_lifecycle_callbacks as life
from gpu_engine import engine_status_residual_callbacks as status,engine_public_events as events
from gpu_engine import public_wire,public_inputs as inputs,v4_encoder as v4,mechanics as m
import test_gpu_engine_leftovers_public as fixtures


class StatusResidualPublicTests(unittest.TestCase):
    prepared=fixtures.LeftoversPublicTests.prepared

    @classmethod
    def setUpClass(cls):
        fixtures.LeftoversPublicTests.setUpClass();base=fixtures.LeftoversPublicTests
        cls.f=base.f;cls.profile=base.profile;cls.heal=base.heal
        cls.helper=status.compile_status_residual(cls.f.metadata,engine_catalog=cls.f.producer)
        cls.fast=staticmethod(jax.jit(lambda stream,frame,public,record,data:full.project_and_fold(
            stream,frame,cls.f.compiled.device,public,cls.f.catalog,
            weather_hp=(record,cls.profile.device),status_residual=data)))

    def frame(self,initial):
        cursor=initial.cursor;state=initial.state;rows=j.arange(2)
        tape=life._tape(cursor,self.f.producer,64);before=[];after=[]
        # A preceding same-frame condition changes the public HP first. Status
        # oldHP must join that frontier, not frame-start or final hidden HP.
        for side in (0,1):
            physical=cursor.physical_active[:,side];maximum=state.max_hp[rows,side,physical]
            value=maximum-7;kind=state.status[rows,side,physical]
            tape=events.condition(tape,j.full(2,side,j.int32),physical,value,maximum,kind,eligible=j.ones(2,bool))
            before.append((value,maximum,kind))
        tape=life.action_end(tape,j.ones(2,bool))
        for side in (0,1):
            old,maximum,kind=before[side];new=old-9
            with patch.object(status,'_unknown',side_effect=lambda tape,eligible:tape):
                tape=status.record_status_residual_device(tape,side,cursor.physical_active[:,side],
                    old,new,maximum,kind,j.zeros(2,j.int32),j.zeros(2,j.int32),self.helper.device,eligible=True)
            after.append((new,maximum,kind))
        end=state._replace(turn=state.turn+1)
        tape=life.finish(tape,end,live=j.ones(2,bool),replacement=j.zeros(2,bool),
            pending_before=j.zeros(2,bool),residual_ready=j.ones(2,bool),
            defer_residual=j.zeros(2,bool),turn_context=j.ones(2,bool))
        return life.LifecycleResult(end,tape.cursor,tape,life.snapshot(cursor),life.snapshot(tape.cursor),
            j.zeros(2,bool),j.zeros(2,j.int32)),before,after

    def reference(self,before,wire,frame,first,last):
        rows=[];tokens={m.BRN:'brn',m.PSN:'psn'}
        for row in range(2):
            active=dict(wire[row].active_idents());lines=[]
            def line(side,values,tagged):
                hp,maximum,kind=(int(a[row]) for a in values)
                if side!=row:hp=min(99,(100*hp+maximum-1)//maximum);maximum=100
                token=tokens[kind]
                return f'|-damage|{active[side]}|{hp}/{maximum} {token}'+('|[from] '+token if tagged else '')
            lines.extend(line(side,first[side],False) for side in (0,1));lines.append('|')
            lines.extend(line(side,last[side],True) for side in (0,1))
            lines.extend(('|upkeep',f'|turn|{int(frame.cursor.turn[row])}'))
            _,ps=public_wire.project_lines(wire[row],lines,**v4.wire_kwargs(self.f.catalog))
            rows.append([event for projection in ps for event in projection.events])
        namespaces={key:np.array(getattr(self.f.compiled.device.namespaces,key)) for key in full._NAMES}
        chunks=list(inputs.pack_chunks(np.array(frame.cursor.stream_id),rows,**namespaces))
        expected=before.public
        for chunk in chunks:
            expected,error=v4.fold_chunk(expected,chunk,self.f.catalog);self.assertEqual(error.tolist(),[0,0])
        return chunks,expected

    def test_exact_original_full_words_and_public_leaves_for_both_statuses_owners_and_nonphysical_slots(self):
        for kind in (m.BRN,m.PSN):
            before,wire,initial,record=self.prepared(kind,physical=5)
            frame,first,last=self.frame(initial);chunks,expected=self.reference(before,wire,frame,first,last)
            result=self.fast(before.projection.state,frame,before.public,record,self.helper.device)
            self.assertEqual(result.projection.error.tolist(),[0,0],msg=str(kind))
            self.f().same(result.public,expected)
            for index,chunk in enumerate(chunks):
                self.f().same(result.projection.frames.words[index],chunk.words)
                self.f().same(result.projection.frames.valid[index],chunk.valid)
            self.assertFalse(bool(result.projection.full_encoder_admitted.any()))
            self.assertFalse(bool(result.projection.B4_qualified.any()))

    def test_missing_registration_and_malformed_unsafe_or_duplicate_receipt_roll_back_whole_row(self):
        before,wire,initial,record=self.prepared(m.BRN,physical=5)
        frame,first,last=self.frame(initial)
        baseline=self.fast(before.projection.state,frame,before.public,record,self.helper.device)
        self.assertEqual(baseline.projection.error.tolist(),[0,0])
        closed=full.project_frame(before.projection.state,frame,self.f.compiled.device,
            weather_hp=(record,self.profile.device))
        self.assertTrue(bool((closed.error!=0).all()))
        locations=np.flatnonzero(np.asarray(frame.callbacks.words[0,:,0])==20);at=int(locations[0])
        blank=int(np.flatnonzero(np.asarray(frame.callbacks.words[0,:,0])==life.ACTION_END)[0])
        for label in ('oldHP','counter','physical','reason','faint','beforeBlank','duplicate','data','profile','prior'):
            tape=frame.callbacks;data=self.helper.device;r=record
            if label=='oldHP':tape=tape._replace(words=tape.words.at[0,at,5].add(1))
            elif label=='counter':tape=tape._replace(words=tape.words.at[0,at,2].set(1))
            elif label=='physical':tape=tape._replace(words=tape.words.at[0,at,4].set(0))
            elif label=='reason':tape=tape._replace(words=tape.words.at[0,at,8].set(m.TOX).at[0,at,9].set(status.TOXIC))
            elif label=='faint':tape=tape._replace(words=tape.words.at[0,at,6].set(0))
            elif label=='beforeBlank':tape=tape._replace(words=tape.words.at[0,blank].set(tape.words[0,at]).at[0,at].set(tape.words[0,blank]))
            elif label=='duplicate':tape=tape._replace(words=tape.words.at[0,int(locations[1])].set(tape.words[0,at]))
            elif label=='data':data=data._replace(source_binding=data.source_binding.at[0].add(j.uint32(1)))
            elif label=='profile':r=r._replace(qualified_birth=r.qualified_birth.at[0].set(False))
            else:tape=tape._replace(refusal=tape.refusal.at[0].set(257))
            failed=self.fast(before.projection.state,frame._replace(callbacks=tape),before.public,r,data)
            self.assertNotEqual(int(failed.projection.error[0]),0,msg=label)
            if label!='data':self.assertEqual(int(failed.projection.error[1]),0,msg=label)
            self.assertFalse(bool(failed.projection.frames.valid[:,0].any()))
            self.assertEqual(int(failed.public.cursor.sequence[0]),int(before.public.cursor.sequence[0]))


if __name__=='__main__':unittest.main()
