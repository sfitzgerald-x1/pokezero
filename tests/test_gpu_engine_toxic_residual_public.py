"""Literal Toxic residual formatter, not a physical-history qualification."""
import unittest
import jax.numpy as j
import numpy as np
from gpu_engine import engine_status_residual_callbacks as status,engine_lifecycle_callbacks as life
from gpu_engine import public_wire,public_inputs,v4_encoder,mechanics as m,engine_full_v4_event_device as full
import test_gpu_engine_status_residual_public as fixture

class ToxicResidualPublicTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture.StatusResidualPublicTests.setUpClass();cls.fixture=fixture.StatusResidualPublicTests()

    def frame(self,initial,before_counter,after_counter):
        f=self.fixture;cursor=initial.cursor;state=initial.state;rows=j.arange(2)
        tape=life.action_end(life._tape(cursor,f.f.producer,64),j.ones(2,bool));values=[]
        for side in (0,1):
            physical=cursor.physical_active[:,side];maximum=state.max_hp[rows,side,physical];new=maximum-13
            tape=status.record_status_residual_device(tape,side,physical,maximum,new,maximum,j.full(2,m.TOX,j.int32),
                j.full(2,before_counter,j.int32),j.full(2,after_counter,j.int32),f.helper.device,
                eligible=True,_defer_own_unknown=True)
            values.append((new,maximum))
        end=state._replace(turn=state.turn+1)
        tape=life.finish(tape,end,live=j.ones(2,bool),replacement=j.zeros(2,bool),pending_before=j.zeros(2,bool),
            residual_ready=j.ones(2,bool),defer_residual=j.zeros(2,bool),turn_context=j.ones(2,bool))
        return life.LifecycleResult(end,tape.cursor,tape,life.snapshot(cursor),life.snapshot(tape.cursor),
            j.zeros(2,bool),j.zeros(2,j.int32)),values

    def test_original_both_owner_status_hp_psn_tag_and_no_private_counter_words(self):
        f=self.fixture;before,wire,initial,record=f.prepared(m.TOX,physical=5)
        frame,values=self.frame(initial,0,1)
        actual=f.fast(before.projection.state,frame,before.public,record,f.helper.device)
        self.assertEqual(actual.projection.error.tolist(),[0,0]);rows=[]
        for row in range(2):
            lines=['|'];active=dict(wire[row].active_idents())
            for side,(hp,maximum) in enumerate(values):
                hp,maximum=int(hp[row]),int(maximum[row])
                if side!=row:hp=min(99,(100*hp+maximum-1)//maximum);maximum=100
                lines.append(f'|-damage|{active[side]}|{hp}/{maximum} tox|[from] psn')
            lines+=['|upkeep',f'|turn|{int(frame.cursor.turn[row])}']
            _,packets=public_wire.project_lines(wire[row],lines,**v4_encoder.wire_kwargs(f.f.catalog))
            rows.append([e for p in packets for e in p.events])
        namespaces={k:np.array(getattr(f.f.compiled.device.namespaces,k)) for k in full._NAMES}
        chunks=list(public_inputs.pack_chunks(np.array(frame.cursor.stream_id),rows,**namespaces))
        expected=before.public
        for at,chunk in enumerate(chunks):
            np.testing.assert_array_equal(actual.projection.frames.words[at],chunk.words)
            np.testing.assert_array_equal(actual.projection.frames.valid[at],chunk.valid)
            expected,error=v4_encoder.fold_chunk(expected,chunk,f.f.catalog);self.assertEqual(error.tolist(),[0,0])
        f.f().same(actual.public,expected)
        self.assertFalse(bool(actual.projection.B4_qualified.any()))

    def test_counter_skip_or_stasis_cannot_defer_and_saturation_is_literal_only(self):
        f=self.fixture;before,wire,initial,record=f.prepared(m.TOX,physical=5)
        for old,new,valid in ((0,0,False),(0,2,False),(14,15,True),(15,15,True)):
            frame,_=self.frame(initial,old,new)
            residual=(j.arange(64)[None,:]<frame.callbacks.count[:,None])&(frame.callbacks.words[:,:,0]==20)
            self.assertEqual(residual.sum(axis=1).tolist(),[2,2])
            self.assertEqual(((frame.callbacks.refusal&status.UNKNOWN_STATUS_RESIDUAL)==0).tolist(),[valid,valid])
