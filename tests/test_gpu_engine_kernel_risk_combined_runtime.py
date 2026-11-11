"""Real owning risk carry and both full folds; tiny fixtures, no forwards/games."""
import unittest
import jax
import jax.numpy as j
import numpy as np
from gpu_engine import engine_retained_encoder_driver as d, engine_kernel_frame_risk as r
from gpu_engine import engine_execution_completeness as accounting, transition as t
import test_gpu_engine_kernel_risk_runtime as fixture


class KernelRiskCombinedRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture.KernelRiskDriverPreparationTests.setUpClass()
        cls.base=fixture.KernelRiskDriverPreparationTests.base
        prepared=fixture.KernelRiskDriverPreparationTests
        cls.plain=prepared.plain
        # Pure sentinel only: lax.cond traces both branches, so a Python spy
        # or raising callable would observe compilation rather than execution.
        fake=lambda values:d.FakeScores(j.full(9,123.,j.float32),j.array(7,j.int32))
        cls.prepared=d.prepare(*prepared.args,**dict(prepared.options,fake_forward=fake))

    def same(self,a,b):
        left,right=jax.tree.leaves(a),jax.tree.leaves(b)
        self.assertEqual(len(left),len(right))
        for x,y in zip(left,right):np.testing.assert_array_equal(x,y)

    def assert_risk_rows_same(self,before,after,rows):
        for name in ('binding','accounting_binding'):
            self.same(getattr(before,name),getattr(after,name))
        for name in r.Carry._fields[2:]:
            self.same(getattr(before,name)[rows],getattr(after,name)[rows])

    def test_genuine_constructor_mixed_frames_and_second_owner_rollback(self):
        # Use the real source constructor and original checked compiler edges;
        # no fabricated Context, retrospective ticket or accepted fold result.
        fixture_instance=self.base()
        session=fixture_instance.construct(8,prepared=self.prepared)
        plain=fixture_instance.construct(8,prepared=self.plain)
        runtime=d._checked(self.prepared)
        before=session.state
        self.same(before.core.engine,plain.state.core.engine)
        self.assertEqual(before.core.admitted.tolist(),[True]*8)
        self.assertEqual(before.core.risks.risk.tolist(),[0]*8)
        self.assertEqual(before.core.risks.frame.tolist(),[1]*8)
        self.assertEqual(before.core.risks.counts[:,r.Branch.CONSTRUCTOR-1].tolist(),[1]*8)
        self.assertEqual(before.core.risks.counts[:,r.Branch.FRAME_END-1].tolist(),[1]*8)
        original=d._session(session)['constructed']
        self.same(original.callbacks.execution.risks.carry,before.core.risks)
        for new,old in zip(before.observers,plain.state.observers):
            self.same(new.retained,old.retained)
            self.same(new.projection,old.projection)
            self.assertEqual(new.retained_healthy.tolist(),[True]*8)
        # Distinct streams remain B rows independently for each owner.
        self.assertEqual(before.core.risks.stream_id.tolist(),list(range(1101,1109)))
        self.assertEqual([o.owner.tolist() for o in before.observers],[[0]*8,[1]*8])

        active=j.array([True,True,True,True,False,False,False,False])
        actions=j.tile(j.array([[0,8]],j.int32),(8,1)).at[1,0].set(1)
        chance=fixture_instance.chance(8)
        # Slot1 is a genuine existing damaging engine move, never a synthetic
        # risk boolean. Its selected table fields must diagnose NON_WEATHER.
        selected=int(before.core.engine.moves[1,0,0,1])
        self.assertTrue(int(runtime['kernel_risk_compiled'].device.selected_risk[selected])&r.NON_WEATHER)
        actual_frame,actual_public=runtime['frame'],runtime['public_fast']
        frames=[];owners=[]
        def capture(*args):
            frame=actual_frame(*args);frames.append(frame);return frame
        def second_owner_failure(observer,frame,profile,mask):
            owners.append(int(observer.owner[0]))
            if owners[-1]==1:
                observer=observer._replace(retained=observer.retained._replace(cursor=observer.retained.cursor._replace(
                    sequence=observer.retained.cursor.sequence.at[2].add(1))))
            return actual_public(observer,frame,profile,mask)
        runtime.update(frame=capture,public_fast=second_owner_failure)
        try:
            result=d.advance(session,actions,active_rows=active,chance_inputs=chance)
        finally:
            runtime.update(frame=actual_frame,public_fast=actual_public)
        self.assertEqual(len(frames),1);self.assertEqual(owners,[0,1])
        frame=frames[0]
        self.assertEqual(d.latest_diagnostics(session).executed_rows.tolist(),active.tolist())
        # Trace-off original numerical transition, including RNG, uses the
        # same literal actions/chance and freezes exactly the inactive rows.
        legacy=jax.jit(lambda state,act,draws:t.step_with_chance(state,act,
            runtime['request']['engine_table'],draws))(before.core.engine._replace(
                done=before.core.engine.done|~active),actions,chance)
        expected=jax.tree.map(lambda old,new:j.where(active.reshape((-1,)+(1,)*(old.ndim-1)),new,old),
            before.core.engine,legacy.state)
        self.same(result.state.core.engine,expected)
        self.same(result.state.core.engine.keys,expected.keys)
        self.assertEqual(result.state.core.admitted.tolist(),[True,False,False,True,True,True,True,True])
        carry=result.state.core.risks
        self.assertEqual(carry.frame.tolist(),[2]*4+[1]*4)
        self.assertEqual(carry.counts[:4,r.Branch.PRE_MOVE-1].tolist(),[2]*4)
        self.assertEqual(carry.counts[:4,r.Branch.FRAME_END-1].tolist(),[1]*4)
        self.assertEqual(carry.risk[j.array([0,2,3])].tolist(),[0,0,0])
        self.assertTrue(int(carry.risk[1])&r.NON_WEATHER)
        self.assertTrue(int(frame.callbacks.refusal[1])&accounting.UNKNOWN_KERNEL_RISK)
        self.assertFalse(bool((frame.callbacks.refusal[j.array([0,2,3])]&accounting.UNKNOWN_KERNEL_RISK).any()))
        # Row2 executed and observed the kernel, although the actual second
        # owner rejects its public sequence. Neither observer may commit it.
        self.same(carry,frame.callbacks.execution.risks.carry)
        self.assertTrue(int(result.state.core.execution.refusal[2])&accounting.CONSUMPTION)
        for old,new in zip(before.observers,result.state.observers):
            self.assertEqual(new.retained_healthy.tolist(),[True,False,False,True,True,True,True,True])
            for row in (1,2):
                chosen=j.arange(8)==row
                self.same(d._public_rows(~chosen,old.retained,new.retained),new.retained)
                self.same(jax.tree.map(lambda x:x[row],old.projection),jax.tree.map(lambda x:x[row],new.projection))
            for row in (0,3):
                self.assertGreater(int(new.retained.cursor.sequence[row]),int(old.retained.cursor.sequence[row]))
        self.assert_risk_rows_same(before.core.risks,carry,~active)
        for old,new in zip(before.observers,result.state.observers):
            self.same(d._observer_rows(active,old,new),new)
        for report in result.owners:
            self.assertFalse(bool(report.scored.any()));self.assertFalse(bool(report.B4_qualified.any()))

        # Real next invocation: refused rows cannot resume or erase risk;
        # active safe rows switch, while all other rows stay bitwise frozen.
        prior=result.state
        next_active=j.array([True,True,True,False,False,False,False,False])
        later=d.advance(session,j.full((8,2),4,j.int32),active_rows=next_active,chance_inputs=chance)
        self.assertEqual(d.latest_diagnostics(session).executed_rows.tolist(),[True,False,False,False,False,False,False,False])
        self.assertEqual(later.state.core.risks.frame.tolist(),[3,2,2,2,1,1,1,1])
        self.assert_risk_rows_same(prior.core.risks,later.state.core.risks,j.arange(8)!=0)
        for name in ('hp','keys','active','turn'):
            self.same(getattr(prior.core.engine,name)[1:],getattr(later.state.core.engine,name)[1:])
        for report in later.owners:
            self.assertFalse(bool(report.scored.any()));self.assertFalse(bool(report.B4_qualified.any()))


if __name__=='__main__':unittest.main()
