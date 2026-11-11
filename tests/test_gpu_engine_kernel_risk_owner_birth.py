"""Real owning risk carry and both full folds; tiny fixtures, no forwards/games.

Light replacement coverage for kernel_risk_combined_runtime, birth half: the real kernel-risk and plain
constructors at B=8, their engines and both owners' public folds identical, and the constructor's risk carry.
The advance half is test_gpu_engine_kernel_risk_owner_rollback.
"""
import unittest
import jax
import jax.numpy as j
import numpy as np
from gpu_engine import engine_retained_encoder_driver as d, engine_kernel_frame_risk as r
from gpu_engine import engine_execution_completeness as accounting, transition as t
import test_gpu_engine_kernel_risk_runtime as fixture


class KernelRiskOwnerBirthTests(unittest.TestCase):
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

    def test_genuine_constructor_birth_identity_and_risk_carry(self):
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


if __name__=='__main__':unittest.main()
