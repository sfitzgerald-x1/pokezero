"""Actual owning weather/heal continuation, without model/game qualification."""
import unittest
import jax.numpy as j
import numpy as np
from gpu_engine import engine_retained_encoder_driver as driver, engine_leftovers_callbacks as leftovers
import test_gpu_engine_weather_hp_runtime as fixture


class RegisteredLeftoversRuntimeTests(unittest.TestCase):
    construct=fixture.WeatherHPRuntimeTests.construct
    advance=fixture.WeatherHPRuntimeTests.advance

    @classmethod
    def setUpClass(cls):
        fixture.WeatherHPRuntimeTests.setUpClass();cls.f=fixture.WeatherHPRuntimeTests.f
        base=driver._checked(fixture.WeatherHPRuntimeTests.prepared)
        cls.heal=leftovers.compile_leftovers(cls.f.metadata,engine_catalog=cls.f.producer)
        cls.prepared=driver.prepare(base['request_compiled'],base['own_compiled'],
            fake_forward=lambda values:driver.FakeScores(j.zeros(9),j.array(1,j.int32)),
            public_mode='retained-full',leftovers_compiled=cls.heal,leftovers_mode='retained')

    def check_final_public_hp(self,result):
        state=result.state.core.engine
        for owner,observer in enumerate(result.state.observers):
            self.assertEqual(observer.retained_healthy.tolist(),[True,True])
            for row in range(2):
                for side in (0,1):
                    physical=int(state.active[row,side]);slot=int(observer.retained.mons.active[row,side])
                    hp=int(state.hp[row,side,physical]);maximum=int(state.max_hp[row,side,physical])
                    visible=hp if owner==side else (100*hp+maximum-1)//maximum
                    if owner!=side and hp<maximum and visible==100:visible=99
                    self.assertEqual(int(observer.retained.mons.hp_numerator[row,side,slot]),visible)
                    self.assertEqual(int(observer.retained.mons.hp_denominator[row,side,slot]),maximum if owner==side else 100)

    def test_actual_weather_heal_and_next_turn_keep_both_public_owners_current(self):
        session=self.construct()
        first,frame=self.advance(session)
        self.assertEqual(first.state.core.admitted.tolist(),[True,True])
        self.assertEqual(frame.callbacks.refusal.tolist(),[1,1])
        for row in range(2):
            words=np.array(frame.callbacks.words[row,:int(frame.callbacks.count[row])])
            heals=words[words[:,0]==17];self.assertEqual(heals[:,3].tolist(),[1,0])
            self.assertEqual(heals[:,4].tolist(),[5,0])
            for heal in heals:
                chip=next(w for w in words if w[0]==16 and w[3]==heal[3])
                self.assertEqual(int(heal[5]),int(chip[6]))
        self.check_final_public_hp(first)
        # A real subsequent owning transition starts from healed physical HP,
        # and the accepted public history must start from that SAME visible HP.
        second=driver.advance(session,j.full((2,2),0,j.int32),chance_inputs=self.f().chance())
        self.assertEqual(second.state.core.admitted.tolist(),[True,True])
        self.check_final_public_hp(second)
        for result in (first,second):
            for owner in result.owners:
                self.assertFalse(bool(owner.scored.any()))
                self.assertFalse(bool(owner.B4_qualified.any()))

    def test_actual_tie_refuses_entire_heal_frame_without_public_commit(self):
        session=self.construct(tie=True);before=session.state
        result,frame=self.advance(session)
        self.assertTrue(bool(((frame.callbacks.refusal&256)!=0).all()))
        self.assertEqual(result.state.core.admitted.tolist(),[False,False])
        for old,new in zip(before.observers,result.state.observers):self.f().same(old.retained,new.retained)


if __name__=='__main__':unittest.main()
