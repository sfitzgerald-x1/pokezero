"""Pure visible-own fact transport only; no world, model or game execution."""
import unittest
from gpu_engine import phase_d_live_worlds as live


class VisibleOwnFacts(unittest.TestCase):
    def test_iv_resolved_hidden_power_uses_device_move_namespace(self):
        names=('', 'tackle','hiddenpower','hiddenpowerflying','hiddenpowerfire')
        self.assertEqual(live._own_move(names,'hiddenpowerflying','hiddenpowerflying70'),3)
        self.assertEqual(live._own_move(names,'hiddenpower','hiddenpowerfire60'),4)
        self.assertEqual(live._own_move(names,'hiddenpowerfire60','hiddenpowerfire60'),4)
        self.assertEqual(live._own_move(('', 'return'),'return102',None),1)
        with self.assertRaisesRegex(live.Refusal,'NAMESPACE_UNAVAILABLE'):
            live._own_move(names,'hiddenpowerice','hiddenpowerice70')
        for literal,resolved in (('hiddenpowerice','hiddenpowerfire70'),('hiddenpowerfire60','hiddenpowerfire70'),
                ('hiddenpower',None)):
            with self.subTest(literal=literal,resolved=resolved),self.assertRaisesRegex(live.Refusal,'OWN_HIDDEN_POWER_LITERAL_DIFFERS'):
                live._own_move(names,literal,resolved)

    def test_literal_max_power_uniquely_proves_own_happiness(self):
        self.assertEqual(live._own_happiness({},('return102',),'own'),255)
        self.assertEqual(live._own_happiness({},('frustration102',),'own'),0)
        self.assertEqual(live._own_happiness({},('return102','frustration1'),'own'),255)
        self.assertIsNone(live._own_happiness({},('tackle',),'own'))

    def test_ambiguous_missing_and_inconsistent_facts_remain_refused(self):
        for moves in (('return',),('frustration',),('return101',),('return1',)):
            with self.subTest(moves=moves),self.assertRaisesRegex(live.Refusal,'OWN_HAPPINESS_UNAVAILABLE'):
                live._own_happiness({},moves,'own')
        for mon,moves in (({'happiness':True},('return102',)),({'happiness':256},()),
                ({'happiness':0},('return102',)),({},('return103',)),({},('return102','frustration102'))):
            with self.subTest(mon=mon,moves=moves),self.assertRaisesRegex(live.Refusal,'OWN_HAPPINESS_INVALID'):
                live._own_happiness(mon,moves,'own')

    def test_explicit_happiness_is_preserved_and_checked_against_literal_power(self):
        for happiness in range(256):
            power=max(1,happiness*10//25)
            self.assertEqual(live._own_happiness({'happiness':happiness},('return'+str(power),),'own'),happiness)
        self.assertEqual(live._own_happiness({'happiness':87},('return',),'own'),87)


if __name__=='__main__':unittest.main()
