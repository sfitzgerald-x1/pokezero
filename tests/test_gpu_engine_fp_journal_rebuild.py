import unittest

from gpu_engine import fp_journal_rebuild as jr


class LineTests(unittest.TestCase):
    def test_public_condition_is_showdowns_percent(self):
        self.assertEqual(jr.public_condition('219/250 tox'), '88/100 tox')
        self.assertEqual(jr.public_condition('299/300'), '99/100')
        self.assertEqual((jr.public_condition('0 fnt'), jr.public_condition('300/300 par')), ('0 fnt', '100/100 par'))

    def test_comparable_drops_noise_and_shows_foul_plays_hp_as_our_seat_saw_it(self):
        env = ['|t:|179', '|', '|request|{}', '|switch|p2a: Lapras|Lapras, L78, M|300/300', '|-ability|p2a: Raikou|Pressure|[silent]',
               '|move|p1a: Wailord|Toxic|p2a: Lapras', '|-damage|p2a: Lapras|282/300 tox|[from] psn', '|-damage|p1a: Wailord|400/438',
               '|inactive|Battle timer is ON', '|-hint|In Gen 3, Intimidate does not activate', '|-sidestart|p1: PokeZero p1|Spikes', '|upkeep', '|turn|2']
        self.assertEqual(jr.comparable(env, 'p2'), ['|switch|p2a: Lapras|Lapras, L78, M|100/100', '|move|p1a: Wailord|Toxic|p2a: Lapras',
                                                     '|-damage|p2a: Lapras|94/100 tox|[from] psn', '|-damage|p1a: Wailord|400/438', '|-sidestart|p1|Spikes',
                                                     '|upkeep', '|turn|2'])
        self.assertEqual(jr.comparable(['|-sidestart|p1: PokeZeroBot|Spikes'], 'p2'), jr.comparable(['|-sidestart|p1: PokeZero p1|Spikes'], 'p2'))
        self.assertEqual(jr.comparable(['|-activate|p1a: Lugia|move: Struggle', '|move|p1a: Lugia|Struggle|p2a: X'], 'p2'), ['|move|p1a: Lugia|Struggle|p2a: X'])

    def test_implied_choice(self):
        self.assertEqual(jr.implied_choice(['|switch|p1a: Wailord|Wailord|438/438', '|move|p2a: Lapras|Ice Beam|p1a: Wailord'], 'p2'), 'move icebeam')
        self.assertEqual(jr.implied_choice(['|switch|p2a: Ho-Oh|Ho-Oh, L70|100/100'], 'p2'), 'switch hooh')
        self.assertEqual(jr.implied_choice(['|cant|p2a: Snorlax|move: Taunt|Curse'], 'p2'), 'move curse')
        self.assertIsNone(jr.implied_choice(['|cant|p2a: Snorlax|par', '|turn|5', '|move|p2a: Snorlax|Rest|p2a: Snorlax'], 'p2'))
        self.assertEqual(jr.implied_choice(['|move|p2a: Gengar|Hidden Power|p1a: X', '|move|p2a: Gengar|Thunderbolt|p1a: X|[from]Sleep Talk'],
                                           'p2'), 'move hiddenpower')

    def test_classify(self):
        c = lambda lines, forced=False: jr.classify(lines, 'p2', forced)
        self.assertEqual(c(['|move|p2a: Snorlax|Body Slam|p1a: X']), ('chosen', 'move', ('move', 'bodyslam')))
        self.assertEqual(c(['|cant|p2a: Snorlax|slp', '|move|p2a: Snorlax|Sleep Talk|p2a: Snorlax',
                            '|move|p2a: Snorlax|Body Slam|p1a: X|[from]Sleep Talk']), ('chosen', 'move', ('move', 'sleeptalk')))
        self.assertEqual(c(['|cant|p2a: Snorlax|par']), ('fill', 'par', None))
        self.assertEqual(c(['|move|p1a: X|Earthquake|p2a: Raikou', '|faint|p2a: Raikou']), ('fill', 'fainted_before_moving', None))
        self.assertEqual(c(['|cant|p2a: Slaking|ability: Truant']), ('fill', 'abilitytruant', None))
        self.assertEqual(c(['|move|p1a: X|Roar|p2a: Raikou', '|drag|p2a: Ho-Oh|Ho-Oh, L70|100/100']), ('fill', 'dragged', None))
        self.assertEqual(c(['|cant|p2a: Snorlax|move: Taunt|Curse']), ('chosen', 'cant_named', ('move', 'curse')))
        self.assertEqual(c(['|switch|p2a: Ho-Oh|Ho-Oh, L70|100/100']), ('chosen', 'switch', ('switch', 'hooh')))
        self.assertEqual(c(['|faint|p1a: X', '|switch|p2a: Ho-Oh|Ho-Oh, L70|100/100'], True), ('chosen', 'replacement', ('switch', 'hooh')))
        self.assertEqual(c(['|move|p1a: X|Baton Pass|p1a: X']), ('fill', 'no_public_action', None))


if __name__ == '__main__':
    unittest.main()
