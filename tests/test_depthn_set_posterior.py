"""set_posterior, stage 1. Pure Python: the pinned Showdown dex and gen3 randbats source, no engine, JAX or Showdown process.

Fixtures (tests/fixtures/set_posterior) are real screen r5a roots: the search request prepare() saw, rebuilt from the owning
wire (wallslib.Wire(run, ordinal).request(decision)), its prefix sha256, the root's opponent belief hard facts (capture_root's
belief view), the r5a census's anchors for that decision and a snapshot of this posterior. Edits of their lines cover the
classes r5a never exercised. Checks: Choice Band collapse and exclusion, a crit, KO censoring, Hidden Power type settled by
damage with the effectiveness line as a consistency refusal, every refusal reason, the off-model drop (in the posterior and
at the sampler), the riskiest false-exclusion paths (a Pursuit intercept, Revenge, a mid-turn replacement), single-variant
supports, draw's byte identity with today's uniform call, the two-way check and provenance. In depthn-light (no skips)."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import random
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from _showdown_root import requires_showdown, showdown_root_str
from pokezero import determinization as D, set_posterior as SP, tier2
from pokezero.belief import PlayerBeliefView, RevealedPokemonBelief
from pokezero.dex import load_showdown_dex_cached, normalize_id
from pokezero.randbat import load_gen3_randbat_source_cached

FIXTURES = Path(__file__).parent / 'fixtures' / 'set_posterior'


def fixture(name):
    data = json.loads((FIXTURES / (name + '.json')).read_text())
    lines = data['request']['lines']
    if hashlib.sha256(''.join(x + '\n' for x in lines).encode()).hexdigest() != data['prefix_sha256']:
        raise ValueError('fixture prefix changed')
    return data


def beliefs(data):
    return tuple(RevealedPokemonBelief(**{k: tuple(v) if isinstance(v, list) else v for k, v in row.items()}) for row in data['beliefs'])


@requires_showdown('set_posterior needs the pinned gen3 randbats source and dex')
class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dex = load_showdown_dex_cached(showdown_root_str())
        cls.src = load_gen3_randbat_source_cached(showdown_root_str())

    def post(self, data, lines=None, own_request=None, **kwargs):
        self.assertEqual(data['source_hash'], self.src.metadata.source_hash)
        return SP.posterior(data['request']['lines'] if lines is None else lines, own_request, source=self.src, owner=data['owner'],
                            beliefs=beliefs(data), format_id=data['format_id'], dex=self.dex, **kwargs)

    def variants(self, key):
        return self.src.universe_for(key).variants

    @staticmethod
    def hard(post):
        return [i for i, h in enumerate(post.hard) if h]

    @staticmethod
    def alive(post):
        return [i for i, w in enumerate(post.weights) if w]

    def strike_at(self, post, line):
        return next(s for s in post.strikes if s.line == line)


class Edits(Base):
    """cb_excluded's one admitted strike (a Snorlax Body Slam on our active), edited line by line."""

    def setUp(self):
        self.data = fixture('cb_excluded')
        self.lines = list(self.data['request']['lines'])
        self.key = 'snorlax'
        self.at = next(s['line'] for s in self.data['census']['strikes'] if s['reason'] is None)
        parts = self.lines[self.at].split('|')
        self.foe_ident, self.me_ident = parts[2], parts[4]
        self.me = self.data['owner']
        self.damage = next(i for i in range(self.at + 1, len(self.lines))
                           if self.lines[i].startswith('|-damage|' + self.me_ident))

    def reason(self, lines, at=None):
        post = self.post(self.data, lines)[self.key]
        return self.strike_at(post, self.at if at is None else at), post

    def inserted(self, *new, before=None):
        before = self.at if before is None else before
        return self.lines[:before] + list(new) + self.lines[before:], self.at + len(new)

    def moved(self, move, flags=''):
        lines = list(self.lines)
        lines[self.at] = f'|move|{self.foe_ident}|{move}|{self.me_ident}' + flags
        return lines


class FixtureAnchors(Base):
    def test_every_fixture_matches_its_snapshot_and_the_r5a_census(self):
        names = sorted(p.stem for p in FIXTURES.glob('*.json'))
        self.assertGreaterEqual(len(names), 16)
        for name in names:
            with self.subTest(fixture=name):
                data = fixture(name)
                posts = self.post(data)
                self.assertEqual(list(posts), [SP.species_key(b['species']) for b in data['beliefs']])
                for key, post in posts.items():
                    snap, census = data['posterior'][key], data['census']['species'][key]
                    # The snapshot: hard support, weights, every strike and its counts, provenance.
                    self.assertEqual(self.hard(post), snap['hard'])
                    self.assertEqual(list(post.weights), snap['weights'])
                    self.assertEqual([dict(line=s.line, move=s.move, admitted=s.admitted, reason=s.reason,
                                           counts=None if s.counts is None else list(s.counts), observed=s.observed, ko=s.ko, crit=s.crit)
                                      for s in post.strikes], snap['strikes'])
                    self.assertEqual(post.provenance, snap['provenance'])
                    # The census: the worlds path's survivors, the variants left alive, the true set kept, proportional weights.
                    ids = [v.variant_id for v in self.variants(key)]
                    self.assertEqual([ids[i] for i in self.hard(post)], census['survivor_ids'])
                    if census['census_n_after'] is not None:
                        self.assertEqual(len(self.alive(post)), census['census_n_after'])
                    if census['true_variant'] is not None:
                        self.assertGreater(post.weights[ids.index(census['true_variant'])], 0)
                    lik = census['census_lik']
                    if lik is not None and max(lik) > 1e-6:
                        top = max(post.weights)
                        for i, x in zip(self.hard(post), lik):
                            self.assertAlmostEqual(post.weights[i] / top, x / max(lik), places=6)
                # Every census strike in the prefix: the same refusal (None: admitted), the true set's fitting rolls.
                for c in data['census']['strikes']:
                    key = SP.species_key(c['attacker'])
                    mine = self.strike_at(posts[key], c['line'])
                    if c['reason'] is None and sum(posts[key].hard) == 1:   # one variant: classified, rolls not computed
                        self.assertEqual((mine.reason, mine.counts), (SP.SINGLE_VARIANT, None), c)
                        continue
                    self.assertEqual(mine.reason or None, c['reason'], c)
                    if c['reason'] is None and c['true_match'] is not None:
                        ids = [v.variant_id for v in self.variants(key)]
                        self.assertEqual(mine.counts[ids.index(c['true_variant'])], c['true_match'])


class Narrowing(Base):
    def test_a_hit_only_choice_band_explains_collapses_to_it(self):
        post = self.post(fixture('cb_collapse'))['rhydon']
        strike = post.strikes[0]
        self.assertTrue(strike.admitted and strike.ko and strike.crit)
        items = {i: normalize_id(self.variants('rhydon')[i].item) for i in self.hard(post)}
        self.assertEqual(sorted(set(items.values())), ['choiceband', 'leftovers'])
        self.assertEqual(self.alive(post), [i for i, item in items.items() if item == 'choiceband'])
        self.assertEqual(post.support, tuple(w > 0 for w in post.weights))

    def test_a_hit_below_every_choice_band_roll_excludes_it(self):
        post = self.post(fixture('cb_excluded'))['snorlax']
        items = {i: normalize_id(self.variants('snorlax')[i].item) for i in self.hard(post)}
        self.assertIn('choiceband', items.values())
        self.assertEqual(self.alive(post), [i for i, item in items.items() if item != 'choiceband'])
        self.assertEqual({post.weights[i] for i in self.alive(post)}, {1})   # one fitting roll each, gcd-reduced

    def test_a_crit_is_a_known_condition(self):
        data = fixture('crit')
        key, = data['posterior']
        line = next(s['line'] for s in data['census']['strikes'] if s['crit'])
        post = self.post(data)[key]
        self.assertTrue(self.strike_at(post, line).crit and self.strike_at(post, line).admitted)
        self.assertLess(len(self.alive(post)), len(self.hard(post)))
        # Without its -crit line the same damage fits no non-crit roll of any survivor: off-model, dropped, support kept.
        lines = [x for x in data['request']['lines'] if not x.startswith('|-crit|')]
        dropped = self.post(data, lines)[key]
        self.assertEqual(self.strike_at(dropped, line).reason, SP.OFF_MODEL)
        self.assertEqual(self.alive(dropped), self.hard(dropped))

    def test_a_ko_is_censored(self):
        data = fixture('ko_censored')
        key, = data['posterior']
        census = next(s for s in data['census']['strikes'] if s['ko'])
        post = self.post(data)[key]
        strike = self.strike_at(post, census['line'])
        self.assertTrue(strike.admitted and strike.ko)
        true = [v.variant_id for v in self.variants(key)].index(census['true_variant'])
        self.assertEqual(strike.counts[true], census['true_match'])
        self.assertGreater(strike.counts[true], 2)   # censored: every roll at or above our HP fits, not one exact value
        self.assertEqual(strike.observed, census['hp_before'])

    def test_damage_settles_hidden_power_type_and_the_effectiveness_line_is_only_a_consistency_refusal(self):
        for name, observed in (('hp_effectiveness', 'supereffective'), ('hp_neutral_unown', None)):
            with self.subTest(fixture=name):
                data = fixture(name)
                key, = data['posterior']
                post = self.post(data)[key]
                strike, = post.strikes
                self.assertTrue(strike.admitted);self.assertEqual(strike.move, 'hiddenpower')
                self.assertEqual(next(c['eff'] for c in data['census']['strikes'] if c['line'] == strike.line), observed)
                defender = data['request']['lines'][strike.line].split('|')[4].split(': ', 1)[1]
                types = tuple(self.dex.species_info(SP.species_key(defender)).types)

                def line_of(i):
                    hp = next(m for m in map(normalize_id, self.variants(key)[i].moves) if m.startswith('hiddenpower'))
                    value = self.dex.effectiveness(self.dex.move_info(hp).type, types)
                    return 'supereffective' if value > 1 else 'resisted' if 0 < value < 1 else None
                # Every survivor's Hidden Power type prints the strike's line; every type that cannot is excluded.
                self.assertTrue(all(line_of(i) == observed for i in self.alive(post)))
                cannot = [i for i in self.hard(post) if line_of(i) != observed]
                self.assertTrue(cannot);self.assertFalse(set(cannot) & set(self.alive(post)))
                # The line itself excludes nothing: rewrite it and the survivors whose rolls fit could not have printed it,
                # so the strike is refused (L = 1) and the whole hard support comes back.
                lines = list(data['request']['lines'])
                if observed is None:
                    lines.insert(strike.line + 1, '|-supereffective|' + lines[strike.line].split('|')[4])
                else:
                    lines = [x for x in lines if not x.startswith('|-supereffective|')]
                refused = self.post(data, lines)[key]
                self.assertEqual(self.strike_at(refused, strike.line).reason, 'effectiveness-disagreement')
                self.assertEqual(self.alive(refused), self.hard(refused))

    def test_a_type_immune_hidden_power_variant_fits_zero_and_the_strike_stays_admitted(self):
        # r5a: an Umbreon Hidden Power on Butterfree and a Kingler one on Snorlax. One hard variant's type cannot touch our mon
        # (the core returns no rolls); damage was dealt, so it fits 0 and the strike is admitted, the true set kept.
        for name, key in (('hp_immune_umbreon', 'umbreon'), ('hp_immune_kingler', 'kingler')):
            with self.subTest(fixture=name):
                data = fixture(name)
                post = self.post(data)[key]
                strike, = post.strikes
                census = next(c for c in data['census']['strikes'] if c['line'] == strike.line)
                self.assertTrue(strike.admitted);self.assertIsNone(census['reason'])
                true = [v.variant_id for v in self.variants(key)].index(census['true_variant'])
                self.assertEqual(strike.counts[true], census['true_match']);self.assertGreater(post.weights[true], 0)
                defender = data['request']['lines'][strike.line].split('|')[4].split(': ', 1)[1]
                types = tuple(self.dex.species_info(SP.species_key(defender)).types)
                hp = lambda i: next(m for m in map(normalize_id, self.variants(key)[i].moves) if m.startswith('hiddenpower'))
                immune = [i for i in self.hard(post) if self.dex.effectiveness(self.dex.move_info(hp(i)).type, types) == 0]
                self.assertTrue(immune);self.assertEqual([strike.counts[i] for i in immune], [0] * len(immune))

    def test_only_type_immunity_turns_empty_rolls_into_zero(self):
        original = tier2._strike_damage
        empty = lambda **kw: replace(original(**kw), rolls=())
        # Empty rolls on a move that does touch our mon (a Body Slam): another cause, still refused.
        with patch.object(tier2, '_strike_damage', side_effect=empty):
            post = self.post(fixture('cb_excluded'))['snorlax']
        self.assertEqual(post.strikes[0].reason, 'core-unsupported:bodyslam')
        # Every hard variant immune: every count 0, so the off-model drop keeps the whole hard support.
        data = fixture('hp_immune_umbreon')
        with patch.object(tier2, '_strike_damage', side_effect=empty), patch.object(SP, '_effectiveness', return_value=0.0):
            post = self.post(data)['umbreon']
        self.assertEqual(post.strikes[0].reason, SP.OFF_MODEL);self.assertEqual(self.alive(post), self.hard(post))

    def test_an_effectiveness_line_a_fitting_variant_cannot_print_refuses_the_strike(self):
        # The KO is censored above every Hidden Power type's rolls on four survivors; drop its -supereffective line and a
        # survivor whose rolls still fit could not have printed what is left: refused (L = 1), never trusted either way.
        data = fixture('hp_crit_ko_effectiveness')
        key, = data['posterior']
        post = self.post(data)[key]
        line = post.strikes[0].line
        self.assertTrue(post.strikes[0].admitted and post.strikes[0].ko)
        lines = [x for x in data['request']['lines'] if not x.startswith('|-supereffective|')]
        refused = self.post(data, lines)[key]
        self.assertEqual(self.strike_at(refused, line).reason, 'effectiveness-disagreement')
        self.assertEqual(self.alive(refused), self.hard(refused))


class Refusals(Edits):
    def test_the_census_refusals_on_real_roots(self):
        expected = dict(refusal_hit_substitute='hit-substitute', refusal_fixed_damage='fixed-damage', refusal_called='called',
                        refusal_trace='ability-overridden', refusal_type_change='type-change', refusal_transform='transform',
                        refusal_item_changed='item-changed', refusal_ambiguous_pinch='ambiguous-band', refusal_ambiguous_flail='ambiguous-band',
                        refusal_endure='endure-or-focusband',
                        refusal_multi_hit='multi-hit')
        for name, reason in expected.items():
            with self.subTest(fixture=name):
                data = fixture(name)
                posts = self.post(data)
                self.assertIn(reason, [s.reason for post in posts.values() for s in post.strikes])
                for post in posts.values():
                    for s in post.strikes:
                        self.assertEqual(s.admitted, s.reason == '')
                        self.assertEqual(s.counts is None, not s.admitted)

    def test_refused_moves(self):
        for move, flags, reason in (('Struggle', '', 'struggle'), ('Seismic Toss', '', 'fixed-damage'), ('Super Fang', '', 'fixed-damage'),
                                    ('Horn Drill', '', 'ohko'), ('Magnitude', '', 'conditional-power:magnitude'),
                                    ('Frustration', '', 'conditional-power:frustration'), ('Rollout', '', 'conditional-power:rollout'),
                                    ('Double Kick', '', 'multi-hit'), ('Body Slam', '|[from]Sleep Talk', 'called'),
                                    ('Body Slam', '|[from]lockedmove', 'called')):
            with self.subTest(move=move, flags=flags):
                strike, post = self.reason(self.moved(move, flags))
                self.assertEqual(strike.reason, reason)
                self.assertFalse(strike.admitted);self.assertIsNone(strike.counts)
                self.assertEqual(self.alive(post), self.hard(post))

    def test_eruption_below_full_hp_and_revenge_after_our_hit(self):
        lines, at = self.inserted(f'|-damage|{self.foe_ident}|50/100')
        lines[at] = f'|move|{self.foe_ident}|Eruption|{self.me_ident}'
        self.assertEqual(self.reason(lines, at)[0].reason, 'conditional-power:eruption')
        lines, at = self.inserted(f'|move|{self.me_ident}|Tackle|{self.foe_ident}', f'|-damage|{self.foe_ident}|90/100')
        lines[at] = f'|move|{self.foe_ident}|Revenge|{self.me_ident}'
        self.assertEqual(self.reason(lines, at)[0].reason, 'conditional-power:revenge')

    def test_public_states_the_core_does_not_model_or_r5a_never_exercised(self):
        side = self.me + ': Owner'
        foe_side = ('p2' if self.me == 'p1' else 'p1') + ': Foe'
        for new, reason in (((f'|-start|{self.foe_ident}|Charge',), 'unmodelled:charge'),
                            ((f'|-start|{self.me_ident}|Mud Sport',), 'unmodelled:mud/water-sport'),
                            ((f'|-start|{self.foe_ident}|move: Water Sport',), 'unmodelled:mud/water-sport'),
                            ((f'|-prepare|{self.me_ident}|Dig',), 'defender-semi-invulnerable'),
                            ((f'|-sidestart|{side}|Reflect',), 'unvalidated:screen'),
                            ((f'|-sidestart|{side}|move: Light Screen',), 'unvalidated:screen'),
                            ((f'|-transform|{self.foe_ident}|{self.me_ident}',), 'transform'),
                            ((f'|-start|{self.me_ident}|typechange|Fire|[from] ability: Color Change',), 'type-change'),
                            ((f'|-enditem|{self.foe_ident}|Leftovers|[from] move: Knock Off|[of] {self.me_ident}',), 'item-changed'),
                            ((f'|-item|{self.me_ident}|Choice Band|[from] move: Trick',), 'item-changed'),
                            ((f'|-ability|{self.foe_ident}|Pressure|[from] ability: Trace|[of] {self.me_ident}',), 'ability-overridden')):
            with self.subTest(reason=reason, lines=new):
                lines, at = self.inserted(*new)
                self.assertEqual(self.reason(lines, at)[0].reason, reason)
        # A screen on the attacker's side, or one that ended, changes nothing.
        lines, at = self.inserted(f'|-sidestart|{foe_side}|Reflect')
        self.assertTrue(self.reason(lines, at)[0].admitted)
        lines, at = self.inserted(f'|-sidestart|{side}|Reflect', f'|-sideend|{side}|Reflect')
        self.assertTrue(self.reason(lines, at)[0].admitted)

    def test_window_refusals(self):
        damage = self.lines[self.damage]
        for window, reason in (((f'|-activate|{self.me_ident}|Substitute|[damage]',), 'hit-substitute'),
                               ((f'|-activate|{self.me_ident}|move: Substitute|[damage]',), 'hit-substitute'),
                               ((f'|-end|{self.me_ident}|Substitute',), 'hit-substitute'),
                               ((f'|-activate|{self.me_ident}|move: Endure', damage), 'endure-or-focusband'),
                               ((f'|-activate|{self.me_ident}|item: Focus Band', damage), 'endure-or-focusband'),
                               ((damage, damage), 'multi-hit'),
                               ((damage, f'|-hitcount|{self.me_ident}|2'), 'multi-hit')):
            with self.subTest(reason=reason, window=window):
                lines = self.lines[:self.damage] + list(window) + self.lines[self.damage + 1:]
                self.assertEqual(self.reason(lines)[0].reason, reason)

    def test_unobservable_damage_and_defender(self):
        damage = self.lines[self.damage].split('|')
        census = next(s for s in self.data['census']['strikes'] if s['line'] == self.at)
        before = f"{census['hp_before']}/{damage[3].split('/', 1)[1]}"
        lines = list(self.lines);lines[self.damage] = '|'.join(damage[:3] + [before] + damage[4:])
        self.assertEqual(self.reason(lines)[0].reason, 'no-hp-drop')
        # Lines that do not end at a decision boundary (a request) leave the last window unclassified.
        self.assertEqual(self.reason(self.lines[:self.damage + 1])[0].reason, 'window-open')
        # Our team comes from the lines' first request, else from own_request; with neither our mon is unknown.
        lines = [('|request|{"wait":true}' if x.startswith('|request|') else x) for x in self.lines]
        self.assertEqual(self.reason(lines)[0].reason, 'unknown-defender')
        first = next(json.loads(x[len('|request|'):]) for x in self.lines if x.startswith('|request|'))
        self.assertEqual(self.strike_at(self.post(self.data, lines, first)[self.key], self.at), self.reason(self.lines)[0])
        # Our switch line's maximum must be the request's.
        switch = next(i for i in range(self.at) if self.lines[i].startswith(('|switch|' + self.me_ident, '|drag|' + self.me_ident)))
        shown = self.lines[switch].split('|')[4].split()[0]
        cur, top = shown.split('/')
        lines = list(self.lines);lines[switch] = self.lines[switch].replace(shown, f'{cur}/{int(top) + 1}')
        self.assertEqual(self.reason(lines)[0].reason, 'defender-maxhp-disagrees')

    def test_off_model_strike_is_dropped_and_never_empties_the_support(self):
        # One HP of damage fits no survivor's rolls: the strike is dropped and recorded, the hard support stays whole.
        damage = self.lines[self.damage].split('|')
        census = next(s for s in self.data['census']['strikes'] if s['line'] == self.at)
        after = f"{census['hp_before'] - 1}/{damage[3].split('/', 1)[1]}"
        lines = list(self.lines);lines[self.damage] = '|'.join(damage[:3] + [after] + damage[4:])
        strike, post = self.reason(lines)
        self.assertEqual((strike.admitted, strike.reason, strike.counts), (False, SP.OFF_MODEL, None))
        self.assertEqual(strike.observed, 1)
        self.assertEqual(self.alive(post), self.hard(post));self.assertEqual(post.support, post.hard)
        self.assertEqual({post.weights[i] for i in self.hard(post)}, {1})


class RiskyPaths(Edits):
    """The paths most likely to exclude a true set by mistake; each keeps it."""

    def true_index(self, data, key):
        return [v.variant_id for v in self.variants(key)].index(data['census']['species'][key]['true_variant'])

    def test_a_pursuit_intercept_on_a_switch_doubles_and_keeps_the_true_set(self):
        data = fixture('pursuit_switch')   # r5a: Houndoom's Pursuit on our switching Alakazam, two hard variants
        census = next(c for c in data['census']['strikes'] if c['move'] == 'pursuit')
        lines = list(data['request']['lines'])
        line = census['line']
        self.assertTrue(lines[line - 1].endswith('|move: Pursuit') and lines[line].endswith('|[from] Pursuit'))
        true = self.true_index(data, 'houndoom')
        strike = self.strike_at(self.post(data)['houndoom'], line)
        self.assertTrue(strike.admitted);self.assertEqual(strike.counts[true], census['true_match'])
        # The -activate lookback alone marks the intercept: the same strike without the move line's [from] Pursuit.
        bare = list(lines);bare[line] = lines[line].replace('|[from] Pursuit', '')
        self.assertEqual(self.strike_at(self.post(data, bare)['houndoom'], line), strike)
        # Without the intercept (base power 40) the damage fits no survivor: dropped, and the true set kept.
        plain = bare[:line - 1] + bare[line:]
        dropped = self.post(data, plain)['houndoom']
        self.assertEqual(self.strike_at(dropped, line - 1).reason, SP.OFF_MODEL)
        self.assertGreater(dropped.weights[true], 0)

    def test_revenge_is_in_no_gen3_set_so_an_unhit_revenge_is_refused_and_keeps_the_true_set(self):
        self.assertFalse(any('revenge' in {normalize_id(m) for m in v.moves} for u in self.src.universes.values() for v in u.variants))
        strike, post = self.reason(self.moved('Revenge'))
        self.assertEqual(strike.reason, 'core-unsupported:revenge')
        self.assertGreater(post.weights[self.true_index(self.data, 'snorlax')], 0)
        self.assertEqual(self.alive(post), self.hard(post))

    def test_a_strike_on_a_mid_turn_replacement_after_a_faint_reads_the_replacement(self):
        # Gen 3 sends a fainted mon's replacement in at once, mid-turn. Our active explodes and faints, another member comes in,
        # and the Snorlax's Body Slam then hits it: our HP, maximum and typing must be the replacement's.
        request = next(json.loads(x[len('|request|'):]) for x in self.lines if x.startswith('|request|'))
        team = tier2.own_team_from_request(request)
        active = SP.species_key(self.me_ident.split(': ', 1)[1])
        mon = next(m for m in team if SP.species_key(m.species) != active and 'hp' in m.stats)
        ident = f'{self.me}a: {mon.species}'
        true = self.true_index(self.data, 'snorlax')
        attacker = SP._variant_participant(self.src, 'snorlax', self.variants('snorlax')[true])
        types = tuple(self.dex.species_info(SP.species_key(mon.species)).types)
        defender = tier2.StrikeParticipant(species_key=SP.species_key(mon.species), level=mon.level, stats=dict(mon.stats),
                                           ability=normalize_id(mon.ability or ''), item=normalize_id(mon.item or ''), moves=mon.moves)
        rolls = tier2._strike_damage(attacker=attacker, defender=defender, defender_types=types, observed_move='bodyslam',
            token=SimpleNamespace(crit=False, weather='', pursuit_intercept=False), context=tier2.StrikeContext('Snorlax', mon.species),
            dex=self.dex, require_move_in_set=True, pinch_ambiguity_band=0.04, split_choice_band_baseline=False).rolls
        top = mon.stats['hp'];damage = rolls[8];self.assertLess(damage, top)
        lines, at = self.inserted(f'|move|{self.me_ident}|Explosion|{self.foe_ident}', f'|-damage|{self.foe_ident}|40/100',
                                  f'|faint|{self.me_ident}', '|request|{"wait":true}', f'|switch|{ident}|{mon.species}, L{mon.level}|{top}/{top}')
        lines[at] = f'|move|{self.foe_ident}|Body Slam|{ident}'
        hit = next(i for i in range(at + 1, len(lines)) if lines[i].startswith('|-damage|' + self.me_ident))
        lines[hit] = f'|-damage|{ident}|{top - damage}/{top}'
        strike, post = self.reason(lines, at)
        self.assertTrue(strike.admitted, strike.reason);self.assertEqual(strike.observed, damage)
        self.assertGreater(strike.counts[true], 0);self.assertGreater(post.weights[true], 0)

    def test_a_single_variant_support_classifies_its_strikes_without_rolls(self):
        posts = self.post(fixture('refusal_item_changed'))
        marowak = posts['marowak']
        self.assertEqual(sum(marowak.hard), 1)
        self.assertEqual([(s.reason, s.counts) for s in marowak.strikes], [(SP.SINGLE_VARIANT, None)] * 2)
        self.assertEqual(self.alive(marowak), self.hard(marowak))
        self.assertEqual(SP.summary(posts['sableye']), dict(species='sableye', hard=1, weighted=1, admitted=0,
            refused={'fixed-damage': 1, SP.SINGLE_VARIANT: 1}, off_model_drops=0, provenance=posts['sableye'].provenance))
        self.assertEqual({k: v for k, v in SP.summary(self.post(fixture('cb_collapse'))['rhydon']).items() if k != 'provenance'},
                         dict(species='rhydon', hard=4, weighted=1, admitted=1, refused={}, off_model_drops=0))


class Draw(Base):
    def uniform(self, hard, n=12, weight=1):
        flags = tuple(i in hard for i in range(n))
        return SP.EncounterPosterior('p2', 'x', flags, flags, tuple(weight if f else 0 for f in flags), (), (), '')

    def test_equal_weights_make_todays_uniform_call(self):
        survivors = [1, 3, 4, 7, 11]
        for weight in (1, 5, 16 ** 20):
            post = self.uniform(survivors, weight=weight)
            for seed in range(300):
                old, new = random.Random(seed), random.Random(seed)
                self.assertEqual(SP.draw(post, new), survivors[old.randrange(len(survivors))])
                self.assertEqual(new.getstate(), old.getstate())

    def test_unequal_weights_draw_by_cumulative_integer_weight(self):
        flags = tuple(i in (1, 3) for i in range(5))
        post = SP.EncounterPosterior('p2', 'x', flags, flags, (0, 3, 0, 1, 0), (), (), '')
        for seed in range(300):
            pick = random.Random(seed).randrange(4)
            self.assertEqual(SP.draw(post, random.Random(seed)), 1 if pick < 3 else 3)
        with self.assertRaisesRegex(ValueError, SP.SAMPLER_EMPTY):
            SP.draw(replace(post, weights=(0,) * 5), random.Random(0))

    def test_restrict_to_the_samplers_own_survivors(self):
        post = self.uniform([1, 3, 4])
        self.assertIs(SP.restrict(post, [1, 3, 4]), post)
        self.assertEqual(SP.restrict(post, [3, 4]).weights[1], 0)
        with self.assertRaisesRegex(ValueError, SP.SAMPLER_DISAGREEMENT):SP.restrict(post, [1, 2])
        with self.assertRaisesRegex(ValueError, SP.SAMPLER_DISAGREEMENT):SP.restrict(post, [])
        # Off-model at the consumer: survivors holding no weight are drawn uniformly, today's call over them.
        fallback = SP.restrict(replace(post, weights=(0, 7) + (0,) * 10), [3, 4])
        self.assertEqual(fallback.weights, tuple(int(i in (3, 4)) for i in range(12)))
        for seed in range(50):
            old, new = random.Random(seed), random.Random(seed)
            self.assertEqual(SP.draw(fallback, new), [3, 4][old.randrange(2)]);self.assertEqual(new.getstate(), old.getstate())

    def test_the_sampler_draws_its_own_zero_weight_survivors_uniformly(self):
        # A replay filter (move slots, public max HP) that leaves only damage-excluded variants: today's uniform call over
        # those survivors, never a refusal. cb_collapse's Rhydon holds weight on its Choice Band variant alone.
        data = fixture('cb_collapse')
        belief, = beliefs(data)
        post = self.post(data)['rhydon']
        payloads = D._canonical_revealed_variant_payloads(belief, set_source=self.src, format_id=data['format_id'])
        ids = [v.variant_id for v in self.variants('rhydon')]
        excluded = [p for p in payloads if post.weights[ids.index(p['variant_id'])] == 0]
        self.assertGreater(len(excluded), 1)
        for seed in range(50):
            old, new = random.Random(seed), random.Random(seed)
            drawn = D._set_posterior_variant(post, excluded, belief, set_source=self.src, rng=new)
            self.assertEqual(drawn, excluded[old.randrange(len(excluded))]);self.assertEqual(new.getstate(), old.getstate())

    def test_the_worlds_sampler_is_byte_identical_under_uniform_weights_and_draws_choice_band_when_settled(self):
        data = fixture('refusal_trace')   # twelve Weezing survivors, its one strike refused: uniform
        belief, = beliefs(data)
        post = self.post(data)['weezing']
        self.assertEqual(set(post.weights), {1});self.assertFalse(post.strikes[0].admitted)
        sample = lambda rng, **kw: D._sample_revealed_opponent_fixture(belief, set_source=self.src, format_id=data['format_id'], rng=rng,
                                                                          canonicalize_revealed_variants=True, **kw)
        for seed in range(64):
            old, new = random.Random(seed), random.Random(seed)
            self.assertEqual(sample(new, set_posterior=post), sample(old))
            self.assertEqual(new.getstate(), old.getstate())
        data = fixture('cb_collapse')
        belief, = beliefs(data)
        post = self.post(data)['rhydon']
        uniform = replace(post, weights=tuple(int(h) for h in post.hard), support=post.hard)
        items = set()
        for seed in range(64):
            old, new = random.Random(seed), random.Random(seed)
            self.assertEqual(sample(new, set_posterior=uniform), sample(old))
            self.assertEqual(new.getstate(), old.getstate())
            items.add(sample(random.Random(seed)).item)
            self.assertEqual(sample(random.Random(seed), set_posterior=post).item, 'Choice Band')
        self.assertGreater(len(items), 1)   # today's uniform draw also gives Rhydon its other items

    def test_the_team_sampler_takes_the_posteriors_lazily_and_names_a_missing_species(self):
        data = fixture('refusal_trace')
        belief, = beliefs(data)
        view = PlayerBeliefView(data['owner'], belief.showdown_slot, (), (belief,))
        calls = []

        def posts():
            calls.append(1);return self.post(data)
        kwargs = dict(set_source=self.src, format_id=data['format_id'], team_size=6, canonicalize_revealed_variants=True)
        old, _ = D._opponent_team_from_belief_result(view, rng=random.Random(5), **kwargs)
        new, _ = D._opponent_team_from_belief_result(view, rng=random.Random(5), set_posteriors=posts, **kwargs)
        self.assertEqual(new, old);self.assertEqual(calls, [1])
        with self.assertRaisesRegex(ValueError, SP.MISSING_SPECIES):
            D._opponent_team_from_belief_result(view, rng=random.Random(5), set_posteriors={}, **kwargs)


class TwoWay(Base):
    def test_hard_support_must_equal_b2s_in_both_directions(self):
        data = fixture('refusal_item_changed')   # five revealed foes
        posts = self.post(data)
        b2 = {key: self.hard(post) for key, post in posts.items()}
        self.assertEqual(SP.two_way(posts, b2), posts)
        golduck = self.hard(posts['golduck'])
        for changed in ({**b2, 'golduck': golduck[1:]}, {**b2, 'golduck': golduck + [max(golduck) + 1]},
                        {**b2, 'golduck': golduck[:-1] + [golduck[-1] + 1]}):
            checked = SP.two_way(posts, changed)
            self.assertEqual({k: p.refusals for k, p in checked.items() if p.refusals}, {'golduck': (SP.B2_DISAGREEMENT,)})
            self.assertEqual(checked['golduck'].weights, posts['golduck'].weights)
        missing = dict(b2);del missing['golduck']
        self.assertEqual(SP.two_way(posts, missing)['golduck'].refusals, (SP.B2_DISAGREEMENT,))
        extra = SP.two_way(posts, {**b2, 'snorlax': [0]})
        self.assertEqual(extra['snorlax'].refusals, (SP.B2_DISAGREEMENT,));self.assertEqual(extra['snorlax'].hard, ())

    def test_damage_narrowing_is_not_a_disagreement(self):
        posts = self.post(fixture('cb_collapse'))
        self.assertLess(len(self.alive(posts['rhydon'])), len(self.hard(posts['rhydon'])))
        self.assertFalse(SP.two_way(posts, {'rhydon': self.hard(posts['rhydon'])})['rhydon'].refusals)


class Provenance(Base):
    def test_a_pure_function_of_the_binding_and_the_game_lines(self):
        data = fixture('cb_excluded')
        lines = data['request']['lines']
        post = self.post(data)['snorlax']
        game = SP.game_lines_sha256(lines)
        self.assertEqual(post.provenance, hashlib.sha256(json.dumps((SP.SCHEMA, self.src.metadata.source_hash, SP.UNIFORM, game),
                                                                    separators=(',', ':')).encode()).hexdigest())
        self.assertEqual(self.post(data)['snorlax'], post)
        clock = [('|t:|1' if x.startswith('|t:|') else x) for x in lines]
        self.assertNotEqual(clock, lines)
        self.assertEqual(self.post(data, clock)['snorlax'].provenance, post.provenance)
        self.assertEqual(self.post(data, clock)['snorlax'].weights, post.weights)
        self.assertNotEqual(self.post(data, binding='b' * 64)['snorlax'].provenance, post.provenance)
        changed = list(lines);changed.insert(-1, '|-weather|none')
        self.assertNotEqual(self.post(data, changed)['snorlax'].provenance, post.provenance)
        with self.assertRaisesRegex(ValueError, 'uniform prior'):
            self.post(data, generator_weights={'snorlax': (1,)})
        with self.assertRaisesRegex(ValueError, SP.DUPLICATE_SPECIES):
            SP.posterior(lines, None, source=self.src, owner=data['owner'], beliefs=beliefs(data) * 2, format_id=data['format_id'], dex=self.dex)


if __name__ == '__main__':
    unittest.main()
