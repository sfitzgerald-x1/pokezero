"""constructor_worlds' peer view b (module docstring, Peer view b): the damage rule on real line forms, held to the policy builder's own
view of the peer's wire (pokezero.policy_opponent_view.build_policy_opponent_view: view b); the projection and its record on
synthetic carries; the completion's flag (the redaction implied); and the real B2 fold for both owners: view b changes the peer's root
view only in its two damage cells (and the redaction's columns), never the owner's view or its B2 parity."""
from dataclasses import replace
import math
import re
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

import jax
import jax.numpy as j
import numpy as np

from gpu_engine import constructor_worlds as cw
from gpu_engine import engine_device_trackers as tr
from tests.test_gpu_engine_constructor_worlds_carry import carry, leaves, reveal_tables, PRIVATE_LINES
import test_peer_view_b_golden as G

OWN_HEAD = ('|player|p1|Subject', '|player|p2|Opponent', '|start', '|switch|p1a: Swampert|Swampert, L84|300/300',
    '|switch|p2a: Snorlax|Snorlax, L80|400/400', '|turn|1')


def wire(lines, exact_seat):
    """`lines` (the omniscient form: both seats' HP exact) as `exact_seat`'s player channel shows them: the other seat's HP at
    Showdown's public percent (getHealth's shared half), '0 fnt' kept."""
    out = []
    for line in lines:
        p = line.split('|')
        if len(p) > 3 and p[1] in ('switch', 'drag', 'replace', '-damage', '-heal', '-sethp'):
            i = 4 if p[1] in ('switch', 'drag', 'replace') else 3
            if p[2][:2] != exact_seat and '/' in p[i]:
                head = p[i].split()[0];hp, top = head.split('/');p[i] = f'{cw.public_percent(int(hp), int(top))}/100' + p[i][len(head):]
        out.append('|'.join(p))
    return out


def mirror(lines):
    """The same game with the seats swapped (p1 <-> p2) in every ident and |player| line."""
    return [re.sub(r'\bp([12])(?=a:|:|\||$)', lambda m: 'p2' if m.group(1) == '1' else 'p1', line) for line in lines]


def last_condition(lines, seat, species):
    """The last condition of `seat`'s `species` on omniscient lines (its request's), '400/400' when none."""
    out = '400/400'
    for line in lines:
        p = line.split('|')
        if len(p) > 3 and p[2].startswith(f'{seat}a: {species}'):
            if p[1] in ('switch', 'drag', 'replace') and len(p) > 4:out = p[4]
            elif p[1] in ('-damage', '-heal', '-sethp'):out = p[3]
    return out


class PublicPercentTests(unittest.TestCase):
    def test_the_integer_rule_is_showdowns_float_rule(self):
        """sim/pokemon.ts getHealth: Math.ceil(100 * hp / maxhp), 99 for a damaged Pokemon that would show 100; the integer rule
        equals it for every hp <= maxhp up to 800 (Python floats are IEEE doubles, as JavaScript's)."""
        for top in range(1, 801):
            for hp in range(0, top + 1):
                shown = math.ceil(100 * hp / top);shown = 99 if shown == 100 and hp < top else shown
                if cw.public_percent(hp, top) != shown:self.fail(f'{hp}/{top}: {cw.public_percent(hp, top)} != {shown}')
        self.assertEqual((cw.public_percent(290, 291), cw.public_percent(291, 291), cw.public_percent(1, 291), cw.public_percent(0, 291)), (99, 100, 1, 0))


class PeerDamageRuleTests(unittest.TestCase):
    """peer_damage_facts on the owner's wire against view b's own cells: the builder on the PEER's wire of the same game."""
    def builder_cells(self, omniscient, owner):
        """(view b's LAST_DAMAGE_TAKEN of our active, its own active's LAST_DAMAGE_DEALT) from the peer's wire of `omniscient`."""
        from pokezero.observation import ObservationFeatureMasks
        from pokezero.policy_opponent_view import build_policy_opponent_view
        from test_policy_opponent_view import VOCAB, request, FakeSetSource, V4_REPLAY_OBSERVATION_SPEC, _dex
        own, peer = f'p{owner + 1}', f'p{2 - owner}'
        mine = request(peer);mine['side']['pokemon'][0]['condition'] = last_condition(omniscient, peer, 'Snorlax')
        view = build_policy_opponent_view(public_lines=wire(omniscient, peer), hp_visibility={peer: 'exact', own: 'percentage'},
            sampled_self_request=mine, opponent_slot=peer, battle_id='t', battle_seed=1, format_id='gen3randombattle', set_source=FakeSetSource(),
            spec=V4_REPLAY_OBSERVATION_SPEC, feature_masks=ObservationFeatureMasks(transition_token_budget=0))
        n = np.asarray(view.observation(category_vocab=VOCAB, dex=_dex()).numeric_features)
        taken = [float(x) for x in n[7:13, 39] if x];dealt = [float(x) for x in n[1:7, 38] if x]   # the active tokens only carry them
        self.assertLessEqual((len(taken), len(dealt)), (1, 1))
        return (taken or [0.])[0], (dealt or [0.])[0], n

    def check(self, body, want, owner=0):
        """The rule on our wire equals view b's cells (float32), and its public values are `want` (taken, dealt)."""
        omniscient = list(OWN_HEAD) + list(body)
        if owner == 1:omniscient = mirror(omniscient)
        d = cw.peer_damage_facts(wire(omniscient, f'p{owner + 1}'), owner)
        taken, dealt, _ = self.builder_cells(omniscient, owner)
        self.assertEqual((np.float32(d.owner_taken), np.float32(d.peer_dealt)), (np.float32(taken), np.float32(dealt)))
        np.testing.assert_allclose((d.owner_taken, d.peer_dealt), want, atol=1e-12)
        return d

    def test_damage_then_a_heal_then_damage(self):
        """The previous reading of a -damage line is the active's last condition, a heal's included; a tagged chip enters taken,
        not dealt; a round's hits sum; only the last settled round shows."""
        body = ('|move|p2a: Snorlax|Body Slam|p1a: Swampert', '|-damage|p1a: Swampert|250/300', '|move|p1a: Swampert|Earthquake|p2a: Snorlax',
            '|-damage|p2a: Snorlax|300/400', '|-heal|p1a: Swampert|268/300|[from] item: Leftovers', '|turn|2',
            '|move|p2a: Snorlax|Body Slam|p1a: Swampert', '|-damage|p1a: Swampert|200/300', '|-damage|p1a: Swampert|163/300|[from] psn', '|turn|3')
        for owner in (0, 1):
            with self.subTest(owner=owner):
                d = self.check(body, (.35, .23), owner)   # 268 -> 200 -> 163 of 300: 90 -> 67 -> 55; dealt 90 -> 67 only
                self.assertEqual((d.owner_taken_exact, d.peer_dealt_exact, d.owner_active, d.peer_active), (105 / 300, 68 / 300, 'swampert', 'snorlax'))
        self.check(body[:5], (0., 0.))   # a round in flight shows the previous one (none yet)
        self.check(body[:6], (.16, .16))   # 300 -> 250: 100 -> 84 (the heal comes after)

    def test_the_99_cap_and_one_point_losses(self):
        self.check(('|move|p2a: Snorlax|Body Slam|p1a: Swampert', '|-damage|p1a: Swampert|299/300', '|turn|2'), (.01, .01))   # 100 -> 99 (capped)
        d = self.check(('|move|p2a: Snorlax|Body Slam|p1a: Swampert', '|-damage|p1a: Swampert|299/300', '|turn|2', '|move|p2a: Snorlax|Body Slam|p1a: Swampert',
            '|-damage|p1a: Swampert|298/300', '|turn|3'), (0., 0.))   # 99 -> 99: nothing shows, though 1 HP was lost
        self.assertEqual(d.owner_taken_exact, 1 / 300)

    def test_a_faint_counts_the_previous_reading_whole(self):
        """'0 fnt' is 0: the KO is the previous public reading; the faint ends the active's readings; a replacement resets our
        taken, never the peer's dealt."""
        body = ('|move|p2a: Snorlax|Body Slam|p1a: Swampert', '|-damage|p1a: Swampert|200/300', '|turn|2', '|move|p2a: Snorlax|Body Slam|p1a: Swampert',
            '|-damage|p1a: Swampert|0 fnt', '|faint|p1a: Swampert', '|upkeep', '|switch|p1a: Starmie|Starmie, L79|250/250', '|turn|3')
        for owner in (0, 1):
            with self.subTest(owner=owner):
                d = self.check(body, (0., .67), owner)   # 200/300 shows 67
                self.assertEqual((d.owner_active, d.peer_dealt_exact), ('starmie', 200 / 300))
        omniscient = list(OWN_HEAD) + list(body[:6]) + ['|-damage|p1a: Swampert|0 fnt']   # after a faint nothing more is recorded
        self.assertEqual(cw.peer_damage_facts(wire(omniscient, 'p1'), 0), cw.peer_damage_facts(wire(list(OWN_HEAD) + list(body[:6]), 'p1'), 0))

    def test_switch_ins_reset_their_own_seat_and_start_the_readings(self):
        """Our switch-in's condition is the next hit's previous reading and resets our taken; the peer's switch-in resets its dealt."""
        ours = ('|switch|p1a: Starmie|Starmie, L79|200/250', '|move|p2a: Snorlax|Body Slam|p1a: Starmie', '|-damage|p1a: Starmie|150/250', '|turn|2')
        self.check(ours, (.20, .20))   # 80 -> 60
        theirs = ('|move|p2a: Snorlax|Body Slam|p1a: Swampert', '|-damage|p1a: Swampert|250/300', '|turn|2',
            '|switch|p2a: Starmie|Starmie, L79|250/250', '|move|p1a: Swampert|Earthquake|p2a: Starmie', '|-damage|p2a: Starmie|200/250', '|turn|3')
        omniscient = list(OWN_HEAD) + list(theirs)
        d = cw.peer_damage_facts(wire(omniscient, 'p1'), 0)
        self.assertEqual((d.owner_taken, d.peer_dealt, d.peer_active), (0., 0., 'starmie'))
        self.assertEqual(cw.peer_damage_facts(wire(omniscient[:-4], 'p1'), 0).peer_dealt, .16)   # before Starmie: Snorlax's settled hit
        mid = list(OWN_HEAD) + ['|move|p2a: Snorlax|Body Slam|p1a: Swampert', '|-damage|p1a: Swampert|250/300', '|switch|p1a: Starmie|Starmie, L79|250/250',
            '|turn|2']   # a hit then our switch in the same round (Pursuit's shape): our taken resets with the switch, the peer's dealt stays
        self.assertEqual((cw.peer_damage_facts(wire(mid, 'p1'), 0).owner_taken, cw.peer_damage_facts(wire(mid, 'p1'), 0).peer_dealt), (0., .16))

    def test_the_move_window(self):
        """Dealt is an untagged -damage on our active while the peer's |move| window is open: cant, a confusion activation, our own
        move and a switch close or move it; a [from] tag never counts."""
        hit = '|-damage|p1a: Swampert|250/300'
        cases = ((('|move|p2a: Snorlax|Body Slam|p1a: Swampert', hit), .16), (('|move|p2a: Snorlax|Body Slam|p1a: Swampert', '|cant|p1a: Swampert|par', hit), 0.),
            (('|move|p2a: Snorlax|Body Slam|p1a: Swampert', '|-activate|p1a: Swampert|confusion', hit), 0.),
            (('|move|p2a: Snorlax|Body Slam|p1a: Swampert', '|move|p1a: Swampert|Earthquake|p2a: Snorlax', hit), 0.),
            (('|move|p2a: Snorlax|Body Slam|p1a: Swampert', '|-damage|p1a: Swampert|250/300|[from] Spikes'), 0.),
            (('|move|p2a: Snorlax|Body Slam|p1a: Swampert', '|-activate|p1a: Swampert|move: Protect', hit), .16))   # not confusion: open
        for body, dealt in cases:
            with self.subTest(body=body):
                self.check(tuple(body) + ('|turn|2',), (.16, dealt))   # 300 -> 250: 100 -> 84

    def test_a_sethp_reading_and_another_ident_are_the_env_parsers(self):
        """-sethp sets the previous reading; a -damage line of another ident than the active's records nothing."""
        self.check(('|move|p1a: Swampert|Pain Split|p2a: Snorlax', '|-sethp|p2a: Snorlax|350/400|[from] move: Pain Split|[silent]',
            '|-sethp|p1a: Swampert|150/300|[from] move: Pain Split', '|move|p2a: Snorlax|Body Slam|p1a: Swampert', '|-damage|p1a: Swampert|120/300', '|turn|2'),
            (.10, .10))   # 150 -> 120: 50 -> 40
        lines = wire(list(OWN_HEAD) + ['|move|p2a: Snorlax|Body Slam|p1a: Swampert', '|-damage|p1: Swampert|250/300', '|turn|2'], 'p1')
        self.assertEqual(cw.peer_damage_facts(lines, 0).owner_taken, 0.)

    def test_line_facts_read_it_only_when_asked(self):
        lines = wire(list(OWN_HEAD) + ['|move|p2a: Snorlax|Body Slam|p1a: Swampert', '|-damage|p1a: Swampert|250/300', '|turn|2'], 'p1')
        off, on = cw.line_facts(lines, 0), cw.line_facts(lines, 0, peer_damage=True)
        self.assertIsNone(off.peer_damage);self.assertEqual(replace(on, peer_damage=None), off)
        self.assertEqual(on.peer_damage, cw.peer_damage_facts(lines, 0))
        with self.assertRaises(ValueError):cw.peer_damage_facts(lines, 2)


def damage_carry(*, taken=.2336, dealt=.2336, slots=(0, 1)):
    """A from_b2-style carry, owner p1: the active slots `slots` (owner, peer), B2's exact owner-HP sums in the owner's taken and the
    peer's dealt, distinct markers everywhere else in the two features."""
    t = carry([[1, 2, 3, 0, 0, 0], [4, 5, 0, 0, 0, 0]], [[0, 1, 2, -1, -1, -1], [0, 1, -1, -1, -1, -1]])
    base = np.arange(12, dtype=np.float32).reshape(1, 2, 6) / 100
    took, dealt_ = base.copy(), base.copy() + .5;took[0, 0, slots[0]] = taken;dealt_[0, 1, slots[1]] = dealt
    mon = dict(t.mon_features, last_damage_dealt=j.asarray(dealt_), last_damage_taken=j.asarray(took))
    return t._replace(active_slot=j.asarray([list(slots)], j.int32), mon_features=mon)


def facts_with(owner_taken=.24, peer_dealt=.23, owner_taken_exact=.2336, peer_dealt_exact=.2336):
    return cw.LineFacts(('beta',), (), peer_damage=cw.PeerDamage(owner_taken, peer_dealt, owner_taken_exact, peer_dealt_exact, 'alpha', 'delta'))


class ProjectionTests(unittest.TestCase):
    def test_the_projection_writes_the_two_cells_only(self):
        t = damage_carry();done = cw.project_peer_damage(t, 0, facts_with())
        took, dealt = (np.asarray(done.mon_features[k]) for k in ('last_damage_taken', 'last_damage_dealt'))
        want_took, want_dealt = (np.asarray(t.mon_features[k]).copy() for k in ('last_damage_taken', 'last_damage_dealt'))
        want_took[0, 0, 0] = np.float32(.24);want_dealt[0, 1, 1] = np.float32(.23)
        np.testing.assert_array_equal(took, want_took);np.testing.assert_array_equal(dealt, want_dealt)
        self.assertEqual((done.mon_features['last_damage_taken'].dtype, list(done.mon_features)), (j.float32, list(t.mon_features)))
        same = leaves(t, ('last_damage_taken', 'last_damage_dealt'))
        self.assertTrue(all(x is y for (_, x), (_, y) in zip(same, leaves(done, ('last_damage_taken', 'last_damage_dealt')))))
        self.assertIs(cw.project_peer_damage(done, 0, facts_with()), done)   # a second pass is a no-op
        theirs = cw.project_peer_damage(t, 1, facts_with())   # owner p2: its own taken (side 1) and p1's dealt
        self.assertEqual((float(np.asarray(theirs.mon_features['last_damage_taken'])[0, 1, 1]), float(np.asarray(theirs.mon_features['last_damage_dealt'])[0, 0, 0])),
            (float(np.float32(.24)), float(np.float32(.23))))
        capped = cw.project_peer_damage(t, 0, facts_with(owner_taken=1.4))   # the writer's min(1, x)
        self.assertEqual(float(np.asarray(capped.mon_features['last_damage_taken'])[0, 0, 0]), 1.)
        idle = damage_carry(slots=(-1, 1));out = cw.project_peer_damage(idle, 0, facts_with())   # no active slot: that cell is left
        np.testing.assert_array_equal(np.asarray(out.mon_features['last_damage_taken']), np.asarray(idle.mon_features['last_damage_taken']))

    def test_the_projection_and_its_record_refuse_what_they_cannot_read(self):
        with self.assertRaisesRegex(ValueError, 'peer_damage'):cw.project_peer_damage(damage_carry(), 0, cw.LineFacts(('beta',), ()))
        with self.assertRaisesRegex(ValueError, 'peer_damage'):cw.project_peer_damage(damage_carry(), 0, None)
        with self.assertRaises(cw.CarryLayoutError):cw.project_peer_damage(damage_carry()._replace(mon_features=None), 0, facts_with())
        bad = damage_carry();bad = bad._replace(mon_features=dict(bad.mon_features, last_damage_taken=j.zeros((1, 2, 6), j.int32)))
        with self.assertRaises(cw.CarryLayoutError):cw.peer_view_b_record(bad, 0, facts_with())

    def test_the_record(self):
        record = cw.peer_view_b_record(damage_carry(), 0, facts_with())
        f32 = lambda x: float(np.float32(x))
        self.assertEqual(record, dict(enabled=True, cells=[
            dict(column='last_damage_taken', side='p1', slot=0, species='alpha', carry=f32(.2336), exact=.2336, public=f32(.24)),
            dict(column='last_damage_dealt', side='p2', slot=1, species='delta', carry=f32(.2336), exact=.2336, public=f32(.23))],
            projected=2, changed=2, exact_agrees=True))
        off = cw.peer_view_b_record(damage_carry(taken=.3), 0, facts_with())   # B2's value is not the exact rule's: recorded, not hidden
        self.assertFalse(off['exact_agrees'])
        idle = cw.peer_view_b_record(damage_carry(slots=(-1, 1)), 0, facts_with())
        self.assertEqual((idle['projected'], idle['cells'][0]['slot'], idle['cells'][0]['carry']), (1, -1, None))

    def test_view_b_in_the_completion_implies_the_redaction(self):
        """complete_public's peer_view_b runs the private-reveal redaction (reused, never duplicated) and then the projection, with or
        without redact_private_reveals; off, neither; without the root's facts it refuses (never a silent off)."""
        calls = [];t = damage_carry();facts = facts_with();keep = lambda public, *a: public
        stand_ins = dict(order_peer=lambda state, public, *a: public, fill_peer_pressure=lambda state, public, *a: public,
            fill_shown_types=lambda state, public, tables: public, orient_credits=keep, fill_negative_items=keep, fill_owner_spikes=keep, fill_baton=keep,
            fill_recharge=keep, _redact=lambda public, *a: calls.append('redact') or public, project_peer_damage=lambda public, *a: calls.append('project') or public)
        with patch.multiple(cw, **stand_ins):
            for keywords, want in ((dict(peer_view_b=True), ['redact', 'project']), (dict(peer_view_b=True, redact_private_reveals=True), ['redact', 'project']),
                    ({}, []), (dict(redact_private_reveals=True), ['redact'])):
                with self.subTest(**keywords):
                    calls.clear();self.assertIs(cw.complete_public(None, t, None, 0, facts=facts, **keywords), t);self.assertEqual(calls, want)
            with self.assertRaisesRegex(ValueError, 'peer_damage'):cw.complete_public(None, t, None, 0, peer_view_b=True)
            self.assertIs(cw.complete_public(None, t, None, 0), t)




class PeerViewBB2Tests(unittest.TestCase):
    """The real B2 fold (tests/test_peer_view_b_golden's scenarios, both owners): B2's carry holds the exact rule's values in the two
    cells (the record's exact_agrees); with view b every world's two cells hold view b's values; the owner's view (and world 0's B2
    parity) is unchanged; the peer's view changes in its own active's LAST_DAMAGE_DEALT (token 1) and our active's LAST_DAMAGE_TAKEN
    (token 7 + its public slot) and, with a side-private Pressure line, REVEALED_ABILITY of that member, nothing else."""
    @classmethod
    def setUpClass(cls):
        from gpu_engine import engine_device_observation as e
        cls.fx = G.b2_fixture();cls.tables = cw.carry_tables(cls.fx.metadata, cls.fx.catalog)
        cls.observation = e.compile_tables(cls.fx.metadata, catalog=cls.fx.catalog);cls.e = e

    def view(self, world, side, b2):
        from gpu_engine import device_public_leaf as leaf
        mask = j.asarray(np.asarray(b2['legal_action_mask'], bool)[None])
        out = leaf.from_public(world.state, world.public, self.observation, observer_sides=j.asarray([side], j.int32), legal_action_mask=mask,
            enable_device_leaf=True).observation
        return {k: np.asarray(getattr(out, k)) for k in ('numeric_features', 'categorical_ids', 'token_type_ids', 'attention_mask')}

    def test_view_b_changes_only_the_peers_damage_cells_and_reveals(self):
        dealt, taken, revealed = self.e._index('LAST_DAMAGE_DEALT'), self.e._index('LAST_DAMAGE_TAKEN'), self.e._index('REVEALED_ABILITY')
        want = dict(heal=(.24, .24), cap=(.01, .01), faint=(0., .69), switch=(.13, .13), pressure=(.14, .14))
        for name in G.SCENARIOS:
            for owner in (0, 1):
                with self.subTest(scenario=name, owner=owner):
                    root, made, lines = G.b2_root(self.fx, name, owner);peer = 1 - owner
                    facts = cw.line_facts(lines, owner, peer_damage=True);record = cw.peer_view_b_record(root.public, owner, facts)
                    self.assertTrue(record['exact_agrees']);self.assertEqual(record['projected'], 2)
                    np.testing.assert_allclose([c['public'] for c in record['cells']], np.float32(want[name]), rtol=0, atol=0)
                    slots = np.asarray(root.public.active_slot)[0];b2 = root.view['_b2_root_inputs']
                    for w in made.worlds:
                        off = cw.complete_world(w, self.tables, owner, facts=facts, root=root.public)
                        on = cw.complete_world(w, self.tables, owner, facts=facts, root=root.public, peer_view_b=True)
                        self.assertIs(on.state, off.state)
                        self.assertEqual((float(np.asarray(on.public.mon_features['last_damage_taken'])[0, owner, slots[owner]]),
                            float(np.asarray(on.public.mon_features['last_damage_dealt'])[0, peer, slots[peer]])), tuple(float(np.float32(x)) for x in want[name]))
                        self.assertIs(cw.complete_world(on, self.tables, owner, facts=facts, root=root.public, peer_view_b=True), on)
                        if w.index > 1:continue   # the views on two worlds (every world's carry is projected alike above)
                        mine_on, mine_off = self.view(on, owner, b2), self.view(off, owner, b2)
                        # the owner's view is unchanged, so world 0's B2 parity (root view against B2's encoding) is the same with view b
                        for k in mine_on:np.testing.assert_array_equal(mine_on[k], mine_off[k], err_msg=k)
                        theirs_on, theirs_off = self.view(on, peer, b2), self.view(off, peer, b2)
                        changed = sorted(map(tuple, np.argwhere(theirs_on['numeric_features'][0] != theirs_off['numeric_features'][0]).tolist()))
                        expected = {(1, dealt): want[name][1], (7 + int(slots[owner]), taken): want[name][0]}
                        moved = {cell for cell, value in expected.items() if float(theirs_off['numeric_features'][0][cell]) != float(np.float32(value))}
                        if name == 'pressure':moved.add((7 + int(slots[owner]), revealed))   # the implied redaction
                        self.assertEqual(set(changed), moved)
                        for cell, value in expected.items():self.assertEqual(float(theirs_on['numeric_features'][0][cell]), float(np.float32(value)))
                        for k in ('categorical_ids', 'token_type_ids', 'attention_mask'):np.testing.assert_array_equal(theirs_on[k], theirs_off[k], err_msg=k)



class ReplyPriorCompositionTests(unittest.TestCase):
    """--peer-view-b with --reply-archive: the br_fp reply prior (gpu_engine/depthn_reply_prior.ReplyPrior) reads the worlds
    prepare_roots completed (DepthNLiveSearch._search: self.reply.apply(blocks, item.made.worlds, owner)) through
    device_depth2_pricer._make_views on the live executor's observe (device_public_teacher.make_executor's: leaf.from_public on the
    observation tables), so its peer inputs carry view b: the projected damage cells and the private-reveal redaction."""
    @classmethod
    def setUpClass(cls):
        from gpu_engine import device_public_leaf as leaf, engine_device_observation as e
        cls.fx = G.b2_fixture();cls.e = e;tables = e.compile_tables(cls.fx.metadata, catalog=cls.fx.catalog)
        cls.executor = NS(legal=jax.jit(lambda state: j.ones((state.active.shape[0], 2, 9), bool)),   # the views' legal masks: not under test
            observe=jax.jit(lambda state, public, owner, legal: leaf.from_public(state, public, tables, observer_sides=owner, legal_action_mask=legal,
                enable_device_leaf=True)))

    def reply_inputs(self, owner, peer_view_b):
        """The reply prior's peer inputs [W, 23, 132] of the 'pressure' root (a hit on our Alpha and its side-private Pressure line) as
        prepare_roots(peer_view_b=...) completes its worlds, and the root's slots and view-b record."""
        from gpu_engine.depthn_reply_prior import ReplyPrior
        root, made, lines = G.b2_root(self.fx, 'pressure', owner)
        constructor = NS(metadata=self.fx.metadata, catalog=self.fx.catalog, count=16, prepare=lambda request, index: (root, made, (True,) * 9))
        with patch.object(cw, 'constructor_binding', return_value='binding'):   # the stand-in constructor's binding is not under test
            item, = cw.prepare_roots(constructor, [dict(request_id='r', lines=list(lines))], **({'peer_view_b': True} if peer_view_b else {}))
        self.assertIs(type(item), cw.Prepared)
        inputs = ReplyPrior(self.executor, None, params=()).inputs(item.made.worlds[:2], owner)
        n = np.asarray(inputs['numeric_features']);return n.reshape(2, -1, n.shape[-1])[:, :23], np.asarray(root.public.active_slot)[0], item

    def test_the_reply_priors_peer_inputs_carry_view_b(self):
        dealt, taken, revealed = self.e._index('LAST_DAMAGE_DEALT'), self.e._index('LAST_DAMAGE_TAKEN'), self.e._index('REVEALED_ABILITY')
        for owner in (0, 1):
            with self.subTest(owner=owner):
                on, slots, item = self.reply_inputs(owner, True);off, _, plain = self.reply_inputs(owner, False)
                ours = 7 + int(slots[owner]);cells = {c['column']: c for c in item.peer_view_b['cells']}
                # view b on: Beta's (the peer's own active, token 1) dealt and our Alpha's taken are the projected public values (291 -> 250:
                # 100 -> 86), and Alpha's Pressure, revealed on our wire alone, is redacted from the peer's view
                self.assertEqual(on[:, 1, dealt].tolist(), [cells['last_damage_dealt']['public']] * 2)
                self.assertEqual(on[:, ours, taken].tolist(), [cells['last_damage_taken']['public']] * 2)
                self.assertEqual((cells['last_damage_taken']['public'], on[:, ours, revealed].tolist()), (float(np.float32(.14)), [0., 0.]))
                # view b off: B2's exact owner-HP sums (41/291) and the reveal, as before the flag
                self.assertEqual(off[:, 1, dealt].tolist(), [cells['last_damage_dealt']['carry']] * 2)
                self.assertEqual(off[:, ours, taken].tolist(), [cells['last_damage_taken']['carry']] * 2)
                self.assertEqual(off[:, ours, revealed].tolist(), [1., 1.]);self.assertIsNone(plain.peer_view_b)
                changed = {tuple(x[1:]) for x in np.argwhere(on != off).tolist()}
                self.assertEqual(changed, {(1, dealt), (ours, taken), (ours, revealed)})   # nothing else of the reply prior's input moves

if __name__ == '__main__':unittest.main()
