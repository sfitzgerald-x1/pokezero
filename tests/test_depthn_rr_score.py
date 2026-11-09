import collections
from contextlib import redirect_stdout
import hashlib
import io
import json
import math
from pathlib import Path
import random
import statistics
import unittest
from tempfile import TemporaryDirectory

from gpu_engine import search_hedge as hedge
from pokezero.phase_d_owning_capture import SCHEMA as CAPTURE_SCHEMA, encoded
from scripts import depthn_rr_score as S

PARTY = ('snorlax', 'skarmory', 'gengar', 'starmie', 'zapdos', 'tyranitar')
MOVES = ('bodyslam', 'earthquake', 'rest', 'curse')


def request(owner, kind='move'):
    body = dict(side=dict(id=owner))
    if kind == 'forceSwitch':body['forceSwitch'] = [True]
    elif kind == 'wait':body['wait'] = True
    else:body['active'] = [dict(moves=[])]
    return '|request|' + json.dumps(body)


def write_capture(path, owner, events, *, ordinal=0):
    """A read_capture-valid owning capture: events are line lists (one WIRE row each) or 'ROOT' (a decision at the current
    request), chained exactly as phase_d_owning_capture.capture_actor emits them."""
    rows, state = [], dict(prior='0' * 64, sequence=0)
    def emit(kind, **fields):
        blob = encoded(dict(schema=CAPTURE_SCHEMA, sequence=state['sequence'], previous_sha256=state['prior'], event=kind, **fields))
        rows.append(blob);state['prior'] = hashlib.sha256(blob).hexdigest();state['sequence'] += 1
    emit('BEGIN', cell=dict(ordinal=ordinal), declaration_sha256='0' * 64, namespace='test', battle_id=f'battle-{ordinal}', owner=owner,
         seed=1, stream_id=1)
    prefix, count, generation, decision, turn, requests = hashlib.sha256(), 0, 0, 0, 0, []
    for event in events:
        if event == 'ROOT':
            emit('ROOT', decision_index=decision, turn=turn, generation=generation, request_count=len(requests),
                 prefix_sha256=prefix.hexdigest(), own_request_line=requests[-1])
            decision += 1
            continue
        for line in event:
            prefix.update(line.encode() + b'\n')
            if line.startswith('|request|'):requests.append(line)
            if line.startswith('|turn|'):turn = int(line.split('|')[2])
        generation += 1
        emit('WIRE', generation=generation, start=count, end=count + len(event), prefix_sha256=prefix.hexdigest(), lines=list(event))
        count += len(event)
    Path(path).write_bytes(b''.join(r + b'\n' for r in rows))


def grid_of(fn):return [[fn(a, b) for b in range(9)] for a in range(9)]


def world_row(grid, *, requested=True, kept=None, logits=None, moves=MOVES, party=PARTY, fainted=None, legal=None,
              move_targets=(0, 1, 2, 3), switch_targets=(1, 2, 3, 4, 5), charge=None):
    """One world as scripts/depthn_rr_reprice.py writes it (describe(): prices_fields, opponent_facts, PeerLanes)."""
    kept = list(kept) if kept is not None else [requested] * 9
    legal = list(legal if legal is not None else kept);fainted = list(fainted or [False] * 6)
    return dict(world=0, index=0, seed=1, kernel_seed=2, refusals=[], constructed=True, priced=True,
                opponent_requested=requested, kept=kept, root_peer_logits=list(logits or [0.] * 9), grid=grid,
                opponent=dict(side='p2', active_slot=0, active_species=party[0], active_moves=list(moves), active_move_engine_ids=[1, 2, 3, 4],
                              charge_move=charge, party=[dict(slot=i, species=sp, species_engine_id=i + 1, fainted=fainted[i], hp=0 if fainted[i] else 100,
                                                              max_hp=100) for i, sp in enumerate(party)]),
                peer=dict(legal=legal, move_targets=list(move_targets), switch_targets=list(switch_targets), error=0, usable=[True] * 4,
                          lanes=[None] * 9), kept_matches_peer_legal=kept == legal if requested else None, requested_matches_peer=True)


def raw_row(worlds, *, ordinal=0, decision_index=0, request_count=1, owner='p1', root_logits=None, root_legal=None, greedy=0,
            live_action=None, kind='move', status='COMPLETE', replay_action=None, source='search'):
    """One rows-<shard>.jsonl row as scripts/depthn_rr_reprice.py's decision_row writes it (the fields this scorer reads)."""
    return dict(schema='pokezero.depthn-rr-reprice.row.v1', shard=0, ordinal=ordinal, seat=0, seed=10, battle_id=f'battle-{ordinal}', owner=owner,
                stream_id=1, decision_index=decision_index, generation=3, prefix_sha256='0' * 64, request_count=request_count, capture_turn=1,
                replay=dict(status=status, cause=None, action_index=replay_action, q_values=None, policy_target=None),
                live=dict(source=source, cause=None, request_kind=kind, turn=1, played_action_index=live_action, greedy_action_index=greedy,
                          prefix_sha256='0' * 64, search_status='COMPLETE', search_cause=None, search_action_index=live_action, search_q_values=None),
                live_problem=None, parity=dict(live_found=True, live_searched=True, replay_complete=True, status_match=True, action_equal=None, q_equal=None),
                root=None if worlds is None else dict(root_logits=list(root_logits or [0.] * 9), root_legal=list(root_legal or [True] * 9)),
                prepared=None, worlds=worlds, notes=[], kept_disagreements=0, capture_ok=True)


def waiting_worlds(columns):
    """Waiting worlds (Q_w(a) = G_w[a][0]): columns[w] = {action: value} on the legal actions."""
    return [world_row(grid_of(lambda a, b, c=c: c.get(a, 0.) if b == 0 else None), requested=False) for c in columns]


def legal_mask(*actions):return [a in actions for a in range(9)]


class ReplyParsingTests(unittest.TestCase):
    """Foul Play is p2 (the owner p1) unless stated; line sequences follow the gen3 Showdown sim's player stream."""

    def reply(self, window, forced=False, owner='p1'):return S.fp_reply(window, owner, forced)

    def test_normal_move(self):
        r = self.reply(['|', '|t:|1700000000', '|move|p1a: Starmie|Surf|p2a: Snorlax', '|-damage|p2a: Snorlax|70/100',
                        '|move|p2a: Snorlax|Body Slam|p1a: Starmie', '|-damage|p1a: Starmie|150/282', '|-status|p1a: Starmie|par', '|', '|upkeep'])
        self.assertEqual((r.kind, r.id, r.cause), ('move', 'bodyslam', None))

    def test_sleep_talk_caller_line_wins(self):
        r = self.reply(['|', '|move|p1a: Breloom|Leech Seed|p2a: Snorlax', '|-start|p2a: Snorlax|move: Leech Seed', '|cant|p2a: Snorlax|slp',
                        '|move|p2a: Snorlax|Sleep Talk|p2a: Snorlax', '|move|p2a: Snorlax|Body Slam|p1a: Breloom|[from] Sleep Talk',
                        '|-damage|p1a: Breloom|120/282', '|', '|upkeep'])
        self.assertEqual((r.kind, r.id, r.cause, r.detail), ('move', 'sleeptalk', None, 'after_cant:slp'))

    def test_metronome_caller_line_wins(self):
        r = self.reply(['|', '|move|p2a: Clefable|Metronome|p2a: Clefable', '|move|p2a: Clefable|Thunderbolt|p1a: Starmie|[from] Metronome'])
        self.assertEqual(r.id, 'metronome')

    def test_locked_and_hidden_power_ids(self):
        self.assertEqual(self.reply(['|move|p2a: Dragonite|Outrage|p1a: Starmie|[from]lockedmove']).id, 'outrage')
        self.assertEqual(self.reply(['|move|p2a: Dragonite|Hidden Power|p1a: Jolteon']).id, 'hiddenpower')
        self.assertEqual(S.showdown_id('Hidden Power Fire'), 'hiddenpower')
        self.assertEqual(S.showdown_id('Will-O-Wisp'), 'willowisp')
        self.assertEqual(self.reply(['|move|p2a: Snorlax|Struggle|p1a: Starmie']).id, 'struggle')

    def test_voluntary_switch(self):
        r = self.reply(['|', '|t:|1700000000', '|switch|p2a: Skarmory|Skarmory, F|100/100', '|move|p1a: Tyranitar|Rock Slide|p2a: Skarmory',
                        '|-damage|p2a: Skarmory|70/100', '|', '|upkeep'])
        self.assertEqual((r.kind, r.id, r.detail), ('switch', 'skarmory', 'voluntary_switch'))
        forme = self.reply(['|switch|p2a: Deoxys|Deoxys-Speed, L73|100/100'])
        self.assertEqual(forme.id, 'deoxysspeed')

    def test_faint_replacement_is_not_a_reply(self):
        # Our Thunderbolt KOs Foul Play's active first; its replacement follows our wait request inside the same turn.
        r = self.reply(['|', '|move|p1a: Starmie|Thunderbolt|p2a: Pikachu', '|-damage|p2a: Pikachu|0 fnt', '|faint|p2a: Pikachu',
                        request('p1', 'wait'), '|', '|switch|p2a: Snorlax|Snorlax, F|100/100', '|', '|upkeep'])
        self.assertEqual((r.kind, r.cause, r.detail), (None, 'ko_first', 'faint_before_acting'))
        # It moved, then fainted (recoil) and was replaced: the move is the reply.
        r = self.reply(['|move|p2a: Snorlax|Double-Edge|p1a: Starmie', '|-damage|p2a: Snorlax|0 fnt', '|faint|p2a: Snorlax',
                        request('p1', 'wait'), '|switch|p2a: Gengar|Gengar, M|100/100'])
        self.assertEqual((r.kind, r.id), ('move', 'doubleedge'))

    def test_pursuit_hitting_a_switch_is_censored(self):
        r = self.reply(['|', '|-activate|p2a: Skarmory|move: Pursuit', '|move|p1a: Tyranitar|Pursuit|p2a: Skarmory|[from] Pursuit',
                        '|-resisted|p2a: Skarmory', '|-damage|p2a: Skarmory|49/100', '|switch|p2a: Snorlax|Snorlax, M|100/100', '|', '|upkeep'])
        self.assertEqual((r.kind, r.cause, r.detail), (None, 'other', 'pursuit_on_switch'))
        # Our Pursuit on a Foul Play that stays in is an ordinary turn: its move is the reply.
        r = self.reply(['|move|p1a: Tyranitar|Pursuit|p2a: Snorlax', '|-damage|p2a: Snorlax|85/100', '|move|p2a: Snorlax|Body Slam|p1a: Tyranitar'])
        self.assertEqual(r.id, 'bodyslam')
        # Foul Play's Pursuit on OUR switch is its move.
        r = self.reply(['|-activate|p1a: Starmie|move: Pursuit', '|move|p2a: Tyranitar|Pursuit|p1a: Starmie|[from] Pursuit',
                        '|switch|p1a: Snorlax|Snorlax|400/400'])
        self.assertEqual((r.kind, r.id), ('move', 'pursuit'))

    def test_ko_first(self):
        r = self.reply(['|', '|move|p1a: Jolteon|Thunderbolt|p2a: Dragonite', '|-damage|p2a: Dragonite|0 fnt', '|faint|p2a: Dragonite'])
        self.assertEqual(r.cause, 'ko_first')

    def test_cant(self):
        r = self.reply(['|', '|move|p1a: Breloom|Spore|p2a: Snorlax', '|-status|p2a: Snorlax|slp|[from] move: Spore', '|cant|p2a: Snorlax|slp', '|', '|upkeep'])
        self.assertEqual((r.cause, r.detail), ('sleep_para_flinch', 'cant:slp'))
        for reason in ('par', 'flinch', 'frz'):self.assertEqual(self.reply([f'|cant|p2a: Snorlax|{reason}']).cause, 'sleep_para_flinch')
        for reason in ('recharge', 'ability: Truant', 'Disable|Body Slam', 'nopp'):self.assertEqual(self.reply([f'|cant|p2a: Snorlax|{reason}']).cause, 'other')
        # Asleep, then KO'd: the sleep decided it.
        r = self.reply(['|cant|p2a: Snorlax|slp', '|move|p1a: Machamp|Cross Chop|p2a: Snorlax', '|-damage|p2a: Snorlax|0 fnt', '|faint|p2a: Snorlax'])
        self.assertEqual(r.cause, 'sleep_para_flinch')
        # Our Roar's drag is never its choice.
        r = self.reply(['|cant|p2a: Snorlax|slp', '|move|p1a: Skarmory|Roar|p2a: Snorlax', '|drag|p2a: Gengar|Gengar, M|100/100'])
        self.assertEqual(r.cause, 'sleep_para_flinch')

    def test_forced_switch_rounds(self):
        # We replace alone: Foul Play waits.
        r = self.reply(['|', '|t:|1700000000', '|switch|p1a: Starmie|Starmie|282/282', '|', '|upkeep'], forced=True)
        self.assertEqual((r.kind, r.cause), ('wait', None))
        # Both replace (double faint): its switch is its reply.
        r = self.reply(['|', '|switch|p1a: Starmie|Starmie|282/282', '|switch|p2a: Snorlax|Snorlax, F|100/100', '|', '|upkeep'], forced=True)
        self.assertEqual((r.kind, r.id), ('switch', 'snorlax'))
        # Our Baton Pass replacement mid-turn: Foul Play's later move belongs to the turn-start decision, not this one.
        r = self.reply(['|', '|switch|p1a: Jolteon|Jolteon|292/292', '|move|p2a: Snorlax|Body Slam|p1a: Jolteon'], forced=True)
        self.assertEqual(r.kind, 'wait')

    def test_no_event_and_owner_p2(self):
        self.assertEqual(self.reply(['|', '|move|p1a: Starmie|Surf|p2a: Snorlax']).detail, 'no_fp_action')
        r = self.reply(['|move|p2a: Starmie|Surf|p1a: Snorlax', '|move|p1a: Snorlax|Rest|p1a: Snorlax'], owner='p2')
        self.assertEqual(r.id, 'rest')


class CaptureAlignmentTests(unittest.TestCase):
    def setUp(self):self.directory = TemporaryDirectory();self.root = Path(self.directory.name);S.load_capture.cache_clear()

    def tearDown(self):self.directory.cleanup()

    def battle(self):
        start = ['|player|p1|Us|', '|player|p2|FP|', '|gen|3', '|start', '|switch|p1a: Magikarp|Magikarp, L5, M|19/19',
                 '|switch|p2a: Snorlax|Snorlax, F|100/100', '|turn|1']
        events = [start, [request('p1')], 'ROOT',
                  ['|', '|t:|1', '|move|p2a: Snorlax|Body Slam|p1a: Magikarp', '|-damage|p1a: Magikarp|0 fnt', '|faint|p1a: Magikarp'],
                  [request('p1', 'forceSwitch')], 'ROOT',
                  ['|', '|t:|2', '|switch|p1a: Starmie|Starmie|282/282', '|', '|upkeep', '|turn|2'], [request('p1')], 'ROOT',
                  ['|', '|switch|p2a: Pikachu|Pikachu, L5, F|21/21', '|move|p1a: Starmie|Thunderbolt|p2a: Pikachu',
                   '|-damage|p2a: Pikachu|0 fnt', '|faint|p2a: Pikachu'], [request('p1', 'wait')],
                  ['|', '|switch|p2a: Snorlax|Snorlax, F|100/100', '|', '|upkeep', '|turn|3'], [request('p1')], 'ROOT',
                  ['|', '|move|p1a: Starmie|Surf|p2a: Snorlax', '|move|p2a: Snorlax|Curse|p2a: Snorlax', '|', '|upkeep', '|turn|4']]
        path = self.root / '0.jsonl';write_capture(path, 'p1', events)
        return path

    def test_decisions_line_up_with_requests(self):
        capture = S.load_capture(str(self.battle()))
        self.assertIsNone(capture.error)
        self.assertEqual(capture.roots, {0: 1, 1: 2, 2: 3, 3: 5})
        expected = {(0, 1, 'move'): ('move', 'bodyslam'), (1, 2, 'switch'): ('wait', None), (2, 3, 'move'): ('switch', 'pikachu'),
                    (3, 5, 'move'): ('move', 'curse')}
        for (index, count, kind), want in expected.items():
            r = S.reply_for(capture, dict(owner='p1', decision_index=index, request_count=count, request_kind=kind))
            self.assertEqual((r.kind, r.id), want, (index, count))
        window = S.turn_window(capture.lines, 1)[1]
        self.assertNotIn('|turn|2', window)
        self.assertIn(request('p1', 'forceSwitch'), window)   # the turn runs past an owner request inside it

    def test_misaligned_missing_and_invalid_captures(self):
        capture = S.load_capture(str(self.battle()))
        r = S.reply_for(capture, dict(owner='p1', decision_index=3, request_count=4, request_kind='move'))
        self.assertEqual((r.cause, r.detail), ('other', 'request_misaligned'))
        r = S.reply_for(capture, dict(owner='p2', decision_index=0, request_count=1, request_kind='move'))
        self.assertEqual(r.detail, 'owner_mismatch')
        r = S.reply_for(capture, dict(owner='p1', decision_index=9, request_count=9, request_kind='move'))
        self.assertEqual(r.detail, 'request_missing')
        self.assertEqual(S.load_capture(str(self.root / 'absent.jsonl')).error, 'capture_missing')
        tampered = self.root / '1.jsonl';tampered.write_bytes(self.battle().read_bytes().replace(b'Body Slam', b'Body Slap'))
        self.assertEqual(S.load_capture(str(tampered)).error, 'capture_invalid')


class MappingTests(unittest.TestCase):
    def setUp(self):
        self.grid = grid_of(lambda a, b: (a - b) / 10)
        self.legal = tuple([True] * 9)

    def map(self, reply, **kw):
        facts = S.normalize(raw_row([world_row(self.grid, **kw)]))['worlds'][0]
        world = S.world_prices(dict(worlds=[facts]))[0]
        return S.map_world(reply, facts, world, self.legal)

    def test_move(self):
        self.assertEqual(self.map(S.Reply('move', 'rest', None, '')), (2, None))
        self.assertEqual(self.map(S.Reply('move', 'bodyslam', None, ''), move_targets=(2, 0, 1, 3)), (1, None))
        self.assertEqual(self.map(S.Reply('move', 'hiddenpower', None, '')), (None, 'move_missing'))
        self.assertEqual(self.map(S.Reply('move', 'rest', None, ''), legal=legal_mask(0, 1, 3, 4)), (None, 'move_not_legal'))
        self.assertEqual(self.map(S.Reply('move', 'struggle', None, '')), (0, None))
        self.assertEqual(self.map(S.Reply('move', 'struggle', None, ''), legal=legal_mask(4, 5)), (None, 'move_not_legal'))

    def test_switch(self):
        self.assertEqual(self.map(S.Reply('switch', 'skarmory', None, '')), (4, None))
        self.assertEqual(self.map(S.Reply('switch', 'gengar', None, ''), switch_targets=(2, 1, 3, 4, 5)), (4, None))
        self.assertEqual(self.map(S.Reply('switch', 'mew', None, '')), (None, 'species_missing'))
        self.assertEqual(self.map(S.Reply('switch', 'gengar', None, ''), fainted=[False, False, True, False, False, False]), (None, 'species_fainted'))
        self.assertEqual(self.map(S.Reply('switch', 'gengar', None, ''), legal=legal_mask(0, 1, 2, 3, 4)), (None, 'switch_not_legal'))

    def test_wait_and_requested_mismatches(self):
        wait = S.Reply('wait', None, None, '')
        self.assertEqual(self.map(wait, requested=False), (0, None))
        self.assertEqual(self.map(wait), (None, 'wait_but_requested'))
        self.assertEqual(self.map(S.Reply('move', 'rest', None, ''), requested=False), (None, 'reply_but_waiting'))
        self.assertEqual(self.map(S.Reply('switch', 'skarmory', None, ''), requested=False), (None, 'reply_but_waiting'))

    def test_charge_move_is_lane_0(self):
        self.assertEqual(self.map(S.Reply('move', 'solarbeam', None, ''), charge='solarbeam', legal=legal_mask(0)), (0, None))
        self.assertEqual(self.map(S.Reply('move', 'solarbeam', None, ''), legal=legal_mask(0)), (None, 'move_missing'))

    def test_adapter_reads_the_reprice_row(self):
        decision = S.normalize(raw_row([world_row(self.grid, fainted=[False, True, False, False, False, False])], greedy=3, live_action=5,
                                       root_logits=[.5] * 9, replay_action=5))
        self.assertEqual((decision['ordinal'], decision['request_count'], decision['greedy_action'], decision['live_action'], decision['replay_action'],
                          decision['request_kind'], decision['source'], decision['replay_status']), (0, 1, 3, 5, 5, 'move', 'search', 'COMPLETE'))
        self.assertEqual(decision['root_logits'], [.5] * 9)
        world = decision['worlds'][0]
        self.assertEqual(world['party_species'], list(PARTY))
        self.assertEqual(world['fainted'], [False, True, False, False, False, False])
        self.assertEqual((world['active_moves'], world['move_targets'], world['requested']), (list(MOVES), [0, 1, 2, 3], True))

    def test_unpriced_column(self):
        self.grid[3][2] = None   # our legal action 3 against lane 2 (rest) is unpriced
        self.assertEqual(self.map(S.Reply('move', 'rest', None, ''), kept=legal_mask(0, 1, 3, 4, 5, 6, 7, 8), legal=[True] * 9), (None, 'unpriced'))


class ScoringTests(unittest.TestCase):
    def score(self, worlds, reply, *, greedy=0, logits=None, legal=None, target=None):
        decision = S.normalize(raw_row(worlds, greedy=greedy, root_logits=logits, root_legal=legal))
        return S.score_decision(decision, reply, target, collections.Counter())

    def test_realized_and_pi_filled(self):
        legal = legal_mask(0, 1, 2)
        rng = random.Random(7)
        values = [grid_of(lambda a, b: round(rng.uniform(-1, 1), 3)) for _ in range(4)]
        logits = [[rng.uniform(-2, 2) for _ in range(9)] for _ in range(4)]
        worlds = [world_row(values[0], logits=logits[0]), world_row(values[1], logits=logits[1], moves=('surf', 'curse', 'rest', 'toxic')),
                  world_row(values[2], logits=logits[2], legal=legal_mask(0, 1, 3)), world_row(values[3], logits=logits[3], move_targets=(1, 0, 2, 3))]
        out = self.score(worlds, S.Reply('move', 'bodyslam', None, ''), legal=legal)
        # Body Slam: world 0 lane 0; world 1 lacks it; world 2 lane 0; world 3 holds slot 0 at lane 1.
        self.assertEqual(out['mapped_worlds'], [[0, 0], [2, 0], [3, 1]])
        self.assertEqual(out['world_drops'], {1: 'move_missing'})
        for a in range(9):
            want = (values[0][a][0] + values[2][a][0] + values[3][a][1]) / 3 if legal[a] else None
            if want is None:self.assertIsNone(out['R']['dropped'][a])
            else:self.assertAlmostEqual(out['R']['dropped'][a], want, places=12)
        self.assertEqual(out['R']['pi_filled'], out['R']['dropped'])
        # Censored: pi_filled R is averaging's q, and equals the hand computation sum_w 1/W sum_b pi_w(b) G_w[a][b].
        censored = self.score(worlds, S.censored('ko_first', ''), legal=legal)
        self.assertIsNone(censored['R']['dropped'])
        average = hedge.averaging(S.world_prices(S.normalize(raw_row(worlds))), tuple(legal))
        self.assertEqual(censored['R']['pi_filled'], list(average.q))
        for a in range(9):
            if not legal[a]:self.assertIsNone(censored['R']['pi_filled'][a]);continue
            hand = 0.
            for w in range(4):
                weights = [math.exp(x) for x in logits[w]];total = sum(weights)
                hand += sum(weights[b] / total * values[w][a][b] for b in range(9)) / 4
            self.assertAlmostEqual(censored['R']['pi_filled'][a], hand, places=12)
        # Unmapped: no world holds the move; reported apart from censored, filled like it.
        unmapped = self.score(worlds, S.Reply('move', 'splash', None, ''), legal=legal)
        self.assertTrue(unmapped['unmapped']);self.assertFalse(unmapped['observed'])
        self.assertEqual(unmapped['R']['pi_filled'], list(average.q))

    def test_d_against_greedy_and_illegal_none(self):
        worlds = waiting_worlds([{0: .5, 1: .1, 4: -.2}] * 2)
        out = self.score(worlds, S.Reply('wait', None, None, ''), greedy=1, legal=legal_mask(0, 1, 4))
        ops = out['operators']
        self.assertEqual(ops['averaging']['choice'], 0)
        self.assertAlmostEqual(ops['averaging']['D']['dropped'], .4)
        self.assertEqual(ops['greedy']['D'], dict(dropped=0., pi_filled=0.))
        self.assertIsNone(out['R']['dropped'][2])

    def test_unscorable(self):
        worlds = waiting_worlds([{0: .5}] * 2)
        with self.assertRaises(S.Unscorable) as error:
            S.score_decision(S.normalize(raw_row(worlds, status='REFUSED')), None, None, None)
        self.assertEqual(error.exception.reason, 'replay_REFUSED')
        worlds[1]['grid'][0][0] = None
        with self.assertRaises(S.Unscorable) as error:
            self.score(worlds, S.Reply('wait', None, None, ''), legal=legal_mask(0, 1))
        self.assertEqual(error.exception.reason, 'unpriced_cell')


class OperatorTests(unittest.TestCase):
    def choices(self, worlds, *, greedy, logits=None, legal, target=None):
        decision = S.normalize(raw_row(worlds, greedy=greedy, root_logits=logits, root_legal=legal))
        out = S.score_decision(decision, S.censored('other', ''), target, collections.Counter())
        return {name: op['choice'] for name, op in out['operators'].items()}, out

    def test_averaging_hedge_and_blend(self):
        # Action 0 best-responds to the predicted reply 0 (prior .9); action 1 is safe against both replies.
        g = grid_of(lambda a, b: {(0, 0): 1., (0, 1): -.2}.get((a, b), .2))
        logits = [math.log(.9), math.log(.1)] + [0.] * 7
        worlds = [world_row(g, kept=legal_mask(0, 1), logits=logits)] * 2
        c, _ = self.choices(worlds, greedy=1, legal=legal_mask(0, 1))
        self.assertEqual((c['averaging'], c['bayes_hedge:lam=0'], c['blend:lam=0.5']), (0, 1, 0))

    def test_prior_regularized(self):
        worlds = waiting_worlds([{0: .5, 1: .1}] * 2)
        logits = [-3., 0.] + [0.] * 7
        c, _ = self.choices(worlds, greedy=1, logits=logits, legal=legal_mask(0, 1))
        self.assertEqual(c['prior_regularized:c=0.02'], 0)
        self.assertEqual(c['prior_regularized:c=0.5'], 1)

    def test_noise_gated(self):
        # gap 0.1 +- 0.3 per world: se = 0.3 * sqrt(16/15) / 4 = 0.0775, so k=1 plays averaging and k=2 greedy.
        worlds = waiting_worlds([{0: .1 + (.3 if w % 2 else -.3), 1: 0.} for w in range(16)])
        c, _ = self.choices(worlds, greedy=1, legal=legal_mask(0, 1))
        self.assertEqual((c['averaging'], c['noise_gated:k=1'], c['noise_gated:k=2']), (0, 0, 1))
        # A consistent gap passes both; agreement with greedy is greedy.
        worlds = waiting_worlds([{0: .1 + .001 * w, 1: 0.} for w in range(16)])
        c, _ = self.choices(worlds, greedy=1, legal=legal_mask(0, 1))
        self.assertEqual((c['noise_gated:k=1'], c['noise_gated:k=2']), (0, 0))
        c, _ = self.choices(worlds, greedy=0, legal=legal_mask(0, 1))
        self.assertEqual(c['noise_gated:k=2'], 0)

    def test_world_lcb(self):
        # Action 0: mean 0.3, per-world spread +-0.6 (se ~0.155); action 1: 0.2 in every world.
        worlds = waiting_worlds([{0: .3 + (.6 if w % 2 else -.6), 1: .2} for w in range(16)])
        c, _ = self.choices(worlds, greedy=1, legal=legal_mask(0, 1))
        self.assertEqual((c['averaging'], c['world_lcb:kappa=0.5'], c['world_lcb:kappa=2']), (0, 0, 1))

    def test_world_lcb_kappa_zero_is_averaging(self):
        rng = random.Random(3)
        for _ in range(20):
            legal = legal_mask(*sorted(rng.sample(range(9), rng.randint(2, 9))))
            worlds = [world_row(grid_of(lambda a, b: round(rng.uniform(-1, 1), 2)), logits=[rng.uniform(-3, 3) for _ in range(9)]) for _ in range(16)]
            prices = S.world_prices(S.normalize(raw_row(worlds)))
            gs = hedge.grids(prices, tuple(legal));average = hedge.averaging(prices, tuple(legal))
            qw = S.per_world_q(gs)
            spread = {a: statistics.stdev([x[a] for x in qw]) / 4 for a in qw[0]}
            self.assertEqual(S.world_lcb(average.q, spread, 0), hedge.top_action(average.strategy))
            for a in qw[0]:self.assertAlmostEqual(statistics.fmean(x[a] for x in qw), average.q[a], places=12)

    def test_target_argmax(self):
        worlds = waiting_worlds([{0: .5, 1: .1, 2: .0}] * 2)
        target = [.1, .2, .7, .0, .0, .0, .0, .0, .9]   # lane 8 is illegal
        c, out = self.choices(worlds, greedy=1, legal=legal_mask(0, 1, 2), target=target)
        self.assertEqual(c['target_argmax'], 2);self.assertFalse(out['target_fallback'])
        c, out = self.choices(worlds, greedy=1, logits=[0., 1., 0.] + [0.] * 6, legal=legal_mask(0, 1, 2))
        self.assertEqual(c['target_argmax'], 1);self.assertTrue(out['target_fallback'])   # argmax of logits + q: 1.1 > 0.5


class StatisticsTests(unittest.TestCase):
    def test_cluster_ci_by_hand(self):
        out = S.cluster_ci({'a': [1., 2.], 'b': [0.], 'c': [-1., 3., 1.]})
        # Battle sums 3, 0, 3: per-battle mean 2, sd sqrt(3), se 1; t(0.95, 2 df) = 2.919986; 6 decisions over 3 battles.
        t = 2.919985580355516
        self.assertEqual((out['battles'], out['decisions']), (3, 6))
        self.assertAlmostEqual(out['per_battle_mean'], 2.);self.assertAlmostEqual(out['per_battle_se'], 1.)
        self.assertAlmostEqual(out['mean'], 1.)
        for got, want in zip(out['per_battle_ci90'], (2 - t, 2 + t)):self.assertAlmostEqual(got, want, places=9)
        for got, want in zip(out['ci90'], ((2 - t) / 2, (2 + t) / 2)):self.assertAlmostEqual(got, want, places=9)
        self.assertIsNone(S.cluster_ci({'a': [1., 2.]})['ci90'])

    def records(self):
        """Six battles (seeds: ordinals 0-2 even, 3-5 odd), two decisions each; 'bayes_hedge:lam=0' gains 0.1 where it
        overrides; the second decision of each battle is censored (dropped excludes it)."""
        out = []
        for ordinal in range(6):
            for k in range(2):
                ops = {name: (0, (0., 0.), '>=50%') for name in S.SETTINGS}
                ops['averaging'] = (1, (None if k else -.2 - .01 * ordinal, -.1 - .01 * ordinal), '5-20%')
                ops['bayes_hedge:lam=0'] = (2, (None if k else .1 + .01 * ordinal, .1 + .02 * ordinal), '1-5%')
                out.append(dict(ordinal=ordinal, half='even' if ordinal < 3 else 'odd', kind='move' if k else 'switch', observed=not k, ops=ops))
        return out

    def test_metrics_populations(self):
        table = S.metrics(self.records())
        self.assertEqual((table['pi_filled']['all']['decisions'], table['dropped']['all']['decisions']), (12, 6))
        self.assertEqual(table['pi_filled']['even']['battles'], 3)
        m = table['dropped']['odd']['operators']['bayes_hedge:lam=0']
        self.assertEqual((m['decisions'], m['override']), (3, 1.))
        self.assertAlmostEqual(m['mean'], .14)
        self.assertEqual(table['pi_filled']['all']['by_request_kind']['move']['averaging']['decisions'], 6)
        self.assertEqual(table['pi_filled']['all']['by_prior_band']['1-5%']['bayes_hedge:lam=0']['decisions'], 12)
        self.assertEqual(table['pi_filled']['all']['by_prior_band']['>=50%']['bayes_hedge:lam=0']['decisions'], 0)
        out = S.decide(table, True)
        self.assertTrue(out['gate']['passed'])
        self.assertEqual(out['selection']['chosen'], 'bayes_hedge:lam=0')
        self.assertEqual(out['verdict'], 'PASS')

    @staticmethod
    def table(averaging_ci, candidates, odd):
        """A metrics table: candidates {name: (even override, even mean)}; odd {name: (pi lower, dropped lower, override)}."""
        def m(override=0., mean=0., ci=(-1., 1.)):return dict(override=override, mean=mean, ci90=list(ci), battles=10, decisions=100)
        table = {s: {h: dict(operators={n: m() for n in S.SETTINGS}) for h in S.HALVES} for s in S.SCORINGS}
        table['pi_filled']['all']['operators']['averaging'] = m(mean=sum(averaging_ci) / 2, ci=averaging_ci)
        for name, (override, mean) in candidates.items():table['pi_filled']['even']['operators'][name] = m(override, mean)
        for name, (pi, dropped, override) in odd.items():
            table['pi_filled']['odd']['operators'][name] = m(override, pi + .1, (pi, pi + .2))
            table['dropped']['odd']['operators'][name] = m(override, dropped + .1, (dropped, dropped + .2))
        return table

    def test_gate(self):
        out = S.decide(self.table((-.05, .001), {'blend:lam=0.5': (.2, .3)}, {}), True)
        self.assertEqual(out['verdict'], 'GATE_FAILED');self.assertIsNone(out['selection']['chosen'])
        self.assertEqual(S.decide(self.table((-.05, -.001), {}, {}), False)['verdict'], 'PARITY_FAILED')

    def test_selection_needs_override(self):
        candidates = {'blend:lam=0.5': (.04, .9), 'noise_gated:k=1': (.05, .02), 'world_lcb:kappa=1': (.3, .01), 'target_argmax': (.5, .02)}
        out = S.decide(self.table((-.05, -.01), candidates, {'noise_gated:k=1': (.01, .02, .06)}), True)
        self.assertEqual(out['selection']['chosen'], 'noise_gated:k=1')   # blend's override is below 5%; a tie keeps declared order
        self.assertEqual(out['verdict'], 'PASS')
        self.assertEqual(S.decide(self.table((-.05, -.01), {}, {}), True)['verdict'], 'NO_CANDIDATE')

    def test_confirmation(self):
        candidates = {'world_lcb:kappa=2': (.1, .05)}
        for odd, passed in (((.01, .02, .06), True), ((.01, -.001, .06), False), ((0., .02, .06), False), ((.01, .02, .049), False)):
            out = S.decide(self.table((-.05, -.01), candidates, {'world_lcb:kappa=2': odd}), True)
            self.assertEqual(out['confirmation']['passed'], passed, odd)
            self.assertEqual(out['verdict'], 'PASS' if passed else 'FAIL')


class EndToEndTests(unittest.TestCase):
    """Two battles (an even and an odd seed) through main(): captures, telemetry, results and a rows file."""

    def setUp(self):self.directory = TemporaryDirectory();self.root = Path(self.directory.name);S.load_capture.cache_clear()

    def tearDown(self):self.directory.cleanup()

    def test_main(self):
        run = self.root / 'run';rows = []
        for sub in ('results', 'telemetry', 'owning-wire'):(run / sub).mkdir(parents=True)
        rng = random.Random(11);legal = legal_mask(0, 1, 2, 4)
        for ordinal, seed in ((0, 10), (1, 11)):
            (run / 'results' / f'{ordinal}.json').write_text(json.dumps(dict(status='COMPLETE', cell=dict(seed=seed, arm='DN2W16', ordinal=ordinal))))
            write_capture(run / 'owning-wire' / f'{ordinal}.jsonl', 'p1', [
                ['|start', '|switch|p1a: Starmie|Starmie|282/282', '|switch|p2a: Snorlax|Snorlax, F|100/100', '|turn|1'], [request('p1')], 'ROOT',
                ['|', '|move|p1a: Starmie|Surf|p2a: Snorlax', '|move|p2a: Snorlax|Rest|p2a: Snorlax', '|', '|upkeep', '|turn|2'], [request('p1')], 'ROOT',
                ['|', '|move|p1a: Starmie|Surf|p2a: Snorlax', '|cant|p2a: Snorlax|slp', '|', '|upkeep', '|turn|3'], [request('p1')]], ordinal=ordinal)
            telemetry = []
            for index in range(2):
                worlds = [world_row(grid_of(lambda a, b: round(rng.uniform(-1, 1), 3)), logits=[rng.uniform(-2, 2) for _ in range(9)]) for _ in range(16)]
                prices = S.world_prices(S.normalize(raw_row(worlds)))
                chosen = hedge.top_action(hedge.averaging(prices, tuple(legal)).strategy)
                rows.append(raw_row(worlds, ordinal=ordinal, decision_index=index, request_count=index + 1, greedy=0, live_action=chosen,
                                    root_logits=[1., 0., 0., 0., 0., 0., 0., 0., 0.], root_legal=legal, replay_action=chosen))
                telemetry.append(dict(decision_index=index, search=dict(policy_target=[.1, .2, .3, 0., .4, 0., 0., 0., 0.])))
            (run / 'telemetry' / f'{ordinal}.jsonl').write_text(''.join(json.dumps(x) + '\n' for x in telemetry))
        rows.append(raw_row(None, ordinal=1, decision_index=5, status='REFUSED'))
        path = self.root / 'rows-0.jsonl';path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
        out = self.root / 'out'
        with redirect_stdout(io.StringIO()):
            self.assertEqual(S.main(['--rows', str(path), '--run-directory', str(run), '--out', str(out)]), 0)
        summary = json.loads((out / 'summary.json').read_text())
        cover = summary['coverage']
        self.assertEqual((cover['searched_decisions'], cover['replay_complete'], cover['scored']), (5, 4, 4))
        self.assertEqual(cover['excluded'], {'replay_REFUSED': 1})
        self.assertEqual(cover['parity']['rate'], 1.)
        self.assertEqual(cover['replies'], {'move': 2, 'censored': 2})
        self.assertEqual(cover['censored']['by_cause']['sleep_para_flinch']['count'], 2)
        self.assertEqual(cover['mapped_worlds']['histogram'], {'16': 2})
        self.assertEqual(cover['halves']['even'], dict(battles=1, decisions=2))
        self.assertIn(summary['verdict'], ('GATE_FAILED', 'NO_CANDIDATE', 'PASS', 'FAIL'))
        decisions = [json.loads(x) for x in (out / 'decisions.jsonl').read_text().splitlines()]
        self.assertEqual(len(decisions), 5)
        first = decisions[0]
        self.assertEqual(first['reply'], dict(kind='move', id='rest', cause=None, detail='move'))
        self.assertEqual(len(first['mapped_worlds']), 16)
        self.assertEqual(set(first['operators']), set(S.SETTINGS))
        self.assertEqual(first['operators']['target_argmax']['choice'], 4)
        self.assertTrue((out / 'summary.md').read_text().startswith('# RR'))
        with self.assertRaises(FileExistsError):S.score_run([str(path)], str(run), str(out))


if __name__ == '__main__':
    unittest.main()
