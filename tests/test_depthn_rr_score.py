import collections
from contextlib import redirect_stderr, redirect_stdout
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
ALL = (True,) * 9


def legal_mask(*actions):return [a in actions for a in range(9)]


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


def real_grid(fn, *, root_legal, requested, kept):
    """A world's grid as the live pricer leaves it: None on our illegal rows; a requested world prices only its kept reply
    columns, a waiting world only column 0."""
    columns = {b for b in range(9) if kept[b]} if requested else {0}
    return [[fn(a, b) if root_legal[a] and b in columns else None for b in range(9)] for a in range(9)]


def lane_labels(moves, move_targets, switch_targets, party, legal, usable, charge):
    """scripts/depthn_rr_reprice.py's lane_labels on these facts (what each legal opponent lane plays)."""
    out, struggle = [], not any(usable)
    for b in range(4):
        if not legal[b]:out.append(None)
        elif b == 0 and struggle:out.append('move:struggle')
        elif b == 0 and charge is not None:out.append(f'move:{charge}')
        else:out.append(f'move:{moves[move_targets[b]]}' if 0 <= move_targets[b] < len(moves) else None)
    for i, slot in enumerate(switch_targets):out.append(None if not legal[4 + i] else f'switch:{party[slot]}' if 0 <= slot < len(party) else None)
    return out


def world_row(fn, *, root_legal=ALL, requested=True, kept=None, legal=None, logits=None, moves=MOVES, party=PARTY, fainted=None,
              move_targets=(0, 1, 2, 3), switch_targets=(1, 2, 3, 4, 5), charge=None, usable=(True,) * 4, lanes=None):
    """One world as scripts/depthn_rr_reprice.py writes it (describe(): prices_fields, opponent_facts, PeerLanes, lane_labels).
    kept (the replies turn 1 enumerated) equals the peer legal lanes unless both are given; a waiting world has neither."""
    if kept is None and legal is None:kept = legal = [requested] * 9
    kept = list(kept if kept is not None else legal);legal = list(legal if legal is not None else kept)
    fainted = list(fainted or [False] * 6)
    if lanes is None:lanes = lane_labels(moves, move_targets, switch_targets, party, legal, usable, charge)
    return dict(world=0, index=0, seed=1, kernel_seed=2, refusals=[], constructed=True, priced=True,
                opponent_requested=requested, kept=kept, root_peer_logits=list(logits or [0.] * 9),
                grid=real_grid(fn, root_legal=root_legal, requested=requested, kept=kept),
                opponent=dict(side='p2', active_slot=0, active_species=party[0], active_moves=list(moves), active_move_engine_ids=[1, 2, 3, 4],
                              charge_move=charge, party=[dict(slot=i, species=sp, species_engine_id=i + 1, fainted=fainted[i], hp=0 if fainted[i] else 100,
                                                              max_hp=100) for i, sp in enumerate(party)]),
                peer=dict(legal=legal, move_targets=list(move_targets), switch_targets=list(switch_targets), error=0, usable=list(usable), lanes=lanes),
                kept_matches_peer_legal=kept == legal if requested else None, requested_matches_peer=requested == any(legal))


def raw_row(worlds, *, ordinal=0, decision_index=0, request_count=1, owner='p1', root_logits=None, root_legal=ALL, greedy=0,
            live_action=None, kind='move', status='COMPLETE', replay_action=None, replay_q=None, source='search', replay_cause=None,
            live_cause=None):
    """One rows-<shard>.jsonl row as scripts/depthn_rr_reprice.py's decision_row writes it (the fields this scorer reads)."""
    return dict(schema='pokezero.depthn-rr-reprice.row.v1', shard=0, ordinal=ordinal, seat=0, seed=10, battle_id=f'battle-{ordinal}', owner=owner,
                stream_id=1, decision_index=decision_index, generation=3, prefix_sha256='0' * 64, request_count=request_count, capture_turn=1,
                replay=dict(status=status, cause=replay_cause, action_index=replay_action, q_values=replay_q, policy_target=None),
                live=dict(source=source, cause=live_cause, request_kind=kind, turn=1, played_action_index=live_action, greedy_action_index=greedy,
                          prefix_sha256='0' * 64, search_status='COMPLETE', search_cause=None, search_action_index=live_action, search_q_values=replay_q),
                live_problem=None, parity=dict(live_found=True, live_searched=True, replay_complete=True, status_match=True, action_equal=None, q_equal=None),
                root=None if worlds is None else dict(root_logits=list(root_logits or [0.] * 9), root_legal=list(root_legal)),
                prepared=None, worlds=worlds, notes=[], kept_disagreements=0, capture_ok=True)


def waiting_worlds(columns, root_legal):
    """Waiting worlds (Q_w(a) = G_w[a][0]): columns[w] = {action: value} on our legal actions."""
    return [world_row(lambda a, b, c=c: c.get(a, 0.), requested=False, root_legal=root_legal) for c in columns]


def decision(worlds, **kw):return S.normalize(raw_row(worlds, **kw))


class ReplyParsingTests(unittest.TestCase):
    """Foul Play is p2 (the owner p1) unless stated; line sequences are the pinned gen3 Showdown sim's player stream
    (pokerena/vendor/pokemon-showdown f76228a13)."""

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
        self.assertEqual(self.reply(['|switch|p2a: Deoxys|Deoxys-Speed, L73|100/100']).id, 'deoxysspeed')

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
        r = self.reply(['|move|p1a: Tyranitar|Pursuit|p2a: Snorlax', '|-damage|p2a: Snorlax|85/100', '|move|p2a: Snorlax|Body Slam|p1a: Tyranitar'])
        self.assertEqual(r.id, 'bodyslam')
        r = self.reply(['|-activate|p1a: Starmie|move: Pursuit', '|move|p2a: Tyranitar|Pursuit|p1a: Starmie|[from] Pursuit',
                        '|switch|p1a: Snorlax|Snorlax|400/400'])
        self.assertEqual((r.kind, r.id), ('move', 'pursuit'))
        # A Foul Play switch after a move line that is not Pursuit's is ambiguous: censored.
        self.assertEqual(self.reply(['|move|p1a: Starmie|Surf|p2a: Snorlax', '|switch|p2a: Gengar|Gengar, M|100/100']).detail, 'switch_after_move')

    def test_ko_first(self):
        r = self.reply(['|', '|move|p1a: Jolteon|Thunderbolt|p2a: Dragonite', '|-damage|p2a: Dragonite|0 fnt', '|faint|p2a: Dragonite'])
        self.assertEqual(r.cause, 'ko_first')

    def test_cant_without_a_move_is_censored(self):
        r = self.reply(['|', '|move|p1a: Breloom|Spore|p2a: Snorlax', '|-status|p2a: Snorlax|slp|[from] move: Spore', '|cant|p2a: Snorlax|slp', '|', '|upkeep'])
        self.assertEqual((r.cause, r.detail), ('sleep_para_flinch', 'cant:slp'))
        for reason in ('par', 'flinch', 'frz'):self.assertEqual(self.reply([f'|cant|p2a: Snorlax|{reason}']).cause, 'sleep_para_flinch')
        for reason in ('recharge', 'ability: Truant', 'Attract', 'nopp'):self.assertEqual(self.reply([f'|cant|p2a: Snorlax|{reason}']).cause, 'other')
        r = self.reply(['|cant|p2a: Snorlax|slp', '|move|p1a: Machamp|Cross Chop|p2a: Snorlax', '|-damage|p2a: Snorlax|0 fnt', '|faint|p2a: Snorlax'])
        self.assertEqual(r.cause, 'sleep_para_flinch')
        r = self.reply(['|cant|p2a: Snorlax|slp', '|move|p1a: Skarmory|Roar|p2a: Snorlax', '|drag|p2a: Gengar|Gengar, M|100/100'])
        self.assertEqual(r.cause, 'sleep_para_flinch')

    def test_cant_naming_the_chosen_move_is_its_reply(self):
        # The sim's own lines: our Taunt, Disable and Imprison block the move Foul Play chose this turn.
        for line, move in (('|cant|p2a: Snorlax|move: Taunt|Thunder Wave', 'thunderwave'), ('|cant|p2a: Snorlax|Disable|Body Slam', 'bodyslam'),
                           ('|cant|p2a: Snorlax|move: Imprison|Body Slam', 'bodyslam'), ('|cant|p2a: Snorlax|Focus Punch|Focus Punch', 'focuspunch')):
            r = self.reply(['|', '|move|p1a: Gengar|Taunt|p2a: Snorlax', '|-start|p2a: Snorlax|move: Taunt', line, '|', '|upkeep'])
            self.assertEqual((r.kind, r.id, r.cause, r.detail), ('move', move, None, 'cant_named:' + line.split('|')[3]))
        # Focus Punch losing focus: the sim writes its [still] move line before the |cant|.
        r = self.reply(['|', '|-singleturn|p2a: Snorlax|move: Focus Punch', '|move|p1a: Gengar|Thunderbolt|p2a: Snorlax', '|-damage|p2a: Snorlax|220/482 par',
                        '|move|p2a: Snorlax|Focus Punch||[still]', '|cant|p2a: Snorlax|Focus Punch|Focus Punch', '|', '|upkeep'])
        self.assertEqual((r.id, r.detail), ('focuspunch', 'move'))
        # Damp: the holder's |cant| names OUR Explosion with [of]; Foul Play's own move follows.
        r = self.reply(['|move|p1a: Gengar|Explosion|p2a: Golduck|[still]', '|cant|p2a: Golduck|ability: Damp|Explosion|[of] p1a: Gengar',
                        '|move|p2a: Golduck|Surf|p1a: Gengar'])
        self.assertEqual(r.id, 'surf')
        self.assertEqual(self.reply(['|cant|p2a: Golduck|ability: Damp|Explosion|[of] p1a: Gengar']).detail, 'no_fp_action')
        r = self.reply(['|cant|p2a: Golduck|ability: Damp|Explosion|[of] p1a: Gengar', '|move|p1a: Gengar|Thunderbolt|p2a: Golduck', '|faint|p2a: Golduck'])
        self.assertEqual(r.cause, 'ko_first')
        self.assertIsNone(S.cant_named_move('|cant|p2a: Snorlax|par'.split('|')))

    def test_forced_switch_rounds(self):
        r = self.reply(['|', '|t:|1700000000', '|switch|p1a: Starmie|Starmie|282/282', '|', '|upkeep'], forced=True)
        self.assertEqual((r.kind, r.cause), ('wait', None))
        r = self.reply(['|', '|switch|p1a: Starmie|Starmie|282/282', '|switch|p2a: Snorlax|Snorlax, F|100/100', '|', '|upkeep'], forced=True)
        self.assertEqual((r.kind, r.id), ('switch', 'snorlax'))
        # The scan ends at the first move line: a Foul Play switch after it is not this round's.
        r = self.reply(['|', '|switch|p1a: Jolteon|Jolteon|292/292', '|move|p2a: Celebi|Baton Pass|p2a: Celebi', '|switch|p2a: Snorlax|Snorlax, F|100/100'],
                       forced=True)
        self.assertEqual(r.kind, 'wait')

    def test_forced_switch_then_residual_faint_and_replacement(self):
        # The reviewer's sequence (the pinned sim): Pikachu KOs our Magikarp, we bring Tyranitar in alone, the sandstorm KOs
        # Pikachu at the end of the turn and Foul Play replaces it after our wait request. That replacement is not a reply.
        window = ['|', '|switch|p1a: Tyranitar|Tyranitar, F|362/362', '|-weather|Sandstorm|[from] ability: Sand Stream|[of] p1a: Tyranitar', '|',
                  '|-weather|Sandstorm|[upkeep]', '|-damage|p2a: Pikachu|0 fnt|[from] Sandstorm', '|faint|p2a: Pikachu', '|upkeep',
                  request('p1', 'wait'), '|', '|switch|p2a: Snorlax|Snorlax, M|482/482']
        self.assertEqual(self.reply(window, forced=True).kind, 'wait')
        self.assertEqual(self.reply([x for x in window if not x.startswith('|request|')], forced=True).kind, 'wait')   # the faint ends it
        self.assertEqual(self.reply([x for x in window if not x.startswith('|faint|')], forced=True).kind, 'wait')   # the request ends it too
        self.assertEqual(self.reply([x for x in window if not x.startswith(('|faint|', '|request|'))], forced=True).kind, 'switch')

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
        self.assertEqual(S.reply_for(capture, dict(owner='p2', decision_index=0, request_count=1, request_kind='move')).detail, 'owner_mismatch')
        self.assertEqual(S.reply_for(capture, dict(owner='p1', decision_index=9, request_count=9, request_kind='move')).detail, 'request_missing')
        self.assertEqual(S.load_capture(str(self.root / 'absent.jsonl')).error, 'capture_missing')
        tampered = self.root / '1.jsonl';tampered.write_bytes(self.battle().read_bytes().replace(b'Body Slam', b'Body Slap'))
        self.assertEqual(S.load_capture(str(tampered)).error, 'capture_invalid')


class MappingTests(unittest.TestCase):
    """Our legal actions are 0, 1, 2, 4 and 6: the illegal rows of every grid are None, as the live pricer leaves them."""

    def setUp(self):self.legal = tuple(legal_mask(0, 1, 2, 4, 6))

    def map(self, reply, **kw):
        facts = decision([world_row(lambda a, b: (a - b) / 10, root_legal=self.legal, **kw)], root_legal=self.legal)['worlds'][0]
        return S.map_world(reply, facts, S.world_prices(dict(worlds=[facts]))[0], self.legal)

    def test_illegal_own_rows_are_not_unpriced(self):
        facts = decision([world_row(lambda a, b: .1, root_legal=self.legal)], root_legal=self.legal)['worlds'][0]
        self.assertIsNone(facts['grid'][3][0])
        self.assertEqual(self.map(S.Reply('move', 'bodyslam', None, '')), (0, None))
        self.assertEqual(self.map(S.Reply('wait', None, None, ''), requested=False), (0, None))

    def test_move(self):
        self.assertEqual(self.map(S.Reply('move', 'rest', None, '')), (2, None))
        self.assertEqual(self.map(S.Reply('move', 'bodyslam', None, ''), move_targets=(2, 0, 1, 3)), (1, None))
        self.assertEqual(self.map(S.Reply('move', 'hiddenpower', None, '')), (None, 'move_missing'))
        self.assertEqual(self.map(S.Reply('move', 'rest', None, ''), legal=legal_mask(0, 1, 3, 4)), (None, 'move_not_legal'))

    def test_struggle_maps_only_where_foul_play_must_struggle(self):
        forced = dict(usable=(False,) * 4, legal=legal_mask(0, 4, 5))
        self.assertEqual(self.map(S.Reply('move', 'struggle', None, ''), **forced), (0, None))
        # A world whose Foul Play still has a usable move: lane 0 is its first move slot, no Struggle exists there.
        self.assertEqual(self.map(S.Reply('move', 'struggle', None, '')), (None, 'struggle_not_forced'))
        self.assertEqual(self.map(S.Reply('move', 'struggle', None, ''), usable=(False, True, False, False)), (None, 'struggle_not_forced'))
        # A world whose Foul Play must Struggle: no other realized move maps (not even through a stale lane).
        self.assertEqual(self.map(S.Reply('move', 'bodyslam', None, ''), **forced), (None, 'struggle_world'))
        self.assertEqual(self.map(S.Reply('move', 'earthquake', None, ''), usable=(False,) * 4, legal=legal_mask(0, 1)), (None, 'struggle_world'))

    def test_charge_move_is_lane_0(self):
        self.assertEqual(self.map(S.Reply('move', 'solarbeam', None, ''), charge='solarbeam', legal=legal_mask(0)), (0, None))
        self.assertEqual(self.map(S.Reply('move', 'bodyslam', None, ''), charge='solarbeam', legal=legal_mask(0)), (None, 'move_not_legal'))
        self.assertEqual(self.map(S.Reply('move', 'solarbeam', None, ''), legal=legal_mask(0)), (None, 'move_missing'))

    def test_lane_labels_must_agree(self):
        labels = lane_labels(MOVES, (0, 1, 2, 3), (1, 2, 3, 4, 5), PARTY, [True] * 9, (True,) * 4, None)
        self.assertEqual(self.map(S.Reply('move', 'rest', None, ''), lanes=labels), (2, None))
        self.assertEqual(self.map(S.Reply('move', 'rest', None, ''), lanes=labels[:2] + ['move:curse'] + labels[3:]), (None, 'lane_label_mismatch'))
        self.assertEqual(self.map(S.Reply('switch', 'skarmory', None, ''), lanes=labels[:4] + ['switch:gengar'] + labels[5:]), (None, 'lane_label_mismatch'))

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

    def test_unkept_legal_lane_is_unpriced(self):
        # The re-pricer records kept != peer legal (kept_matches_peer_legal false): the legal lane 2 was never priced.
        self.assertEqual(self.map(S.Reply('move', 'rest', None, ''), kept=legal_mask(0, 1, 3, 4, 5, 6, 7, 8), legal=[True] * 9), (None, 'unpriced'))


class ScoringTests(unittest.TestCase):
    def score(self, worlds, reply, *, greedy=0, logits=None, legal=ALL, target=None):
        return S.score_decision(decision(worlds, greedy=greedy, root_logits=logits, root_legal=legal), reply, target, collections.Counter())

    def test_realized_and_pi_filled(self):
        legal = legal_mask(0, 1, 2);rng = random.Random(7)
        values = [[[round(rng.uniform(-1, 1), 3) for _ in range(9)] for _ in range(9)] for _ in range(4)]
        logits = [[rng.uniform(-2, 2) for _ in range(9)] for _ in range(4)]
        kept = [ALL, ALL, legal_mask(0, 1, 3), ALL]
        worlds = [world_row(lambda a, b, w=w: values[w][a][b], root_legal=legal, logits=logits[w], kept=kept[w],
                            **[{}, dict(moves=('surf', 'curse', 'rest', 'toxic')), {}, dict(move_targets=(1, 0, 2, 3))][w]) for w in range(4)]
        out = self.score(worlds, S.Reply('move', 'bodyslam', None, ''), legal=legal)
        # Body Slam: world 0 lane 0; world 1 lacks it; world 2 lane 0; world 3 holds slot 0 at lane 1.
        self.assertEqual(out['mapped_worlds'], [[0, 0], [2, 0], [3, 1]])
        self.assertEqual(out['world_drops'], {1: 'move_missing'})
        for a in range(9):
            if not legal[a]:self.assertIsNone(out['R']['dropped'][a]);continue
            self.assertAlmostEqual(out['R']['dropped'][a], (values[0][a][0] + values[2][a][0] + values[3][a][1]) / 3, places=12)
        self.assertEqual(out['R']['pi_filled'], out['R']['dropped'])
        # Censored: pi_filled R is averaging's q, and the hand computation sum_w 1/W sum_b pi_w(b) G_w[a][b] over kept replies.
        censored = self.score(worlds, S.censored('ko_first', ''), legal=legal)
        self.assertIsNone(censored['R']['dropped'])
        average = hedge.averaging(S.world_prices(decision(worlds, root_legal=legal)), tuple(legal))
        self.assertEqual(censored['R']['pi_filled'], list(average.q))
        for a in range(9):
            if not legal[a]:self.assertIsNone(censored['R']['pi_filled'][a]);continue
            hand = 0.
            for w in range(4):
                weights = [math.exp(logits[w][b]) if kept[w][b] else 0. for b in range(9)];total = sum(weights)
                hand += sum(weights[b] / total * values[w][a][b] for b in range(9) if kept[w][b]) / 4
            self.assertAlmostEqual(censored['R']['pi_filled'][a], hand, places=12)
        unmapped = self.score(worlds, S.Reply('move', 'splash', None, ''), legal=legal)
        self.assertTrue(unmapped['unmapped']);self.assertFalse(unmapped['observed'])
        self.assertEqual(unmapped['R']['pi_filled'], list(average.q))

    def test_d_against_greedy_and_illegal_none(self):
        legal = legal_mask(0, 1, 4)
        out = self.score(waiting_worlds([{0: .5, 1: .1, 4: -.2}] * 2, legal), S.Reply('wait', None, None, ''), greedy=1, legal=legal)
        ops = out['operators']
        self.assertEqual(ops['averaging']['choice'], 0)
        self.assertAlmostEqual(ops['averaging']['D']['dropped'], .4)
        self.assertEqual(ops['greedy']['D'], dict(dropped=0., pi_filled=0.))
        self.assertIsNone(out['R']['dropped'][2])

    def test_unscorable(self):
        legal = legal_mask(0, 1)
        with self.assertRaises(S.Unscorable) as error:S.recompute(decision(waiting_worlds([{0: .5}] * 2, legal), root_legal=legal, status='REFUSED'))
        self.assertEqual(error.exception.reason, 'replay_REFUSED')
        worlds = waiting_worlds([{0: .5}] * 2, legal);worlds[1]['grid'][1][0] = None   # a legal action against the waiting column
        with self.assertRaises(S.Unscorable) as error:self.score(worlds, S.Reply('wait', None, None, ''), legal=legal)
        self.assertEqual(error.exception.reason, 'unpriced_cell')
        with self.assertRaises(S.Unscorable) as error:self.score(waiting_worlds([{0: .5}] * 2, legal), S.Reply('wait', None, None, ''), legal=legal, greedy=3)
        self.assertEqual(error.exception.reason, 'greedy_missing')


class ParityTests(unittest.TestCase):
    def setUp(self):
        self.legal = legal_mask(0, 1, 4)
        self.worlds = [world_row(lambda a, b, w=w: round(.1 * a - .05 * b + .01 * w, 4), root_legal=self.legal, kept=legal_mask(0, 2, 5)) for w in range(4)]
        prices = S.world_prices(decision(self.worlds, root_legal=self.legal))
        self.average = hedge.averaging(prices, tuple(self.legal));self.action = hedge.top_action(self.average.strategy)
        self.q = json.loads(json.dumps(list(self.average.q)))

    def check(self, **kw):
        row = decision(self.worlds, root_legal=self.legal, **kw)
        try:base, reason = S.recompute(row), None
        except S.Unscorable as error:base, reason = None, error.reason
        return S.parity_row(row, base, reason)

    def test_match_and_mismatches(self):
        self.assertEqual(self.check(live_action=self.action, replay_q=self.q), dict(match=True, action_equal=True, q_equal=True, cause=None))
        self.assertEqual(self.check(live_action=1 - self.action, replay_q=self.q)['cause'], 'action_differs')
        off = list(self.q);off[4] = math.nextafter(off[4], 2.)   # one ulp: exact equality only
        self.assertEqual(self.check(live_action=self.action, replay_q=off)['cause'], 'q_differs')
        self.assertEqual(self.check(live_action=self.action, replay_q=None)['cause'], 'q_differs')
        self.assertEqual(self.check(live_action=self.action, replay_q=self.q[:8])['cause'], 'q_differs')
        self.assertEqual(self.check(live_action=None, replay_q=None)['cause'], 'action_and_q_differ')
        self.assertEqual(self.check(live_action=self.action, replay_q=self.q, status='REFUSED', replay_cause='depthn_refused')['cause'],
                         'replay_REFUSED:depthn_refused')
        self.worlds[0]['grid'] = None
        self.assertEqual(self.check(live_action=self.action, replay_q=self.q)['cause'], 'recompute_world_unpriced')

    def test_replay_mode_holds_the_replay_to_itself(self):
        row = decision(self.worlds, root_legal=self.legal, live_action=1 - self.action, replay_action=self.action, replay_q=self.q)
        base = S.recompute(row)
        self.assertEqual(S.parity_row(row, base, None)['cause'], 'action_differs')   # live mode: the live action differs
        self.assertEqual(S.parity_row(row, base, None, 'replay'), dict(match=True, action_equal=True, q_equal=True, cause=None))
        off = decision(self.worlds, root_legal=self.legal, live_action=self.action, replay_action=1 - self.action, replay_q=self.q)
        self.assertEqual(S.parity_row(off, S.recompute(off), None, 'replay')['cause'], 'action_differs')
        with self.assertRaises(ValueError):S.parity_row(row, base, None, 'other')

    def test_threshold_is_99_percent_inclusive(self):
        self.assertTrue(S.parity_passed(99, 100))
        self.assertFalse(S.parity_passed(98, 100))
        self.assertFalse(S.parity_passed(0, 0))


class OperatorTests(unittest.TestCase):
    def choices(self, worlds, *, greedy, logits=None, legal, target=None):
        out = S.score_decision(decision(worlds, greedy=greedy, root_logits=logits, root_legal=legal), S.censored('other', ''), target,
                               collections.Counter())
        return {name: op['choice'] for name, op in out['operators'].items()}, out

    def test_averaging_hedge_and_blend_differ(self):
        # Replies 0 (prior .9) and 1. Action 0: [1, -1] (best vs the predicted reply); 1: [.2, .2] (safe); 2: [.6, -.1].
        # Averaging (lam 1) plays 0, the hedge (lam 0) 1, the lam 0.5 blend 2.
        cells = {0: (1., -1.), 1: (.2, .2), 2: (.6, -.1)};legal = legal_mask(0, 1, 2)
        worlds = [world_row(lambda a, b: cells[a][b], root_legal=legal, kept=legal_mask(0, 1), logits=[math.log(.9), math.log(.1)] + [0.] * 7)] * 2
        c, _ = self.choices(worlds, greedy=1, legal=legal)
        self.assertEqual((c['averaging'], c['bayes_hedge:lam=0'], c['blend:lam=0.5']), (0, 1, 2))

    def test_prior_regularized(self):
        legal = legal_mask(0, 1)
        c, _ = self.choices(waiting_worlds([{0: .5, 1: .1}] * 2, legal), greedy=1, logits=[-3., 0.] + [0.] * 7, legal=legal)
        self.assertEqual((c['prior_regularized:c=0.02'], c['prior_regularized:c=0.5']), (0, 1))

    def test_noise_gated(self):
        legal = legal_mask(0, 1)
        # gap 0.1 +- 0.3 per world: se = 0.3 * sqrt(16/15) / 4 = 0.0775, so k=1 plays averaging and k=2 greedy.
        c, _ = self.choices(waiting_worlds([{0: .1 + (.3 if w % 2 else -.3), 1: 0.} for w in range(16)], legal), greedy=1, legal=legal)
        self.assertEqual((c['averaging'], c['noise_gated:k=1'], c['noise_gated:k=2']), (0, 0, 1))
        c, _ = self.choices(waiting_worlds([{0: .1 + .001 * w, 1: 0.} for w in range(16)], legal), greedy=1, legal=legal)
        self.assertEqual((c['noise_gated:k=1'], c['noise_gated:k=2']), (0, 0))
        # Strict: a zero gap with zero spread (averaging picks 0 by index) is not > 0: greedy.
        c, _ = self.choices(waiting_worlds([{0: .1, 1: .1}] * 16, legal), greedy=1, legal=legal)
        self.assertEqual((c['averaging'], c['noise_gated:k=1']), (0, 1))
        c, _ = self.choices(waiting_worlds([{0: .2, 1: .1}] * 16, legal), greedy=1, legal=legal)
        self.assertEqual(c['noise_gated:k=2'], 0)

    def test_world_lcb(self):
        legal = legal_mask(0, 1)
        c, _ = self.choices(waiting_worlds([{0: .3 + (.6 if w % 2 else -.6), 1: .2} for w in range(16)], legal), greedy=1, legal=legal)
        self.assertEqual((c['averaging'], c['world_lcb:kappa=0.5'], c['world_lcb:kappa=2']), (0, 0, 1))
        self.assertEqual(S.world_lcb({3: .5, 5: .5}, {3: .1, 5: .1}, 1), 3)   # ties: lowest index

    def test_world_lcb_kappa_zero_is_averaging(self):
        rng = random.Random(3)
        for _ in range(20):
            legal = legal_mask(*sorted(rng.sample(range(9), rng.randint(2, 9))))
            worlds = [world_row(lambda a, b: round(rng.uniform(-1, 1), 2), root_legal=legal, logits=[rng.uniform(-3, 3) for _ in range(9)]) for _ in range(16)]
            base = S.recompute(decision(worlds, root_legal=legal))
            qw = S.per_world_q(base.grids)
            spread = {a: statistics.stdev([x[a] for x in qw]) / 4 for a in qw[0]}
            self.assertEqual(S.world_lcb(base.average.q, spread, 0), hedge.top_action(base.average.strategy))
            for a in qw[0]:self.assertAlmostEqual(statistics.fmean(x[a] for x in qw), base.average.q[a], places=12)

    def test_target_argmax(self):
        legal = legal_mask(0, 1, 2);worlds = waiting_worlds([{0: .5, 1: .1, 2: .0}] * 2, legal)
        c, out = self.choices(worlds, greedy=1, legal=legal, target=[.1, .2, .7, .0, .0, .0, .0, .0, .9])   # lane 8 is illegal
        self.assertEqual(c['target_argmax'], 2);self.assertFalse(out['target_fallback'])
        # Without the live target: argmax of logits + q = (.5, .4, 0) plays 0; the logits alone would play 1.
        c, out = self.choices(worlds, greedy=1, logits=[0., .3, 0.] + [0.] * 6, legal=legal)
        self.assertEqual(c['target_argmax'], 0);self.assertTrue(out['target_fallback'])

    def test_bands(self):
        self.assertEqual([S.band(p) for p in (.0099, .01, .0499, .05, .1999, .2, .4999, .5, 1.)],
                         ['<1%', '1-5%', '1-5%', '5-20%', '5-20%', '20-50%', '20-50%', '>=50%', '>=50%'])


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

    def test_halves(self):
        self.assertEqual((S.half_of(10), S.half_of(11), S.half_of(0), S.half_of(None)), ('even', 'odd', 'even', None))

    def records(self):
        """Six battles (ordinals 0-2 even, 3-5 odd), two decisions each; 'bayes_hedge:lam=0' gains where it overrides; the
        second decision of each battle is censored (dropped excludes it) and a 'wait' decision."""
        out = []
        for ordinal in range(6):
            for k in range(2):
                ops = {name: (0, (0., 0.), '>=50%') for name in S.SETTINGS}
                ops['averaging'] = (1, (None if k else -.2 - .01 * ordinal, -.1 - .01 * ordinal), '5-20%')
                ops['bayes_hedge:lam=0'] = (2, (None if k else .1 + .01 * ordinal, .1 + .02 * ordinal), '1-5%')
                ops['world_lcb:kappa=1'] = (1, (None if k else .05, .05), '5-20%')   # plays averaging's action
                out.append(dict(ordinal=ordinal, half='even' if ordinal < 3 else 'odd', kind='move' if k else 'switch', observed=not k,
                                fp_requested=not k, ops=ops))
        return out

    def test_metrics_populations(self):
        table = S.metrics(self.records())
        self.assertEqual((table['pi_filled']['all']['decisions'], table['dropped']['all']['decisions']), (12, 6))
        self.assertEqual(table['pi_filled']['even']['battles'], 3)
        m = table['dropped']['odd']['operators']['bayes_hedge:lam=0']
        self.assertEqual((m['decisions'], m['override']), (3, 1.))
        self.assertAlmostEqual(m['mean'], .14)
        # Override is against greedy: averaging (action 1 vs greedy's 0) and a setting playing averaging's action override always.
        operators = table['pi_filled']['all']['operators']
        self.assertEqual((operators['averaging']['override'], operators['world_lcb:kappa=1']['override'], operators['greedy']['override'],
                          operators['noise_gated:k=1']['override']), (1., 1., 0., 0.))
        self.assertEqual(table['pi_filled']['all']['by_request_kind']['move']['averaging']['decisions'], 6)
        self.assertEqual(table['pi_filled']['all']['by_prior_band']['1-5%']['bayes_hedge:lam=0']['decisions'], 12)
        self.assertEqual(table['pi_filled']['all']['by_prior_band']['>=50%']['bayes_hedge:lam=0']['decisions'], 0)
        requested = S.metrics([r for r in self.records() if r['fp_requested']])
        self.assertEqual(requested['pi_filled']['all']['decisions'], 6)
        out = S.decide(table, True)
        self.assertTrue(out['gate']['passed'])
        self.assertEqual(out['selection']['chosen'], 'bayes_hedge:lam=0')
        self.assertEqual(out['verdict'], 'PASS')

    @staticmethod
    def table(averaging_ci, candidates, odd, averaging_even=None):
        """A metrics table: candidates {name: (even override, even mean)}; odd {name: (pi lower, dropped lower, override)}."""
        def m(override=0., mean=0., ci=(-1., 1.)):return dict(override=override, mean=mean, ci90=list(ci), battles=10, decisions=100)
        table = {s: {h: dict(operators={n: m() for n in S.SETTINGS}) for h in S.HALVES} for s in S.SCORINGS}
        table['pi_filled']['all']['operators']['averaging'] = m(mean=sum(averaging_ci) / 2, ci=averaging_ci)
        if averaging_even is not None:table['pi_filled']['even']['operators']['averaging'] = m(*averaging_even)
        for name, (override, mean) in candidates.items():table['pi_filled']['even']['operators'][name] = m(override, mean)
        for name, (pi, dropped, override) in odd.items():
            table['pi_filled']['odd']['operators'][name] = m(override, pi + .1, (pi, pi + .2))
            table['dropped']['odd']['operators'][name] = m(override, dropped + .1, (dropped, dropped + .2))
        return table

    def test_gate(self):
        out = S.decide(self.table((-.05, .001), {'blend:lam=0.5': (.2, .3)}, {}), True)
        self.assertEqual(out['verdict'], 'GATE_FAILED');self.assertIsNone(out['selection']['chosen'])
        self.assertEqual(S.decide(self.table((-.05, 0.), {'blend:lam=0.5': (.2, .3)}, {}), True)['verdict'], 'GATE_FAILED')   # strict < 0
        self.assertEqual(S.decide(self.table((-.05, -.001), {}, {}), False)['verdict'], 'PARITY_FAILED')

    def test_selection(self):
        candidates = {'blend:lam=0.5': (.04, .9), 'noise_gated:k=1': (.05, .02), 'world_lcb:kappa=1': (.3, .01), 'target_argmax': (.5, .02)}
        out = S.decide(self.table((-.05, -.01), candidates, {'noise_gated:k=1': (.01, .02, .06)}), True)
        self.assertEqual(out['selection']['chosen'], 'noise_gated:k=1')   # blend's override is below 5%; 5% exactly is eligible; ties keep order
        self.assertEqual(out['verdict'], 'PASS')
        self.assertEqual(S.decide(self.table((-.05, -.01), {}, {}), True)['verdict'], 'NO_CANDIDATE')
        # Averaging is never a candidate, whatever its even-seed mean.
        out = S.decide(self.table((-.05, -.01), {'blend:lam=0.5': (.2, .03)}, {}, averaging_even=(.5, .9)), True)
        self.assertEqual(out['selection']['chosen'], 'blend:lam=0.5')
        self.assertNotIn('averaging', [r['setting'] for r in out['selection']['candidates']])

    def test_confirmation(self):
        candidates = {'world_lcb:kappa=2': (.1, .05)}
        cases = (((.01, .02, .06), True), ((.01, .02, .05), True), ((.01, -.001, .06), False), ((.01, 0., .06), False), ((0., .02, .06), False),
                 ((.01, .02, .049), False))
        for odd, passed in cases:
            out = S.decide(self.table((-.05, -.01), candidates, {'world_lcb:kappa=2': odd}), True)
            self.assertEqual(out['confirmation']['passed'], passed, odd)
            self.assertEqual(out['verdict'], 'PASS' if passed else 'FAIL')


class EndToEndTests(unittest.TestCase):
    """Two battles through main(): an even seed with four scored decisions (one a forced switch Foul Play waits through, one
    played greedy after a live refusal) and an odd seed with two, plus a live-searched row whose replay was refused."""

    def setUp(self):self.directory = TemporaryDirectory();self.root = Path(self.directory.name);S.load_capture.cache_clear()

    def tearDown(self):self.directory.cleanup()

    def rows(self, run, rng):
        moves, switch = legal_mask(0, 1, 2, 4), legal_mask(4, 5)
        start = ['|start', '|switch|p1a: Starmie|Starmie|282/282', '|switch|p2a: Snorlax|Snorlax, F|100/100', '|turn|1']
        battles = {
            (0, 10): [('move', 'search', [start, [request('p1')], 'ROOT', ['|', '|move|p1a: Starmie|Surf|p2a: Snorlax',
                       '|move|p2a: Snorlax|Body Slam|p1a: Starmie', '|', '|upkeep', '|turn|2']]),
                      ('move', 'search', [[request('p1')], 'ROOT', ['|', '|move|p2a: Snorlax|Body Slam|p1a: Starmie', '|-damage|p1a: Starmie|0 fnt',
                       '|faint|p1a: Starmie']]),
                      ('switch', 'search', [[request('p1', 'forceSwitch')], 'ROOT', ['|', '|switch|p1a: Jolteon|Jolteon|292/292', '|', '|upkeep', '|turn|3']]),
                      ('move', 'greedy_fallback', [[request('p1')], 'ROOT', ['|', '|move|p1a: Jolteon|Thunderbolt|p2a: Snorlax', '|cant|p2a: Snorlax|par',
                       '|', '|upkeep', '|turn|4']])],
            (1, 11): [('move', 'search', [start, [request('p1')], 'ROOT', ['|', '|move|p1a: Starmie|Surf|p2a: Snorlax',
                       '|move|p2a: Snorlax|Rest|p2a: Snorlax', '|', '|upkeep', '|turn|2']]),
                      ('move', 'search', [[request('p1')], 'ROOT', ['|', '|move|p1a: Starmie|Surf|p2a: Snorlax', '|cant|p2a: Snorlax|slp', '|', '|upkeep',
                       '|turn|3']]),
                      ('move', 'refused', [[request('p1')], 'ROOT'])]}
        rows = []
        for (ordinal, seed), plan in battles.items():
            (run / 'results' / f'{ordinal}.json').write_text(json.dumps(dict(status='COMPLETE', cell=dict(seed=seed, arm='DN2W16', ordinal=ordinal))))
            write_capture(run / 'owning-wire' / f'{ordinal}.jsonl', 'p1', [e for _, _, events in plan for e in events], ordinal=ordinal)
            telemetry = []
            for index, (kind, source, _) in enumerate(plan):
                legal = switch if kind == 'switch' else moves
                if kind == 'switch':worlds = waiting_worlds([{a: round(rng.uniform(-1, 1), 3) for a in (4, 5)} for _ in range(16)], legal)
                else:worlds = [world_row(lambda a, b: round(rng.uniform(-1, 1), 3), root_legal=legal, logits=[rng.uniform(-2, 2) for _ in range(9)])
                               for _ in range(16)]
                base = S.recompute(decision(worlds, root_legal=legal))
                chosen, q = hedge.top_action(base.average.strategy), json.loads(json.dumps(list(base.average.q)))
                common = dict(ordinal=ordinal, decision_index=index, request_count=index + 1, greedy=4 if kind == 'switch' else 0, kind=kind,
                              root_logits=[1., 0., 0., 0., 0., 0., 0., 0., 0.], root_legal=legal)
                if source == 'refused':rows.append(raw_row(None, status='REFUSED', replay_cause='depthn_refused', live_action=chosen, **common))
                elif source == 'greedy_fallback':
                    rows.append(raw_row(worlds, source=source, live_cause='deadline', live_action=None, replay_action=chosen, replay_q=q, **common))
                else:rows.append(raw_row(worlds, live_action=chosen, replay_action=chosen, replay_q=q, **common))
                telemetry.append(dict(decision_index=index, search=dict(policy_target=[.1, .2, .3, 0., .4, 0., 0., 0., 0.])))
            (run / 'telemetry' / f'{ordinal}.jsonl').write_text(''.join(json.dumps(x) + '\n' for x in telemetry))
        return rows

    def test_main(self):
        run = self.root / 'run'
        for sub in ('results', 'telemetry', 'owning-wire'):(run / sub).mkdir(parents=True)
        path = self.root / 'rows-0.jsonl';path.write_text(''.join(json.dumps(r) + '\n' for r in self.rows(run, random.Random(11))))
        out = self.root / 'out'
        with redirect_stdout(io.StringIO()):
            self.assertEqual(S.main(['--rows', str(path), '--run-directory', str(run), '--out', str(out)]), 0)
        summary = json.loads((out / 'summary.json').read_text());cover = summary['coverage']
        self.assertEqual((cover['searched_decisions'], cover['replay_complete'], cover['scored']), (7, 6, 6))
        self.assertEqual(cover['excluded'], {'replay_REFUSED': 1})
        # Parity counts the refused replay and leaves out the row played greedy live: 5 of 6, below 99%.
        parity = cover['parity']
        self.assertEqual((parity['live_searched'], parity['matches'], parity['action_equal'], parity['q_equal']), (6, 5, 5, 5))
        self.assertEqual(parity['mismatches_by_cause'], {'replay_REFUSED:depthn_refused': 1})
        self.assertFalse(parity['passed']);self.assertEqual(summary['verdict'], 'PARITY_FAILED');self.assertIsNone(summary['selection']['chosen'])
        self.assertEqual(cover['status_mismatches']['by_cause'], {'replay_REFUSED:depthn_refused': 1, 'live_greedy_fallback:deadline': 1})
        self.assertEqual(cover['replies'], {'move': 3, 'wait': 1, 'censored': 2})
        self.assertEqual(cover['censored']['by_cause']['sleep_para_flinch']['count'], 2)
        self.assertEqual(cover['mapped_worlds']['histogram'], {'16': 4})
        self.assertEqual(cover['halves'], dict(even=dict(battles=1, decisions=4), odd=dict(battles=1, decisions=2), no_seed=0))
        self.assertEqual(cover['fp_requested']['decisions'], 5)
        self.assertEqual(summary['metrics_fp_requested']['pi_filled']['all']['decisions'], 5)
        self.assertEqual(summary['metrics']['pi_filled']['all']['decisions'], 6)
        decisions = [json.loads(x) for x in (out / 'decisions.jsonl').read_text().splitlines()]
        self.assertEqual(len(decisions), 7)
        self.assertEqual([d['half'] for d in decisions], ['even'] * 4 + ['odd'] * 3)
        first, forced = decisions[0], decisions[2]
        self.assertEqual(first['reply'], dict(kind='move', id='bodyslam', cause=None, detail='move'))
        self.assertEqual((len(first['mapped_worlds']), first['parity']['match']), (16, True))
        self.assertEqual(set(first['operators']), set(S.SETTINGS))
        self.assertEqual(first['operators']['target_argmax']['choice'], 4)
        self.assertEqual((first['target_source'], cover['target_sources']), ('live_telemetry', {'live_telemetry': 6}))   # the rows carry no replay target
        self.assertEqual((forced['reply']['kind'], len(forced['mapped_worlds'])), ('wait', 16))
        self.assertIsNone(decisions[3]['parity'])
        text = (out / 'summary.md').read_text()
        self.assertTrue(text.startswith('# RR'));self.assertIn('requested only: dropped, all', text)
        # The reading of D is stated before the verdict: a priced gain, and the gate as the RR's validation.
        self.assertEqual(text.splitlines()[2], S.SIGN_NOTE)
        self.assertIn('D > 0 = priced above greedy', S.SIGN_NOTE);self.assertIn('upper 90% bound < 0', S.SIGN_NOTE)
        # Without --reply-prior: no external operators anywhere; tempered is scored like the others.
        self.assertEqual(summary['settings'], list(S.SETTINGS));self.assertEqual(summary['reported_only'], [])
        self.assertIsNone(cover['external_prior']);self.assertNotIn('external_br', first['operators'])
        self.assertEqual(summary['metrics']['pi_filled']['all']['operators']['tempered:T=inf']['decisions'], 6)
        with self.assertRaises(FileExistsError):S.score_run([str(path)], str(run), str(out))

    def test_the_re_priced_rows_target_wins_over_the_live_telemetry(self):
        run = self.root / 'run'
        for sub in ('results', 'telemetry', 'owning-wire'):(run / sub).mkdir(parents=True)
        rows = self.rows(run, random.Random(11))
        for r in rows:   # the re-priced target on a lane neither the telemetry (4) nor the logits + q fallback picks: 2 for moves, 5 for switches
            legal = (r.get('root') or {}).get('root_legal')
            if legal:lane = 2 if legal[2] else 5;r['replay']['policy_target'] = [1. if a == lane else 0. for a in range(9)]
        rows[1]['live']['greedy_action_index'] = None   # excluded inside the scoring (greedy_missing): not counted as a target source
        path = self.root / 'rows-0.jsonl';path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
        out = self.root / 'out'
        with redirect_stdout(io.StringIO()):self.assertEqual(S.main(['--rows', str(path), '--run-directory', str(run), '--out', str(out)]), 0)
        cover = json.loads((out / 'summary.json').read_text())['coverage']
        self.assertEqual((cover['scored'], cover['target_sources'], cover['target_argmax_fallbacks']), (5, {'replay': 5}, 0))
        first = json.loads((out / 'decisions.jsonl').read_text().splitlines()[0])
        op = first['operators']['target_argmax'];self.assertEqual((first['target_source'], op.get('raw_choice', op['choice'])), ('replay', 2))

    def test_a_replay_mode_re_scoring_never_takes_the_live_trees_target(self):
        run = self.root / 'run'
        for sub in ('results', 'telemetry', 'owning-wire'):(run / sub).mkdir(parents=True)
        path = self.root / 'rows-0.jsonl';path.write_text(''.join(json.dumps(r) + '\n' for r in self.rows(run, random.Random(11))))   # no replay targets
        out = self.root / 'out'
        with redirect_stdout(io.StringIO()):
            self.assertEqual(S.main(['--rows', str(path), '--run-directory', str(run), '--out', str(out), '--parity-mode', 'replay']), 0)
        cover = json.loads((out / 'summary.json').read_text())['coverage']
        self.assertEqual((cover['target_sources'], cover['target_argmax_fallbacks']), ({'missing': 6}, 6))   # the telemetry is never read

    def test_row_policy_target_prefers_a_valid_replay_target(self):
        legal = [True, True, False] + [False] * 6
        class Run:
            def __init__(self, target):self.target = target
            def policy_target(self, ordinal, index):return self.target
        live = Run([.3, .7] + [0.] * 7)
        good = dict(replay_policy_target=[.9, .1] + [0.] * 7, root_legal=legal)
        self.assertEqual(S.row_policy_target(good, live, 0, 0), ([.9, .1] + [0.] * 7, 'replay'))
        nan_illegal = dict(replay_policy_target=[.9, .1, float('nan'), None] + [0.] * 5, root_legal=legal)   # only legal lanes count
        self.assertEqual(S.row_policy_target(nan_illegal, live, 0, 0)[1], 'replay')
        for bad in (None, [.9] * 8, [float('nan'), .1] + [0.] * 7, 'x', [True, False] + [0.] * 7):
            self.assertEqual(S.row_policy_target(dict(replay_policy_target=bad, root_legal=legal), live, 0, 0), ([.3, .7] + [0.] * 7, 'live_telemetry'))
        none = dict(replay_policy_target=None, root_legal=legal)
        self.assertEqual(S.row_policy_target(none, Run(None), 0, 0), (None, 'missing'))
        self.assertEqual(S.row_policy_target(none, Run([.3, .7]), 0, 0), (None, 'live_telemetry_invalid'))
        self.assertEqual(S.row_policy_target(none, live, 0, 0, live_fallback=False), (None, 'missing'))   # a modified tree: never the live target

    def test_real_telemetry_reader_preserves_malformed_target_provenance(self):
        telemetry = self.root / 'telemetry';telemetry.mkdir()
        malformed = ['invalid', {'lane': 1}, False, 42, [.5] * 8, [True] * 9]
        rows = [dict(decision_index=i, search=dict(policy_target=target)) for i, target in enumerate(malformed)]
        rows += [dict(decision_index=6, search=dict(policy_target=None)), dict(decision_index=7, search={})]
        (telemetry / '0.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
        run = S.RunDirectory(self.root)
        decision = dict(root_legal=[True] * 9)
        for i, target in enumerate(malformed):
            with self.subTest(target=target):
                self.assertEqual(run.policy_target(0, i), target)
                self.assertEqual(S.row_policy_target(decision, run, 0, i), (None, 'live_telemetry_invalid'))
        for i in (6, 7, 8):
            self.assertEqual(S.row_policy_target(decision, run, 0, i), (None, 'missing'))

    def test_main_counts_malformed_telemetry_targets_as_invalid(self):
        run = self.root / 'run'
        for sub in ('results', 'telemetry', 'owning-wire'):(run / sub).mkdir(parents=True)
        rows = self.rows(run, random.Random(11))
        for path in (run / 'telemetry').glob('*.jsonl'):
            telemetry = [json.loads(line) for line in path.read_text().splitlines()]
            for row in telemetry:row['search']['policy_target'] = 'invalid'
            path.write_text(''.join(json.dumps(row) + '\n' for row in telemetry))
        path = self.root / 'rows-0.jsonl';path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        out = self.root / 'out'
        with redirect_stdout(io.StringIO()):
            self.assertEqual(S.main(['--rows', str(path), '--run-directory', str(run), '--out', str(out)]), 0)
        cover = json.loads((out / 'summary.json').read_text())['coverage']
        self.assertEqual((cover['target_sources'], cover['target_argmax_fallbacks']), ({'live_telemetry_invalid': 6}, 6))

    def test_main_with_reply_prior(self):
        run = self.root / 'run'
        for sub in ('results', 'telemetry', 'owning-wire'):(run / sub).mkdir(parents=True)
        rows = self.rows(run, random.Random(11))
        path = self.root / 'rows-0.jsonl';path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
        prior = []
        for row in rows:
            if row['worlds'] is None or row['live']['request_kind'] == 'switch':continue   # the forced switch's worlds all wait: none needed
            if (row['ordinal'], row['decision_index']) == (1, 1):continue   # no logits at all: excluded
            for w, world in enumerate(row['worlds']):
                if (row['ordinal'], row['decision_index'], w) == (0, 1, 3):continue   # one requested world missing: excluded
                prior.append(dict(ordinal=row['ordinal'], decision_index=row['decision_index'], world=w, logits=world['root_peer_logits'][::-1]))
        prior_path = self.root / 'prior.jsonl';prior_path.write_text(''.join(json.dumps(r) + '\n' for r in prior))
        out = self.root / 'out'
        with redirect_stdout(io.StringIO()):
            S.main(['--rows', str(path), '--run-directory', str(run), '--out', str(out), '--reply-prior', str(prior_path)])
        summary = json.loads((out / 'summary.json').read_text())
        self.assertEqual(summary['settings'], list(S.SETTINGS + S.EXTERNAL_SETTINGS))
        self.assertEqual(summary['reported_only'], list(S.EXTERNAL_SETTINGS));self.assertEqual(summary['candidates'], list(S.CANDIDATES))
        external = summary['coverage']['external_prior']
        self.assertEqual((external['scored'], external['with_logits'], external['excluded'], external['missing_worlds']), (6, 4, 2, 17))
        for name in S.EXTERNAL_SETTINGS:self.assertEqual(summary['metrics']['pi_filled']['all']['operators'][name]['decisions'], 4)
        self.assertEqual(summary['metrics']['pi_filled']['all']['operators']['averaging']['decisions'], 6)
        decisions = [json.loads(x) for x in (out / 'decisions.jsonl').read_text().splitlines()]
        self.assertEqual(decisions[1]['external_missing_worlds'], [3]);self.assertIsNone(decisions[1]['operators']['external_br']['choice'])
        self.assertEqual(decisions[2]['external_missing_worlds'], []);self.assertIsNotNone(decisions[2]['operators']['external_br']['choice'])
        self.assertIn('external_mix:alpha=0.5', (out / 'summary.md').read_text())


def guard_record(*, b=False, turn1=(), turn2=(), lanes=(None, None, None, None), a=False, tier_b=False, masses=()):
    """A minimal effect-guard record (pokezero.depthn_effect_guard.evaluate's fields the scorer reads): (b) reasons per source as
    (move, tier), our guarded lanes as (tier, class) or None."""
    reasons = [dict(move=m, side='opp', source='turn1', tier=t, effect_class='x', max_mass=.3, worlds=[0]) for m, t in turn1]
    reasons += [dict(move=m, side='own', source='turn2', tier=t, effect_class='x', max_mass=.2, worlds=[1]) for m, t in turn2]
    return dict(b=b or bool(reasons), b_turn1=bool(turn1), b_turn2=bool(turn2), reasons=reasons, a=a, tier_b_legal=tier_b,
                own_guarded_lanes=[None if x is None else list(x) for x in lanes], turn2_available=True,
                masses=dict(turn1=[], turn2=list(masses)), conditions=dict(own=dict(members={}), opp=dict(members={})))


def reach_guard_record(*, turn1=(), reach_tiers=None, unavailable=False, any_cell=False):
    """A reach-rule record (the live rule) for GuardTests' decision (two waiting worlds, our legal lanes 0, 1, 4): reach_tiers
    {side: {tier: mass}} puts a guarded turn-2 entry (Destiny Bond for A, Spore for C) of that mass at every expanded cell, so
    its reach M is the sum of its tiers' masses whatever our prior; unavailable a lost capture; any_cell the any-cell arm's hit."""
    r = guard_record(turn1=turn1);moves = dict(A='destinybond', C='spore')
    tiers = {side: {**dict(A=0., C=0.), **(reach_tiers or {}).get(side, {})} for side in ('own', 'opp')}
    entries = [dict(move=moves[t], side=side, world=w, cell=a * 9, mass=v, guarded=True)
               for side, per in tiers.items() for t, v in per.items() if v for w in (0, 1) for a in (0, 1, 4)]
    m = {side: None if unavailable else sum(tiers[side].values()) for side in ('own', 'opp')}
    trips = unavailable or max(m.values()) >= S.effect_guard.THRESHOLD
    lost = [dict(move=None, side=None, source='turn2_unavailable', tier=None, effect_class=None, max_mass=None, worlds=[])] if unavailable else []
    hit = [dict(move='curse', side='opp', source='turn2', tier='A', effect_class='x', max_mass=.6, worlds=[0])] if any_cell else []
    r.update(rule=S.effect_guard.RULE, b=bool(turn1) or trips, b_turn2=trips, m_own=m['own'], m_opp=m['opp'],
             reach=dict(m_own=m['own'], m_opp=m['opp'], tripped=trips, unavailable=unavailable, tiers=None if unavailable else tiers),
             any_cell_turn2=bool(lost or hit), any_cell_b=bool(turn1 or lost or hit), any_cell_tripped=bool(turn1 or lost or hit),
             any_cell_reasons=lost + hit, reasons=[x for x in r['reasons'] if x['source'] == 'turn1'] + lost,
             turn2_available=not unavailable, masses=dict(turn1=[], turn2=None if unavailable else entries))
    return r


class GuardTests(unittest.TestCase):
    """Every operator but greedy plays greedy where the own-recharge guard or UNREPRESENTED_EFFECT trips on its own choice."""
    legal = legal_mask(0, 1, 4)

    def score(self, record=None, own_moves=None):
        d = decision(waiting_worlds([{0: .5, 1: .1, 4: -.2}] * 2, self.legal), greedy=1, root_legal=self.legal)   # averaging plays 0
        d.update(effect_guard=record, own_moves=own_moves)
        return S.score_decision(d, S.Reply('wait', None, None, ''), None, collections.Counter())

    def test_b_guards_every_operator(self):
        out = self.score(guard_record(turn1=[('destinybond', 'A')]))
        for name, o in out['operators'].items():
            self.assertEqual((o['choice'], o['D']['dropped']), (1, 0.), name)   # greedy plays
            if name != 'greedy' and o['raw_choice'] != 1:self.assertEqual(o['guard_causes'], ['UNREPRESENTED_EFFECT'], name)
        self.assertEqual(out['raw_operators']['averaging']['choice'], 0);self.assertAlmostEqual(out['raw_operators']['averaging']['D']['dropped'], .4)
        self.assertEqual((out['guard']['b'], out['guard']['tiers_turn1']), (True, ['A']))

    def test_c_guards_only_the_operators_choosing_the_lane(self):
        out = self.score(guard_record(lanes=(('A', 'other'), None, None, None)))   # lane 0 (averaging's choice) guarded for us
        self.assertEqual((out['operators']['averaging']['choice'], out['operators']['averaging']['guard_causes']), (1, ['UNREPRESENTED_EFFECT']))
        self.assertEqual(out['guard']['c']['averaging'], True)
        kept = [n for n, o in out['raw_operators'].items() if o['choice'] not in (0, None)]
        for name in kept:self.assertEqual(out['operators'][name]['choice'], out['raw_operators'][name]['choice'], name)   # not lane 0: unguarded
        clear = self.score(guard_record())
        self.assertEqual((clear['operators']['averaging']['choice'], clear['operators']['averaging']['guard_causes']), (0, None))

    def test_the_recharge_guard_and_a_row_without_the_record(self):
        out = self.score(None, own_moves=['hyperbeam', 'bodyslam', 'rest', 'protect'])
        self.assertEqual((out['operators']['averaging']['choice'], out['operators']['averaging']['guard_causes'], out['guard']),
            (1, ['OWN_RECHARGE_UNREPRESENTED'], None))
        both = self.score(guard_record(lanes=(('A', 'multi_turn'), None, None, None)), own_moves=['hyperbeam', 'bodyslam', 'rest', 'protect'])
        self.assertEqual(both['operators']['averaging']['guard_causes'], ['OWN_RECHARGE_UNREPRESENTED', 'UNREPRESENTED_EFFECT'])

    def test_guard_report(self):
        def record(ordinal, guard, d=.1, raw=.3):
            ops = {n: (0, (d, d), '<1%') for n in S.SETTINGS};ops['greedy'] = (1, (0., 0.), '<1%')
            raws = {n: (0, (raw, raw), '<1%') for n in S.SETTINGS};raws['greedy'] = (1, (0., 0.), '<1%')
            return dict(ordinal=ordinal, half='even', kind='move', observed=True, fp_requested=True, ops=ops, raw_ops=raws, guard=guard)
        facts = lambda r, c=False: S.guard_facts(r, {'averaging': 0 if c else 4, 'greedy': 1})
        records = [record(0, facts(guard_record(turn1=[('curse', 'A')])), d=0.), record(1, facts(guard_record(turn2=[('spore', 'C')])), d=0.),
                   record(2, facts(guard_record(a=True, tier_b=True))), record(3, facts(guard_record(lanes=(('A', 'x'), None, None, None)), c=True)),
                   record(4, None), record(5, facts(S.effect_guard.error_record('unreadable')), d=0.)]
        g = S.guard_report(records, S.SETTINGS, opponent_recharge=2)
        self.assertEqual((g['priced_roots'], g['without_record'], g['opponent_recharge_refused']), (5, 1, 2))
        # the error record (b, neither turn-1 nor turn-2 reasons) is its own part of the guarded share
        self.assertEqual((g['union']['turn1'], g['union']['turn2_added'], g['union']['unavailable_or_error'], g['union']['total'], g['guarded_share']),
            (1, 1, 1, 3, .6))
        self.assertEqual((g['tiers']['A']['turn1'], g['tiers']['C']['turn2_added']), (1, 1));self.assertTrue(g['above_20pct'])
        self.assertEqual((g['c']['averaging']['c_only'], g['c']['averaging']['union_share']), (1, .8))
        self.assertEqual((g['primary']['roots'], g['guarded_context']['roots'], g['tier_b']['legal']['roots'], g['sensitivity']['roots']), (2, 3, 1, 1))
        alone = S.guard_report([record(0, facts(guard_record())), record(1, facts(S.effect_guard.error_record('x')))], S.SETTINGS)
        self.assertEqual((alone['guarded_share'], alone['above_20pct'], alone['union']['unavailable_or_error']), (.5, True, 1))   # errors alone flag
        self.assertAlmostEqual(g['guarded_context']['table']['pi_filled']['averaging']['mean'], .3)   # raw choices for context
        self.assertAlmostEqual(g['primary']['table']['pi_filled']['averaging']['mean'], .1)
        text = '\n'.join(S.guard_markdown(g, S.SETTINGS))
        self.assertIn('**above 20%**', text);self.assertIn('Primary: RR on unguarded roots (2)', text)

    def test_end_to_end(self):
        """A small synthetic replay through main(): the rows carry their guard records; the guarded root plays greedy everywhere."""
        e2e = EndToEndTests('test_main');e2e.setUp();self.addCleanup(e2e.tearDown)
        run = e2e.root / 'run'
        for sub in ('results', 'telemetry', 'owning-wire'):(run / sub).mkdir(parents=True)
        rows = e2e.rows(run, random.Random(11))
        for i, row in enumerate(rows):
            row['effect_guard'] = guard_record(turn1=[('knockoff', 'A')]) if i == 0 else guard_record(a=i == 1)
            if row['root'] is not None:row['root']['own_moves'] = ['surf', 'icebeam', 'recover', 'thunderbolt']
        rows[-1]['root'] = dict(decided=False, root_logits=None, root_legal=None, refusal_cause='OPPONENT_RECHARGE_UNREPRESENTED')
        path = e2e.root / 'rows-0.jsonl';path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
        with redirect_stdout(io.StringIO()):self.assertEqual(S.main(['--rows', str(path), '--run-directory', str(run), '--out', str(e2e.root / 'out')]), 0)
        summary = json.loads((e2e.root / 'out' / 'summary.json').read_text());g = summary['guard']
        self.assertEqual((g['priced_roots'], g['union']['total'], g['opponent_recharge_refused'], g['primary']['roots']), (6, 1, 1, 5))
        decisions = [json.loads(x) for x in (e2e.root / 'out' / 'decisions.jsonl').read_text().splitlines()];first = decisions[0]
        self.assertTrue(all(o['choice'] == first['greedy_action'] for o in first['operators'].values()))
        self.assertIn('# Effect guard', (e2e.root / 'out' / 'summary.md').read_text())
        # the reach fields on every scored decision (the measurement session keys on these names)
        self.assertEqual(set(first['reach_turn2']), {'m_own_lower', 'm_own_upper', 'm_opp_lower', 'm_opp_upper', 'tripped', 'exact'})
        self.assertEqual((first['reach_turn2']['tripped'], first['reach_scope_tripped_lower'], first['reach_scope_tripped_upper']), (False, True, True))   # turn 1
        self.assertEqual((decisions[1]['reach_scope_tripped_lower'], decisions[1]['reach_scope_tripped_upper']), (False, False))
        r = g['reach'];self.assertEqual((r['rows'], r['ambiguous'], r['lower']['roots'], r['upper']['turn1']), (6, 0, 1, 1))
        self.assertEqual(r['lower']['cross_tab'], {'any_cell/reach': 1, 'any_cell/reach_clear': 0, 'any_cell_clear/reach': 0, 'any_cell_clear/reach_clear': 5})
        self.assertIn('# Guard reach', (e2e.root / 'out' / 'summary.md').read_text())

    def test_the_primary_follows_the_records_live_rule(self):
        """A reach record (rule 'reach'): its b (turn-1 (b) or the reach M >= 5%) guards the primary; its any-cell arm is reported
        only. Here the any-cell arm trips (a 60% guarded prior at one cell) while the reach does not: the operators play their own
        choices, the any-cell scope's play greedy, and the cross-tab says so."""
        def live_record(*, reach_trips):
            r = guard_record(turn2=[('destinybond', 'A')])
            r.update(rule=S.effect_guard.RULE, b=reach_trips, b_turn1=False, b_turn2=reach_trips, any_cell_turn2=True, any_cell_b=True,
                     any_cell_tripped=True, m_own=0., m_opp=.08 if reach_trips else .01, arms=['reach'] if reach_trips else [],
                     reach=dict(m_own=0., m_opp=.08 if reach_trips else .01, tripped=reach_trips, unavailable=False,
                                tiers=dict(own=dict(A=0., C=0.), opp=dict(A=.08 if reach_trips else .01, C=0.))))
            return r
        quiet = self.score(live_record(reach_trips=False))
        self.assertEqual((quiet['operators']['averaging']['choice'], quiet['operators']['averaging']['guard_causes']), (0, None))   # unguarded
        self.assertEqual(quiet['reach_operators']['any_cell']['averaging']['choice'], 1)   # the any-cell scope would play greedy
        self.assertEqual((quiet['guard']['rule'], quiet['guard']['b'], quiet['guard']['any_cell']['b'], quiet['guard']['m_opp']), ('reach', False, True, .01))
        loud = self.score(live_record(reach_trips=True))
        self.assertEqual((loud['operators']['averaging']['choice'], loud['guard']['tiers_turn2'], loud['guard']['turn2']), (1, ['A'], True))
        def record(guard, ops):
            return dict(ordinal=0, half='even', kind='move', observed=True, fp_requested=True, guard=guard, raw_ops=ops, reach_ops=dict(any_cell=ops, lower=ops, upper=ops),
                        ops=ops)
        ops = {n: (0, (.1, .1), '<1%') for n in S.SETTINGS};ops['greedy'] = (1, (0., 0.), '<1%')
        g = S.guard_report([record(quiet['guard'], ops), record(loud['guard'], ops)], S.SETTINGS)
        self.assertEqual((g['rules'], g['union']['total'], g['union']['turn2_added']), ({'reach': 2}, 1, 1))
        self.assertEqual((g['any_cell']['roots'], g['any_cell']['turn2_added'], g['any_cell']['cross_tab']),
            (2, 2, {'any_cell/live': 1, 'any_cell/live_clear': 1, 'any_cell_clear/live': 0, 'any_cell_clear/live_clear': 0}))
        self.assertIn('# Any-cell scope', '\n'.join(S.any_cell_markdown(g['any_cell'], S.SETTINGS)))

    def test_the_report_only_scopes_on_old_and_reach_rows(self):
        """Old rows: the any-cell scope is the live one (a diagonal cross-tab). Reach rows: the union splits a reach trip (turn 2)
        from an unavailable capture; a trip lists every tier whose mass contributes (A + C); the reach section's any-cell column
        reads the record's any-cell fields, not its live (b)."""
        def record(guard):
            ops = {n: (0, (.1, .1), '<1%') for n in S.SETTINGS};ops['greedy'] = (1, (0., 0.), '<1%')
            return dict(ordinal=0, half='even', kind='move', observed=True, fp_requested=True, guard=guard, raw_ops=ops, ops=ops,
                        reach_ops=dict(any_cell=ops, lower=ops, upper=ops))
        old = [self.score(guard_record(turn1=[('curse', 'A')])), self.score(guard_record())]
        g = S.guard_report([record(x['guard']) for x in old], S.SETTINGS)
        self.assertEqual((g['rules'], g['any_cell']['rows'], g['any_cell']['roots'], g['any_cell']['unguarded']['roots']), ({'any_cell': 2}, 2, 1, 1))
        self.assertEqual(g['any_cell']['cross_tab'], {'any_cell/live': 1, 'any_cell/live_clear': 0, 'any_cell_clear/live': 0, 'any_cell_clear/live_clear': 1})
        both = self.score(reach_guard_record(reach_tiers=dict(own=dict(A=.03, C=.03))))   # neither tier alone crosses 5%
        self.assertEqual((both['guard']['turn2'], both['guard']['tiers_turn2']), (True, ['A', 'C']))
        self.assertEqual(both['guard']['tier_masses_turn2'], dict(own=dict(A=.03, C=.03)))
        lost = self.score(reach_guard_record(unavailable=True))
        clear = self.score(reach_guard_record(any_cell=True))   # only the any-cell arm would trip
        self.assertEqual((lost['guard']['turn2'], lost['guard']['unavailable'], clear['guard']['b'], clear['guard']['any_cell']['b']), (False, True, False, True))
        g = S.guard_report([record(x['guard']) for x in (both, lost, clear)], S.SETTINGS)
        u = g['union'];self.assertEqual((u['turn1'], u['turn2_added'], u['unavailable_or_error'], u['total']), (0, 1, 1, 2))
        self.assertEqual((g['tiers']['A']['turn2_added'], g['tiers']['C']['turn2_added']), (1, 1))
        # both: the reach trips, the any-cell arm does not; lost: both trip, fail closed; clear: the record's any-cell arm, not its (b)
        self.assertEqual(g['reach']['lower']['cross_tab'], {'any_cell/reach': 1, 'any_cell/reach_clear': 1, 'any_cell_clear/reach': 1, 'any_cell_clear/reach_clear': 0})
        self.assertEqual((g['reach']['lower']['tiers']['A']['roots'], g['reach']['lower']['tiers']['C']['roots']), (2, 2))   # both and lost

    def test_reach_scopes_per_row(self):
        """The reach on a scored row: a turn-2 mass whose bounds straddle 5% is ambiguous (lower clear, upper guarded); the operators
        follow each scope's own guard; the report counts it and its cross-tab against the any-cell scope."""
        legal = self.legal;d = decision(waiting_worlds([{0: .5, 1: .1, 4: -.2}] * 2, legal), greedy=1, root_legal=legal)   # 2 waiting worlds
        # Spore (tier C, no guarded flag) at our lane 0 against the waiting reply 0 in both worlds: M_own upper = pi_own(0) * .5
        spore = [dict(side='own', world=w, move='spore', mass=.5, cell=0 * 9 + 0) for w in (0, 1)]
        d.update(effect_guard=guard_record(turn2=[('spore', 'C')], masses=spore), own_moves=None)
        out = S.score_decision(d, S.Reply('wait', None, None, ''), None, collections.Counter())
        reach = out['guard']['reach'];prior = S.effect_guard.masses(d['root_logits'], d['root_legal'])
        self.assertAlmostEqual(reach['m_own_upper'], prior[0] * .5, places=15);self.assertEqual(reach['m_own_lower'], 0.)
        self.assertEqual((reach['tripped'], out['guard']['scopes']), (None, dict(lower=False, upper=True)))
        self.assertEqual(out['reach_operators']['lower']['averaging']['choice'], 0)   # the lower scope's averaging plays its own choice
        self.assertEqual(out['reach_operators']['upper']['averaging']['choice'], 1)   # the upper scope guards it: greedy
        self.assertEqual(out['operators']['averaging']['choice'], 1)   # and the any-cell scope (b from its turn-2 reason) too
        fields = S.reach_fields(out['guard'])
        self.assertEqual((fields['reach_turn2']['tripped'], fields['reach_scope_tripped_lower'], fields['reach_scope_tripped_upper']), (None, False, True))


OWN_MOVES = ['surf', 'icebeam', 'recover', 'thunderbolt']


def definitive_decision(worlds, *, record='clear', own_moves=OWN_MOVES, **kw):
    """A definitive row: the guard record (guard_record(), unless given; None drops it) and our own moves."""
    d = decision(worlds, **kw);d.update(effect_guard=guard_record() if record == 'clear' else record, own_moves=own_moves)
    return d


class DefinitiveOperatorTests(unittest.TestCase):
    """The definitive set on test_averaging_hedge_and_blend_differ's grid: replies 0 and 1; action 0 [1, -1], 1 [.2, .2], 2 [.6, -.1];
    the champion's prior (.9, .1): averaging 0, hedge 1, blend 0.5 2."""
    cells = {0: (1., -1.), 1: (.2, .2), 2: (.6, -.1)}
    champion = [math.log(.9), math.log(.1)] + [0.] * 7
    other_reply = [math.log(.1), math.log(.9)] + [0.] * 7

    def worlds(self, count=2, cells=None):
        legal = legal_mask(0, 1, 2);cells = cells or [self.cells] * count
        return [world_row(lambda a, b, c=c: c[a][b], root_legal=legal, kept=legal_mask(0, 1), logits=self.champion) for c in cells]

    def score(self, worlds, *, model, logliks=None, record='clear', own_moves=OWN_MOVES, greedy=1):
        legal = legal_mask(0, 1, 2);d = definitive_decision(worlds, record=record, own_moves=own_moves, greedy=greedy, root_legal=legal)
        prior = {w: list(model) for w in range(len(worlds))}
        weights = {w: x for w, x in enumerate(logliks or [0.] * len(worlds))}
        out = S.score_decision(d, S.censored('other', ''), None, collections.Counter(), definitive=dict(prior=prior, weights=weights))
        return {name: op['choice'] for name, op in out['operators'].items()}, out

    def test_the_champion_as_reply_model_reproduces_averaging_and_blend(self):
        c, out = self.score(self.worlds(), model=self.champion)
        self.assertEqual(set(c), set(S.DEFINITIVE_SETTINGS + S.LOW_BETA_REWEIGHTED))
        self.assertEqual((c['averaging'], c['br_fp'], c['bayes_hedge:lam=0']), (0, 0, 1))
        self.assertEqual((c['blend:lam=0.5'], c['rnr:lam=0.5']), (2, 2))
        self.assertEqual((c['reweighted_averaging:beta=0.5'], c['reweighted_br_fp:beta=0.5']), (0, 0))
        self.assertEqual(out['world_weights'], [.5, .5]);self.assertEqual((out['effective_worlds'], out['max_world_weight']), (2., .5))

    def test_a_reply_model_that_predicts_the_other_reply_moves_br_and_rnr(self):
        # The model predicts reply 1 at .9: Q = (-.8, .2, -.03), so br_fp plays 1; at lam .75 rnr stays on the safe action too.
        c, _ = self.score(self.worlds(), model=self.other_reply)
        self.assertEqual((c['averaging'], c['br_fp'], c['rnr:lam=0.75'], c['reweighted_br_fp:beta=0.5']), (0, 1, 1, 1))

    def test_reweighting_follows_the_likely_world(self):
        # World 0 is the grid above; in world 1 action 2 dominates. Equal logliks: averaging's choice; loglik 10 on world 1
        # (beta .5: weight e^5 / (1 + e^5)) plays world 1's best.
        other = {0: (-.5, -.5), 1: (0., 0.), 2: (.9, .9)}
        c, out = self.score(self.worlds(cells=[self.cells, other]), model=self.champion)
        self.assertEqual(c['reweighted_averaging:beta=0.5'], c['averaging'])
        c, out = self.score(self.worlds(cells=[self.cells, other]), model=self.champion, logliks=[0., 10.])
        self.assertEqual((c['reweighted_averaging:beta=0.5'], c['reweighted_br_fp:beta=0.5']), (2, 2))
        self.assertAlmostEqual(out['world_weights'][1], math.exp(5) / (1 + math.exp(5)), places=12)
        self.assertAlmostEqual(out['effective_worlds'], 1 / math.fsum(w * w for w in out['world_weights']), places=12)
        self.assertEqual(out['max_world_weight'], max(out['world_weights']))
        # beta 0.25 halves the evidence: weight e^2.5 / (1 + e^2.5) on world 1, still enough to move both operators here
        self.assertEqual((c['reweighted_averaging:beta=0.25'], c['reweighted_br_fp:beta=0.25']), (2, 2))
        self.assertEqual(S.world_weights([-1e6, 0.]), [0., 1.])

    def test_the_guards_apply_to_the_definitive_operators(self):
        free, _ = self.score(self.worlds(), model=self.other_reply, greedy=0)
        self.assertEqual((free['br_fp'], free['bayes_hedge:lam=0'], free['blend:lam=0.5']), (1, 1, 2))
        # (b) at the root: every operator plays greedy; the raw choices are kept for the context table
        c, out = self.score(self.worlds(), model=self.other_reply, greedy=0, record=guard_record(turn1=[('destinybond', 'A')]))
        self.assertEqual(set(c.values()), {0});self.assertEqual({n: o['choice'] for n, o in out['raw_operators'].items()}, free)
        self.assertEqual(out['operators']['br_fp']['guard_causes'], ['UNREPRESENTED_EFFECT']);self.assertTrue(out['guard']['b'])
        for name, op in out['operators'].items():self.assertEqual(op['D']['pi_filled'], 0., name)
        # (c) on our lane 1: the operators choosing it play greedy; the others keep their choice
        c, out = self.score(self.worlds(), model=self.other_reply, greedy=0, record=guard_record(lanes=(None, ('A', 'x'), None, None)))
        self.assertEqual((c['br_fp'], c['bayes_hedge:lam=0'], c['blend:lam=0.5']), (0, 0, 2))
        self.assertEqual((out['guard']['b'], out['guard']['c']['br_fp'], out['guard']['c']['blend:lam=0.5']), (False, True, False))
        # our lane 1 is a recharge move: an override into it plays greedy; greedy itself may be one
        moves = ['surf', 'hyperbeam', 'recover', 'thunderbolt']
        c, out = self.score(self.worlds(), model=self.other_reply, greedy=0, own_moves=moves)
        self.assertEqual((c['br_fp'], c['blend:lam=0.5']), (0, 2));self.assertEqual(out['operators']['br_fp']['guard_causes'], ['OWN_RECHARGE_UNREPRESENTED'])
        c, out = self.score(self.worlds(), model=self.other_reply, greedy=1, own_moves=moves)
        self.assertEqual((c['greedy'], c['br_fp'], out['operators']['br_fp']['guard_causes']), (1, 1, None))

    def test_each_world_gets_its_own_reply_logits_and_br_fp_is_averaging_exactly(self):
        # world 0 rewards action 0 against reply 0, world 1 rewards action 1 against reply 0; reply priors per world decide br_fp
        legal = legal_mask(0, 1, 2);cells = [{0: (1., -1.), 1: (0., 0.), 2: (-1., -1.)}, {0: (0., 0.), 1: (1., -1.), 2: (-1., -1.)}]
        worlds = self.worlds(cells=cells);d = definitive_decision(worlds, greedy=2, root_legal=legal)
        to_reply0, to_reply1 = [math.log(.9), math.log(.1)] + [0.] * 7, [math.log(.1), math.log(.9)] + [0.] * 7
        def br(prior):
            out = S.score_decision(d, S.censored('other', ''), None, collections.Counter(), definitive=dict(prior=prior, weights={0: 0., 1: 0.}))
            return out['operators']['br_fp']['choice']
        aligned, swapped = br({0: to_reply0, 1: to_reply1}), br({0: to_reply1, 1: to_reply0})
        self.assertEqual((aligned, swapped), (0, 1))   # a world-index shuffle would swap them
        base = S.recompute(d)
        want = hedge.averaging(S.model_worlds(base.worlds, {0: tuple(to_reply0), 1: tuple(to_reply1)}), base.legal)
        self.assertEqual(aligned, hedge.top_action(want.strategy))
        three = self.worlds(cells=cells + [cells[0]]);d3 = definitive_decision(three, greedy=2, root_legal=legal)   # 3 worlds: not a power of 2
        prior = {0: to_reply0, 1: to_reply1, 2: to_reply1}
        got = S.score_decision(d3, S.censored('other', ''), None, collections.Counter(), definitive=dict(prior=prior, weights={0: 0., 1: 0., 2: 0.}))
        b3 = S.recompute(d3);want = hedge.averaging(S.model_worlds(b3.worlds, {k: tuple(v) for k, v in prior.items()}), b3.legal)
        self.assertEqual(got['operators']['br_fp']['choice'], hedge.top_action(want.strategy))

    def test_inputs_from_other_rows_are_refused(self):
        legal = legal_mask(0, 1, 2);d = definitive_decision(self.worlds(), greedy=1, root_legal=legal)
        for reason, prior, weights in (('world_weights_mismatch', {0: self.champion, 1: self.champion}, {0: 0., 1: 0., 2: 0.}),
                                       ('reply_prior_mismatch', {0: self.champion, 1: self.champion, 5: self.champion}, {0: 0., 1: 0.})):
            with self.assertRaises(S.Unscorable) as caught:
                S.score_decision(d, S.censored('other', ''), None, collections.Counter(), definitive=dict(prior=prior, weights=weights))
            self.assertEqual(caught.exception.reason, reason)

    def test_missing_inputs_leave_the_decision_out_of_every_operator(self):
        legal = legal_mask(0, 1, 2);worlds = self.worlds()
        cases = (('effect_guard_missing', dict(record=None), {0: self.champion, 1: self.champion}, {0: 0., 1: 0.}),
                 ('reply_prior_missing', {}, {0: self.champion}, {0: 0., 1: 0.}),
                 ('world_weights_missing', {}, {0: self.champion, 1: self.champion}, {0: 0.}),
                 ('world_weights_missing', {}, {0: self.champion, 1: self.champion}, None))
        for reason, kw, prior, weights in cases:
            d = definitive_decision(worlds, greedy=1, root_legal=legal, **kw)
            with self.assertRaises(S.Unscorable) as caught:
                S.score_decision(d, S.censored('other', ''), None, collections.Counter(), definitive=dict(prior=prior, weights=weights))
            self.assertEqual(caught.exception.reason, reason)
        # Rows carrying reply_peer_logits need no --reply-prior; a waiting world needs none at all.
        rows = [dict(w, reply_peer_logits=self.champion) for w in worlds] + [world_row(lambda a, b: .1, requested=False, root_legal=legal)]
        d = definitive_decision(rows, greedy=1, root_legal=legal)
        out = S.score_decision(d, S.censored('other', ''), None, collections.Counter(), definitive=dict(prior=None, weights={0: 0., 1: 0., 2: 0.}))
        self.assertEqual(out['operators']['br_fp']['choice'], out['operators']['averaging']['choice'])

    def test_load_world_weights(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / 'w.jsonl'
            path.write_text(json.dumps(dict(ordinal=1, decision_index=2, world=0, loglik=-1.5)) + '\n\n' + json.dumps(dict(ordinal=1, decision_index=2, world=1, loglik=0)) + '\n')
            self.assertEqual(S.load_world_weights(path), {(1, 2): {0: -1.5, 1: 0.}})
            for bad in (dict(ordinal=1, decision_index=2, world=0, loglik=float('nan')), dict(ordinal=1, decision_index=2, world='0', loglik=0.)):
                path.write_text(json.dumps(bad) + '\n')
                with self.assertRaises(ValueError):S.load_world_weights(path)
            path.write_text(json.dumps(dict(ordinal=1, decision_index=2, world=0, loglik=0.)) + '\n' + json.dumps(dict(ordinal=1, decision_index=2, world=0, loglik=1.)) + '\n')
            with self.assertRaises(ValueError):S.load_world_weights(path)


class DefinitiveEndToEndTests(EndToEndTests):
    """EndToEndTests' two battles under --operator-set definitive: every row carries its guard record; decision (1, 1) has no
    world weights."""

    def test_main(self):pass

    def test_main_with_reply_prior(self):pass

    def definitive_inputs(self, slope, guarded=(), tier_b=(), a_only=()):
        run = self.root / 'run'
        for sub in ('results', 'telemetry', 'owning-wire'):(run / sub).mkdir(parents=True, exist_ok=True)
        rows = self.rows(run, random.Random(11));weights, prior = [], []
        for row in rows:
            key = (row['ordinal'], row['decision_index'])
            if row['worlds'] is None:   # a refused root: no guard record by construction (the live actor refused it)
                row['root'] = dict(decided=False, root_logits=None, root_legal=None, refusal_cause='OPPONENT_RECHARGE_UNREPRESENTED');continue
            row['effect_guard'] = guard_record(turn1=[('knockoff', 'A')] if key in guarded else (), a=key in a_only, tier_b=key in tier_b)
            row['root']['own_moves'] = OWN_MOVES
            for w, world in enumerate(row['worlds']):
                if key != (1, 1):weights.append(dict(ordinal=row['ordinal'], decision_index=row['decision_index'], world=w, loglik=-w * slope))
                if row['ordinal'] == 0:world['reply_peer_logits'] = world['root_peer_logits'][::-1]
                else:prior.append(dict(ordinal=row['ordinal'], decision_index=row['decision_index'], world=w, logits=world['root_peer_logits'][::-1]))
        path = self.root / 'rows-0.jsonl';path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
        wpath = self.root / 'weights.jsonl';wpath.write_text(''.join(json.dumps(r) + '\n' for r in weights))
        ppath = self.root / 'prior.jsonl';ppath.write_text(''.join(json.dumps(r) + '\n' for r in prior))
        return run, path, wpath, ppath

    def run_definitive(self, run, path, wpath, ppath, *extra, out='out'):
        with redirect_stdout(io.StringIO()):
            S.main(['--rows', str(path), '--run-directory', str(run), '--out', str(self.root / out), '--operator-set', 'definitive',
                    '--world-weights', str(wpath), '--reply-prior', str(ppath), *extra])
        return json.loads((self.root / out / 'summary.json').read_text())

    def test_main_definitive(self):
        run, path, wpath, ppath = self.definitive_inputs(.25)
        summary = self.run_definitive(run, path, wpath, ppath);cover = summary['coverage'];out = self.root / 'out'
        self.assertEqual((summary['schema'], summary['operator_set'], summary['scope']), (S.SCHEMA, 'definitive', 'unguarded roots (effect_guard.b false)'))
        self.assertEqual(summary['settings'], list(S.DEFINITIVE_SETTINGS));self.assertEqual(summary['candidates'], list(S.DEFINITIVE_CANDIDATES))
        self.assertNotIn('averaging', summary['reported_only']);self.assertIn('tempered:T=2', summary['reported_only'])
        self.assertIn('rnr:lam=0.25', summary['reported_only']);self.assertNotIn('rnr:lam=0.5', summary['reported_only'])
        self.assertEqual((cover['scored'], cover['excluded']), (5, {'replay_REFUSED': 1, 'world_weights_missing': 1}))
        # the refused root without a record keeps the screen set's accounting: excluded by its replay status, not effect_guard_missing
        self.assertEqual(summary['guard']['opponent_recharge_refused'], 1)
        with redirect_stdout(io.StringIO()):S.main(['--rows', str(path), '--run-directory', str(run), '--out', str(self.root / 'screen')])
        screen = json.loads((self.root / 'screen' / 'summary.json').read_text())['coverage']
        self.assertEqual((screen['scored'], screen['excluded']), (6, {'replay_REFUSED': 1}))
        w = cover['world_weights']   # loglik -w/4 at beta .5: p_w ∝ r^w, r = e^(-1/8), ESS = (sum r^w)^2 / sum r^2w ~ 12.2 on every root
        self.assertEqual(w['roots'], 5);self.assertAlmostEqual(w['median_ess'], 12.2, places=1);self.assertEqual(w['share_ess_under_2'], 0.)
        self.assertAlmostEqual(w['median_max_weight'], (1 - math.exp(-1 / 8)) / (1 - math.exp(-2)), places=12)
        self.assertFalse(w['low_beta_reported']);self.assertNotIn('reweighted_br_fp:beta=0.25', summary['settings'])
        self.assertEqual(summary['scoring_rule'], S.SCORING_RULE);self.assertIn('Scoring rule:', (out / 'summary.md').read_text())
        self.assertEqual(cover['dropped_battles']['battles'], 0);self.assertIsNone(cover['external_prior'])
        self.assertEqual((summary['guard']['priced_roots'], summary['guard']['primary']['roots'], summary['guard']['guarded_share']), (5, 5, 0.))
        for name in S.DEFINITIVE_SETTINGS:self.assertEqual(summary['metrics']['pi_filled']['all']['operators'][name]['decisions'], 5, name)
        self.assertIn('world weights (definitive set)', (out / 'summary.md').read_text())
        with redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
            S.main(['--rows', str(path), '--run-directory', str(run), '--out', str(self.root / 'out2'), '--operator-set', 'definitive'])
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):   # no --reply-prior
            S.main(['--rows', str(path), '--run-directory', str(run), '--out', str(self.root / 'out4'), '--operator-set', 'definitive',
                    '--world-weights', str(wpath)])

    def test_the_verdict_runs_on_the_unguarded_roots(self):
        run, path, wpath, ppath = self.definitive_inputs(.25, guarded={(0, 0)}, tier_b={(0, 2), (0, 3)}, a_only={(0, 3)})
        summary = self.run_definitive(run, path, wpath, ppath);g = summary['guard']
        for name in summary['settings']:self.assertEqual(summary['metrics']['pi_filled']['all']['operators'][name]['decisions'], 4, name)
        self.assertEqual((g['priced_roots'], g['primary']['roots'], g['guarded_context']['roots']), (5, 4, 1))
        self.assertEqual((g['tier_b']['legal']['roots'], g['tier_b']['not_legal']['roots'], g['sensitivity']['roots']), (2, 2, 1))
        for name in summary['settings']:   # the verdict's table is the guard section's primary table
            self.assertEqual(summary['metrics']['pi_filled']['all']['operators'][name]['mean'], g['primary']['table']['pi_filled'][name]['mean'], name)
        decisions = {(d['ordinal'], d['decision_index']): d for d in map(json.loads, (self.root / 'out' / 'decisions.jsonl').read_text().splitlines())}
        root = decisions[(0, 0)];self.assertTrue(root['guard']['b'])
        self.assertEqual({op['choice'] for op in root['operators'].values()}, {root['greedy_action']})
        self.assertIn('Primary: RR on unguarded roots (4)', (self.root / 'out' / 'summary.md').read_text())

    def test_steep_weights_report_beta_quarter_and_dropped_battles_leave_every_operator(self):
        run, path, wpath, ppath = self.definitive_inputs(4.)   # ESS ~ 1 at beta .5
        drop = self.root / 'drop.json';drop.write_text(json.dumps([dict(ordinal=1, reason='journal_validation')]))
        summary = self.run_definitive(run, path, wpath, ppath, '--drop-battles', str(drop));cover = summary['coverage']
        undropped = self.run_definitive(run, path, wpath, ppath, out='undropped')['coverage']
        self.assertEqual(cover['parity'], undropped['parity'])   # the parity gate still counts a dropped battle's rows
        self.assertTrue(cover['world_weights']['low_beta_reported']);self.assertEqual(cover['world_weights']['share_ess_under_2'], 1.)
        self.assertEqual(summary['settings'], list(S.DEFINITIVE_SETTINGS + S.LOW_BETA_REWEIGHTED))
        self.assertIn('reweighted_br_fp:beta=0.25', summary['reported_only']);self.assertEqual(summary['candidates'], list(S.DEFINITIVE_CANDIDATES))
        self.assertEqual(cover['dropped_battles'], dict(battles=1, by_reason={'journal_validation': 1}, rows={'journal_validation': 3}, ordinals=[1]))
        for name in summary['settings']:self.assertEqual(summary['metrics']['pi_filled']['all']['operators'][name]['decisions'], 4, name)
        self.assertEqual(cover['scored'], 4)
        decisions = [json.loads(x) for x in (self.root / 'out' / 'decisions.jsonl').read_text().splitlines()]
        self.assertEqual([d['reason'] for d in decisions if d['ordinal'] == 1], ['battle_dropped:journal_validation'] * 3)
        drop.write_text(json.dumps([dict(ordinal=1, reason='a'), dict(ordinal=1, reason='b')]))
        with self.assertRaises(ValueError):S.load_dropped_battles(drop)
        with redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
            S.main(['--rows', str(path), '--run-directory', str(run), '--out', str(self.root / 'out3'), '--world-weights', str(wpath)])


class TemperedTests(unittest.TestCase):
    """Sixteen worlds, our legal actions 0, 1, 2, 4 and 6, kept sets that differ by world, and four waiting worlds."""

    def setUp(self):
        rng = random.Random(5);self.legal = legal_mask(0, 1, 2, 4, 6);worlds = []
        for w in range(16):
            if w % 4 == 3:worlds.append(world_row(lambda a, b: round(rng.uniform(-1, 1), 4), requested=False, root_legal=self.legal));continue
            kept = legal_mask(*sorted(rng.sample(range(9), rng.randint(1, 9))))
            worlds.append(world_row(lambda a, b: round(rng.uniform(-1, 1), 4), root_legal=self.legal, kept=kept,
                                    logits=[rng.uniform(-4, 4) for _ in range(9)]))
        self.rows = worlds;self.base = S.recompute(decision(worlds, root_legal=self.legal))

    def test_t1_reproduces_averaging_exactly(self):
        q, action = S.tempered(self.base.grids, self.base.worlds, 1)
        self.assertEqual(q, {a: self.base.average.q[a] for a in range(9) if self.legal[a]})
        self.assertEqual(action, hedge.top_action(self.base.average.strategy))

    def test_t_inf_is_uniform_over_kept_lanes(self):
        q, _ = S.tempered(self.base.grids, self.base.worlds, math.inf)
        for a in q:
            hand = 0.
            for world in self.rows:
                g = world['grid']
                lanes = [b for b in range(9) if world['kept'][b]] if world['opponent_requested'] else [0]
                hand += sum(g[a][b] for b in lanes) / len(lanes) / 16
            self.assertAlmostEqual(q[a], hand, places=12)

    def test_t2_by_hand(self):
        q, _ = S.tempered(self.base.grids, self.base.worlds, 2)
        for a in q:
            hand = 0.
            for world in self.rows:
                g = world['grid']
                if not world['opponent_requested']:hand += g[a][0] / 16;continue
                weights = {b: math.exp(world['root_peer_logits'][b] / 2) for b in range(9) if world['kept'][b]};total = sum(weights.values())
                hand += sum(weights[b] / total * g[a][b] for b in weights) / 16
            self.assertAlmostEqual(q[a], hand, places=12)

    def test_waiting_world_is_unaffected_by_t(self):
        for world in self.base.worlds:
            if not world.opponent_requested:
                for t in (1, 2, 4, math.inf):self.assertEqual(S.tempered_prior(world, t), S.WAITING)
        waiting = S.recompute(decision(waiting_worlds([{0: .3, 1: -.2, 2: .1, 4: .5, 6: 0.}] * 16, self.legal), root_legal=self.legal))
        for t in (2, 4, math.inf):
            self.assertEqual(S.tempered(waiting.grids, waiting.worlds, t)[0], {a: waiting.average.q[a] for a in range(9) if self.legal[a]})

    def test_temperature_moves_the_action(self):
        # Replies 0 (prior .95) and 1; action 0: [.5, -.8], action 1: [.1, .1]. q0 at T = 1, 2, 4, inf: .435, .257, .079, -.15.
        legal = legal_mask(0, 1);cells = {0: (.5, -.8), 1: (.1, .1)}
        worlds = [world_row(lambda a, b: cells[a][b], root_legal=legal, kept=legal_mask(0, 1), logits=[math.log(.95), math.log(.05)] + [0.] * 7)] * 16
        out = S.score_decision(decision(worlds, greedy=1, root_legal=legal), S.censored('other', ''), None, collections.Counter())
        c = {n: o['choice'] for n, o in out['operators'].items()}
        self.assertEqual((c['averaging'], c['tempered:T=2'], c['tempered:T=4'], c['tempered:T=inf']), (0, 0, 1, 1))

    def test_tempered_is_a_candidate_after_the_existing_settings(self):
        self.assertEqual(S.SETTINGS[-4:], ('target_argmax', 'tempered:T=2', 'tempered:T=4', 'tempered:T=inf'))
        self.assertEqual(S.CANDIDATES, S.SETTINGS[2:])
        self.assertFalse(set(S.EXTERNAL_SETTINGS) & set(S.CANDIDATES))


class ExternalPriorTests(unittest.TestCase):
    def setUp(self):self.directory = TemporaryDirectory();self.root = Path(self.directory.name)

    def tearDown(self):self.directory.cleanup()

    def write(self, rows):
        path = self.root / f'prior-{len(list(self.root.iterdir()))}.jsonl'
        path.write_text(''.join((r if type(r) is str else json.dumps(r)) + '\n' for r in rows));return path

    def test_load(self):
        prior = S.load_reply_prior(self.write([dict(ordinal=3, decision_index=1, world=w, logits=[float(w)] * 9) for w in (0, 5)] + ['']))
        self.assertEqual(set(prior), {(3, 1)});self.assertEqual(sorted(prior[3, 1]), [0, 5]);self.assertEqual(list(prior[3, 1][5]), [5.] * 9)
        bad = (dict(ordinal=3, decision_index=1, world=0, logits=[0.] * 8), dict(ordinal=3, decision_index=1, world='0', logits=[0.] * 9),
               '{"ordinal": 3, "decision_index": 1, "world": 0, "logits": [NaN, 0, 0, 0, 0, 0, 0, 0, 0]}')
        for row in bad:
            with self.assertRaises(ValueError):S.load_reply_prior(self.write([row]))
        with self.assertRaises(ValueError):S.load_reply_prior(self.write([dict(ordinal=3, decision_index=1, world=0, logits=[0.] * 9)] * 2))

    def worlds(self, n=4):
        legal = legal_mask(0, 1);cells = {0: (.5, -.8), 1: (.1, .1)}
        rows = [world_row(lambda a, b: cells[a][b], root_legal=legal, kept=legal_mask(0, 1), logits=[math.log(.95), math.log(.05)] + [0.] * 7)] * n
        return legal, S.recompute(decision(rows, root_legal=legal))

    def test_external_br_on_the_champions_logits_is_averaging(self):
        legal, base = self.worlds()
        priors = [S.external_prior(w, w.opponent_logits) for w in base.worlds]
        self.assertEqual(S.prior_q(base.grids, priors), {a: base.average.q[a] for a in (0, 1)})
        found, missing = S.external_operators(base.grids, base.worlds, {w: base.worlds[w].opponent_logits for w in range(4)})
        self.assertEqual((found['external_br'], missing), (hedge.top_action(base.average.strategy), []))

    def test_external_and_mixes(self):
        # The champion predicts reply 0 (.95), the external model reply 1 (.95). P(reply 0) under the alpha mix: .725, .5, .275.
        legal, base = self.worlds()
        found, _ = S.external_operators(base.grids, base.worlds, {w: [math.log(.05), math.log(.95)] + [0.] * 7 for w in range(4)})
        self.assertEqual(found, {'external_br': 1, 'external_mix:alpha=0.25': 0, 'external_mix:alpha=0.5': 1, 'external_mix:alpha=0.75': 1})
        mixed = S.external_prior(base.worlds[0], [math.log(.05), math.log(.95)] + [0.] * 7, .25)
        self.assertAlmostEqual(mixed[0], .725);self.assertEqual(mixed[2:], (0.,) * 7)

    def test_missing_worlds_exclude_and_waiting_worlds_need_none(self):
        legal, base = self.worlds()
        found, missing = S.external_operators(base.grids, base.worlds, {w: [0.] * 9 for w in (0, 1, 3)})
        self.assertEqual((found, missing), (dict.fromkeys(S.EXTERNAL_SETTINGS), [2]))
        waiting = S.recompute(decision(waiting_worlds([{0: .3, 1: .1}] * 4, legal), root_legal=legal))
        found, missing = S.external_operators(waiting.grids, waiting.worlds, {})
        self.assertEqual((set(found.values()), missing), ({0}, []))


if __name__ == '__main__':
    unittest.main()
