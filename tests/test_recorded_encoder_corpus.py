import copy
import json
import unittest

from pokezero.mirror_value_probe import canonical,digest
from pokezero.recorded_encoder_corpus import RecordedEncoderCursor,VISIBILITY
from test_recorded_mirror_roots import boundary


class RecordedEncoderCorpusTests(unittest.TestCase):
    battle={'battle_id':'routing-fixture','seed':1234}

    def row(self):
        row=boundary(self.battle)
        for seat in ('p1','p2'):row['requests'][seat]['private_fixture']=seat+' secret'
        row['protocol_delta']=['|turn|3','|-damage|p2a: Absol|133/299']+[
            '|request|'+json.dumps(row['requests'][seat]) for seat in ('p1','p2')]
        row['protocol_end']=len(row['protocol_delta']);row['public_prefix_sha256']=digest(row['protocol_delta'])
        row['root_id']=digest([row['battle_id'],0,row['public_prefix_sha256'],row['requests']])
        return row

    def test_own_request_isolation_preserves_trusted_conditions_and_hashes(self):
        row=self.row();cursor=RecordedEncoderCursor(**self.battle);projection=cursor.advance(row);views=projection.views
        self.assertEqual(cursor.source_prefix_sha256,digest(row['protocol_delta']))
        self.assertEqual(len(views),2);self.assertEqual(len({v.decision_id for v in views}),2)
        for view in views:
            self.assertEqual(json.loads(view.own_request_json),row['requests'][view.seat])
            other='p2' if view.seat=='p1' else 'p1'
            self.assertNotIn((other+' secret').encode(),view.own_request_json)
            self.assertFalse(any(line.startswith('|request|') for line in projection.trusted_nonrequest_delta))
            self.assertIn('|-damage|p2a: Absol|133/299',projection.trusted_nonrequest_delta)
            self.assertEqual(view.visibility,VISIBILITY);self.assertFalse(view.public_wire_qualified)
            self.assertEqual(projection.trusted_nonrequest_prefix_sha256,digest(list(projection.trusted_nonrequest_delta)))
        row['requests']['p1']['private_fixture']='mutated'
        self.assertEqual(json.loads(views[0].own_request_json)['private_fixture'],'p1 secret')

    def test_asymmetric_boundary_retains_inactive_request_digest_without_exposing_it(self):
        first=self.row();cursor=RecordedEncoderCursor(**self.battle);cursor.advance(first)
        row=copy.deepcopy(first);row['decision_round_index']=1
        row['requests']['p2']['wait']=True;row['requested_players']=['p1'];row['paired']=False
        row['observations']={'p1':row['observations']['p1']};row['actions']={'p1':0}
        row['protocol_start']=first['protocol_end'];row['protocol_delta']=['|turn|4','|request|'+json.dumps(row['requests']['p2'])]
        row['protocol_end']=row['protocol_start']+len(row['protocol_delta'])
        row['public_prefix_sha256']=digest(first['protocol_delta']+row['protocol_delta'])
        row['root_id']=digest([row['battle_id'],1,row['public_prefix_sha256'],row['requests']])
        projection=cursor.advance(row);views=projection.views;self.assertEqual([v.seat for v in views],['p1'])
        self.assertEqual(projection.trusted_nonrequest_delta,('|turn|4',));self.assertNotIn(b'p2 secret',views[0].own_request_json)

    def test_bad_boundary_never_advances_or_silently_drops_source_lines(self):
        row=self.row()
        cases=(lambda r:r.update(protocol_start=1),lambda r:r.update(public_prefix_sha256='0'*64),
            lambda r:r['requests']['p2'].update(private_fixture='mismatch'),
            lambda r:r['protocol_delta'].__setitem__(0,'|turn|3\n|turn|4'),
            lambda r:r['observations']['p2'].update(turn_number=-1),
            lambda r:r.update(requested_players=['p1']))
        for mutate in cases:
            cursor=RecordedEncoderCursor(**self.battle);bad=copy.deepcopy(row);mutate(bad)
            with self.assertRaises((ValueError,KeyError)):cursor.advance(bad)
            self.assertEqual(cursor.source_lines,0);self.assertEqual(cursor.next_round,0)
            self.assertEqual(cursor.source_prefix_sha256,digest([]));self.assertEqual(cursor._requests,{})
        cursor=RecordedEncoderCursor(**self.battle);cursor.advance(row)
        with self.assertRaisesRegex(ValueError,'gap or duplicate'):cursor.advance(row)
        terminal={'event':'end','protocol_start':cursor.source_lines,'protocol_end':cursor.source_lines+1,'protocol_delta':['|win|p1']}
        cursor.close(terminal);self.assertTrue(cursor.closed)
        with self.assertRaisesRegex(ValueError,'closed'):cursor.advance(row)

    def test_hash_stream_matches_canonical_prefix_for_escaped_text_and_bad_requests(self):
        row=self.row();row['protocol_delta'].insert(1,'|message|quote " and unicode é and backslash \\')
        row['protocol_end']=len(row['protocol_delta']);row['public_prefix_sha256']=digest(row['protocol_delta'])
        row['root_id']=digest([row['battle_id'],0,row['public_prefix_sha256'],row['requests']])
        cursor=RecordedEncoderCursor(**self.battle);cursor.advance(row)
        self.assertEqual(cursor.source_prefix_sha256,digest(row['protocol_delta']))
        for raw in ('|request|{broken','|request|{"side":[]}','|request|{"side":{"id":"p3"}}'):
            bad=self.row();bad['protocol_delta'][2]=raw
            cursor=RecordedEncoderCursor(**self.battle)
            with self.assertRaises(ValueError):cursor.advance(bad)
            self.assertEqual(cursor.next_round,0);self.assertEqual(cursor.source_lines,0)

    def test_terminal_decision_empty_winner_and_bool_seed_refuse_atomically(self):
        for mutate in (lambda r:r['protocol_delta'].insert(0,'|win|p1'),lambda r:r.update(seed=True)):
            row=self.row();mutate(row);row['protocol_end']=len(row['protocol_delta'])
            row['public_prefix_sha256']=digest(row['protocol_delta'])
            row['root_id']=digest([row['battle_id'],0,row['public_prefix_sha256'],row['requests']])
            cursor=RecordedEncoderCursor(battle_id=self.battle['battle_id'],seed=1 if row['seed'] is True else self.battle['seed'])
            with self.assertRaises(ValueError):cursor.advance(row)
            self.assertEqual(cursor.next_round,0);self.assertEqual(cursor.source_lines,0)
        for line in ('|win|','|win|  ','|win|p1|extra'):
            cursor=RecordedEncoderCursor(**self.battle)
            with self.assertRaises(ValueError):cursor.close({'event':'end','protocol_start':0,'protocol_end':1,'protocol_delta':[line]})
            self.assertFalse(cursor.closed);self.assertEqual(cursor.source_lines,0)
        for line in ('|tie','|tie|','|win|Scott player'):
            cursor=RecordedEncoderCursor(**self.battle)
            cursor.close({'event':'end','protocol_start':0,'protocol_end':1,'protocol_delta':[line]})
            self.assertTrue(cursor.closed)

    def test_shared_boundary_delta_is_emitted_once_not_on_each_owning_view(self):
        cursor=RecordedEncoderCursor(**self.battle);projection=cursor.advance(self.row())
        self.assertEqual(len(projection.views),2)
        self.assertEqual(projection.trusted_nonrequest_delta,('|turn|3','|-damage|p2a: Absol|133/299'))
        for view in projection.views:self.assertFalse(hasattr(view,'trusted_nonrequest_delta'))
