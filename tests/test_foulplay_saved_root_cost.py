"""Authored parser/role/shape controls only; no opponent imports or native calls."""
from copy import deepcopy
import json
import hashlib
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch
from types import SimpleNamespace as NS
import unittest
from pokezero.foulplay_saved_root_cost import actor_protocol_line, FoulPlayParserAPI, SavedRootFoulPlayParser


def request(seat, moves=('splash',), **extra):
    return dict(side=dict(id=seat, secret='private-'+seat),
                active=[dict(moves=[dict(id=m) for m in moves])], **extra)


def line(req):
    return '|request|' + json.dumps(req)


class ProjectionTests(unittest.TestCase):
    def test_health_matches_pinned_percentage_mod_including_nearly_full(self):
        for exact, public in [('335/335','100/100'),('334/335','99/100'),
                              ('1/335','1/100'),('169/335 par','51/100 par'),('0 fnt','0 fnt')]:
            for actor,peer in [('p1','p2'),('p2','p1')]:
                for command,tail in [('switch','Toy, L80|'),('drag','Toy, L80|'),
                                     ('replace','Toy, L80|'),('-damage',''),('-heal','')]:
                    original=f'|{command}|{peer}a: Toy|{tail}{exact}'
                    self.assertEqual(actor_protocol_line(original,actor),
                                     f'|{command}|{peer}a: Toy|{tail}{public}')
                    own=original.replace(peer+'a:',actor+'a:')
                    self.assertEqual(actor_protocol_line(own,actor),own)
    def test_sethp_projects_each_owning_role_separately(self):
        self.assertEqual(actor_protocol_line('|-sethp|p1a: One|73/335|p2a: Two|334/335','p1'),
                         '|-sethp|p1a: One|73/335|p2a: Two|99/100')
    def test_private_requests_never_cross_actor_role(self):
        for actor in ('p1','p2'):
            own=line(request(actor));foreign=line(request('p2' if actor=='p1' else 'p1'))
            self.assertEqual(actor_protocol_line(own,actor),own)
            self.assertIsNone(actor_protocol_line(foreign,actor))
    def test_unknown_or_malformed_health_refuses(self):
        for value in ['|unknown|p2a: Toy|3/335','|-damage|p2a: Toy|336/335',
                      '|-heal|missing-owner|3/335','|-heal|p2a: Toy|1/0',
                      '|-heal|p2a: Toy|unrecognized','|request|{"wait":true}']:
            with self.subTest(value=value),self.assertRaises(ValueError):actor_protocol_line(value,'p1')


class User:
    def __init__(self):
        self.name='';self.active=NS(name='ownmon',moves=[]);self.reserve=[NS(name='benchmon',index=2)]
    def initialize_first_turn_user_from_json(self,req):self.update_from_request_json(req)
    def update_from_request_json(self,req):
        self.active.moves=[NS(name=x['id']) for x in req.get('active',[{'moves':[]}])[0]['moves']]


class Battle:
    def __init__(self,tag):
        self.battle_tag=tag;self.user=User();self.opponent=User();self.team_preview=False
        self.msg_list=[];self.turn=1;self.started=False;self.request_json=None


class RuntimeSnapshotTests(unittest.TestCase):
    def factory(self, cache, digest, dataset, **config_changes):
        settings=dict(pokemon_format='gen3randombattle',search_time_ms=1000,
                      parallelism=1,search_threads=1,search_iterations=0)
        settings.update(config_changes)
        modules={
            'constants':NS(BattleType=NS(RANDOM_BATTLE='random')),
            'data.pkmn_sets':NS(PKMN_SETS_CACHE_DIR=str(cache),RandomBattleTeamDatasets=dataset),
            'fp.battle':NS(Battle=Battle,LastUsedMove=lambda *args:args),
            'fp.battle_modifier':NS(process_battle_updates=lambda b:None,update_battle=lambda b,m:None),
            'config':NS(FoulPlayConfig=NS(**settings)),
        }
        with patch.dict(sys.modules,modules):
            return FoulPlayParserAPI.from_bound_runtime(randbats_cache_sha256=digest)
    def test_exact_dataset_snapshot_survives_later_path_replacement_without_network(self):
        with tempfile.TemporaryDirectory() as directory:
            cache=Path(directory);target=cache/'gen3randombattle.json'
            blob=b'{"authored_A":{"set":2}}';target.write_bytes(blob);seen=[]
            dataset=NS(_initialize_pkmn_sets=lambda:seen.append(deepcopy(dataset.raw_pkmn_sets)))
            api=self.factory(cache,hashlib.sha256(blob).hexdigest(),dataset)
            target.write_bytes(b'{"authored_B":{"set":999}}')
            api.initialize_dataset('gen3');api.initialize_dataset('gen3')
            self.assertEqual(seen,[{'authored_A':{'set':2}}]*2)
            self.assertEqual(dataset.pkmn_mode,'gen3')
            with self.assertRaises(ValueError):api.initialize_dataset('gen4')
    def test_missing_changed_or_empty_cache_and_wrong_configuration_refuse(self):
        with tempfile.TemporaryDirectory() as directory:
            cache=Path(directory);target=cache/'gen3randombattle.json';dataset=NS()
            with self.assertRaises(FileNotFoundError):self.factory(cache,'0'*64,dataset)
            target.write_text('{}')
            for digest in ('0'*64,hashlib.sha256(target.read_bytes()).hexdigest()):
                with self.assertRaises(ValueError):self.factory(cache,digest,dataset)
            target.write_text('{"authored":1}');digest=hashlib.sha256(target.read_bytes()).hexdigest()
            for changed in ({'search_time_ms':500},{'parallelism':2},{'search_threads':2},
                            {'search_iterations':100},{'pokemon_format':'gen4randombattle'}):
                with self.subTest(changed=changed),self.assertRaises(ValueError):
                    self.factory(cache,digest,dataset,**changed)


class ParserTests(unittest.TestCase):
    def setUp(self):
        self.calls=[]
        def process(b):
            self.calls.append(('updates',list(b.msg_list)));b.msg_list.clear()
        def update(b,msg):
            self.calls.append(('update',msg));parts=msg.split('\n');b.request_json=json.loads(parts[-1][len('|request|'):]);b.msg_list.extend(parts[:-1]);process(b)
        self.api=FoulPlayParserAPI(Battle,'random',lambda gen:self.calls.append(('dataset',gen)),
                                  update,process,lambda *args:args)
    def opening(self,actor='p1'):
        own=request(actor,('hiddenpowerice70','splash'));peer='p2' if actor=='p1' else 'p1';foreign=request(peer)
        return dict(battle_id='authored',decision_round_index=0,protocol_delta=['|player|p1|One','|start',
                    f'|switch|{actor}a: Own|Toy, L80|335/335',
                    f'|switch|{peer}a: Peer|Toy, L80|334/335',line(own),line(foreign)],
                    requested_players=['p1','p2'],requests={actor:own,peer:foreign},actions={actor:0,peer:0})
    def test_opening_each_seat_uses_production_init_without_own_switch_duplication(self):
        for seat in ('p1','p2'):
            parser=SavedRootFoulPlayParser(actor=seat,battle_id='authored',api=self.api)
            parser.feed_boundary(self.opening(seat),selected=True)
            candidate=parser.selected_decision_copy();self.assertEqual(candidate.user.active.moves[0].name,'hiddenpowerice70')
            self.assertEqual(candidate.request_json['rqid'],1)
            proof=parser.proof();self.assertEqual(proof['foreign_requests_excluded'],1)
            self.assertFalse(proof['selected_root_action_consumed']);self.assertEqual(proof['native_search_queries'],0)
            updates=self.calls[-1][1];self.assertEqual(len(updates),1)
            self.assertIn('99/100',updates[0]);self.assertNotIn('|switch|'+seat,updates[0])
    def test_past_recorded_choice_updates_bookkeeping_without_a_decision(self):
        parser=SavedRootFoulPlayParser(actor='p1',battle_id='authored',api=self.api)
        parser.feed_boundary(self.opening(),selected=False,recorded_own_choice='move 1')
        self.assertEqual(parser.battle.user.last_selected_move,('ownmon','hiddenpowerice70',1))
        row=self.opening();row['decision_round_index']=1;row['protocol_delta']=['|turn|2',line(row['requests']['p1'])]
        parser.feed_boundary(row,selected=True);self.assertEqual(parser.proof()['historical_own_choices_mapped_without_execution'],1)
        self.assertEqual(parser.selected_decision_copy().request_json['rqid'],2)
    def test_recorded_switch_maps_production_request_slot(self):
        parser=SavedRootFoulPlayParser(actor='p1',battle_id='authored',api=self.api)
        parser.feed_boundary(self.opening(),selected=False,recorded_own_choice='switch 2')
        self.assertEqual(parser.battle.user.last_selected_move,('ownmon','switch benchmon',1))
    def test_selected_action_is_never_read(self):
        class NoActions(dict):
            def __getitem__(self,key):
                if key=='actions':raise AssertionError('selected action leaked')
                return super().__getitem__(key)
        parser=SavedRootFoulPlayParser(actor='p1',battle_id='authored',api=self.api)
        parser.feed_boundary(NoActions(self.opening()),selected=True)
        with self.assertRaises(ValueError):parser.feed_boundary(self.opening(),selected=True)
    def test_role_without_selected_request_cannot_think(self):
        parser=SavedRootFoulPlayParser(actor='p1',battle_id='authored',api=self.api)
        parser.feed_boundary(self.opening(),selected=False,recorded_own_choice='move 1')
        own=request('p1',wait=True);row=dict(battle_id='authored',decision_round_index=1,protocol_delta=[line(own)],
                  requested_players=['p2'],requests={'p1':own})
        parser.feed_boundary(row,selected=True);self.assertFalse(parser.proof()['selected_opponent_requested'])
        with self.assertRaises(ValueError):parser.selected_decision_copy()
    def test_missing_wrong_extra_historical_choices_refuse(self):
        for choice,selected in [(None,False),('move 1',True),('move 9',False),('switch 9',False),('guess',False)]:
            parser=SavedRootFoulPlayParser(actor='p1',battle_id='authored',api=self.api)
            with self.subTest(choice=choice,selected=selected),self.assertRaises(ValueError):
                parser.feed_boundary(self.opening(),selected=selected,recorded_own_choice=choice)
    def test_duplicate_or_unknown_source_roles_refuse(self):
        for roles in (['p1','p1','p2'],['p3'],[]):
            row=self.opening();row['requested_players']=roles
            parser=SavedRootFoulPlayParser(actor='p1',battle_id='authored',api=self.api)
            with self.subTest(roles=roles),self.assertRaises(ValueError):parser.feed_boundary(row,selected=True)
    def test_missing_prefix_or_changed_request_refuses(self):
        for change in ('start','request','order'):
            row=self.opening()
            if change=='start':row['protocol_delta'].remove('|start')
            elif change=='request':row['requests']['p1']['side']['secret']='changed'
            else:row['decision_round_index']=3
            parser=SavedRootFoulPlayParser(actor='p1',battle_id='authored',api=self.api)
            with self.subTest(change=change),self.assertRaises(ValueError):parser.feed_boundary(row,selected=True)

if __name__=='__main__':unittest.main()
