"""Authored JSON and actor/clock fakes only; no JAX/model/game/corpus calls."""
from copy import deepcopy
from dataclasses import dataclass, fields
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pokezero import owning_encoder_cost as cost
from pokezero import owning_encoder_study as study
from pokezero.recorded_mirror_roots import CHAMPION_CONTRACT
from pokezero.live_protocol_bridge import OwningWireSession, OwningActorFactory
from pokezero.live_protocol_actor import ActorContract
from pokezero.foulplay_bridge import ControlledFoulPlayConfig as RealConfig


@dataclass
class Controls:
    owning_decision_timeout_seconds: float = .25
    owning_public_single_line: bool = True


class FakeActor:
    def __init__(self, events, fail=False):
        self.events, self.fail = events, fail
        self.generation, self.lines = 0, []
        self.prefix_sha256 = hashlib.sha256().hexdigest()

    def begin_game(self, **identity):
        self.identity = identity
        self.events.append(('begin', identity))

    def accept_owner_stream(self, **data):
        assert data['generation'] == self.generation and data['prefix_sha256'] == self.prefix_sha256
        assert data['battle_id'] == self.identity['battle_id'] and data['owner'] == self.identity['owner']
        assert len(data['lines']) == 1
        self.events.append(('literal', data['lines'][0]))
        self.lines.extend(data['lines']); self.generation += 1
        self.prefix_sha256 = hashlib.sha256(''.join(s+'\n' for s in self.lines).encode()).hexdigest()

    def snapshot_boundary(self):
        self.events.append(('snapshot',))
        self.snapshot = SimpleNamespace(prefix_sha256=self.prefix_sha256,
            own_request_line=[s for s in self.lines if s.startswith('|request|')][-1])
        return self.snapshot

    def decide(self, snapshot, *, rng):
        assert snapshot is self.snapshot
        self.events.append(('ordinary_actor_decide',))
        if self.fail:
            raise RuntimeError('authored encode refusal')

    def choice(self, snapshot):
        assert snapshot is self.snapshot
        self.events.append(('choice',))
        return 'move 1'

    def close_game(self):
        self.events.append(('close',))
        return dict(status='ABORTED_UNQUALIFIED', error=None, close_error=None)


class FakeSession:
    def __init__(self,factory,runtime,row,events):
        self.actor=factory.actor(runtime)
        self.row,self.events=row,events
        self.owner=row['seat']
        self.started=False
        self.terminal_deadline=None

    def begin(self):
        self.actor.begin_game(battle_id=self.row['battle_id'],owner=self.owner,seed=self.row['seed'],stream_id=1)

    def boundary(self,**kwargs):
        self.events.append(('live_boundary_guard',))
        return OwningWireSession.boundary(self,**kwargs)


class CostTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.addCleanup(patch.stopall)
        patch('subprocess.Popen',side_effect=AssertionError('no process')).start()
        patch('subprocess.run',side_effect=AssertionError('no source/runtime process')).start()
        patch.object(cost,'_runtime',side_effect=AssertionError('no actual runtime')).start()
        patch.object(study,'validate_prespec',side_effect=self.check_fake_study).start()
        patch('pokezero.foulplay_bridge.ControlledFoulPlayConfig',Controls).start()
        self.study_spec = dict(schema='pokezero.B3-owning-study-prespec.v2',
            owning_decision_timeout_seconds=.25,
            controls=dict(owning_decision_timeout_seconds=.25,owning_public_single_line=True),
            execution=dict(public_dispatch='one_literal_line_per_call'),
            admission=dict(collection_sha256='c'*64,selection_sha256='a'*64,selection_metadata_sha256='b'*64))
        self.study_ref = self.put('study.json',self.study_spec)
        self.selection = dict(path='/authored/selection',sha256='a'*64)
        self.metadata_ref = dict(path='/authored/metadata',sha256='b'*64)
        self.selected, frozen = [], []
        for index in range(24):
            battle = dict(index=index,battle_id=f'authored-{index}',seed=100+index,split='train')
            directory = self.root/f'battle-{index:04d}'; directory.mkdir()
            prefixes, rows = dict(p1=[],p2=[]), []
            for round_index in range(2):
                wires = {}
                for seat in ('p1','p2'):
                    packet = dict(side=dict(id=seat,pokemon=[]),active=[dict(moves=[dict(id='tackle',pp=35)])],rqid=round_index)
                    lines = [f'|turn|{round_index+1}','|request|'+study.canonical(packet).decode(),'|']
                    start = len(prefixes[seat]); prefixes[seat].extend(lines)
                    wires[seat] = dict(start=start,end=len(prefixes[seat]),delta=lines,
                                       prefix_sha256=study.digest(prefixes[seat]))
                rows.append(dict(battle_id=battle['battle_id'],seed=battle['seed'],root_id=f'root-{index}-{round_index}',
                    decision_round_index=round_index,requested_players=['p1','p2'],owning_wire=wires,
                    observations={'DO_NOT_DESERIALIZE_TENSOR':[999]},actions={'DO_NOT_SELECT_BY_ACTION':8},
                    outcome='DO_NOT_SELECT_BY_OUTCOME'))
            rows.append(dict(event='end',battle_id=battle['battle_id']))
            path = directory/'boundaries.jsonl.gz'
            with gzip.open(path,'wt') as stream:
                for row in rows:
                    stream.write(study.canonical(row).decode()+'\n')
            receipt = dict(battle,contract=CHAMPION_CONTRACT,files={'boundaries.jsonl.gz':cost.file_sha(path)})
            reference = self.put(f'battle-{index:04d}/battle.json',receipt)
            frozen.append(dict(battle,battle_receipt_sha256=reference['sha256'],
                source_file_sha256=cost.file_sha(path),boundaries=2,requested_views=4))
            self.selected.append(battle)
        patch('pokezero.actual_saved_wire._selection',return_value=({},self.selected,dict(selected=frozen))).start()
        self.spec = self.prepare()

    def check_fake_study(self, spec):
        # Only a fixture seam while the parent control API is unmerged.
        self.assertEqual(spec,self.study_spec)

    def put(self, name, value):
        path = self.root/name; path.write_bytes(study.canonical(value)+b'\n')
        return dict(path=str(path),sha256=cost.file_sha(path))

    def prepare(self):
        return cost.prepare_cost(study_prespec=self.study_ref,collection=self.root,
            selection=self.selection,selection_metadata=self.metadata_ref,source_commit='d'*40)

    def run_fake(self, *, fail_at=None, setup_fail=False, admission_fail=False, out='attempt'):
        events, actors = [], []
        def actor(runtime):
            self.assertEqual(runtime,('authored-model','authored-result','authored-env'))
            persisted=json.loads((self.root/out/'ledger.json').read_text())
            self.assertEqual(persisted['roots'][len(actors)]['status'],'ATTEMPTING_UNKNOWN')
            value = FakeActor(events,len(actors) == fail_at); actors.append(value)
            return value
        def runtime(source):
            self.assertTrue((self.root/out/'ledger.json').exists())
            persisted = json.loads((self.root/out/'ledger.json').read_text())
            self.assertEqual([r['status'] for r in persisted['roots']],['NOT_ATTEMPTED']*20)
            events.append(('setup',))
            if setup_fail:raise RuntimeError('authored setup failure')
            return (SimpleNamespace(actor=actor),('authored-model','authored-result','authored-env'),
                    {'fake':True},SimpleNamespace(authored=True))
        def admission(source):
            events.append(('admission',))
            if admission_fail:raise ValueError('authored B2 refusal')
            return {'fake_gate':True}
        value = [0.]
        def clock():
            events.append(('clock',))
            value[0] += .001
            return value[0]
        ledger = cost.run_cost(self.spec,self.root/out,runtime_builder=runtime,source_check=lambda spec:None,
            admission=admission,clock=clock,
            session_builder=lambda config,factory,runtime,row:FakeSession(factory,runtime,row,events))
        self.assertEqual(ledger,json.loads((self.root/out/'ledger.json').read_text()))
        return ledger,events,actors

    def test_metadata_skip_does_not_decode_tensor_action_outcome_values(self):
        text = '{"observations":{"DO_NOT_DECODE":[1,{"nested":"}\\\""}]},"actions":[2],"outcome":"win","root_id":"fixed"}'
        decoder = json.JSONDecoder.raw_decode
        def checked(instance,text,index=0):
            self.assertFalse(text[index:].startswith(('{"DO_NOT_DECODE"','[2]','"win"')))
            return decoder(instance,text,index)
        with patch.object(json.JSONDecoder,'raw_decode',checked):
            self.assertEqual(cost.metadata_object(text,{'root_id'}),{'root_id':'fixed'})

    def test_top_level_duplicates_and_trailing_input_refuse(self):
        for text in ('{"x":1,"x":2}','{"x":1} extra','{"x":1,}','{"x":[1}'):
            with self.subTest(text=text),self.assertRaises((ValueError,IndexError)):
                cost.metadata_object(text,set())

    def test_fixed_hash_selection_distinct_battles_seat_balance_order_independent(self):
        rows = self.spec['roster']
        self.assertEqual(len({r['battle_id'] for r in rows}),20)
        self.assertEqual(sum(r['seat'] == 'p1' for r in rows),10)
        self.assertEqual(self.prepare(),self.spec)
        candidates = [{k:v for k,v in r.items() if k in ('index','battle_id','seed','split','seat','root_id',
            'decision_round_index','source_file_sha256','battle_receipt_sha256','collection_sha256',
            'selection_sha256','selection_metadata_sha256')} for r in rows]
        self.assertEqual(cost.select_roster(candidates),cost.select_roster(list(reversed(candidates))))
        with self.assertRaisesRegex(ValueError,'insufficient'):
            cost.select_roster(candidates[:19])

    def test_chosen_owner_only_and_prefix_request_binding(self):
        for row in self.spec['roster']:
            data = cost._owner_input(self.root,row)
            self.assertEqual(data['binding'],{key:row[key] for key in data['binding']})
            other = 'p2' if row['seat'] == 'p1' else 'p1'
            self.assertTrue(all('"id":"'+other+'"' not in s for s in data['prior']+data['delta']))

    def test_selected_file_drift_has_no_refill(self):
        path = cost._battle_path(self.root,self.spec['roster'][0])
        with path.open('ab') as stream:stream.write(b'changed')
        with self.assertRaisesRegex(ValueError,'hash mismatch'):
            cost._owner_input(self.root,self.spec['roster'][0])

    def test_explicit_limit_and_merged_fields_fail_closed(self):
        for limit in (None,0,-1,True,float('inf'),float('nan')):
            spec = deepcopy(self.study_spec); spec['owning_decision_timeout_seconds']=limit
            with self.subTest(limit=limit),self.assertRaisesRegex(ValueError,'LIMIT_UNBOUND'):
                cost.validate_controls(spec)
        @dataclass
        class Legacy:
            owning_terminal_timeout_seconds:float=5.
        with self.assertRaisesRegex(ValueError,'unavailable'):
            cost.validate_controls(self.study_spec,config_class=Legacy)
        spec = deepcopy(self.study_spec); spec['controls']['owning_decision_timeout_seconds']=1.
        with self.assertRaisesRegex(ValueError,'must match'):cost.validate_controls(spec)
        spec = deepcopy(self.study_spec); spec['schema']='pokezero.B3-owning-study-prespec.v1'
        with self.assertRaisesRegex(ValueError,'prespec v2'):cost.validate_controls(spec)

    def test_real_entry_refuses_unmerged_control_api_before_any_runtime(self):
        @dataclass
        class Legacy:
            owning_terminal_timeout_seconds:float=5.
        with patch('pokezero.foulplay_bridge.ControlledFoulPlayConfig',Legacy),self.assertRaisesRegex(
                ValueError,'unavailable'):
            cost.run_cost(self.spec,self.root/'missing-controls',
                runtime_builder=lambda source:self.fail('runtime must not be called'))
        self.assertFalse((self.root/'missing-controls').exists())

    def test_one_setup_fresh_actor_every_root_timer_order_and_cold_samples(self):
        ledger,events,actors = self.run_fake()
        self.assertEqual(ledger['actual_exit'],0)
        self.assertEqual(len(actors),20)
        self.assertEqual(len({id(a) for a in actors}),20)
        self.assertEqual(sum(e[0] == 'setup' for e in events),1)
        self.assertLess(events.index(('admission',)),events.index(('setup',)))
        self.assertEqual(ledger['summary']['valid'],20)
        self.assertEqual(ledger['summary']['p95_rank'],19)
        for row in ledger['roots']:
            self.assertGreater(row['actor_catalog_seconds'],0)
            self.assertGreater(row['prior_priming_seconds'],0)
            self.assertAlmostEqual(row['decision_seconds'],.001)
        # The selected timer encloses delta literals, snapshot, selection and choice.
        first = actors[0]; begin = events.index(('begin',first.identity))
        close = events.index(('close',),begin)
        subset = events[begin:close]
        choice = subset.index(('choice',))
        self.assertEqual(subset[choice-1],('ordinary_actor_decide',))
        self.assertEqual(subset[choice-2],('snapshot',))
        self.assertEqual(subset[choice-3],('live_boundary_guard',))
        self.assertEqual(subset[choice+1],('clock',))
        self.assertTrue(any(e[0] == 'literal' for e in subset))
        for actor in actors:
            peer='p2' if actor.identity['owner']=='p1' else 'p1'
            self.assertTrue(all('"id":"'+peer+'"' not in line for line in actor.lines))

    def test_failure_unknown_and_remaining_not_attempted_no_retry(self):
        ledger,events,actors = self.run_fake(fail_at=3)
        self.assertEqual(ledger['actual_exit'],1)
        self.assertEqual(len(actors),4)
        self.assertEqual([r['status'] for r in ledger['roots']],['VALID']*3+['FAILED_UNKNOWN']+['NOT_ATTEMPTED']*16)
        self.assertEqual(ledger['summary']['unknown'],17)
        self.assertFalse(ledger['summary']['cost_rule_satisfied'])
        self.assertIsNone(ledger['summary']['p95_seconds'])
        self.assertEqual(ledger['failure']['stage'],'selected_decision')

    def test_setup_and_admission_failures_publish_all_unknown(self):
        for i,kwargs in enumerate((dict(setup_fail=True),dict(admission_fail=True))):
            ledger,events,actors = self.run_fake(out=f'failure-{i}',**kwargs)
            self.assertEqual(ledger['actual_exit'],1)
            self.assertFalse(actors)
            self.assertEqual(ledger['summary']['unknown'],20)
            self.assertEqual([r['status'] for r in ledger['roots']],['NOT_ATTEMPTED']*20)
            if kwargs.get('admission_fail'):self.assertNotIn(('setup',),events)

    def test_bad_selected_binding_fails_before_actor_no_replacement(self):
        original = cost._owner_input
        def wrong(collection,row):
            data = original(collection,row);data['binding']['own_request_sha256']='0'*64
            return data
        with patch.object(cost,'_owner_input',side_effect=wrong):
            ledger,_,actors=self.run_fake()
        self.assertEqual(ledger['failure']['stage'],'input')
        self.assertFalse(actors)
        self.assertEqual(ledger['roots'][0]['status'],'FAILED_UNKNOWN')

    def test_peer_cache_is_hash_bound_host_only_no_fabricated_request(self):
        original=cost._owner_input
        def wrong(collection,row):
            data=original(collection,row)
            peer='p2' if row['seat']=='p1' else 'p1'
            data['host_requests'][peer]='|request|'+study.canonical(dict(side=dict(id=peer),wait=True)).decode()
            return data
        with patch.object(cost,'_owner_input',side_effect=wrong):
            ledger,_,actors=self.run_fake()
        self.assertEqual(ledger['failure']['stage'],'input')
        self.assertFalse(actors)

    def test_existing_session_factory_contract_accepts_both_fresh_registrations(self):
        # Actual config/factory/session validation, without beginning providers
        # or constructing any backend, model, catalog, encoder or policy.
        from pokezero.foulplay_bridge import ControlledFoulPlayConfig as Patched
        self.assertIs(Patched,Controls)
        extra={key:value for key,value in dict(owning_decision_timeout_seconds=.25,
            owning_public_single_line=True).items() if key in {f.name for f in fields(RealConfig)}}
        config=RealConfig(checkpoint=Path('/authored-checkpoint'),showdown_root=Path('/authored-showdown'),
            encoder_kind='jax-owning',policy_mode='raw',device='cpu',belief_set_source=True,
            allow_search_fallback=False,opponent_crash_retries=0,**extra)
        def forbidden(*args,**kwargs):self.fail('provider/model must not be instantiated')
        factory=OwningActorFactory('jax-owning',ActorContract.champion(),forbidden,forbidden)
        sessions=[]
        for seat in ('p1','p2'):
            row=dict(battle_id='authored-'+seat,seed=42,seat=seat)
            session=cost._session(config,factory,('model','result','env'),row)
            self.assertIs(session.factory,factory)
            self.assertEqual(session.owner,seat)
            self.assertEqual(session.config.seed_start,42)
            self.assertEqual(session.config.games,1)
            self.assertEqual(session.actor.status,'IDLE')
            sessions.append(session)
        self.assertEqual(config.pokezero_player,'p1')
        self.assertIsNot(sessions[0].actor,sessions[1].actor)

    def test_reducer_nearest_rank_includes_cold_outlier_and_max(self):
        ledger,_,_=self.run_fake()
        for i,row in enumerate(ledger['roots']):row['decision_seconds']=i/100.
        summary=cost.summarize(self.spec,ledger)
        self.assertEqual(summary['p95_seconds'],.18)
        self.assertEqual(summary['maximum_seconds'],.19)
        self.assertEqual(summary['threshold_seconds'],.125)
        self.assertFalse(summary['cost_rule_satisfied'])
        ledger['roots'][-1]['decision_seconds']=1000.
        self.assertEqual(cost.summarize(self.spec,ledger)['maximum_seconds'],1000.)
        ledger['roots'][-1]['status']='FAILED_UNKNOWN'
        self.assertIsNone(cost.summarize(self.spec,ledger)['p95_seconds'])

    def test_durable_roster_before_setup_and_io_failure_blocks_model(self):
        writes=[]
        original=study.atomic_json
        def write(path,value):
            writes.append(Path(path).name)
            if len(writes)==2:raise OSError('authored pre-runtime IO failure')
            original(path,value)
        builder=lambda source:self.fail('model setup after failed ledger publication')
        with patch.object(study,'atomic_json',side_effect=write),self.assertRaises(OSError):
            cost.run_cost(self.spec,self.root/'io-fail',runtime_builder=builder)
        self.assertEqual(writes,['prespec.json','ledger.json'])

    def test_new_directory_and_full_roster_sync_precede_admission_and_model(self):
        events=[]
        sync,atomic=study._sync_directory,study.atomic_json
        def sync_directory(path):
            events.append(('directory',str(path)));sync(path)
        def publish(path,value):
            atomic(path,value);events.append(('published',Path(path).name))
        def admission(source):
            events.append(('admission',));raise ValueError('authored stop before runtime')
        with patch.object(study,'_sync_directory',side_effect=sync_directory),patch.object(
                study,'atomic_json',side_effect=publish):
            ledger=cost.run_cost(self.spec,self.root/'durable',source_check=lambda spec:None,
                admission=admission,runtime_builder=lambda source:self.fail('unexpected model'))
        self.assertEqual(events[0],('directory',str(self.root)))
        self.assertLess(events.index(('published','prespec.json')),events.index(('published','ledger.json')))
        self.assertLess(events.index(('published','ledger.json')),events.index(('admission',)))
        self.assertEqual(ledger['actual_exit'],1)

    def test_directory_sync_failure_refuses_before_any_model_or_root(self):
        with (patch.object(study,'_sync_directory',side_effect=OSError('authored directory failure')),
              self.assertRaises(OSError)):
            cost.run_cost(self.spec,self.root/'directory-fail',
                runtime_builder=lambda source:self.fail('unexpected model'))

    def test_no_resume_or_overwrite(self):
        self.run_fake()
        with self.assertRaises(FileExistsError):self.run_fake()

    def test_cli_prepare_run_and_reduce_have_no_implicit_execution(self):
        location=Path(__file__).resolve().parents[1]/'scripts/owning_encoder_cost.py'
        spec=importlib.util.spec_from_file_location('cost_cli',location)
        cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)
        with patch.object(cli,'prepare_cost',return_value=self.spec) as prepare,patch.object(cli,'run_cost') as run:
            self.put('input.json',{})
            self.assertEqual(cli.main(['prepare','--input',str(self.root/'input.json'),'--out',str(self.root/'prepared.json')]),0)
            prepare.assert_called_once_with();run.assert_not_called()


if __name__ == '__main__':unittest.main()
