"""Rest tail recovery stays fail-closed without changing a qualified treatment."""
import ast
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest import mock

from pokezero.mcts_eval.followthrough import continuation_seed
from pokezero.mcts_eval.wider_rest_tail_recovery import (
    IDENTITY,PARENT,QUALIFIED,PUBLIC_OBSERVER,SEMANTIC_OBSERVER,VALID,
    RemoveRestTailGuard,bind_fresh_audits,bind_proofs,resume_capture,
    validate_prefix_work,validate_resumed_workers,
)
from pokezero.mcts_eval.wider_search import ARMS,SEATS,game_identity,study_seed
from wider_search_comparison import register


def clock_draw():
    return dict(status='DEADLINE_CANCELLED',sampling_diagnostic=dict(
        schema='pokezero.world-sampling-deadline.v1',accepted_world=False,backed_up=False,
        checked_at=11.,deadline_at=10.))


def witness(boundary,cancelled=False):
    receipts=[dict(worker=i,batch=dict(world_draws=2 if cancelled else 1,
        trajectories=1,transitions=1,deadline_exhausted=cancelled),evidence=dict(draws=[
            dict(status='ROOT_VALIDATED',released=True),*([clock_draw()] if cancelled else [])]))
        for i in range(20)]
    return dict(worker_receipts=receipts,result=dict(trajectories=20,transitions=20,
        world_draws=40 if cancelled else 20),decision_id=378+boundary,statistics_checkpoint='accepted393')


class PrefixTests(unittest.TestCase):
    def fixture(self):
        paths={f'boundary-{i:03d}.json.gz':f'/accepted/boundary-{i:03d}.json.gz' for i in range(16)}
        old=dict(per_game_wall_seconds=2400.,repair_retention=dict(resume=dict(
            identity=IDENTITY,prefix_steps=dict(list(paths.items())[:15]),
            elapsed_before_resume=337.,wall_clock_envelope=dict(additional_attempts=[dict(charged_seconds=14.)]))))
        capture=dict(status='ACCEPTED393_PRESERVED_NEXT394_ALL20_CLOCK_CANCELLED',source_commit=PARENT,
            accepted_boundaries=16,last_accepted_decision_id=393,failed_decision_id=394,
            failed_work_or_rng_restored=False,failure_scored_as_loss=False,strength_inference=False,
            accepted_prefix_steps=paths,accepted_prefix_hashes={p:'hash' for p in paths},
            accepted_checkpoint=paths['boundary-015.json.gz'],accepted_checkpoint_sha256='hash',
            worker_ordinals=[17]*20,elapsed_before_attempt=337.,cumulative_game_seconds=363.,
            wall_clock_envelope=dict(start_birthtime=100.,terminal_mtime=125.,slack_seconds=1.,charged_seconds=26.))
        rows={paths[f'boundary-{i:03d}.json.gz']:dict(boundary=i,actions={'p1':3},
            chance_seed=continuation_seed('wider-search:474486314:p1',0,i,'chance'),
            evidence=dict(p1=dict(selector='paper_reference',elapsed_seconds=10.,
                seed=continuation_seed('wider-search:474486314:p1',0,i,'search'),
                search_evidence=witness(i,cancelled=i==15)))) for i in range(16)}
        return capture,old,rows

    def resume(self,capture,old,rows):
        with mock.patch('pokezero.mcts_eval.wider_rest_tail_recovery.read_step',side_effect=lambda p:rows[str(p)]),\
             mock.patch.object(Path,'stat',side_effect=[NS(st_birthtime=100.),NS(st_mtime=125.)]):
            return resume_capture(capture,old,'/failed',sha=lambda p:'hash',
                restore_reference_checkpoint=lambda c:NS(battle_id='wider-search:'+IDENTITY))

    def test_all16_393_original_attempted_ordinals_and_cumulative_wall_preserved(self):
        capture,old,rows=self.fixture()
        result=self.resume(capture,old,rows)
        self.assertEqual(result['start_boundary'],16)
        self.assertEqual(result['decision_id'],393)
        self.assertEqual(result['worker_ordinals'],[17]*20)
        self.assertEqual(result['elapsed_before_resume'],363.)
        self.assertEqual(result['checkpoint_step'],capture['accepted_checkpoint'])
        self.assertFalse(result['canonical_terminal_is_truncated'])
        self.assertEqual(old['repair_retention']['resume']['elapsed_before_resume'],337.)

    def test_failed394_or_diagnostics_never_replace_accepted393(self):
        for key,value in (('last_accepted_decision_id',394),('failed_decision_id',395),
                ('worker_ordinals',[18]*20),('failed_work_or_rng_restored',True),
                ('failure_scored_as_loss',True),('strength_inference',True),('accepted_boundaries',15)):
            capture,old,rows=self.fixture()
            capture[key]=value
            with self.subTest(key=key),self.assertRaises(RuntimeError):
                self.resume(capture,old,rows)

    def test_chance_selector_prefix_checkpoint_or_paid_time_drift_refuses(self):
        for mode in ('chance','selector','prefix','checkpoint','wall','allowance','uncounted_cancellation'):
            capture,old,rows=self.fixture()
            row=rows['/accepted/boundary-015.json.gz']
            if mode=='chance':row['chance_seed']+=1
            elif mode=='selector':row['evidence']['p1']['selector']='raw'
            elif mode=='prefix':old['repair_retention']['resume']['prefix_steps']['boundary-000.json.gz']='/other'
            elif mode=='checkpoint':capture['accepted_checkpoint']='/diagnostic'
            elif mode=='wall':capture['cumulative_game_seconds']-=26.
            elif mode=='allowance':old['per_game_wall_seconds']=360.
            else:capture['worker_ordinals']=[16]*20
            with self.subTest(mode=mode),self.assertRaises(RuntimeError):
                self.resume(capture,old,rows)

    def test_clock_backup_shared_deadline_and_work_mutations_refuse(self):
        edits=[lambda w:w['worker_receipts'][0]['evidence']['draws'][1]['sampling_diagnostic'].update(backed_up=True),
            lambda w:w['worker_receipts'][0]['evidence']['draws'][1]['sampling_diagnostic'].update(deadline_at=9.),
            lambda w:w['worker_receipts'][0]['evidence']['draws'][1].update(substitute_policy_conditioning={}),
            lambda w:w['worker_receipts'][0]['batch'].update(trajectories=2),
            lambda w:w['result'].update(world_draws=20),lambda w:w['result'].update(trajectories=0),
            lambda w:w['worker_receipts'].pop()]
        for index,edit in enumerate(edits):
            w=witness(15,True)
            edit(w)
            with self.subTest(index=index),self.assertRaises(RuntimeError):
                validate_prefix_work(w)


class QualificationTests(unittest.TestCase):
    def fixture(self):
        root=Path('/qualification')
        sem,pub=Path('/audits/semantic.json'),Path('/audits/public.json')
        seeds=[study_seed(i,qualification=True) for i in range(2)]
        identities=[game_identity(s,seat,arm) for s in seeds for seat in SEATS for arm in ARMS]
        m=dict(source_commit=QUALIFIED,phase='QUALIFICATION_NOT_STRENGTH',registered_games=8,
            seeds=seeds,reference=dict(staged_substitute_conditioning=True))
        data={str(root/'registration.json'):m}
        for identity in identities:
            data[str(root/(identity+'.json'))]=dict(status='COMPLETE',registration_sha256='hash',step_hashes={'boundary-000.json.gz':'hash'})
        for path,observer,status in ((sem,SEMANTIC_OBSERVER,VALID),(pub,PUBLIC_OBSERVER,'PUBLIC_AND_SUBSTITUTE_CERTIFICATES_VALID')):
            data[str(path)]=dict(observer_sha256=observer,source_commit=QUALIFIED,registration_sha256='hash',
                audited_complete_games=8,complete_roster_valid=True,strength_inference=False,
                games=[dict(identity=i,status=status,result_sha256='hash') for i in identities],input_hashes={})
        data[str(pub)].update(schema='pokezero.wider-search.rest-tail-conditioning-audit.v4',full_replay_audit_sha256='hash')
        return root,sem,pub,data

    def bind(self,root,sem,pub,data):
        def digest(path):
            if Path(path).name=='validate_guarded_qualification_games_r31.py':return SEMANTIC_OBSERVER
            if Path(path).name=='validate_rest_tail_qualification_public_r61.py':return PUBLIC_OBSERVER
            return 'hash'
        # Autospec preserves the particular path for each mocked file read.
        with mock.patch.object(Path,'read_text',autospec=True,side_effect=lambda p:json.dumps(data[str(p)])),\
             mock.patch.object(Path,'iterdir',side_effect=lambda:iter([Path('boundary-000.json.gz')])),\
             mock.patch.object(Path,'is_file',return_value=True):
            return bind_fresh_audits(sem,pub,root/'READOUT.json',sha=digest)

    def test_exact8_v2_qualification_bound(self):
        root,sem,pub,data=self.fixture()
        inputs=self.bind(root,sem,pub,data)
        self.assertIn(str(root/'registration.json'),inputs)

    def test_partial_old_source_or_v1_report_cannot_transfer(self):
        for mode in ('partial','old_source','v1','wrong_semantic','strength','incomplete','duplicate'):
            root,sem,pub,data=self.fixture()
            report=data[str(pub)]
            if mode=='partial':report['games'].pop()
            elif mode=='old_source':data[str(root/'registration.json')]['source_commit']=PARENT
            elif mode=='v1':report['schema']='pokezero.wider-search.substitute-conditioning-audit.v3'
            elif mode=='wrong_semantic':report['full_replay_audit_sha256']='other'
            elif mode=='strength':report['strength_inference']=True
            elif mode=='incomplete':report['complete_roster_valid']=False
            else:report['games'][-1]=deepcopy(report['games'][0])
            with self.subTest(mode=mode),self.assertRaises(RuntimeError):
                self.bind(root,sem,pub,data)


class OperationalTests(unittest.TestCase):
    def test_only_explicit_guard_is_permitted_run_change(self):
        before=ast.parse('def run():\n    if old:\n        action()\n    else:\n        final()\n')
        after=ast.parse("""def run():
    if old:
        action()
    elif recovery.get('kind') == 'native-and-reference-rest-tail-repair':
        from pokezero.mcts_eval.wider_rest_tail_recovery import validate_resumed_workers
        validate_resumed_workers(measured.worker_receipts, probe)
    else:
        final()
""")
        self.assertEqual(ast.dump(before),ast.dump(RemoveRestTailGuard().visit(after)))
        changed=ast.parse("""if recovery.get('kind') == 'native-and-reference-rest-tail-repair':
    from pokezero.mcts_eval.wider_rest_tail_recovery import validate_resumed_workers
    validate_resumed_workers(diagnostic_statistics, probe)
""")
        with self.assertRaises(RuntimeError):RemoveRestTailGuard().visit(changed)

    def test_explicit_qualification_and_recovery_modes_cannot_mix(self):
        with self.assertRaisesRegex(RuntimeError,'exactly one'):
            register(NS(staged_substitute_conditioning=True,qualification=False,
                guarded_staged_recover_from=Path('old'),rest_tail_recover_from=Path('new')))
        for qualification,staged in ((True,True),(False,False)):
            with self.subTest(qualification=qualification),self.assertRaisesRegex(RuntimeError,'confirmation'):
                register(NS(staged_substitute_conditioning=staged,qualification=qualification,
                    rest_tail_recover_from=Path('new')))

    def test_missing_proof_hashes_fail_closed(self):
        with self.assertRaisesRegex(RuntimeError,'proof drift'):
            bind_proofs('unknown',sha=lambda p:'bad')

    def test_all20_expected_fingerprints_even_if_actual_first_draw_cancels(self):
        draw=dict(ordinal=0,status='ROOT_VALIDATED',released=True,packed_team_sha256='team',
            materialization_seed=19,substitute_policy_conditioning=dict(schema='v2'))
        probe=dict(measurement=dict(worker_receipts=[dict(worker=i,evidence=dict(draws=[deepcopy(draw)])) for i in range(20)]))
        actual=deepcopy(probe['measurement']['worker_receipts'])
        actual[0]['evidence']['draws']=[dict(ordinal=0,**clock_draw())]
        validate_resumed_workers(actual,probe)
        probe['measurement']['worker_receipts'][0]['evidence']['draws']=[dict(ordinal=0,**clock_draw())]
        with self.assertRaisesRegex(RuntimeError,'first-draw proof incomplete'):
            validate_resumed_workers(actual,probe)


if __name__=='__main__':
    unittest.main()
