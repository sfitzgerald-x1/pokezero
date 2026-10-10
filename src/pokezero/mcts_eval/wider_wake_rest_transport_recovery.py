"""Create-only qualified prefix/joint-tail repair; retain accepted398, never failed399.

The treatment is byte-identical to fresh all8 qualification. This module only
binds the registry, immutable proofs and accepted original worker streams.
"""
import ast
import json
import math
from pathlib import Path

from .followthrough import continuation_seed
from .wider_guarded_staged_recovery import validate_resumed_workers
from .wider_native_recovery import CONTRACT_KEYS, verify_historical
from .wider_recovery import combined_steps, read_step, require
from .wider_search import ARMS, SEATS, analyze, game_identity, study_seed
from .wider_substitute_recovery import VALID, validate_prior_audit

KIND='native-and-reference-wake-rest-transport-repair'
QUALIFIED='efa1384c897f723157e8c974aeafc36d65be498b'
PARENT='0acad37c2e3ca82c7e652425ae48a565310eef9f'
IDENTITY='seed-474486314-p1-paper_reference'
SEMANTIC_OBSERVER='07abc045ba0bc26dc35332b90c1aa87e780ffa99cf9d70a4b04c335136075739'
PUBLIC_OBSERVER='93f0ba691b4f73bd3826ef83a80c0a09fb7ae8ed81eef3b97fc02b193bee6250'
PROOFS={
    'capture_prefix_joint_tail_next_root_failure_r101.json':'ff5f5737b82546d5ab633ade689008c5e13384141609c57ab232fcd5dfaafbc9',
    'probe_wake_rest_transport_original20_r109.json':'26e235c47b86f5bfca3f6d08dccdbd6f880baec1277869f2e293083dd08a00e0',
    'validate_wake_rest_transport_witnesses_r110.json':'abc2eae94c264178efbbb53347ea013a9ff72701140c524ba98af746bc9a1fbb',
    'validate_wake_rest_transport_stream_r111.json':'27e84834dc17a51c01a69b7c7a8d6ae658b2d3ad683c0a6cfb412cf5d8723669',
    'audit_wake_rest_transport_source_law_r112.json':'d99c9b77958c0f620894b7f22f91774b2df7392e32be2e1aaed5ca92f6004e32',
    'test_wake_rest_transport_public_observer_r114.json':'e1905dadd34be6f2d8287b1f3d833fd30892432393a5bd2a17dddf115907fcbb',
    'probe_wake_rest_transport_first_streams_r116.json':'9967ff707d3aacf009dbef53ecedb223a0125fbcd097845dd91138c751efe2ac',
    'validate_wake_rest_transport_first_streams_r117.json':'834f4d483d575f0c67cbd8a3f82556cca4e318efbfd165fc713a6504cc1ab556',
}


class RemoveWakeRestTransportGuard(ast.NodeTransformer):
    def visit_If(self,node):
        self.generic_visit(node)
        if ast.unparse(node.test)=="recovery.get('kind') == 'native-and-reference-wake-rest-transport-repair'":
            require(len(node.body)==2 and isinstance(node.body[0],ast.ImportFrom)
                and node.body[0].module=='pokezero.mcts_eval.wider_wake_rest_transport_recovery'
                and ast.unparse(node.body[1])=='validate_resumed_workers(measured.worker_receipts, probe)',
                'undeclared behavior in Rest tail resume guard')
            return node.orelse
        return node


def bind_source(registered,current,*,repo,git,sha):
    origin=Path(registered['source_root'])
    require(registered['source_commit']==QUALIFIED and git('rev-parse','HEAD',root=origin)==QUALIFIED
        and not git('status','--porcelain',root=origin),'fresh qualification source must remain pinned and clean')
    allowed={'scripts/wider_search_comparison.py','src/pokezero/mcts_eval/wider_wake_rest_transport_recovery.py',
        'third_party/foul-play'}
    names=set()
    for root in (origin,repo):
        names.update(git('ls-files','src','scripts','rust/pokezero-search','third_party','pyproject.toml',root=root).splitlines())
    inputs={}
    for name in sorted(names-allowed):
        require((origin/name).is_file() and (repo/name).is_file() and sha(origin/name)==sha(repo/name),
            'unqualified scientific source change: '+name)
        inputs[str(origin/name)]=sha(origin/name)
    functions=[]
    for root in (origin,repo):
        tree=ast.parse((root/'scripts/wider_search_comparison.py').read_text())
        functions.append({n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)})
    require(functions[0].keys()==functions[1].keys(),'undeclared driver helper')
    for name in functions[0].keys()-{'verify','register','validate_qualification','main','readout'}:
        candidate=functions[1][name]
        if name=='run':
            candidate=RemoveWakeRestTransportGuard().visit(candidate)
        require(ast.dump(functions[0][name])==ast.dump(candidate),'qualified driver behavior drift: '+name)
    for key in CONTRACT_KEYS:
        if key not in ('seeds','registered_games'):
            require(current[key]==registered[key],'fresh qualification treatment drift: '+key)
    require(current['native_package']==registered['native_package'],'qualified native package drift')
    return inputs


def bind_proofs(audit_dir,*,sha):
    records,inputs={},{}
    for name,expected in PROOFS.items():
        require(len(expected)==64,'unfinished original20 proof cannot qualify recovery: '+name)
        path=Path(audit_dir)/name
        require(sha(path)==expected,'independent wake/Rest transport proof drift: '+name)
        record=json.loads(path.read_text())
        records[name]=record
        inputs[str(path)]=expected
        inputs[str(path.with_suffix('.py'))]=sha(path.with_suffix('.py'))
        for filename,digest in record['input_hashes'].items():
            require(sha(filename)==digest,'proof input drift: '+filename)
            require(filename not in inputs or inputs[filename]==digest,'conflicting proof input')
            inputs[filename]=digest
    probe=records['probe_wake_rest_transport_original20_r109.json']
    replay=records['validate_wake_rest_transport_witnesses_r110.json']
    accepted_stream=records['validate_wake_rest_transport_stream_r111.json']
    fingerprints=records['probe_wake_rest_transport_first_streams_r116.json']
    streams=records['validate_wake_rest_transport_first_streams_r117.json']
    require(all(r['source_commit']==QUALIFIED for r in (probe,replay,accepted_stream,fingerprints,streams))
        and probe['restored_decision_id']==398 and probe['measurement']['decision_id']==399
        and probe['action_played'] is False and probe['accepted_prefix_extended'] is False
        and probe['failed_statistics_or_rng_restored'] is False
        and probe['diagnostic_statistics_or_rng_restored'] is False and probe['strength_inference'] is False
        and probe['nominal_decision_seconds']==10. and probe['measurement']['result']['trajectories']==1
        and replay['status']=='ACCEPTED_WAKE_REST_TRANSPORT_WITNESSES_REPLAYED_WITH_ORIGINAL_ANCHOR_POLICY_RNG_AND_NEW_BACKUPS'
        and len(replay['verified'])==replay['accepted_worlds']==1 and replay['checked_cancellations']==20
        and accepted_stream['status']=='ACCEPTED_ORIGINAL_STREAM_ALL_REJECTED_PROPOSALS_REPLAYED_WITH_FULL_TRANSPORT'
        and len(accepted_stream['verified'])==1
        and accepted_stream['verified'][0]['replayed_chance_proposals']==1020,
        'original nominal10s NEW work and independently replayed accepted/rejected stream required')
    require(fingerprints['status']=='ALL20_ORIGINAL_FIRST_WORLD_FINGERPRINTS_NOT_SEARCH_OR_STRENGTH'
        and fingerprints['restored_decision_id']==398 and fingerprints['decision_id']==399
        and fingerprints['search_trajectories']==0 and fingerprints['search_statistics_restored'] is False
        and fingerprints['action_played'] is False and fingerprints['game_wall_allowance_changed'] is False
        and fingerprints['diagnostic_statistics_or_rng_restored'] is False
        and streams['status']=='ALL20_ORIGINAL_FIRST_STREAMS_AND_REJECTED_PROPOSALS_INDEPENDENTLY_REPLAYED'
        and len(streams['verified'])==20 and {r['worker'] for r in streams['verified']}==set(range(20))
        and streams['action_played'] is False and streams['study_resumed'] is False
        and streams['strength_inference'] is False,
        'all20 original first-stream fingerprints and legacy replay required')
    controls=records['test_wake_rest_transport_public_observer_r114.json']
    law=records['audit_wake_rest_transport_source_law_r112.json']
    require(controls['status']=='PASS' and controls['controls']==54
        and controls['observer_sha256']==PUBLIC_OBSERVER
        and law['status']=='FIFTH_STAGE_SOURCE_DEPENDENCY_AND_TRANSPORT_ARGUMENT_BOUND_NOT_EXHAUSTIVE'
        and law['compiled_closure']=='1ee403ea5274fe32bc30329ead814c4c117ff548f5f576242d7f0c3d108642b1',
        'independent v4/v5 mutation controls and explicit conditional source-law argument required')
    return records,inputs


def bind_fresh_audits(semantic_path,public_path,qualification_path,*,sha):
    semantic_path,public_path,qualification_path=map(Path,(semantic_path,public_path,qualification_path))
    registration=qualification_path.parent/'registration.json'
    m=json.loads(registration.read_text())
    expected={game_identity(s,seat,arm) for s in m['seeds'] for seat in SEATS for arm in ARMS}
    require(m['source_commit']==QUALIFIED and m['phase']=='QUALIFICATION_NOT_STRENGTH'
        and m['registered_games']==8 and len(expected)==8
        and m['seeds']==[study_seed(i,qualification=True) for i in range(2)]
        and m['reference']['staged_substitute_conditioning'] is True,'fresh disjoint all8 wake/Rest transport qualification required')
    inputs={str(registration):sha(registration),str(qualification_path):sha(qualification_path)}
    for path,name,observer_hash,status in (
        (semantic_path,'validate_guarded_qualification_games_r31.py',SEMANTIC_OBSERVER,VALID),
        (public_path,'validate_wake_rest_transport_qualification_public_r113.py',PUBLIC_OBSERVER,'PUBLIC_AND_SUBSTITUTE_CERTIFICATES_VALID')):
        audit=json.loads(path.read_text())
        observer=path.with_name(name)
        require(sha(observer)==observer_hash==audit['observer_sha256']
            and audit['source_commit']==QUALIFIED and audit['registration_sha256']==sha(registration)
            and audit['audited_complete_games']==len(audit['games'])==8
            and audit['complete_roster_valid'] is True and audit['strength_inference'] is False
            and {r['identity'] for r in audit['games']}==expected
            and all(r['status']==status for r in audit['games']),'complete independently replayed all8 qualification required')
        if path==public_path:
            require(audit['schema']=='pokezero.wider-search.wake-rest-transport-conditioning-audit.v6'
                and audit['full_replay_audit_sha256']==sha(semantic_path)
                and isinstance(audit.get('input_hashes'),dict),'distinct v4/v5 public audit/semantic binding drift')
        else:
            # The immutable R31 semantic observer predates an explicit input
            # inventory. Rebind all audited terminals and their complete step
            # inventories below; do not require a field it never emitted or
            # weaken the newer public observer's explicit input bindings.
            require(audit['schema']=='pokezero.wider-search.read-only-audit.v1'
                and audit['phase']=='QUALIFICATION_NOT_STRENGTH' and audit['registered_games']==8
                and audit['canonical_observer_basis_sha256']=='df7748a6c812a913cd5314128860c1a84ea166c4357315722304664508557b7e',
                'immutable semantic observer wire schema drift')
        inputs.update({str(path):sha(path),str(observer):observer_hash})
        for row in audit['games']:
            terminal=qualification_path.parent/(row['identity']+'.json')
            cell=json.loads(terminal.read_text())
            files=sorted((qualification_path.parent/row['identity']).iterdir())
            require(row['result_sha256']==sha(terminal) and cell['status']=='COMPLETE'
                and cell['registration_sha256']==sha(registration)
                and files and all(p.is_file() for p in files)
                and {p.name:sha(p) for p in files}==cell['step_hashes'],'audited all8 terminal/step binding drift')
            inputs.update({str(p):sha(p) for p in (terminal,*files)})
        for filename,digest in audit.get('input_hashes',{}).items():
            require(sha(filename)==digest,'independent all8 audit input drift: '+filename)
            inputs[filename]=digest
    return inputs


def validate_qualification(path,current,recovery,*,repo,git,sha,verify,bound_rows):
    path=Path(path)
    require(path.name=='READOUT.json' and str(path)==recovery['qualification_readout'],'canonical fresh qualification path required')
    registered=verify_historical(path.parent/'registration.json')
    inputs=bind_source(registered,current,repo=repo,git=git,sha=sha)
    rows=bound_rows(path.parent,registered)
    result=analyze(registered['seeds'],rows)
    result.update(source_commit=registered['source_commit'],input_hashes=registered['input_hashes'],
        phase=registered['phase'],statistically_supported_advantage=False,inferential_test_allowed=False,
        status='QUALIFICATION_COMPLETE_NOT_STRENGTH' if not result['missing_seed_clusters'] else 'QUALIFICATION_INCOMPLETE')
    require(result==json.loads(path.read_text()) and result['status']=='QUALIFICATION_COMPLETE_NOT_STRENGTH'
        and len(rows)==8 and all(r['status']=='COMPLETE' for r in rows),'complete fresh all8 readout required')
    inputs.update(bind_fresh_audits(recovery['qualification_semantic_audit'],
        recovery['qualification_conditioning_audit'],path,sha=sha))
    _,proofs=bind_proofs(recovery['staged_audit_dir'],sha=sha)
    inputs.update(proofs)
    return dict(mode='fresh all8 wake/Rest transport qualification and independent v4/v5 audits; unchanged treatment, operational recovery only',
        fresh_eight_game_qualification=True,qualified_source_commit=QUALIFIED,
        readout_sha256=sha(path),registration_sha256=sha(path.parent/'registration.json'),semantic_input_hashes=inputs)


def validate_prefix_work(witness):
    require(witness['result']['trajectories']>0
        and {r['worker'] for r in witness['worker_receipts']}==set(range(20)),'accepted prefix requires NEW work and original20')
    draws=trajectories=transitions=0
    clocks,updates=set(),set()
    for receipt in witness['worker_receipts']:
        batch,worlds=receipt['batch'],receipt['evidence']['draws']
        require(type(receipt['worker']) is int and 0<=receipt['worker']<20,'invalid prefix worker identity')
        require(type(receipt.get('sequence')) is int and receipt['sequence']>0
            and (receipt['worker'],receipt['sequence']) not in updates,
            'duplicate or invalid prefix worker update')
        updates.add((receipt['worker'],receipt['sequence']))
        require(len(worlds)==batch['world_draws'],'prefix draw count drift')
        accepted=0
        for world in worlds:
            if world['status']=='ROOT_VALIDATED':
                require(world['released'] is True,'unreleased accepted prefix world')
                accepted+=1
            else:
                clock=world.get('sampling_diagnostic',{})
                require(world['status']=='DEADLINE_CANCELLED' and batch['deadline_exhausted'] is True
                    and clock.get('schema')=='pokezero.world-sampling-deadline.v1'
                    and clock.get('accepted_world') is False and clock.get('backed_up') is False
                    and all(type(clock.get(k)) in (int,float) and math.isfinite(clock[k]) for k in ('checked_at','deadline_at'))
                    and clock['checked_at']>=clock['deadline_at']
                    and 'substitute_policy_conditioning' not in world and 'pending_policy_conditioning' not in world,
                    'prefix cancellation lacks clock/zero-backup evidence')
                clocks.add(clock['deadline_at'])
        require(0<=batch['trajectories']<=accepted and (accepted or batch['transitions']==0),
            'cancelled prefix work backed up')
        draws+=len(worlds)
        trajectories+=batch['trajectories']
        transitions+=batch['transitions']
    require(len(clocks)<=1 and (draws,trajectories,transitions)==tuple(witness['result'][k]
        for k in ('world_draws','trajectories','transitions')),'prefix aggregate/shared-clock drift')


def resume_capture(capture,old,previous,*,sha,restore_reference_checkpoint):
    require(capture['status']=='ACCEPTED398_PRESERVED_NEXT399_ALL20_CLOCK_CANCELLED'
        and capture['source_commit']==PARENT and capture['accepted_boundaries']==21
        and capture['last_accepted_decision_id']==398 and capture['failed_decision_id']==399
        and capture['failed_work_or_rng_restored'] is False and capture['failure_scored_as_loss'] is False
        and capture['strength_inference'] is False,'exact accepted398/refused399 capture required')
    names=[f'boundary-{i:03d}.json.gz' for i in range(21)]
    require(sorted(capture['accepted_prefix_steps'])==names,'all21 accepted prefixes required')
    ordinals,selections,elapsed,last=[0]*20,0,0.,None
    for boundary,name in enumerate(names):
        path=Path(capture['accepted_prefix_steps'][name])
        require(path.name==name and sha(path)==capture['accepted_prefix_hashes'][name],'accepted prefix hash drift')
        row=read_step(path)
        require(row['boundary']==boundary and row['chance_seed']==continuation_seed('wider-search:474486314:p1',0,boundary,'chance'),
            'accepted prefix boundary/chance domain drift')
        if 'p1' not in row['actions']:
            continue
        own=row['evidence']['p1']
        require(own['selector']=='paper_reference' and own['seed']==continuation_seed('wider-search:474486314:p1',0,boundary,'search'),
            'accepted prefix own selector/search domain drift')
        selections+=1
        elapsed+=own['elapsed_seconds']
        last=own['search_evidence']
        validate_prefix_work(last)
        for receipt in last['worker_receipts']:
            ordinals[receipt['worker']]+=len(receipt['evidence']['draws'])
    checkpoint=restore_reference_checkpoint(last['statistics_checkpoint'])
    require(last['decision_id']==398 and checkpoint.battle_id=='wider-search:'+IDENTITY
        and ordinals==capture['worker_ordinals']
        and capture['accepted_checkpoint']==capture['accepted_prefix_steps'][names[-1]]
        and capture['accepted_checkpoint_sha256']==sha(capture['accepted_checkpoint']),
        'only accepted checkpoint398 and original stream ordinals may be restored')
    inherited=old['repair_retention']['resume']
    require(all(capture['accepted_prefix_steps'].get(name)==path for name,path in inherited['prefix_steps'].items()),
        'historical accepted prefix replaced')
    directory,terminal=Path(previous)/IDENTITY,Path(previous)/(IDENTITY+'.json')
    born,stop=directory.stat().st_birthtime,terminal.stat().st_mtime
    envelope=capture['wall_clock_envelope']
    charge=stop-born+1.
    require(born==envelope['start_birthtime'] and stop==envelope['terminal_mtime']
        and envelope['slack_seconds']==1. and charge==envelope['charged_seconds']
        and math.isfinite(charge) and 0<charge<120,'actual failed399 wall envelope drift')
    charged=inherited['elapsed_before_resume']+charge
    require(capture['elapsed_before_attempt']==inherited['elapsed_before_resume']
        and capture['cumulative_game_seconds']==charged and elapsed<=charged<old['per_game_wall_seconds'],
        'original cumulative2400s allowance cannot reset')
    wall=dict(inherited['wall_clock_envelope'])
    wall['additional_attempts']=[*wall.get('additional_attempts',[]),dict(directory=str(directory),terminal=str(terminal),
        directory_birth=born,terminal_mtime=stop,margin_seconds=1.,charged_seconds=charge,
        accepted_boundaries_added=4,failed_decision_id=399,failed_work_or_rng_restored=False)]
    return dict(inherited,result_path=str(terminal),canonical_terminal_is_truncated=False,
        checkpoint_step=capture['accepted_checkpoint'],prefix_steps=capture['accepted_prefix_steps'],
        start_boundary=21,prior_selections=selections,worker_ordinals=ordinals,decision_id=398,
        elapsed_before_resume=charged,wall_clock_envelope=wall,
        recovery='all21 accepted prefixes/checkpoint398 and original20 streams retained; failed399 and all diagnostics never restored')


def prepare(previous,audit_dir,qualification_path,semantic_path,public_path,current,*,
            repo,git,sha,verify,bound_rows,step_files,restore_reference_checkpoint):
    previous,audit_dir=Path(previous),Path(audit_dir)
    registration=previous/'registration.json'
    old=verify_historical(registration)
    retained=old['repair_retention']
    require(old['source_commit']==PARENT and old['phase']=='FIXED_64_SEED_CONFIRMATION'
        and old['seeds']==[study_seed(i) for i in range(64)] and old['registered_games']==256
        and retained['kind']=='native-and-reference-prefix-joint-tail-repair'
        and len(retained['retained_complete'])==32 and len(set(retained['uncertain_seed_clusters']))==9,
        'preserved full64/256 historical inventory required')
    for key in CONTRACT_KEYS:
        require(current[key]==old[key],'original study contract drift: '+key)
    records,inputs=bind_proofs(audit_dir,sha=sha)
    capture=records['capture_prefix_joint_tail_next_root_failure_r101.json']
    require(capture['registration']==str(registration) and capture['registration_sha256']==sha(registration)
        and capture['uncertain_seed_clusters']==retained['uncertain_seed_clusters'],
        'captured previous registration/all9 uncertainty roster drift')
    rows=bound_rows(previous,old)
    require(len(rows)==33 and len({r['identity'] for r in rows})==33
        and {r['identity'] for r in rows if r['status']=='COMPLETE'}==set(retained['retained_complete'])
        and [r['identity'] for r in rows if r['status']!='COMPLETE']==[IDENTITY],
        'retain all32 complete and exact one refused399, never drop/redraw')
    original=verify_historical(retained['original_registration'])
    audited=validate_prior_audit(json.loads(Path(retained['historical_semantic_audit']).read_text()),
        original,sha(retained['original_registration']))
    require(set(audited)==set(retained['retained_complete']),'every original complete game must retain semantic audit')
    for bindings in (old['input_hashes'],old['retained_input_hashes']):
        for filename,digest in bindings.items():
            require(sha(filename)==digest,'historical source/evidence drift: '+filename)
            inputs[filename]=digest
    for identity,binding in retained['retained_complete'].items():
        path=Path(binding['path'])
        m=json.loads(Path(binding['registration']).read_text())
        cell=json.loads(path.read_text())
        files=combined_steps(path.parent,cell,m,sha=sha,step_files=step_files)
        require(cell['status']=='COMPLETE' and binding['sha256']==sha(path)==audited[identity]['result_sha256']
            and {p.name:sha(p) for p in files}==cell['step_hashes'],'retained complete semantic/hash inventory drift')
        inputs.update({str(p):sha(p) for p in (path,Path(binding['registration']),*files)})
    probe=records['probe_wake_rest_transport_first_streams_r116.json']
    require(probe['worker_ordinals']==capture['worker_ordinals'],'original first-stream proof must bind accepted398')
    result=dict(retained,kind=KIND,qualification_readout=str(qualification_path),
        qualification_semantic_audit=str(semantic_path),qualification_conditioning_audit=str(public_path),
        qualified_source_commit=QUALIFIED,audited_repair_source_commit=QUALIFIED,
        audited_repair_source_root=str(json.loads((Path(qualification_path).parent/'registration.json').read_text())['source_root']),
        staged_audit_dir=str(audit_dir),exact_failed_draw_certificate=str(audit_dir/'probe_wake_rest_transport_first_streams_r116.json'),
        previous_failed_registration=str(registration),fresh_eight_game_qualification=True,
        wake_rest_transport_certificate='pokezero.constant-chance-wake-rest-prefix-joint-tail.v5',
        historical_prefix_joint_tail_capture=str(audit_dir/'capture_prefix_joint_tail_next_root_failure_r101.json'),
        disclosure=retained['disclosure']+'; fresh disjoint all8 wake/Rest transport qualification, separate v4/v5 source-law/public/semantic audits; '
            'all32 historical COMPLETE games and21 accepted boundaries/checkpoint398 retained; original20 streams and cumulative wall preserved; '
            'failed399/diagnostic Q/N/RNG never restored; mixed sources and all9 outcome-blind sensitivity clusters remain; '
            'deterministic first-world fingerprint observer has no search clock and is NOT longer-budget search; '
            'actual resumed search remains nominal10s; unsupported programs stay joint; deadline selection effects not proven away')
    q=validate_qualification(qualification_path,current,result,repo=repo,git=git,sha=sha,verify=verify,bound_rows=bound_rows)
    inputs.update(q['semantic_input_hashes'])
    inputs[str(registration)]=sha(registration)
    result['resume']=resume_capture(capture,old,previous,sha=sha,restore_reference_checkpoint=restore_reference_checkpoint)
    result['input_hashes']=inputs
    return result
