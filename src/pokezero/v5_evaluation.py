"""Explicit frozen-snapshot panel connector; import/inspection executes nothing.

The external declaration supplies every seed, artifact, runtime and CPU. Saved
data must be genuine fixed heldout public positions and actual action siblings.
Hashes bind those inputs; they do not attest the upstream collector's truth.
No legacy A1/A2/A4 fit, capture or panel entrypoint is used here.
"""
import argparse
import asyncio
from dataclasses import asdict,dataclass,fields,replace
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace
import uuid

from . import v5_run as run

SCHEMA='pokezero.v5.milestone_evaluation.v1'
POSITIONS='pokezero.v5.fixed_outcome_positions.v1'
SIBLINGS='pokezero.v5.actual_fixed_siblings.v1'
HOST_REQUEST_FIELDS=('progress_callback','prior_completions')


def _keys(value,names):
    if type(value) is not dict or set(value)!=set(names):raise ValueError('complete explicit evaluation fields required')


@dataclass(frozen=True)
class Artifact:
    path:str
    sha256:str

    def __post_init__(self):
        if type(self.path) is not str or not Path(self.path).is_absolute():raise ValueError('absolute original artifact path required')
        run.c._binding(self.sha256)

    def verify(self):
        from .v5_worker import _file_sha
        if _file_sha(self.path)!=self.sha256:raise ValueError('original evaluation artifact digest differs')
        return Path(self.path)

    def document(self):
        blob=Path(self.path).read_bytes()
        if hashlib.sha256(blob).hexdigest()!=self.sha256:raise ValueError('original evaluation document digest differs')
        return json.loads(blob)


def _config(document):
    from .foulplay_bridge import ControlledFoulPlayConfig
    _keys(document,(f.name for f in fields(ControlledFoulPlayConfig)))
    values=dict(document)
    for field in fields(ControlledFoulPlayConfig):
        value=values[field.name]
        if value is not None and 'Path' in str(field.type):values[field.name]=Path(value)
        elif value is not None and 'tuple' in str(field.type):values[field.name]=tuple(value)
    return ControlledFoulPlayConfig(**values)


def _require_foulplay_source(protocol, roots):
    """Bind the dependency's original PATCHED tracked worktree, not a clean clone.

    The dependency build commit is independent of the training/evaluator commit.
    Its caller-bound component receipt supplies the actual diff and binary bytes;
    only the current original FP/native build inputs must remain identical.
    """
    component=Artifact(**protocol['foulplay_source']).document()
    commit=component.get('source_commit')
    if (component.get('status')!='ORIGINAL_FOULPLAY_COMPONENT_BUILD_COMPLETE_NOT_SELECTED'
            or type(commit) is not str or len(commit)!=40 or any(c not in '0123456789abcdef' for c in commit)
            or component.get('foulplay_gitlink')!=protocol['foulplay_commit']
            or component.get('native_patch_count')!=100):
        raise ValueError('original digest-bound FoulPlay component source required')
    runtime=component['runtime'];python=runtime['python_executable']
    if python['path']!=protocol['foulplay_python']['path'] or python['sha256']!=protocol['foulplay_python']['sha256']:
        raise ValueError('FoulPlay component interpreter differs from protocol')
    for name in ('python_executable','pyvenv_config','native'):
        record=runtime[name];Artifact(record['path'],record['sha256']).verify()
    source=Path(__file__).resolve().parents[2]
    manifest=source/'third_party/poke-engine-gen3-patches.txt'
    patches=[line.strip() for line in manifest.read_text().splitlines() if line.strip() and not line.lstrip().startswith('#')]
    if len(patches)!=100 or hashlib.sha256(manifest.read_bytes()).hexdigest()!=component['native_patch_manifest_sha256']:
        raise ValueError('original FoulPlay native manifest differs')
    names=('.gitmodules','scripts/setup_foulplay_eval.sh','scripts/setup_poke_engine.sh',
        'scripts/verify_foulplay_poke_engine.py','scripts/verify_poke_engine_source.py',
        'scripts/apply_poke_engine_patches.py','third_party/poke-engine-base-source.json',
        'third_party/poke-engine-gen3-patches.txt',
        *(f'third_party/{name}' for name in patches),
        *(f'third_party/foulplay-{name}.patch' for name in ('local-nosec','rest-cant','capture')))
    for name in names:
        if hashlib.sha256((source/name).read_bytes()).hexdigest()!=component['input_sha256'].get(name):
            raise ValueError('original FoulPlay build input differs: '+name)
    expected=Path(python['path']).parents[2]/'source/third_party/foul-play'
    actual={Path(root).resolve(strict=True) for root in roots}
    if actual!={expected.resolve(strict=True)}:
        raise ValueError('FoulPlay worktree differs from selected component runtime')
    root=next(iter(actual))
    if subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()!=protocol['foulplay_commit']:
        raise ValueError('original declared FoulPlay source differs')
    difference=subprocess.check_output(['git','-C',str(root),'diff','--binary','--no-ext-diff','--no-textconv','HEAD','--'])
    if hashlib.sha256(difference).hexdigest()!=component['foulplay_applied_diff_sha256']:
        raise ValueError('original patched FoulPlay tracked worktree differs')


def _pairs(protocol,*,parse_configs,checkpoint=None):
    panels={}
    for entry in protocol['panels']:
        _keys(entry,('declared_games','seeds','configs'))
        count=entry['declared_games'];seeds=entry['seeds']
        if any(type(x) is not dict for x in entry['configs']):raise ValueError('literal complete panel configs required')
        required=('checkpoint','games','search_time_ms','foulplay_mcts_iterations','policy_mode','opponent_policy_mode',
            'encoder_kind','allow_search_fallback','belief_set_source','opponent_crash_retries','live_continuation_smoke',
            'live_continuation_oracle','belief_start_overrides','value_checkpoint','foulplay_cpu_affinity','seed_start',
            'pokezero_player','showdown_root','node_binary','foulplay_python','foulplay_root','owning_decision_timeout_seconds')
        if any(not set(required)<=set(x) for x in entry['configs']):raise ValueError('complete fixed panel controls required')
        configs=tuple(SimpleNamespace(**x) for x in entry['configs'])
        if (type(count) is not int or count<=0 or count%500000 or count in panels
                or type(seeds) is not list or len(seeds)!=500 or any(type(x) is not int or not 0<=x<2**32 for x in seeds)
                or len(set(seeds))!=500 or len(configs)!=2 or tuple(c.pokezero_player for c in configs)!=('p1','p2')):
            raise ValueError('exact500 declared seeds ×both seats at each500k milestone required')
        for config in configs:
            if (config.checkpoint is not None or config.games!=500 or config.search_time_ms!=1000
                    or config.foulplay_mcts_iterations is not None or config.policy_mode!='raw'
                    or config.opponent_policy_mode!='foul-play' or config.encoder_kind!='jax-owning'
                    or config.allow_search_fallback or config.belief_set_source is not True
                    or config.opponent_crash_retries!=0 or config.live_continuation_smoke or config.live_continuation_oracle
                    or config.belief_start_overrides or config.value_checkpoint is not None
                    or config.foulplay_cpu_affinity is not None
                    or config.owning_decision_timeout_seconds!=60.
                    or seeds!=list(range(config.seed_start,config.seed_start+500))):
                raise ValueError('original raw owning FP1000ms finite panel without fallback/retry required')
        if parse_configs:
            if checkpoint is None:raise ValueError('actual published checkpoint required to bind panel templates')
            configs=tuple(_config(dict(x,checkpoint=str(checkpoint))) for x in entry['configs'])
            from .foulplay_bridge import _per_seed_foulplay_random_seed_schedule_for_offsets
            for config in configs:_per_seed_foulplay_random_seed_schedule_for_offsets(config,offsets=tuple(range(500)))
        panels[count]=configs
    if not panels or sorted(panels)!=list(range(500000,max(panels)+1,500000)):
        raise ValueError('complete exact500k milestone declaration required')
    return panels


def _protocol(reference,*,parse_configs=False,checkpoint=None):
    value=reference.document()
    _keys(value,('schema','model_sha256','label_vocabulary_sha256','runtime','worker_python','foulplay_python',
        'foulplay_commit','foulplay_source','positions','siblings','panels','actor_cpus','foulplay_cpus','watchdog','bins','batch_size','deterministic'))
    if value['schema']!=SCHEMA:raise ValueError('explicit v5 evaluation protocol required')
    if value['deterministic'] is not True:raise ValueError('explicit common raw greedy panel policy required')
    for key in ('model_sha256','label_vocabulary_sha256'):run.c._binding(value[key])
    if type(value['foulplay_commit']) is not str or len(value['foulplay_commit'])!=40 or any(x not in '0123456789abcdef' for x in value['foulplay_commit']):
        raise ValueError('original FoulPlay source revision required')
    for key in ('runtime','worker_python','foulplay_python','foulplay_source','positions','siblings'):Artifact(**value[key]).verify()
    runtime=Artifact(**value['runtime']).document()
    from .v5_worker import SCHEMA as runtime_schema,_memory
    if runtime.get('schema')!=runtime_schema:raise ValueError('original explicit worker runtime required')
    _memory(runtime['memory_policy'],apply=False)
    if runtime['precision_policy']['inference']!='bfloat16':raise ValueError('declared original bf16 inference required')
    actor=value['actor_cpus'];fp=value['foulplay_cpus']
    if (type(actor) is not list or not actor or type(fp) is not list or not 1<=len(fp)<=4
            or any(type(cpu) is not int or cpu<0 for cpu in actor+fp) or len(set(actor+fp))!=len(actor+fp)):
        raise ValueError('disjoint declared actor CPUs and one tofour dedicated FP cores required')
    run.Watchdog(**value['watchdog'])
    if any(type(value[k]) is not int or value[k]<=0 for k in ('bins','batch_size')):
        raise ValueError('explicit finite mechanism bins/batch required')
    panels=_pairs(value,parse_configs=parse_configs,checkpoint=checkpoint)
    for configs in panels.values():
        for config in configs:
            if (Path(config.showdown_root)!=Path(runtime['showdown']['root'])
                    or Path(config.node_binary)!=Path(runtime['showdown']['node']['path'])
                    or config.foulplay_python is None or Path(config.foulplay_python)!=Path(value['foulplay_python']['path'])):
                raise ValueError('original panel runtime paths differ')
            if not Path(config.foulplay_root).is_dir() or not Path(config.showdown_root).is_dir():
                raise run.MissingDependency('actual declared FoulPlay and Showdown roots required before panel')
    # Inspect headers only here. Actual record validation is streamed before
    # model loading in the child, and all source artifacts are rechecked there.
    vocabulary=Artifact(**runtime['category_vocabulary']).document()
    category_sha=run._sha(dict(tokens=vocabulary['tokens'],oov_buckets=vocabulary['oov_buckets'],aliases=vocabulary['aliases']))
    from . import v5_device_evaluation_bank as device_bank
    device_headers=[]
    for key,schema,device_schema in (('positions',POSITIONS,device_bank.POSITIONS),('siblings',SIBLINGS,device_bank.SIBLINGS)):
        with Path(value[key]['path']).open() as stream:header=json.loads(next(stream))
        base_header={'schema','split','rows','category_vocabulary_sha256','label_vocabulary_sha256'}
        device = header['schema']==device_schema
        device_headers.append(device)
        if set(header) not in (base_header,base_header|{'capture_bank','capture_source'},base_header|{'device_bank','device_source','category_count'}):
            raise ValueError('complete fixed mechanism header required')
        if (header['schema'] not in (schema,device_schema) or header['split']!='PHASE_C_FIXED_HELDOUT'
                or type(header['rows']) is not int or header['rows']<=0
                or header['label_vocabulary_sha256']!=value['label_vocabulary_sha256']
                or header['category_vocabulary_sha256']!=category_sha):
            raise run.MissingDependency('fixed actual heldout positions and full sibling data required before FP')
    if device_headers[0]!=device_headers[1]:raise ValueError('cannot mix legacy and device mechanism banks')
    return value,panels


def _records(reference,schema):
    with Path(reference.path).open('rb') as stream:
        before=os.fstat(stream.fileno());digest=hashlib.sha256();first=next(stream);digest.update(first)
        header=json.loads(first);seen=0
        if header['schema']!=schema:raise ValueError('fixed mechanism artifact schema differs')
        for line in stream:
            digest.update(line)
            if not line.strip():raise ValueError('literal ordered mechanism records required')
            seen+=1;yield header,json.loads(line)
        if seen!=header['rows']:raise ValueError('full fixed mechanism denominator differs')
        after=os.fstat(stream.fileno());identity=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)
        if identity(before)!=identity(after) or digest.hexdigest()!=reference.sha256:
            raise ValueError('original mechanism bytes changed while reading')


def _position(document,header):
    from .v5_data import row_from_dict
    row=row_from_dict(document)
    if (row.privileged is not None or row.observation.view!='PUBLIC'
            or row.observation.category_vocabulary_sha256!=header['category_vocabulary_sha256']
            or row.observation.label_vocabulary.sha256!=header['label_vocabulary_sha256']):
        raise ValueError('same-namespace original public-only heldout observation required')
    return row


def _identity(row):return (row.battle_id,row.seed,row.observation.address.stream,row.observation.address.frame,row.observation.address.owner,row.source_frame_sha256)


def _mechanism_data(protocol):
    """Unit record-structure parser, not attestation of actual capture origin."""
    positions={};known=0
    for header,document in _records(Artifact(**protocol['positions']),POSITIONS):
        row=_position(document,header);key=_identity(row)
        if key in positions:raise ValueError('duplicate fixed heldout position')
        positions[key]=row
        if row.labels.terminal_status=='COMPLETE' and row.labels.outcome is not None and row.terminal_source_sha256 is not None:known+=1
    groups=[]
    for header,group in _records(Artifact(**protocol['siblings']),SIBLINGS):
        _keys(group,('root_identity','world_id','chance_id','opponent_action','selection','source_receipt_sha256','branches'))
        key=tuple(group['root_identity']);root=positions.get(key)
        if (root is None or group['selection']!='ALL_LEGAL_ACTIONS_BEFORE_VALUES'
                or any(type(group[k]) is not str or not group[k] for k in ('world_id','chance_id'))
                or type(group['opponent_action']) is not int or not 0<=group['opponent_action']<9):
            raise ValueError('actual same heldout root/world/chance/reply sibling group required')
        run.c._binding(group['source_receipt_sha256'])
        legal=[i for i,ok in enumerate(root.observation.legal_action_mask) if ok];seen=set();children=[]
        for branch in group['branches']:
            _keys(branch,('action','root_identity','world_id','chance_id','opponent_action','terminal','terminal_outcome',
                'terminal_source_sha256','decision','child_state_sha256'))
            action=branch['action'];run.c._binding(branch['child_state_sha256'])
            if (type(action) is not int or action not in legal or action in seen
                    or any(branch[k]!=group[k] for k in ('root_identity','world_id','chance_id','opponent_action'))
                    or type(branch['terminal']) is not bool):raise ValueError('sibling actual parent/context/action differs')
            seen.add(action)
            if branch['terminal']:
                if (branch['decision'] is not None or type(branch['terminal_outcome']) not in (int,float)
                        or branch['terminal_outcome'] not in (-1,0,1)):
                    raise ValueError('literal uncapped sibling terminal, no invented successor required')
                run.c._binding(branch['terminal_source_sha256']);child=None
            else:
                if branch['terminal_outcome'] is not None or branch['terminal_source_sha256'] is not None:
                    raise ValueError('nonterminal sibling cannot claim a final outcome')
                child=_position(branch['decision'],header);a,b=root.observation.address,child.observation.address
                if (child.battle_id!=root.battle_id or child.seed!=root.seed or b.stream!=a.stream or b.owner!=a.owner or b.frame!=a.frame+1):
                    raise ValueError('actual successor must retain original root owner and stream')
            children.append((action,child))
        if len(legal)<2 or seen!=set(legal):raise run.MissingDependency('complete actual legal-action siblings required before FP')
        groups.append((group,tuple(sorted(children))))
    keys=[(tuple(g['root_identity']),g['world_id'],g['chance_id'],g['opponent_action']) for g,_ in groups]
    if len(set(keys))!=len(keys) or not known or not any(all(row is not None for _,row in children) for _,children in groups):
        raise run.MissingDependency('actual outcome labels and nonterminal full siblings required before FP')
    return positions,groups


def _columns(rows,config):
    import numpy as np
    if any(row.observation.view!='PUBLIC' or row.observation.category_vocabulary_sha256!=config.category_vocabulary_sha256
            or row.observation.label_vocabulary.sha256!=config.label_vocabulary.sha256 for row in rows):
        raise ValueError('original observation namespaces must match the actual frozen head')
    value=dict(schema=run.c.TRAINING_CONTRACT_SCHEMA,observation_schema=run.c.OBSERVATION_SCHEMA_VERSION_V5,
        category_vocabulary_sha256=config.category_vocabulary_sha256,label_vocabulary_sha256=config.label_vocabulary.sha256,view='PUBLIC')
    for name,dtype in (('categorical_ids','int32'),('numeric_features','float32'),('token_type_ids','int32'),('attention_mask','bool')):
        value[name]=np.asarray([[row.observation.model_inputs()[name]] for row in rows],dtype=dtype)
    value['history_mask']=np.ones((len(rows),1),bool)
    return value


def _mechanism_arithmetic(protocol,factory,data):
    """Unit arithmetic only. Production requires the original capture resolver."""
    import numpy as np
    from gpu_engine import v5_model
    from gpu_engine.v5_live_protocol import check_factory
    from .value_precision_probe import calibration_metrics
    positions,groups=data
    def predict(rows):
        values=[]
        for start in range(0,len(rows),protocol['batch_size']):
            check_factory(factory);batch=rows[start:start+protocol['batch_size']]
            result=factory.forward(v5_model.prepare_inputs(_columns(batch,factory.config),factory.config))
            if not np.asarray(result.valid).all():raise ValueError('heldout actual public rows refused by original v5 head')
            chunk=np.asarray(result.value,dtype=np.float64)
            if chunk.shape!=(len(batch),) or not np.isfinite(chunk).all():raise ValueError('literal finite current-head values required')
            values.extend(map(float,chunk))
        return values
    known=[row for row in positions.values() if row.labels.terminal_status=='COMPLETE' and row.labels.outcome is not None]
    metrics=calibration_metrics(predict(known),[r.labels.outcome for r in known],bins=protocol['bins']);rmse=math.sqrt(metrics['mse'])
    gaps=[];terminal=0
    for group,children in groups:
        if any(row is None for _,row in children):terminal+=1;continue
        values=predict([row for _,row in children]);order=sorted(range(len(values)),key=lambda i:(-values[i],children[i][0]));a,b=order[:2]
        gap=values[a]-values[b]
        gaps.append(dict(root_identity=group['root_identity'],world_id=group['world_id'],chance_id=group['chance_id'],
            opponent_action=group['opponent_action'],actions=[a for a,_ in children],values=values,
            top_two_actions=[children[a][0],children[b][0]],gap=gap,gap_exceeds_outcome_rmse=gap>rmse))
    return dict(calibration=metrics,heldout_positions=len(positions),known_outcomes=len(known),unknown_outcomes=len(positions)-len(known),
        outcome_rmse=rmse,sibling_groups=len(groups),terminal_groups_excluded=terminal,scored_sibling_groups=len(gaps),
        top_two_rows=gaps,share_gap_exceeds_outcome_rmse=sum(g['gap_exceeds_outcome_rmse'] for g in gaps)/len(gaps),
        interpretation='Descriptive gap versus external outcome RMSE; residual variance includes outcome randomness and model error.',
        actual_ranking_accuracy='UNMEASURED_NO_MATCHED_TERMINAL_CONTINUATIONS',fitted_models=0)


def _capture_resolver(protocol=None):
    """Fixed original producer integration; never a caller readiness boolean.

    The fixed resolver requires the original nontraining collector/branch
    records. A numeric fixture or hexadecimal tag alone cannot replace them.
    """
    if protocol is not None:
        from . import v5_device_evaluation_bank as device_bank
        with Path(protocol['positions']['path']).open() as stream:header=json.loads(next(stream))
        if header['schema']==device_bank.POSITIONS:
            return SimpleNamespace(require_original_capture=device_bank.require_device_bank,
                resolve_original_capture=device_bank.resolve_device_bank,evaluate_original_capture=device_bank.evaluate_device_bank)
    name=__package__+'.v5_evaluation_capture'
    try:module=importlib.import_module(name)
    except ModuleNotFoundError as error:
        if error.name!=name:raise
        raise run.MissingDependency('genuine original heldout/sibling capture resolver is not integrated') from error
    if any(not callable(getattr(module,name,None)) for name in
            ('require_original_capture','resolve_original_capture','evaluate_original_capture')):
        raise run.MissingDependency('complete original capture producer/resolver API required')
    return module


def mechanism(protocol,factory,data):
    """Production entry cannot turn hand-authored record metadata into claims."""
    return _capture_resolver(protocol).evaluate_original_capture(protocol,factory,data)


def _atomic(path,value):
    path=Path(path);temporary=path.with_name('.'+path.name+'.pending')
    with temporary.open('w') as stream:
        stream.write(run._json(value));stream.flush();os.fsync(stream.fileno())
    temporary.replace(path)


def _fixed_rows(configs):
    from .foulplay_bridge import _per_seed_foulplay_random_seed_schedule_for_offsets
    schedules=[_per_seed_foulplay_random_seed_schedule_for_offsets(c,offsets=tuple(range(500))) for c in configs]
    return [dict(cell_ordinal=2*i+side,offset=i,seat=('p1','p2')[side],seed=configs[side].seed_start+i,
        fp_seed=schedules[side][i],status='NOT_ATTEMPTED',score=None,duration_seconds=None,source_result_sha256=None,
        source_game=None,reason=None,invocation_id=None) for i in range(500) for side in range(2)]


def _completed_receipts(rows,expected,configs,*,protocol_path=None):
    from . import v5_foulplay as fp
    if type(rows) is not list or len(rows)!=1000:raise ValueError('full original1000cell denominator required')
    ended=False;completed=[]
    for target,row in zip(expected,rows):
        if any(row[k]!=target[k] for k in ('cell_ordinal','offset','seed','seat','fp_seed')):
            raise ValueError('fixed original panel cell order differs')
        if row['status'] not in ('NOT_ATTEMPTED','RUNNING','COMPLETE','CAPPED','REFUSED','FAILED'):
            raise ValueError('literal original FP disposition required')
        if row['status']=='NOT_ATTEMPTED':ended=True
        elif ended:raise ValueError('panel skipped an earlier not-attempted cell')
        if row['status']!='COMPLETE':
            if row['score'] is not None:raise ValueError('unknown panel cell cannot carry a score')
            continue
        config=configs[row['cell_ordinal']%2]
        receipt=run.EvaluationCompletion(row['cell_ordinal'],row['offset'],row['seed'],row['seat'],row['duration_seconds'],
            run._json(row['source_game']),row['source_result_sha256'],config.pokezero_username,config.foulplay_username,protocol_path,row['fp_seed'])
        run._checked_evaluation_completion(receipt)
        if row['score']!=fp._score(config,row['source_game']):raise ValueError('actual source terminal score differs')
        completed.append(receipt)
    return completed


def _lost_inflight(rows):
    """A dispatched cell is never relaunched, even if its result was lost."""
    rows=json.loads(run._json(rows))
    for row in rows:
        if row['status']=='RUNNING':
            row.update(status='FAILED',reason='interrupted-inflight-result-unknown-not-replayed')
    return rows


def _panel_predecessor(panel,request_document,protocol_reference,configs,*,prior_completions=()):
    """Read the declared attempt chain; retain all prior non-dispatchable rows."""
    attempts=sorted(panel.glob('attempt-*'));previous=None;reference=None
    expected=_fixed_rows(configs)
    for index,directory in enumerate(attempts):
        if directory.name!=f'attempt-{index:06d}' or not directory.is_dir():
            raise ValueError('original ordered panel attempts required')
        if not (directory/'request.json').exists():
            if any(directory.iterdir()):raise ValueError('unbound panel attempt contains data')
            continue  # Parent interruption before request publication: no dispatch.
        payload=json.loads((directory/'request.json').read_bytes())
        if (payload['request']!=request_document or payload['protocol']!=asdict(protocol_reference)
                or payload['expected_run_declaration_sha256']!=request_document['declaration_sha256']
                or payload['directory']!=str(directory)
                or payload.get('resume_progress')!=(asdict(reference)if reference else None)):
            raise ValueError('original panel continuation binding differs')
        progress=directory/'progress.json'
        report=directory/'report.json'
        saved=None
        if report.exists():
            saved=json.loads(report.read_bytes())
            if (saved.get('schema')!=SCHEMA or saved.get('namespace')!=run._sha(payload)
                    or saved.get('request')!=request_document):
                raise ValueError('original immutable panel report binding differs')
        if not progress.exists():
            if saved is not None and saved['rows']!=_lost_inflight(previous if previous is not None else expected):
                raise ValueError('panel report records dispatch without its progress ledger')
            continue  # No dispatch is possible before this publication.
        ref=Artifact(str(progress),run._file_sha(progress));document=ref.document()
        if document['namespace']!=run._sha(payload):raise ValueError('original panel progress namespace differs')
        rows=document['rows'];_completed_receipts(rows,expected,configs)
        if saved is not None and saved['rows']!=rows:
            raise ValueError('panel progress changed from its immutable report')
        if previous is not None:
            for old,current in zip(_lost_inflight(previous),rows):
                if old['status']!='NOT_ATTEMPTED' and current!=old:
                    raise ValueError('previously attempted panel cell changed or replayed')
        previous=rows;reference=ref
    carried=_lost_inflight(previous if previous is not None else expected)
    actual={r.cell_ordinal:r for r in _completed_receipts(carried,expected,configs,protocol_path=protocol_reference.path)}
    for receipt in prior_completions:
        run._checked_evaluation_completion(receipt)
        if actual.get(receipt.cell_ordinal)!=receipt:
            raise ValueError('panel progress lost or changed a durable run-journal completion')
    return len(attempts),reference,carried


async def _concurrent_panel(configs,factory,protocol,progress_path,namespace,*,initial_rows=None):
    from . import v5_foulplay as fp
    expected=_fixed_rows(configs)
    rows=expected if initial_rows is None else _lost_inflight(initial_rows)
    _completed_receipts(rows,expected,configs)
    invocation=uuid.uuid4().hex;next_cell=next((i for i,row in enumerate(rows)if row['status']=='NOT_ATTEMPTED'),len(rows));stop=False
    active={};slots=len(protocol['foulplay_cpus'])
    def publish():
        _atomic(progress_path,dict(schema=fp.SCHEMA,namespace=namespace,invocation_id=invocation,rows=rows))
    publish()
    async def cell(index,slot):
        nonlocal stop
        expected=rows[index];config=replace(configs[index%2],foulplay_cpu_affinity=protocol['foulplay_cpus'][slot])
        def update(value):
            actual=value['rows'][0]
            if any(actual[k]!=expected[k] for k in ('offset','seed','fp_seed')):raise ValueError('actual original FP cell differs')
            if actual['status']=='NOT_ATTEMPTED':return  # Parent's durable dispatch is already RUNNING.
            expected.update(actual,cell_ordinal=index,seat=config.pokezero_player,invocation_id=invocation,
                source_panel_namespace=value.get('namespace'),source_invocation_id=value.get('invocation_id'))
            publish()
        result=await fp.run_panel(config,factory,game_offsets=(expected['offset'],),progress_callback=update,enable_v5=True)
        update(result)
        if str(expected.get('reason','')).startswith('TimeoutError:'):stop=True
    try:
        while next_cell<len(rows) or active:
            while next_cell<len(rows) and not stop:
                slot=next_cell%slots
                if slot in active:break  # Fixed ordinal→core, never whichever outcome frees first.
                index=next_cell;next_cell+=1
                rows[index].update(status='RUNNING',invocation_id=invocation);publish()
                active[slot]=asyncio.create_task(cell(index,slot))
            if not active:break
            done,_=await asyncio.wait(tuple(active.values()),return_when=asyncio.FIRST_COMPLETED)
            for slot,task in list(active.items()):
                if task in done:
                    del active[slot];task.result()
    finally:
        for task in active.values():task.cancel()
        if active:await asyncio.gather(*active.values(),return_exceptions=True)
    return json.loads(Path(progress_path).read_bytes())


def _factory(protocol,request):
    """Original public catalog for THIS archive; never load a champion model."""
    from . import v5_worker as worker
    from .v5_phasec import category_binding
    runtime=Artifact(**protocol['runtime']).document();worker._source(runtime)
    worker._probe(runtime)
    from .dex import load_showdown_dex_cached
    from .randbat import load_gen3_randbat_source_cached
    from .category_vocab import CategoryVocabulary
    from .live_protocol_actor import ActorContract,_vocab_sha
    from gpu_engine import v4_encoder,public_belief
    from gpu_engine.v5_live_protocol import load_factory
    script=Path(__file__).resolve().parents[2]/'gpu_engine'/'gender_policy.cjs'
    code="const p=require(process.argv[1]); console.log(JSON.stringify(p.compiledTree(process.argv[2])));"
    compiled=json.loads(subprocess.check_output([runtime['showdown']['node']['path'],'-e',code,str(script),
        str(Path(runtime['showdown']['root'])/'dist')],text=True))
    if compiled['sha256']!=runtime['showdown']['compiled_tree_sha256']:raise ValueError('original compiled Showdown tree differs')
    vocabulary=Artifact(**runtime['category_vocabulary']).document();original=ActorContract.champion()
    v=CategoryVocabulary(tuple(vocabulary['tokens']),vocabulary['oov_buckets'],vocabulary['aliases'])
    if _vocab_sha(v)!=original.category_vocab_sha256:raise ValueError('original categorical row order required')
    source=load_gen3_randbat_source_cached(runtime['showdown']['root'])
    if source.metadata.source_hash!=original.belief_set_source_hash:raise ValueError('original public source differs')
    contract=replace(original,checkpoint_sha256=request.snapshot.checkpoint_sha256,model_config_sha256=request.snapshot.model_sha256)
    catalog=v4_encoder.compile_encoder(public_belief.compile_source(source),load_showdown_dex_cached(runtime['showdown']['root']),v,
        checkpoint_sha256=contract.checkpoint_sha256,feature_masks=dict(contract.feature_masks))
    factory=load_factory(request.archive_path,catalog=catalog,contract=contract,
        expected_archive_sha256=request.snapshot.archive_sha256,expected_checkpoint_sha256=request.snapshot.checkpoint_sha256,
        expected_category_vocabulary_sha256=category_binding(v),
        expected_label_vocabulary_sha256=protocol['label_vocabulary_sha256'],compute_dtype='bfloat16',learner_seconds=60.,
        deterministic=protocol['deterministic'],enable_v5=True)
    if run._sha(factory.config.metadata())!=protocol['model_sha256']:raise ValueError('exact frozen evaluation model configuration differs')
    return factory


def _request_document(request):
    return {key:(asdict(request.snapshot) if key=='snapshot' else str(getattr(request,key)) if key.endswith('_path') or key=='run_directory'
        else getattr(request,key)) for key in request.__dataclass_fields__ if key not in HOST_REQUEST_FIELDS}


def _pin_actor(protocol):
    actor=protocol['actor_cpus'];fp=protocol['foulplay_cpus']
    if (type(actor) is not list or not actor or type(fp) is not list or not 1<=len(fp)<=4
            or any(type(cpu) is not int or cpu<0 for cpu in actor+fp) or len(set(actor+fp))!=len(actor+fp)
            or not set(actor+fp)<=set(os.sched_getaffinity(0))):
        raise ValueError('declared dedicated CPUs absent or invalid')
    # Imported NumPy/Torch can already have native threads. Restrict every
    # current actor thread, not just the main TID; future threads inherit this.
    threads=Path('/proc/self/task')
    for task in threads.iterdir():
        try:os.sched_setaffinity(int(task.name),set(actor))
        except ProcessLookupError:pass  # An exited thread cannot consume a core.
    for task in threads.iterdir():
        try:
            if set(os.sched_getaffinity(int(task.name)))!=set(actor):raise ValueError('actor thread can consume a dedicated FP core')
        except ProcessLookupError:pass


def _restore_request(value):
    _keys(value,(k for k in run.EvaluationRequest.__dataclass_fields__ if k not in HOST_REQUEST_FIELDS))
    return run.EvaluationRequest(**dict(value,snapshot=run.Snapshot(**value['snapshot']),
        **{key:Path(value[key]) for key in ('run_directory','checkpoint_path','archive_path')}))


def _check_request(request,protocol,reference,expected_run_sha256):
    if (type(request) is not run.EvaluationRequest or type(request.snapshot) is not run.Snapshot or request.protocol_sha256!=reference.sha256
            or request.declaration_sha256!=expected_run_sha256 or request.snapshot.model_sha256!=protocol['model_sha256']):
        raise ValueError('original frozen milestone snapshot/protocol/run binding required')
    declaration=json.loads((request.run_directory/'declaration.json').read_bytes())
    if (run._sha(declaration)!=expected_run_sha256 or declaration['evaluation']['protocol_sha256']!=reference.sha256
            or run._sha(declaration['model'])!=request.snapshot.model_sha256):
        raise ValueError('original accepted run declaration differs')
    for path,digest in ((request.checkpoint_path,request.snapshot.checkpoint_sha256),(request.archive_path,request.snapshot.archive_sha256)):
        if not Path(path).resolve().is_relative_to(request.run_directory.resolve()) or run._file_sha(path)!=digest:
            raise ValueError('original owned frozen evaluation input differs')
    if (request.checkpoint_path!=(request.run_directory/request.snapshot.checkpoint).resolve()
            or request.archive_path!=(request.run_directory/request.snapshot.archive).resolve()):
        raise ValueError('exact published Snapshot path required')


@dataclass(frozen=True)
class EvaluationHook:
    protocol:Artifact
    expected_run_declaration_sha256:str  # External AFTER plan freeze; avoids circular protocol/run hash.

    def __post_init__(self):
        if type(self.protocol) is not Artifact:raise ValueError('original digest-bound protocol artifact required')
        run.c._binding(self.expected_run_declaration_sha256);protocol,_=_protocol(self.protocol)
        _capture_resolver(protocol).require_original_capture(protocol)

    def __call__(self,request):
        from . import v5_foulplay as fp
        protocol,panels=_protocol(self.protocol,parse_configs=True,checkpoint=request.checkpoint_path);_check_request(request,protocol,self.protocol,self.expected_run_declaration_sha256)
        _capture_resolver(protocol).require_original_capture(protocol)
        _capture_resolver(protocol).resolve_original_capture(protocol,request)
        if request.declared_games not in panels:raise run.MissingDependency('exact declared milestone panel missing')
        if not callable(request.progress_callback):raise run.MissingDependency('original durable run completion sink required before panel')
        document=_request_document(request)
        panel=request.run_directory/'artifacts'/('evaluation-'+str(request.declared_games)+'-'+run._sha(document))
        panel.mkdir(parents=True,exist_ok=True)
        configs=panels[request.declared_games];expected=_fixed_rows(configs)
        attempt,predecessor,carried=_panel_predecessor(panel,document,self.protocol,configs,prior_completions=request.prior_completions)
        directory=panel/f'attempt-{attempt:06d}'
        directory.mkdir(parents=True,exist_ok=False)
        payload=dict(protocol=asdict(self.protocol),expected_run_declaration_sha256=self.expected_run_declaration_sha256,
            request=document,directory=str(directory),resume_progress=asdict(predecessor)if predecessor else None)
        child=directory/'request.json';run._publish(child,run._json(payload).encode());namespace=run._sha(payload)
        progress=directory/'progress.json';seen={}
        def drain():
            if not progress.is_file():return
            document=json.loads(progress.read_bytes())
            if document['namespace']!=namespace or len(document['rows'])!=1000:raise ValueError('original full panel progress binding differs')
            current={receipt.cell_ordinal:receipt for receipt in _completed_receipts(document['rows'],expected,configs,protocol_path=self.protocol.path)}
            if any(current.get(index)!=receipt for index,receipt in seen.items()):
                raise ValueError('completed panel source disappeared or mutated')
            for receipt in current.values():
                if receipt.cell_ordinal in seen:
                    if seen[receipt.cell_ordinal]!=receipt:raise ValueError('completed panel source mutated')
                else:
                    if not callable(request.progress_callback):raise ValueError('original run completion sink required')
                    request.progress_callback(receipt);seen[receipt.cell_ordinal]=receipt
        def poll(seconds):drain();time.sleep(seconds)
        wd=run.Watchdog(**protocol['watchdog'])
        result=fp.supervise_panel([protocol['worker_python']['path'],'-m','pokezero.v5_evaluation',
            '--request',str(child),'--sha256',run._file_sha(child)],progress_path=progress,
            initial_window_seconds=wd.initial_window_seconds,longest_multiplier=wd.longest_multiplier,
            backstop_seconds=wd.backstop_seconds,poll_seconds=wd.poll_seconds,grace_seconds=wd.terminate_grace_seconds,sleep=poll)
        drain();report=directory/'report.json';worker_report=directory/'worker-report.json'
        worker_digest=None
        if not worker_report.is_file():
            value=dict(schema=SCHEMA,status='UNKNOWN',reason='owned-panel-ended-without-final-report',namespace=namespace,
                request=payload['request'],rows=json.loads(progress.read_bytes())['rows'] if progress.is_file() else carried,
                scientific_qualification=False)
        else:
            blob=worker_report.read_bytes();worker_digest=hashlib.sha256(blob).hexdigest();value=json.loads(blob)
        if value['namespace']!=namespace or value['request']!=payload['request']:raise ValueError('final snapshot report binding differs')
        if value.get('schema')!=SCHEMA:raise ValueError('original evaluation report schema differs')
        if value['status']=='RESULT' and (not progress.is_file() or value['rows']!=json.loads(progress.read_bytes())['rows']):
            raise ValueError('final actual cell evidence differs from original progress')
        _check_request(request,protocol,self.protocol,self.expected_run_declaration_sha256)
        final={receipt.cell_ordinal:receipt for receipt in _completed_receipts(value['rows'],expected,configs,protocol_path=self.protocol.path)}
        if any(final.get(index)!=receipt for index,receipt in seen.items()):
            raise ValueError('completed source disappeared or mutated in final report')
        scores=[row['score'] for row in value['rows'] if row['status']=='COMPLETE'];unknown=1000-len(scores)
        value=dict(value,known_completed=len(scores),unknown_cells=unknown,
            completed_only_score=sum(scores)/len(scores) if scores else None,
            score_bounds=[sum(scores)/1000,(sum(scores)+unknown)/1000],
            disposition_counts={status:sum(row['status']==status for row in value['rows']) for status in
                ('NOT_ATTEMPTED','RUNNING','COMPLETE','CAPPED','REFUSED','FAILED')})
        value=dict(value,supervisor=result,worker_report_sha256=worker_digest)
        if result['exit_code']!=0 or result['stop_reason']:
            value.update(status='UNKNOWN',reason=value.get('reason') or 'owned-panel-unknown')
        run._publish(report,run._json(value).encode())
        relative=str(report.relative_to(request.run_directory));digest=run._file_sha(report)
        if result['exit_code']!=0 or result['stop_reason'] or value['status']!='RESULT':
            return run.EvaluationOutcome('UNKNOWN',relative,digest,reason=value.get('reason') or 'owned-panel-unknown')
        return run.EvaluationOutcome('RESULT',relative,digest)


def child(reference):
    """Fixed owned worker entry only; never called by preparation or import."""
    payload=reference.document();request=_restore_request(payload['request']);directory=Path(payload['directory'])
    # The FP bridge imports Torch. Bind CPU/memory BEFORE parsing its typed
    # configs, not merely before the later factory/JAX compiler call.
    unparsed=Artifact(**payload['protocol']).document()
    from .v5_worker import _memory
    runtime=Artifact(**unparsed['runtime']).document()
    _pin_actor(unparsed)
    _memory(runtime['memory_policy'],apply=True)
    protocol,panels=_protocol(Artifact(**payload['protocol']),parse_configs=True,checkpoint=request.checkpoint_path);namespace=run._sha(payload)
    _check_request(request,protocol,Artifact(**payload['protocol']),payload['expected_run_declaration_sha256'])
    panel=request.run_directory/'artifacts'/('evaluation-'+str(request.declared_games)+'-'+run._sha(payload['request']))
    if (directory.parent!=panel or Path(reference.path)!=directory/'request.json'
            or not directory.name.startswith('attempt-')):
        raise ValueError('original owned evaluation request directory required')
    initial_rows=None
    if payload.get('resume_progress') is not None:
        prior=Artifact(**payload['resume_progress'])
        if (Path(prior.path).parent.parent!=panel or Path(prior.path).name!='progress.json'
                or Path(prior.path).parent.name>=directory.name):
            raise ValueError('earlier original panel progress required')
        initial_rows=prior.document()['rows']
    from .owning_worker_runtime import configure_compilation_cache
    configure_compilation_cache(request.run_directory/'jax-compilation-cache')
    configs=tuple(replace(c,checkpoint=request.checkpoint_path) for c in panels[request.declared_games]);report=directory/'worker-report.json'
    try:
        # No actual panel before source/data/runtime checks and genuine metric inputs.
        from .v5_worker import _source
        runtime=Artifact(**protocol['runtime']).document();_source(runtime)
        _require_foulplay_source(protocol,tuple(config.foulplay_root for config in configs))
        captured=_capture_resolver(protocol).resolve_original_capture(protocol,request);factory=_factory(protocol,request)
        mechanisms=mechanism(protocol,factory,captured)
        progress=asyncio.run(_concurrent_panel(configs,factory,protocol,directory/'progress.json',namespace,initial_rows=initial_rows))
        rows=progress['rows'];known=[r['score'] for r in rows if r['status']=='COMPLETE'];unknown=1000-len(known)
        partial=any(r['status'] in ('RUNNING','NOT_ATTEMPTED') for r in rows)
        value=dict(schema=SCHEMA,status='UNKNOWN' if partial else 'RESULT',reason='unattempted/inflight-original-cells-remain' if partial else None,
            namespace=namespace,request=payload['request'],rows=rows,known_completed=len(known),unknown_cells=unknown,
            completed_only_score=sum(known)/len(known) if known else None,score_bounds=[sum(known)/1000,(sum(known)+unknown)/1000],
            mechanisms=mechanisms,scientific_qualification=False,physics_qualified=False,strength_qualified=False)
    except Exception as error:
        value=dict(schema=SCHEMA,status='UNKNOWN',reason=type(error).__name__+': '+str(error),namespace=namespace,request=payload['request'],
            rows=json.loads((directory/'progress.json').read_bytes())['rows'] if (directory/'progress.json').is_file() else _lost_inflight(initial_rows if initial_rows is not None else _fixed_rows(configs)),
            scientific_qualification=False)
    run._publish(report,run._json(value).encode());return 0 if value['status']=='RESULT' else 1


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--request',required=True);parser.add_argument('--sha256',required=True)
    args=parser.parse_args(argv);return child(Artifact(args.request,args.sha256))


if __name__=='__main__':raise SystemExit(main())
