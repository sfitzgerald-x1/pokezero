"""Paired DepthN / our-RAW Foul Play panel on Codex's frozen ladder cells.

Codex's ladder declaration is a read-only, sha256-pinned input. Our cells use its
formula and must equal its RAW cells on (pair, offset, seat, seed, fp_seed).
Every cell is claimed once (write_new) before model or game work; an attempted
cell is never re-run, and resume dispatches only not-attempted cells. One game
at a time per dedicated worker process. A search arm is dispatched only after a
live server probe passes. Readouts are pinned to the run's code identity and use
our own estimator. Import starts nothing.

The searched-share breakage floor (50% of non-single-legal decisions) only
catches a server that refuses everything. The pre-registered expected searched
share (from the constructor-world gate on 40 Showdown roots, bound to that run id
and report sha256) must be set before any dispatch; readouts report observed vs
expected as a descriptive comparison that never changes a decision.

Eval readiness (the lead's rulings):
- Precision: the declaration pins matmul_precision 'highest' (the gates' certified
  precision); the probe fails, and the readout reports breakage, on any COMPLETE
  response whose pricer.matmul_precision differs.
- Deadline: the declaration's deadline (3, 5, 10 or 15 s) and reserve reach every
  search-arm client (worker and probe); the latency p95 breakage bound equals it.
- Screen latency rule (outcome-blind): once the screen's first 1,000 searched
  decisions (non-single-legal decisions of search-arm cells, cells in completion
  order, rows in decision order) exist, their timeout rate is evaluated exactly
  once and recorded in run state; above 1% the screen is VOID_LATENCY: dispatch
  stops, no screen statistic is computed, the readout reports the end-to-end wall
  distribution and each candidate deadline's coverage (wall + reserve), and the
  run is never resumed (a restart is a fresh run with a fresh declaration).
- Arm fairness: Foul Play's 1,000 ms is wall clock, so both arms run interleaved
  (pair by pair, RAW then search, neither arm more than the concurrency ahead),
  Foul Play keeps Codex's dedicated actor/FP core pair, each cell records its core
  pair, worker identity and load average, and Foul Play's per-move cost (MCTS
  iterations, else wait seconds, from the bridge's opponent_think rows) is compared
  by arm: more than 5% apart (relative to RAW, once each arm has 1,000 measured
  moves) flags FLAG_ARM_FAIRNESS and stops dispatch until the lead acknowledges.
"""
import asyncio
from collections import Counter
import hashlib
from itertools import islice
import json
import math
import os
from pathlib import Path
import socket
from statistics import fmean,variance
import subprocess
import sys
import time
from types import SimpleNamespace
import uuid

from . import phase_d_ladder_panel as ladder
from .depthn_live_actor import ARMS,CAUSES,CHAMPION_ARCHIVE_SHA256,REQUEST_KINDS,ROW_SCHEMA,SEARCH_ARMS,SOURCES

SCHEMA='depthn-foulplay-panel.declaration.v1'
RUN_SCHEMA='depthn-foulplay-panel.run.v1'
REQUEST_SCHEMA='depthn-foulplay-panel.request.v1'
CELL_SCHEMA='depthn-foulplay-panel.cell.v1'
READOUT_SCHEMA='depthn-foulplay-panel.readout.v1'
PROBE_SCHEMA='depthn-foulplay-panel.probe.v1'
WORLDS={'DN2W8':8,'DN2W16':16}
SCREEN_RULE='STOP_HARM_IF_UPPER_BOUND_LT_0'
BREAKAGE=dict(timeout_forfeits_max=0,unknown_cells_max=0,transport_failure_share_max=.02,searched_share_min=.5)
DEADLINES=(3.,5.,10.,15.)
LATENCY_RULE=dict(searched_decisions=1000,timeout_rate_max=.01,coverage_deadlines=[5.,10.,15.],coverage_min=.995)
INTERLEAVE='ALTERNATE_ARMS_BY_PAIR_ORDINAL_LEAD_AT_MOST_CONCURRENCY'
ARM_FAIRNESS=dict(metric='MEAN_FOULPLAY_ITERATIONS_PER_MOVE',fallback='MEAN_FOULPLAY_WAIT_SECONDS_PER_MOVE',reference='RAW',
                  relative_difference_max=.05,min_moves_per_arm=1000)
WORKER_IDENTITY_ENVIRONMENT=('POD_NAME','NODE_NAME')
TRANSPORT=('timeout','connection_error','server_error')
T_QUANTILE=.95  # Two-sided 90%; validate() pins confidence to .9.
KEYS={'schema','ladder_declaration_sha256','checkpoint_sha256','archive_sha256','arms','search','root_operator','deadline_seconds',
      'reserve_seconds','screen_paired_cells','max_paired_cells','confidence','primary_comparison','bar','screen_rule','breakage',
      'expected_searched_share','expected_searched_share_roster','matmul_precision','screen_latency_rule','bench_p95_wall_seconds',
      'bench_source','server_processes_per_gpu','interleave','arm_fairness'}
REQUEST_KEYS={'schema','run_directory','declaration','declaration_sha256','ladder_declaration','ladder_declaration_sha256',
              'cell','checkpoint','checkpoint_sha256','base_config'}
encoded,sha,read_bound,write_new=ladder.encoded,ladder.sha,ladder.read_bound,ladder.write_new


def _hex(value,size=64):return type(value)is str and len(value)==size and all(c in '0123456789abcdef'for c in value)


def validate(declaration):
    """Our pre-registered eval declaration: exact keys, so no paths or extras."""
    from .live_protocol_actor import ActorContract
    d=declaration
    if type(d)is not dict or set(d)!=KEYS:raise ValueError('exact DepthN panel declaration keys required')
    arms=d['arms']
    if (d['schema']!=SCHEMA or not _hex(d['ladder_declaration_sha256'])
            or d['checkpoint_sha256']!=ActorContract.champion().checkpoint_sha256 or d['archive_sha256']!=CHAMPION_ARCHIVE_SHA256
            or type(arms)is not list or arms!=['RAW']+[a for a in SEARCH_ARMS if a in arms]or len(arms)<2
            or d['search']!={a:dict(depth=2,worlds=WORLDS[a])for a in arms[1:]}
            or d['root_operator']!='ARGMAX_MEAN_COMPLETED_Q'or type(d['deadline_seconds'])is not float
            or d['deadline_seconds']not in DEADLINES or d['reserve_seconds']!=.15
            or d['screen_paired_cells']!=500 or d['max_paired_cells']!=2000 or d['confidence']!=.9
            or d['primary_comparison']!='SEARCH_MINUS_OUR_RAW_PAIRED_BY_CELL'
            or d['bar']!='LOWER_BOUND_GT_0_AT_2000_FULL_ROSTER_NO_BREAKAGE'or d['screen_rule']!=SCREEN_RULE
            or d['breakage']!=dict(BREAKAGE,latency_p95_max_seconds=d['deadline_seconds'])or d['matmul_precision']!='highest'
            or d['screen_latency_rule']!=LATENCY_RULE or d['interleave']!=INTERLEAVE or d['arm_fairness']!=ARM_FAIRNESS):
        raise ValueError('declared DepthN 500/2000 paired panel against our RAW required')
    expected=d['expected_searched_share']
    if expected is not None and (type(expected)is not dict or set(expected)!={'value','source'}
            or type(expected['value'])is not float or not 0.<=expected['value']<=1.
            or type(expected['source'])is not dict or set(expected['source'])!={'run_id','report_sha256'}
            or type(expected['source']['run_id'])is not str or not expected['source']['run_id'].strip()
            or not _hex(expected['source']['report_sha256'])):
        raise ValueError('expected searched share: null, or a value in [0, 1] bound to the gate run id and report sha256')
    roster=d['expected_searched_share_roster'];bench=d['bench_source'];p95=d['bench_p95_wall_seconds']
    if (roster is not None and(type(roster)is not float or not 0.<=roster<=1.)
            or type(p95)is not float or not math.isfinite(p95)or p95<=0
            or type(bench)is not dict or set(bench)!={'path','sha256'}or type(bench['path'])is not str or not bench['path'].strip()
            or not _hex(bench['sha256'])or type(d['server_processes_per_gpu'])is not int or d['server_processes_per_gpu']<1):
        raise ValueError('descriptive fields: roster share in [0, 1] or null, bench p95 > 0 with its summary path and sha256, '
                         'server processes per GPU >= 1')
    return d


def _pairing(cell):return tuple(cell[k]for k in ('pair_ordinal','offset','seat','seed','fp_seed'))


def cells(ladder_declaration,arm,*,limit=2000):
    """Codex's formula; asserted equal to Codex's RAW cells, so pairs join seed-for-seed."""
    ladder.validate(ladder_declaration)
    if arm not in ARMS or limit not in (500,2000):raise ValueError('declared arm and fixed500 screen or2000 prefix required')
    axis=ARMS.index(arm);d=ladder_declaration
    ours=tuple(dict(ordinal=len(ARMS)*pair+axis,pair_ordinal=pair,offset=pair//2,seat=pair%2,arm=arm,
        seed=d['battle_seed_start']+pair//2,fp_seed=d['foulplay_seed_start']+pair//2)for pair in range(limit))
    if list(map(_pairing,ours))!=list(map(_pairing,ladder.cells(d,'RAW',limit=limit))):
        raise ValueError('derived cells differ from the ladder RAW cells')
    return ours


def checked_cell(ladder_declaration,cell):
    if type(cell)is not dict or type(cell.get('pair_ordinal'))is not int or not 0<=cell['pair_ordinal']<2000:
        raise ValueError('declared panel cell required')
    if cell!=cells(ladder_declaration,cell.get('arm'))[cell['pair_ordinal']]:raise ValueError('panel cell address/seed/seat differs')
    return cell


def _git(*args):
    try:
        out=subprocess.run(['git','-C',str(Path(__file__).resolve().parent),*args],capture_output=True,text=True,timeout=30)
    except (OSError,subprocess.SubprocessError):return None
    return out.stdout if out.returncode==0 else None


def tree_identity(declared=None):
    """Commit, dirty flag and a digest of the worker code (package, gpu_engine, bridge scripts).

    Outside git, an explicit 40-hex commit is required and recorded as declared;
    the source digest is then the verified identity.
    """
    digest=code_digest(Path(__file__).resolve().parents[2])
    head=_git('rev-parse','HEAD');status=_git('status','--porcelain','--untracked-files=no')
    if head is not None and status is not None:
        commit=head.strip()
        if declared is not None and declared!=commit:raise ValueError('declared tree commit differs from git HEAD')
        return dict(commit=commit,dirty=bool(status.strip()),source_sha256=digest,commit_source='git')
    if not _hex(declared,40):raise ValueError('outside git an explicit 40-hex tree commit is required')
    return dict(commit=declared,dirty=None,source_sha256=digest,commit_source='declared')


CODE=(('src/pokezero','**/*.py'),('gpu_engine','**/*.py'),('scripts','*.mjs'))


def code_digest(repo):
    digest=hashlib.sha256()
    for base,pattern in CODE:
        for path in sorted((repo/base).glob(pattern)):
            if '__pycache__'not in path.parts:digest.update(str(path.relative_to(repo)).encode()+b'\0'+path.read_bytes())
    return digest.hexdigest()


def same_tree(a,b):
    """Same commit and code digest; the dirty flags must agree, and an unknown
    (outside git) flag matches only a clean checkout. Fail-closed otherwise."""
    if a['commit']!=b['commit']or a['source_sha256']!=b['source_sha256']:return False
    if a['dirty']is None or b['dirty']is None:return not (a['dirty']or b['dirty'])
    return a['dirty']==b['dirty']


def require_tree(run,declared=None):
    tree=tree_identity(declared)
    if not same_tree(tree,run['tree']):raise ValueError('code identity differs from the run manifest')
    return tree


def init(run_directory,*,declaration,declaration_sha256,ladder_declaration,ladder_declaration_sha256,checkpoint,base_config,
         base_config_sha256,tree_commit=None):
    """Bind one run directory to both pinned declarations, one base config and the code identity."""
    d=validate(read_bound(declaration,declaration_sha256));lad=ladder.validate(read_bound(ladder_declaration,ladder_declaration_sha256))
    if d['ladder_declaration_sha256']!=ladder_declaration_sha256 or lad['checkpoint_sha256']!=d['checkpoint_sha256']:
        raise ValueError('panel declaration must pin this ladder declaration and champion')
    base=read_bound(base_config,base_config_sha256)
    if type(base)is not dict:raise ValueError('literal base configuration required')
    root=Path(run_directory).resolve()
    run=dict(schema=RUN_SCHEMA,run_directory=str(root),declaration=str(Path(declaration).resolve()),declaration_sha256=declaration_sha256,
        ladder_declaration=str(Path(ladder_declaration).resolve()),ladder_declaration_sha256=ladder_declaration_sha256,
        checkpoint=str(Path(checkpoint).resolve()),checkpoint_sha256=d['checkpoint_sha256'],base_config=base,
        base_config_sha256=base_config_sha256,tree=tree_identity(tree_commit))
    write_new(root/'run.json',run)
    return run


def load(run_directory):
    root=Path(run_directory).resolve();run=json.loads((root/'run.json').read_bytes())
    if run.get('schema')!=RUN_SCHEMA or run.get('run_directory')!=str(root):raise ValueError('this panel run manifest required')
    d=validate(read_bound(run['declaration'],run['declaration_sha256']))
    lad=ladder.validate(read_bound(run['ladder_declaration'],run['ladder_declaration_sha256']))
    if d['ladder_declaration_sha256']!=run['ladder_declaration_sha256']:raise ValueError('panel declaration pins another ladder')
    return root,run,d,lad


def request(run,cell):
    return dict(schema=REQUEST_SCHEMA,run_directory=run['run_directory'],declaration=run['declaration'],
        declaration_sha256=run['declaration_sha256'],ladder_declaration=run['ladder_declaration'],
        ladder_declaration_sha256=run['ladder_declaration_sha256'],cell=cell,checkpoint=run['checkpoint'],
        checkpoint_sha256=run['checkpoint_sha256'],base_config=run['base_config'])


def pending(run_directory,*,arm,limit,resume=False,screen=None,tree_commit=None):
    """Packets for not-attempted cells; request files are immutable once written.

    Cells past the screen need a pinned 500-cell readout that continued, and
    recomputing that readout now must reproduce its bytes exactly.
    """
    root,run,d,lad=load(run_directory);require_tree(run,tree_commit)
    if arm not in d['arms']:raise ValueError('undeclared arm')
    if d['expected_searched_share']is None:raise ValueError('expected searched share from the constructor-world gate must be set before dispatch')
    refuse_by_state(root)
    if limit==d['max_paired_cells']:
        value=None if screen is None else read_bound(*screen)
        if (type(value)is not dict or value.get('schema')!=READOUT_SCHEMA or value.get('declaration_sha256')!=run['declaration_sha256']
                or value.get('paired_cells')!=d['screen_paired_cells']or value.get('decision')!='CONTINUE_TO_2000'
                or value.get('arm')!=arm and arm!='RAW'):
            raise ValueError('pinned 500-cell screen readout continuing to 2000 required')
        again=readout(root,arm=value['arm'],limit=d['screen_paired_cells'],codex_raw_directory=value['codex_raw_directory'],
            server_telemetry=value['server_telemetry'],tree_commit=tree_commit)
        if sha(again)!=screen[1]:raise ValueError('screen readout differs from its recomputation')
    todo=[];attempted=0
    for cell in cells(lad,arm,limit=limit):
        if (root/'claims'/f"{cell['ordinal']}.json").exists():attempted+=1;continue
        path=root/'requests'/f"{cell['ordinal']}.json";value=request(run,cell)
        if path.exists():
            if path.read_bytes()!=encoded(value):raise ValueError('existing cell request differs')
        else:write_new(path,value)
        todo.append(dict(path=str(path),sha256=sha(value),arm=arm,pair_ordinal=cell['pair_ordinal']))
    if attempted and not resume:raise ValueError('attempted cells exist; resume explicitly')
    return todo


def interleaved(run_directory,*,search_arm,limit,resume=False,screen=None,tree_commit=None):
    """Both arms' not-attempted cells in the declared interleave (pair by pair, RAW then the search arm) and each arm's
    attempted count (claims), which dispatch's lead rule starts from, so a resume keeps the interleave."""
    root,run,d,lad=load(run_directory)
    if search_arm not in d['arms'][1:]:raise ValueError('declared search arm required')
    packets=[]
    for arm in ('RAW',search_arm):packets+=pending(root,arm=arm,limit=limit,resume=resume,screen=screen,tree_commit=tree_commit)
    packets.sort(key=lambda p:(p['pair_ordinal'],p['arm']!='RAW'))
    attempted={arm:sum((root/'claims'/f"{c['ordinal']}.json").exists()for c in cells(lad,arm,limit=limit))for arm in ('RAW',search_arm)}
    return packets,attempted


def state_path(root,name):return Path(root)/'state'/name


def refuse_by_state(root):
    """A voided run is never resumed; a fairness flag holds dispatch until the lead acknowledges it."""
    rule=state_path(root,'latency_rule.json')
    if rule.exists()and json.loads(rule.read_bytes())['status']=='VOID_LATENCY':
        raise ValueError('VOID_LATENCY: a voided screen is never resumed; restart with a fresh run and declaration')
    if state_path(root,'arm_fairness_flag.json').exists()and not state_path(root,'arm_fairness_acknowledged.json').exists():
        raise ValueError("FLAG_ARM_FAIRNESS: dispatch waits for the lead's acknowledgement")


def acknowledge_arm_fairness(run_directory,*,note):
    """The lead's acknowledgement of a recorded fairness flag (written once); online fairness stops end for this run."""
    root=Path(run_directory).resolve();flag=state_path(root,'arm_fairness_flag.json')
    if not flag.exists():raise ValueError('no arm fairness flag to acknowledge')
    if type(note)is not str or not note.strip():raise ValueError('a note is required')
    value=dict(flag_sha256=hashlib.sha256(flag.read_bytes()).hexdigest(),note=note,acknowledged_unix_seconds=time.time())
    write_new(state_path(root,'arm_fairness_acknowledged.json'),value)
    return value


def probe_capture(run_directory,*,limit=500):
    """The first closed RAW cell's owning-wire capture in this run, if any."""
    root,run,d,lad=load(run_directory)
    for cell in cells(lad,'RAW',limit=limit):
        path=root/'owning-wire'/f"{cell['ordinal']}.jsonl"
        if (root/'results'/f"{cell['ordinal']}.json").exists()and path.exists():return path
    raise ValueError('no closed RAW capture in this run; pass a capture (the arms run interleaved)')


def probe(run_directory,*,arm,server,capture,roots=5,tree_commit=None):
    """Live search exchanges on captured roots, under fresh battle ids, before any dispatch.

    Pass: a COMPLETE response on the declared rung, W, root operator and matmul precision. Every
    server response must carry the declared champion archive_sha256. Any transport
    failure, server_error, bad_request or expired request fails at once; a
    constructor, DepthN or root-class refusal moves to the next root. The record
    is written once.
    """
    from .depthn_live_actor import Client,clock
    from .phase_d_owning_capture import read_capture
    root,run,d,lad=load(run_directory);require_tree(run,tree_commit)
    if arm not in d['arms'][1:]:raise ValueError('declared search arm required')
    client=Client(server['host'],server['port'],rung=arm,deadline_seconds=d['deadline_seconds'],reserve_seconds=d['reserve_seconds'])
    tried=[];status='FAIL';reason='no COMPLETE response on the first captured roots'
    for row in islice(read_capture(capture),roots):
        snapshot=SimpleNamespace(battle_id='probe-'+uuid.uuid4().hex,**{k:row[k]for k in ('owner','stream_id','generation','prefix_sha256','lines','own_request_line')})
        response=client.request(snapshot,decision_index=row['decision_index'],started=clock())
        tried.append({k:response.get(k)for k in ('request_id','status','cause','refusal','requested_worlds','root_operator',
            'tree_commit','archive_sha256','pricer','client_wall_seconds')})
        if response.get('client_refused')is False and response.get('archive_sha256')!=d['archive_sha256']:
            reason='server archive differs from the declared champion archive';break
        if response['status']=='COMPLETE':
            good=(response.get('requested_worlds')==d['search'][arm]['worlds']and response.get('root_operator')==d['root_operator']
                and (response.get('pricer')or {}).get('matmul_precision')==d['matmul_precision'])
            status,reason=('PASS',None)if good else('FAIL','COMPLETE response with another W, root operator or matmul precision');break
        if response['cause']not in ('constructor_refused','depthn_refused','class_refused'):reason='refused: '+str(response['cause']);break
    value=dict(schema=PROBE_SCHEMA,arm=arm,server=server,capture=str(Path(capture).resolve()),status=status,reason=reason,tried=tried)
    write_new(root/'probes'/f'{arm}-{time.time_ns()}.json',value)
    if status!='PASS':raise ValueError('search server probe failed: '+reason)
    return value


async def worker(packet,*,expected_pair,original,runtime_class=None):
    """Exactly one cell. All validation precedes the claim; the claim precedes any game work."""
    if type(packet)is not dict or set(packet)!={'path','sha256','server'}:raise ValueError('one panel cell packet required')
    path,digest,server=packet['path'],packet['sha256'],packet['server']
    value=read_bound(path,digest)
    if type(value)is not dict or set(value)!=REQUEST_KEYS or value['schema']!=REQUEST_SCHEMA:raise ValueError('literal panel request required')
    d=validate(read_bound(value['declaration'],value['declaration_sha256']))
    lad=ladder.validate(read_bound(value['ladder_declaration'],value['ladder_declaration_sha256']))
    cell=checked_cell(lad,value['cell']);root=Path(value['run_directory']).resolve()
    if (d['ladder_declaration_sha256']!=value['ladder_declaration_sha256']or cell['arm']not in d['arms']
            or value['checkpoint_sha256']!=lad['checkpoint_sha256']):raise ValueError('declared arm/ladder/champion required')
    if Path(path).resolve()!=root/'requests'/f"{cell['ordinal']}.json":raise ValueError('owned cell request path required')
    if (server is None)!=(cell['arm']=='RAW')or server is not None and (type(server)is not dict or set(server)!={'host','port'}):
        raise ValueError('RAW runs without a server; a search arm needs one')
    config=ladder.configuration(value['base_config'],cell,checkpoint=value['checkpoint'],cpu_pair=expected_pair)
    from .live_protocol_runtime import _checkpoint_sha
    if _checkpoint_sha(config.checkpoint)!=value['checkpoint_sha256']:raise ValueError('original champion bytes differ')
    from .depthn_live_actor import Client,Runtime
    client=None if server is None else Client(server['host'],server['port'],rung=cell['arm'],
        deadline_seconds=d['deadline_seconds'],reserve_seconds=d['reserve_seconds'])
    telemetry=root/'telemetry'/f"{cell['ordinal']}.jsonl"
    runtime=(Runtime if runtime_class is None else runtime_class)(client=client,telemetry_path=telemetry,arm=cell['arm'],cell=cell,
        original=original,archive_sha256=d['archive_sha256'])
    write_new(root/'claims'/f"{cell['ordinal']}.json",dict(request_sha256=digest,cell=cell,cpu_pair=list(expected_pair),
        server=server,pid=os.getpid(),claimed_unix_seconds=time.time()))
    started=time.monotonic();document=game=None;load=dict(start=list(os.getloadavg()))
    try:
        with ladder.owning_capture(root/'owning-wire',cell=cell,declaration_sha256=value['declaration_sha256']):result=await runtime.run(config)
        document=result.to_dict()
        if (result.config!=config or result.checkpoint_sha256!=value['checkpoint_sha256']
                or result.foulplay_random_seed_schedule!=(cell['fp_seed'],)or len(result.games)!=1):
            raise ValueError('literal one-cell original runtime result required')
        game=result.games[0].to_dict()
        headers=dict(checkpoint_sha256=value['checkpoint_sha256'],policy_mode='raw',seed_start=cell['seed'],
            pokezero_player=config.pokezero_player,encoder_kind=config.encoder_kind,capture_driver=config.capture_driver,format_id=config.format_id)
        if (game.get('seed')!=cell['seed']or document.get('game_results')!=[game]
                or any(document.get(k)!=v for k,v in headers.items())
                or document.get('foulplay_random_seed_schedule',{}).get('seeds')!=[cell['fp_seed']]):
            raise ValueError('cell/scored game/serialized benchmark join required')
        from .v5_foulplay import _score
        score=_score(config,game);outcome=dict(status='COMPLETE'if score is not None else'UNKNOWN',score=score,error=None)
    except Exception as error:outcome=dict(status='UNKNOWN',score=None,error=type(error).__name__+': '+str(error))
    outcome.update(schema=CELL_SCHEMA,request_sha256=digest,declaration_sha256=value['declaration_sha256'],
        ladder_declaration_sha256=value['ladder_declaration_sha256'],cell=cell,checkpoint_sha256=value['checkpoint_sha256'],
        cpu_pair=list(expected_pair),server=server,duration_seconds=time.monotonic()-started,source_game=game,
        benchmark=document,telemetry=str(telemetry),
        # The bridge's per-game result keeps no protocol log; the owner's literal wire is this capture.
        owning_wire=str(root/'owning-wire'/f"{cell['ordinal']}.jsonl"),worker=worker_identity(),load_average=dict(load,end=list(os.getloadavg())))
    write_new(root/'results'/f"{cell['ordinal']}.json",outcome)
    return 0


def worker_identity():
    """This worker's host and, when the environment provides them, its pod and node: run outputs only."""
    return dict(hostname=socket.gethostname(),**{k.lower():os.environ[k]for k in WORKER_IDENTITY_ENVIRONMENT if os.environ.get(k)})


async def serve(pair,cache_directory):
    """CPU-only worker: one game at a time, packets on stdin until EOF."""
    from .b4_canary_endpoint import pin_owned_actor
    pin_owned_actor(pair);os.environ['JAX_PLATFORMS']='cpu';os.environ['CUDA_VISIBLE_DEVICES']=''
    import contextlib,jax,torch
    jax.config.update('jax_enable_x64',False);jax.config.update('jax_default_matmul_precision','highest')
    from .owning_worker_runtime import ChampionWorkerRuntime,configure_compilation_cache
    configure_compilation_cache(cache_directory);torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision('highest');torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    original=ChampionWorkerRuntime();output=sys.stdout
    for line in sys.stdin:
        with contextlib.redirect_stdout(sys.stderr):code=await worker(json.loads(line),expected_pair=pair,original=original)
        output.write(json.dumps({'exit':code})+'\n');output.flush()


def worker_command(pair,cache_directory):
    return [sys.executable,'-m','pokezero.depthn_live_panel','--cpu-pair',str(pair[0]),str(pair[1]),'--cache-directory',str(cache_directory)]


async def dispatch(packets,*,pairs,servers,cells_per_worker=1,command=worker_command,cache_directory=None,attempted=None,after=None):
    """One worker process per core pair at a time, fresh after cells_per_worker cells.

    servers maps every arm to its server (None for RAW). Interleave: a cell is taken
    (in packet order) only if its arm stays at most len(pairs) cells ahead of every
    other arm, counting attempted cells; when no cell qualifies, dispatch stops
    ('interleave') rather than run one arm alone. after(packet), called once per
    completed cell, may stop dispatch by returning a reason. A worker that fails to
    report stops all dispatch: its cell stays claimed or unclaimed as the worker
    left it, and no further cell is attempted this run.
    """
    if (type(cells_per_worker)is not int or cells_per_worker<1 or not pairs
            or len({p for pair in pairs for p in pair})!=2*len(pairs)):raise ValueError('disjoint core pairs and positive cells per worker required')
    if type(servers)is not dict or not servers or any(p.get('arm')not in servers for p in packets):raise ValueError('a server entry for every arm required')
    queue=list(packets);done=[];failed=[];stopped=[];lead=len(pairs)
    started=Counter({arm:0 for arm in servers});started.update(attempted or {})
    def take():
        for i,packet in enumerate(queue):
            arm=packet['arm']
            if all(started[arm]+1-started[other]<=lead for other in servers if other!=arm):
                started[arm]+=1;return queue.pop(i)
        stopped.append(f'interleave: no arm may run more than {lead} cells ahead of another');return None
    async def lane(pair):
        while queue and not failed and not stopped:
            process=await asyncio.create_subprocess_exec(*command(pair,cache_directory),stdin=asyncio.subprocess.PIPE,stdout=asyncio.subprocess.PIPE)
            try:
                for _ in range(cells_per_worker):
                    if not queue or failed or stopped:break
                    packet=take()
                    if packet is None:break
                    wire=dict(path=packet['path'],sha256=packet['sha256'],server=servers[packet['arm']])
                    process.stdin.write((json.dumps(wire)+'\n').encode());await process.stdin.drain()
                    line=await process.stdout.readline()
                    try:ok=json.loads(line)=={'exit':0}
                    except ValueError:ok=False
                    (done if ok else failed).append(packet['path'])
                    if not ok:break
                    reason=None if after is None else after(packet)
                    if reason:stopped.append(reason);break
            finally:
                process.stdin.close();code=await process.wait()
            if code!=0:failed.append(f'worker exit {code}')
    await asyncio.gather(*(lane(pair)for pair in pairs))
    return dict(dispatched=len(done),failed=failed,not_dispatched=len(queue),stopped=stopped[0]if stopped else None)


def completions(root):
    path=state_path(root,'completions.jsonl')
    return [json.loads(x)for x in path.read_bytes().splitlines()]if path.exists()else []


def telemetry_rows(root,ordinal):
    path=Path(root)/'telemetry'/f'{ordinal}.jsonl'
    return [json.loads(x)for x in path.read_bytes().splitlines()]if path.exists()else []


def latency_rule(root,d,*,search_arm,log):
    """The screen latency rule on the first N searched decisions (non-single-legal decisions of search-arm screen cells,
    cells in completion order, rows in decision order): None until N exist; outcome-blind (timeout causes only)."""
    rule=d['screen_latency_rule'];n=rule['searched_decisions'];searched=[];used=[]
    for entry in log:
        if entry['arm']!=search_arm or entry['pair_ordinal']>=d['screen_paired_cells']:continue
        used.append(entry['ordinal'])
        searched+=[r for r in telemetry_rows(root,entry['ordinal'])if r.get('source')in ('search','greedy_fallback')]
        if len(searched)>=n:break
    if len(searched)<n:return None
    timeouts=sum(r.get('cause')=='timeout'for r in searched[:n]);rate=timeouts/n
    return dict(status='VOID_LATENCY'if rate>rule['timeout_rate_max']else'CONTINUE',searched_decisions=n,timeouts=timeouts,timeout_rate=rate,
        timeout_rate_max=rule['timeout_rate_max'],cells=used,scope='first searched decisions of the screen in completion order; timeout causes only')


def foulplay_cost(rows):
    """Foul Play's per-move cost in these cell results: the bridge's opponent_think rows (MCTS iterations; wait seconds)."""
    iterations=[];waits=[]
    for row in rows:
        for think in (row.get('source_game')or {}).get('opponent_think')or ():
            if type(think.get('iterations'))is int:iterations.append(think['iterations'])
            if type(think.get('wait_seconds'))in (int,float)and math.isfinite(think['wait_seconds']):waits.append(float(think['wait_seconds']))
    return dict(cells=len(rows),iteration_moves=len(iterations),mean_iterations=fmean(iterations)if iterations else None,
        wait_moves=len(waits),mean_wait_seconds=fmean(waits)if waits else None)


def arm_fairness(raw,search,rule):
    """Search vs RAW relative difference of Foul Play's mean per-move iterations (wait seconds when either arm lacks the
    iteration count), once both arms have the declared minimum of measured moves."""
    least=rule['min_moves_per_arm'];metric=a=b=None
    if raw['iteration_moves']>=least and search['iteration_moves']>=least:metric,a,b=rule['metric'],raw['mean_iterations'],search['mean_iterations']
    elif raw['wait_moves']>=least and search['wait_moves']>=least:metric,a,b=rule['fallback'],raw['mean_wait_seconds'],search['mean_wait_seconds']
    difference=None if metric is None or not a else abs(b-a)/a
    return dict(metric=metric,raw=raw,search=search,relative_difference=difference,relative_difference_max=rule['relative_difference_max'],
        flagged=difference is not None and difference>rule['relative_difference_max'],
        status='EVALUATED'if difference is not None else'INSUFFICIENT_MOVES')


class Monitor:
    """Run state after each completed cell: completion order, the screen latency rule (evaluated once, recorded) and arm
    fairness (a flag is recorded once and stops dispatch until acknowledged). Returns a stop reason or None."""
    def __init__(self,run_directory,*,search_arm):
        self.root,self.run,self.d,self.lad=load(run_directory);self.arm=search_arm;self.costs={}
        if search_arm not in self.d['arms'][1:]:raise ValueError('declared search arm required')
        known={entry['ordinal']for entry in completions(self.root)}
        # A cell whose result exists but never reached the log (dispatch stopped in between) joins it now, by ordinal.
        for arm in ('RAW',search_arm):
            for cell in cells(self.lad,arm):
                if cell['ordinal']not in known and (self.root/'results'/f"{cell['ordinal']}.json").exists():self._append(cell)

    def _append(self,cell):
        path=state_path(self.root,'completions.jsonl');path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('ab')as stream:
            stream.write(encoded(dict(ordinal=cell['ordinal'],arm=cell['arm'],pair_ordinal=cell['pair_ordinal']))+b'\n')
            stream.flush();os.fsync(stream.fileno())

    def latency(self):
        path=state_path(self.root,'latency_rule.json')
        if path.exists():return json.loads(path.read_bytes())
        value=latency_rule(self.root,self.d,search_arm=self.arm,log=completions(self.root))
        if value is not None:write_new(path,value)
        return value

    def fairness(self):
        rows={'RAW':[], self.arm:[]}
        for entry in completions(self.root):
            if entry['arm']in rows:
                if entry['ordinal']not in self.costs:
                    path=self.root/'results'/f"{entry['ordinal']}.json"
                    self.costs[entry['ordinal']]=json.loads(path.read_bytes())if path.exists()else {}
                rows[entry['arm']].append(self.costs[entry['ordinal']])
        return arm_fairness(foulplay_cost(rows['RAW']),foulplay_cost(rows[self.arm]),self.d['arm_fairness'])

    def __call__(self,packet):
        ordinal=int(Path(packet['path']).stem);arm=ARMS[ordinal%len(ARMS)]
        if arm!=packet['arm']:raise ValueError('completed packet arm differs from its cell')
        self._append(cells(self.lad,arm)[ordinal//len(ARMS)])
        rule=self.latency()
        if rule is not None and rule['status']=='VOID_LATENCY':return 'VOID_LATENCY'
        if state_path(self.root,'arm_fairness_acknowledged.json').exists():return None
        fair=self.fairness()
        if fair['flagged']:
            flag=state_path(self.root,'arm_fairness_flag.json')
            if not flag.exists():write_new(flag,fair)
            return 'FLAG_ARM_FAIRNESS'
        return None


def _quantiles(values):
    values=sorted(values)
    if not values:return None
    rank=lambda q:values[max(0,math.ceil(q*len(values))-1)]
    return dict(n=len(values),p50=rank(.5),p95=rank(.95),p99=rank(.99),max=values[-1])


def _share(a,b):return None if b==0 else a/b


def decisions(rows,*,deadline,worlds=None,archive=None,precision=None):
    """Shares over non-single-legal decisions: search, fallback by cause and kind, transport.

    Also latency quantiles, search-vs-greedy, and COMPLETE responses whose
    requested_worlds is not the arm's W, whose archive_sha256 is not the declared
    champion archive, or whose pricer.matmul_precision is not the pinned precision. class_refused is a counted fallback cause; it is
    neither a transport failure nor breakage.
    """
    seen=set();sources=Counter();causes=Counter();kinds={};differs=late=wrong_worlds=wrong_archive=wrong_precision=0
    times={k:[]for k in ('select','encode','greedy','client','server_wall','queue','prepare','build','price','reduce')}
    provenance={k:Counter()for k in ('tree_commit','archive_sha256','root_operator','pricer')}
    for row in rows:
        key=(row.get('battle_id'),row.get('owner'),row.get('decision_index'),row.get('prefix_sha256'))
        source,cause,kind=row.get('source'),row.get('cause'),row.get('request_kind')
        if (row.get('schema')!=ROW_SCHEMA or key in seen or source not in SOURCES or kind not in REQUEST_KINDS
                or (cause is None)!=(source!='greedy_fallback')or cause is not None and cause not in CAUSES
                or (row.get('search')is None)!=(source in ('raw_arm','single_legal'))
                or (source=='single_legal')!=(kind=='forced'and source!='raw_arm')):
            raise ValueError('literal unique decision rows required')
        seen.add(key);sources[source]+=1
        k=kinds.setdefault(kind,dict(decisions=0,searched=0,fallback=0,differs=0,causes=Counter()));k['decisions']+=1
        if source=='search':
            k['searched']+=1;hit=row['search_differs']is True;k['differs']+=hit;differs+=hit
        elif source=='greedy_fallback':k['fallback']+=1;k['causes'][cause]+=1;causes[cause]+=1
        times['select'].append(row['select_seconds']);late+=row['select_seconds']>deadline
        times['encode'].append(row['encode_seconds']);times['greedy'].append(row['greedy']['seconds']);search=row['search']
        if search is None:continue
        if search.get('client_wall_seconds')is not None:times['client'].append(search['client_wall_seconds'])
        if search.get('server_wall_seconds')is not None:times['server_wall'].append(search['server_wall_seconds'])
        for name,value in (search.get('timings')or {}).items():
            if name.endswith('_seconds')and name[:-8]in times and value is not None:times[name[:-8]].append(value)
        if search['status']=='COMPLETE':
            wrong_worlds+=worlds is not None and search.get('requested_worlds')!=worlds
            wrong_archive+=archive is not None and search.get('archive_sha256')!=archive
            wrong_precision+=precision is not None and (search.get('pricer')if isinstance(search.get('pricer'),dict)else {}).get('matmul_precision')!=precision
            for name in provenance:
                value=search.get(name);provenance[name][value if type(value)is not dict else encoded(value).decode()]+=1
    n=len(seen);searched=sources['search'];eligible=n-sources['single_legal']
    return dict(decisions=n,eligible_decisions=eligible,by_source=dict(sources),searched_share=_share(searched,eligible),
        fallback_share=_share(sources['greedy_fallback'],eligible),
        fallback_by_cause={c:dict(count=v,share=v/eligible)for c,v in sorted(causes.items())},
        transport_failure_share=_share(sum(causes[c]for c in TRANSPORT),eligible),
        by_request_kind={kind:dict(decisions=v['decisions'],searched=v['searched'],fallback=v['fallback'],
            searched_share=_share(v['searched'],v['decisions']),fallback_share=_share(v['fallback'],v['decisions']),
            fallback_by_cause=dict(sorted(v['causes'].items())),search_differs=v['differs'],
            search_differs_share=_share(v['differs'],v['searched']))for kind,v in sorted(kinds.items())},
        search_differs=differs,search_differs_share_of_searched=_share(differs,searched),search_differs_share_of_all=_share(differs,n),
        wrong_worlds=wrong_worlds,wrong_archive=wrong_archive,wrong_precision=wrong_precision,latency={k:_quantiles(v)for k,v in times.items()},
        late_decisions=late,deadline_seconds=deadline,
        provenance={k:sorted(map(str,v))for k,v in provenance.items()})


def _timeout_forfeit(row):
    error=row.get('error')
    return row['status']!='COMPLETE'and type(error)is str and ('TimeoutError'in error or 'deadline'in error)


def arm_summary(root,lad,*,arm,limit,declaration_sha256,deadline,worlds=None,archive=None,precision=None):
    rows=[];telemetry=[];mismatch=[];lost=unattempted=0
    for cell in cells(lad,arm,limit=limit):
        path=root/'results'/f"{cell['ordinal']}.json"
        if not path.exists():
            if (root/'claims'/f"{cell['ordinal']}.json").exists():lost+=1
            else:unattempted+=1
            continue
        row=json.loads(path.read_bytes())
        if row.get('schema')!=CELL_SCHEMA or row.get('cell')!=cell or row.get('declaration_sha256')!=declaration_sha256:
            raise ValueError('foreign panel result')
        rows.append(row);log=root/'telemetry'/f"{cell['ordinal']}.jsonl"
        found=[json.loads(x)for x in log.read_bytes().splitlines()]if log.exists()else []
        if any(r.get('cell')!=cell or r.get('arm')!=arm for r in found):raise ValueError('foreign decision telemetry')
        telemetry.extend(found)
        if row['status']=='COMPLETE'and len(found)!=(row.get('source_game')or {}).get('pokezero_decisions'):mismatch.append(cell['pair_ordinal'])
    complete=[r for r in rows if r['status']=='COMPLETE']
    return rows,dict(results=len(rows),complete=len(complete),unknown=len(rows)-len(complete)+lost,claimed_without_result=lost,
        not_attempted=unattempted,timeout_forfeits=sum(map(_timeout_forfeit,rows)),
        score_mean=None if not complete else sum(r['score']for r in complete)/len(complete),
        telemetry_mismatches=len(mismatch),telemetry_mismatch_pairs=mismatch[:50],
        decisions=decisions(telemetry,deadline=deadline,worlds=worlds,archive=archive,precision=precision),
        foulplay_cost=foulplay_cost(rows),
        load_average_1m=_quantiles([r['load_average']['end'][0]for r in rows if isinstance(r.get('load_average'),dict)]),
        workers=len({encoded(r['worker']).decode()for r in rows if isinstance(r.get('worker'),dict)}))


def paired(raw,search,*,limit):
    """Seed-clustered paired t (two-sided 90%) on search - our RAW; missing cells as bounds.

    Same statistics as Codex's phase_d_ladder_results.paired (a test pins
    equality): the two seats of a seed form one cluster; the point estimate
    exists only with every cell known; missing cells give identification bounds.
    """
    from scipy.stats import t
    def scores(rows):
        out={}
        for row in rows:
            known=row['status']=='COMPLETE';score=row['score']
            if known and score not in (0.,.5,1.)or not known and score is not None:raise ValueError('literal known terminal score or unknown required')
            if known:out[row['cell']['pair_ordinal']]=score
        return out
    a,b=scores(raw),scores(search);lower=[];upper=[];known={}
    for pair in range(limit):
        x,y=a.get(pair),b.get(pair)
        lower.append((0. if y is None else y)-(1. if x is None else x));upper.append((1. if y is None else y)-(0. if x is None else x))
        if x is not None and y is not None:known[pair]=y-x
    clusters=[(known[s]+known[s+1])/2 for s in range(0,limit,2)if s in known and s+1 in known]
    interval=None
    if len(clusters)>=2 and variance(clusters)>0:
        radius=float(t.ppf(T_QUANTILE,len(clusters)-1))*math.sqrt(variance(clusters)/len(clusters))
        interval=[max(-1.,fmean(clusters)-radius),min(1.,fmean(clusters)+radius)]
    n,k=limit,len(known);changed=sum(v!=0 for v in known.values())
    return dict(paired_cells=n,known_pairs=k,unknown_pairs=n-k,point_difference=fmean(lower)if k==n else None,
        full_roster_identification_bounds=[fmean(lower),fmean(upper)],complete_seed_clusters=len(clusters),
        complete_seed_difference=fmean(clusters)if clusters else None,interval90=interval,
        interval_status='AVAILABLE'if interval is not None else'UNAVAILABLE_TOO_FEW_OR_ZERO_SAMPLE_VARIANCE',
        discordant_known=changed,discordant_known_rate=None if k==0 else changed/k,discordant_full_bounds=[changed/n,(changed+n-k)/n])


def codex_raw_agreement(lad,ladder_sha256,ours,directory,*,limit):
    """Plumbing check only: our RAW against Codex's RAW on shared seeds."""
    codex={}
    for cell in ladder.cells(lad,'RAW',limit=limit):
        path=Path(directory)/'results'/f"{cell['ordinal']}.json"
        if not path.exists():continue
        row=json.loads(path.read_bytes())
        if ladder.checked_cell(lad,row.get('cell'))!=cell or row.get('declaration_sha256')!=ladder_sha256:raise ValueError('foreign ladder RAW result')
        codex[cell['pair_ordinal']]=row
    mine={r['cell']['pair_ordinal']:r for r in ours}
    shared=[p for p in sorted(codex)if p in mine and codex[p]['status']==mine[p]['status']=='COMPLETE']
    differ=[p for p in shared if codex[p]['score']!=mine[p]['score']]
    mean=lambda rows:None if not shared else sum(rows[p]['score']for p in shared)/len(shared)
    return dict(scope='PLUMBING_CHECK_ONLY_NOT_THE_COMPARISON',codex_results=len(codex),shared_complete=len(shared),
        agree=len(shared)-len(differ),agreement_share=_share(len(shared)-len(differ),len(shared)),
        our_score_mean=mean(mine),codex_score_mean=mean(codex),disagreeing_pairs=differ[:50])


def server_seconds(paths):
    """request_id -> the server's own seconds from arrival to result, from depthn_live_server --telemetry files."""
    out={}
    for path in paths or ():
        for line in Path(path).read_bytes().splitlines():
            row=json.loads(line)
            if (row.get('kind')=='request'and isinstance(row.get('response'),dict)and type(row.get('server_seconds'))in (int,float)
                    and math.isfinite(row['server_seconds'])):out[row['response'].get('request_id')]=row['server_seconds']
    return out


def deadline_coverage(rows,*,reserve,deadlines,minimum,late=None):
    """End-to-end wall (select_seconds) of searched decisions and, per candidate deadline, the share whose wall + reserve
    fits. A timeout's wall is censored at the client deadline: with the server's telemetry it is the pre-send time plus the
    server's own seconds, else it is unknown (lower bound counts it as not fitting, upper bound as fitting)."""
    walls=[]
    for row in rows:
        wall=row['select_seconds']
        if row.get('cause')=='timeout':
            search=row.get('search')or {};seconds=(late or {}).get(search.get('request_id'));client=search.get('client_wall_seconds')
            wall=None if seconds is None or client is None else row['select_seconds']-client+seconds
        walls.append(wall)
    known=[w for w in walls if w is not None];n=len(walls);censored=n-len(known)
    table={str(limit):dict(lower=None if not n else sum(w+reserve<=limit for w in known)/n,
        upper=None if not n else (sum(w+reserve<=limit for w in known)+censored)/n)for limit in deadlines}
    covering=next((limit for limit in deadlines if n and table[str(limit)]['lower']>=minimum),None)
    return dict(decisions=n,censored_timeouts=censored,wall=_quantiles(known),coverage=table,coverage_min=minimum,
        smallest_covering_deadline=covering,reserve_seconds=reserve)


def readout(run_directory,*,arm,limit,codex_raw_directory=None,server_telemetry=None,tree_commit=None):
    root,run,d,lad=load(run_directory);tree=require_tree(run,tree_commit)
    if arm not in d['arms'][1:]or limit not in (d['screen_paired_cells'],d['max_paired_cells']):raise ValueError('declared search arm and cell count required')
    rows={};arms={}
    for name in ('RAW',arm):
        rows[name],arms[name]=arm_summary(root,lad,arm=name,limit=limit,declaration_sha256=run['declaration_sha256'],
            deadline=d['deadline_seconds'],worlds=None if name=='RAW'else d['search'][arm]['worlds'],
            archive=None if name=='RAW'else d['archive_sha256'],precision=None if name=='RAW'else d['matmul_precision'])
    recorded=state_path(root,'latency_rule.json')
    latency=json.loads(recorded.read_bytes())if recorded.exists()else latency_rule(root,d,search_arm=arm,log=completions(root))
    telemetry_paths=None if server_telemetry is None else [str(Path(x).resolve())for x in server_telemetry]
    fair=arm_fairness(arms['RAW']['foulplay_cost'],arms[arm]['foulplay_cost'],d['arm_fairness'])
    acknowledged=state_path(root,'arm_fairness_acknowledged.json').exists()
    fair.update(acknowledged=acknowledged,recorded_flag=state_path(root,'arm_fairness_flag.json').exists())
    void=latency is not None and latency['status']=='VOID_LATENCY'
    coverage=None
    if void:   # the input for re-declaring the deadline: every searched decision of the screen so far
        searched=[r for cell in cells(lad,arm,limit=d['screen_paired_cells'])for r in telemetry_rows(root,cell['ordinal'])
                  if r.get('source')in ('search','greedy_fallback')]
        coverage=deadline_coverage(searched,reserve=d['reserve_seconds'],deadlines=d['screen_latency_rule']['coverage_deadlines'],
            minimum=d['screen_latency_rule']['coverage_min'],late=server_seconds(telemetry_paths))
    stats=None if void else paired(rows['RAW'],rows[arm],limit=limit)
    interval=None if stats is None else stats['interval90'];search=arms[arm]['decisions'];rules=d['breakage'];breakage=[]
    if sum(a['unknown']for a in arms.values())>rules['unknown_cells_max']:breakage.append('unknown_cells')
    if sum(a['timeout_forfeits']for a in arms.values())>rules['timeout_forfeits_max']:breakage.append('timeout_forfeits')
    p95=(search['latency']['select']or {}).get('p95')
    if p95 is not None and p95>rules['latency_p95_max_seconds']:breakage.append('latency_p95')
    if any(a['telemetry_mismatches']for a in arms.values()):breakage.append('telemetry')
    origin=search['provenance']
    if any(len(v)>1 for v in origin.values())or origin['root_operator']not in ([],[d['root_operator']]):breakage.append('search_provenance')
    if search['wrong_worlds']:breakage.append('requested_worlds')
    if search['wrong_archive']:breakage.append('archive')
    if search['wrong_precision']:breakage.append('matmul_precision')
    if search['transport_failure_share']is not None and search['transport_failure_share']>rules['transport_failure_share_max']:
        breakage.append('transport_failures')
    if search['searched_share']is not None and search['searched_share']<rules['searched_share_min']:breakage.append('searched_share')
    attempted=all(a['not_attempted']==0 for a in arms.values());harm=interval is not None and interval[1]<0;bar=None
    held=fair['flagged']and not acknowledged
    if void:decision='VOID_LATENCY'
    elif held:decision='FLAG_ARM_FAIRNESS'
    elif limit==d['screen_paired_cells']:
        decision='INCOMPLETE'if not attempted else'STOP_HARM'if harm else'STOP_BREAKAGE'if breakage else'CONTINUE_TO_2000'
    else:
        met=interval is not None and interval[0]>0
        roster=stats['unknown_pairs']==0 or stats['full_roster_identification_bounds'][0]>0
        bar=dict(rule=d['bar'],met=met,met_full_roster=met and roster,breakage_free=not breakage)
        decision='INCOMPLETE'if not attempted else'BAR_MET'if bar['met_full_roster']and not breakage else'BAR_NOT_MET'
    # Descriptive only: computed after the decision and never read by it.
    declared=d['expected_searched_share'];observed=search['searched_share']
    expected=dict(scope='DESCRIPTIVE_NOT_A_GATE',observed=observed,expected=None if declared is None else declared['value'],
        difference=None if declared is None or observed is None else observed-declared['value'],
        source=None if declared is None else declared['source'])
    return dict(schema=READOUT_SCHEMA,declaration_sha256=run['declaration_sha256'],ladder_declaration_sha256=run['ladder_declaration_sha256'],
        tree=tree,arm=arm,paired_cells=limit,comparison=d['primary_comparison'],confidence=d['confidence'],
        estimator='seed-clustered paired t over the two seats per seed, two-sided 90%; missing cells as identification bounds',
        paired=stats,decision=decision,harm=harm,breakage=breakage,screen_rule=d['screen_rule'],breakage_rules=rules,bar=bar,
        latency_rule=latency,deadline_coverage=coverage,arm_fairness=fair,deadline_seconds=d['deadline_seconds'],reserve_seconds=d['reserve_seconds'],
        dilution=dict(searched_share=search['searched_share'],fallback_share=search['fallback_share'],
            search_differs_share_of_all=search['search_differs_share_of_all']),expected_searched_share=expected,arms=arms,
        codex_raw_directory=None if codex_raw_directory is None else str(Path(codex_raw_directory).resolve()),server_telemetry=telemetry_paths,
        codex_raw_agreement=None if codex_raw_directory is None else
            codex_raw_agreement(lad,run['ladder_declaration_sha256'],rows['RAW'],codex_raw_directory,limit=limit))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cpu-pair',nargs=2,type=int,required=True)
    p.add_argument('--cache-directory',required=True)
    args=p.parse_args();asyncio.run(serve(tuple(args.cpu_pair),args.cache_directory))
