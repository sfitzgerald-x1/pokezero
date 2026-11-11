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
- One search arm per run (the lead fixes W at declaration time): arms are exactly
  RAW and one search arm.
- Arm fairness: Foul Play's 1,000 ms is wall clock, so both arms run interleaved
  (pair by pair, RAW then search, neither arm more than the concurrency ahead),
  Foul Play keeps Codex's dedicated actor/FP core pair, and each cell records its
  core pair, worker identity and load average. Whether Foul Play's realized work is
  observable (the bridge's iterations_observable) is declared up front with its
  evidence (fairness_observable); a cell contradicting it is breakage. When
  observable, depthn_live_fairness reads iterations_per_budget_second stratified by
  budget stratum x turn band; the Monitor evaluates it only at checkpoints (every
  50 completed pairs) once both floors hold (50 pairs, 1,000 rated moves per arm),
  per phase (screen, extension) and per acknowledgement window. A flag stops
  dispatch until `acknowledge --cause benign|fault`: benign re-arms over the moves
  after it (depthn_live_fairness.flag's re-arm rule), fault voids the run
  (VOID_FAIRNESS, never resumed). The extension starts un-acknowledged. The declared
  rule is depthn_live_fairness.RULE (screen-r1, screen-r2) or RULE_R3 (screen-r3: our
  and their exact alive bands from the owner capture, minimum support); the
  checkpoints and the readout read under the declared one. Flags,
  acknowledgements and checks are phase-scoped files, so a pinned screen readout
  recomputes identically whatever the extension records. When not observable, the
  gate is UNMEASURABLE: it rests on structural isolation plus interleaving, and the
  readout reports Foul Play's wait + overlap per arm as description only.
- Calibration (RAW only, seeds disjoint from every eval cell): `calibrate` runs
  RAW games solo (one lane) and at the declared worker-lane count, measures Foul
  Play's iterations_per_budget_second per budget stratum (at least the declared
  floor of rated moves per stratum and setting: 60 for 4x500ms, 300 for
  2x1000ms) and writes calibration.json; the declaration records its
  sha256 and the concurrent/solo ratios, init verifies them, and dispatch refuses
  unless every ratio is >= 0.90, or, when Foul Play is not observable, an explicit
  accept-uncalibrated is recorded.
- Search choice and provenance (descriptive, recorded before any dispatch): the
  evidence that chose the search arm's W (the cells' commit, their files by
  sha256, a summary of the result), and the opponent's and engine's identity.
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

from . import depthn_live_fairness as fairness
from . import phase_d_ladder_panel as ladder
from .depthn_live_actor import ARMS,CAUSES,CHAMPION_ARCHIVE_SHA256,REQUEST_KINDS,ROW_SCHEMA,SEARCH_ARMS,SERVER_CAUSES,SOURCES

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
ARM_FAIRNESS=fairness.RULE
ARM_FAIRNESS_RULES=(fairness.RULE,fairness.RULE_R3)   # the declarable arm fairness rules: screen-r1/r2's, screen-r3's
PHASES=('screen','extension')
CALIBRATION_SCHEMA='depthn-foulplay-panel.calibration.v1'
CALIBRATION_RUN_SCHEMA='depthn-foulplay-panel.calibration-run.v1'
CALIBRATION_REQUEST_SCHEMA='depthn-foulplay-panel.calibration-request.v1'
PREFLIGHT_REQUEST_SCHEMA='depthn-foulplay-panel.preflight-request.v1'
PREFLIGHT_SCHEMA='depthn-foulplay-panel.preflight.v1'
PREFLIGHT_WALL_MARGIN=.25   # the margin in the lead's bound on the search arm's end-to-end decision wall p95 (preflight_wall_p95_max: deadline - reserve - margin)


def preflight_wall_p95_max(deadline_seconds,reserve_seconds):
    """The lead's bound on the pre-flight's search-arm end-to-end decision wall p95: deadline - reserve - PREFLIGHT_WALL_MARGIN.
    2.6 s at the 3.0 s deadline and 0.15 s reserve (every screen through screen-r4a); a deadline re-declared under the latency rule
    moves it (4.6 s at 5.0 s), since a longer deadline lets the tail complete into the measured wall instead of timing out."""
    return round(float(deadline_seconds)-float(reserve_seconds)-PREFLIGHT_WALL_MARGIN,6)
PREFLIGHT_TIMEOUT_SHARE_MAX=.01   # the lead's bound on the search arm's timeout share (of non-single-legal decisions)
PREFLIGHT_REFUSED_SHARE_MAX=.25   # the lead's bound on the search arm's unpriced share (of non-single-legal decisions): searches the servers
# did not price (every server cause: refusals, expired in queue, server errors, bad requests) or the client could not reach them; pre-flights and
# screens through screen-r4a2 ran 4.6-10.4% (constructor refusals); a pre-flight whose searches were mostly unpriced measured only greedy walls
# (screen-r4a2's first 5.0 s pre-flight: 100% refused, wall 0.32 s). Timeouts have their own bound; a fallback after a priced response
# (an unrepresented effect, an illegal action, a bad q) carries a real search wall and does not count.
PREFLIGHT_REFUSED_CAUSES=SERVER_CAUSES+('connection_error',)
PREFLIGHT_REUSE_REFUSALS=('SEARCH_PREFIX_NOT_ORIGINAL_SUCCESSOR','SEARCH_REQUEST_ALREADY_ATTEMPTED')   # a live server still held the battle
# (its session and attempted addresses, kept until evicted beyond the session cap): the pre-flight reused a battle an earlier run in the same
# server lifetime had played; a pre-flight passes only with no search refused for this reason


NON_HOME_SHARE_MAX=.01   # the lead's breakage bound: answered search decisions off their battle's home server


def non_home(rows):
    """Keys (battle_id, owner, decision_index) of answered decisions off their battle's home: per battle, in decision order, the home
    is the first server (the response's server_id) that answered; a decision answered by another server, and every answered decision
    of that battle after it, is non_home (its server's constructor session missed decisions: a path the gates never validated).
    Unanswered decisions (no server_id) neither set nor break a home."""
    battles={}
    for r in rows:
        x=r.get('search')if isinstance(r.get('search'),dict)else {};sid=x.get('server_id')
        if type(sid)is str and sid and x.get('client_refused')is not True:
            battles.setdefault((r.get('battle_id'),r.get('owner')),[]).append((r.get('decision_index'),sid))
    out=set()
    for (battle,owner),seen in battles.items():
        home=None;broken=False
        for index,sid in sorted(seen,key=lambda x:(x[0]if type(x[0])is int else -1)):
            if home is None:home=sid
            broken=broken or sid!=home
            if broken:out.add((battle,owner,index))
    return out


def refusal_kind(text):
    """A refusal's kind: its text with numbers masked, first 120 characters (for counting kinds)."""
    import re
    return re.sub(r'[0-9]+','#',str(text or ''))[:120]
CALIBRATION_RATIO_MIN=.9
CALIBRATION_FLOORS=dict(fairness.CALIBRATION_FLOORS)   # rated moves per budget stratum and setting (the lead's ruling)
CALIBRATION_Z90=1.6448536269514722   # two-sided 90% normal quantile, for each stratum's descriptive ratio CI
CALIBRATION_START_NOTE=('concurrent lanes start together, so early-stratum contention is worse than the eval\'s staggered steady state: '
    'a conservative direction')
WORKER_IDENTITY_ENVIRONMENT=('POD_NAME','NODE_NAME')
TRANSPORT=('timeout','connection_error','server_error')
T_QUANTILE=.95  # Two-sided 90%; validate() pins confidence to .9.
KEYS={'schema','ladder_declaration_sha256','checkpoint_sha256','archive_sha256','arms','search','root_operator','deadline_seconds',
      'reserve_seconds','screen_paired_cells','max_paired_cells','confidence','primary_comparison','bar','screen_rule','breakage',
      'expected_searched_share','expected_searched_share_roster','matmul_precision','screen_latency_rule','bench_p95_wall_seconds',
      'bench_source','server_processes_per_gpu','interleave','arm_fairness','worker_lanes','fairness_observable','calibration','provenance',
      'search_choice','foulplay_move_wall','actor','server_tree_commit'}
ROOT_OPERATORS=('ARGMAX_MEAN_COMPLETED_Q','NOISE_GATED_K2')   # depthn_live_search.ROOT_OPERATORS: the declarable live root operators
REPLY_PRIOR_KEYS={'archive_sha256','checkpoint_sha256','view','peer_view_b','union_guard','views'}   # optional: a br_fp screen's reply model
ACTOR_KEYS={'cells_per_worker','warmup_decisions','search_overlaps_greedy','warm_p95_seconds','preflight_search_wall_p95_seconds','source'}
SEARCH_CHOICE_KEYS={'arm','cell_commit','evidence','summary'}
MOVE_WALL_SCHEMA='depthn-foulplay-panel.move-walls.v1'
MOVE_WALL_KEYS={'search_time_ms','label','strata','source'}
MOVE_WALL_STRATUM_KEYS={'sampled_battles','ms_per_sample','solo','concurrent'}
PROVENANCE_KEYS={'foulplay','showdown_commit','engine_tables','base_config_sha256'}
FOULPLAY_KEYS={'commit','applied_diff_sha256','patch_blobs','poke_engine','runtime_metadata_sha256','python'}
POKE_ENGINE_KEYS={'version','features','sdist_sha256','setup_script_sha256','gen3_patch_list_sha256','native_so_sha256'}
ENGINE_TABLES_KEYS={'digest','constructor_metadata_sha256','bank_metadata_sha256'}
REQUEST_KEYS={'schema','run_directory','declaration','declaration_sha256','ladder_declaration','ladder_declaration_sha256',
              'cell','checkpoint','checkpoint_sha256','base_config'}
encoded,sha,read_bound,write_new=ladder.encoded,ladder.sha,ladder.read_bound,ladder.write_new


def _hex(value,size=64):return type(value)is str and len(value)==size and all(c in '0123456789abcdef'for c in value)


def validate(declaration):
    """Our pre-registered eval declaration: exact keys, so no paths or extras."""
    from .live_protocol_actor import ActorContract
    d=declaration
    if type(d)is not dict or set(d)not in (KEYS,KEYS|{'reply_prior'}):raise ValueError('exact DepthN panel declaration keys required')
    arms=d['arms']
    if (d['schema']!=SCHEMA or not _hex(d['ladder_declaration_sha256'])
            or d['checkpoint_sha256']!=ActorContract.champion().checkpoint_sha256 or d['archive_sha256']!=CHAMPION_ARCHIVE_SHA256
            or type(arms)is not list or len(arms)!=2 or arms[0]!='RAW'or arms[1]not in SEARCH_ARMS
            or d['search']!={a:dict(depth=2,worlds=WORLDS[a])for a in arms[1:]}
            or d['root_operator']not in ROOT_OPERATORS or type(d['deadline_seconds'])is not float
            or d['deadline_seconds']not in DEADLINES or d['reserve_seconds']!=.15
            or d['screen_paired_cells']!=500 or d['max_paired_cells']!=2000 or d['confidence']!=.9
            or d['primary_comparison']!='SEARCH_MINUS_OUR_RAW_PAIRED_BY_CELL'
            or d['bar']!='LOWER_BOUND_GT_0_AT_2000_FULL_ROSTER_NO_BREAKAGE'or d['screen_rule']!=SCREEN_RULE
            or d['breakage']!=dict(BREAKAGE,latency_p95_max_seconds=d['deadline_seconds'])or d['matmul_precision']!='highest'
            or d['screen_latency_rule']!=LATENCY_RULE or d['interleave']!=INTERLEAVE or d['arm_fairness']not in ARM_FAIRNESS_RULES):
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
    observable=d['fairness_observable'];cal=d['calibration']
    if (type(d['worker_lanes'])is not int or d['worker_lanes']<1 or type(observable)is not dict or set(observable)!={'value','evidence'}
            or type(observable['value'])is not bool or not _evidence(observable['evidence'])
            or type(cal)is not dict or set(cal)!={'status','sha256','ratios','seed_band','floors'}or not _hex(cal['sha256'])
            or cal['floors']!=CALIBRATION_FLOORS or any(type(x)is not int for x in cal['floors'].values())
            or cal['status']!=('MEASURED'if observable['value']else'UNMEASURABLE')
            or (cal['ratios']is not None if cal['status']=='UNMEASURABLE'else type(cal['ratios'])is not dict
                or set(cal['ratios'])!=set(fairness.CALIBRATION_STRATA)
                or any(type(r)is not float or not math.isfinite(r)or r<=0 for r in cal['ratios'].values()))
            or type(cal['seed_band'])is not dict or set(cal['seed_band'])!={'battle_seed_start','foulplay_seed_start','count'}
            or any(type(v)is not int or v<1 for v in cal['seed_band'].values())):
        raise ValueError('worker lanes >= 1, fairness observability with its evidence, and a calibration (MEASURED with '
                         'positive per-stratum ratios when observable, else UNMEASURABLE) with its sha256 and seed band required')
    choice=d['search_choice']
    if (type(choice)is not dict or set(choice)!=SEARCH_CHOICE_KEYS or choice['arm']!=arms[1]or not _hex(choice['cell_commit'],40)
            or type(choice['evidence'])is not list or not choice['evidence']or not all(map(_evidence,choice['evidence']))
            or type(choice['summary'])is not str or not choice['summary'].strip()):
        raise ValueError('search choice: the declared search arm, the commit of the cells that chose it, their files (path and sha256) '
                         'and a summary of the result required')
    validate_move_wall(d['foulplay_move_wall']);validate_actor(d['actor'],d['deadline_seconds'],d['reserve_seconds'])
    if not _hex(d['server_tree_commit'],40):raise ValueError('the servers\' tree commit (40 hex) required')
    validate_provenance(d['provenance'])
    if 'reply_prior'in d:validate_reply_prior(d['reply_prior'])
    return d


def validate_reply_prior(value):
    """A br_fp screen's reply model (depthn_live_server --reply-archive): its archive and checkpoint sha256, the view it was trained
    on ('b' needs peer_view_b true, the server's own rule), and the union guard (automatic with a reply model) declared true. Every
    COMPLETE search must then carry it (reply_prior.reply_archive_sha256); without this key none may."""
    v=value
    if (type(v)is not dict or set(v)!=REPLY_PRIOR_KEYS or not _hex(v['archive_sha256'])or not _hex(v['checkpoint_sha256'])
            or v['view']not in ('b','env')or type(v['peer_view_b'])is not bool or (v['view']=='b')>v['peer_view_b']
            or v['union_guard']is not True or v['views']not in ('eager','jit')):
        raise ValueError('reply prior: archive and checkpoint sha256, view b (with peer_view_b) or env, union_guard true and views eager '
                         'or jit required')
    return v


def served_reply(response):
    """(the reply model's archive sha256 a response names, or None; whether it was built with peer view b; its views mode or None)."""
    reply=response.get('reply_prior')if isinstance(response.get('reply_prior'),dict)else {}
    return reply.get('reply_archive_sha256'),'peer_view_b'in response,reply.get('views')


def declared_reply(d):
    """(the declared reply archive sha256 or None; whether the declaration's searches are built with peer view b; views or None)."""
    reply=d.get('reply_prior')or {}
    return reply.get('archive_sha256'),bool(reply.get('peer_view_b')),reply.get('views')


def validate_actor(value,deadline_seconds,reserve_seconds):
    """The actor's time against search's budget (the lead's check; dispatch enforces cells_per_worker): persistent workers
    (cells_per_worker >= 2), each warmed once before its first claim (warm_actor on warmup_decisions captured decisions), the
    search request sent before the greedy forward, the measured warm p95s (encode, greedy) and the pre-flight's search-arm
    end-to-end decision wall p95 (at most preflight_wall_p95_max of the declared deadline and reserve), with their source (the pre-flight's preflight.json)."""
    v=value;warm=v.get('warm_p95_seconds')if type(v)is dict else None
    if (type(v)is not dict or set(v)!=ACTOR_KEYS or type(v['cells_per_worker'])is not int or v['cells_per_worker']<2
            or type(v['warmup_decisions'])is not int or v['warmup_decisions']!=WARMUP_DECISIONS or v['search_overlaps_greedy']is not True or not _evidence(v['source'])
            or type(warm)is not dict or set(warm)!={'encode','greedy'}
            or any(type(x)is not float or not math.isfinite(x)or x<=0 for x in (*warm.values(),v['preflight_search_wall_p95_seconds']))
            or v['preflight_search_wall_p95_seconds']>preflight_wall_p95_max(deadline_seconds,reserve_seconds)):
        raise ValueError('actor: persistent warmed workers (cells_per_worker >= 2, the declared warmup), the search request overlapping '
                         'greedy, the measured warm encode / greedy p95 and a pre-flight search wall p95 within the bound, with its source required')
    return v


def validate_move_wall(value):
    """Descriptive: what 'Foul Play at search_time_ms' means in this run. The base config's search_time_ms (init checks it),
    a label, and per budget stratum its sampled battles and ms per sample (as Foul Play reports them) with the per-move wall
    (the bridge's wait + overlap seconds: request forwarded to choice received) p50 / p95 over the calibration's moves in each
    setting, from calibration_move_walls' file (source)."""
    v=value;strata=v.get('strata')if type(v)is dict else None
    good=lambda q:(type(q)is dict and set(q)=={'moves','p50','p95'}and type(q['moves'])is int and q['moves']>0
                   and all(type(q[k])is float and math.isfinite(q[k])and q[k]>0 for k in ('p50','p95')))
    if (type(v)is not dict or set(v)!=MOVE_WALL_KEYS or type(v['search_time_ms'])is not int or v['search_time_ms']<=0
            or type(v['label'])is not str or not v['label'].strip()or not _evidence(v['source'])
            or type(strata)is not dict or set(strata)!=set(fairness.CALIBRATION_STRATA)
            or any(type(x)is not dict or set(x)!=MOVE_WALL_STRATUM_KEYS or any(type(x[k])is not int or x[k]<=0 for k in ('sampled_battles','ms_per_sample'))
                   or f"{x['sampled_battles']}x{x['ms_per_sample']}ms"!=c or not good(x['solo'])or not good(x['concurrent'])for c,x in strata.items())):
        raise ValueError('foulplay move wall: the base config\'s search_time_ms, a label, and per budget stratum its sampled battles, ms per '
                         'sample and per-move wall p50 / p95 per setting, with its source file, required')
    return v


def validate_provenance(value):
    """The opponent's and the engine's identity, recorded in the declaration (descriptive; init only checks the base config sha):
    the Foul Play runtime (its commit, applied patch diff and patch blobs, the poke-engine build: version, features, sdist, setup
    script, gen3 patch list and native .so sha256s, runtime metadata, python), the Showdown commit, the live executor's compiled
    engine tables (constructor_worlds.engine_tables_digest and both metadata shas the R1-C bridge compares) and the base config."""
    v=value;fp=v.get('foulplay')if type(v)is dict else None;pe=fp.get('poke_engine')if type(fp)is dict else None
    tables=v.get('engine_tables')if type(v)is dict else None
    if (type(v)is not dict or set(v)!=PROVENANCE_KEYS or type(fp)is not dict or set(fp)!=FOULPLAY_KEYS
            or not _hex(fp['commit'],40)or not _hex(fp['applied_diff_sha256'])or not _hex(fp['runtime_metadata_sha256'])
            or type(fp['python'])is not str or not fp['python'].strip()
            or type(fp['patch_blobs'])is not dict or not fp['patch_blobs']
            or any(type(k)is not str or not k.strip()or type(b)is not str or len(b)<8 or not _hex(b,len(b))for k,b in fp['patch_blobs'].items())
            or type(pe)is not dict or set(pe)!=POKE_ENGINE_KEYS or type(pe['version'])is not str or not pe['version'].strip()
            or type(pe['features'])is not str or not pe['features'].strip()
            or not all(_hex(pe[k])for k in ('sdist_sha256','setup_script_sha256','gen3_patch_list_sha256','native_so_sha256'))
            or not _hex(v['showdown_commit'],40)or type(tables)is not dict or set(tables)!=ENGINE_TABLES_KEYS
            or not all(_hex(tables[k])for k in ENGINE_TABLES_KEYS)or not _hex(v['base_config_sha256'])):
        raise ValueError('provenance: the Foul Play runtime identity, the Showdown commit, the engine tables digest with both metadata '
                         'shas, and the base config sha required')
    return v


def _evidence(value):
    return (type(value)is dict and set(value)=={'path','sha256'}and type(value['path'])is str and bool(value['path'].strip())
            and _hex(value['sha256']))


def calibration_cells(ladder_declaration,band):
    """RAW calibration cells on the declared band: both seats of `count` seeds, disjoint from every eval cell's battle and
    Foul Play seeds (the ladder's 1,000 seeds from each start)."""
    d=ladder.validate(ladder_declaration)
    if type(band)is not dict or set(band)!={'battle_seed_start','foulplay_seed_start','count'}or any(type(v)is not int or v<1 for v in band.values()):
        raise ValueError('calibration seed band required')
    for start,key in ((band['battle_seed_start'],'battle_seed_start'),(band['foulplay_seed_start'],'foulplay_seed_start')):
        if start<d[key]+d['seeds']and d[key]<start+band['count']:raise ValueError('calibration seeds overlap the eval cells')
    return tuple(dict(ordinal=k,pair_ordinal=k,offset=k//2,seat=k%2,arm='RAW',seed=band['battle_seed_start']+k//2,
        fp_seed=band['foulplay_seed_start']+k//2,calibration=True)for k in range(2*band['count']))


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


def server_tree_check(server_commit,eval_commit,declaration):
    """The lead's rule for servers started before the declaration was committed: they may run another commit than the eval tree
    only if `git diff --name-only server eval` is exactly the declaration's own repo path (or nothing: the same tree). Anything
    else, or no git answer, is not ok. Returns both commits, the declaration path, the differing files and ok."""
    path=_git('ls-files','--full-name','--error-unmatch',str(Path(declaration).resolve()))
    names=_git('diff','--name-only',str(server_commit),str(eval_commit))if _hex(server_commit,40)and _hex(eval_commit,40)else None
    files=None if names is None else sorted(x for x in names.splitlines()if x)
    declared=None if path is None else path.strip()
    ok=files is not None and declared is not None and files in ([],[declared])
    return dict(server_commit=server_commit,eval_commit=eval_commit,declaration_path=declared,files=files,ok=ok)


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
         base_config_sha256,calibration,tree_commit=None):
    """Bind one run directory to both pinned declarations, one base config, the calibration and the code identity."""
    d=validate(read_bound(declaration,declaration_sha256));lad=ladder.validate(read_bound(ladder_declaration,ladder_declaration_sha256))
    if d['ladder_declaration_sha256']!=ladder_declaration_sha256 or lad['checkpoint_sha256']!=d['checkpoint_sha256']:
        raise ValueError('panel declaration must pin this ladder declaration and champion')
    if d['provenance']['base_config_sha256']!=base_config_sha256:raise ValueError('the declaration\'s provenance must pin this base config')
    cal=d['calibration'];calibration_cells(lad,cal['seed_band']);measured=read_bound(calibration,cal['sha256'])
    if (type(measured)is not dict or measured.get('schema')!=CALIBRATION_SCHEMA or measured.get('status')!=cal['status']
            or measured.get('ratios')!=cal['ratios']or measured.get('seed_band')!=cal['seed_band']
            or measured.get('lanes')!=d['worker_lanes']or measured.get('iterations_observable')is not d['fairness_observable']['value']
            or measured.get('ladder_declaration_sha256')!=ladder_declaration_sha256
            or measured.get('base_config_sha256')!=base_config_sha256 or measured.get('checkpoint_sha256')!=d['checkpoint_sha256']):
        raise ValueError('calibration.json must match the declared status, ratios, seed band, lanes, observability, ladder, '
                         'base config (Foul Play budget included) and champion checkpoint')
    if cal['status']=='MEASURED':   # valid evidence: the move floor held, and the ratios are the recorded rates' own
        strata=measured.get('strata')
        if (measured.get('floors')!=CALIBRATION_FLOORS or type(strata)is not dict
                or set(strata)!=set(fairness.CALIBRATION_STRATA)
                or any(type(strata[s].get(k))is not dict or type(strata[s][k].get('moves'))is not int or strata[s][k]['moves']<CALIBRATION_FLOORS[s]
                       or type(strata[s][k].get('rate'))is not float or strata[s][k]['rate']<=0 for s in strata for k in ('solo','concurrent'))
                or any(strata[s]['concurrent']['rate']/strata[s]['solo']['rate']!=cal['ratios'][s]for s in strata)):
            raise ValueError(f'calibration.json is not valid evidence: the declared floors {CALIBRATION_FLOORS} of rated moves per stratum and setting, '
                             'and ratios recomputed from its rates, required')
    base=read_bound(base_config,base_config_sha256)
    if type(base)is not dict:raise ValueError('literal base configuration required')
    if base.get('search_time_ms')!=d['foulplay_move_wall']['search_time_ms']:raise ValueError('the declared Foul Play search_time_ms must be the base config\'s')
    root=Path(run_directory).resolve();tree=tree_identity(tree_commit)
    if (root/'calibration-run.json').exists():raise ValueError('a calibration or pre-flight directory is never an eval run directory')
    require_clean_tree(tree);check_preflight(d,tree)
    servers=server_tree_check(d['server_tree_commit'],tree['commit'],declaration)
    if not servers['ok']:raise ValueError(f'the servers\' tree may differ from the eval tree only in the declaration: {servers}')
    run=dict(schema=RUN_SCHEMA,run_directory=str(root),declaration=str(Path(declaration).resolve()),declaration_sha256=declaration_sha256,
        ladder_declaration=str(Path(ladder_declaration).resolve()),ladder_declaration_sha256=ladder_declaration_sha256,
        checkpoint=str(Path(checkpoint).resolve()),checkpoint_sha256=d['checkpoint_sha256'],base_config=base,
        base_config_sha256=base_config_sha256,calibration=str(Path(calibration).resolve()),tree=tree,server_tree=servers)
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
            server_telemetry=value['server_telemetry'],tree_commit=tree_commit,nodes=value.get('nodes'))
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


def phase_of(d,pair_ordinal):return 'screen'if pair_ordinal<d['screen_paired_cells']else'extension'


def fairness_directory(root,phase):return state_path(root,'fairness')/phase


def fairness_state(root,phase):
    """One phase's fairness record: its flags and acknowledgements in order, its checkpoint checks and any fault void."""
    directory=fairness_directory(root,phase)
    def numbered(prefix):
        found=sorted(directory.glob(prefix+'-*.json'),key=lambda p:int(p.stem.split('-')[1]))if directory.exists()else []
        return [json.loads(p.read_bytes())for p in found]
    checks=directory/'checks.jsonl';void=directory/'void.json'
    return dict(flags=numbered('flag'),acks=numbered('ack'),void=json.loads(void.read_bytes())if void.exists()else None,
        checks=[json.loads(x)for x in checks.read_bytes().splitlines()]if checks.exists()else [])


def refuse_by_state(root):
    """A voided run is never resumed; a fairness flag holds dispatch until the lead acknowledges it."""
    rule=state_path(root,'latency_rule.json')
    if rule.exists()and json.loads(rule.read_bytes())['status']=='VOID_LATENCY':
        raise ValueError('VOID_LATENCY: a voided screen is never resumed; restart with a fresh run and declaration')
    for phase in PHASES:
        state=fairness_state(root,phase)
        if state['void']is not None:raise ValueError('VOID_FAIRNESS: a run voided by an arm fairness fault is never resumed; restart fresh')
        if len(state['flags'])>len(state['acks']):raise ValueError("FLAG_ARM_FAIRNESS: dispatch waits for the lead's acknowledgement")


def acknowledge_arm_fairness(run_directory,*,cause,note):
    """The lead's acknowledgement of the pending fairness flag. benign re-arms the check over the moves after it; fault
    voids the run (VOID_FAIRNESS). The note must name the diagnosed cause."""
    root,run,d,lad=load(run_directory)
    if cause not in ('benign','fault')or type(note)is not str or not note.strip():raise ValueError('cause benign|fault and a note naming the cause required')
    backfill(root,lad,d['arms'][1])   # the window after this acknowledgement starts after every cell completed before it
    for phase in PHASES:
        state=fairness_state(root,phase)
        if len(state['flags'])>len(state['acks']):break
    else:raise ValueError('no arm fairness flag to acknowledge')
    flag=state['flags'][-1];number=len(state['flags']);directory=fairness_directory(root,phase)
    value=dict(phase=phase,flag=number,flag_sha256=hashlib.sha256((directory/f'flag-{number}.json').read_bytes()).hexdigest(),cause=cause,
        note=note,difference=flag['reading']['difference'],coverage_difference=flag['reading']['coverage_difference'],
        log_length=len(completions(root)),acknowledged_unix_seconds=time.time())
    write_new(directory/f'ack-{number}.json',value)
    if cause=='fault':write_new(directory/'void.json',dict(value,status='VOID_FAIRNESS'))
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
            'tree_commit','archive_sha256','pricer','client_wall_seconds')}|dict(zip(('reply_archive_sha256','peer_view_b','reply_views'),served_reply(response))))
        if response.get('client_refused')is False and response.get('archive_sha256')!=d['archive_sha256']:
            reason='server archive differs from the declared champion archive';break
        if response.get('client_refused')is False and response.get('tree_commit')!=d['server_tree_commit']:
            reason='server tree commit differs from the declared server tree';break
        if response['status']=='COMPLETE':
            good=(response.get('requested_worlds')==d['search'][arm]['worlds']and response.get('root_operator')==d['root_operator']
                and (response.get('pricer')or {}).get('matmul_precision')==d['matmul_precision'])
            status,reason=('PASS',None)if good else('FAIL','COMPLETE response with another W, root operator or matmul precision')
            if good and served_reply(response)!=declared_reply(d):status,reason='FAIL','COMPLETE response with another reply model or peer view'
            break
        if response['cause']not in ('constructor_refused','depthn_refused','class_refused'):reason='refused: '+str(response['cause']);break
    value=dict(schema=PROBE_SCHEMA,arm=arm,server=server,capture=str(Path(capture).resolve()),status=status,reason=reason,tried=tried)
    write_new(root/'probes'/f'{arm}-{time.time_ns()}.json',value)
    if status!='PASS':raise ValueError('search server probe failed: '+reason)
    return value


async def worker(packet,*,expected_pair,original,runtime_class=None,warmup=None):
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
    warmed=_warm_once(original,config,warmup)
    write_new(root/'claims'/f"{cell['ordinal']}.json",dict(request_sha256=digest,cell=cell,cpu_pair=list(expected_pair),
        server=server,pid=os.getpid(),claimed_unix_seconds=time.time(),warmup=warmed))
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


def calibration_init(directory,*,ladder_declaration,ladder_declaration_sha256,checkpoint,base_config,base_config_sha256,band,lanes,tree_commit=None):
    """A calibration directory (its own claims and results; never an eval run's) bound to the ladder, base config, the seed
    band and the planned worker-lane count."""
    lad=ladder.validate(read_bound(ladder_declaration,ladder_declaration_sha256));calibration_cells(lad,band)
    if type(lanes)is not int or lanes<1:raise ValueError('planned worker lanes >= 1 required')
    base=read_bound(base_config,base_config_sha256)
    if type(base)is not dict:raise ValueError('literal base configuration required')
    root=Path(directory).resolve()
    if (root/'run.json').exists():raise ValueError('an eval run directory is never a calibration or pre-flight directory')
    run=dict(schema=CALIBRATION_RUN_SCHEMA,run_directory=str(root),ladder_declaration=str(Path(ladder_declaration).resolve()),
        ladder_declaration_sha256=ladder_declaration_sha256,checkpoint=str(Path(checkpoint).resolve()),checkpoint_sha256=lad['checkpoint_sha256'],
        base_config=base,base_config_sha256=base_config_sha256,seed_band=band,lanes=lanes,tree=tree_identity(tree_commit))
    write_new(root/'calibration-run.json',run)
    return run


def calibration_load(directory):
    root=Path(directory).resolve();manifest=root/'calibration-run.json'
    if not manifest.exists():raise ValueError('a calibration directory required')
    run=json.loads(manifest.read_bytes())
    if run.get('schema')!=CALIBRATION_RUN_SCHEMA or run.get('run_directory')!=str(root):raise ValueError('this calibration manifest required')
    lad=ladder.validate(read_bound(run['ladder_declaration'],run['ladder_declaration_sha256']))
    return root,run,lad,calibration_cells(lad,run['seed_band'])


def calibration_packets(directory,*,setting):
    """Packets for the band's not-attempted calibration cells, in order, for one setting (solo or concurrent)."""
    root,run,lad,band=calibration_load(directory);todo=[]
    if setting not in ('solo','concurrent'):raise ValueError('setting solo or concurrent required')
    for cell in band:
        if (root/'claims'/f"{cell['ordinal']}.json").exists():continue
        path=root/'requests'/f"{cell['ordinal']}.json"
        value=dict(schema=CALIBRATION_REQUEST_SCHEMA,run_directory=run['run_directory'],cell=cell,checkpoint=run['checkpoint'],
            checkpoint_sha256=run['checkpoint_sha256'],base_config=run['base_config'])
        if path.exists():
            if path.read_bytes()!=encoded(value):raise ValueError('existing calibration request differs')
        else:write_new(path,value)
        todo.append(dict(path=str(path),sha256=sha(value),arm='RAW',mode='calibration',setting=setting,pair_ordinal=cell['pair_ordinal']))
    return todo


def preflight_packets(directory,*,arm,games,deadline_seconds,reserve_seconds):
    """Pre-flight packets (the lead's check of the overlapped path, outcome-blind, never counted): the band's first `games`
    not-attempted cells, alternately RAW and `arm`, each request carrying its arm and the declared client deadline and reserve."""
    root,run,lad,band=calibration_load(directory);todo=[]
    if arm not in SEARCH_ARMS or type(games)is not int or games<1:raise ValueError('a declared search arm and games >= 1 required')
    if (root/'calibration.json').exists()or any(json.loads(c.read_bytes()).get('setting')in ('solo','concurrent')for c in (root/'claims').glob('*.json')):
        raise ValueError('a pre-flight runs in its own directory, never a calibration\'s')
    for k,cell in enumerate(c for c in band if not (root/'claims'/f"{c['ordinal']}.json").exists()):
        if k>=games:break
        path=root/'requests'/f"{cell['ordinal']}.json"
        value=dict(schema=PREFLIGHT_REQUEST_SCHEMA,run_directory=run['run_directory'],cell=cell,checkpoint=run['checkpoint'],
            checkpoint_sha256=run['checkpoint_sha256'],base_config=run['base_config'],arm='RAW'if k%2==0 else arm,
            deadline_seconds=float(deadline_seconds),reserve_seconds=float(reserve_seconds))
        if path.exists():
            if path.read_bytes()!=encoded(value):raise ValueError('existing pre-flight request differs')
        else:write_new(path,value)
        todo.append(dict(path=str(path),sha256=sha(value),arm=value['arm'],mode='calibration',setting='preflight',pair_ordinal=cell['pair_ordinal']))
    return todo


def preflight_summary(directory,*,dispatch=None,server=None,cells_per_worker=None,code_tree=None,tree_commit=None):
    """The pre-flight's readings: per arm, each game's first-decision wall against the warm (decision >= 1) distribution's p99
    (the warmup's check), and the search arm's end-to-end decision wall (select_seconds, non-single-legal decisions) p95
    against preflight_wall_p95_max of the pre-flight's deadline and reserve, with fallback causes; plus the workers' warmups."""
    root,run,lad,band=calibration_load(directory);arms={};conditions=set()
    for cell in band:
        request=root/'requests'/f"{cell['ordinal']}.json"
        if request.exists():
            q=json.loads(request.read_bytes())
            if q.get('schema')==PREFLIGHT_REQUEST_SCHEMA and q.get('arm')in SEARCH_ARMS:conditions.add((q['arm'],q['deadline_seconds'],q['reserve_seconds']))
        result=root/'results'/f"{cell['ordinal']}.json"
        if not result.exists():continue
        value=json.loads(result.read_bytes())
        if value.get('setting')!='preflight':continue
        arm=value.get('arm');rows=telemetry_rows(root,cell['ordinal']);off=non_home(rows)
        a=arms.setdefault(arm,dict(games=0,complete=0,first=[],warm=[],encode=[],greedy=[],search=[],causes=Counter(),kinds=Counter(),non_home=0,answered=0,
            first_actor=[],warm_actor=[],reuse=0));
        a['non_home']+=len(off);a['answered']+=sum(1 for r in rows if type((r.get('search')or {}).get('server_id'))is str and (r.get('search')or {}).get('client_refused')is not True)
        a['games']+=1;a['complete']+=value.get('status')=='COMPLETE'
        for r in rows:
            (a['first']if r.get('decision_index')==0 else a['warm']).append(r['select_seconds'])
            (a['first_actor']if r.get('decision_index')==0 else a['warm_actor']).append(r['encode_seconds']+r['greedy']['seconds'])
            if r.get('decision_index')!=0:a['encode'].append(r['encode_seconds']);a['greedy'].append(r['greedy']['seconds'])
            if r.get('source')in ('search','greedy_fallback'):
                a['search'].append(r['select_seconds'])
                if r.get('cause')is not None:a['causes'][r['cause']]+=1
                if r.get('cause')=='constructor_refused':a['kinds'][refusal_kind((r.get('search')or {}).get('refusal'))]+=1
                if any(k in str((r.get('search')or {}).get('refusal')or '')for k in PREFLIGHT_REUSE_REFUSALS):a['reuse']+=1
    out={}
    for arm,a in sorted(arms.items(),key=lambda x:str(x[0])):
        warm=_quantiles(a['warm']);p99=None if warm is None else warm['p99']
        actor=_quantiles(a['warm_actor']);actor_p99=None if actor is None else actor['p99']
        out[arm]=dict(games=a['games'],complete=a['complete'],first_decision=sorted(a['first']),warm=warm,
            warm_encode=_quantiles(a['encode']),warm_greedy=_quantiles(a['greedy']),
            first_within_warm_p99=None if p99 is None else all(x<=p99 for x in a['first']),   # descriptive: includes the server's time
            first_end_to_end_note=None if arm not in SEARCH_ARMS else ('descriptive: a search arm\'s end-to-end first decision carries '
                'the server\'s time too; all lanes start together, so the opening search battles price at once and share GPUs (see the '
                'server telemetry\'s price times); the warmup is gated on the actor time (first_actor_within_warm_p99)'),
            first_actor=sorted(a['first_actor']),warm_actor=actor,   # the warmup's check: the actor's own time (encode + greedy) on first decisions
            first_actor_within_warm_p99=None if actor_p99 is None else all(x<=actor_p99 for x in a['first_actor']),
            search_wall=_quantiles(a['search']),fallback_causes=dict(sorted(a['causes'].items())),
            timeout_share=_share(a['causes']['timeout'],len(a['search'])),constructor_refused_share=_share(a['causes']['constructor_refused'],len(a['search'])),
            searched_share_of_non_timeouts=_share(len(a['search'])-sum(a['causes'].values()),len(a['search'])-a['causes']['timeout']),
            refused_share=_share(sum(a['causes'][c]for c in PREFLIGHT_REFUSED_CAUSES),len(a['search'])),reuse_refusals=a['reuse'],
            constructor_refusal_kinds=dict(a['kinds'].most_common(10)),non_home=a['non_home'],answered=a['answered'],
            non_home_share=_share(a['non_home'],a['answered']))
    if len(conditions)!=1:raise ValueError(f'one search arm, deadline and reserve per pre-flight required, found {sorted(conditions)}')
    (arm,deadline,reserve),=conditions
    walls=[out[a]['search_wall']for a in out if a in SEARCH_ARMS and out[a]['search_wall']]
    timeouts=[out[a]['timeout_share']for a in out if a in SEARCH_ARMS and out[a]['timeout_share']is not None]
    refused=[out[a]['refused_share']for a in out if a in SEARCH_ARMS and out[a]['refused_share']is not None]
    reuse=[out[a]['reuse_refusals']for a in out if a in SEARCH_ARMS]
    claims=[json.loads(p.read_bytes())for p in sorted((root/'claims').glob('*.json'))]
    return dict(schema=PREFLIGHT_SCHEMA,arm=arm,deadline_seconds=deadline,reserve_seconds=reserve,server=server,cells_per_worker=cells_per_worker,
        dispatch=dispatch,arms=out,search_wall_p95_max=preflight_wall_p95_max(deadline,reserve),
        search_wall_within=None if not walls else all(w['p95']<=preflight_wall_p95_max(deadline,reserve)for w in walls),
        timeout_share_max=PREFLIGHT_TIMEOUT_SHARE_MAX,timeouts_within=None if not timeouts else all(t<=PREFLIGHT_TIMEOUT_SHARE_MAX for t in timeouts),
        refused_share_max=PREFLIGHT_REFUSED_SHARE_MAX,refused_within=None if not refused else all(x<=PREFLIGHT_REFUSED_SHARE_MAX for x in refused),
        reuse_refusals=None if not reuse else sum(reuse),
        warmups=[c['warmup']for c in claims if c.get('setting')=='preflight'and c.get('warmup')is not None],
        seed_band=run['seed_band'],lanes=run['lanes'],calibration_tree=run['tree'],
        code_tree=code_tree if code_tree is not None else tree_identity(tree_commit))


def require_clean_tree(tree):
    """init's eval tree: a clean git checkout (dirty False), since the server-tree rule compares commits, not working trees."""
    if tree.get('dirty')is not False:raise ValueError('a clean git tree required: the server-tree rule compares commits')
    return tree


def check_preflight(d,tree):
    """The declared actor block against its source (the pre-flight's preflight.json, sha256-bound): the declared arm, deadline,
    reserve and cells per worker; a dispatch that neither failed nor stopped; every game's first decision's actor time (encode +
    greedy) within its arm's warm actor p99 (the warmup worked; the end-to-end first decision, which also carries the server's time
    and the start-together GPU wave, is descriptive); the search wall within the bound, which is the declaration's
    (preflight_wall_p95_max of its deadline and reserve); timeouts and server refusals (PREFLIGHT_REFUSED_CAUSES) within
    their bounds and no reuse refusal (PREFLIGHT_REUSE_REFUSALS: a battle the live servers already held); the declared numbers equal to the file's; and the pre-flight's code
    the eval's (the worker code digest: the declaration JSON is not part of it)."""
    a=d['actor'];value=read_bound(a['source']['path'],a['source']['sha256']);arm=d['arms'][1]
    raw=(value.get('arms')or {}).get('RAW')or {};search=(value.get('arms')or {}).get(arm)or {}
    outcome=value.get('dispatch')if isinstance(value.get('dispatch'),dict)else {}
    pick=lambda q,k:(q or {}).get(k)
    if (value.get('schema')!=PREFLIGHT_SCHEMA or value.get('arm')!=arm or value.get('deadline_seconds')!=d['deadline_seconds']
            or value.get('reserve_seconds')!=d['reserve_seconds']or value.get('cells_per_worker')!=a['cells_per_worker']
            or value.get('search_wall_p95_max')!=preflight_wall_p95_max(d['deadline_seconds'],d['reserve_seconds'])
            or outcome.get('failed')!=[]or outcome.get('stopped')is not None or value.get('search_wall_within')is not True
            or value.get('timeouts_within')is not True or value.get('refused_within')is not True or value.get('reuse_refusals')!=0
            or value.get('refused_share_max')!=PREFLIGHT_REFUSED_SHARE_MAX
            or raw.get('first_actor_within_warm_p99')is not True or search.get('first_actor_within_warm_p99')is not True
            or pick(search.get('search_wall'),'p95')!=a['preflight_search_wall_p95_seconds']
            or pick(raw.get('warm_encode'),'p95')!=a['warm_p95_seconds']['encode']or pick(raw.get('warm_greedy'),'p95')!=a['warm_p95_seconds']['greedy']
            or pick(value.get('code_tree'),'source_sha256')!=tree['source_sha256']):
        raise ValueError('the declared actor must be its pre-flight\'s: same arm, deadline, reserve and cells per worker, a clean dispatch, '
                         'first decisions within the warm p99, the search wall, timeouts and server refusals within their bounds, no reused battle, '
                         'the same numbers and the same code')
    return value


async def calibration_worker(packet,*,expected_pair,original,runtime_class=None,warmup=None):
    """One RAW calibration game (or, in setting 'preflight', one RAW or search-arm pre-flight game: outcome-blind, never
    counted) through the same Runtime, bridge and Foul Play path; claims and results stay in the calibration directory, and
    only band cells (disjoint from every eval cell) are accepted."""
    if type(packet)is not dict or set(packet)!={'path','sha256','server','setting'}:raise ValueError('one calibration packet required')
    value=read_bound(packet['path'],packet['sha256']);root=Path(value.get('run_directory','')).resolve()
    _,run,lad,band=calibration_load(root);cell=value.get('cell');preflight=packet['setting']=='preflight'
    keys={'schema','run_directory','cell','checkpoint','checkpoint_sha256','base_config'}|({'arm','deadline_seconds','reserve_seconds'}if preflight else set())
    if (set(value)!=keys or value['schema']!=(PREFLIGHT_REQUEST_SCHEMA if preflight else CALIBRATION_REQUEST_SCHEMA)
            or cell not in band or Path(packet['path']).resolve()!=root/'requests'/f"{cell['ordinal']}.json"
            or packet['setting']not in ('solo','concurrent','preflight')):raise ValueError('a band calibration request at its own path required')
    arm=value['arm']if preflight else'RAW';server=packet['server']
    if arm not in ARMS or (server is None)!=(arm=='RAW')or server is not None and (type(server)is not dict or set(server)!={'host','port'}):
        raise ValueError('RAW runs without a server; a search arm needs one')
    config=ladder.configuration(value['base_config'],cell,checkpoint=value['checkpoint'],cpu_pair=expected_pair)
    from .live_protocol_runtime import _checkpoint_sha
    if _checkpoint_sha(config.checkpoint)!=value['checkpoint_sha256']:raise ValueError('original champion bytes differ')
    from .depthn_live_actor import Client,Runtime
    client=None if server is None else Client(server['host'],server['port'],rung=arm,deadline_seconds=value['deadline_seconds'],
        reserve_seconds=value['reserve_seconds'])
    runtime=(Runtime if runtime_class is None else runtime_class)(client=client,telemetry_path=root/'telemetry'/f"{cell['ordinal']}.jsonl",
        arm=arm,cell=cell,original=original)
    warmed=_warm_once(original,config,warmup)
    write_new(root/'claims'/f"{cell['ordinal']}.json",dict(request_sha256=packet['sha256'],cell=cell,setting=packet['setting'],
        cpu_pair=list(expected_pair),arm=arm,pid=os.getpid(),warmup=warmed))
    load=dict(start=list(os.getloadavg()));document=game=None
    try:
        with ladder.owning_capture(root/'owning-wire',cell=cell,declaration_sha256=sha(run)):result=await runtime.run(config)
        document=result.to_dict();game=result.games[0].to_dict();status,error='COMPLETE',None
    except Exception as error_:status,error=('UNKNOWN',type(error_).__name__+': '+str(error_))
    write_new(root/'results'/f"{cell['ordinal']}.json",dict(schema=CELL_SCHEMA,status=status,error=error,cell=cell,setting=packet['setting'],arm=arm,
        cpu_pair=list(expected_pair),source_game=game,benchmark=document,owning_wire=str(root/'owning-wire'/f"{cell['ordinal']}.jsonl"),
        worker=worker_identity(),load_average=dict(load,end=list(os.getloadavg()))))
    return 0


def ratio_ci90(concurrent,solo):
    """A descriptive two-sided 90% CI of mean(concurrent)/mean(solo) by the delta method (independent samples), or None with fewer
    than two of either."""
    if len(concurrent)<2 or len(solo)<2:return None
    mc,ms=fmean(concurrent),fmean(solo)
    if mc<=0 or ms<=0:return None
    r=mc/ms;se=r*math.sqrt(variance(concurrent)/(len(concurrent)*mc*mc)+variance(solo)/(len(solo)*ms*ms))
    return [r-CALIBRATION_Z90*se,r+CALIBRATION_Z90*se]


def calibration_summary(directory,*,floors=CALIBRATION_FLOORS):
    """Per budget stratum and setting: rated moves and mean iterations_per_budget_second; the concurrent/solo ratio; the
    observability the games' headers report; status MEASURED, INSUFFICIENT or UNMEASURABLE."""
    root,run,lad,band=calibration_load(directory);results=[read_result(root,c['ordinal'])for c in band]
    results=[r for r in results if r.get('status')=='COMPLETE'and r.get('setting')in ('solo','concurrent')]   # pre-flight games never count
    headers={((r.get('benchmark')or {}).get('foulplay_think')or {}).get('iterations_observable')for r in results}
    # UNMEASURABLE only when no game reports True; None is the bridge's failed probe, not a reading. Games that report both
    # True and False contradict each other: INSUFFICIENT (fail closed: the screen waits until someone looks).
    mixed=True in headers and False in headers
    observable=True if True in headers else False if False in headers else None
    rates={s:{c:[]for c in fairness.CALIBRATION_STRATA}for s in ('solo','concurrent')}
    for r in results:
        for stratum,_,rate in fairness.moves(r)['rated']:   # per budget stratum only: no turn bands needed
            if stratum in fairness.CALIBRATION_STRATA:rates[r['setting']][stratum].append(rate)
    strata={c:dict(solo=dict(moves=len(rates['solo'][c]),rate=fmean(rates['solo'][c])if rates['solo'][c]else None),
        concurrent=dict(moves=len(rates['concurrent'][c]),rate=fmean(rates['concurrent'][c])if rates['concurrent'][c]else None))
        for c in fairness.CALIBRATION_STRATA}
    for c,value in strata.items():
        value['ratio']=(value['concurrent']['rate']/value['solo']['rate']if value['solo']['rate']and value['concurrent']['rate']is not None else None)
        value['ratio_ci90']=ratio_ci90(rates['concurrent'][c],rates['solo'][c])   # descriptive; the gate is the point estimate
        value['ratio_ci90_note']='moves treated as independent; they cluster by game, so this interval is too narrow (descriptive only)'
    enough=all(v[s]['moves']>=floors[c]for c,v in strata.items()for s in ('solo','concurrent'))
    status='INSUFFICIENT'if mixed else'UNMEASURABLE'if observable is not True else'MEASURED'if enough else'INSUFFICIENT'
    return dict(schema=CALIBRATION_SCHEMA,status=status,iterations_observable=observable,observability_mixed=mixed,strata=strata,
        ratios={c:v['ratio']for c,v in strata.items()}if status=='MEASURED'else None,ratio_min=CALIBRATION_RATIO_MIN,floors=dict(floors),
        concurrent_start=CALIBRATION_START_NOTE,
        lanes=run['lanes'],seed_band=run['seed_band'],ladder_declaration_sha256=run['ladder_declaration_sha256'],
        base_config_sha256=run['base_config_sha256'],checkpoint_sha256=run['checkpoint_sha256'],
        games={s:sum(r['setting']==s for r in results)for s in ('solo','concurrent')},
        hosts=sorted({encoded(r['worker']).decode()for r in results if isinstance(r.get('worker'),dict)}),
        core_pairs=[list(pair)for pair in sorted({tuple(r['cpu_pair'])for r in results if r.get('cpu_pair')})],tree=run['tree'])


def calibration_move_walls(directory,*,tree_commit=None):
    """Descriptive (no rule reads it): per budget stratum, the sampled battles and ms per sample Foul Play reported and, per
    setting, its per-move wall: the bridge's wait + overlap seconds (request forwarded to its choice received), over the
    calibration's COMPLETE cells. With search parallelism 1 the sampled battles run one after another, so the wall is about
    sampled battles x ms per sample plus overhead."""
    root,run,lad,band=calibration_load(directory);results=[read_result(root,c['ordinal'])for c in band]
    walls={c:{s:[]for s in ('solo','concurrent')}for c in fairness.CALIBRATION_STRATA};shape={c:Counter()for c in fairness.CALIBRATION_STRATA}
    for r in results:
        if r.get('status')!='COMPLETE'or r.get('setting')not in ('solo','concurrent'):continue
        for row in (r.get('source_game')or {}).get('opponent_think')or ():
            c=row.get('stratum')
            if c not in walls or not all(type(row.get(k))in (int,float)for k in ('wait_seconds','overlap_seconds')):continue
            walls[c][r['setting']].append(row['wait_seconds']+row['overlap_seconds']);shape[c][(row.get('sampled_battles'),row.get('budget_ms_per_sample'))]+=1
    quantiles=lambda v:None if not v else dict(moves=len(v),**{k:x for k,x in _quantiles(v).items()if k in ('p50','p95')})
    return dict(schema=MOVE_WALL_SCHEMA,search_time_ms=run['base_config'].get('search_time_ms')if isinstance(run.get('base_config'),dict)else None,
        strata={c:dict(shapes={f'{b}x{m}ms':n for (b,m),n in sorted(shape[c].items(),key=str)},**{s:quantiles(walls[c][s])for s in ('solo','concurrent')})
            for c in fairness.CALIBRATION_STRATA},seed_band=run['seed_band'],lanes=run['lanes'],calibration_tree=run['tree'],code_tree=tree_identity(tree_commit))


async def calibrate(directory,*,pairs,cache_directory=None,command=None,floors=CALIBRATION_FLOORS,dispatch_function=None):
    """Solo (one lane) then concurrent (the planned lanes) RAW games on band seeds until every budget stratum has its floor of
    rated moves in the setting (or Foul Play's work proves unobservable, or the band runs out); writes calibration.json once."""
    root,run,lad,band=calibration_load(directory);run_dispatch=dispatch if dispatch_function is None else dispatch_function
    command=worker_command if command is None else command
    if len(pairs)!=run['lanes']:raise ValueError('calibrate with exactly the planned worker lanes')
    for setting,lanes in (('solo',pairs[:1]),('concurrent',pairs)):
        def enough():
            value=calibration_summary(root,floors=floors)
            if value['iterations_observable']is False:return 'UNOBSERVABLE'
            return 'ENOUGH'if all(v[setting]['moves']>=floors[c]for c,v in value['strata'].items())else None
        state=enough()
        if state=='UNOBSERVABLE':break
        if state=='ENOUGH':continue
        outcome=await run_dispatch(calibration_packets(root,setting=setting),pairs=lanes,servers={'RAW':None},cache_directory=cache_directory,
            command=command,after=lambda packet:enough())
        if outcome.get('stopped')=='UNOBSERVABLE'or outcome.get('failed'):break
    value=calibration_summary(root,floors=floors);write_new(root/'calibration.json',value)
    return value


def require_calibration(root,d,*,pairs,accept_uncalibrated=False):
    """The screen starts only within the calibrated concurrency and with every stratum's concurrent/solo ratio >= 0.90;
    an unobservable (UNMEASURABLE) calibration needs an explicit accept-uncalibrated, recorded once."""
    if len(pairs)>d['worker_lanes']:raise ValueError(f"more worker lanes than the {d['worker_lanes']} calibrated")
    cal=d['calibration']
    if cal['status']=='MEASURED':
        low={s:r for s,r in cal['ratios'].items()if r<CALIBRATION_RATIO_MIN}
        if low:raise ValueError(f'calibration concurrent/solo ratios below {CALIBRATION_RATIO_MIN}: {low}; lower the concurrency or give Foul Play '
                                'more cores, then recalibrate')
        return None
    if not accept_uncalibrated:raise ValueError('calibration UNMEASURABLE (Foul Play not observable): pass --accept-uncalibrated to proceed')
    path=state_path(root,'accept_uncalibrated.json')
    if not path.exists():write_new(path,dict(status=cal['status'],accepted_unix_seconds=time.time()))
    return json.loads(path.read_bytes())


def worker_identity():
    """This worker's host and, when the environment provides them, its pod and node: run outputs only."""
    return dict(hostname=socket.gethostname(),**{k.lower():os.environ[k]for k in WORKER_IDENTITY_ENVIRONMENT if os.environ.get(k)})


async def run_packet(packet,*,pair,original,warmup=None):
    """One packet to its worker: calibration and pre-flight packets (mode 'calibration') to calibration_worker, eval cells to
    worker, each with this process's warmup capture."""
    run=calibration_worker if packet.get('mode')=='calibration'else worker
    if run is calibration_worker:packet={k:v for k,v in packet.items()if k!='mode'}
    return await run(packet,expected_pair=pair,original=original,warmup=warmup)


async def serve(pair,cache_directory,warmup=None):
    """CPU-only worker: one game at a time, packets on stdin until EOF; with a warmup capture, warm_actor once before the
    first claim."""
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
        with contextlib.redirect_stdout(sys.stderr):code=await run_packet(json.loads(line),pair=pair,original=original,warmup=warmup)
        output.write(json.dumps({'exit':code})+'\n');output.flush()


def worker_command(pair,cache_directory,warmup=None):
    return [sys.executable,'-m','pokezero.depthn_live_panel','--cpu-pair',str(pair[0]),str(pair[1]),'--cache-directory',str(cache_directory),
        *(()if warmup is None else ('--warmup-capture',str(warmup)))]


WARMUP_DECISIONS=12
_WARMED={}


def warm_actor(original,config,capture,*,decisions=WARMUP_DECISIONS):
    """A worker's warmup (the lead's ruling), once, before its first claim: the champion's own encode + greedy forward on
    `decisions` captured decisions spread over one owning-wire capture, each in a fresh game of the original raw actor (no
    telemetry, no search, no Foul Play), so the encoder's jit and torch's first forward are paid outside any game. The raw
    factory and model are the ones every later game of this worker uses (ChampionWorkerRuntime.prepare). Returns the
    capture, the decision count and each decision's seconds."""
    import random
    from .phase_d_owning_capture import read_capture
    rows=list(read_capture(capture))
    if type(decisions)is not int or decisions<1 or len(rows)<decisions:raise ValueError(f'a warmup capture with at least {decisions} decisions required')
    rows=rows[::max(1,len(rows)//decisions)][:decisions]
    factory=original.prepare(config);model,result,env=original.runtime[:3]
    # Every game's first step in ChampionWorkerRuntime.run, verbatim: the bridge's policy (it puts the shared model in eval mode on
    # CPU, which the owning policy factory requires before any actor is built).
    from . import foulplay_bridge as bridge
    bridge._build_policy(config=config,model=model,result=result,value_model=model,value_result=result,env_config=env,
        rollout_config=bridge.RolloutConfig(max_decision_rounds=config.max_decision_rounds,format_id=config.format_id),
        policy_id=str(result.model_config.policy_id))
    actor=factory.actor((model,result,env));seconds=[]
    for k,row in enumerate(rows):
        actor.begin_game(battle_id=f'warmup-{os.getpid()}-{k}',owner=row['owner'],seed=k,stream_id=k+1)
        try:
            actor.accept_owner_stream(battle_id=actor.battle_id,owner=row['owner'],generation=0,prefix_sha256=hashlib.sha256().hexdigest(),
                lines=tuple(row['lines']))
            started=time.monotonic();actor.decide(actor.snapshot_boundary(),rng=random.Random(k));seconds.append(time.monotonic()-started)
        finally:actor.close_game()
    return dict(capture=str(Path(capture).resolve()),decisions=len(seconds),seconds=seconds)


def _warm_once(original,config,warmup):
    """warm_actor on this worker's first cell only (None afterwards, or without a warmup capture)."""
    if warmup is None or id(original)in _WARMED:return None
    value=warm_actor(original,config,warmup);_WARMED[id(original)]=(original,value)   # held: its id is never reused
    return value


async def dispatch(packets,*,pairs,servers,cells_per_worker=1,command=worker_command,cache_directory=None,attempted=None,after=None,warmup=None):
    """One worker process per core pair at a time, fresh after cells_per_worker cells.

    servers maps every arm to its server (None for RAW). Interleave: a cell is taken
    (in packet order) only if its arm stays at most len(pairs) cells ahead of every
    other arm, counting attempted cells; when no cell qualifies, dispatch stops
    ('interleave') rather than run one arm alone. With warmup (a capture path), each
    worker process runs warm_actor once before its first claim, so no game pays the
    encoder's jit or torch's first forward. after(packet), called once per
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
            argv=command(pair,cache_directory)if warmup is None else command(pair,cache_directory,warmup=warmup)
            process=await asyncio.create_subprocess_exec(*argv,stdin=asyncio.subprocess.PIPE,stdout=asyncio.subprocess.PIPE)
            try:
                for _ in range(cells_per_worker):
                    if not queue or failed or stopped:break
                    packet=take()
                    if packet is None:break
                    wire=dict(path=packet['path'],sha256=packet['sha256'],server=servers[packet['arm']],
                        **{k:packet[k]for k in ('mode','setting')if k in packet})
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


def missing_from_log(root,lad,search_arm,log):
    """Cells with a result but no completion-log line (dispatch stopped in between), in their one deterministic place:
    RAW then the search arm, each by ordinal."""
    known={entry['ordinal']for entry in log}
    return [dict(ordinal=c['ordinal'],arm=c['arm'],pair_ordinal=c['pair_ordinal'])for arm in ('RAW',search_arm)for c in cells(lad,arm)
            if c['ordinal']not in known and (Path(root)/'results'/f"{c['ordinal']}.json").exists()]


def effective_log(root,lad,search_arm):
    """The completion log as the Monitor holds it: the recorded order, then any missing cells in their deterministic place.
    Readouts evaluate the latency rule and fairness windows on exactly this."""
    log=completions(root);return log+missing_from_log(root,lad,search_arm,log)


def backfill(root,lad,search_arm):
    """Append the missing cells to the completion log (what effective_log reads), once."""
    for entry in missing_from_log(root,lad,search_arm,completions(root)):_log(root,entry)


def _log(root,entry):
    path=state_path(root,'completions.jsonl');path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('ab')as stream:stream.write(encoded(entry)+b'\n');stream.flush();os.fsync(stream.fileno())


def read_result(root,ordinal):
    path=Path(root)/'results'/f'{ordinal}.json'
    return json.loads(path.read_bytes())if path.exists()else {}


def cell_moves(root,ordinal,result=None,rule=ARM_FAIRNESS):
    """One cell's Foul Play moves (depthn_live_fairness.moves), banded by its owner capture's turns; under a rule with
    alive_bands (RULE_R3) also labelled by both sides' alive bands from that capture (round_states)."""
    result=read_result(root,ordinal)if result is None else result
    capture=result.get('owning_wire')or Path(root)/'owning-wire'/f'{ordinal}.jsonl'
    if fairness.alive_labelled(rule):return fairness.moves(result,fairness.round_states(capture),alive=True)
    return fairness.moves(result,fairness.round_turns(capture))


def completed_pairs(entries,search_arm):
    arms={}
    for entry in entries:arms.setdefault(entry['pair_ordinal'],set()).add(entry['arm'])
    return sum(found>={'RAW',search_arm}for found in arms.values())


def observability(results,declared):
    """Cells whose bridge header reports a bool iterations_observable other than the declaration (contradictions: breakage),
    and cells reporting None (the bridge's probe failed: counted apart, never breakage)."""
    seen=[((r.get('benchmark')or {}).get('foulplay_think')or {}).get('iterations_observable')for r in results if r.get('status')=='COMPLETE']
    return dict(contradictions=sum(type(v)is bool and v is not declared for v in seen),unknown=sum(v is None for v in seen))


class Monitor:
    """Run state after each completed cell: completion order, the screen latency rule (evaluated once, recorded) and arm
    fairness at its checkpoints (phase-scoped; a flag is recorded and stops dispatch until acknowledged). Returns a stop
    reason or None."""
    def __init__(self,run_directory,*,search_arm):
        self.root,self.run,self.d,self.lad=load(run_directory);self.arm=search_arm;self.moves={}
        if search_arm!=self.d['arms'][1]:raise ValueError('the declared search arm required')
        backfill(self.root,self.lad,search_arm)

    def latency(self):
        path=state_path(self.root,'latency_rule.json')
        if path.exists():return json.loads(path.read_bytes())
        value=latency_rule(self.root,self.d,search_arm=self.arm,log=completions(self.root))
        if value is not None:write_new(path,value)
        return value

    def checkpoint(self,phase):
        """Evaluate this phase's fairness at the next checkpoint of its current window, if reached; record it."""
        rule=self.d['arm_fairness'];state=fairness_state(self.root,phase)
        if len(state['flags'])>len(state['acks']):return 'FLAG_ARM_FAIRNESS'
        ack=state['acks'][-1]if state['acks']else None;start=ack['log_length']if ack else 0
        entries=[e for e in completions(self.root)[start:]if phase_of(self.d,e['pair_ordinal'])==phase]
        pairs=completed_pairs(entries,self.arm);done=[c for c in state['checks']if c['window']==start]
        if pairs<(done[-1]['pairs']if done else 0)+rule['checkpoint_pairs']:return None
        for entry in entries:
            if entry['ordinal']not in self.moves:self.moves[entry['ordinal']]=cell_moves(self.root,entry['ordinal'],rule=rule)
        arms={arm:fairness.pool([self.moves[e['ordinal']]for e in entries if e['arm']==arm])for arm in ('RAW',self.arm)}
        value=fairness.reading(arms['RAW'],arms[self.arm],rule)
        floors=pairs>=rule['min_pairs']and min(value['moves'].values())>=rule['min_moves_per_arm']
        reason=fairness.flag(value,rule,ack)if floors else None
        check=dict(window=start,pairs=pairs//rule['checkpoint_pairs']*rule['checkpoint_pairs'],floors=floors,reason=reason,
            weighted_ratio=value['weighted_ratio'],difference=value['difference'],coverage_difference=value['coverage_difference'],moves=value['moves'])
        _append_line(fairness_directory(self.root,phase)/'checks.jsonl',check)
        if reason is None:return None
        write_new(fairness_directory(self.root,phase)/f"flag-{len(state['flags'])+1}.json",dict(phase=phase,reason=reason,window=start,
            pairs=check['pairs'],acknowledged=ack,reading=value))
        return 'FLAG_ARM_FAIRNESS'

    def __call__(self,packet):
        ordinal=int(Path(packet['path']).stem);arm=ARMS[ordinal%len(ARMS)]
        if arm!=packet['arm']:raise ValueError('completed packet arm differs from its cell')
        cell=cells(self.lad,arm)[ordinal//len(ARMS)]
        _log(self.root,dict(ordinal=ordinal,arm=arm,pair_ordinal=cell['pair_ordinal']))
        rule=self.latency()
        if rule is not None and rule['status']=='VOID_LATENCY':return 'VOID_LATENCY'
        if not self.d['fairness_observable']['value']:return None   # UNMEASURABLE: isolation and interleaving only
        return self.checkpoint(phase_of(self.d,cell['pair_ordinal']))


def _append_line(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('ab')as stream:stream.write(encoded(value)+b'\n');stream.flush();os.fsync(stream.fileno())


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
    neither a transport failure nor breakage. by_server: per server_id (the process that answered, behind a pool;
    'unknown' when the client got no answer to read it from), the responses by status and their select, client and
    server-wall latency quantiles.
    """
    seen=set();sources=Counter();causes=Counter();kinds={};differs=late=wrong_worlds=wrong_archive=wrong_precision=0
    times={k:[]for k in ('select','encode','greedy','client','server_wall','queue','prepare','build','price','reduce')}
    provenance={k:Counter()for k in ('tree_commit','archive_sha256','root_operator','pricer','reply')};servers={};kinds_refused=Counter()
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
        elif source=='greedy_fallback':
            k['fallback']+=1;k['causes'][cause]+=1;causes[cause]+=1
            if cause=='constructor_refused':kinds_refused[refusal_kind((row.get('search')or {}).get('refusal'))]+=1
        times['select'].append(row['select_seconds']);late+=row['select_seconds']>deadline
        times['encode'].append(row['encode_seconds']);times['greedy'].append(row['greedy']['seconds']);search=row['search']
        if search is None:continue
        server_id=search.get('server_id');server_id=server_id if type(server_id)is str and server_id else 'unknown'
        by=servers.setdefault(server_id,dict(statuses=Counter(),select=[],client=[],server_wall=[]))
        by['statuses'][str(search.get('status'))]+=1;by['select'].append(row['select_seconds'])
        if search.get('client_wall_seconds')is not None:times['client'].append(search['client_wall_seconds']);by['client'].append(search['client_wall_seconds'])
        if search.get('server_wall_seconds')is not None:times['server_wall'].append(search['server_wall_seconds']);by['server_wall'].append(search['server_wall_seconds'])
        for name,value in (search.get('timings')or {}).items():
            if name.endswith('_seconds')and name[:-8]in times and value is not None:times[name[:-8]].append(value)
        if search['status']=='COMPLETE':
            wrong_worlds+=worlds is not None and search.get('requested_worlds')!=worlds
            wrong_archive+=archive is not None and search.get('archive_sha256')!=archive
            wrong_precision+=precision is not None and (search.get('pricer')if isinstance(search.get('pricer'),dict)else {}).get('matmul_precision')!=precision
            for name in provenance:
                value=list(served_reply(search))if name=='reply'else search.get(name)
                provenance[name][value if type(value)is not dict and name!='reply'else encoded(value).decode()]+=1
    off=non_home(rows);answered=nh=nh_refused=nh_searched=nh_differs=0
    for row in rows:
        x=row.get('search')if isinstance(row.get('search'),dict)else {}
        if type(x.get('server_id'))is not str or x.get('client_refused')is True:continue
        answered+=1
        if (row.get('battle_id'),row.get('owner'),row.get('decision_index'))in off:
            nh+=1;nh_refused+=row.get('cause')=='constructor_refused'
            if row.get('source')=='search':nh_searched+=1;nh_differs+=row.get('search_differs')is True
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
        provenance={k:sorted(map(str,v))for k,v in provenance.items()},
        constructor_refusal_kinds=dict(kinds_refused.most_common(10)),
        non_home=dict(decisions=nh,answered=answered,share_of_answered=_share(nh,answered),constructor_refused_share=_share(nh_refused,nh),
            override_rate=_share(nh_differs,nh_searched),max_share=NON_HOME_SHARE_MAX),
        by_server={k:dict(responses=sum(v['statuses'].values()),statuses=dict(sorted(v['statuses'].items())),
            latency={t:_quantiles(v[t])for t in ('select','client','server_wall')})for k,v in sorted(servers.items())})


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
    """request_id -> the server's own seconds from arrival to result, from depthn_live_server --telemetry files, for replies
    that were COMPLETE only: any other reply (expired_in_queue reports just its queue wait) leaves the decision censored."""
    out={}
    for path in paths or ():
        for line in Path(path).read_bytes().splitlines():
            row=json.loads(line)
            if (row.get('kind')=='request'and isinstance(row.get('response'),dict)and row['response'].get('status')=='COMPLETE'
                    and type(row.get('server_seconds'))in (int,float)and math.isfinite(row['server_seconds'])):
                out[row['response'].get('request_id')]=row['server_seconds']
    return out


def deadline_coverage(rows,*,reserve,deadlines,minimum,late=None):
    """End-to-end wall (select_seconds) of searched decisions and, per candidate deadline, the share whose wall + reserve
    fits. A timeout's wall is censored at the client deadline: with the server's telemetry showing its reply COMPLETE it is the
    pre-send time plus the server's own seconds, else it is unknown (lower bound counts it as not fitting, upper bound as
    fitting)."""
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


def readout(run_directory,*,arm,limit,codex_raw_directory=None,server_telemetry=None,tree_commit=None,nodes=None):
    root,run,d,lad=load(run_directory);tree=require_tree(run,tree_commit)
    if arm not in d['arms'][1:]or limit not in (d['screen_paired_cells'],d['max_paired_cells']):raise ValueError('declared search arm and cell count required')
    rows={};arms={}
    for name in ('RAW',arm):
        rows[name],arms[name]=arm_summary(root,lad,arm=name,limit=limit,declaration_sha256=run['declaration_sha256'],
            deadline=d['deadline_seconds'],worlds=None if name=='RAW'else d['search'][arm]['worlds'],
            archive=None if name=='RAW'else d['archive_sha256'],precision=None if name=='RAW'else d['matmul_precision'])
    recorded=state_path(root,'latency_rule.json')   # else the Monitor's own window: the effective completion log
    latency=json.loads(recorded.read_bytes())if recorded.exists()else latency_rule(root,d,search_arm=arm,log=effective_log(root,lad,arm))
    telemetry_paths=None if server_telemetry is None else [str(Path(x).resolve())for x in server_telemetry]
    observable=d['fairness_observable']['value']
    fair=dict(observable=d['fairness_observable'],rule=d['arm_fairness'],calibration=d['calibration'],
        accepted_uncalibrated=state_path(root,'accept_uncalibrated.json').exists(),phases={})
    for phase in PHASES[:1]if limit==d['screen_paired_cells']else PHASES:   # the screen readout reads screen-phase state only
        state=fairness_state(root,phase)
        pooled={name:fairness.pool([cell_moves(root,r['cell']['ordinal'],r,d['arm_fairness'])for r in rows[name]
                                    if phase_of(d,r['cell']['pair_ordinal'])==phase])
                for name in rows}
        reading=(fairness.reading(pooled['RAW'],pooled[arm],d['arm_fairness'])if observable else dict(status='UNMEASURABLE',
            basis='structural isolation (dedicated Foul Play cores; servers in their own pods) plus interleaving',
            descriptive_request_to_answer_seconds=fairness.descriptive_waits(pooled['RAW'],pooled[arm])))
        fair['phases'][phase]=dict(reading=reading,flags=state['flags'],acks=state['acks'],void=state['void'],checks=state['checks'])
    held=any(len(v['flags'])>len(v['acks'])for v in fair['phases'].values())
    fault=any(v['void']is not None for v in fair['phases'].values())
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
    if (any(len(v)>1 for v in origin.values())or origin['root_operator']not in ([],[d['root_operator']])
            or origin.get('reply',[])not in ([],[encoded(list(declared_reply(d))).decode()])):breakage.append('search_provenance')
    servers=dict(server_tree_check(d['server_tree_commit'],tree['commit'],run['declaration']),observed=origin['tree_commit'])
    if not servers['ok']or origin['tree_commit']not in ([],[d['server_tree_commit']]):breakage.append('server_tree')
    if (search['non_home']['share_of_answered']or 0.)>NON_HOME_SHARE_MAX:breakage.append('non_home')   # routing not holding
    if search['wrong_worlds']:breakage.append('requested_worlds')
    if search['wrong_archive']:breakage.append('archive')
    if search['wrong_precision']:breakage.append('matmul_precision')
    if search['transport_failure_share']is not None and search['transport_failure_share']>rules['transport_failure_share_max']:
        breakage.append('transport_failures')
    if search['searched_share']is not None and search['searched_share']<rules['searched_share_min']:breakage.append('searched_share')
    fair['observability']=observability(rows['RAW']+rows[arm],observable)
    if fair['observability']['contradictions']:breakage.append('fairness_observability')
    attempted=all(a['not_attempted']==0 for a in arms.values());harm=interval is not None and interval[1]<0;bar=None
    if fault:decision='VOID_FAIRNESS'
    elif void:decision='VOID_LATENCY'
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
        tree=tree,server_tree=servers,nodes=nodes,arm=arm,paired_cells=limit,comparison=d['primary_comparison'],confidence=d['confidence'],
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
    p.add_argument('--cache-directory',required=True);p.add_argument('--warmup-capture')
    args=p.parse_args();asyncio.run(serve(tuple(args.cpu_pair),args.cache_directory,warmup=args.warmup_capture))
