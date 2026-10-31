"""Explicit finite v5 raw-vs-FP dispatch; legacy loaders/defaults stay separate.

This is callable source, not an experiment declaration or launch. The caller
supplies original catalog/archive/seed bank. All planned cells retain their
places and failures, with no replay or outcome-based replacement. A blocked
device thread needs the owned-worker supervisor below, not another game in the
same process. No legacy bridge score for a cap becomes a v5 observed tie.
"""
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time


SCHEMA='pokezero.v5.raw_foulplay_panel.v1'


def _json_default(value):
    if isinstance(value,Path):return str(value)
    raise TypeError('explicit panel JSON value required')


def _sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False,
        default=_json_default).encode()).hexdigest()


class _TransportOnly:
    policy_id='v5-original-owning-actor-routes-all-learner-choices'
    def reset(self):pass
    def select_action(self,*args,**kwargs):raise RuntimeError('legacy policy must never decide a v5 owning action')


def _score(config,game):
    """Only literal actual terminal evidence supplies an observed panel score."""
    if game['capped']:return None
    witness=game.get('terminal_witness')
    if not isinstance(witness,dict):raise ValueError('actual owner terminal witness missing')
    if witness.get('capped',False):raise ValueError('capped terminal cannot become observed outcome')
    winner=witness.get('winner_player');line=witness.get('public_terminal_line')
    if witness.get('status')=='PUBLIC_TIE' and winner is None and line in ('|tie','|tie|'):
        if game.get('winner') is not None or game.get('pokezero_won') or not game.get('tied'):
            raise ValueError('original tie outcome fields disagree')
        return .5
    if witness.get('status')=='PUBLIC_WIN' and winner in ('p1','p2'):
        name=config.pokezero_username if winner==config.pokezero_player else config.foulplay_username
        if line=='|win|'+name:
            if (game.get('winner')!=name or game.get('tied')
                    or game.get('pokezero_won')!=(winner==config.pokezero_player)):
                raise ValueError('original winner outcome fields disagree')
            return float(winner==config.pokezero_player)
    raise ValueError('uncorroborated original terminal remains unknown')


def _recipe(config,factory):
    from . import foulplay_bridge as old
    from .live_protocol_bridge import require_actor_factory
    from .live_protocol_runtime import _require_source,_checkpoint_sha
    if type(config) is not old.ControlledFoulPlayConfig:raise ValueError('original explicit FP configuration required')
    require_actor_factory(config,factory,actor_version='v5')
    old._validate_external_paths(config);_require_source(config,factory.contract)
    if _checkpoint_sha(config.checkpoint)!=factory.contract.checkpoint_sha256:
        raise ValueError('original explicit v5 checkpoint path differs from archived source')
    resolved=json.loads(json.dumps(asdict(config),default=_json_default,allow_nan=False))
    return dict(schema=SCHEMA,actor_version='v5',config=resolved,actor_namespace=factory.contract.namespace,
        archive_sha256=json.loads(factory.metadata)['weights_archive_sha256'],
        checkpoint_sha256=factory.contract.checkpoint_sha256,
        category_vocabulary_sha256=factory.config.category_vocabulary_sha256,
        label_vocabulary_sha256=factory.config.label_vocabulary.sha256,
        deterministic=factory.deterministic,inference_precision=factory.compute_dtype)


async def load_and_run(config,archive,*,catalog,contract,expected_archive_sha256,
        expected_checkpoint_sha256,expected_category_vocabulary_sha256,expected_label_vocabulary_sha256,
        deterministic=True,compute_dtype='float32',enable_v5=False,**run_options):
    """Actual explicit archive load -> owning actor -> original FP game body."""
    if enable_v5 is not True:raise ValueError('explicit v5 FP enablement required')
    from gpu_engine.v5_live_protocol import load_factory
    factory=load_factory(archive,catalog=catalog,contract=contract,
        expected_archive_sha256=expected_archive_sha256,expected_checkpoint_sha256=expected_checkpoint_sha256,
        expected_category_vocabulary_sha256=expected_category_vocabulary_sha256,
        expected_label_vocabulary_sha256=expected_label_vocabulary_sha256,deterministic=deterministic,
        compute_dtype=compute_dtype,enable_v5=True)
    return await run_panel(config,factory,enable_v5=True,**run_options)


async def run_panel(config,factory,*,previous=None,expected_previous_sha256=None,progress_callback=None,
        trajectory_callback=None,enable_v5=False):
    """One original finite declared panel; exceptions remain unknown cells.

    The original low-level runner receives explicit offsets of the ORIGINAL
    config bank. Thus seed, FP seed and owning stream ordinal do not change on
    failure/resume. The old checkpoint loader is never called. Timeout stops
    further attempts until a fresh owned worker resumes not-attempted cells.
    """
    if enable_v5 is not True:raise ValueError('explicit v5 FP enablement required')
    from . import foulplay_bridge as old
    from .observation import ObservationFeatureMasks
    from .showdown import V4_REPLAY_OBSERVATION_SPEC
    recipe=_recipe(config,factory);namespace=_sha(recipe)
    fp=old._per_seed_foulplay_random_seed_schedule(config,count=config.games)
    rows=[dict(offset=i,seed=config.seed_start+i,fp_seed=fp[i],status='NOT_ATTEMPTED',score=None,
        duration_seconds=None,reason=None,source_result_sha256=None,source_game=None) for i in range(config.games)]
    started=time.time()
    if previous is not None:
        if (type(previous) is not dict or _sha(previous)!=expected_previous_sha256
                or previous.get('schema')!=SCHEMA or previous.get('namespace')!=namespace
                or type(previous.get('rows')) is not list or len(previous['rows'])!=len(rows)):
            raise ValueError('same original finite v5 panel declaration required')
        ended=False
        for expected,actual in zip(rows,previous['rows']):
            if any(actual.get(k)!=expected[k] for k in ('offset','seed','fp_seed')):raise ValueError('original panel cell order differs')
            status=actual.get('status')
            if status not in ('NOT_ATTEMPTED','RUNNING','COMPLETE','CAPPED','REFUSED','FAILED') or ended and status!='NOT_ATTEMPTED':
                raise ValueError('original attempted prefix required, no skipped cells')
            ended|=status=='NOT_ATTEMPTED';expected.update(actual)
            game=actual.get('source_game')
            if game is not None and (_sha(game)!=actual.get('source_result_sha256') or game.get('seed')!=expected['seed']):
                raise ValueError('saved actual original panel game changed')
            if status=='COMPLETE' and (game is None or actual.get('score')!=_score(config,game) or game['capped']):
                raise ValueError('saved complete score differs from literal original terminal')
            if status!='COMPLETE' and actual.get('score') is not None:raise ValueError('unknown panel outcome cannot carry a score')
            if status=='RUNNING':expected.update(status='FAILED',score=None,reason='interrupted-original-root-not-replayed')
        started=previous['started_unix']
    masks=dict(factory.contract.feature_masks);masks['opponent_tendency_stats_block']=masks.pop('stats_block')
    masks=ObservationFeatureMasks(**masks)
    def report():
        completed=[row['score'] for row in rows if row['status']=='COMPLETE']
        unknown=len(rows)-len(completed);total=sum(completed)
        return dict(schema=SCHEMA,namespace=namespace,recipe=recipe,started_unix=started,
            rows=[dict(row) for row in rows],counts={s:sum(row['status']==s for row in rows)
                for s in ('NOT_ATTEMPTED','RUNNING','COMPLETE','CAPPED','REFUSED','FAILED')},
            completed_only_score=total/len(completed) if completed else None,
            score_bounds=[total/len(rows),(total+unknown)/len(rows)],unknown_cells=unknown,
            full_B4_qualified=False,physics_qualified=False,strength_qualified=False)
    def publish():
        if progress_callback is not None:progress_callback(report())
    publish()
    for row in rows:
        if row['status']!='NOT_ATTEMPTED':continue
        row['status']='RUNNING';publish();clock=time.monotonic()
        try:
            result=await old._run_controlled_foulplay_games(config,policy=_TransportOnly(),
                policy_id='v5-raw-'+recipe['archive_sha256'][:12],vocab=factory.catalog.vocabulary,
                dex=factory.catalog.dex,observation_spec=V4_REPLAY_OBSERVATION_SPEC,feature_masks=masks,
                checkpoint_sha256=factory.contract.checkpoint_sha256,trajectory_callback=trajectory_callback,
                owning_actor_factory=factory,actor_runtime=factory,actor_version='v5',game_offsets=(row['offset'],))
            if (len(result.games)!=1 or result.games[0].seed!=row['seed']
                    or result.foulplay_random_seed_schedule!=(row['fp_seed'],)):
                raise ValueError('actual original FP game cell/seed differs')
            game=result.games[0];source=game.to_dict()
            row.update(source_result_sha256=_sha(source),source_game=source)
            if game.capped:row.update(status='CAPPED',score=None,reason='original-round-cap')
            else:row.update(status='COMPLETE',score=_score(config,source))
        except Exception as error:
            row.update(status='REFUSED' if isinstance(error,ValueError) else 'FAILED',score=None,
                reason=type(error).__name__+': '+str(error)[:1024])
            if isinstance(error,TimeoutError):
                row['duration_seconds']=time.monotonic()-clock;publish();break
        row['duration_seconds']=time.monotonic()-clock;publish()
    return report()


def supervise_panel(command,*,progress_path,initial_window_seconds,longest_multiplier=5.,
        backstop_seconds=30*24*3600.,poll_seconds=1.,grace_seconds=10.,popen=subprocess.Popen,
        group_exists=None,signal_group=None,sleep=time.sleep):
    """Hard bound a NEW owned worker/group, including blocked device threads.

    Direct async callers retain late-result rejection but cannot preempt JAX.
    Run that caller inside this supervisor for actual stall/backstop enforcement.
    Only the newly created session/process group may receive cleanup signals.
    """
    values=(initial_window_seconds,longest_multiplier,backstop_seconds,poll_seconds,grace_seconds)
    if (os.name!='posix' or any(type(x) not in (int,float) or not 0<x<float('inf') for x in values)
            or longest_multiplier<3 or backstop_seconds<10*initial_window_seconds):
        raise ValueError('finite positive owned-group watchdog and distant backstop required')
    signal_group=os.killpg if signal_group is None else signal_group
    if group_exists is None:
        def group_exists(group):
            try:os.killpg(group,0);return True
            except ProcessLookupError:return False
    process=popen(command,start_new_session=True);group=process.pid
    if type(group) is not int or group<=0 or group==os.getpgrp():raise ValueError('new isolated owned worker group required')
    started=last=time.monotonic();completed=0;longest=0.;reason=None
    failure=None;namespace=None
    try:
        while process.poll() is None:
            path=Path(progress_path)
            if path.is_file():
                current=json.loads(path.read_bytes())
                if current['schema']!=SCHEMA or namespace is not None and current['namespace']!=namespace:
                    raise ValueError('original finite panel progress namespace changed')
                namespace=current['namespace'];done=[row for row in current['rows'] if row['status']=='COMPLETE']
                if len(done)<completed:raise ValueError('completed panel progress regressed')
                if len(done)>completed:
                    completed=len(done);last=time.monotonic();longest=max(longest,max(row['duration_seconds'] for row in done))
                if time.time()-current['started_unix']>backstop_seconds:reason='DISTANT_BACKSTOP'
            now=time.monotonic()
            if now-started>backstop_seconds:reason='DISTANT_BACKSTOP'
            elif reason is None and now-last>max(initial_window_seconds,longest_multiplier*longest):reason='NO_COMPLETED_GAME_WITHIN_ADAPTIVE_WINDOW'
            if reason:break
            sleep(poll_seconds)
    except BaseException as error:failure=error
    finally:
        if process.poll() is None or group_exists(group):
            if group_exists(group):signal_group(group,signal.SIGTERM)
            try:process.wait(timeout=grace_seconds)
            except subprocess.TimeoutExpired:
                if group_exists(group):signal_group(group,signal.SIGKILL)
                process.wait(timeout=grace_seconds)
            if group_exists(group):signal_group(group,signal.SIGKILL)
            until=time.monotonic()+grace_seconds
            while group_exists(group) and time.monotonic()<until:sleep(min(poll_seconds,.2))
    if failure is not None:raise failure
    return dict(exit_code=process.returncode,owned_pid=process.pid,owned_group=group,
        cleanup_confirmed=not group_exists(group),stop_reason=reason)
