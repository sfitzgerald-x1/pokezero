"""Fixed B3 owning-encoder study; preparation is not game authority.

Only the explicit async runner invokes the existing controlled benchmark.
Missing, failed and capped cells remain unknown on the original denominator.
No result from this module automatically qualifies B3 or an equivalence claim.
"""
from dataclasses import asdict
from collections import deque
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import fmean, stdev
import tempfile

from .live_protocol_actor import ActorContract


COUNT = 500
SEED_START = 30100000
FP_SEED_START = 40100000
ORDER = (('python-owning', 'p1'), ('jax-owning', 'p1'),
         ('jax-owning', 'p2'), ('python-owning', 'p2'))
SPEC_SCHEMA = 'pokezero.B3-owning-study-prespec.v1'
LEDGER_SCHEMA = 'pokezero.B3-owning-study-ledger.v1'
VISIBILITY = 'ACTUAL_CAPTURED_OWNING_LINES_AND_IMMUTABLE_CONSUMED_PYTHON_TENSORS'
RUNTIME_KEYS = ('checkpoint', 'showdown_root', 'foulplay_root', 'foulplay_python', 'node_binary')
# Derived from hash-bound compact saved configs/receipts before choosing B3
# seeds. A4's screen reused the original A2 cohort; it collected no new games.
CONSUMED = (
    ('a2-mirror-20261029-5e4338a880d9', 256,
     '5361a3715955146c0f08a81b543e1a9437adafc3db0c76b63ddc5a1e817f0f9f'),
    ('a2-corrected-owning-warmstart-20261030-r1', 4096,
     'ab063a77bdb34390b41d0e8b1e16438581ed935666b1e3ff5471343cdd284198'))


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def _failure_diagnostics(error):
    """Retain bounded failure evidence without frame locals or source payloads.

    The runner consumes the exception instead of letting Python print it, so
    type alone permanently loses the reason for a failed non-resumable attempt.
    Keep the innermost code locations even when the stack exceeds the bound.
    """
    try:
        message = str(error)
    except BaseException:
        message = '<exception message unavailable>'
    frames = deque(maxlen=16)
    count = 0
    trace = error.__traceback__
    while trace is not None:
        code = trace.tb_frame.f_code
        frames.append(dict(file=code.co_filename[-512:], line=trace.tb_lineno,
                           function=code.co_name[:256]))
        count += 1
        trace = trace.tb_next
    return dict(error_message=message[:4096], error_message_truncated=len(message) > 4096,
        error_traceback=list(frames), error_traceback_truncated=count > len(frames))


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def prespec_file_sha256(spec):
    return hashlib.sha256(canonical(spec) + b'\n').hexdigest()


def _hex(value, size=64):
    return type(value) is str and len(value) == size and all(c in '0123456789abcdef' for c in value)


def _reference(value):
    require(type(value) is dict and set(value) == {'path', 'sha256'}
        and type(value['path']) is str and value['path'] and _hex(value['sha256']),
        'hash-bound artifact reference required')
    return dict(value)


def read_bound(reference):
    reference = _reference(reference)
    payload = Path(reference['path']).read_bytes()
    require(hashlib.sha256(payload).hexdigest() == reference['sha256'], 'artifact hash changed')
    return json.loads(payload)


@lru_cache(maxsize=1)
def _seed_audit_bytes():
    from .mirror_value_probe import cohort
    consumed = set()
    for root, count, expected in CONSUMED:
        rows = cohort(root, count)
        require(digest(rows) == expected, 'consumed seed derivation changed')
        consumed.update(row['seed'] for row in rows)
    proposed = set(range(SEED_START, SEED_START + COUNT)) | set(range(FP_SEED_START, FP_SEED_START + COUNT))
    require(not proposed.intersection(consumed) and len(proposed) == 2 * COUNT,
            'B3 seeds overlap consumed A2/A4 schedules')
    return canonical(dict(consumed_cohorts=[dict(root=r, games=n, roster_sha256=h) for r, n, h in CONSUMED],
        consumed_unique_seeds=len(consumed), consumed_seed_sha256=digest(sorted(consumed)),
        overlap=[], new_games=0))


def seed_audit():
    return json.loads(_seed_audit_bytes())


def make_prespec(*, study_id, runtime, admission, max_decision_rounds=250,
                 owning_terminal_timeout_seconds=5):
    require(type(study_id) is str and study_id and len(study_id) <= 128
        and all(c.isalnum() or c in '-_' for c in study_id), 'canonical study ID required')
    require(type(runtime) is dict and set(runtime) == set(RUNTIME_KEYS)
        and all(type(runtime[k]) is str and runtime[k] for k in RUNTIME_KEYS), 'explicit runtime paths required')
    require(type(max_decision_rounds) is int and 0 < max_decision_rounds <= 4096,
            'fixed finite round cap required')
    require(type(owning_terminal_timeout_seconds) in (int, float)
        and math.isfinite(owning_terminal_timeout_seconds) and 0 < owning_terminal_timeout_seconds <= 30,
        'bounded terminal timeout required')
    require(type(admission) is dict and set(admission) == {'comparison', 'scientific_review',
        'source_commit', 'selection_sha256', 'selection_metadata_sha256', 'collection_sha256'},
        'exact B2 comparison/review/source/selection bindings required')
    for key in ('selection_sha256', 'selection_metadata_sha256', 'collection_sha256'):
        require(_hex(admission[key]), 'exact B2 identity required')
    require(_hex(admission['source_commit'], 40), 'exact B2 source commit required')
    admission = dict(admission, comparison=_reference(admission['comparison']),
                     scientific_review=_reference(admission['scientific_review']))
    return dict(schema=SPEC_SCHEMA, study_id=study_id, runtime=dict(runtime), admission=admission,
        champion_contract=json.loads(canonical(asdict(ActorContract.champion()))),
        seed_start=SEED_START, foulplay_seed_start=FP_SEED_START, seed_count=COUNT,
        invocations=[dict(ordinal=i, encoder_kind=k, seat=s, games=COUNT) for i, (k, s) in enumerate(ORDER)],
        seed_audit=seed_audit(), max_decision_rounds=max_decision_rounds,
        owning_terminal_timeout_seconds=owning_terminal_timeout_seconds,
        controls=dict(policy_mode='raw', capture_driver='checkpoint', format_id='gen3randombattle',
            opponent_policy_mode='foul-play', device='cpu', temperature=1., belief_set_source=True,
            opponent_legal_mask_mode='hidden', search_time_ms=1000, foulplay_mcts_iterations=None,
            opponent_crash_retries=0, allow_search_fallback=False),
        precision=dict(cpu_only=True, torch_dtype='float32', mha_fastpath=False, tf32=False,
            torch_matmul_precision='highest', jax_enable_x64=False,
            jax_matmul_precision='highest', torch_threads=1, torch_interop_threads=1),
        execution=dict(invocation_count=4, games_per_invocation=500,
            failure_policy='stop_study_on_first_failure', resume=False, retries=0,
            replacement_games=0, model_and_factory='fresh_per_invocation'),
        analysis=dict(primary='win_rate', secondary='score_rate', contrast='jax-minus-python',
            cluster='battle_seed_both_seats_together', confidence=.90,
            interval='two_sided_paired_t_499df_full_normal_roster_only',
            zero_variance='interval_unavailable', equivalence_margin=None,
            unknown_cell_bounds=[0., 1.], automatic_B3_qualification=False))


def validate_prespec(spec):
    require(type(spec) is dict and spec.get('schema') == SPEC_SCHEMA, 'B3 prespec required')
    expected = make_prespec(study_id=spec['study_id'], runtime=spec['runtime'], admission=spec['admission'],
        max_decision_rounds=spec['max_decision_rounds'],
        owning_terminal_timeout_seconds=spec['owning_terminal_timeout_seconds'])
    require(canonical(spec) == canonical(expected), 'fixed B3 prespec drift')


def validate_admission(spec):
    """Objective's actual-owning 173-column/FP32 gate, not broader flags."""
    binding = spec['admission']
    document = read_bound(binding['comparison'])
    require(document.get('schema') == 'pokezero.B2-actual-saved-wire-comparison.v1'
        and document.get('visibility') == VISIBILITY and document.get('actual_captured_wire') is True
        and document.get('original_reference_values_compared') is True,
        'actual owning lines and original consumed references required; projected tables forbidden')
    from .actual_saved_wire import (PRODUCER_SOURCE, PRODUCER_FILES,
        PRODUCER_COHORT_FILE_SHA, PRODUCER_ROSTER_DIGEST)
    require(document.get('producer_source') == PRODUCER_SOURCE
        and document.get('producer_scientific_file_sha256') == PRODUCER_FILES
        and document.get('cohort_file_sha256') == PRODUCER_COHORT_FILE_SHA
        and document.get('canonical_roster_digest') == PRODUCER_ROSTER_DIGEST,
        'original consumed-reference producer/cohort lineage required')
    for key in ('selection_sha256', 'selection_metadata_sha256', 'collection_sha256'):
        require(document.get(key) == binding[key], 'B2 selection/collection binding differs')
    from .recorded_mirror_roots import CHAMPION_CONTRACT
    require(document['runtime']['contract'] == CHAMPION_CONTRACT,
        'exact champion B2 runtime contract required')
    metrics = document['metrics']
    n = metrics.get('expected_views')
    require(type(n) is int and n >= 10000 and all(type(metrics.get(k)) is int and metrics[k] == n for k in
        ('views', 'processed_views', 'model_forward_views'))
        and all(metrics.get(k) is True for k in ('complete_view_count', 'complete_roster_processed', 'model_forward_complete'))
        and all(type(metrics.get(k)) is int and metrics[k] == 0 for k in
            ('refused_input_views', 'refused_forward_views', 'unprocessed_views')),
        'complete fixed B2 roster and forward denominator without refusals required')
    require(metrics.get('schema') == 'pokezero.B2-actual-saved-wire-metrics.v1'
        and metrics.get('visibility') == VISIBILITY and metrics.get('original_reference_values_compared') is True
        and metrics.get('actual_captured_wire') is True and metrics.get('captured_player_wire') is True,
        'actual original-reference B2 metric identity required')
    for family, width in (('numeric_columns', 132), ('categorical_columns', 41)):
        rows = metrics[family]
        require(type(rows) is list and len(rows) == width
            and all(type(r.get('index')) is int for r in rows)
            and [r.get('index') for r in rows] == list(range(width)), 'all 173 ordered columns required')
        for row in rows:
            require(all(type(row.get(k)) is int and row[k] == n * 23
                for k in ('cells', 'expected_cells', 'compared_cells'))
                and all(type(row.get(k)) is int and row[k] == 0 for k in
                    ('mismatched_cells', 'refused_cells', 'unprocessed_cells')), 'B2 column has unknown/mismatched cells')
            if family == 'numeric_columns':
                require(type(row.get('maximum_absolute_error')) in (int, float)
                    and math.isfinite(row['maximum_absolute_error'])
                    and 0 <= row['maximum_absolute_error'] <= 1e-6, 'B2 numeric tolerance failed')
    require(set(metrics['other_tensor_mismatched_cells']) == {'token_type_ids', 'attention_mask', 'legal_action_mask'}
        and all(type(v) is int and v == 0 for v in metrics['other_tensor_mismatched_cells'].values()),
        'B2 tensor masks differ')
    for name, width in (('token_type_ids',23), ('attention_mask',23), ('legal_action_mask',9)):
        require(all(type(metrics.get(key, {}).get(name)) is int and metrics[key][name] == n*width
            for key in ('other_tensor_cells','other_tensor_expected_cells'))
            and type(metrics.get('other_tensor_refused_cells', {}).get(name)) is int
            and metrics['other_tensor_refused_cells'][name] == 0, 'B2 tensor denominator differs')
    for key in ('policy_argmax_agreement', 'legal_policy_argmax_agreement'):
        require(type(metrics.get(key)) in (int, float) and .999 <= metrics[key] <= 1, 'B2 FP32 argmax gate failed')
    require(type(metrics.get('maximum_value_difference')) in (int, float)
        and 0 <= metrics['maximum_value_difference'] < 1e-4, 'B2 FP32 value gate failed')
    review = read_bound(binding['scientific_review'])
    expected = dict(comparison_sha256=binding['comparison']['sha256'], reviewed_commit=binding['source_commit'],
        checkpoint_sha256=ActorContract.champion().checkpoint_sha256,
        **{k: binding[k] for k in ('selection_sha256', 'selection_metadata_sha256', 'collection_sha256')})
    require(review.get('verdict') == 'CLEAR' and all(review.get(k) == v for k, v in expected.items()),
            'independent B2 result review must bind exact artifact/source/checkpoint/selection')
    return dict(comparison_sha256=binding['comparison']['sha256'],
        review_sha256=binding['scientific_review']['sha256'],
        full_encoder_admitted=document.get('full_encoder_admitted'),
        raw_bridge_chunk_bytes_captured=document.get('raw_bridge_chunk_bytes_captured'))


def planned_cells():
    return [dict(invocation=i, encoder_kind=kind, seat=seat, seed=SEED_START + offset,
        foulplay_seed=FP_SEED_START + offset, status='NOT_ATTEMPTED', score=None, win=None)
        for i, (kind, seat) in enumerate(ORDER) for offset in range(COUNT)]


class CappedGame(ValueError):
    """An observed cap is unknown, including in synthetic benchmark controls."""


def checked_game(row, spec, invocation, offset):
    from .foulplay_bridge import DEFAULT_BATTLE_ID_PREFIX
    kind, seat = ORDER[invocation]
    seed = SEED_START + offset
    require(type(row) is dict and type(row.get('seed')) is int and row['seed'] == seed
        and row.get('encoder_kind') == kind and row.get('battle_id') == f'{DEFAULT_BATTLE_ID_PREFIX}-{seed}',
        'game identity differs from planned cell')
    if row.get('capped') is True:
        raise CappedGame('capped game is unknown, never a draw')
    require(row.get('capped') is False, 'explicit uncapped game required')
    players = {seat: 'PokeZeroBot', 'p2' if seat == 'p1' else 'p1': 'FoulPlayBot'}
    witness, provenance = row['terminal_witness'], row['encoder_provenance']
    actor = provenance['actor_receipt']
    require(provenance.get('encoder_kind') == kind and provenance.get('status') == 'CLOSED_UNQUALIFIED'
        and provenance.get('input_source') == 'ACTUAL_GETPLAYERSTREAMS_OWNER_LINES_UNQUALIFIED'
        and provenance.get('owner_terminal_timeout_seconds') == spec['owning_terminal_timeout_seconds']
        and provenance.get('actor_namespace') == ActorContract.champion().namespace
        and provenance.get('actual_owner_terminal_seen') is True
        and provenance.get('completed_terminal_streams') == ['omniscient', 'p1', 'p2']
        and provenance.get('bridge_terminal_marker') is True and provenance.get('bridge_started_witness') is True
        and actor.get('status') == 'TERMINAL_UNQUALIFIED' and actor.get('battle_id') == row['battle_id']
        and actor.get('owner') == seat and actor.get('terminal_seen') is True
        and actor.get('error') is None and actor.get('close_error') is None,
        'actual owning terminal/clean provider closure required')
    require(witness.get('schema') == 'pokezero.controlled-terminal-witness.v1'
        and witness.get('players') == players and witness.get('capped') is False
        and witness.get('max_decision_rounds') == spec['max_decision_rounds']
        and type(row.get('decision_rounds')) is int and 0 <= row['decision_rounds'] <= spec['max_decision_rounds']
        and witness.get('decision_rounds') == row['decision_rounds'], 'terminal witness binding differs')
    winner = witness.get('winner_player')
    if winner is None:
        require(witness.get('status') == 'PUBLIC_TIE' and witness.get('public_terminal_line') == '|tie'
            and row.get('winner') is None and row.get('tied') is True and row.get('pokezero_won') is False,
            'genuine owning draw required')
        score, win = .5, 0
    else:
        require(winner in players and witness.get('status') == 'PUBLIC_WIN'
            and witness.get('public_terminal_line') == '|win|' + players[winner]
            and row.get('winner') == players[winner] and row.get('tied') is False
            and row.get('pokezero_won') is (winner == seat), 'owning win witness differs')
        score = win = int(winner == seat)
    lifecycle = actor.get('terminal_lifecycle', {})
    line = witness['public_terminal_line']
    require(_hex(actor.get('prefix_sha256')) and lifecycle.get('consumed') is True
        and lifecycle.get('kind') == line.split('|')[1]
        and lifecycle.get('line_sha256') == hashlib.sha256(line.encode()).hexdigest()
        and lifecycle.get('scope') == 'OWNING_HOST_LIFECYCLE_NOT_A_DEVICE_FEATURE',
        'owning actor terminal lifecycle differs from public witness')
    decisions = row.get('pokezero_decisions')
    require(type(decisions) is int and decisions >= 0
        and row.get('pokezero_decision_players') == [seat] * decisions
        and row.get('pokezero_submitted_choice_players') == [seat] * decisions
        and type(row.get('pokezero_score')) in (int, float)
        and row['pokezero_score'] == score, 'actual decision/submission seat or score differs')
    return json.loads(canonical(dict(status='COMPLETE', score=score, win=win,
        terminal_witness=witness, encoder_provenance=provenance,
        game_sha256=digest(row), pokezero_decisions=decisions)))


def summarize(spec, ledger):
    validate_prespec(spec)
    require(ledger.get('schema') == LEDGER_SCHEMA and ledger.get('prespec_sha256') == prespec_file_sha256(spec)
        and ledger.get('status') in ('PREPARING','RUNNING','FAILED','COMPLETE'), 'bound study ledger required')
    require(len(ledger['invocations']) == 4 and all(all(actual.get(k) == expected[k]
        for k in ('ordinal','encoder_kind','seat','games'))
        for actual, expected in zip(ledger['invocations'], spec['invocations'])), 'invocation roster differs')
    expected = planned_cells()
    cells = ledger['cells']
    require(len(cells) == len(expected), 'every planned cell must remain in ledger')
    for actual, planned in zip(cells, expected):
        require(all(actual.get(k) == planned[k] for k in ('invocation', 'encoder_kind', 'seat', 'seed', 'foulplay_seed')),
                'ledger roster order/identity differs')
        require(actual['status'] in ('COMPLETE', 'FAILED', 'CAPPED', 'NOT_ATTEMPTED'), 'known ledger status required')
        if actual['status'] == 'COMPLETE':
            require(type(actual.get('score')) in (int, float) and actual['score'] in (0, .5, 1)
                and type(actual.get('win')) is int and actual['win'] == int(actual['score'] == 1), 'valid observed score required')
            witness = actual['terminal_witness']
            row = dict(seed=actual['seed'], encoder_kind=actual['encoder_kind'],
                battle_id=actual['encoder_provenance']['actor_receipt']['battle_id'],
                capped=False, terminal_witness=witness, encoder_provenance=actual['encoder_provenance'],
                decision_rounds=witness['decision_rounds'],
                winner=witness['players'].get(witness.get('winner_player')),
                tied=actual['score'] == .5, pokezero_won=actual['win'] == 1,
                pokezero_score=actual['score'], pokezero_decisions=actual['pokezero_decisions'],
                pokezero_decision_players=[actual['seat']]*actual['pokezero_decisions'],
                pokezero_submitted_choice_players=[actual['seat']]*actual['pokezero_decisions'])
            checked_game(row, spec, actual['invocation'], actual['seed']-SEED_START)
            require(_hex(actual.get('game_sha256')), 'observed game hash required')
        else:
            require(actual.get('score') is None and actual.get('win') is None, 'unknown cell cannot carry a score')
    normal = ledger['status'] == 'COMPLETE' and type(ledger.get('actual_exit')) is int and ledger['actual_exit'] == 0 \
        and ledger.get('failure') is None and all(i['status'] == 'COMPLETE' for i in ledger['invocations'])
    arms = {}
    for kind in ('python-owning', 'jax-owning'):
        rows = [r for r in cells if r['encoder_kind'] == kind]
        known = [r for r in rows if r['status'] == 'COMPLETE']
        arms[kind] = dict(planned=1000, known=len(known), unknown=1000-len(known),
            statuses={s: sum(r['status'] == s for r in rows) for s in ('COMPLETE','FAILED','CAPPED','NOT_ATTEMPTED')},
            **{metric: dict(known_rate=fmean(r[field] for r in known) if known else None,
                full_roster_bounds=[sum(r[field] for r in known)/1000,
                    (sum(r[field] for r in known)+1000-len(known))/1000])
               for metric, field in (('win_rate', 'win'), ('score_rate', 'score'))})
    indexed = {(r['encoder_kind'], r['seat'], r['seed']): r for r in cells}
    contrasts = {}
    for metric, field in (('win_rate', 'win'), ('score_rate', 'score')):
        lower, upper = [], []
        for seed in range(SEED_START, SEED_START+COUNT):
            bounds = {}
            for kind in ('python-owning', 'jax-owning'):
                rows = [indexed[kind, seat, seed] for seat in ('p1','p2')]
                bounds[kind] = [sum(r[field] if r['status']=='COMPLETE' else endpoint for r in rows)/2
                                for endpoint in (0,1)]
            lower.append(bounds['jax-owning'][0]-bounds['python-owning'][1])
            upper.append(bounds['jax-owning'][1]-bounds['python-owning'][0])
        complete = lower == upper
        interval = None
        if normal and complete:
            standard_error = stdev(lower)/math.sqrt(COUNT)
            if standard_error:
                from scipy.stats import t
                radius = float(t.ppf(.95, COUNT-1))*standard_error
                interval = [fmean(lower)-radius, fmean(lower)+radius]
        contrasts[metric] = dict(full_roster_bounds=[fmean(lower), fmean(upper)],
            point_estimate=fmean(lower) if complete else None, interval90=interval,
            interval_status=('APPROXIMATE_PAIRED_T_499DF' if interval is not None else
                'DEGENERATE_VARIANCE' if normal and complete else 'UNAVAILABLE_INCOMPLETE_OR_ABNORMAL'),
            seed_clusters=COUNT, complete_roster=complete)
    return dict(arms=arms, paired_jax_minus_python=contrasts, B3_qualified=False,
        equivalence_claim=False, interpretation='Unknown bounds are identification bounds, not confidence intervals; no automatic qualification.')


def _sync_directory(path):
    directory_fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def atomic_json(path, document):
    """Publish durable bytes and name: file fsync, replace, directory fsync."""
    path = Path(path)
    fd, name = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(canonical(document) + b'\n'); stream.flush(); os.fsync(stream.fileno())
        os.replace(name, path)
        _sync_directory(path.parent)
    finally:
        Path(name).unlink(missing_ok=True)


def _config(spec, ordinal):
    from .foulplay_bridge import ControlledFoulPlayConfig
    kind, seat = ORDER[ordinal]
    runtime = dict(spec['runtime'])
    for key in RUNTIME_KEYS[:-1]:
        runtime[key] = Path(runtime[key])
    return ControlledFoulPlayConfig(**runtime, **spec['controls'], encoder_kind=kind,
        pokezero_player=seat, games=COUNT, seed_start=SEED_START, foulplay_random_seed=FP_SEED_START,
        max_decision_rounds=spec['max_decision_rounds'],
        owning_terminal_timeout_seconds=spec['owning_terminal_timeout_seconds'])


def _configure_precision():
    for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS',
                 'VECLIB_MAXIMUM_THREADS','POKEZERO_TORCH_NUM_THREADS','POKEZERO_TORCH_NUM_INTEROP_THREADS'):
        os.environ[name] = '1'
    os.environ['JAX_PLATFORMS'] = 'cpu'
    from .neural_policy import require_torch
    torch = require_torch()
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    import jax
    jax.config.update('jax_enable_x64', False)
    jax.config.update('jax_default_matmul_precision', 'highest')
    require(all(d.platform == 'cpu' for d in jax.devices()), 'CPU-only study required')
    require(torch.get_num_threads() == 1 and torch.get_num_interop_threads() == 1,
            'matched one-thread CPU runtime required')
    return dict(torch_version=torch.__version__, jax_version=jax.__version__,
        torch_threads=torch.get_num_threads(), torch_interop_threads=torch.get_num_interop_threads(),
        mha_fastpath=torch.backends.mha.get_fastpath_enabled(),
        tf32=bool(torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32),
        torch_matmul_precision=torch.get_float32_matmul_precision(),
        jax_enable_x64=jax.config.jax_enable_x64,
        jax_matmul_precision=jax.config.jax_default_matmul_precision,
        devices=[str(device) for device in jax.devices()])


async def run_study(spec, output, *, benchmark=None, factory_builder=None, configure=None):
    """One non-resumable four-invocation attempt; first failure stops the study.

    Injectable dependencies support synthetic source tests. Production defaults
    use only the existing controlled benchmark/factory. No retry or refill.
    """
    validate_prespec(spec)
    spec = json.loads(canonical(spec))
    output = Path(output)
    output.mkdir(exist_ok=False)
    _sync_directory(output.parent)
    atomic_json(output/'prespec.json', spec)
    ledger = dict(schema=LEDGER_SCHEMA, prespec_sha256=prespec_file_sha256(spec), status='PREPARING',
        cells=planned_cells(), invocations=[dict(item, status='NOT_ATTEMPTED') for item in spec['invocations']],
        failure=None, actual_exit=None, B3_qualified=False)
    def publish():
        ledger['recorded_games'] = sum(row['status'] == 'COMPLETE' for row in ledger['cells'])
        atomic_json(output/'ledger.json', ledger)
    publish()
    ordinal, completed, stage = 0, 0, 'admission'
    try:
        ledger['admission'] = validate_admission(spec)
        stage = 'precision'
        ledger['runtime_precision'] = (configure or _configure_precision)()
        if benchmark is None:
            from .foulplay_bridge import run_controlled_foulplay_benchmark
            benchmark = run_controlled_foulplay_benchmark
        if factory_builder is None:
            from .live_protocol_runtime import champion_actor_factory
            factory_builder = champion_actor_factory
        for ordinal in range(len(ORDER)):
            completed, stage = 0, 'factory'
            config = _config(spec, ordinal)
            ledger['status'] = 'RUNNING'
            ledger['invocations'][ordinal]['status'] = 'RUNNING'
            publish()
            factory = factory_builder(config)
            def progress(result):
                nonlocal completed
                document = result.to_dict()
                require(document.get('encoder_kind') == ORDER[ordinal][0]
                    and document.get('pokezero_player') == ORDER[ordinal][1]
                    and document.get('checkpoint_sha256') == ActorContract.champion().checkpoint_sha256
                    and document.get('seed_start') == SEED_START and document.get('games') == COUNT,
                    'benchmark summary binding differs')
                rows = document['game_results']
                require(type(rows) is list and completed <= len(rows) <= COUNT
                    and document.get('completed_games') == len(rows), 'chronological cumulative prefix required')
                require(document.get('policy_mode') == 'raw' and document.get('format_id') == 'gen3randombattle'
                    and document.get('foulplay_random_seed_schedule', {}).get('seeds') ==
                        list(range(FP_SEED_START, FP_SEED_START+len(rows)))
                    and all(type(document.get('execution_integrity', {}).get(k)) is int
                        and document['execution_integrity'][k] == 0 for k in ('retries','errors',
                            'refusal_records','refusal_recorder_instrument_errors','refusal_records_unrowed')),
                    'benchmark opponent seed schedule or execution integrity differs')
                for offset, row in enumerate(rows):
                    checked = checked_game(row, spec, ordinal, offset)
                    cell = ledger['cells'][ordinal*COUNT+offset]
                    if offset < completed:
                        require(cell['game_sha256'] == checked['game_sha256'], 'completed prefix changed')
                    else:
                        cell.update(checked)
                        completed = offset+1
                        publish()
            stage = 'benchmark'
            result = await benchmark(config, owning_actor_factory=factory, progress_callback=progress)
            progress(result)
            require(completed == COUNT, 'benchmark returned an incomplete invocation')
            ledger['invocations'][ordinal]['status'] = 'COMPLETE'
            publish()
        ledger['status'], ledger['actual_exit'] = 'COMPLETE', 0
    except BaseException as error:
        ledger['status'], ledger['actual_exit'] = 'FAILED', 1
        ledger['invocations'][ordinal]['status'] = 'FAILED'
        failure = dict(invocation=ordinal, stage=stage, error_type=type(error).__name__,
            **_failure_diagnostics(error),
            owning_encoder_receipt=getattr(error, 'owning_encoder_receipt', None))
        if stage == 'benchmark' and completed < COUNT:
            cell = ledger['cells'][ordinal*COUNT+completed]
            cell['status'] = 'CAPPED' if isinstance(error, CappedGame) or (
                isinstance(error, RuntimeError) and failure['error_message'] ==
                'owning game round cap without actual owner terminal') else 'FAILED'
            failure['seed'] = cell['seed']
        if isinstance(error, PermissionError):
            failure['missing_permission'] = dict(errno=error.errno, filename=error.filename)
        ledger['failure'] = failure
    ledger['readout'] = summarize(spec, ledger)
    publish()
    return ledger
