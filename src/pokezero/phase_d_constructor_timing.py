"""Finite, quarantined constructor timing on the original twenty saved roots.

This does not price actions, load an evaluator, play games or qualify coverage.
Each world budget has one explicit warmup pass and one measured pass. Fresh
constructor request state shares only immutable compiled encoder functions.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace

from .phase_d_ladder_admission import timing_roster, refusal_kind
from .phase_d_ladder_panel import encoded, read_bound, write_new
from .phase_d_saved_roots import address, band


SCHEMA = 'pokezero.phase-d.r7-timing.v1'
ROSTER_SHA256 = '16302b49eb8aa03c56446ca921aab423780374f20336a59b1eb1618d0048c7e9'
BUDGETS = (8, 16)
PASSES = ('warmup', 'warm')


def validate_config(config):
    expected = dict(schema=SCHEMA, rung='R7', worlds=32, seed=20261103,
                    lanes=4096, branch_cap=262144, compute_dtype='float32',
                    saved_root_admission=True)
    if any(type(config.get(key)) is not type(value) or config[key] != value
           for key, value in expected.items()):
        raise ValueError('Original R7 numerical configuration required')
    warm = config['warm_constructor']
    if (warm['worlds'] != list(BUDGETS) or warm['passes'] != list(PASSES)
            or warm['roots_per_pass'] != 20
            or warm['before_full_request_timing'] is not True
            or config['roster']['sha256'] != ROSTER_SHA256):
        raise ValueError('Fixed twenty-root, two-budget, two-pass protocol required')
    return config


def p95(values):
    """Missing slots retain infinity; never use a survivor denominator."""
    if len(values) != 20:
        raise ValueError('Original twenty timing slots required')
    known = []
    for value in values:
        if value is None:
            known.append(math.inf)
        elif type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError('Finite nonnegative literal time required')
        else:
            known.append(value)
    result = sorted(known)[18]
    return result if math.isfinite(result) else None


def summarize(roster, records, *, worlds, phase):
    if (len(roster) != 20 or type(worlds) is not int
            or worlds not in BUDGETS or phase not in PASSES):
        raise ValueError('Fixed constructor timing condition required')
    expected = {tuple(address(row)): i for i, row in enumerate(roster)}
    if len(expected) != 20:
        raise ValueError('Duplicate original root address')
    found = {}
    for row in records:
        key = tuple(row['address'])
        if (key not in expected or key in found or type(row['ordinal']) is not int
                or row['ordinal'] != expected[key] or type(row['worlds']) is not int
                or row['worlds'] != worlds or row['phase'] != phase
                or row['turn'] != roster[expected[key]]['turn']):
            raise ValueError('Foreign, duplicate or changed timing condition')
        seconds = row['prepare_seconds']
        if seconds is not None:
            if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds < 0:
                raise ValueError('Finite nonnegative literal time required')
        count = row['constructed_worlds']
        if count is not None and (type(count) is not int or not 0 <= count <= worlds):
            raise ValueError('Literal returned-world count required')
        if row['status'] not in ('COMPLETE', 'REFUSED', 'INTERRUPTED_UNKNOWN'):
            raise ValueError('Original disposition required')
        if row['status'] == 'COMPLETE' and (count != worlds or seconds is None):
            raise ValueError('Complete constructor result must retain all worlds and time')
        if row['status'] == 'INTERRUPTED_UNKNOWN' and (count is not None or seconds is not None):
            raise ValueError('Interrupted attempt cannot invent a world count or time')
        if type(row['warm_context_qualified']) is not bool:
            raise ValueError('Explicit warm-context qualification required')
        found[key] = row
    attempts = []
    full = []
    statuses = Counter()
    refusals = Counter()
    by_band = {}
    qualified = 0
    for source in roster:
        row = found.get(tuple(address(source)))
        counts = by_band.setdefault(str(band(source['turn'])),
                                    dict(requested=0, returned=0, fully_constructed=0,
                                         qualified_times=0))
        counts['requested'] += 1
        if row is None:
            statuses['UNAVAILABLE'] += 1
            attempts.append(None)
            full.append(None)
            continue
        statuses[row['status']] += 1
        counts['returned'] += row['prepare_seconds'] is not None
        complete = row['status'] == 'COMPLETE'
        counts['fully_constructed'] += complete
        valid_time = (phase == 'warmup' or row['warm_context_qualified'])
        seconds = row['prepare_seconds'] if valid_time else None
        counts['qualified_times'] += seconds is not None
        qualified += seconds is not None
        attempts.append(seconds)
        full.append(seconds if complete else None)
        kinds = [refusal_kind(text) for text in row['refusals']]
        for kind in set(kinds):
            refusals[kind] += 1
    attempt_p95, complete_p95 = p95(attempts), p95(full)
    return dict(requested_roots=20, worlds=worlds, phase=phase,
                dispositions=dict(statuses), by_turn_band=by_band,
                refusal_roots_by_kind=dict(refusals), qualified_attempt_times=qualified,
                full_roster_attempt_p95_seconds=attempt_p95,
                full_roster_attempt_p95_unavailable=attempt_p95 is None,
                full_roster_complete_p95_seconds=complete_p95,
                full_roster_complete_p95_unavailable=complete_p95 is None,
                p95_untimed_rule='INFINITY_IN_ORIGINAL_TWENTY_SLOTS',
                metric='ProtocolConstructor.prepare plus readiness of returned state/public arrays',
                scope='COMPONENT_TIMING_ONLY_NO_COVERAGE_OR_SEARCH_ADMISSION')


def request_for(meta, *, worlds, phase, ordinal):
    from . import phase_d_owning_capture as capture
    from .phase_d_search_server import SCHEMA as request_schema
    found = next((row for row in capture.read_capture(meta['capture_path'])
                  if row['capture_row'] == meta['capture_row']), None)
    if found is None or address(found) != address(meta):
        raise ValueError('Same original captured root unavailable')
    # These identities also feed the original constructor/sampler keys; an
    # unchanged prefix/address alone cannot authorize a substituted battle.
    for key in ('battle_id', 'stream_id', 'generation'):
        if type(found.get(key)) is not type(meta[key]) or found.get(key) != meta[key]:
            raise ValueError('Frozen captured root identity differs: ' + key)
    return dict(schema=request_schema,
                request_id=f'{ROSTER_SHA256}:R7:constructor:{worlds}:{phase}:{ordinal}',
                **{key: found[key] for key in ('battle_id', 'owner', 'stream_id',
                   'decision_index', 'generation', 'prefix_sha256', 'lines', 'own_request_line')})


def measure(args):
    # All dependency/model/device imports are below input validation. The
    # constructor loader is the exact original one used by the search route.
    config = validate_config(read_bound(args.config, args.expected_config_sha256))
    review = read_bound(args.producer_review, args.expected_producer_review_sha256)
    if (not review['status'].startswith('SCOPED_CLEAR')
            or review['source_commit'] != config['source_commit']
            or review['config_sha256'] != args.expected_config_sha256):
        raise ValueError('Same reviewed numerical configuration required')
    head = subprocess.check_output(['git', '-C', config['source_directory'],
                                    'rev-parse', 'HEAD'], text=True).strip()
    if head != config['source_commit']:
        raise ValueError('Frozen measurement source required')
    declaration = read_bound(config['roster']['path'], ROSTER_SHA256)
    roster = timing_roster(declaration['selected'])
    root = Path(args.output)
    if str(root) != config['warm_constructor']['output']:
        raise ValueError('Declared component output required')
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'result.json').exists():
        raise ValueError('Completed constructor timing point must not be rerun')
    settings = dict(config_sha256=args.expected_config_sha256,
                    producer_review_sha256=args.expected_producer_review_sha256,
                    addresses=[list(address(row)) for row in roster],
                    worlds=list(BUDGETS), passes=list(PASSES), seed=config['seed'])
    settings_path = root / 'settings.json'
    if settings_path.exists():
        if settings_path.read_bytes() != encoded(settings):
            raise ValueError('Immutable constructor timing settings changed')
    else:
        write_new(settings_path, settings)
    setup_start = time.monotonic()
    import jax
    from gpu_engine.phase_d_protocol_constructor import ProtocolConstructor, load_inputs
    from .live_protocol_runtime import CompiledEncoderFunctions
    devices = jax.devices('gpu')
    if len(devices) != 1:
        raise ValueError('One visible GPU required for this component timing point')
    catalog, source, metadata, owning = load_inputs(SimpleNamespace(**config['constructor_inputs']))
    functions = CompiledEncoderFunctions(catalog)
    setup_seconds = time.monotonic() - setup_start
    invocation = str(time.time_ns())
    summaries = []
    for worlds in BUDGETS:
        warmups_here = 0
        for phase in PASSES:
            constructor = ProtocolConstructor(catalog=catalog, source=source, metadata=metadata,
                owning_config=owning, count=worlds, seed=config['seed'], rung='R7',
                saved_root_admission=True, functions=functions)
            records = []
            for ordinal, meta in enumerate(roster):
                folder = root / str(worlds) / phase
                target = folder / 'records' / f'{ordinal}.json'
                claim = folder / 'claims' / f'{ordinal}.json'
                expected_claim = dict(ordinal=ordinal, address=list(address(meta)),
                    settings_sha256=hashlib.sha256(encoded(settings)).hexdigest())
                old_claim = json.loads(claim.read_bytes()) if claim.exists() else None
                if old_claim is not None and any(old_claim.get(key) != value
                                                for key, value in expected_claim.items()):
                    raise ValueError('Original timing claim differs')
                if target.exists():
                    if old_claim is None:
                        raise ValueError('Timing result lacks its original claim')
                    records.append(json.loads(target.read_bytes()))
                    continue
                record = dict(ordinal=ordinal, address=list(address(meta)), turn=meta['turn'],
                    worlds=worlds, phase=phase, invocation=invocation,
                    warm_context_qualified=phase == 'warm' and warmups_here == 20,
                    status='INTERRUPTED_UNKNOWN', constructed_worlds=None,
                    prepare_seconds=None, refusals=[])
                if claim.exists():
                    record['refusals'] = ['INTERRUPTED_ORIGINAL_ATTEMPT_UNKNOWN_NO_REPLAY']
                else:
                    write_new(claim, dict(**expected_claim, invocation=invocation))
                    # Capture lookup and request assembly are outside this
                    # component timer; full-request timing measures its route.
                    request = request_for(meta, worlds=worlds, phase=phase, ordinal=ordinal)
                    started = time.monotonic()
                    try:
                        _, made, _ = constructor.prepare(request, ordinal)
                        for world in made.worlds:
                            if world.state is not None:
                                jax.block_until_ready((world.state, world.public))
                        record['constructed_worlds'] = sum(w.state is not None for w in made.worlds)
                        record['status'] = 'COMPLETE' if made.fully_constructed else 'REFUSED'
                        record['refusals'] = sorted({reason for world in made.worlds
                                                    for reason in world.refusals})
                        if not made.fully_constructed and not record['refusals']:
                            record['refusals'] = ['WORLD_CONSTRUCTOR_REFUSED']
                    except (ValueError, KeyError, TypeError, IndexError, AttributeError) as error:
                        record['status'] = 'REFUSED'
                        record['refusals'] = [type(error).__name__ + ': ' + str(error)]
                    record['prepare_seconds'] = time.monotonic() - started
                    if phase == 'warmup':
                        warmups_here += 1
                write_new(target, record)
                records.append(record)
                print(json.dumps({key: record[key] for key in ('ordinal', 'worlds', 'phase',
                    'status', 'prepare_seconds', 'constructed_worlds', 'warm_context_qualified')}), flush=True)
            summaries.append(summarize(roster, records, worlds=worlds, phase=phase))
    result = dict(settings=settings, setup_seconds_this_invocation=setup_seconds,
                  setup_metric='Dependency imports, visible-device initialization and catalog/source loading; CLI/input lookup is outside this component setup timer',
                  invocation=invocation, devices=[str(device) for device in devices],
                  summaries=summaries,
                  interruption_rule='No replay. Warm timings require all20 warmup calls in the same invocation; resumed unconditioned times stay unqualified.',
                  scope='QUARANTINED_SAVED_ROOT_COMPONENT_DIAGNOSTIC_NO_MODEL_FORWARD_OR_GAMES')
    write_new(root / f'invocation-{invocation}.json', result)
    write_new(root / 'result.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'expected-config-sha256', 'producer-review',
                 'expected-producer-review-sha256', 'output'):
        parser.add_argument('--' + name, required=True)
    result = measure(parser.parse_args())
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
