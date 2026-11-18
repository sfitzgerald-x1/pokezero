"""File-invoked source-pinned fresh200 capture/candidate replay entry.

The launcher owns allocation, deadlines, utilisation and release. This entry
does one declared stage and exits. It never scores a battle, finalizes a run,
generates reference data, samples a new opponent, or publishes qualification.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

HELPERS = ('fresh200_driver.py', 'fresh200_capture.py', 'fresh200_replay.py', 'device_corpus_attestation.py')


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n').encode()


def read_document(path, expected, *, max_bytes=64*1024**2):
    path = Path(path)
    if not path.is_absolute() or not re.fullmatch('[0-9a-f]{64}', expected or ''):
        raise ValueError('absolute input and explicit SHA256 required')
    if path.stat().st_size > max_bytes:raise ValueError('input document exceeds bound')
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:raise ValueError('input document digest differs: '+str(path))
    return json.loads(data)


def runtime_derivation(parent, derived, commit):
    expected = dict(parent, source_commit=commit)
    if derived != expected:raise ValueError('only the runtime source_commit may change')
    return derived


def validate_launch(value, *, integration_commit, producer_commit, policy):
    if (value.get('schema') != 'pokezero.fresh200.launch.v1'
            or value.get('integration_commit') != integration_commit or value.get('producer_commit') != producer_commit
            or value.get('selection_policy') != policy):
        raise ValueError('launch source or fixed root policy differs')
    resources = value.get('resources', {})
    if (resources.get('gpus') != 1 or resources.get('cpu') != 8 or resources.get('memory_bytes') != 48*1024**3
            or resources.get('max_allocation_seconds') != 7200 or resources.get('utilisation_target_percent') != 30):
        raise ValueError('fresh collection requires declared single-GPU/8CPU/48Gi/7200s bounds')
    if value.get('scope') != 'one fixed fresh200 device corpus and same-prefix candidate replay; no scoring or refill':
        raise ValueError('fresh collection scope differs')
    return value


def _git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()


def check_source(root, commit):
    root = Path(root).resolve()
    if not re.fullmatch('[0-9a-f]{40}', commit or '') or _git(root, 'rev-parse', 'HEAD') != commit:
        raise ValueError('exact source checkout required')
    if _git(root, 'status', '--porcelain', '--', 'gpu_engine', 'src/pokezero'):
        raise ValueError('clean engine and Python source required')
    return root


def load_helper(root, name):
    path = root/'gpu_engine'/name
    spec = importlib.util.spec_from_file_location('_fresh200_'+path.stem, path)
    module = importlib.util.module_from_spec(spec);sys.modules[spec.name] = module;spec.loader.exec_module(module)
    return module


def import_engine(root):
    """A fresh process imports all engine/project modules from one checkout."""
    if any(name == 'gpu_engine' or name.startswith(('gpu_engine.', 'pokezero.')) or name == 'pokezero' for name in sys.modules):
        raise ValueError('project namespace already imported; use a separate stage process')
    producer = Path(__file__).resolve().parents[1]
    sys.path[:] = [entry for entry in sys.path if not Path(entry or '.').resolve().is_relative_to(producer)]
    sys.path[:0] = [str(root), str(root/'src')]
    from pokezero import v5_worker as worker
    assert_engine_modules(root)
    return worker


def assert_engine_modules(root):
    for name, module in list(sys.modules.items()):
        if name not in ('gpu_engine', 'pokezero') and not name.startswith(('gpu_engine.', 'pokezero.')):continue
        paths = [module.__file__] if getattr(module, '__file__', None) else list(getattr(module, '__path__', ()))
        if not paths or any(not Path(path).resolve().is_relative_to(root) for path in paths):
            raise ValueError('project module shadowed outside pinned source: '+name)


def read_journal(manifest):
    if type(manifest) is not list or not 0 < len(manifest) <= 64:raise ValueError('bounded exact initial journal manifest required')
    paths = [row['path'] for row in manifest]
    if len(set(paths)) != len(paths) or paths != sorted(paths):raise ValueError('original journal file order required')
    events = []
    for row in manifest:
        if set(row) != {'path', 'sha256', 'kind'}:raise ValueError('complete journal reference required')
        value = read_document(row['path'], row['sha256'], max_bytes=4*1024**2)
        if value['kind'] != row['kind']:raise ValueError('journal event kind changed')
        events.append(value)
    if (sum(event['kind'] == 'SNAPSHOT' and event['payload']['iteration'] == -1 for event in events) != 1
            or sum(event['kind'] == 'BATCH_STARTED' for event in events) != 1 or events[-1]['kind'] != 'BATCH_STARTED'):
        raise ValueError('one initial actor and prefix ending at first bank required')
    return events


def prepare_context(worker, config, runtime, *, arm):
    """Use original factories; runtime/precision probe failures are fatal."""
    from pokezero import v5_run as runs, v5_contract as contract, v5_device_smoke as smoke
    labels = contract.LabelVocabulary(**worker._record(config['labels']))
    profile, vocabulary, gender, category = worker._runtime_metadata(runtime, expected_pool_sha256=config['bank']['pool_sha256'])
    if gender is None:raise ValueError('original prospective gender policy required')
    worker._source(runtime);worker._memory(runtime['memory_policy'], apply=True);worker._probe(runtime)
    plan = smoke.build_plan(config['bank'], profile, labels, category, arm=arm, iterations=1)
    inputs = worker.Inputs(plan, profile, json.dumps(runtime), json.dumps(vocabulary), json.dumps(gender), runs._sha(plan.manifest()))
    prepared, pool, _, _ = worker.prepare_device_runtime(inputs)
    return dict(config=config, runtime=runtime, plan=plan, prepared=prepared, pool=pool, gender=gender)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('collect', 'replay'))
    for name in ('engine-source', 'producer-source', 'config', 'runtime', 'journal-manifest', 'native-attestation', 'launch-declaration', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    for name in ('engine-commit', 'producer-commit', 'integration-commit', 'config-sha256', 'runtime-sha256',
            'journal-sha256', 'native-attestation-sha256', 'launch-sha256'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--reference-run', type=Path)
    parser.add_argument('--arm', choices=('A', 'B'), default='B')
    parser.add_argument('--capture-reference', type=Path)
    parser.add_argument('--capture-reference-sha256')
    args = parser.parse_args(argv);started = time.monotonic()
    producer = check_source(args.producer_source, args.producer_commit)
    engine = check_source(args.engine_source, args.engine_commit)
    if Path(__file__).resolve() != producer/'gpu_engine/fresh200_driver.py':raise ValueError('execute the exact producer entry')
    if args.engine_commit != (args.integration_commit if args.stage == 'collect' else args.producer_commit):
        raise ValueError('stage engine differs from declared integration/candidate')
    capture = load_helper(producer, 'fresh200_capture.py');attestation = load_helper(producer, 'device_corpus_attestation.py')
    launch_value = read_document(args.launch_declaration, args.launch_sha256)
    validate_launch(launch_value, integration_commit=args.integration_commit, producer_commit=args.producer_commit,
        policy=capture.selection_policy())
    expected_inputs = dict(config=dict(path=str(args.config), sha256=args.config_sha256),
        journal_manifest=dict(path=str(args.journal_manifest), sha256=args.journal_sha256),
        native_attestation=dict(path=str(args.native_attestation), sha256=args.native_attestation_sha256))
    expected_inputs[args.stage+'_runtime'] = dict(path=str(args.runtime), sha256=args.runtime_sha256)
    if (any(launch_value.get('inputs', {}).get(name) != value for name, value in expected_inputs.items())
            or launch_value.get('arm') != args.arm
            or args.stage == 'collect' and launch_value.get('reference_run') != str(args.reference_run)):
        raise ValueError('stage inputs differ from the prelaunch declaration')
    config = read_document(args.config, args.config_sha256)
    parent = read_document(config['runtime']['path'], config['runtime']['sha256'])
    runtime = runtime_derivation(parent, read_document(args.runtime, args.runtime_sha256), args.engine_commit)
    journal = read_document(args.journal_manifest, args.journal_sha256);events = read_journal(journal)
    native = read_document(args.native_attestation, args.native_attestation_sha256)
    if (native.get('passed') is not True or native.get('original_ivs_present') is not True
            or native['compiled_tree']['sha256'] != runtime['showdown']['compiled_tree_sha256']
            or native['original_pool']['reference'] != runtime['constructor_pool']):
        raise ValueError('completed Native source/literal-IV identity receipt required')
    os.environ['PATH'] = str(Path(runtime['showdown']['node']['path']).parent)+os.pathsep+os.environ['PATH']
    for key, value in runtime['memory_policy']['environment'].items():
        if value is not None:os.environ[key] = value
    worker = import_engine(engine);context = prepare_context(worker, config, runtime, arm=args.arm)
    assert_engine_modules(engine)
    import jax
    import torch
    devices = jax.devices();device = devices[0]
    identity = dict(runtime=runtime, runtime_document_sha256=args.runtime_sha256,
        python=dict(path=sys.executable, resolved=str(Path(sys.executable).resolve()), sha256=attestation.sha256_file(Path(sys.executable).resolve())),
        device=dict(platform=device.platform, kind=device.device_kind, id=device.id, process_index=device.process_index),
        precision=dict(jax_x64=jax.config.jax_enable_x64, jax_matmul=jax.config.jax_default_matmul_precision,
            torch_matmul=torch.get_float32_matmul_precision()),
        engine_commit=args.engine_commit, producer_commit=args.producer_commit,
        helpers={name:attestation.sha256_file(producer/'gpu_engine'/name) for name in HELPERS})
    identity_bytes = canonical(identity)
    source = dict(engine_commit=args.engine_commit, producer_commit=args.producer_commit,
        entry_sha256=identity['helpers']['fresh200_driver.py'], runtime_sha256=hashlib.sha256(identity_bytes).hexdigest())
    root = args.out.resolve();root.mkdir(parents=True, exist_ok=True)
    with (root/(args.stage+'-runtime-identity.json')).open('xb') as output:output.write(identity_bytes)
    runtime_ref = attestation.seal_manifest(root, args.stage+'-runtime', identity, parents=[], source=source)
    if args.stage == 'collect':
        if args.reference_run is None:raise ValueError('original actor archive directory required')
        from gpu_engine import v2_bench_collector as bench
        snapshot, params, actor_metadata = bench.actor(args, context['plan'], events)
        input_ref = attestation.seal_manifest(root, 'inputs', dict(config=config, config_sha256=args.config_sha256,
            parent_runtime=parent, parent_runtime_reference=config['runtime'], derived_runtime=runtime,
            journal=journal, journal_manifest_sha256=args.journal_sha256, actor_metadata=actor_metadata,
            native_attestation=native, native_attestation_sha256=args.native_attestation_sha256), parents=[runtime_ref], source=source)
        launch_ref = attestation.seal_manifest(root, 'launch', launch_value, parents=[input_ref], source=source)
        result = capture.collect_capsule(root, context=context, snapshot=snapshot, params=params, events=events,
            source=source, launch=launch_ref, attestation=attestation)
    else:
        if args.capture_reference is None:raise ValueError('sealed capture reference required')
        reference = read_document(args.capture_reference, args.capture_reference_sha256)
        recorded = json.loads(attestation.verify_artifact(root, reference));original_source = recorded['source']
        if (original_source['engine_commit'] != args.integration_commit or original_source['producer_commit'] != args.producer_commit
                or original_source['entry_sha256'] != source['entry_sha256']
                or original_source['runtime_sha256'] != attestation.sha256_file(root/'collect-runtime-identity.json')):
            raise ValueError('integration capture source/runtime identity changed')
        replay = load_helper(producer, 'fresh200_replay.py')
        result = replay.replay(root, reference, prepared=context['prepared'], integration_source=original_source,
            candidate_source=source, capture=capture, attestation=attestation, runtime_parent=runtime_ref)
    assert_engine_modules(engine)
    document = json.loads(attestation.verify_artifact(root, result))
    report = dict(stage=args.stage, reference=result, source=source, status=document['payload']['status'],
        seconds=time.monotonic()-started, qualification=False)
    with (root/(args.stage+'-result.json')).open('xb') as output:output.write(canonical(report))
    print('FRESH200 '+json.dumps(report, sort_keys=True), flush=True)
    if report['status'] not in ('CAPTURED_ONLY', 'DEVICE_REPLAYED_ONLY'):raise SystemExit(1)


if __name__ == '__main__':main()
