"""Run the two S0 PROGRESS panels against a committed V2 milestone."""

import argparse
from dataclasses import asdict, replace
from functools import partial
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time

from .v2_engine_evaluation import PANEL_PAIRS, SEATS, Terminal, schedule, score


CONFIG_SCHEMA = 'pokezero.v2.engine_eval.inputs.v1'
RECEIPT_SCHEMA = 'pokezero.v2.engine_eval.receipt.v1'


def _artifact(ref):
    from pokezero.v5_evaluation import Artifact
    return Artifact(**ref)


def _decode(starts, *, metadata, show):
    import numpy as np
    from .team_pool import FIELDS

    node = _artifact(show['node']).verify()
    packed = [row['teams'][owner] for row in starts for owner in SEATS]
    script = Path(__file__).with_name('v2_engine_eval_starts.cjs')
    decoded = json.loads(subprocess.check_output(
        [node, script, show['root'], show['compiled_tree_sha256']], input=json.dumps(packed), text=True))
    if decoded['vocabulary'] != metadata['vocabulary'] or len(decoded['teams']) != len(packed):
        raise ValueError('decoded teams differ from the engine vocabulary or registered count')
    fields = (*FIELDS, 'happiness', 'gender')
    arrays = {name: np.asarray([[mon[name] for mon in team] for team in decoded['teams']], np.int32)
              for name in fields}
    return arrays, decoded['genderPolicy']


def _raw(arrays, gender_policy, cells):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from . import transition
    from .gender import validate_policy

    indices = 2 * np.asarray([cell.offset for cell in cells], np.int32)[:, None] + np.arange(2)
    seeds = jnp.asarray([cell.seed for cell in cells], jnp.uint32)
    chance = jax.vmap(jax.random.PRNGKey)(seeds)
    state = transition.state_from_pool(arrays, indices)._replace(keys=chance)
    codes = validate_policy(gender_policy, gender_policy['species'])
    fixed = codes[arrays['species'][indices]]
    explicit = arrays['gender'][indices]
    if np.any((fixed < 0) & (explicit < 0)):
        raise ValueError('registered start has an unresolved gender law')

    def mixed(key):
        key = jax.random.fold_in(key, 0x067B3D92)
        return jax.random.randint(key, (2, 6), 1, 3, dtype=jnp.int32)

    gender = jnp.where(explicit >= 0, explicit, jnp.where(fixed == 3, jax.vmap(mixed)(chance), fixed))
    return state._replace(gender=gender), seeds


def _prepare(config):
    from pokezero.category_vocab import CategoryVocabulary
    from pokezero.dex import load_showdown_dex_cached
    from pokezero.live_protocol_actor import ActorContract, _vocab_sha
    from pokezero.randbat import load_gen3_randbat_source_cached
    from pokezero.v5_progress_evaluation import SCHEMA as PROGRESS_SCHEMA
    from . import champion_evaluator, engine_device_observation, public_belief, transition, v4_encoder

    profile = _artifact(config['progress_profile']).document()
    if profile['schema'] != PROGRESS_SCHEMA or type(profile['max_requests']) is not int or profile['max_requests'] <= 0:
        raise ValueError('registered PROGRESS profile and cap required')
    runtime = _artifact(profile['runtime']).document()
    show = runtime['showdown']
    node = _artifact(show['node']).verify()
    if node.name != 'node':
        raise ValueError('pinned Node executable must have its normal name')
    os.environ['PATH'] = os.pathsep.join((str(node.parent), os.environ['PATH']))

    metadata = _artifact(config['constructor_metadata']).document()
    if metadata['compiled_tree']['sha256'] != show['compiled_tree_sha256']:
        raise ValueError('engine and registered PROGRESS dex trees differ')
    document = _artifact(runtime['category_vocabulary']).document()
    vocabulary = CategoryVocabulary(tuple(document['tokens']), document['oov_buckets'], document['aliases'])
    original = ActorContract.champion()
    source = load_gen3_randbat_source_cached(show['root'])
    if (show['commit'] != original.showdown_revision or _vocab_sha(vocabulary) != original.category_vocab_sha256
            or source.metadata.source_hash != original.belief_set_source_hash):
        raise ValueError('original acting vocabulary and belief source required')
    dex = load_showdown_dex_cached(show['root'])
    belief = public_belief.compile_source(source)

    def compile_catalog(contract):
        return v4_encoder.compile_encoder(belief, dex, vocabulary,
            checkpoint_sha256=contract.checkpoint_sha256, feature_masks=contract.encoder_masks())

    archive = _artifact(config['champion_archive'])
    champion = champion_evaluator.load(archive.path, expected_archive_sha256=archive.sha256,
        expected_checkpoint_sha256=profile['champion']['sha256'], precision='float32')
    champion_catalog = compile_catalog(original)
    champion_evaluator.check_binding(champion, champion_catalog)
    champion_tables = engine_device_observation.compile_tables(metadata, catalog=champion_catalog)
    engine = transition.tables_from_metadata(metadata)
    return profile, show, metadata, vocabulary, compile_catalog, dex, engine, champion, champion_tables


def _candidate(request, vocab, metadata, compile_catalog):
    from pokezero import v5_deferred_evaluation as deferred, v5_run
    from pokezero.checkpoint_self_moves import archive_self_moves
    from pokezero.live_protocol_actor import ActorContract
    from pokezero.v5_model import V5ModelConfig
    from pokezero.v5_phasec import category_binding
    from . import engine_device_observation, v5_model

    declaration = deferred.read_evaluation_declaration(request.run_directory, request.declaration_sha256)
    declared_model = V5ModelConfig.from_metadata(declaration['model'])
    candidate, model_config, actor = v5_model.load(request.archive_path,
        expected_archive_sha256=request.snapshot.archive_sha256,
        expected_checkpoint_sha256=request.snapshot.checkpoint_sha256,
        expected_category_vocabulary_sha256=category_binding(vocab),
        expected_label_vocabulary_sha256=declared_model.label_vocabulary.sha256, enable_v5=True)
    if model_config != declared_model or v5_run._sha(model_config.metadata()) != request.snapshot.model_sha256:
        raise ValueError('the candidate model differs from the committed milestone')
    contract = replace(ActorContract.champion(), checkpoint_sha256=request.snapshot.checkpoint_sha256,
                       model_config_sha256=request.snapshot.model_sha256, self_moves=archive_self_moves(actor))
    if contract.self_moves != declaration.get('self_moves', False):
        raise ValueError('candidate self-move encoding differs from its declaration')
    tables = engine_device_observation.compile_tables(metadata, catalog=compile_catalog(contract))
    return candidate, model_config, tables


def _play(cells, arrays, gender_policy, execute, candidate, opponent, batch_size, *, progress=None):
    import jax
    import jax.numpy as jnp
    from .v5_device_rollout import CAPPED, COMPLETE

    outcomes = []
    for offset in range(0, len(cells), batch_size):
        real = cells[offset:offset + batch_size]
        batch = real + [real[-1]] * (batch_size - len(real))
        state, seeds = _raw(arrays, gender_policy, batch)
        seats = jnp.asarray([cell.seat for cell in batch], jnp.int32)
        enabled = jnp.arange(batch_size) < len(real)
        status, winners, requests = jax.device_get(execute(state, seats, seeds, enabled, candidate, opponent))
        for cell, state_code, winner, frames in zip(real, status, winners, requests):
            if state_code not in (COMPLETE, CAPPED):
                raise ValueError(f'engine refused seed {cell.seed} on seat {cell.seat}')
            capped = bool(state_code == CAPPED)
            outcomes.append(Terminal(cell.seed, cell.seat, int(frames), capped, None if capped else int(winner)))
        if progress is not None:
            progress(len(outcomes))
    return outcomes


def _publish(path, receipt):
    """Publish one complete receipt even if a dispatcher repeats this look."""
    from pokezero.v5_deferred_evaluation import sync_directory

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.pending-', delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write((json.dumps(receipt, sort_keys=True, allow_nan=False) + '\n').encode())
            stream.flush()
            os.fchmod(stream.fileno(), 0o444)
            os.fsync(stream.fileno())
            try:
                os.link(temporary, path)
            except FileExistsError:
                return _existing(path, receipt['binding'])
            sync_directory(path.parent)
        finally:
            temporary.unlink(missing_ok=True)
    return receipt


def _existing(path, binding):
    try:
        receipt = json.loads(path.read_bytes())
    except FileNotFoundError:
        return None
    if receipt['schema'] != RECEIPT_SCHEMA or receipt['binding'] != binding:
        raise ValueError('existing receipt belongs to a different evaluator or milestone')
    return receipt


def evaluate(dispatch, config_ref, output, *, source_sha, batch_size, progress=None):
    import jax
    from pokezero import v5_deferred_evaluation as deferred
    from . import v2_engine_eval_max_damage, v2_engine_evaluation_device as device

    start = time.perf_counter()
    output = Path(output).resolve()
    binding = dict(dispatch_sha256=dispatch.sha256, config_sha256=config_ref.sha256,
                   engine_source_sha=source_sha, precision='float32', batch_size=batch_size)
    request = deferred.read_owned_milestone(dispatch)
    if output.is_relative_to(request.run_directory):
        raise ValueError('write evaluation receipts outside the immutable training root')
    if receipt := _existing(output, binding):
        return receipt
    if len(jax.devices()) != 1 or jax.devices()[0].platform != 'gpu':
        raise ValueError('the panel runner requires one GPU')
    config = config_ref.document()
    if config['schema'] != CONFIG_SCHEMA or set(config['banks']) != set(PANEL_PAIRS):
        raise ValueError('the two original PROGRESS team banks are required')
    profile, show, metadata, vocab, compile_catalog, dex, engine, champion, champion_tables = _prepare(config)
    candidate, model_config, candidate_tables = _candidate(request, vocab, metadata, compile_catalog)
    max_damage = v2_engine_eval_max_damage.compile_tables(metadata['vocabulary'], dex)
    rivals = {
        'champion_9375': (device.champion_opponent(champion, champion_tables), champion.params),
        'max_damage': (v2_engine_eval_max_damage.opponent(max_damage, engine.struggle), None),
    }
    cap = profile['max_requests']
    panels = {}
    for name, (opponent, weights) in rivals.items():
        panel_start = time.perf_counter()
        bank = _artifact(config['banks'][name]).document()
        if len(bank) != PANEL_PAIRS[name]:
            raise ValueError(f'wrong registered pair count for {name}')
        cells = schedule(bank)
        notify = None if progress is None else partial(progress, name, len(cells))
        if notify is not None:
            notify(0)
        arrays, gender_policy = _decode(bank, metadata=metadata, show=show)
        execute = device.make_batch(engine, candidate_tables, model_config, opponent, max_requests=cap)
        outcomes = _play(cells, arrays, gender_policy, execute, candidate, weights, batch_size, progress=notify)
        stats = score(cells, outcomes, max_requests=cap, interim=name == 'champion_9375')
        elapsed = time.perf_counter() - panel_start
        panels[name] = dict(stats, wall_seconds=elapsed, games_per_second=stats['games'] / elapsed,
                            outcomes=[asdict(terminal) for terminal in outcomes])
    deferred.require_live_milestone(request)
    elapsed = time.perf_counter() - start
    receipt = dict(schema=RECEIPT_SCHEMA, binding=binding, declared_games=request.declared_games,
                   checkpoint_sha256=request.snapshot.checkpoint_sha256,
                   champion_checkpoint_sha256=champion.checkpoint_sha256, max_requests=cap,
                   wall_seconds=elapsed, gpu_seconds=elapsed,
                   games_per_second=sum(panel['games'] for panel in panels.values()) / elapsed, panels=panels)
    return _publish(output, receipt)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dispatch', type=Path, required=True)
    parser.add_argument('--dispatch-sha256', required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--config-sha256', required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=256)
    args = parser.parse_args(argv)
    source_sha = os.environ.get('V2_SOURCE_COMMIT', '')
    if not re.fullmatch('[0-9a-f]{40}', source_sha) or args.batch_size <= 0:
        parser.error('a bound launcher source SHA and positive batch size are required')
    receipt = evaluate(_artifact(dict(path=str(args.dispatch.resolve()), sha256=args.dispatch_sha256)),
        _artifact(dict(path=str(args.config.resolve()), sha256=args.config_sha256)),
        args.receipt.resolve(), source_sha=source_sha, batch_size=args.batch_size)
    print(json.dumps({name: panel['games'] for name, panel in receipt['panels'].items()}), flush=True)


if __name__ == '__main__':
    main()
