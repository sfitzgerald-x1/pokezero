"""Create-only, preregistered whole-game paper-reference/incumbent comparison.

Qualification uses disjoint seeds and never enters the confirmatory analysis.
The confirmatory roster is fixed at 64 seed clusters (256 games). Run sequentially
on the laptop; never deploy, retrain, merge PRs, redraw or overwrite a result.
"""
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import random
import signal
import subprocess
import time

from pokezero.mcts_eval.wider_search import ARMS, SEATS, analyze, game_identity, play_game, study_seed

REPO = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def evidence_json_default(value):
    # Failure receipts contain the same BatchResult dataclasses converted by
    # asdict(measured) on successful decisions. Serialize their full fields;
    # never stringify/omit an unsupported diagnostic or admit it as search work.
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    raise TypeError(f'unsupported evidence type: {type(value).__name__}')


def save(path, value):
    # Validate/serialize before opening a create-only output. A bad diagnostic
    # must not leave a truncated canonical terminal that hides its own error.
    encoded = json.dumps(value, indent=2, sort_keys=True, default=evidence_json_default)
    with Path(path).open('x') as stream:
        stream.write(encoded + '\n')


def save_step(path, value):
    # Q/N/M/F and per-world receipts are losslessly retained. Compression keeps
    # a full fixed roster feasible without deleting or thinning decision data.
    data = json.dumps(value, sort_keys=True).encode()
    with path.open('xb') as stream:
        stream.write(gzip.compress(data, compresslevel=3, mtime=0))


def staged_conditioning_enabled(registration):
    value = registration.get('reference', {}).get('staged_substitute_conditioning', False)
    if type(value) is not bool:
        raise RuntimeError('staged conditioning opt-in must be a registered boolean')
    return value


def validate_reference_measurement(measured):
    """Cancelled attempts retain RNG ownership but never qualify as search work.

    The parallel engine separately checks the new root backup delta. Here the
    whole-game driver checks every draw and batch, including cancelled-only
    batches; it must not reject a valid completed decision for reaching its clock.
    """
    if len(set(measured.worker_pids)) != 20 or measured.result.trajectories <= 0:
        raise RuntimeError('reference requires twenty workers and NEW complete trajectories')
    count = completed = transitions = 0
    for receipt in measured.worker_receipts:
        batch, draws = receipt['batch'], receipt['evidence']['draws']
        if len(draws) != batch.world_draws:
            raise RuntimeError('reference batch/draw count drift')
        valid = 0
        for draw in draws:
            if draw['status'] == 'ROOT_VALIDATED' and draw.get('released') is True:
                valid += 1
            elif draw['status'] == 'DEADLINE_CANCELLED':
                diagnostic = draw.get('sampling_diagnostic', {})
                checked, deadline = diagnostic.get('checked_at'), diagnostic.get('deadline_at')
                if (diagnostic.get('schema') != 'pokezero.world-sampling-deadline.v1'
                        or type(checked) not in (int, float) or type(deadline) not in (int, float)
                        or not math.isfinite(checked) or not math.isfinite(deadline) or checked < deadline
                        or diagnostic.get('accepted_world') is not False
                        or diagnostic.get('backed_up') is not False
                        or not batch.deadline_exhausted):
                    raise RuntimeError('cancelled reference world lacks clock/zero-backup evidence')
            else:
                raise RuntimeError('refused or unreleased reference world cannot enter a decision')
        if not 0 <= batch.trajectories <= valid or (valid == 0 and batch.transitions != 0):
            raise RuntimeError('cancelled reference attempts counted as work')
        count += len(draws)
        completed += batch.trajectories
        transitions += batch.transitions
    if (count != measured.result.world_draws or completed != measured.result.trajectories
            or transitions != measured.result.transitions):
        raise RuntimeError('reference aggregate work receipt drift')


def step_files(root):
    return sorted(path for path in root.iterdir() if path.is_file())


def git(*args, root=REPO):
    return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()


def verify(m, *, source_root=REPO):
    if git('rev-parse', 'HEAD', root=source_root) != m['source_commit'] or git('status', '--porcelain', root=source_root):
        raise RuntimeError('source must remain pinned and clean')
    for path, digest in m['input_hashes'].items():
        if sha(path) != digest:
            raise RuntimeError('source/input binding drift: ' + path)
    if m.get('native_build_receipt'):
        receipt = json.loads(Path(m['native_build_receipt']).read_text())
        for path, digest in receipt['source_hashes'].items():
            if sha(path) != digest:
                raise RuntimeError('native build/runtime receipt input drift: '+path)
        if receipt.get('python_engine_package'):
            import poke_engine
            if Path(poke_engine.__file__).parent != Path(receipt['python_engine_package']):
                raise RuntimeError('wrong Python engine package imported')
        if receipt.get('python_executable'):
            import sys
            if Path(sys.executable).resolve() != Path(receipt['python_executable']).resolve():
                raise RuntimeError('wrong model Python runtime imported')
        if receipt.get('torch_version'):
            import torch
            if (torch.__version__ != receipt['torch_version']
                    or str(Path(torch.__file__)) not in receipt['source_hashes']):
                raise RuntimeError('wrong receipt-bound model library imported')
    if (git('rev-parse', 'HEAD', root=m['showdown_root']) != m['showdown_commit']
            or git('status', '--porcelain', root=m['showdown_root'])):
        raise RuntimeError('pinned simulator drift')
    import pokezero_search
    from pokezero.randbat import load_gen3_randbat_source_cached
    native = Path(pokezero_search.__file__).parent
    if native != Path(m['native_package']):
        raise RuntimeError('wrong native package imported')
    if load_gen3_randbat_source_cached(m['showdown_root']).metadata.source_hash != m['set_source_hash']:
        raise RuntimeError('set-source binding drift')
    for path, digest in m.get('retained_input_hashes', {}).items():
        if sha(path) != digest:
            raise RuntimeError('retained evidence binding drift: '+path)
    recovery = m.get('repair_retention')
    if recovery:
        old = json.loads(Path(recovery['original_registration']).read_text())
        if recovery.get('kind') in ('native-semantic-repair', 'native-and-reference-trapping-repair',
                                    'native-and-reference-faint-repair', 'native-and-reference-substitute-repair',
                                    'native-and-reference-substitute-tail-repair',
                                    'native-and-reference-guarded-staged-repair'):
            from pokezero.mcts_eval.wider_native_recovery import verify_historical
            verify_historical(recovery['original_registration'])
        else:
            verify(old, source_root=Path(old['source_root']))
        repair = recovery['audited_repair_source_root']
        if (git('rev-parse', 'HEAD', root=repair) != recovery['audited_repair_source_commit']
                or git('status', '--porcelain', root=repair)):
            raise RuntimeError('audited repair source drift')


def register(args):
    staged = getattr(args, 'staged_substitute_conditioning', False)
    guarded_recovery = getattr(args, 'guarded_staged_recover_from', None)
    if staged and not args.qualification and guarded_recovery is None:
        raise RuntimeError('staged confirmation requires fresh technical qualification and explicit all8-qualified guarded recovery')
    if guarded_recovery is not None and (args.qualification or not staged):
        raise RuntimeError('guarded staged recovery requires confirmation and explicit staged opt-in')
    if git('status', '--porcelain'):
        raise RuntimeError('commit reviewed source before registration')
    import pokezero_search
    from pokezero.randbat import load_gen3_randbat_source_cached
    # Reuse an immutable compiled binary only when ALL its Rust source hashes
    # match the separately retained build evidence. Python conditioning uses the
    # new homogeneous source, not the old diagnostic Python files.
    receipt = json.loads(args.native_binding.read_text())
    native = Path(pokezero_search.__file__).parent
    hashes = {str(args.checkpoint): sha(args.checkpoint), str(args.native_binding): sha(args.native_binding)}
    for path in sorted(native.glob('*')):
        if path.is_file():
            hashes[str(path)] = sha(path)
    compiled_source_root = Path(next(path for path in receipt['source_hashes']
        if path.endswith('/rust/pokezero-search/Cargo.toml'))).parents[2]
    for relative in git('ls-files', 'src', 'scripts', 'rust/pokezero-search', 'third_party', 'pyproject.toml').splitlines():
        if relative == 'third_party/foul-play':
            # Gitlink for an unrelated opponent implementation, not a loaded
            # file or native build input in this champion-only comparison.
            continue
        path = REPO / relative
        hashes[str(path)] = sha(path)
        if (relative.startswith('rust/pokezero-search/')
                and (path.suffix == '.rs' or path.name in ('Cargo.toml', 'Cargo.lock'))):
            expected = [value for original, value in receipt['source_hashes'].items()
                if original.endswith('/' + relative)]
            if expected and (len(expected) != 1 or expected[0] != sha(path)):
                raise RuntimeError('compiled binary source binding differs: ' + relative)
            if not expected:
                # The retained receipt hashes Cargo and src/*.rs, not build.rs.
                # Check and pin that additional input against the unchanged
                # repair checkout; do not claim it was in the earlier receipt.
                original = compiled_source_root / relative
                if not original.is_file() or sha(original) != sha(path):
                    raise RuntimeError('supplementary build input differs: ' + relative)
                hashes[str(original)] = sha(original)
        if relative.startswith('third_party/'):
            original = compiled_source_root / relative
            if not original.is_file() or sha(original) != sha(path):
                raise RuntimeError('compiled engine source/patch binding differs: ' + relative)
            hashes[str(original)] = sha(original)
    binary = list(native.glob('*.so'))
    if len(binary) != 1 or sha(binary[0]) not in receipt['source_hashes'].values():
        raise RuntimeError('native binary absent from retained build evidence')
    source = load_gen3_randbat_source_cached(args.showdown_root)
    if staged:
        from pokezero.mcts_eval.paper_reference_staged_chance import verify_compiled_tree
        verify_compiled_tree(args.showdown_root)
        for path in sorted((args.showdown_root/'dist').rglob('*')):
            if path.is_file() and path.suffix in ('.js', '.json'):
                hashes[str(path)] = sha(path)
    seeds = [study_seed(i, qualification=args.qualification) for i in range(2 if args.qualification else 64)]
    if len(set(seeds)) != len(seeds):
        raise RuntimeError('seed collision; do not silently replace seeds')
    m = dict(schema='pokezero.wider-search.v1', created_at=datetime.now(timezone.utc).isoformat(),
        phase='QUALIFICATION_NOT_STRENGTH' if args.qualification else 'FIXED_64_SEED_CONFIRMATION',
        source_commit=git('rev-parse', 'HEAD'), source_root=str(REPO), seeds=seeds, arms=ARMS, seats=SEATS,
        registered_games=4*len(seeds), checkpoint=str(args.checkpoint), checkpoint_sha256=sha(args.checkpoint),
        showdown_root=str(args.showdown_root), showdown_commit=git('rev-parse', 'HEAD', root=args.showdown_root),
        set_source_hash=source.metadata.source_hash, native_package=str(native), input_hashes=hashes,
        native_build_receipt=str(args.native_binding),
        native_binding='immutable repaired binary and historical Cargo/src receipt; supplementary build/engine inputs byte-checked and pinned against preserved repair checkout',
        nominal_decision_seconds=10., per_decision_safety_seconds=120,
        max_boundaries=200, per_game_wall_seconds=2400.,
        qualification_required_before_confirmation=True,
        durable_evidence='lossless gzip JSON per boundary; every file hash bound by terminal game',
        opponent='same immutable champion; full masked policy sample at every requested opponent decision',
        initial_state='fresh Gen3 random battle; same generated teams/seed for each arm and candidate seat',
        first_action='selected by respective search, not a historical frozen intervention',
        rng='shared seed/seat/boundary domains across arms; opponent and chance independent of search',
        incumbent=dict(depth=6, sims=4096, batch=16, worlds=4, world_workers=1,
            model_priors=True, use_opponent_priors=False, leaf_eval='model', native_batch_guard_ms=64),
        reference=dict(workers=20, exchange_trajectories=10, alpha=.5, beta=1.,
            allow_earlier_compatible_template=True, max_known_set_draws=128,
            canonical_public_rules=True, retain_Q_N_M_F=True,
            cooperative_sampling_deadline=True,
            unfinished_worlds='explicit deadline cancellation; zero backup; attempted RNG ordinal retained',
            selection_requires_new_complete_trajectories=True),
        primary_analysis=dict(unit='battle seed; two seats clustered', score='win=1, draw=.5, loss=0',
            contrast='mean reference score minus mean incumbent score within seed', alpha=.05,
            test='exact two-sided seed-cluster sign flip; exchangeability assumption disclosed',
            confidence='distribution-free Hoeffding bounded-mean 95% interval',
            positive_claim='full fixed roster; p<=.05 AND mean confidence lower bound>0',
            fixed_N=True, optional_stopping=False, historical_data_pooled=False,
            missing='no complete-case inference; full-roster worst-case bounds only'),
        limitations=['not equal CPU allocation: reference twenty workers vs incumbent one',
            'nominal 10s matched decision budgets, actual latency and work recorded separately',
            'world sampling checks the cooperative decision clock; cancelled worlds do not enter Q/N/M; individual simulator/model calls remain atomic and can overrun',
            '128-draw compatible-template adaptation, not an exact hidden-state posterior',
            'local public HP surface may differ from online percentage-only observations',
            'same champion opponent; not Foul Play or unrestricted all-opponent superiority',
            '64 independent clusters may be insufficient for a modest advantage; no equivalence inference',
            'qualification outcomes never enter confirmation; refusal halts, no raw fallback or redraw'])
    m['reference']['staged_substitute_conditioning'] = staged
    if staged:
        m['reference']['staged_kernel'] = 'pokezero.constant-chance-substitute.v1'
        m['limitations'].append('guarded constant-chance replay program explicitly opted in; unsupported public programs keep joint conditioning; deadline-induced sample-selection effects are not proven away')
    recovery_modes = (args.recover_from, args.native_recover_from, args.trapping_recover_from,
                      args.pending_qualification_recover_from, getattr(args, 'faint_recover_from', None),
                      getattr(args, 'substitute_recover_from', None),
                      getattr(args, 'substitute_tail_recover_from', None), guarded_recovery)
    if sum(value is not None for value in recovery_modes) > 1:
        raise RuntimeError('only one explicitly registered recovery mode is permitted')
    if args.recover_from is not None:
        if args.qualification or not all((args.repair_certificate, args.repair_probe, args.qualification_readout)):
            raise RuntimeError('compatible recovery requires confirmation and all qualification certificates')
        from pokezero.mcts_eval.wider_recovery import prepare
        from search_followthrough_diagnostic import restore_reference_checkpoint
        recovery = prepare(args.recover_from, args.repair_certificate, args.repair_probe,
            args.qualification_readout, m, repo=REPO, bound_rows=bound_rows, git=git,
            sha=sha, verify=verify, restore_reference_checkpoint=restore_reference_checkpoint)
        m.update(schema='pokezero.wider-search.compatible-repair.v2', repair_retention=recovery,
                 retained_input_hashes=recovery['input_hashes'],
                 recovery_claim_rule='a positive advantage must satisfy the original full-roster rule AND survive all nine possible score contrasts for the recovered seed cluster')
        m['limitations'].append(recovery['disclosure'])
        m['limitations'].append('accepted-prefix worker ordinals and aggregate restored; priors recomputed in fresh processes; operational cap charged from the preserved filesystem wall-clock envelope')
        validate_qualification(args.qualification_readout, m)
    if args.native_recover_from is not None:
        if args.recover_from is not None or args.qualification or not all((
                args.native_repair_certificate, args.retained_audit, args.qualification_readout,
                args.native_qualification_audit)):
            raise RuntimeError('native recovery requires a new native qualification with full replay audit, repair certificate, and retained audits')
        from pokezero.mcts_eval.wider_native_recovery import prepare
        recovery = prepare(args.native_recover_from, args.native_repair_certificate,
            args.retained_audit, args.qualification_readout, m, repo=REPO, git=git, sha=sha,
            verify=verify, bound_rows=bound_rows, step_files=step_files,
            qualification_audit_path=args.native_qualification_audit)
        m.update(schema='pokezero.wider-search.native-repair.v3', repair_retention=recovery,
            retained_input_hashes=recovery['input_hashes'],
            recovery_claim_rule='original full-roster rule plus worst-case scores for every historically touched cluster')
        m['limitations'].append(recovery['disclosure'])
        m['limitations'].append('incumbent private caches are fresh after accepted-action prefix replay; the full historical wall-clock envelope is charged against the unchanged game cap')
    if args.trapping_recover_from is not None:
        if args.qualification or not all((args.repair_probe, args.trapping_retained_audit,
                args.qualification_readout, args.native_qualification_audit)):
            raise RuntimeError('trapping recovery requires exact draw proof and full historical/new qualification audits')
        from pokezero.mcts_eval.wider_trapping_recovery import prepare
        from search_followthrough_diagnostic import restore_reference_checkpoint
        recovery = prepare(args.trapping_recover_from, args.repair_probe, args.trapping_retained_audit,
            args.qualification_readout, args.native_qualification_audit, m,
            repo=REPO, git=git, sha=sha, verify=verify, bound_rows=bound_rows, step_files=step_files,
            restore_reference_checkpoint=restore_reference_checkpoint)
        m.update(schema='pokezero.wider-search.mixed-trapping-repair.v4', repair_retention=recovery,
            retained_input_hashes=recovery['input_hashes'],
            recovery_claim_rule='full 64-seed roster plus all possible scores for every historical native/reference repair cluster')
        m['limitations'].append(recovery['disclosure'])
        m['limitations'].append('accepted Q/N/M/F and all worker draw positions restored; original prefix and historical wall cap retained; no refused-decision work restored')
    if args.pending_qualification_recover_from is not None:
        if not args.qualification or args.recover_from or args.native_recover_from or not all((
                args.repair_probe, args.pending_retained_audit)):
            raise RuntimeError('pending recovery requires disjoint qualification, exact probe and retained audit')
        from pokezero.mcts_eval.wider_pending_qualification import prepare
        from search_followthrough_diagnostic import restore_reference_checkpoint
        recovery = prepare(args.pending_qualification_recover_from, args.repair_probe,
            args.pending_retained_audit, m, repo=REPO, git=git, sha=sha, verify=verify,
            bound_rows=bound_rows, step_files=step_files, restore_reference_checkpoint=restore_reference_checkpoint)
        m.update(schema='pokezero.wider-search.pending-qualification-repair.v1', repair_retention=recovery,
            retained_input_hashes=recovery['input_hashes'])
        m['limitations'].append(recovery['disclosure'])
    if getattr(args, 'faint_recover_from', None) is not None:
        if args.qualification or not all((args.repair_probe, args.trapping_retained_audit,
                args.qualification_readout, args.native_qualification_audit, args.conditioning_qualification_audit)):
            raise RuntimeError('faint recovery needs exact-root proof and both full qualification audits')
        from pokezero.mcts_eval.wider_faint_recovery import prepare
        from search_followthrough_diagnostic import restore_reference_checkpoint
        recovery = prepare(args.faint_recover_from, args.repair_probe, args.trapping_retained_audit,
            args.qualification_readout, args.native_qualification_audit,
            args.conditioning_qualification_audit, m, repo=REPO, git=git, sha=sha, verify=verify,
            bound_rows=bound_rows, step_files=step_files, restore_reference_checkpoint=restore_reference_checkpoint)
        m.update(schema='pokezero.wider-search.mixed-faint-repair.v5', repair_retention=recovery,
            retained_input_hashes=recovery['input_hashes'],
            recovery_claim_rule='full 64-seed roster and all score assignments for all five historical repair clusters')
        m['limitations'].append(recovery['disclosure'])
        m['limitations'].append('accepted Q/N/M/F, worker draw positions and original wall allowance retained; no refused work restored')
    if getattr(args, 'substitute_recover_from', None) is not None:
        if args.qualification or not all((args.repair_probe, args.trapping_retained_audit,
                args.qualification_readout, args.native_qualification_audit, args.conditioning_qualification_audit)):
            raise RuntimeError('Substitute recovery needs exact-root proof and both full qualification audits')
        from pokezero.mcts_eval.wider_substitute_recovery import prepare
        from search_followthrough_diagnostic import restore_reference_checkpoint
        recovery = prepare(args.substitute_recover_from, args.repair_probe, args.trapping_retained_audit,
            args.qualification_readout, args.native_qualification_audit,
            args.conditioning_qualification_audit, m, repo=REPO, git=git, sha=sha, verify=verify,
            bound_rows=bound_rows, step_files=step_files, restore_reference_checkpoint=restore_reference_checkpoint)
        m.update(schema='pokezero.wider-search.mixed-substitute-repair.v6', repair_retention=recovery,
            retained_input_hashes=recovery['input_hashes'],
            recovery_claim_rule='full64 seed roster and all387420489 score assignments across9 historical clusters')
        m['limitations'].append(recovery['disclosure'])
        m['limitations'].append('all accepted Q/N/M/F,20 worker positions, original prefix and wall charge retained')
    if getattr(args, 'substitute_tail_recover_from', None) is not None:
        if args.qualification or not args.tail_transfer_audit_dir:
            raise RuntimeError('tail recovery requires fixed confirmation and explicit independent transfer audits')
        from pokezero.mcts_eval.wider_substitute_tail_recovery import prepare
        from search_followthrough_diagnostic import restore_reference_checkpoint
        m['conditioning_limits'] = dict(substitute=2048, pending=128, compatible_template=128)
        recovery = prepare(args.substitute_tail_recover_from, args.tail_transfer_audit_dir, m,
            repo=REPO, git=git, sha=sha, verify=verify, bound_rows=bound_rows,
            step_files=step_files, restore_reference_checkpoint=restore_reference_checkpoint)
        m.update(schema='pokezero.wider-search.mixed-substitute-tail-repair.v7', repair_retention=recovery,
            retained_input_hashes=recovery['input_hashes'],
            recovery_claim_rule='full64 seed roster and all387420489 score assignments across9 historical clusters')
        m['limitations'].append(recovery['disclosure'])
        m['limitations'].append('original decision390 Q/N/M/F and20 worker positions retained; '
            'the failed active attempt is also charged against the original game wall allowance')
    if guarded_recovery is not None:
        if not all((args.staged_audit_dir, args.qualification_readout,
                args.native_qualification_audit, args.conditioning_qualification_audit)):
            raise RuntimeError('guarded staged recovery needs complete fresh all8 and independent proof/audit bindings')
        from pokezero.mcts_eval.wider_guarded_staged_recovery import prepare
        from search_followthrough_diagnostic import restore_reference_checkpoint
        m['conditioning_limits'] = dict(substitute=2048, pending=128, compatible_template=128)
        recovery = prepare(guarded_recovery, args.staged_audit_dir, args.qualification_readout,
            args.native_qualification_audit, args.conditioning_qualification_audit, m,
            repo=REPO, git=git, sha=sha, verify=verify, bound_rows=bound_rows,
            step_files=step_files, restore_reference_checkpoint=restore_reference_checkpoint)
        m.update(schema='pokezero.wider-search.mixed-guarded-staged-repair.v8', repair_retention=recovery,
            retained_input_hashes=recovery['input_hashes'],
            recovery_claim_rule='full64/256 roster and all387420489 score assignments across9 outcome-blind historical clusters')
        m['limitations'].append(recovery['disclosure'])
        m['limitations'].append('only15 accepted boundaries/checkpoint392 and original20 RNG positions restored; prior active failed wall retained; no diagnostic or failed work restored')
    verify(m)
    args.output.mkdir(exist_ok=False)
    save(args.output/'registration.json', m)


def timeout(*args):
    raise TimeoutError('registered decision safety cap; no raw fallback')


def bound_rows(output, m):
    rows = []
    recovery = m.get('repair_retention', {})
    retained_groups = {}
    for identity, binding in recovery.get('retained_complete', {}).items():
        path = Path(binding['path'])
        if (output/(identity+'.json')).exists() or sha(path) != binding['sha256']:
            raise RuntimeError('retained complete game duplicate or drift')
        if binding['registration'] not in retained_groups:
            old = json.loads(Path(binding['registration']).read_text())
            retained_groups[binding['registration']] = bound_rows(path.parent, old)
        retained = [row for row in retained_groups[binding['registration']] if row['identity'] == identity]
        if len(retained) != 1 or retained[0]['status'] != 'COMPLETE':
            raise RuntimeError('retained complete game missing')
        rows.append(retained[0])
    for path in sorted(output.glob('seed-*.json')):
        row = json.loads(path.read_text())
        if row['registration_sha256'] != sha(output/'registration.json'):
            raise RuntimeError('result registration binding drift')
        from pokezero.mcts_eval.wider_recovery import combined_steps
        files = combined_steps(output, row, m, sha=sha, step_files=step_files)
        actual = {p.name: sha(p) for p in files}
        if actual != row['step_hashes'] or not actual:
            raise RuntimeError('durable decision evidence drift or missing evidence')
        for step in files:
            data = gzip.decompress(step.read_bytes()) if step.name.endswith('.json.gz') else step.read_bytes()
            json.loads(data)
        rows.append(row)
    return rows


def validate_qualification(path, confirmation):
    if path.name != 'READOUT.json':
        raise RuntimeError('qualification requires the canonical durable readout')
    registration = json.loads((path.parent/'registration.json').read_text())
    recovery = confirmation.get('repair_retention')
    if recovery is not None and recovery.get('kind') == 'native-and-reference-guarded-staged-repair':
        from pokezero.mcts_eval.wider_guarded_staged_recovery import validate_qualification as validate_staged
        return validate_staged(path, confirmation, recovery, repo=REPO, git=git,
            sha=sha, verify=verify, bound_rows=bound_rows)
    if recovery is not None and recovery.get('kind') == 'native-and-reference-substitute-tail-repair':
        from pokezero.mcts_eval.wider_substitute_tail_recovery import validate_transfer
        return validate_transfer(path, confirmation, recovery, repo=REPO, git=git,
            sha=sha, verify=verify, bound_rows=bound_rows)
    if recovery is not None and recovery.get('kind') in ('native-semantic-repair', 'native-and-reference-trapping-repair',
                                                       'native-and-reference-faint-repair', 'native-and-reference-substitute-repair'):
        if str(path) != recovery['qualification_readout']:
            raise RuntimeError('native repair qualification path drift')
        from pokezero.mcts_eval.wider_native_recovery import validate_native_qualification
        return validate_native_qualification(path, confirmation, repo=REPO, git=git,
            sha=sha, verify=verify, bound_rows=bound_rows)
    verify(registration, source_root=Path(registration.get('source_root', REPO)))
    compatible = (recovery is not None and str(path) == recovery['qualification_readout']
                  and registration['source_commit'] == recovery['original_source_commit'])
    if (registration['phase'] != 'QUALIFICATION_NOT_STRENGTH'
            or registration['seeds'] != [study_seed(i, qualification=True) for i in range(2)]
            or registration['registered_games'] != 8
            or staged_conditioning_enabled(registration) != staged_conditioning_enabled(confirmation)
            or not compatible and (registration['source_commit'] != confirmation['source_commit']
                or registration['input_hashes'] != confirmation['input_hashes'])):
        raise RuntimeError('qualification roster or source/input binding differs')
    recomputed = analyze(registration['seeds'], bound_rows(path.parent, registration))
    recomputed.update(source_commit=registration['source_commit'], input_hashes=registration['input_hashes'],
        phase=registration['phase'], statistically_supported_advantage=False, inferential_test_allowed=False,
        status='QUALIFICATION_COMPLETE_NOT_STRENGTH' if not recomputed['missing_seed_clusters'] else 'QUALIFICATION_INCOMPLETE')
    if (json.loads(path.read_text()) != recomputed
            or recomputed['status'] != 'QUALIFICATION_COMPLETE_NOT_STRENGTH'):
        raise RuntimeError('qualification incomplete or readout does not derive from durable evidence')
    result = {'readout_sha256': sha(path), 'registration_sha256': sha(path.parent/'registration.json')}
    if compatible:
        result.update(mode='retained qualification with explicit constructor noninterference and recovery qualification',
                      original_source_commit=registration['source_commit'],
                      disclosure=recovery['disclosure'])
    return result


def run(args, m):
    import torch
    from pokezero.collection import env_config_with_policy_spec_masks
    from pokezero.local_showdown import LocalShowdownConfig, LocalShowdownEnv, showdown_choice_for_action
    from pokezero.mcts_eval.head_to_head import public_only_context
    from pokezero.mcts_eval.manifest import SearchConfig
    from pokezero.mcts_eval.paper_reference import ReferenceConfig
    from pokezero.mcts_eval.paper_reference_parallel import ParallelTrajectorySearch
    from pokezero.mcts_eval.paper_reference_runtime import PublicRootRequest, ShowdownWorkerFactory
    from pokezero.mcts_eval.paper_reference_showdown import ChampionEvaluator, decision_state
    from pokezero.mcts_eval.policy_opponent_profile import make_profile_decider, validate_selection
    from pokezero.mcts_eval.resolver import resolve_checkpoint_contract
    from pokezero.neural_policy import load_transformer_policy
    from pokezero.policy import PolicyContext
    from pokezero.trajectory import BattleTrajectory, TrajectoryStep
    from search_followthrough_diagnostic import reference_statistics_checkpoint
    verify(m)
    if m['phase'] != 'QUALIFICATION_NOT_STRENGTH':
        if args.qualification_readout is None:
            raise RuntimeError('confirmation requires completed disjoint qualification')
        qualification = validate_qualification(args.qualification_readout, m)
        save(args.output/'QUALIFICATION_BINDING.json', qualification)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    evaluator = ChampionEvaluator(load_transformer_policy(Path(m['checkpoint']), device='cpu',
        deterministic=True, exploration_epsilon=0., sampling_temperature=1., family_gated_selection=False))
    cfg = env_config_with_policy_spec_masks(LocalShowdownConfig(showdown_root=Path(m['showdown_root']),
        set_belief_source=True), [f"neural:{m['checkpoint']}"], context='wider full-game search comparison')
    live = LocalShowdownEnv(cfg)
    contract = resolve_checkpoint_contract(m['checkpoint'], expected_sha256=m['checkpoint_sha256'],
        showdown_root=m['showdown_root'], showdown_source_sha256=m['set_source_hash'])
    decider = make_profile_decider(contract, m['showdown_root'], arm='incumbent_mcts',
        mode='matched_deadline', opponent_seed=0, deadline_ms=10000, native_batch_guard_ms=64)
    search_cfg = SearchConfig(depth=6, sims=4096, batch=16, worlds=4, inference_mode='local')
    native = decider._policy_for(search_cfg)
    pool = None
    signal.signal(signal.SIGALRM, timeout)
    try:
        pool = ParallelTrajectorySearch(ReferenceConfig(.5, 1.), ShowdownWorkerFactory(
            m['checkpoint'], m['checkpoint_sha256'], m['showdown_root'], m['set_source_hash'],
            allow_earlier_compatible_template=True, max_known_set_draws=128,
            staged_substitute_conditioning=staged_conditioning_enabled(m)))
        print(json.dumps(dict(status='POOL_READY', startup_seconds=pool.startup_seconds)), flush=True)
        for ordinal, seed in enumerate(m['seeds']):
            for subject in SEATS:
                order = ARMS if (ordinal + SEATS.index(subject)) % 2 else tuple(reversed(ARMS))
                for arm in order:
                    identity = game_identity(seed, subject, arm)
                    recovery = m.get('repair_retention', {})
                    if identity in recovery.get('retained_complete', {}):
                        print(json.dumps(dict(identity=identity, status='RETAINED_COMPLETE_NOT_RERUN')), flush=True)
                        continue
                    path = args.output/(identity+'.json')
                    if path.exists():
                        row = json.loads(path.read_text())
                        if row['status'] != 'COMPLETE' or row['registration_sha256'] != sha(args.output/'registration.json'):
                            raise RuntimeError('extant noncomplete game cannot be overwritten or retried')
                        continue
                    steps = args.output/identity
                    steps.mkdir(exist_ok=False)
                    live.reset(seed=seed)
                    battle_id = 'wider-search:' + identity
                    trajectory = BattleTrajectory(battle_id, 'gen3randombattle', seed)
                    # Only the actor's previous public root and own played action
                    # can certify a pending Baton Pass. Never retain live opponent
                    # action indices or requests in the worker transport.
                    from pokezero.mcts_eval.paper_reference_pending import (
                        FaintReplacementTransition, PendingPolicyTransition, requires_faint_encore_replay)
                    previous_public_transition = [None]
                    from pokezero.mcts_eval.paper_reference_substitute import SubstituteHistoryTracker
                    substitute_history = SubstituteHistoryTracker()

                    def pending_for(public):
                        substitute = substitute_history.certificate(public)
                        if substitute is not None:
                            return substitute
                        pending = previous_public_transition[0] if public.deferred_opponent_action_player is not None else None
                        if requires_faint_encore_replay(public):
                            previous = previous_public_transition[0]
                            if previous is None:
                                raise RuntimeError('forced Encore replacement lacks its public prior root')
                            pending = FaintReplacementTransition(previous.before_state,
                                previous.before_observation, previous.own_action, previous.set_source_hash,
                                previous.prior_transition)
                        return pending
                    native.reset()
                    resume = recovery.get('resume') if recovery.get('resume', {}).get('identity') == identity else None
                    if resume:
                        from pokezero.mcts_eval.wider_recovery import read_step
                        from search_followthrough_diagnostic import restore_reference_checkpoint
                        for name, retained in sorted(resume['prefix_steps'].items()):
                            row = read_step(retained)
                            if (name != f"boundary-{row['boundary']:03d}.json.gz"
                                    or set(row['actions']) != set(live.requested_players())):
                                raise RuntimeError('retained prefix request boundary drift')
                            for actor, action in row['actions'].items():
                                observation = live.observe(actor)
                                if not observation.legal_action_mask[action]:
                                    raise RuntimeError('retained played action is no longer legal')
                                if actor == subject:
                                    public = live.public_materialization_state(subject)
                                    public_request = PublicRootRequest.capture(public, observation,
                                        pending_transition=pending_for(public)) if arm == 'paper_reference' else None
                                    if public_request is not None:
                                        substitute_history.accept_own(public_request, action)
                                    if not public.self_request.get('forceSwitch') and public.deferred_opponent_action_player is None:
                                        previous_public_transition[0] = PendingPolicyTransition.capture(
                                            public_request or PublicRootRequest.capture(public, observation), action)
                                    trajectory.append(TrajectoryStep(player_id=subject, turn_index=row['boundary'],
                                        observation=observation, legal_action_mask=tuple(observation.legal_action_mask),
                                        action_index=action))
                            if arm == 'paper_reference' and subject not in row['actions']:
                                substitute_history.accept_opponent_only()
                            live.reseed_simulator_rng(row['chance_seed'])
                            live.step(row['actions'])
                        checkpoint_sha256 = None
                        if arm == 'paper_reference':
                            checkpoint = read_step(resume['checkpoint_step'])['evidence'][subject]['search_evidence']['statistics_checkpoint']
                            pool.restore_statistics(restore_reference_checkpoint(checkpoint),
                                worker_ordinals=tuple(resume['worker_ordinals']), decision_id=resume['decision_id'])
                            checkpoint_sha256 = sha(resume['checkpoint_step'])
                        elif recovery.get('kind') != 'native-semantic-repair':
                            raise RuntimeError('incumbent prefix recovery requires native semantic repair registration')
                        save(args.output/'RECOVERY_RECEIPT.json', dict(identity=identity,
                            boundary=resume['start_boundary'], worker_ordinals=resume['worker_ordinals'],
                            decision_id=resume['decision_id'], checkpoint_sha256=checkpoint_sha256,
                            failed_partial_work_restored=False, private_priors_restored=False))

                    def select(observation, boundary, rng_seed):
                        signal.alarm(m['per_decision_safety_seconds'])
                        begun = time.perf_counter()
                        try:
                            if arm == 'deep_incumbent':
                                context = public_only_context(PolicyContext(player_id=subject,
                                    decision_round_index=boundary, battle_id=battle_id, format_id='gen3randombattle',
                                    seed=seed, observation=observation, requested_players=tuple(live.requested_players()),
                                    trajectory=trajectory, requested_observations={subject: observation},
                                    requested_legal_action_masks={subject: tuple(observation.legal_action_mask)},
                                    public_materialization_state=live.public_materialization_state(subject)))
                                before = decider._snapshot_stats(native)
                                decision = native.select_action_with_context(context, rng=random.Random(rng_seed))
                                after = decider._snapshot_stats(native)
                                index = int(decision.action_index)
                                evidence = dict(root_action=showdown_choice_for_action(live._state_for_player(subject), index),
                                    max_depth_reached=decider._changed_depth(before['depth_reached_histogram'], after['depth_reached_histogram']),
                                    fallbacks=after['fallback_decisions']-before['fallback_decisions'],
                                    prior_fallbacks=after['prior_fallbacks']-before['prior_fallbacks'],
                                    invalid_actions=int(not observation.legal_action_mask[index]),
                                    total_iterations=after['total_iterations']-before['total_iterations'],
                                    model_evals=after['model_evals']-before['model_evals'],
                                    engine_mcts=dict(decision.metadata.get('engine_mcts', {})))
                                if sum(observation.legal_action_mask) > 1:
                                    try:
                                        validate_selection(evidence, arm='incumbent_mcts', mode='matched_deadline',
                                            config=search_cfg, mask=observation.legal_action_mask, opponent_seed=rng_seed,
                                            deadline_ms=10000, native_batch_guard_ms=64)
                                    except Exception as error:
                                        error.evidence = evidence
                                        raise
                                elif evidence['fallbacks'] or evidence['prior_fallbacks']:
                                    raise RuntimeError('forced request fell back')
                            else:
                                public = live.public_materialization_state(subject)
                                pending = pending_for(public)
                                request = PublicRootRequest.capture(public, observation, pending_transition=pending)
                                remaining = 10 - (time.perf_counter() - begun)
                                if remaining <= 0:
                                    raise TimeoutError('reference deadline expired during public capture')
                                measured = pool.search(request, decision_state(observation, player=subject),
                                    battle_id=battle_id, seed=rng_seed,
                                    deadline_seconds=remaining)
                                index = int(measured.result.action.split(':')[1])
                                validate_reference_measurement(measured)
                                if resume and boundary == resume['start_boundary']:
                                    probe = json.loads(Path(recovery['exact_failed_draw_certificate']).read_text())
                                    worker_zero = [draw for receipt in measured.worker_receipts if receipt['worker'] == 0
                                                   for draw in receipt['evidence']['draws']]
                                    if recovery.get('kind') == 'native-and-reference-faint-repair':
                                        from pokezero.mcts_eval.wider_faint_recovery import validate_resumed_draw
                                        validate_resumed_draw(worker_zero, probe)
                                    elif recovery.get('kind') == 'native-and-reference-substitute-repair':
                                        from pokezero.mcts_eval.wider_substitute_recovery import validate_resumed_draw
                                        validate_resumed_draw(worker_zero, probe)
                                    elif recovery.get('kind') == 'native-and-reference-substitute-tail-repair':
                                        from pokezero.mcts_eval.wider_substitute_tail_recovery import validate_resumed_workers
                                        validate_resumed_workers(measured.worker_receipts, probe)
                                    elif recovery.get('kind') == 'native-and-reference-guarded-staged-repair':
                                        from pokezero.mcts_eval.wider_guarded_staged_recovery import validate_resumed_workers
                                        validate_resumed_workers(measured.worker_receipts, probe)
                                    elif worker_zero and (worker_zero[0]['packed_team_sha256'] != probe['draw']['packed_team_sha256']
                                            or worker_zero[0]['materialization_seed'] != probe['draw']['materialization_seed']):
                                        raise RuntimeError('recovered first hidden draw differs from exact failed-draw qualification')
                                evidence = asdict(measured)
                                evidence['statistics_checkpoint'] = reference_statistics_checkpoint(pool._master.snapshot())
                                substitute_history.accept_own(request, index)
                                if not public.self_request.get('forceSwitch') and public.deferred_opponent_action_player is None:
                                    previous_public_transition[0] = PendingPolicyTransition.capture(
                                        request, index)
                            trajectory.append(TrajectoryStep(player_id=subject, turn_index=boundary,
                                observation=observation, legal_action_mask=tuple(observation.legal_action_mask), action_index=index))
                            return index, dict(selector=arm, elapsed_seconds=time.perf_counter()-begun,
                                seed=rng_seed, search_evidence=evidence)
                        finally:
                            signal.alarm(0)

                    def emit(row):
                        save_step(steps/f"boundary-{row['boundary']:03d}.json.gz", row)
                        if arm == 'paper_reference' and subject not in row['actions']:
                            substitute_history.accept_opponent_only()
                        print(json.dumps(dict(identity=identity, boundary=row['boundary'],
                            actions=row['actions'], decision_seconds=row['evidence'].get(subject, {}).get('elapsed_seconds'))), flush=True)

                    result = dict(identity=identity, seed=seed, subject=subject, arm=arm,
                        registration_sha256=sha(args.output/'registration.json'))
                    if resume:
                        result.update(retained_prefix_steps=resume['prefix_steps'], recovery=resume['recovery'])
                    try:
                        result.update(play_game(live, subject=subject, decision_id=f'wider-search:{seed}:{subject}',
                            selector=select, opponent=evaluator, emit=emit,
                            max_boundaries=m['max_boundaries'], wall_seconds=m['per_game_wall_seconds'],
                            start_boundary=resume['start_boundary'] if resume else 0,
                            prior_selections=resume['prior_selections'] if resume else 0,
                            elapsed_before_resume=resume['elapsed_before_resume'] if resume else 0.))
                    except Exception as error:
                        result.update(status='REFUSED', signed_outcome=None,
                            error=f'{type(error).__name__}: {error}', failure_evidence=getattr(error, 'evidence', None))
                    from pokezero.mcts_eval.wider_recovery import combined_steps
                    result['step_hashes'] = {p.name: sha(p) for p in
                        combined_steps(args.output, result, m, sha=sha, step_files=step_files)}
                    save(path, result)
                    print(json.dumps({key: result.get(key) for key in ('identity', 'status', 'signed_outcome', 'error')}), flush=True)
                    if result['status'] != 'COMPLETE':
                        raise RuntimeError('noncomplete registered game; preserve evidence, no fallback/redraw')
        verify(m)
        readout(args, m)
    finally:
        signal.alarm(0)
        if pool is not None:
            pool.close()
        live.close()
        decider.close()


def readout(args, m):
    verify(m)
    rows = bound_rows(args.output, m)
    result = analyze(m['seeds'], rows)
    result.update(source_commit=m['source_commit'], input_hashes=m['input_hashes'], phase=m['phase'])
    if 'repair_retention' in m:
        result.update(literal_homogeneous_source=False, recovery_disclosure=m['repair_retention']['disclosure'],
            retained_complete_games=sorted(m['repair_retention']['retained_complete']),
            recovered_game=m['repair_retention']['resume']['identity'])
        if m['phase'] == 'QUALIFICATION_NOT_STRENGTH':
            pass  # Qualification outcomes NEVER enter a significance claim.
        elif m['repair_retention'].get('kind') == 'native-semantic-repair':
            from pokezero.mcts_eval.wider_native_recovery import add_sensitivity
            add_sensitivity(result, m)
        elif m['repair_retention'].get('kind') in ('native-and-reference-trapping-repair', 'native-and-reference-faint-repair'):
            from pokezero.mcts_eval.wider_trapping_recovery import add_mixed_sensitivity
            add_mixed_sensitivity(result, m)
        elif m['repair_retention'].get('kind') in ('native-and-reference-substitute-repair',
                                                 'native-and-reference-substitute-tail-repair',
                                                 'native-and-reference-guarded-staged-repair'):
            from pokezero.mcts_eval.wider_substitute_recovery import add_substitute_sensitivity
            add_substitute_sensitivity(result, m)
        else:
            from pokezero.mcts_eval.wider_recovery import add_recovery_sensitivity
            add_recovery_sensitivity(result, m, sha=sha)
    if m['phase'] == 'QUALIFICATION_NOT_STRENGTH':
        result['statistically_supported_advantage'] = False
        result['inferential_test_allowed'] = False
        result['status'] = 'QUALIFICATION_COMPLETE_NOT_STRENGTH' if not result['missing_seed_clusters'] else 'QUALIFICATION_INCOMPLETE'
    save(args.output/'READOUT.json', result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('register', 'run', 'readout'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--showdown-root', type=Path)
    parser.add_argument('--native-binding', type=Path)
    parser.add_argument('--qualification', action='store_true')
    parser.add_argument('--staged-substitute-conditioning', action='store_true',
        help='Explicit default-off guarded replay kernel; technical qualification only until recovery is validated')
    parser.add_argument('--qualification-readout', type=Path)
    parser.add_argument('--recover-from', type=Path)
    parser.add_argument('--repair-certificate', type=Path)
    parser.add_argument('--repair-probe', type=Path)
    parser.add_argument('--native-recover-from', type=Path)
    parser.add_argument('--native-repair-certificate', type=Path)
    parser.add_argument('--native-qualification-audit', type=Path)
    parser.add_argument('--trapping-recover-from', type=Path)
    parser.add_argument('--trapping-retained-audit', type=Path)
    parser.add_argument('--retained-audit', type=Path, action='append')
    parser.add_argument('--pending-qualification-recover-from', type=Path)
    parser.add_argument('--pending-retained-audit', type=Path)
    parser.add_argument('--faint-recover-from', type=Path)
    parser.add_argument('--substitute-recover-from', type=Path)
    parser.add_argument('--substitute-tail-recover-from', type=Path)
    parser.add_argument('--tail-transfer-audit-dir', type=Path)
    parser.add_argument('--guarded-staged-recover-from', type=Path)
    parser.add_argument('--staged-audit-dir', type=Path)
    parser.add_argument('--conditioning-qualification-audit', type=Path)
    args = parser.parse_args()
    if args.mode == 'register':
        if not all((args.checkpoint, args.showdown_root, args.native_binding)):
            parser.error('registration requires checkpoint, showdown-root and native-binding')
        register(args)
    else:
        m = json.loads((args.output/'registration.json').read_text())
        (run if args.mode == 'run' else readout)(args, m)


if __name__ == '__main__':
    main()
