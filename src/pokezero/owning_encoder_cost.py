"""Fixed TRAIN-root offline cost attempt for the configured owning live path.

No import-time runtime, corpus access, model load, timing, or game authority.
Selection skips observation/action/outcome values; execution uses ordinary live
actors, never SavedWireAdapter.forward or a reconstructed native battle.
"""
from dataclasses import fields, replace
import gzip
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
import time

from . import owning_encoder_study as study
from .live_protocol_actor import ActorContract, _actionable, _packet, _terminal
from .recorded_encoder_corpus import _Prefix

SCHEMA = 'pokezero.B3-owning-cost-prespec.v1'
LEDGER_SCHEMA = 'pokezero.B3-owning-cost-ledger.v1'
COUNT = 20
PUBLIC_DISPATCH = 'one_literal_line_per_call_terminal_suffix_atomic'
SOURCE_FILES = ('src/pokezero/owning_encoder_cost.py', 'scripts/owning_encoder_cost.py')
META_FIELDS = {'event', 'battle_id', 'seed', 'root_id', 'decision_round_index', 'requested_players'}
require, canonical, digest = study.require, study.canonical, study.digest


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _skip(text, start):
    """Skip a JSON value without constructing its tensors/actions/outcomes."""
    if text[start] == '"':
        return json.decoder.scanstring(text, start + 1, True)[1]
    if text[start] not in '[{':
        return json.JSONDecoder().raw_decode(text, start)[1]
    stack, i = [], start
    while i < len(text):
        c = text[i]
        if c == '"':
            i = json.decoder.scanstring(text, i + 1, True)[1]
            continue
        if c in '[{':
            stack.append(c)
        elif c in ']}':
            require(stack and stack.pop() == ( '[' if c == ']' else '{'), 'invalid JSON nesting')
            if not stack:
                return i + 1
        i += 1
    raise ValueError('unterminated JSON value')


def metadata_object(text, wanted):
    """Decode selected top-level fields only; reject duplicate top-level keys.

    Compressed bytes must be read/hash-checked, but skipped values are neither
    deserialized nor inspected for selection. This is not a general JSON API.
    """
    decoder, i, result, seen = json.JSONDecoder(), 0, {}, set()
    def space(i):
        while i < len(text) and text[i].isspace():
            i += 1
        return i
    i = space(i)
    require(text[i:i+1] == '{', 'boundary JSON object required')
    i = space(i + 1)
    while text[i:i+1] != '}':
        key, i = decoder.raw_decode(text, i)
        require(type(key) is str and key not in seen, 'duplicate/invalid top-level key')
        seen.add(key)
        i = space(i)
        require(text[i:i+1] == ':', 'JSON colon required')
        i = space(i + 1)
        if key in wanted:
            result[key], i = decoder.raw_decode(text, i)
        else:
            i = _skip(text, i)
        i = space(i)
        if text[i:i+1] == '}':
            break
        require(text[i:i+1] == ',', 'JSON delimiter required')
        i = space(i + 1)
        require(text[i:i+1] != '}', 'trailing JSON comma')
    require(text[i:i+1] == '}' and space(i + 1) == len(text), 'trailing/incomplete JSON')
    return result


def metadata_rows(path, expected_sha, wanted):
    require(file_sha(path) == expected_sha, 'captured boundary file hash mismatch')
    with gzip.open(path, 'rt') as stream:
        for line in stream:
            yield metadata_object(line, wanted)
    require(file_sha(path) == expected_sha, 'captured boundary file changed during read')


def validate_controls(spec, *, config_class=None):
    """Fail before any model calls when v2 controls have not been integrated."""
    require(spec.get('schema') == 'pokezero.B3-owning-study-prespec.v2',
            'merged B3 prespec v2 with explicit learner limit required')
    limit = spec.get('owning_decision_timeout_seconds')
    require(type(limit) in (int, float) and math.isfinite(limit) and 0 < limit <= 3600,
            'LIMIT_UNBOUND: explicit finite positive learner decision limit required')
    require(spec['controls'].get('owning_decision_timeout_seconds') == limit
        and type(spec['controls'].get('owning_decision_timeout_seconds')) in (int, float)
        and spec['controls'].get('owning_public_single_line') is True
        and spec['execution'].get('public_dispatch') == PUBLIC_DISPATCH,
        'configured learner limit and canonical one-line dispatch must match')
    if config_class is None:
        from .foulplay_bridge import ControlledFoulPlayConfig
        config_class = ControlledFoulPlayConfig
    names = {f.name for f in fields(config_class)}
    require({'owning_decision_timeout_seconds', 'owning_public_single_line'} <= names,
            'merged owning runtime controls unavailable; no model calls permitted')
    require(dict(ActorContract.champion().feature_masks)['transition_token_budget'] == 0,
            'cost producer requires exact zero-history-token champion contract')
    return limit


def select_roster(candidates):
    """Hash-ranked metadata addresses, greedy quotas, no outcome-driven refill."""
    ordered = sorted(candidates, key=lambda row: (digest(row), canonical(row)))
    used, seats, selected = set(), dict(p1=0, p2=0), []
    for row in ordered:
        require(row['split'] == 'train' and row['seat'] in seats, 'TRAIN own-actionable metadata required')
        if row['battle_id'] in used or seats[row['seat']] == 10:
            continue
        selected.append(dict(row, metadata_rank_sha256=digest(row)))
        used.add(row['battle_id']); seats[row['seat']] += 1
        if len(selected) == COUNT:
            break
    require(len(selected) == COUNT and seats == dict(p1=10, p2=10),
            'fixed metadata roster insufficient; no refill/redraw permitted')
    return selected


def _battle_path(collection, row):
    return Path(collection)/f"battle-{row['index']:04d}"/'boundaries.jsonl.gz'


def _owner_input(collection, row):
    """Only selected owner's literals; observation/reference arrays stay skipped."""
    prior = []
    prefixes = {seat:_Prefix() for seat in ('p1','p2')}
    requests = dict.fromkeys(prefixes)
    found = None
    for boundary in metadata_rows(_battle_path(collection, row), row['source_file_sha256'],
                                   META_FIELDS | {'owning_wire'}):
        require(boundary.get('battle_id') == row['battle_id']
            and (boundary.get('event') == 'end' or boundary.get('seed') == row['seed']),
                'captured battle identity drift')
        if found is not None:
            continue  # Finish source hash verification without examining later owner values.
        require(boundary.get('event') is None, 'selected boundary missing before terminal')
        before = prefixes[row['seat']].sha256()
        for seat in ('p1','p2'):
            wire = boundary['owning_wire'][seat]
            lines = wire['delta']
            require(type(lines) is list and all(type(s) is str and s.startswith('|')
                and '\n' not in s and '\r' not in s and not _terminal(s) for s in lines),
                'literal nonterminal captured lines required')
            prefix = prefixes[seat]
            require(wire['start'] == prefix.count and type(wire['start']) is int
                and wire['end'] == prefix.count + len(lines) and type(wire['end']) is int,
                'captured owner cursor discontinuity')
            prefixes[seat] = prefix.append(lines)
            require(prefixes[seat].sha256() == wire['prefix_sha256'], 'captured owner prefix mismatch')
            for line in lines:
                if line.startswith('|request|'):
                    _packet(line, seat)
                    requests[seat] = line
        lines = boundary['owning_wire'][row['seat']]['delta']
        prefix, request = prefixes[row['seat']], requests[row['seat']]
        if boundary['decision_round_index'] == row['decision_round_index']:
            require(boundary['root_id'] == row['root_id'] and row['seat'] in boundary['requested_players']
                and request is not None and _actionable(_packet(request, row['seat'])),
                'frozen selected own-actionable address differs; no replacement')
            binding = dict(own_prior_prefix_sha256=before, own_prefix_sha256=prefix.sha256(),
                own_delta_sha256=digest(lines), own_request_sha256=digest(_packet(request, row['seat'])),
                own_request_line_sha256=hashlib.sha256(request.encode()).hexdigest(),
                actor_prefix_sha256=hashlib.sha256(b''.join((s+'\n').encode() for s in prior+lines)).hexdigest(),
                prior_line_count=len(prior), delta_line_count=len(lines),
                host_request_cache_sha256=digest(requests),
                host_owner_prefixes_sha256={seat:p.sha256() for seat,p in prefixes.items()},
                requested_players=boundary['requested_players'])
            require(all(type(requests[s]) is str and _actionable(_packet(requests[s],s))
                == (s in boundary['requested_players']) for s in ('p1','p2')),
                'captured host request cache differs from ready roles')
            found = dict(prior=tuple(prior), delta=tuple(lines), binding=binding,
                host_requests=dict(requests), requested_players=tuple(boundary['requested_players']))
        else:
            require(boundary['decision_round_index'] < row['decision_round_index'], 'selected boundary skipped')
            prior.extend(lines)
    require(found is not None, 'selected root absent; no replacement')
    return found


def prepare_cost(*, study_prespec, collection, selection, selection_metadata, source_commit):
    """Explicit remote metadata preparation, with no model/encoder calls."""
    spec = study.read_bound(study_prespec)
    validate_controls(spec)
    study.validate_prespec(spec)
    from .actual_saved_wire import _selection, PRODUCER_SOURCE, PRODUCER_COHORT_FILE_SHA
    from .recorded_encoder_inventory import _inside
    from .recorded_mirror_roots import CHAMPION_CONTRACT
    root = Path(collection).resolve(strict=True)
    admission = spec['admission']
    require(selection['sha256'] == admission['selection_sha256']
        and selection_metadata['sha256'] == admission['selection_metadata_sha256'], 'selection binding mismatch')
    manifest, selected, metadata = _selection(root, Path(selection['path']), selection['sha256'],
        admission['collection_sha256'], Path(selection_metadata['path']), selection_metadata['sha256'])
    candidates = []
    for battle, frozen in zip(selected, metadata['selected']):
        receipt_path = _inside(root, f"battle-{battle['index']:04d}/battle.json")
        payload = receipt_path.read_bytes()
        require(hashlib.sha256(payload).hexdigest() == frozen['battle_receipt_sha256'],
                'battle receipt hash mismatch')
        # The receipt also carries initial requests and outcomes. Only its
        # identity, contract and file manifest are decoded for selection.
        receipt = metadata_object(payload.decode(), set(battle) | {'contract','files'})
        require(all(receipt.get(k) == v for k, v in battle.items())
            and receipt.get('contract') == CHAMPION_CONTRACT
            and receipt['files']['boundaries.jsonl.gz'] == frozen['source_file_sha256'],
            'champion battle receipt/source identity mismatch')
        boundaries = views = 0
        ended = False
        for row in metadata_rows(_battle_path(root, battle), frozen['source_file_sha256'], META_FIELDS):
            require(not ended and row.get('battle_id') == battle['battle_id']
                and (row.get('event') == 'end' or row.get('seed') == battle['seed']),
                    'fixed captured boundary identity/order mismatch')
            if row.get('event') == 'end':
                ended = True
                continue
            require(row.get('event') is None and type(row.get('decision_round_index')) is int
                and row['decision_round_index'] == boundaries and type(row.get('root_id')) is str,
                'fixed boundary address chronology required')
            requested = row['requested_players']
            require(type(requested) is list and requested and requested == [s for s in ('p1','p2') if s in requested],
                    'canonical own-actionable metadata roles required')
            for seat in requested:
                candidates.append(dict(index=battle['index'], battle_id=battle['battle_id'], seed=battle['seed'],
                    split='train', seat=seat, root_id=row['root_id'], decision_round_index=boundaries,
                    source_file_sha256=frozen['source_file_sha256'],
                    battle_receipt_sha256=frozen['battle_receipt_sha256'],
                    collection_sha256=admission['collection_sha256'], selection_sha256=selection['sha256'],
                    selection_metadata_sha256=selection_metadata['sha256']))
            boundaries += 1; views += len(requested)
        require(ended and boundaries == frozen['boundaries'] and views == frozen['requested_views'],
                'fixed B2 metadata denominator differs')
    roster = select_roster(candidates)
    # Selection is final before own request/literal values are examined. A bad
    # selected root refuses preparation, never draws a replacement.
    for row in roster:
        row.update(_owner_input(root, row)['binding'])
    repo = Path(__file__).resolve().parents[2]
    require(study._hex(source_commit, 40), 'exact cost source commit required')
    return dict(schema=SCHEMA, study_prespec=study._reference(study_prespec), collection=str(root),
        selection=study._reference(selection), selection_metadata=study._reference(selection_metadata),
        source_commit=source_commit, source_files={name:file_sha(repo/name) for name in SOURCE_FILES},
        producer_source=PRODUCER_SOURCE, cohort_file_sha256=PRODUCER_COHORT_FILE_SHA,
        roster=roster, roster_sha256=digest(roster), roots=COUNT, seats=dict(p1=10,p2=10),
        decision_limit_seconds=spec['owning_decision_timeout_seconds'],
        timer_scope='selected_delta_single_lines_live_boundary_guard_173jax_encode_raw_FP32_selection_choice',
        dispatch=PUBLIC_DISPATCH,
        transport_scope='configured_prospective_dispatch_not_historical_transport_chunks',
        policy_history='fresh_per_root_zero_transition_tokens_window1_mean',
        setup_and_prior_prefix_priming='record_separately_no_warmup_or_discarded_selected_samples',
        factory_lifecycle='one_model_factory_catalog_both_owners_fresh_per_root_registration_config_and_session',
        host_peer_request_scope='exact_captured_peer_request_only_in_existing_boundary_guard_never_actor_backend_or_JAX',
        reducer=dict(p95='nearest_rank_19_of_20', required_valid=20, threshold='actual_decision_limit/2',
                     unknowns='cannot_pass', retries=0, refill=False, automatic_B3_qualification=False))


def _validate_cost(spec):
    require(spec.get('schema') == SCHEMA and spec.get('roots') == COUNT
        and len(spec.get('roster', ())) == COUNT and digest(spec['roster']) == spec.get('roster_sha256'),
        'fixed hash-bound 20-root cost prespec required')
    rows = spec['roster']
    require(len({r['battle_id'] for r in rows}) == COUNT
        and {s:sum(r['seat'] == s for r in rows) for s in ('p1','p2')} == dict(p1=10,p2=10)
        and all(r['split'] == 'train' for r in rows), '20 distinct TRAIN battles/10 per seat required')
    source = study.read_bound(spec['study_prespec'])
    limit = validate_controls(source)
    study.validate_prespec(source)
    require(spec['decision_limit_seconds'] == limit, 'cost/live learner limit drift')
    return source


def _source_identity(spec):
    repo = Path(__file__).resolve().parents[2]
    require(spec['source_files'] == {name:file_sha(repo/name) for name in SOURCE_FILES}, 'cost source bytes drift')
    revision = subprocess.run(['git','-C',str(repo),'rev-parse','HEAD'], check=True,
                              capture_output=True,text=True).stdout.strip()
    dirty = subprocess.run(['git','-C',str(repo),'status','--porcelain','--untracked-files=no'],
                           check=True,capture_output=True,text=True).stdout
    require(revision == spec['source_commit'] and not dirty, 'exact clean cost producer source required')
    # Recompute the deterministic metadata roster and every selected owning
    # binding; a self-consistent but hand-edited roster cannot change selection.
    expected = prepare_cost(study_prespec=spec['study_prespec'], collection=spec['collection'],
        selection=spec['selection'], selection_metadata=spec['selection_metadata'],
        source_commit=spec['source_commit'])
    require(canonical(expected) == canonical(spec), 'fixed cost prespec/metadata derivation drift')


def _runtime(spec):
    """One ordinary champion model and owning factory; no simulator creation."""
    precision = study._configure_precision()
    config = study._config(spec, 1)  # Fixed JAX/p1; per-root owner is registered separately.
    require(config.encoder_kind == 'jax-owning'
        and config.owning_decision_timeout_seconds == spec['owning_decision_timeout_seconds']
        and config.owning_public_single_line is True, 'actual config differs from bound live controls')
    from .live_protocol_runtime import champion_actor_factory
    factory = champion_actor_factory(config)
    from .neural_policy import (load_transformer_checkpoint, observation_spec_from_model_config,
        category_vocab_from_model_config, feature_masks_from_model_config)
    from .local_showdown import LocalShowdownConfig
    model, result = load_transformer_checkpoint(config.checkpoint, map_location='cpu')
    model.float().eval()
    env = LocalShowdownConfig(showdown_root=config.showdown_root, node_binary=config.node_binary,
        observation_spec=observation_spec_from_model_config(result.model_config),
        category_vocab=category_vocab_from_model_config(result.model_config, config.showdown_root),
        feature_masks=feature_masks_from_model_config(result.model_config))
    return factory, (model, result, env), precision, config


def _session(config, factory, runtime, row):
    from .live_protocol_bridge import OwningWireSession
    # The existing factory contract binds axis/model/catalog, not an owner.
    # Its immutable original config is retained. A fresh registration config
    # supplies this saved root's seat/seed; all scientific controls are copied.
    registration = replace(config,pokezero_player=row['seat'],seed_start=row['seed'],games=1)
    return OwningWireSession(registration,factory,runtime,seed=row['seed'],
        battle_id=row['battle_id'],start_seed=row['seed'])


def summarize(spec, ledger):
    _validate_cost(spec)
    require(ledger.get('schema') == LEDGER_SCHEMA
        and ledger['prespec_sha256'] == digest(spec) and len(ledger['roots']) == COUNT,
            'cost ledger/prespec mismatch')
    valid = []
    for planned, row in zip(spec['roster'], ledger['roots']):
        require(row['root'] == planned and row['status'] in
            ('VALID','ATTEMPTING_UNKNOWN','FAILED_UNKNOWN','NOT_ATTEMPTED'),
                'cost ledger roster/status drift')
        if row['status'] == 'VALID':
            value = row['decision_seconds']
            require(type(value) in (int,float) and math.isfinite(value) and value >= 0,
                    'invalid selected cost sample')
            valid.append(value)
    complete = (len(valid) == COUNT and ledger.get('actual_exit') == 0
        and ledger.get('status') == 'COMPLETE_REQUIRES_SCIENTIFIC_REVIEW'
        and ledger.get('failure') is None)
    p95 = sorted(valid)[18] if complete else None
    limit = spec['decision_limit_seconds']
    return dict(planned=COUNT, valid=len(valid), unknown=COUNT-len(valid), decision_limit_seconds=limit,
        threshold_seconds=limit/2, p95_seconds=p95, p95_rank=19 if complete else None,
        maximum_seconds=max(valid) if valid else None,
        cost_rule_satisfied=bool(complete and p95 <= limit/2),
        status='COMPLETE_REQUIRES_SCIENTIFIC_REVIEW' if complete else 'INCOMPLETE_UNKNOWN',
        B3_qualified=False, live_transport_chunks_measured=False)


def run_cost(spec, output, *, runtime_builder=None, input_loader=None, admission=None,
             source_check=None, session_builder=None, clock=None):
    """One non-resumable attempt. Injectable dependencies are for fake tests."""
    source = _validate_cost(spec)
    spec = json.loads(canonical(spec))
    output = Path(output)
    output.mkdir(exist_ok=False)
    study._sync_directory(output.parent)
    study.atomic_json(output/'prespec.json', spec)
    ledger = dict(schema=LEDGER_SCHEMA, prespec_sha256=digest(spec), status='PREPARING',
        roots=[dict(root=row,status='NOT_ATTEMPTED') for row in spec['roster']],
        actual_exit=None, failure=None, B3_qualified=False)
    def publish():
        ledger['summary'] = summarize(spec, ledger)
        study.atomic_json(output/'ledger.json', ledger)
    publish()
    clock = clock or time.perf_counter
    stage, current, actor = 'source', None, None
    try:
        (source_check or _source_identity)(spec)
        stage = 'B2_admission'
        ledger['admission'] = (admission or study.validate_admission)(source)
        stage = 'setup'
        started = clock()
        try:
            factory, runtime, precision, config = (runtime_builder or _runtime)(source)
        finally:
            ledger['setup_seconds'] = clock()-started
        ledger['runtime_precision'] = precision
        for ordinal, current in enumerate(ledger['roots']):
            stage = 'input'
            current['status'] = 'ATTEMPTING_UNKNOWN'
            ledger['status'] = 'RUNNING'
            # A hard interruption must not mislabel an already started root
            # as not attempted. An interrupted attempt has no resumable path.
            publish()
            row = current['root']
            data = (input_loader or _owner_input)(spec['collection'], row)
            require(data['binding'] == {key:row[key] for key in data['binding']}
                and digest(data['host_requests']) == row['host_request_cache_sha256']
                and list(data['requested_players']) == row['requested_players'], 'frozen owning input drift')
            stage = 'fresh_actor_catalog'
            started = clock()
            try:
                session = (session_builder or _session)(config,factory,runtime,row)
                actor = session.actor
                session.begin()
                session.started = True
                session.next_round = row['decision_round_index']
            finally:
                current['actor_catalog_seconds'] = clock()-started
            def feed(lines):
                for line in lines:
                    actor.accept_owner_stream(battle_id=row['battle_id'],owner=row['seat'],
                        generation=actor.generation,prefix_sha256=actor.prefix_sha256,lines=(line,))
            stage = 'prior_prefix_priming'
            started = clock()
            try:
                feed(data['prior'])
            finally:
                current['prior_priming_seconds'] = clock()-started
            stage = 'selected_decision'
            current['first_selected_encode_in_attempt'] = ordinal == 0
            started = clock()
            try:
                feed(data['delta'])
                snapshot = session.boundary(requested_players=data['requested_players'],
                    decision_round=row['decision_round_index'],requests=data['host_requests'])
                actor.decide(snapshot, rng=random.Random(f"{row['seed']}:{row['seat']}:{row['decision_round_index']}"))
                choice = actor.choice(snapshot)
            finally:
                elapsed = clock()-started
                current['decision_attempt_seconds'] = elapsed
            # The ordinary actor converts/validates all five device tensors
            # and executes FP32 RawPolicy selection before returning choice.
            require(type(choice) is str and choice and math.isfinite(elapsed) and elapsed >= 0,
                    'complete finite decision cost required')
            require(snapshot.prefix_sha256 == row['actor_prefix_sha256']
                and digest(_packet(snapshot.own_request_line,row['seat'])) == row['own_request_sha256'],
                'timed own snapshot provenance differs')
            current['decision_seconds'] = elapsed
            stage = 'close'
            current['actor_receipt'] = actor.close_game()
            actor = None
            require(current['actor_receipt']['status'] == 'ABORTED_UNQUALIFIED'
                and current['actor_receipt']['error'] is None and current['actor_receipt']['close_error'] is None,
                'selected nonterminal actor did not close cleanly')
            current['status'] = 'VALID'
            ledger['status'] = 'RUNNING'
            publish()
        ledger.update(status='COMPLETE_REQUIRES_SCIENTIFIC_REVIEW', actual_exit=0)
    except Exception as error:
        if actor is not None:
            try:
                receipt = actor.close_game()
                if current is not None:
                    current['actor_receipt'] = receipt
            except Exception as close_error:
                if current is not None:
                    current['close_error_type'] = type(close_error).__name__
        if current is not None:
            current['status'] = 'FAILED_UNKNOWN'
        ledger.update(status='FAILED_UNKNOWN',actual_exit=1,
            failure=dict(stage=stage,error_type=type(error).__name__,reason=str(error)))
    publish()
    return ledger
