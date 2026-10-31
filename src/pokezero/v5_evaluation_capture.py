"""Original nontraining capture resolver; importing it performs no collection.

The external protocol binds the produced bytes and the predeclared bank. This
checks those records against the ORIGINAL collector/branch schemas, rather than
treating arbitrary hex receipt tags or PPO targets as outcome evidence. A source
binding describes the producing code, not external attestation of a file's truth.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType
import weakref

from . import v5_contract as c, v5_run as run

ROSTER = 'pokezero.v5.game_roster.v1'
BANK = 'pokezero.v5.evaluation_capture_bank.v1'
CAPTURE = 'pokezero.v5.original_evaluation_capture.v1'
NONTRAINING = 'NONTRAINING_EVALUATION'
TRAINING = 'TRAINING_ROSTER'
POSITION = 'pokezero.v5.original_outcome_position.v1'
SIBLING = 'pokezero.v5.original_public_siblings.v1'
SOURCE_FILES = (
    'gpu_engine/v5_evaluation_capture.py', 'src/pokezero/v5_evaluation_capture.py',
    'gpu_engine/v5_selfplay.py', 'gpu_engine/v5_collection.py',
    'gpu_engine/v5_collection_data.py', 'gpu_engine/v5_owning_history.py',
    'gpu_engine/v5_public_world.py', 'gpu_engine/v5_public_branch.py',
    'gpu_engine/v5_observation.py', 'gpu_engine/transition.py',
    'gpu_engine/engine_retained_encoder_driver.py', 'gpu_engine/engine_source_frame_witness.py',
    'src/pokezero/v5_teacher_public_root.py', 'src/pokezero/v5_gumbel.py',
    'src/pokezero/determinization.py', 'src/pokezero/v5_contract.py',
)
_REGISTERED_SOURCE_FILES=SOURCE_FILES


def source_binding():
    if type(SOURCE_FILES) is not tuple or SOURCE_FILES!=_REGISTERED_SOURCE_FILES:
        raise ValueError('original capture source inventory changed')
    root = Path(__file__).resolve().parents[2]
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in SOURCE_FILES}


def _keys(value, names):
    if type(value) is not dict or set(value) != set(names):
        raise ValueError('complete original capture fields required')


def _artifact(value):
    from .v5_evaluation import Artifact
    _keys(value, ('path', 'sha256'))
    return Artifact(**value)


def _roster(reference, role):
    value = _artifact(reference).document()
    _keys(value, ('schema', 'role', 'arm', 'replicate', 'pool_sha256', 'batches'))
    if value['schema'] != ROSTER or value['role'] != role:
        raise ValueError('predeclared original game roster role required')
    identity = (value['arm'], value['replicate'])
    if type(value['replicate']) is not int or (role == TRAINING and identity not in (('A',0),('B',0),('D',0),('B',1))):
        raise ValueError('exact declared training arm/replicate required')
    if role == NONTRAINING and identity != ('EVALUATION', 0):
        raise ValueError('heldout roster must be separately declared nontraining')
    c._binding(value['pool_sha256'])
    if type(value['batches']) is not list or not value['batches']:
        raise ValueError('finite ordered original batches required')
    batches = []; games = {}
    for document in value['batches']:
        _keys(document, run.Batch.__dataclass_fields__)
        _keys_games = ('stream_id','battle_id','seed','team_indices')
        for game in document['games']:_keys(game, _keys_games)
        batch = run.Batch(**dict(document, games=tuple(run.Game(**g) for g in document['games'])))
        batches.append(batch)
        for game in batch.games:
            if game.stream_id in games:raise ValueError('roster reused a stream')
            games[game.stream_id] = game
    if (len({g.battle_id for g in games.values()}) != len(games)
            or len({t for g in games.values() for t in g.team_indices}) != 2 * len(games)):
        raise ValueError('roster reused a battle or original team row')
    return value, tuple(batches), games


@dataclass(frozen=True)
class CaptureBank:
    reference: object
    document: dict
    batches: tuple
    games: object
    selectors: tuple
    training_rosters: tuple


def load_bank(reference):
    """No model/evaluation-protocol hash in rosters: no declaration hash cycle."""
    reference = _artifact(reference); value = reference.document()
    _keys(value, ('schema','role','heldout_roster','training_rosters','selectors','world_recipe',
        'opponent_replies','collector_archive_sha256','opponent_archive_sha256','collector_seats','max_requests'))
    if value['schema'] != BANK or value['role'] != NONTRAINING:
        raise ValueError('predeclared NONTRAINING evaluation bank required')
    for key in ('collector_archive_sha256','opponent_archive_sha256'):c._binding(value[key])
    if type(value['max_requests']) is not int or value['max_requests']<=0:raise ValueError('fixed finite heldout request cap required')
    heldout, batches, games = _roster(value['heldout_roster'], NONTRAINING)
    references = value['training_rosters']
    if type(references) is not list or len(references) != 4 or len({_artifact(r).sha256 for r in references}) != 4:
        raise ValueError('all four distinct original training roster identities required')
    training = tuple(_roster(r, TRAINING)[0] for r in references)
    if {(r['arm'],r['replicate']) for r in training} != {('A',0),('B',0),('D',0),('B',1)}:
        raise ValueError('all initial arms and replicate must participate in heldout exclusion')
    streams = set(games); battles = {g.battle_id for g in games.values()}
    teams = {t for g in games.values() for t in g.team_indices}
    # Original empty_state owns fold_in(constructor_seed, batch row), whereas
    # Game.seed is only a trajectory identity. Renaming streams is not enough.
    roots={(b.constructor_seed,row) for b in batches for row,_ in enumerate(b.games)}
    genders={(b.gender_seed,b.gender_domain,root) for b in batches for root in b.gender_root_ids}
    if len(roots)!=len(games):raise ValueError('heldout reused an original constructor RNG root')
    for roster in training:
        for batch in roster['batches']:
            for row,game in enumerate(batch['games']):
                if (game['stream_id'] in streams or game['battle_id'] in battles
                        or (batch['constructor_seed'],row) in roots
                        or batch['gender_seed'] is not None and (batch['gender_seed'],batch['gender_domain'],batch['gender_root_ids'][row]) in genders
                        or roster['pool_sha256'] == heldout['pool_sha256'] and any(t in teams for t in game['team_indices'])):
                    raise ValueError('nontraining original game/stream/team/RNG identity overlaps a training arm')
    seats = value['collector_seats']
    if type(seats) is not list or len(seats) != len(games) or any(type(s) is not int or s not in (0,1) for s in seats):
        raise ValueError('predeclared collector seat for every heldout game required')
    seat_by_stream = dict(zip(games, seats, strict=True)); selectors = []
    for item in value['selectors']:
        _keys(item, ('stream','frame','owner')); address = c.DecisionAddress(**item)
        if address.stream not in games or address.owner != seat_by_stream[address.stream]:
            raise ValueError('fixed selector is outside the original collector bank/seat')
        selectors.append(address)
    if not selectors or len(set(selectors)) != len(selectors):raise ValueError('fixed unique heldout selectors required')
    _keys(value['world_recipe'], ('seed',))
    from .v5_gumbel import Recipe
    Recipe(**value['world_recipe'])
    replies = value['opponent_replies']
    if type(replies) is not list or len(replies) != len(selectors) or any(
            type(row) is not list or len(row) != 2 or any(type(x) is not int or not 0<=x<9 for x in row) for row in replies):
        raise ValueError('two fixed literal replies per selected root required')
    return CaptureBank(reference, value, batches, MappingProxyType(games), tuple(selectors), training)


def check_training_declaration(bank, request):
    declaration = json.loads((request.run_directory / 'declaration.json').read_bytes())
    if run._sha(declaration) != request.declaration_sha256:raise ValueError('accepted run declaration differs')
    identity = (declaration['model']['arm'], declaration['model']['replicate'])
    roster = next((r for r in bank.training_rosters if (r['arm'],r['replicate']) == identity), None)
    if (roster is None or roster['pool_sha256'] != declaration['pool_sha256']
            or roster['batches'] != [batch for iteration in declaration['iterations'] for batch in iteration]):
        raise ValueError('actual accepted Plan is not its independently fixed training roster')


def observation_to_dict(observation):
    if type(observation) is not c.V5Observation or observation.view != 'PUBLIC':
        raise ValueError('actual public observation only, never a fake trajectory child')
    value = {name:getattr(observation,name) for name in ('categorical_ids','numeric_features','token_type_ids',
        'attention_mask','legal_action_mask','category_vocabulary_sha256','view','schema_version')}
    value.update(address=asdict(observation.address), label_vocabulary=asdict(observation.label_vocabulary),
        posteriors=[{key:getattr(p,key) for key in ('species','effective_mask','prior_weights','reason','pin_conflict')}
            for p in observation.posteriors])
    return value


def observation_from_dict(value):
    _keys(value, ('address','label_vocabulary','posteriors','categorical_ids','numeric_features','token_type_ids',
        'attention_mask','legal_action_mask','category_vocabulary_sha256','view','schema_version'))
    vocabulary = c.LabelVocabulary(**value['label_vocabulary'])
    return c.V5Observation(**dict({k:v for k,v in value.items() if k!='label_vocabulary'},
        address=c.DecisionAddress(**value['address']),posteriors=tuple(c.PublicPosterior(vocabulary,**p) for p in value['posteriors'])))


def _header(protocol, key, schema):
    from .v5_evaluation import Artifact
    reference = Artifact(**protocol[key]); reference.verify()
    with Path(reference.path).open() as stream:header = json.loads(next(stream))
    _keys(header, ('schema','split','rows','category_vocabulary_sha256','label_vocabulary_sha256','capture_bank','capture_source'))
    if (header['schema'] != schema or header['split'] != 'PHASE_C_FIXED_HELDOUT'
            or header['capture_source'] != source_binding()):
        raise run.MissingDependency('original reviewed capture code/inventory required, not fixture metadata')
    c._binding(header['category_vocabulary_sha256']); c._binding(header['label_vocabulary_sha256'])
    return header


def _terminal(record, game, pool_sha):
    from gpu_engine.v5_collection_data import TerminalCapture, sha
    _keys(record, ('body','capture'))
    body = record['body']; terminal = TerminalCapture(**record['capture'])
    expected = dict(stream=game.stream_id,frame=terminal.frame,status=terminal.status,reason=body.get('reason'),pool_sha256=pool_sha)
    if terminal.status == 'COMPLETE':expected.update(winner=terminal.winner,hp=[list(x) for x in terminal.hp],max_hp=[list(x) for x in terminal.max_hp])
    if (body != expected or sha(body) != terminal.source_sha256
            or body['reason'] is None or type(body['reason']) is not str):
        raise ValueError('literal original collector terminal body differs')
    return terminal


@dataclass(frozen=True,eq=False)
class OriginalCapture:
    bank: CaptureBank
    positions: tuple
    groups: tuple
    protocol_sha256: str


_RESOLVED=weakref.WeakKeyDictionary()


def _capture_digest(capture):
    return run._sha(dict(bank=asdict(capture.bank.reference),bank_document=capture.bank.document,
        positions=[dict(address=asdict(a),observation=None if o is None else observation_to_dict(o),outcome=y)
            for a,o,y in capture.positions],groups=capture.groups,protocol=capture.protocol_sha256))


def _checked_capture(capture,protocol):
    expected=_RESOLVED.get(capture) if type(capture) is OriginalCapture else None
    if expected is None or capture.protocol_sha256!=run._sha(protocol) or _capture_digest(capture)!=expected:
        raise ValueError('original immutable resolved capture for this exact protocol required')


def require_original_capture(protocol):
    """Validate genuine schemas/whole denominators before any FP allocation."""
    return resolve_original_capture(protocol, None)


def resolve_original_capture(protocol, request):
    from . import v5_evaluation as evaluation, v5_data
    from gpu_engine import v5_collection_data as original
    headers = [_header(protocol,key,schema) for key,schema in (('positions',evaluation.POSITIONS),('siblings',evaluation.SIBLINGS))]
    first, second = headers
    if any(first[key] != second[key] for key in ('capture_bank','capture_source','category_vocabulary_sha256','label_vocabulary_sha256')):
        raise ValueError('positions and siblings must name the same original capture bank/source')
    if first['label_vocabulary_sha256'] != protocol['label_vocabulary_sha256']:raise ValueError('capture label source differs')
    bank = load_bank(first['capture_bank'])
    if request is not None:check_training_declaration(bank, request)
    positions = []; by_address = {}; terminals = {}
    heldout = _artifact(bank.document['heldout_roster']).document()
    for index,(_,record) in enumerate(evaluation._records(evaluation.Artifact(**protocol['positions']),evaluation.POSITIONS)):
        _keys(record, ('schema','selector','game','status','decision','execution','terminal','reason'))
        address = c.DecisionAddress(**record['selector'])
        if index >= len(bank.selectors) or address != bank.selectors[index]:raise ValueError('fixed selector order/denominator differs')
        game = bank.games[address.stream]
        if record['schema'] != POSITION or record['game'] != json.loads(run._json(asdict(game))):raise ValueError('original game identity differs')
        terminal = None
        if record['terminal'] is not None:
            terminal = _terminal(record['terminal'],game,heldout['pool_sha256'])
            if address.stream in terminals and terminals[address.stream] != terminal:raise ValueError('mutated terminal within one original game')
            terminals[address.stream] = terminal
        if record['status'] == 'UNKNOWN':
            if record['decision'] is not None or record['execution'] is not None or type(record['reason']) is not str or not record['reason']:
                raise ValueError('unobserved selector must remain explicitly unknown')
            positions.append((address,None,None));by_address[address]=None;continue
        if record['status'] != 'CAPTURED' or record['reason'] is not None:raise ValueError('original decision capture status differs')
        row = v5_data.row_from_dict(record['decision']); metadata = row.step.metadata.get('v5_capture')
        if type(metadata) is not dict:raise ValueError('missing original capture_decision receipt')
        captured = original.CapturedDecision(row,tuple(metadata['physical']))
        execution = record['execution']
        _keys(execution, ('stream','before_frame','after_frame','executed','invalid','source_frame_sha256','after_state','raw_frame'))
        if (row.observation.address != address or (row.battle_id,row.seed) != (game.battle_id,game.seed)
                or row.privileged is not None or row.observation.view != 'PUBLIC'
                or row.observation.category_vocabulary_sha256 != first['category_vocabulary_sha256']
                or row.observation.label_vocabulary.sha256 != first['label_vocabulary_sha256']
                or metadata['pool_sha256'] != heldout['pool_sha256'] or tuple(metadata['team_indices']) != game.team_indices
                or metadata['actor_archive_sha256'] != bank.document['collector_archive_sha256']
                or metadata['opponent_archive_sha256'] != bank.document['opponent_archive_sha256']
                or execution['stream'] != address.stream or execution['before_frame'] != address.frame
                or execution['after_frame'] != address.frame+1 or execution['executed'] is not True or execution['invalid'] is not False
                or execution['source_frame_sha256'] != row.source_frame_sha256
                or type(execution['after_state']) is not dict or type(execution['raw_frame']) is not dict):
            raise ValueError('actual same-call collector execution/input binding differs')
        # The original finish routine alone attaches outcome and physical labels.
        if row.labels.terminal_status != 'IN_PROGRESS' or row.terminal_source_sha256 is not None:
            raise ValueError('unfinalized original decision required; no returned/GAE outcome substitute')
        outcome = None
        if terminal is not None:
            finished = original.finish((captured,),(terminal,))[0].row
            outcome = finished.labels.outcome if finished.labels.terminal_status == 'COMPLETE' else None
        positions.append((address,row.observation,outcome));by_address[address]=row
    if len(positions) != len(bank.selectors):raise ValueError('missing fixed position denominator')
    groups = []
    records = list(evaluation._records(evaluation.Artifact(**protocol['siblings']),evaluation.SIBLINGS))
    if len(records) != 2*len(bank.selectors):raise ValueError('both declared worlds retained for every selector')
    for ordinal,(_,group) in enumerate(records):
        _keys(group, ('schema','selector','world','reply','status','reason','root_source_sha256','origin',
            'chance','root_legal','branches'))
        index,world = divmod(ordinal,2); address = bank.selectors[index]; root = by_address[address]
        if (group['schema'] != SIBLING or group['selector'] != asdict(address) or group['world'] != world
                or group['reply'] != bank.document['opponent_replies'][index][world]):
            raise ValueError('actual declared sibling root/world/reply order differs')
        if group['status'] == 'UNKNOWN':
            if group['branches'] or type(group['reason']) is not str or not group['reason']:
                raise ValueError('unknown world must not manufacture child values')
            groups.append(group);continue
        if (group['status'] != 'CAPTURED' or group['reason'] is not None or root is None
                or group['root_source_sha256'] != root.source_frame_sha256
                or group['root_legal'] != list(root.observation.legal_action_mask)):
            raise ValueError('original sibling root source/public legal mask differs')
        origin=group['origin'];_keys(origin,('body','sha256','history_sha256','history','owning_lines','before_state','root_view',
            'sampled_start','posterior_seed','kernel_seed'))
        from .v5_gumbel import Recipe
        recipe=Recipe(**bank.document['world_recipe'])
        if (run._sha(origin['body']) != origin['sha256'] or origin['body']['scope'] != 'HYPOTHETICAL_PUBLIC_WORLD_NOT_ORDINARY_SESSION'
                or origin['posterior_seed'] != recipe.world_seed(address,world)
                or origin['kernel_seed'] != int.from_bytes(recipe._digest(address,'hypothetical-kernel',world)[:4],'big')):
            raise ValueError('original sampled world origin differs')
        history=origin['history'];_keys(history,('stream','owner','own_constructor','frames'))
        if (run._sha(history)!=origin['history_sha256'] or (history['stream'],history['owner'])!=(address.stream,address.owner)
                or not history['frames'] or history['frames'][-1]['number']!=address.frame
                or type(origin['owning_lines']) is not list or not origin['owning_lines']
                or any(type(line) is not str for line in origin['owning_lines'])
                or hashlib.sha256(''.join(line+'\n' for line in origin['owning_lines']).encode()).hexdigest()!=origin['body']['root']
                or origin['sampled_start']['player_teams']!=origin['body']['sampled_teams']
                or origin['before_state'].get('done')!=[False]):
            raise ValueError('original owning history/sampled start record is missing or differs')
        import numpy as np
        view=origin['root_view'];_keys(view,('schema','observation_schema','category_vocabulary_sha256','label_vocabulary_sha256','view',
            'categorical_ids','numeric_features','token_type_ids','attention_mask','history_mask'))
        if (view['schema']!=c.TRAINING_CONTRACT_SCHEMA or view['observation_schema']!=c.OBSERVATION_SCHEMA_VERSION_V5
                or view['category_vocabulary_sha256']!=root.observation.category_vocabulary_sha256
                or view['label_vocabulary_sha256']!=root.observation.label_vocabulary.sha256 or view['view']!='PUBLIC'):
            raise ValueError('original root policy source namespace differs')
        for name in ('categorical_ids','numeric_features','token_type_ids','attention_mask'):
            dtype=np.float32 if name=='numeric_features' else None
            expected=np.asarray(getattr(root.observation,name),dtype=dtype)[None,None]
            if not np.array_equal(np.asarray(view[name],dtype=dtype),expected):
                raise ValueError('same original public root policy bytes differ')
        if view['history_mask']!=[[True]]:raise ValueError('literal original root history mask differs')
        chance=group['chance'];_keys(chance,('fields','sha256'))
        if run._sha(chance['fields']) != chance['sha256']:raise ValueError('original once-per-world chance differs')
        legal=[a for a,ok in enumerate(root.observation.legal_action_mask) if ok]
        if [b['action'] for b in group['branches']] != legal:raise ValueError('complete original legal sibling denominator required')
        for branch in group['branches']:
            _keys(branch,('action','choices','status','refusals','state','raw_frame','diagnostics','observation','terminal','receipt_sha256'))
            returned={k:v for k,v in branch.items() if k!='receipt_sha256'}
            if branch['receipt_sha256']!=run._sha(dict(selector=asdict(address),world=world,reply=group['reply'],
                    origin=origin['sha256'],chance=group['chance']['sha256'],returned=returned)):
                raise ValueError('same original returned branch receipt differs')
            choices=[group['reply']]*2;choices[address.owner]=branch['action']
            if branch['choices'] != choices:raise ValueError('actual root/reply physical seats differ')
            if branch['status'] == 'UNKNOWN':
                if branch['observation'] is not None or branch['terminal'] is not None or not branch['refusals']:
                    raise ValueError('refused branch cannot carry a value/outcome')
                continue
            if branch['status'] != 'CAPTURED' or branch['refusals'] or type(branch['state']) is not dict or type(branch['raw_frame']) is not dict:
                raise ValueError('same actual returned public branch state required')
            state=branch['state'];done=state['done'];winner=state['winner']
            if done != [True] and done != [False]:raise ValueError('literal B1 returned done bit required')
            if done == [True]:
                terminal=branch['terminal']
                if (branch['observation'] is not None or terminal != dict(state_sha256=run._sha(state),winner=winner[0])
                        or type(winner[0]) is not int or winner[0] not in (0,1,2)):
                    raise ValueError('only actual returned child terminal may supply a terminal outcome')
            else:
                if branch['terminal'] is not None:raise ValueError('missing child terminal continuation remains unknown')
                observation=observation_from_dict(branch['observation'])
                if (observation.address != c.DecisionAddress(address.stream,address.frame+1,address.owner)
                        or observation.view != 'PUBLIC' or observation.category_vocabulary_sha256 != first['category_vocabulary_sha256']
                        or observation.label_vocabulary.sha256 != first['label_vocabulary_sha256']):
                    raise ValueError('original public successor identity/source differs')
        groups.append(group)
    result=OriginalCapture(bank,tuple(positions),tuple(groups),run._sha(protocol))
    _RESOLVED[result]=_capture_digest(result)
    return result


def evaluate_original_capture(protocol, factory, capture):
    """Pure forward aggregation after genuine source resolution; no data access."""
    _checked_capture(capture,protocol)
    import numpy as np
    from gpu_engine import v5_model
    from gpu_engine.v5_live_protocol import check_factory
    from .v5_evaluation import _columns
    from .value_precision_probe import calibration_metrics
    from types import SimpleNamespace
    def predict(observations):
        values=[]
        for start in range(0,len(observations),protocol['batch_size']):
            check_factory(factory)
            rows=[SimpleNamespace(observation=o) for o in observations[start:start+protocol['batch_size']]]
            output=factory.forward(v5_model.prepare_inputs(_columns(rows,factory.config),factory.config))
            a=np.asarray(output.value,dtype=np.float64)
            if not np.asarray(output.valid).all() or a.shape!=(len(rows),) or not np.isfinite(a).all():
                raise ValueError('original frozen public head refused heldout input')
            values.extend(map(float,a))
        return values
    known=[(o,y) for _,o,y in capture.positions if o is not None and y is not None]
    if not known:raise run.MissingDependency('no actual uncapped collector-policy terminal outcomes for calibration')
    metrics=calibration_metrics(predict([o for o,_ in known]),[y for _,y in known],bins=protocol['bins']);rmse=math.sqrt(metrics['mse'])
    rows=[];unknown=0;terminal_branches=0
    for group in capture.groups:
        if group['status']!='CAPTURED' or any(b['status']!='CAPTURED' for b in group['branches']):unknown+=1;continue
        if len(group['branches'])<2:unknown+=1;continue
        # Literal returned terminal outcomes are measured branches, not missing
        # leaves. Only genuine nonterminal public children enter the head.
        indices=[i for i,b in enumerate(group['branches']) if b['terminal'] is None]
        predicted=predict([observation_from_dict(group['branches'][i]['observation']) for i in indices])
        values=[None]*len(group['branches']);sources=[]
        for i,value in zip(indices,predicted,strict=True):values[i]=value
        owner=group['selector']['owner']
        for i,branch in enumerate(group['branches']):
            terminal=branch['terminal'];sources.append('CURRENT_PUBLIC_HEAD' if terminal is None else 'ACTUAL_TERMINAL_WINNER')
            if terminal is not None:
                winner=terminal['winner'];values[i]=0. if winner==2 else 1. if winner==owner else -1.
                terminal_branches+=1
        order=sorted(range(len(values)),key=lambda i:(-values[i],group['branches'][i]['action']))[:2]
        gap=values[order[0]]-values[order[1]]
        rows.append(dict(selector=group['selector'],world=group['world'],reply=group['reply'],
            actions=[b['action'] for b in group['branches']],values=values,value_sources=sources,
            top_two_actions=[group['branches'][i]['action'] for i in order],gap=gap,gap_exceeds_outcome_rmse=gap>rmse))
    _checked_capture(capture,protocol)
    return dict(calibration=metrics,heldout_positions=len(capture.positions),known_outcomes=len(known),
        unknown_outcomes=len(capture.positions)-len(known),outcome_rmse=rmse,sibling_groups=len(capture.groups),
        unknown_sibling_groups=unknown,terminal_groups_excluded=0,actual_terminal_branches=terminal_branches,
        scored_sibling_groups=len(rows),top_two_rows=rows,
        share_gap_exceeds_outcome_rmse=sum(r['gap_exceeds_outcome_rmse'] for r in rows)/len(rows) if rows else None,
        interpretation='Descriptive public successor gap versus fixed collector-policy terminal-return RMSE; not each evaluated arm counterfactual value.',
        actual_ranking_accuracy='UNMEASURED_NO_MATCHED_TERMINAL_CONTINUATIONS',fitted_models=0,
        bank_role=NONTRAINING,scientific_qualification=False)
