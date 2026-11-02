"""Fixed device heldout-bank schema/resolver. Import performs no collection.

Raw float32 public29 tensors are preserved, not rebuilt through Python's
double-precision posterior constructor. Ordinary terminal outcomes calibrate
the head; one-step siblings support descriptive gaps, never ranking truth.
"""
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from . import v5_contract as c, v5_run as run

DECLARATION = 'pokezero.device_heldout_declaration.v1'
POSITIONS = 'pokezero.device_fixed_outcome_positions.v1'
SIBLINGS = 'pokezero.device_actual_fixed_siblings.v1'
ROOT_INDEX = 'pokezero.device_heldout_saved_roots.v1'
OBSERVATION = 'pokezero.device_heldout_public29.v1'
STOP_RULE = 'ONE_DECLARED_256_GAME_COLLECTION_NO_REFILL_ONE_PRODUCER_REPAIR_IF_INVALID'
FRAMES = (0, 4, 12, 24)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def read_artifact(reference):
    from .v5_evaluation import Artifact
    return Artifact(**reference).document()


def cost_selectors(games, seed):
    """Exactly40 hash-selected addresses, chosen before outcomes are present."""
    from .v5_gumbel import Recipe
    recipe = Recipe(seed); used = set(); result = []
    for frame in FRAMES:
        selected = 0
        for ordinal, game in enumerate(games):
            address = c.DecisionAddress(game.stream_id, frame, ordinal % 2)
            if game.stream_id not in used and recipe.selected(address):
                result.append(dict(stream=address.stream, frame=frame, owner=address.owner))
                used.add(game.stream_id); selected += 1
                if selected == 10: break
        if selected != 10: raise ValueError('declared streams cannot fill40 fixed cost selectors')
    return result


def declaration(reference):
    value = read_artifact(reference)
    required = {'schema','role','stop_rule','source_commit','source_sha256','pool_sha256',
        'category_vocabulary_sha256','category_count','label_vocabulary','actors','batches',
        'max_requests','output_directory','training_roster_index','cost_recipe_seed','cost_selectors'}
    if set(value) != required or value['schema'] != DECLARATION or value['role'] != 'NONTRAINING_EVALUATION' or value['stop_rule'] != STOP_RULE:
        raise ValueError('explicit prospective nontraining device-bank declaration required')
    if (type(value['source_commit']) is not str or len(value['source_commit']) != 40
            or any(ch not in '0123456789abcdef' for ch in value['source_commit'])):
        raise ValueError('exact frozen device source required')
    for key in ('source_sha256','pool_sha256','category_vocabulary_sha256'): c._binding(value[key])
    labels = c.LabelVocabulary(**value['label_vocabulary'])
    if type(value['category_count']) is not int or value['category_count'] <= 0:
        raise ValueError('actual category namespace size required')
    if type(value['max_requests']) is not int or value['max_requests'] != 1024 or not Path(value['output_directory']).is_absolute():
        raise ValueError('fixed original1024 request cap and absolute fresh output required')
    if type(value['actors']) is not list or len(value['actors']) != 2:
        raise ValueError('two frozen original public actors required')
    for actor in value['actors']:
        if set(actor) != {'archive','checkpoint_sha256'}: raise ValueError('exact archive/checkpoint binding required')
        c._binding(actor['archive']['sha256']); c._binding(actor['checkpoint_sha256'])
        if not Path(actor['archive']['path']).is_absolute(): raise ValueError('absolute original archive required')
    batches = tuple(run.Batch(**dict(row, games=tuple(run.Game(**g) for g in row['games']))) for row in value['batches'])
    games = tuple(g for batch in batches for g in batch.games)
    if len(batches) != 16 or any(len(b.games) != 16 for b in batches) or len(games) != 256:
        raise ValueError('fixed16x16 genuine games required')
    if (len({g.stream_id for g in games}) != 256 or len({g.battle_id for g in games}) != 256
            or len({i for g in games for i in g.team_indices}) != 512
            or len({(b.constructor_seed,r) for b in batches for r in range(16)}) != 256):
        raise ValueError('declared game/stream/team/constructor roots overlap')
    # Root owns prospective disjointness against this already-fixed four-arm
    # index; the bank preserves that original reference, never a training role.
    index = value['training_roster_index']
    c._binding(index['sha256'])
    if not Path(index['path']).is_absolute(): raise ValueError('original training-index exclusion reference required')
    if value['cost_selectors'] != cost_selectors(games, value['cost_recipe_seed']):
        raise ValueError('cost selectors differ from fixed pre-outcome hash selection')
    return value, batches, games, labels


def selectors(games):
    return tuple(dict(stream=g.stream_id, frame=f, owner=i % 2) for i,g in enumerate(games) for f in FRAMES)


def _freeze(value):
    if type(value) is dict: return MappingProxyType({k:_freeze(v) for k,v in value.items()})
    if type(value) is list: return tuple(_freeze(v) for v in value)
    return value


@dataclass(frozen=True)
class PublicObservation:
    document: object
    category_vocabulary_sha256: str
    label_vocabulary: c.LabelVocabulary
    view: str = 'PUBLIC'

    def model_inputs(self):
        return {k:self.document[k] for k in ('numeric_features','categorical_ids','token_type_ids','attention_mask','legal_action_mask')}


def observation(value, header, labels):
    if not isinstance(value, (dict, MappingProxyType)) or set(value) != {'schema','numeric_features','categorical_ids','token_type_ids','attention_mask','legal_action_mask'} or value['schema'] != OBSERVATION:
        raise ValueError('literal device public observation required')
    for key,width,kind in (('numeric_features',132,(int,float)),('categorical_ids',41,(int,))):
        rows = value[key]
        if type(rows) not in (list,tuple) or len(rows) != 29 or any(type(r) not in (list,tuple) or len(r) != width for r in rows):
            raise ValueError('actual29token device tensor geometry differs')
        for row in rows:
            for x in row:
                if type(x) not in kind or not math.isfinite(x): raise ValueError('finite literal device tensor required')
                if key == 'categorical_ids' and not 0 <= x < header['category_count']:
                    raise ValueError('device category outside original namespace')
    if tuple(value['token_type_ids']) != (0,)+(1,)*6+(2,)*6+(3,)*9+(5,)+(7,)*6:
        raise ValueError('canonical public29 roles required')
    for key,n in (('attention_mask',29),('legal_action_mask',9)):
        if type(value[key]) not in (list,tuple) or len(value[key]) != n or any(type(x) is not bool for x in value[key]):
            raise ValueError('literal device masks required')
    return PublicObservation(_freeze(value), header['category_vocabulary_sha256'], labels)


@dataclass(frozen=True)
class BankData:
    positions: tuple
    groups: tuple
    declaration: object


def _records(reference, schema):
    from .v5_evaluation import Artifact
    artifact = Artifact(**reference); blob = Path(artifact.path).read_bytes()
    if hashlib.sha256(blob).hexdigest() != artifact.sha256: raise ValueError('device bank bytes differ')
    lines = blob.splitlines(); header = json.loads(lines[0]); rows = [json.loads(line) for line in lines[1:]]
    if header['schema'] != schema or header['rows'] != len(rows):
        raise ValueError('complete fixed device-bank denominator required')
    return header, rows


def require_device_bank(protocol):
    return resolve_device_bank(protocol, None)


def resolve_device_bank(protocol, request):
    ph, positions = _records(protocol['positions'], POSITIONS)
    sh, groups = _records(protocol['siblings'], SIBLINGS)
    if (ph['split']!='PHASE_C_FIXED_HELDOUT' or sh['split']!=ph['split']
            or any(ph[k] != sh[k] for k in ('device_bank','device_source','category_vocabulary_sha256','label_vocabulary_sha256','category_count'))):
        raise ValueError('positions/siblings name different device banks')
    plan, _, games, labels = declaration(ph['device_bank']); expected = selectors(games)
    if (ph['device_source'] != plan['source_commit'] or ph['category_vocabulary_sha256'] != plan['category_vocabulary_sha256']
            or ph['label_vocabulary_sha256'] != labels.sha256 or labels.sha256 != protocol['label_vocabulary_sha256']
            or ph['category_count'] != plan['category_count'] or len(positions) != 1024 or len(groups) != 1024):
        raise ValueError('actual fixed bank source/namespaces differ')
    if request is not None:
        # Same source label/category namespaces across arms; heldout outcomes
        # are not relabeled as the currently evaluated arm's policy outcomes.
        model = request.model if hasattr(request, 'model') else None
        if model is not None and model.label_vocabulary.sha256 != labels.sha256:
            raise ValueError('evaluation model label namespace differs')
    resolved = []
    for selector, row, group in zip(expected, positions, groups, strict=True):
        if row['selector'] != selector or group['selector'] != selector:
            raise ValueError('fixed selector order changed or omitted')
        if row['status'] not in ('OBSERVED','UNREACHED','REFUSED','FAILED','NOT_ATTEMPTED'):
            raise ValueError('literal selected-root disposition required')
        obs = None if row['observation'] is None else observation(row['observation'], ph, labels)
        if (row['status'] == 'OBSERVED') != (obs is not None): raise ValueError('root observation disposition differs')
        terminal = row['terminal']; outcome = None
        if terminal is not None:
            if terminal['status'] not in ('COMPLETE','CAPPED','REFUSED','FAILED'): raise ValueError('actual game disposition required')
            if terminal['status'] == 'COMPLETE':
                if type(terminal['winner']) is not int or terminal['winner'] not in (0,1,2) or terminal['done'] is not True: raise ValueError('literal completed winner required')
                outcome = 0 if terminal['winner'] == 2 else 1 if terminal['winner'] == selector['owner'] else -1
            elif terminal['winner'] is not None or terminal['done'] is not False: raise ValueError('unknown game cannot invent outcome')
        resolved.append((c.DecisionAddress(**selector), obs, outcome if obs is not None else None))
        if group['status'] == 'OBSERVED':
            if obs is None or group['root_id'] != row['root_id'] or group['world_id'] != row['world_id']:
                raise ValueError('siblings must share the actual saved root/world')
            if digest(group['chance']) != group['chance_id'] or type(group['reply']) is not int or not 0 <= group['reply'] < 9:
                raise ValueError('same literal chance and fixed reply required')
            legal = [a for a,ok in enumerate(obs.document['legal_action_mask']) if ok]
            if [b['action'] for b in group['branches']] != legal: raise ValueError('every legal sibling required once before values')
            for branch in group['branches']:
                if any(branch[k] != group[k] for k in ('root_id','world_id','chance_id','reply')):
                    raise ValueError('sibling root/world/chance/reply changed')
                if branch['status'] == 'OBSERVED':
                    observation(branch['observation'], ph, labels)
                    if branch['terminal'] is not None:raise ValueError('nonterminal child cannot invent outcome')
                elif branch['status'] == 'COMPLETE':
                    if branch['terminal']['done'] is not True or type(branch['terminal']['winner']) is not int or branch['terminal']['winner'] not in (0,1,2): raise ValueError('actual child terminal required')
                    if branch['observation'] is not None: raise ValueError('terminal child cannot invent successor')
                elif branch['status'] not in ('REFUSED','FAILED') or branch['observation'] is not None or branch['terminal'] is not None: raise ValueError('unknown sibling disposition required')
        elif group['branches'] or group['status'] not in ('UNREACHED','REFUSED','FAILED','NOT_ATTEMPTED'):
            raise ValueError('unknown root cannot manufacture sibling branches')
    return BankData(tuple(resolved), tuple(_freeze(g) for g in groups), _freeze(plan))


def evaluate_device_bank(protocol, factory, data):
    from .value_precision_probe import calibration_metrics
    from .v5_evaluation import _columns
    from gpu_engine import v5_model
    from gpu_engine.v5_live_protocol import check_factory
    import numpy as np
    if type(data) is not BankData: raise ValueError('resolved fixed device bank required')
    labels = factory.config.label_vocabulary
    header = dict(category_vocabulary_sha256=factory.config.category_vocabulary_sha256, category_count=factory.config.category_vocab_size)
    def predict(observations):
        values = []
        for start in range(0,len(observations),protocol['batch_size']):
            check_factory(factory)
            rows = [SimpleNamespace(observation=o) for o in observations[start:start+protocol['batch_size']]]
            result = factory.forward(v5_model.prepare_inputs(_columns(rows,factory.config),factory.config))
            if not np.asarray(result.valid).all(): raise ValueError('actual frozen head refused bank input')
            y = np.asarray(result.value)
            if y.shape != (len(rows),) or not np.isfinite(y).all(): raise ValueError('finite actual values required')
            values.extend(map(float,y))
        return values
    known = [(o,y) for _,o,y in data.positions if o is not None and y is not None]
    metrics = calibration_metrics(predict([o for o,_ in known]),[y for _,y in known],bins=protocol['bins']) if known else None
    rmse = None if metrics is None else math.sqrt(metrics['mse']); rows = []; unknown = 0
    for group in data.groups:
        if group['status'] != 'OBSERVED' or len(group['branches']) < 2 or any(b['status'] not in ('OBSERVED','COMPLETE') for b in group['branches']):
            unknown += 1; continue
        children = [observation(dict(b['observation']),header,labels) for b in group['branches'] if b['status'] == 'OBSERVED']
        predicted = iter(predict(children)); values = []
        for b in group['branches']:
            if b['status'] == 'OBSERVED': values.append(next(predicted))
            else:
                winner = b['terminal']['winner']; values.append(0. if winner == 2 else 1. if winner == group['selector']['owner'] else -1.)
        order = sorted(range(len(values)),key=lambda i:(-values[i],group['branches'][i]['action']))[:2]
        gap = values[order[0]]-values[order[1]]
        rows.append(dict(selector=dict(group['selector']),gap=gap,actions=[b['action'] for b in group['branches']],values=values,
            gap_exceeds_outcome_rmse=None if rmse is None else gap > rmse))
    return dict(calibration=metrics,outcome_rmse=rmse,heldout_positions=len(data.positions),known_outcomes=len(known),
        unknown_outcomes=len(data.positions)-len(known),sibling_groups=len(data.groups),unknown_sibling_groups=unknown,
        scored_sibling_groups=len(rows),top_two_rows=rows,
        share_gap_exceeds_outcome_rmse=None if not rows or rmse is None else sum(r['gap_exceeds_outcome_rmse'] for r in rows)/len(rows),
        actual_ranking_accuracy='UNMEASURED_NO_MATCHED_TERMINAL_CONTINUATIONS',fitted_models=0,
        interpretation='Actual same-root one-step successor gaps versus frozen collector-policy terminal RMSE; no alternate-child return labels.',
        bank_role='NONTRAINING_EVALUATION',scientific_qualification=False)
