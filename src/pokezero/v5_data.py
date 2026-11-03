"""Explicit v5 producer rows, TrainingBatch sidecars and array caches.

Labels enter only as typed same-frame source data. Legacy slot-index actions,
adjacent states, PPO returns and advantages never manufacture auxiliary truth.
Default trajectory serializers and v2 training caches remain unchanged. Actor
inputs contain only the public29-token observation; optional peer views are
separate training-only sidecars. Original collection owns capture provenance.
"""
from dataclasses import asdict, dataclass, fields, replace
import hashlib
import json
from pathlib import Path
from types import MappingProxyType

import numpy as np

from . import v5_contract as c
from .dataset import TrainingBatch
from .trajectory import TrajectoryStep, _step_from_dict, _step_to_dict

CACHE_SCHEMA = 'pokezero.training_cache.v5'
ROW_SCHEMA = 'pokezero.trajectory_decision.v5'
INPUTS = ('categorical_ids', 'numeric_features', 'token_type_ids', 'attention_mask')
SEARCH = ('search_policy_targets','search_policy_target_mask','search_value_targets','search_value_target_mask')
STATIC = ('schema','observation_schema','category_vocabulary_sha256','label_vocabulary_sha256',
    'source_binding_sha256','label_vocabulary')
META_ROWS = ('addresses','source_frame_sha256','terminal_status','search_status','search_refusal',
    'root_improvement_records','terminal_source_sha256')
ARRAY_SHAPES = dict(opponent_semantic_action_targets=(),opponent_semantic_action_target_mask=(),
    hidden_set_targets=(6,64),hidden_set_target_mask=(6,),hidden_set_species=(6,),
    survival_targets=(12,),final_hp_targets=(12,),pokemon_end_target_mask=(12,),
    outcome_targets=(),outcome_target_mask=(),privileged_view_mask=(),
    privileged_categorical_ids=(1,29,41),privileged_numeric_features=(1,29,132),
    privileged_token_type_ids=(1,29),privileged_attention_mask=(1,29))


def _sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def _file_sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def semantic_opponent_action(address, peer_address, *, kind, identity):
    """Literal same-frame peer choice; never convert an actor's nine-slot index."""
    if (type(address) is not c.DecisionAddress or type(peer_address) is not c.DecisionAddress
            or peer_address != replace(address,owner=1-address.owner)):
        raise ValueError('same stream/frame and opposite original owner required')
    return c.OpponentAction(kind,identity)


@dataclass(frozen=True)
class DecisionRow:
    battle_id: str
    seed: int
    format_id: str
    step: TrajectoryStep
    observation: c.V5Observation
    labels: c.V5Labels
    source_frame_sha256: str
    privileged: c.V5Observation | None = None
    terminal_source_sha256: str | None = None

    def __post_init__(self):
        if (type(self.battle_id) is not str or not self.battle_id or type(self.seed) is not int
                or type(self.format_id) is not str or not self.format_id
                or type(self.step) is not TrajectoryStep or type(self.observation) is not c.V5Observation
                or self.observation.view != 'PUBLIC' or type(self.labels) is not c.V5Labels):
            raise ValueError('explicit original trajectory/v5 public decision required')
        c._binding(self.source_frame_sha256)
        if self.terminal_source_sha256 is not None:c._binding(self.terminal_source_sha256)
        if self.step.observation.schema_version != c.OBSERVATION_SCHEMA_VERSION_V4:
            raise ValueError('original v4 step schema required; no legacy cache relabeling')
        self.step.observation.validate(c.V4_SPEC)
        if self.step.player_id != ('p1','p2')[self.observation.address.owner]:
            raise ValueError('original trajectory owner differs from v5 address')
        for name in (*INPUTS,'legal_action_mask'):
            actual=getattr(self.observation,name)
            if name in INPUTS:actual=actual[:23]
            if tuple(actual) != tuple(getattr(self.step.observation,name)):
                raise ValueError('v5 must preserve the actual original v4 observation prefix')
        self.labels.columns(self.observation)
        if self.labels.terminal_status=='COMPLETE' and self.terminal_source_sha256 is None:
            raise ValueError('actual terminal capture binding required for complete labels')
        if self.privileged is not None and (type(self.privileged) is not c.V5Observation
                or self.privileged.view != 'PRIVILEGED' or self.privileged.address != self.observation.address
                or self.privileged.category_vocabulary_sha256 != self.observation.category_vocabulary_sha256
                or self.privileged.label_vocabulary.sha256 != self.observation.label_vocabulary.sha256):
            raise ValueError('separate same-address training-only privileged view required')


def with_terminal(row, *, address, status, outcome=None, pokemon_end=(None,)*12, source_sha256):
    """Attach literal final source data, never infer it from return/GAE/HP deltas."""
    if (type(row) is not DecisionRow or type(address) is not c.DecisionAddress
            or address.stream != row.observation.address.stream or address.owner != row.observation.address.owner
            or address.frame < row.observation.address.frame):
        raise ValueError('same actual stream/owner terminal boundary required')
    c._binding(source_sha256)
    return replace(row,labels=replace(row.labels,terminal_status=status,outcome=outcome,
        pokemon_end=tuple(pokemon_end)),terminal_source_sha256=source_sha256)


def row_to_dict(row):
    if type(row) is not DecisionRow:raise ValueError('typed v5 decision required')
    def observation(value):
        if value is None:return None
        result={name:getattr(value,name) for name in (*INPUTS,'legal_action_mask','view','schema_version')}
        result['address']=asdict(value.address)
        result['category_vocabulary_sha256']=value.category_vocabulary_sha256
        result['posteriors']=[{name:getattr(p,name) for name in
            ('species','effective_mask','prior_weights','reason','pin_conflict')} for p in value.posteriors]
        return result
    return dict(schema=ROW_SCHEMA,battle_id=row.battle_id,seed=row.seed,format_id=row.format_id,
        step=_step_to_dict(row.step),observation=observation(row.observation),labels=asdict(row.labels),
        label_vocabulary=asdict(row.observation.label_vocabulary),source_frame_sha256=row.source_frame_sha256,
        privileged=observation(row.privileged),terminal_source_sha256=row.terminal_source_sha256)


def row_from_dict(payload):
    if payload.get('schema') != ROW_SCHEMA:raise ValueError('explicit v5 trajectory schema required')
    vocabulary=c.LabelVocabulary(**payload['label_vocabulary'])
    def observation(value):
        if value is None:return None
        data=dict(value);data['address']=c.DecisionAddress(**data['address'])
        data['posteriors']=tuple(c.PublicPosterior(vocabulary,**p) for p in data['posteriors'])
        return c.V5Observation(**data)
    target=dict(payload['labels']);target['address']=c.DecisionAddress(**target['address'])
    target['opponent_next_action']=None if target['opponent_next_action'] is None else c.OpponentAction(**target['opponent_next_action'])
    target['hidden_sets']=tuple(None if v is None else c.HiddenSet(**v) for v in target['hidden_sets'])
    target['pokemon_end']=tuple(None if v is None else c.PokemonEnd(**v) for v in target['pokemon_end'])
    target['root_improvement']=None if target['root_improvement'] is None else c.RootImprovement(**target['root_improvement'])
    return DecisionRow(payload['battle_id'],payload['seed'],payload['format_id'],_step_from_dict(payload['step']),
        observation(payload['observation']),c.V5Labels(**target),payload['source_frame_sha256'],
        observation(payload.get('privileged')),payload.get('terminal_source_sha256'))


def attach_batch(base, rows):
    """Existing behavior/PPO batch -> explicit original v5 actor/auxiliary data."""
    rows=tuple(rows)
    if (type(base) is not TrainingBatch or base.v5_columns is not None or base.window_size != 1
            or base.window_row_indices is not None or len(rows) != base.batch_size
            or any(type(row) is not DecisionRow for row in rows)):
        raise ValueError('expanded original one-frame behavior batch and aligned v5 rows required')
    for i,row in enumerate(rows):
        if ((base.battle_ids[i],base.seeds[i],base.format_ids[i],base.player_ids[i],base.turn_indices[i]) !=
                (row.battle_id,row.seed,row.format_id,row.step.player_id,row.step.turn_index)
                or int(base.action_indices[i]) != row.step.action_index
                or tuple(base.legal_action_mask[i]) != row.step.legal_action_mask
                or bool(base.terminal_capped[i]) != (row.labels.terminal_status=='CAPPED')):
            raise ValueError('v5 row differs from its actual original behavior/terminal record')
        for name in INPUTS:
            dtype=np.float32 if name=='numeric_features' else None
            if not np.array_equal(np.asarray(getattr(base,name)[i][0],dtype=dtype),
                    np.asarray(getattr(row.step.observation,name),dtype=dtype)):
                raise ValueError('original behavior observation and v5 prefix differ')
    observations=tuple(row.observation for row in rows);labels=tuple(row.labels for row in rows)
    packed=c.batch_columns(observations,labels)
    for target,mask in (('search_policy_targets','search_policy_target_mask'),
            ('search_value_targets','search_value_target_mask')):
        old_mask=getattr(base,mask)
        if old_mask is not None:
            active=np.asarray(old_mask)
            new_mask=np.asarray(packed[mask])
            if (np.any(active & ~new_mask) or not np.allclose(np.asarray(getattr(base,target))[active],
                    np.asarray(packed[target])[active],atol=1e-6,rtol=0)):
                raise ValueError('existing search columns conflict with original D labels')
    sidecar={name:value for name,value in packed.items()
        if name not in (*INPUTS,'history_mask','legal_action_mask','batch_size',*SEARCH)}
    sidecar.update(source_binding_sha256=observations[0].label_vocabulary.source_binding_sha256,
        label_vocabulary=asdict(observations[0].label_vocabulary),
        source_frame_sha256=tuple(row.source_frame_sha256 for row in rows),
        terminal_source_sha256=tuple(row.terminal_source_sha256 for row in rows),
        terminal_status=tuple(label.terminal_status for label in labels),
        outcome_targets=tuple(0.0 if label.outcome is None else label.outcome for label in labels),
        outcome_target_mask=tuple(label.outcome is not None and label.terminal_status=='COMPLETE' for label in labels),
        root_improvement_records=tuple(None if label.root_improvement is None else asdict(label.root_improvement) for label in labels),
        privileged_view_mask=tuple(row.privileged is not None for row in rows))
    for name,width in (('categorical_ids',41),('numeric_features',132),('token_type_ids',None),('attention_mask',None)):
        empty=tuple(tuple(0 for _ in range(width)) for _ in range(29)) if width else (False,)*29 if name=='attention_mask' else (0,)*29
        sidecar['privileged_'+name]=tuple((empty if row.privileged is None else getattr(row.privileged,name),) for row in rows)
    return replace(base,**{name:packed[name] for name in (*INPUTS,'history_mask','legal_action_mask',*SEARCH)},
        opponent_action_indices=(-1,)*len(rows),opponent_action_mask=(False,)*len(rows),
        v5_columns=MappingProxyType(sidecar))


def validate_batch(batch):
    """Validate active targets; masked poison is never converted into truth."""
    cols=batch.v5_columns;n=batch.batch_size
    if not hasattr(cols,'keys') or set(cols) != set((*STATIC,*META_ROWS,*ARRAY_SHAPES)):
        raise ValueError('exact explicit v5 sidecar columns required')
    if cols['schema']!=c.TRAINING_CONTRACT_SCHEMA or cols['observation_schema']!=c.OBSERVATION_SCHEMA_VERSION_V5:
        raise ValueError('v5 batch schema differs')
    vocabulary=c.LabelVocabulary(**dict(cols['label_vocabulary']))
    for name in ('category_vocabulary_sha256','label_vocabulary_sha256','source_binding_sha256'):c._binding(cols[name])
    if vocabulary.sha256!=cols['label_vocabulary_sha256'] or vocabulary.source_binding_sha256!=cols['source_binding_sha256']:
        raise ValueError('original v5 label source namespace differs')
    if batch.window_row_indices is not None or batch.window_size!=1:
        raise ValueError('v5 contract currently requires one expanded frame')
    for name in META_ROWS:
        if len(cols[name])!=n:raise ValueError('v5 metadata row count differs')
    for name in ('source_frame_sha256','terminal_source_sha256'):
        for value in cols[name]:
            if name=='source_frame_sha256' or value is not None:c._binding(value)
    for i,address in enumerate(cols['addresses']):
        original=c.DecisionAddress(**address)
        if batch.player_ids[i] != ('p1','p2')[original.owner]:
            raise ValueError('v5 address owner differs from original behavior owner')
    for name,shape in ARRAY_SHAPES.items():
        a=np.asarray(cols[name])
        if a.shape!=(n,*shape):raise ValueError('v5 target shape differs: '+name)
        if name.endswith('mask') and name!='privileged_attention_mask':
            if a.dtype!=np.bool_:raise ValueError('typed v5 target mask required')
    if np.any(np.asarray(batch.opponent_action_mask)) or np.any(np.asarray(batch.opponent_action_indices)!=-1):
        raise ValueError('legacy actor-slot opponent targets cannot enter v5')
    for name,shape in (('categorical_ids',(n,1,29,41)),('numeric_features',(n,1,29,132)),
            ('token_type_ids',(n,1,29)),('attention_mask',(n,1,29))):
        if np.asarray(getattr(batch,name)).shape!=shape:raise ValueError('public v5 actor input shape differs')
    categories=np.asarray(batch.categorical_ids);numeric=np.asarray(batch.numeric_features)
    if categories.dtype.kind not in 'iu' or np.any(categories<0) or not np.isfinite(numeric).all():
        raise ValueError('public actor category/numeric values invalid')
    expected_roles=(0,)+(1,)*6+(2,)*6+(3,)*9+(5,)+(7,)*6
    if not np.all(np.asarray(batch.token_type_ids)==np.asarray(expected_roles)[None,None]):
        raise ValueError('v4 prefix/appended belief roles differ')
    private=np.asarray(cols['privileged_view_mask'])
    if private.any():
        # Validate only present training views, without coercing objects/bools
        # into real numeric cells. Masked private payload never enters actor
        # inputs and supplies no value/category/type/mask constraints.
        pc=np.asarray(cols['privileged_categorical_ids'])[private]
        pn=np.asarray(cols['privileged_numeric_features'])[private]
        pt=np.asarray(cols['privileged_token_type_ids'])[private]
        pa=np.asarray(cols['privileged_attention_mask'])[private]
        if (pc.dtype.kind not in 'iu' or np.any(pc<0) or pn.dtype.kind not in 'iuf'
                or not np.isfinite(pn).all() or pt.dtype.kind not in 'iu'
                or not np.all(pt==np.asarray(expected_roles)[None,None])
                or pa.dtype!=np.bool_ or not np.all(pa[:,:,23:])):
            raise ValueError('active privileged category/numeric/type/mask values invalid')
    action=np.asarray(cols['opponent_semantic_action_targets']);mask=np.asarray(cols['opponent_semantic_action_target_mask'])
    if action.dtype.kind not in 'iu' or np.any((action[mask]<0)|(action[mask]>=len(vocabulary.opponent_classes))):
        raise ValueError('active semantic opponent target outside source namespace')
    hidden=np.asarray(cols['hidden_set_targets']);species=np.asarray(cols['hidden_set_species']);hm=np.asarray(cols['hidden_set_target_mask'])
    if species.dtype.kind not in 'iu':raise ValueError('typed hidden-set species labels required')
    for i,j in zip(*np.nonzero(hm)):
        sid=int(species[i,j]);p=hidden[i,j]
        if not 0<=sid<len(vocabulary.species):raise ValueError('hidden species outside source namespace')
        count=vocabulary.variant_count(vocabulary.species[sid])
        if not np.isfinite(p).all() or np.any(p<0) or not np.isclose(p.sum(),1,atol=1e-6) or np.any(p[count:]!=0):
            raise ValueError('active hidden-set distribution differs from complete source axis')
    outcome=np.asarray(cols['outcome_targets']);om=np.asarray(cols['outcome_target_mask'])
    survival=np.asarray(cols['survival_targets']);hp=np.asarray(cols['final_hp_targets']);pm=np.asarray(cols['pokemon_end_target_mask'])
    if not np.isin(outcome[om],[-1,0,1]).all():raise ValueError('literal masked game outcome required')
    for i,status in enumerate(cols['terminal_status']):
        if bool(batch.terminal_capped[i]) != (status=='CAPPED'):
            raise ValueError('v5 terminal status differs from original capped disposition')
        if status not in ('COMPLETE','IN_PROGRESS','CAPPED','REFUSED') or status!='COMPLETE' and (om[i] or pm[i].any()):
            raise ValueError('unknown terminal cannot supervise outcome or Pokemon end state')
        if status=='COMPLETE' and cols['terminal_source_sha256'][i] is None:
            raise ValueError('actual terminal capture binding missing')
    if (not np.isfinite(hp[pm]).all() or np.any((hp[pm]<0)|(hp[pm]>1))
            or not np.isin(survival[pm],[0,1]).all() or np.any((hp[pm]>0)!=survival[pm].astype(bool))):
        raise ValueError('active literal survival/final-HP labels differ')
    for i,status in enumerate(cols['search_status']):
        policy=bool(np.asarray(batch.search_policy_target_mask)[i]);value=bool(np.asarray(batch.search_value_target_mask)[i])
        record=cols['root_improvement_records'][i]
        if (status not in ('LABELED','REFUSED','UNSELECTED') or policy!=(status=='LABELED')
                or value and (not policy or not om[i]) or (record is not None)!=(status=='LABELED')):
            raise ValueError('independent D policy/value masks differ from captured disposition')
        if record is not None:
            teacher=c.RootImprovement(**record);teacher.validate(tuple(batch.legal_action_mask[i]))
            if not np.allclose(teacher.policy_target,np.asarray(batch.search_policy_targets)[i],atol=1e-6,rtol=0):
                raise ValueError('captured D policy differs')
            if value and not np.isclose(np.asarray(batch.search_value_targets)[i],(outcome[i]+teacher.root_value)/2,atol=1e-6):
                raise ValueError('D outcome/root target was remixed or inferred from PPO')


def slice_columns(columns,start,stop):
    return MappingProxyType({name:value if name in STATIC else value[start:stop] for name,value in columns.items()})


def combine_columns(parts):
    if any(part is None for part in parts):raise ValueError('cannot mix legacy and v5 training batches')
    if any(any(_sha(part[name])!=_sha(parts[0][name]) for name in STATIC) for part in parts[1:]):
        raise ValueError('cannot combine different v5 source/category/label namespaces')
    return MappingProxyType({name:parts[0][name] if name in STATIC else
        np.concatenate([np.asarray(part[name]) for part in parts],axis=0) if name in ARRAY_SHAPES else
        tuple(value for part in parts for value in part[name]) for name in parts[0]})


def write_cache(path,batch):
    """One explicit v5 shard; no legacy cache is relabeled or overwritten."""
    if type(batch) is not TrainingBatch or batch.v5_columns is None:raise ValueError('explicit v5 TrainingBatch required')
    validate_batch(batch);path=Path(path);path.mkdir(exist_ok=False)
    metadata=dict(schema_version=CACHE_SCHEMA,example_count=batch.batch_size,
        v5_static={name:batch.v5_columns[name] for name in STATIC},
        v5_rows={name:batch.v5_columns[name] for name in META_ROWS},base_json={},base_none=[],arrays={})
    def save(name,value):
        array=np.asarray(value)
        if array.dtype.hasobject:raise ValueError('object arrays cannot enter v5 cache')
        file=path/(name+'.npy');np.save(file,array,allow_pickle=False)
        metadata['arrays'][name]=dict(shape=list(array.shape),dtype=array.dtype.str,
            sha256=_file_sha(file))
    for field in fields(batch):
        name=field.name;value=getattr(batch,name)
        if name=='v5_columns':continue
        if value is None:metadata['base_none'].append(name)
        elif name=='step_metadata':metadata['base_json'][name]=list(value)
        else:save('base__'+name,value)
    for name in ARRAY_SHAPES:save('v5__'+name,batch.v5_columns[name])
    metadata['metadata_sha256']=_sha(metadata)
    (path/'metadata.json').write_text(json.dumps(metadata,sort_keys=True,allow_nan=False)+'\n')
    return metadata


def _file_identity(path):
    value=Path(path).stat()
    return (value.st_dev,value.st_ino,value.st_size,value.st_mtime_ns,value.st_ctime_ns)


def _require_cache_identities(identities):
    try:unchanged=all(_file_identity(path)==identity for path,identity in identities.items())
    except OSError:unchanged=False
    if not unchanged:raise ValueError('verified v5 cache files changed during reader session')


class VerifiedCache:
    """One immutable, digest-verified shard, reused only within its owner process.

    Every new opening retains the original metadata/array SHA checks. Reuse
    checks file identity, size and both modification/change timestamps; any
    change fails closed. This object is never serialized into a checkpoint.
    """
    def __init__(self,path,*,resident_bytes=None):
        if resident_bytes is not None and (type(resident_bytes) is not int or resident_bytes<=0):
            raise ValueError('positive finite resident cache byte budget required')
        copied_bytes=0
        path=Path(path);file=path/'metadata.json'
        identities={file:_file_identity(file)}
        metadata=json.loads(file.read_text())
        check=dict(metadata);claimed=check.pop('metadata_sha256',None)
        if metadata.get('schema_version')!=CACHE_SCHEMA or claimed!=_sha(check):raise ValueError('v5 cache schema/metadata changed')
        arrays={}
        for name,description in metadata['arrays'].items():
            file=path/(name+'.npy');identities[file]=_file_identity(file)
            if _file_sha(file)!=description['sha256']:raise ValueError('v5 cache array bytes changed')
            value=np.load(file,mmap_mode='r',allow_pickle=False)
            if list(value.shape)!=description['shape'] or value.dtype.str!=description['dtype']:raise ValueError('v5 cache shape/dtype changed')
            if resident_bytes is not None:
                copied_bytes+=value.nbytes
                if copied_bytes>resident_bytes:
                    raise ValueError('resident cache exceeds explicit byte budget')
                value=np.array(value,copy=True,order='K')
                value.flags.writeable=False
            arrays[name]=value
        _require_cache_identities(identities)
        base={field.name:None for field in fields(TrainingBatch) if field.name!='v5_columns'}
        base.update(metadata['base_json']);base.update({name[6:]:value for name,value in arrays.items() if name.startswith('base__')})
        cols={**metadata['v5_static'],**metadata['v5_rows'],**{name[4:]:value for name,value in arrays.items() if name.startswith('v5__')}}
        self._batch=TrainingBatch(**base,v5_columns=MappingProxyType(cols))
        if self._batch.batch_size!=metadata['example_count']:raise ValueError('v5 cache row denominator changed')
        self._identities=identities

    def batches(self,*,batch_size,start_batch=0):
        if type(batch_size) is not int or batch_size<=0:raise ValueError('positive v5 batch size required')
        count=(self._batch.batch_size+batch_size-1)//batch_size
        if type(start_batch) is not int or not 0<=start_batch<=count:raise ValueError('v5 cache batch cursor outside finite shard')
        _require_cache_identities(self._identities)
        from .dataset import slice_training_batch
        for start in range(start_batch*batch_size,self._batch.batch_size,batch_size):
            yield slice_training_batch(self._batch,start,min(start+batch_size,self._batch.batch_size))


def iter_cache_batches(path,*,batch_size,start_batch=0):
    if type(batch_size) is not int or batch_size<=0:raise ValueError('positive v5 batch size required')
    yield from VerifiedCache(path).batches(batch_size=batch_size,start_batch=start_batch)
