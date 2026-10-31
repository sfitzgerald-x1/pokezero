"""Actual PUBLIC-only depth1 teacher orchestration, never an acting policy.

Two ORIGINAL posterior draws are shared by the entire legal root/reply grid.
Every cell uses the original hypothetical kernel and owning FULL folds. Unknown
positive mass refuses the whole label. Readiness here means source connection;
actual coverage is recorded and a D batch with no usable labels cannot train.
"""
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as j
import numpy as np

from pokezero import v5_contract as c,v5_gumbel as g,v5_teacher_public_root as planner
from pokezero.live_protocol_actor import ActorContract,_mask_values,_vocab_sha
from pokezero.local_showdown import LocalShowdownConfig
from . import v5_collection as collection,v5_observation,transition
from . import v5_owning_history as capture,v5_public_world as worlds,v5_public_branch as branches
from pokezero.v5_run import MissingDependency

SCHEMA='pokezero.v5.actual_depth1_provider.v1'


def source_binding():
    """Portable source receipt; no model bytes or local paths enter it."""
    root=Path(__file__).resolve().parents[1]
    names=('gpu_engine/v5_depth1_provider.py','gpu_engine/v5_public_world.py',
        'gpu_engine/v5_public_branch.py','gpu_engine/v5_hypothetical_state.py',
        'gpu_engine/v5_hypothetical_profile.py','gpu_engine/v5_owning_history.py',
        'gpu_engine/v5_observation.py','gpu_engine/v5_model.py','gpu_engine/v5_collection.py',
        'gpu_engine/v5_selfplay.py','gpu_engine/v5_collection_data.py','gpu_engine/transition.py',
        'gpu_engine/engine_source_frame_witness.py','src/pokezero/v5_gumbel.py',
        'src/pokezero/v5_teacher_public_root.py','src/pokezero/v5_contract.py',
        'src/pokezero/determinization.py','src/pokezero/engine_world.py')
    values={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in names}
    batch='gpu_engine/v5_hypothetical_batch.py'
    values[batch]=hashlib.sha256((root/batch).read_bytes()).hexdigest() if (root/batch).exists() else 'MISSING'
    return c._sha(dict(schema=SCHEMA,files=values))


@dataclass(frozen=True)
class DecisionContext:
    address:c.DecisionAddress
    source_frame_sha256:str
    public_observation:c.V5Observation
    raw_policy_logits:tuple[float,...]
    raw_public_value:float
    actor_archive_sha256:str
    opponent_archive_sha256:str
    history:capture.History|None
    own_static:worlds.OwnStatic|None
    battle_id:str
    battle_seed:int

    def __post_init__(self):
        if (type(self.address) is not c.DecisionAddress or type(self.public_observation) is not c.V5Observation
                or self.public_observation.address!=self.address or self.public_observation.view!='PUBLIC'):
            raise ValueError('same actual PUBLIC observation/address required')
        for value in (self.source_frame_sha256,self.actor_archive_sha256,self.opponent_archive_sha256):c._binding(value)
        logits=tuple(c._real(value,low=-float('inf'),high=float('inf')) for value in self.raw_policy_logits)
        if len(logits)!=9:raise ValueError('actual same-inference nine logits required')
        object.__setattr__(self,'raw_policy_logits',logits)
        object.__setattr__(self,'raw_public_value',c._real(self.raw_public_value,low=-1,high=1))
        if type(self.battle_id) is not str or not self.battle_id or type(self.battle_seed) is not int or not 0<=self.battle_seed<2**32:
            raise ValueError('literal original battle identity required')


@dataclass(frozen=True)
class Label:
    status:str
    improvement:c.RootImprovement|None
    refusal:str|None
    report:dict


@dataclass(frozen=True)
class GridCell:
    address:c.DecisionAddress
    world:int
    root_index:int
    root_action:int
    reply_action:int

    def __post_init__(self):
        if (type(self.address) is not c.DecisionAddress or type(self.world) is not int or self.world not in (0,1)
                or type(self.root_index) is not int or self.root_index<0
                or any(type(a) is not int or not 0<=a<9 for a in (self.root_action,self.reply_action))):
            raise ValueError('exact address/world/root/action/reply grid cell required')


@dataclass(frozen=True)
class GridBatch:
    roots:tuple
    chance_inputs:tuple
    cells:tuple[GridCell,...]

    def __post_init__(self):
        for name in ('roots','chance_inputs','cells'):object.__setattr__(self,name,tuple(getattr(self,name)))
        if not self.roots or not self.cells or len(self.roots)!=len(self.chance_inputs):
            raise ValueError('nonempty explicit root/world grid and original chance per world required')
        if any(type(root) is not worlds.HypotheticalRoot for root in self.roots):raise ValueError('separate reviewed hypotheses required')
        first=self.roots[0];shared={};worlds_by_root={};seen=set()
        for root,chance in zip(self.roots,self.chance_inputs):
            if root.prepared is not first.prepared or root.history.catalog is not first.history.catalog:
                raise ValueError('one original prepared/catalog grid namespace required')
            address=root.root.address
            if address in shared and root.public is not shared[address]:raise ValueError('same world-independent public root carry required')
            shared[address]=root.public
            if type(chance) is not transition.ChanceInputs:raise ValueError('original chance block required')
            shapes=((1,2,12),(1,2),(1,2),(1,2),(1,2),(1,))
            if any(value.shape!=shape or value.dtype!=j.int32 for value,shape in zip(chance,shapes,strict=True)):
                raise ValueError('original B1 finite integer chance layout required')
        for cell in self.cells:
            if type(cell) is not GridCell or cell.root_index>=len(self.roots):raise ValueError('typed in-bounds grid root index required')
            root=self.roots[cell.root_index]
            if cell.address!=root.root.address or cell.address.owner!=root.history.owner:raise ValueError('same owning root address required')
            if cell.root_index in worlds_by_root and worlds_by_root[cell.root_index]!=cell.world:raise ValueError('one world per root index required')
            worlds_by_root[cell.root_index]=cell.world
            key=(cell.address,cell.world,cell.root_action,cell.reply_action)
            if key in seen:raise ValueError('no duplicated required grid cell')
            seen.add(key)


@dataclass(frozen=True)
class GridChunk:
    cells:tuple[GridCell,...]
    model_indices:tuple[int,...]
    inputs:dict|None
    admitted:object
    refusals:tuple[tuple[str,...],...]
    diagnostics:object=()
    device_rows:int=0
    active_rows:int=0
    padding:int=0

    def __post_init__(self):
        object.__setattr__(self,'cells',tuple(self.cells));object.__setattr__(self,'model_indices',tuple(self.model_indices))
        object.__setattr__(self,'refusals',tuple(tuple(x) for x in self.refusals))
        mask=np.asarray(self.admitted)
        if (not self.cells or any(type(x) is not GridCell for x in self.cells) or mask.shape!=(len(self.cells),)
                or mask.dtype!=np.bool_ or len(self.refusals)!=len(self.cells)
                or self.model_indices!=tuple(map(int,np.flatnonzero(mask)))):
            raise ValueError('whole declared chunk and exact genuine-model row scatter required')
        if any(type(n) is not int or n<0 for n in (self.device_rows,self.active_rows,self.padding)):
            raise ValueError('literal actual execution row counts required')
        if self.active_rows>len(self.cells) or self.device_rows!=self.active_rows+self.padding:
            raise ValueError('exact actual active/device/padding row accounting required')
        if not self.model_indices:
            if self.inputs is not None:raise ValueError('no fabricated model inputs for refused cells')
        elif (type(self.inputs) is not dict or self.inputs.get('view')!='PUBLIC'
                or any(name not in self.inputs or self.inputs[name].shape[0]!=len(self.model_indices) for name in collection.MODEL_INPUTS)):
            raise ValueError('only genuine original public29 model rows required')


def _scalar_execute(batch,adapter,*,max_rows=128):
    """Correctness component only; real plan execution requires iter_execute.

    This exact grid boundary is shared with the fixed batched executor. There
    is no caller-supplied execution callback or readiness flag.
    """
    if type(batch) is not GridBatch:raise ValueError('exact GridBatch required')
    for cell in batch.cells:
        root=batch.roots[cell.root_index];choices=[None,None]
        choices[cell.address.owner]=cell.root_action;choices[1-cell.address.owner]=cell.reply_action
        try:
            step=branches.step(root,tuple(choices),chance_inputs=batch.chance_inputs[cell.root_index],hypothetical_profiles=True)
            rows=int(step.raw_frame is not None)
            if step.refusals:
                yield GridChunk((cell,),(),None,np.array([False]),(step.refusals,),step.diagnostics,rows,rows,0);continue
            appended=v5_observation.append_device(step.encoded[cell.address.owner],step.public[cell.address.owner],
                adapter,observer_sides=j.array([cell.address.owner],j.int32))
            if not bool(np.asarray(appended.admitted).all()):raise ValueError('actual-branch-public29-refused')
            yield GridChunk((cell,),(0,),v5_observation.actor_inputs(appended,adapter),np.array([True]),((),),step.diagnostics,rows,rows,0)
        except Exception as error:
            yield GridChunk((cell,),(),None,np.array([False]),((type(error).__name__+': '+str(error),),))


def _grid_executor():
    try:
        from .v5_hypothetical_batch import iter_execute
        return iter_execute,'BATCHED_ORIGINAL_GRID'
    except ModuleNotFoundError as error:
        if error.name!='gpu_engine.v5_hypothetical_batch':raise
        return _scalar_execute,'SCALAR_CORRECTNESS_COMPONENT_NOT_PRODUCTION_READY'


def _actor(actor,prepared):
    """Original registered immutable actor; no live Session/peer facts needed."""
    record=collection._ACTORS.get(actor) if type(actor) is collection.Actor else None
    if (record is None or record['prepared']() is not prepared
            or any(getattr(actor,name) is not record[name] for name in ('params','config','adapter','metadata'))):
        raise ValueError('same original registered public actor required')
    if v5_observation.check_adapter(actor.adapter) is not collection.driver._checked(prepared)['catalog']:
        raise ValueError('same original public actor catalog required')
    return record,json.loads(actor.metadata)['weights_archive_sha256']


def _forward(record,inputs):
    if set(collection.MODEL_INPUTS)-set(inputs):raise ValueError('all original five public tensors required')
    output=jax.block_until_ready(record['forward']({name:inputs[name] for name in collection.MODEL_INPUTS}))
    logits=np.asarray(output.policy_logits);value=np.asarray(output.value);valid=np.asarray(output.valid)
    if (logits.shape!=(1,9) or value.shape!=(1,) or valid.shape!=(1,) or valid.dtype!=np.bool_
            or not valid[0] or not np.isfinite(logits).all() or not np.isfinite(value).all() or abs(value[0])>1):
        raise ValueError('current public actor output refused/nonfinite')
    return tuple(map(float,logits[0])),float(value[0])


def _leaf_values(record,chunk):
    """CURRENT learner value head on genuine chunk rows; scatter is explicit."""
    if type(chunk) is not GridChunk:raise ValueError('original typed grid chunk required')
    if not chunk.model_indices:return {},{}
    count=len(chunk.model_indices)
    output=jax.block_until_ready(record['forward']({name:chunk.inputs[name] for name in collection.MODEL_INPUTS}))
    logits=np.asarray(output.policy_logits);values=np.asarray(output.value);valid=np.asarray(output.valid)
    if logits.shape!=(count,9) or values.shape!=(count,) or valid.shape!=(count,) or valid.dtype!=np.bool_:
        raise ValueError('current public leaf head chunk shape differs')
    good=valid&np.isfinite(logits).all(-1)&np.isfinite(values)&(np.abs(values)<=1)
    result={};refused={}
    for model_index,cell_index in enumerate(chunk.model_indices):
        if good[model_index]:result[cell_index]=float(values[model_index])
        else:refused[cell_index]='current-public-leaf-head-refused-or-nonfinite'
    return result,refused


def _same_root(view,observation):
    for name in ('categorical_ids','numeric_features','token_type_ids','attention_mask'):
        expected=np.asarray(getattr(observation,name),dtype=np.float32 if name=='numeric_features' else None)
        if not np.array_equal(np.asarray(view.inputs[name])[0,0],expected):raise ValueError('same actual public root input:'+name)
    if not np.array_equal(np.asarray(view.legal_action_mask)[0],observation.legal_action_mask):
        raise ValueError('same actual full legal root denominator required')


@dataclass(frozen=True)
class Provider:
    prepared:object
    public_config:LocalShowdownConfig
    contract:ActorContract
    recipe:g.Recipe
    producer_sha256:str=''

    def __post_init__(self):
        if type(self.recipe) is not g.Recipe or type(self.public_config) is not LocalShowdownConfig or type(self.contract) is not ActorContract:
            raise ValueError('explicit original public parser and bounded teacher recipe required')
        r=collection.driver._checked(self.prepared);catalog=r['catalog']
        if (self.public_config.set_belief_source is not True or self.public_config.category_vocab is not catalog.vocabulary
                or _vocab_sha(catalog.vocabulary)!=self.contract.category_vocab_sha256
                or _mask_values(self.public_config.feature_masks)!=self.contract.feature_masks
                or self.contract.belief_set_source_hash!=catalog.source.source_hash):
            raise ValueError('frozen original v4 public source/category/masks required')
        if self.producer_sha256:raise ValueError('producer binding is computed from actual source, never supplied')
        object.__setattr__(self,'producer_sha256',source_binding())

    def require_ready(self):
        r=collection.driver._checked(self.prepared);collection.driver._checked_witness(r)
        if source_binding()!=self.producer_sha256 or r['speed_profile'] is None:
            raise ValueError('original connected hypothetical source/profile changed or missing')
        _,mode=_grid_executor()
        return dict(schema=SCHEMA,producer_sha256=self.producer_sha256,seed=self.recipe.seed,
            recipe=g.DECISIONS,execution_mode=mode,max_grid_rows=128,
            scope='SOURCE_CONNECTED_RUNTIME_COVERAGE_REQUIRED_NOT_QUALIFICATION')

    def require_batched(self):
        metadata=self.require_ready()
        if metadata['execution_mode']!='BATCHED_ORIGINAL_GRID':
            raise MissingDependency('D has only scalar correctness execution; original batched grid is not connected')
        return metadata

    def label(self,context,*,learner,opponent):
        if type(context) is not DecisionContext:raise ValueError('typed owning PUBLIC decision context required')
        self.require_ready();address=context.address;legal=tuple(context.public_observation.legal_action_mask)
        report=dict(schema=SCHEMA,address=dict(stream=address.stream,frame=address.frame,owner=address.owner),
            source_frame_sha256=context.source_frame_sha256,producer_sha256=self.producer_sha256,
            actor_archive_sha256=context.actor_archive_sha256,opponent_archive_sha256=context.opponent_archive_sha256,
            selected=self.recipe.selected(address),root_actions=sum(legal),world_slots=[],
            required_pairs=None,known_required_pairs=0,unknown_pairs=0,unknown_worlds=0,
            completed_pairs=0,branch_refusals={},required_pairs_known=False)
        if not report['selected']:return Label('UNSELECTED',None,None,report)
        reasons=Counter();prices=[];grid_roots=[];grid_chance=[];grid_cells=[];grids={};policies={}
        report['required_worlds']=2
        report['world_slots']=[dict(index=index,posterior_seed=self.recipe.world_seed(address,index),
            status='REFUSED',required_pairs=None,completed_pairs=0,unknown_pairs=0,reason='not-materialized') for index in (0,1)]
        try:
            order=self.recipe.pricing_order(address,context.raw_policy_logits,legal)
            report['pricing_order']=order
            left,archive=_actor(learner,self.prepared);right,peer_archive=_actor(opponent,self.prepared)
            if (archive,peer_archive)!=(context.actor_archive_sha256,context.opponent_archive_sha256):
                raise ValueError('actual current learner/opponent archive binding differs')
            if learner.config.arm!='D' or opponent.config!=learner.config:raise ValueError('same original D architecture required')
            if context.history is None or context.own_static is None:raise ValueError('missing-owning-history-or-static')
            if ((context.history.stream_id,context.history.frames[-1].number,context.history.owner)!=
                    (address.stream,address.frame,address.owner)):
                raise ValueError('same current owning history address required')
            root=capture.public_root(context.history,config=self.public_config,contract=self.contract,
                battle_id=context.battle_id,seed=context.battle_seed)
            if root.root is None:raise ValueError('owning-history-decoder:'+str(root.decoded.refusal_counts))
            public_root=root.root;source=public_root.context.public_materialization_state.belief_engine.set_source
            plans=planner.plan_starts(public_root,self.recipe,set_source=source) # exactly2; no retry/refill
            r=collection.driver._checked(self.prepared);shared=worlds.public_carry(context.history,r)
            for index,(start,reason) in enumerate(zip(plans.starts,plans.refusals,strict=True)):
                slot=report['world_slots'][index];slot['reason']=reason
                if start is None:
                    report['unknown_worlds']+=1;reasons[reason or 'public-sampler-unavailable']+=1;continue
                seed=int.from_bytes(self.recipe._digest(address,'hypothetical-kernel',index)[:4],'big')
                result=worlds.materialize(public_root,start,prepared=self.prepared,history=context.history,
                    own_static=context.own_static,seed=seed,retained=shared)
                if result.branch is None:
                    slot['reason']='materialize:'+','.join(result.refusals);report['unknown_worlds']+=1;reasons[slot['reason']]+=1;continue
                branch=result.branch
                slot.update(hypothetical_origin_sha256=branch.origin_sha256,kernel_seed=seed)
                try:
                    _same_root(branches.policy_observation(branch,learner.adapter,owner=address.owner),context.public_observation)
                    reply=branches.policy_observation(branch,opponent.adapter,owner=1-address.owner)
                    reply_logits,_=_forward(right,reply.inputs)
                    reply_legal=tuple(map(bool,np.asarray(reply.legal_action_mask)[0]))
                    if not any(reply_legal):raise ValueError('unsupported-wait-or-replacement-reply-phase')
                except (ValueError,KeyError,IndexError) as error:
                    slot['reason']='root-policy:'+str(error);report['unknown_worlds']+=1;reasons[slot['reason']]+=1;continue
                replies=tuple(i for i,active in enumerate(reply_legal) if active)
                slot.update(required_pairs=len(order)*len(replies),reply_legal=reply_legal,reply_logits=reply_logits,reason=None)
                report['known_required_pairs']+=slot['required_pairs']
                chance=transition.sample_chance_inputs(branch.state) # same original chance block for this world's grid
                root_index=len(grid_roots);grid_roots.append(branch);grid_chance.append(chance)
                grids[index]=[[None]*9 for _ in range(9)];policies[index]=(reply_legal,reply_logits)
                for action in order:
                    for opponent_action in replies: # includes numerically underflowed positive probabilities
                        grid_cells.append(GridCell(address,index,root_index,action,opponent_action))
            executor,mode=_grid_executor();report['execution']=dict(mode=mode,chunks=[],declared_cells=len(grid_cells))
            if grid_cells:
                batch=GridBatch(tuple(grid_roots),tuple(grid_chance),tuple(grid_cells));seen=[]
                for chunk in executor(batch,learner.adapter,max_rows=128):
                    if type(chunk) is not GridChunk:raise ValueError('fixed executor must return original GridChunk')
                    seen.extend(chunk.cells)
                    report['execution']['chunks'].append(dict(device_rows=chunk.device_rows,active_rows=chunk.active_rows,
                        padding=chunk.padding,declared_cells=len(chunk.cells),model_rows=len(chunk.model_indices)))
                    try:values,head_errors=_leaf_values(left,chunk)
                    except Exception as error:values={};head_errors={i:type(error).__name__+': '+str(error) for i in chunk.model_indices}
                    for index,cell in enumerate(chunk.cells):
                        slot=report['world_slots'][cell.world]
                        if index in values:
                            grids[cell.world][cell.root_action][cell.reply_action]=values[index]
                            slot['completed_pairs']+=1;report['completed_pairs']+=1
                        else:
                            reason=head_errors.get(index) or ','.join(chunk.refusals[index]) or 'unknown-required-public-leaf'
                            slot['unknown_pairs']+=1;report['unknown_pairs']+=1;reasons[reason]+=1
                if tuple(seen)!=batch.cells:raise ValueError('executor omitted/duplicated/reordered required grid cells')
            for index,(reply_legal,reply_logits) in sorted(policies.items()):
                slot=report['world_slots'][index];slot['joint_values']=grids[index]
                slot['status']='COMPLETE' if not slot['unknown_pairs'] else 'REFUSED'
                prices.append(g.WorldPrices(True,reply_legal,reply_logits,grids[index]))
            report['branch_refusals']=dict(sorted(reasons.items()))
            if len(prices)!=2:
                return Label('REFUSED',None,'unknown-required-world-or-reply-denominator',report)
            reduced=g.reduce_completed_q(root_logits=context.raw_policy_logits,root_legal=legal,worlds=prices)
            report.update(required_pairs=reduced.required_pairs,required_pairs_known=True,
                unknown_pairs=reduced.unknown_pairs,unknown_reply_mass=reduced.unknown_reply_mass)
            if reduced.improvement is not None:
                completed=tuple(logit+(q or 0.) for logit,q in zip(context.raw_policy_logits,reduced.improvement.completed_q))
                report['gumbel_argmax_diagnostic']=self.recipe.pricing_order(address,completed,legal)[0]
            return Label(reduced.status,reduced.improvement,reduced.reason,report)
        except Exception as error:
            reason=type(error).__name__+': '+str(error)
            report['unknown_worlds']=2-len(prices);report['branch_refusals']={reason:1}
            return Label('REFUSED',None,reason,report)
