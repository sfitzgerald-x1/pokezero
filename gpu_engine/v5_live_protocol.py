"""Explicit v5 raw actor over the original owning protocol encoder.

No runner/CLI dispatch changes. A factory owns immutable archived weights and
pure JITs; each game owns a fresh original public fold and only its own request.
The original V4 observation remains authoritative for routing. The SAME actual
encode_step result supplies the reviewed six public belief rows exactly once.
"""
from dataclasses import dataclass
import json
import math
from types import MappingProxyType
import time
import weakref

import jax
import jax.numpy as j
import numpy as np

from pokezero import live_protocol_actor as live, v5_contract as c
from pokezero.live_protocol_runtime import CompiledEncoderFunctions
from pokezero.neural_policy import _greedy_action_index, _sample_action
from pokezero.policy import PolicyDecision
from . import v5_model as model, v5_observation as observation


@dataclass(frozen=True,eq=False)
class Factory:
    contract:object
    catalog:object
    params:object
    config:object
    adapter:object
    metadata:str
    encoder:object
    append:object
    forward:object
    learner_seconds:float
    deterministic:bool
    compute_dtype:str

    @property
    def encoder_kind(self):return 'jax-owning'

    def actor(self,runtime):
        if runtime is not self:raise ValueError('explicit original v5 factory runtime required, no legacy tuple')
        return self.create_actor()

    def create_actor(self):
        _checked(self)
        holder={}
        def backend(contract,**game):
            result=_Backend(self,contract,**game);holder['backend']=result;return result
        def policy(contract):
            current=holder['backend']
            select,reset=current.select_action,current.reset_policy
            result=live.RawPolicy(contract,select,reset)
            def validate():
                _checked(self);current._check_namespace()
                if (current.factory is not self or not live._same_callable(result._select_action,select)
                        or not live._same_callable(result.reset,reset)):
                    raise ValueError('original v5 raw callback/owning provider drift')
            result._validate=validate;return result
        return Actor(self,backend_factory=backend,policy_factory=policy)


_FACTORIES=weakref.WeakKeyDictionary()


def _host():
    if isinstance(j.zeros((),j.int32),jax.core.Tracer):raise ValueError('v5 protocol host boundary cannot be traced')


def load_factory(archive,*,catalog,contract,expected_archive_sha256,expected_checkpoint_sha256,
        expected_category_vocabulary_sha256,expected_label_vocabulary_sha256,compute_dtype='float32',
        learner_seconds=60.,deterministic=True,enable_v5=False):
    """Load only the externally digest-bound explicit v5 archive, never legacy."""
    _host()
    if enable_v5 is not True:raise ValueError('explicit v5 protocol enablement required')
    if (type(contract) is not live.ActorContract or type(deterministic) is not bool or compute_dtype not in ('float32','bfloat16')
            or isinstance(learner_seconds,bool) or not isinstance(learner_seconds,(int,float))
            or not math.isfinite(learner_seconds) or not 0<learner_seconds<=60):
        raise ValueError('original owning contract and learner budget<=60s required')
    encoder=CompiledEncoderFunctions(catalog)
    params,config,metadata=model.load(archive,enable_v5=True,
        expected_archive_sha256=expected_archive_sha256,expected_checkpoint_sha256=expected_checkpoint_sha256,
        expected_category_vocabulary_sha256=expected_category_vocabulary_sha256,
        expected_label_vocabulary_sha256=expected_label_vocabulary_sha256)
    if (contract.checkpoint_sha256!=expected_checkpoint_sha256
            or contract.model_config_sha256!=c._sha(config.metadata())
            # The original routing contract hashes literal tokens. V5's model
            # namespace additionally hashes aliases/OOV settings; the original
            # append compiler checks that full hash against THIS same catalog.
            or contract.category_vocab_sha256!=live._vocab_sha(catalog.vocabulary)
            or config.category_vocabulary_sha256!=expected_category_vocabulary_sha256
            or contract.belief_set_source_hash!=catalog.source.source_hash
            or contract.categorical_vocab_size!=config.category_vocab_size
            or catalog.checkpoint_sha256!=expected_checkpoint_sha256
            or metadata['weights_archive_sha256']!=expected_archive_sha256):
        raise ValueError('same explicit v5 archive/config/source/encoder namespace required')
    adapter=observation.compile_adapter(catalog,model_config=config)
    # JAX arrays are immutable. No mutable NumPy tensor can drift underneath a
    # cached inference closure after the original archive digest check.
    params=MappingProxyType({name:j.asarray(value) for name,value in params.items()})
    append=jax.jit(lambda encoded,state,side:observation.append_device(encoded,state,adapter,observer_sides=side))
    forward=jax.jit(lambda inputs:model.forward(params,inputs,config,compute_dtype=compute_dtype))
    result=Factory(contract,catalog,params,config,adapter,json.dumps(metadata,sort_keys=True,allow_nan=False),
        encoder,append,forward,float(learner_seconds),deterministic,compute_dtype)
    _FACTORIES[result]=tuple(getattr(result,name) for name in result.__dataclass_fields__)
    from pokezero.owning_worker_runtime import register_factory
    register_factory(result,lambda:dict(fold=encoder.step,encode=encoder.encode_step,append=append,forward=forward))
    _checked(result);return result


def _checked(factory):
    _host();record=_FACTORIES.get(factory) if type(factory) is Factory else None
    if record is None or any(getattr(factory,name) is not value for name,value in zip(factory.__dataclass_fields__,record)):
        raise ValueError('original immutable v5 protocol factory required')
    factory.encoder.require_catalog(factory.catalog)
    if observation.check_adapter(factory.adapter) is not factory.catalog:
        raise ValueError('same original retained catalog required')
    return factory


def check_factory(factory):return _checked(factory)


class _CaptureFunctions:
    """Per-game read-only capture at the existing compiled-function API seam."""
    def __init__(self,backend,encoder):
        self.encoder=encoder;self.catalog=encoder.catalog;self.step=encoder.step;self.calls=0
        def encode(state,own,side):
            # The original JaxRecordedAdapter calls THIS once, then validates
            # admission/column coverage and returns the identical V4 prefix.
            result=encoder.encode_step(state,own,side)
            self.calls+=1
            backend.captured=(state,result,side)
            return result
        self.encode_step=encode;self._encode=encode

    def require_catalog(self,catalog):
        self.encoder.require_catalog(catalog)
        if catalog is not self.catalog or self.step is not self.encoder.step or self.encode_step is not self._encode:
            raise ValueError('owning encode capture function drift')


class _Backend(live.JaxBackend):
    def __init__(self,factory,contract,**game):
        _checked(factory)
        self.factory=factory;self.captured=None;self.pending=None;self.deadline=None;self.last_inputs=None
        self.pending_capture=None;self.last_capture=None
        functions=_CaptureFunctions(self,factory.encoder)
        super().__init__(contract,catalog=factory.catalog,compiled_functions=functions,**game)

    def ingest(self,lines):
        # The caller supplies one owning stream. Reject the available definite
        # violations before the original fold; do not turn an omniscient split
        # or an exact peer-HP packet into public actor facts.
        peer='p2' if self.owner=='p1' else 'p1'
        for line in lines:
            parts=line.split('|')
            if len(parts)>1 and parts[1]=='split':raise ValueError('isolated owning stream required')
            health=[]
            if len(parts)>4 and parts[1] in ('switch','drag','replace'):health.append((parts[2],parts[4]))
            elif len(parts)>3 and parts[1] in ('-damage','-heal','-sethp'):
                health.append((parts[2],parts[3]))
                if parts[1]=='-sethp' and len(parts)>5:health.append((parts[4],parts[5]))
            if any(ident.startswith(peer) and '/' in hp and hp.split()[0].split('/',1)[1]!='100' for ident,hp in health):
                raise ValueError('exact peer HP cannot enter owning public actor')
        self.pending=None;self.last_inputs=None;self.pending_capture=None;self.last_capture=None
        super().ingest(lines)

    def budget(self):
        if self.deadline is None or time.monotonic()>=self.deadline:
            raise TimeoutError('v5 learner encode/forward/choice exceeded declared budget')

    def reset_policy(self):
        self.pending=None;self.last_inputs=None;self.pending_capture=None;self.last_capture=None

    def encode(self,snapshot):
        self.budget();_checked(self.factory);self.captured=None;self.pending=None
        self.pending_capture=None;self.last_capture=None
        arrays,targets=super().encode(snapshot)
        self.budget()
        if self.captured is None:raise ValueError('original same-call EncodedReport capture missing')
        state,encoded,side=self.captured;self.captured=None
        if state is not self.adapter.state:raise ValueError('owning public state changed during encode')
        appended=jax.block_until_ready(self.factory.append(encoded,state,side));self.budget()
        if not bool(np.asarray(appended.admitted).all()):
            raise ValueError('v5 public observation refused: '+str(np.asarray(appended.error).tolist()))
        columns=observation.actor_inputs(appended,self.factory.adapter)
        inputs={name:columns[name] for name in model.MODEL_INPUTS}
        for name in live.SHAPES:
            value=appended.legal_action_mask if name=='legal_action_mask' else getattr(appended,name)[:,:23]
            if not np.array_equal(np.asarray(value)[0],arrays[name]):
                raise ValueError('original V4 routing prefix differs from v5 actor input')
        self.pending_capture=(snapshot,state,encoded,appended,side)
        self.pending=(snapshot,arrays,inputs);return arrays,targets

    def select_action(self,routing,*,rng):
        self.budget();_checked(self.factory)
        if self.pending is None:raise ValueError('original owning encode must precede v5 raw policy')
        snapshot,arrays,inputs=self.pending
        if (routing.metadata.get('owner_prefix_sha256')!=snapshot.prefix_sha256
                or routing.metadata.get('generation')!=snapshot.generation
                or routing.perspective.player_id!=self.owner
                or any(not np.array_equal(np.asarray(getattr(routing,name)),arrays[name]) for name in live.SHAPES)):
            raise ValueError('same original owning routing observation required')
        self.pending=None
        output=jax.block_until_ready(self.factory.forward(inputs));self.budget()
        logits=np.asarray(output.policy_logits)[0];value=float(np.asarray(output.value)[0])
        legal=np.asarray(routing.legal_action_mask,bool)
        if (np.asarray(output.valid).shape!=(1,) or not bool(np.asarray(output.valid)[0])
                or logits.shape!=(9,) or not np.isfinite(logits).all() or not math.isfinite(value) or not -1<=value<=1):
            raise ValueError('finite valid v5 raw public policy/value required')
        probabilities=np.asarray(jax.nn.softmax(j.where(j.asarray(legal),j.asarray(logits),-j.inf))).tolist()
        legal_actions=tuple(np.flatnonzero(legal).tolist())
        action=(_greedy_action_index(probabilities=probabilities,legal=legal_actions,family_gated=False)
            if self.factory.deterministic else _sample_action(probabilities,legal_actions,rng))
        self.budget();self.last_inputs=inputs
        self.last_capture=self.pending_capture;self.pending_capture=None
        metadata=json.loads(self.factory.metadata)
        return PolicyDecision(action,'v5-raw-'+metadata['weights_archive_sha256'][:12],1. if self.factory.deterministic else probabilities[action],
            dict(policy_family='v5-raw-owning-protocol',actor='RAW_POLICY',view='PUBLIC',
                deterministic=self.factory.deterministic,inference_precision=self.factory.compute_dtype,
                archive_sha256=metadata['weights_archive_sha256'],policy_distribution=tuple(probabilities),
                source_prefix_sha256=snapshot.prefix_sha256),value)


class Actor(live.LiveProtocolActor):
    """Original routing/lifecycle with one wall budget over the whole decision.

    No late result is accepted. A synchronous blocked device call itself must
    be interrupted by the game runner's existing owned-worker watchdog; this
    in-process adapter cannot kill or preempt a library call.
    """
    def __init__(self,factory,**providers):
        self.factory=self._original_factory=factory;self._request_started=None;self._ingest_seconds=0.
        super().__init__(factory.contract,**providers)
        self._original_factories=(self.backend_factory,self.policy_factory);self._providers=None

    def _check_providers(self):
        if (self.factory is not self._original_factory or self.backend_factory is not self._original_factories[0]
                or self.policy_factory is not self._original_factories[1]):
            raise ValueError('original v5 owning factory callbacks required')
        if self._providers is not None:
            backend,policy,select,reset,validate=self._providers
            if (self._backend is not backend or self._policy is not policy or policy._select_action is not select
                    or policy.reset is not reset or policy._validate is not validate):
                raise ValueError('original v5 raw callback/owning provider drift')

    def begin_game(self,**game):
        self._check_providers();self._request_started=None;self._ingest_seconds=0.;self._providers=None
        result=super().begin_game(**game)
        self._providers=(self._backend,self._policy,self._policy._select_action,self._policy.reset,self._policy._validate)
        return result

    def decision_v5_observation(self,snapshot,*,frame):
        """Serialize the SAME successful public encode, without another call.

        The trusted collector supplies its shared pre-step frame ordinal. No
        simulator state, peer request, hidden label or new forward is accepted.
        A new wire packet, failed decision or closed game invalidates access.
        """
        routing=self.decision_observation(snapshot)
        self._check_providers();_checked(self.factory)
        capture=self._backend.last_capture
        if capture is None or capture[0] is not snapshot:
            raise ValueError('same completed original v5 encode capture required')
        _,state,encoded,appended,side=capture
        from .v5_collection_data import public_observation
        observer=int(np.asarray(side)[0])
        species=np.asarray(state.beliefs.species)[0,observer]
        old,public=public_observation(encoded,appended,row=0,owner=int(self.owner[1])-1,
            address=c.DecisionAddress(self.stream_id,frame,int(self.owner[1])-1),
            adapter=self.factory.adapter,public_species=species)
        for name in live.SHAPES:
            if not np.array_equal(np.asarray(getattr(old,name)),np.asarray(getattr(routing,name))):
                raise ValueError('captured v5 prefix differs from original routing input')
        return old,public

    def accept_owner_stream(self,**chunk):
        self._check_providers()
        started=time.monotonic()
        result=super().accept_owner_stream(**chunk)
        requests=[line for line in chunk['lines'] if line.startswith('|request|')]
        if requests:
            self._request_started=(started-self._ingest_seconds
                if live._actionable(live._packet(requests[-1],self.owner)) else None)
        self._ingest_seconds+=time.monotonic()-started
        return result

    def decide(self,snapshot,*,rng):
        # The original FP transport runs decisions in asyncio.to_thread; JAX
        # precision contexts are thread-local. Establish the factory's explicit
        # matched recipe in that worker, restoring its prior context afterwards.
        with jax.default_matmul_precision('highest'),jax.enable_x64(False):
            return self._decide(snapshot,rng=rng)

    def _decide(self,snapshot,*,rng):
        started=time.monotonic() if self._request_started is None else self._request_started
        try:
            _checked(self.factory);self._check_providers();self._current(snapshot)
            self._backend.deadline=started+self.factory.learner_seconds
            decision=super().decide(snapshot,rng=rng)
            self._backend.budget();self._ingest_seconds=0.;self._request_started=None
            return decision
        except live.NoAction:
            raise  # A legitimate waiting/terminal boundary does not poison the actor.
        except BaseException as error:
            self._poison(error);raise
