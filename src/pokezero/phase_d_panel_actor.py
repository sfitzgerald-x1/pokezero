"""Explicit search actor factory over the original B2 owning panel transport.

CPU workers retain the original B2 encoding and original LiveProtocolActor
choice routing. Their policy sends one own snapshot to ONE shared GPU server.
The caller must label this panel route as search in its fixed cell declaration;
this module creates no games, alters no scorer and never changes FP1000ms.
"""
from .live_protocol_actor import (ActorContract, LiveProtocolActor, JaxBackend,
    RawPolicy, _same_callable)
from .live_protocol_bridge import OwningActorFactory, validate_encoder_config
from .policy import PolicyDecision
from .phase_d_search_server import Client


def factory(config, *, catalog, client):
    """The unchanged bridge factory type supports the explicit GPU client policy.

    No CPU forward is run by this factory. A future winner needs an explicit
    winner contract/evaluator adapter; this prerequisite is bound to9375 only.
    """
    validate_encoder_config(config)
    contract=ActorContract.champion()
    if config.encoder_kind!='jax-owning' or type(client) is not Client or catalog.checkpoint_sha256!=contract.checkpoint_sha256:
        raise ValueError('original B2 champion catalog and shared GPU client required')
    from .live_protocol_runtime import CompiledEncoderFunctions
    functions=CompiledEncoderFunctions(catalog)
    pending={}

    class Backend(JaxBackend):
        def encode(self,snapshot):
            result=super().encode(snapshot)
            pending['snapshot']=snapshot
            return result

    def backend(c,*,env_config,**identity):
        if c!=contract:raise ValueError('same Phase D champion contract required')
        pending.clear();pending['index']=0
        return Backend(c,catalog=catalog,compiled_functions=functions,**identity)

    def policy(c,*,model,result):
        if c!=contract:raise ValueError('same Phase D champion contract required')
        def select(observation,*,rng):
            snapshot=pending.pop('snapshot',None)
            if snapshot is None or observation.metadata.get('owner_prefix_sha256')!=snapshot.prefix_sha256:
                raise ValueError('search must use current original B2 encoded own snapshot')
            index=pending['index'];pending['index']+=1
            response=client.request(snapshot,decision_index=index)
            action=response['action_index']
            if type(action) is not int or not 0<=action<9 or not observation.legal_action_mask[action]:
                raise ValueError('shared GPU returned nonliteral/illegal request action')
            return PolicyDecision(action,'phase-d-shared-gpu-search',action_probability=1.,
                value_estimate=response['q_values'][action],metadata={'search':response})
        return RawPolicy(c,select,lambda:None)
    return OwningActorFactory('jax-owning',contract,backend,policy)
