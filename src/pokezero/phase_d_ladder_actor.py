"""Explicit searched actor over the unchanged original B2 CPU panel transport.

The control raw policy is also the declared fallback on a counted search
refusal. A timeout never retries a root; the original60-second actor deadline
still applies to search plus fallback. Actual panel row.arm supplies the search
axis; the legacy transport config's policy_mode remains raw by contract.
"""
from dataclasses import replace
import os
from pathlib import Path
import time
from .live_protocol_actor import ActorContract,RawPolicy
from .live_protocol_bridge import OwningActorFactory,require_actor_factory
from .policy import PolicyDecision
from .phase_d_ladder_transport import Client
from .phase_d_ladder_panel import encoded


def factory(config,*,raw_factory,client,telemetry_path):
    if (type(raw_factory)is not OwningActorFactory or raw_factory.contract!=ActorContract.champion()
            or type(client)is not Client):raise ValueError('same9375 raw fallback factory and explicit GPU rung required')
    require_actor_factory(config,raw_factory)
    pending={};path=Path(telemetry_path);path.parent.mkdir(parents=True,exist_ok=True)
    def emit(row):
        with path.open('ab')as stream:stream.write(encoded(row)+b'\n');stream.flush();os.fsync(stream.fileno())
    def backend(contract,*,env_config,**identity):
        pending.clear();pending['index']=0
        provider=raw_factory.backend_factory(contract,env_config=env_config,**identity)
        encode=provider.encode
        def capture(snapshot):
            output=encode(snapshot);pending['snapshot']=snapshot;return output
        provider.encode=capture
        return provider
    def policy(contract,*,model,result):
        fallback=raw_factory.policy_factory(contract,model=model,result=result)
        class SearchedPolicy(RawPolicy):
            def claim(self,c):fallback.claim(c);super().claim(c)
            def close(self):fallback.close()
        def select(observation,*,rng):
            snapshot=pending.pop('snapshot',None)
            if snapshot is None or observation.metadata.get('owner_prefix_sha256')!=snapshot.prefix_sha256:
                raise ValueError('same current B2 owning root required')
            index=pending['index'];pending['index']+=1
            started=time.monotonic();response=client.request(snapshot,decision_index=index)
            row=dict(rung=client.rung,battle_id=snapshot.battle_id,owner=snapshot.owner,decision_index=index,
                prefix_sha256=snapshot.prefix_sha256,search=response,fell_back=False)
            if response['status']=='COMPLETE':
                action=response.get('action_index');q=response.get('q_values')
                if (type(action)is not int or not 0<=action<9 or not observation.legal_action_mask[action]
                        or type(q)not in (tuple,list)or len(q)!=9 or q[action]is None):
                    raise ValueError('literal legal searched action and completed Q required')
                decision=PolicyDecision(action,'champion-search-ladder-'+client.rung,action_probability=1.,value_estimate=q[action],
                    metadata=dict(search=response))
            else:
                decision=fallback.select_action(observation,rng=rng)
                row['fell_back']=True;row['fallback_action_index']=decision.action_index
                decision=replace(decision,metadata={**dict(decision.metadata),
                    'search_ladder':dict(rung=client.rung,status='REFUSED_RAW_FALLBACK',refusal=response.get('refusal'))})
            row['acting_wall_seconds']=time.monotonic()-started;emit(row)
            return decision
        return SearchedPolicy(contract,select,fallback.reset)
    value=OwningActorFactory('jax-owning',raw_factory.contract,backend,policy)
    from .owning_worker_runtime import register_factory,compilation_snapshot
    # The original raw factory remains the owner of the shared compiled B2
    # functions. Runtime reports its counters directly; no fake cache-hit claim.
    register_factory(value,lambda:{})
    return require_actor_factory(config,value)


class Runtime:
    def __init__(self,*,client,telemetry_path):
        from .owning_worker_runtime import ChampionWorkerRuntime
        self.original=ChampionWorkerRuntime();self.client=client;self.telemetry_path=telemetry_path;self.factory=None
    async def run(self,config):
        from . import foulplay_bridge as bridge
        original=self.original.prepare(config)
        if self.factory is None:self.factory=factory(config,raw_factory=original,client=self.client,telemetry_path=self.telemetry_path)
        model,result,env,spec,vocab,dex,masks,digest=self.original.runtime
        policy_id='champion-search-ladder-'+self.client.rung
        policy=bridge._build_policy(config=config,model=model,result=result,value_model=model,value_result=result,
            env_config=env,rollout_config=bridge.RolloutConfig(max_decision_rounds=config.max_decision_rounds,
            format_id=config.format_id),policy_id=policy_id)
        return await bridge._run_controlled_foulplay_games(config,policy=policy,policy_id=policy_id,
            vocab=vocab,dex=dex,observation_spec=spec,feature_masks=masks,checkpoint_sha256=digest,
            owning_actor_factory=self.factory,actor_runtime=(model,result,env))
