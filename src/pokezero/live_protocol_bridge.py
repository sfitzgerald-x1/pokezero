"""Source-only opt-in seams for actual owning-line raw champion evaluation.

No backend assembly, model loading, process launch or qualification lives here.
Factories are explicit reviewed dependencies. The transport still needs separate
actual-wire/runtime/B2 admission before any B3 scientific interpretation.
"""
from dataclasses import dataclass
import asyncio
import hashlib
import json
import math
import os
import time
from typing import Callable

from .live_protocol_actor import (ActorContract, JaxBackend, LiveProtocolActor,
    OwnedProvider, PythonBackend, UNQUALIFIED, _actionable, _packet, _require, _terminal)


ENCODERS = ('legacy-mixed', 'python-owning', 'jax-owning')


def bridge_event_snapshot(event):
    """Failure-only bounded envelope; never an actor input or feature."""
    try:payload=json.dumps(event,sort_keys=True,separators=(',',':'),ensure_ascii=True)
    except Exception:payload='[unserializable bridge event]'
    def field(name,limit):
        value=event.get(name) if isinstance(event,dict) else None
        return value[:limit] if type(value) is str else None
    return dict(event_type=field('type',128),battle_id=field('battleId',256),
        message=field('message',2048),payload_json=payload[:8192],
        payload_truncated=len(payload)>8192,payload_sha256=hashlib.sha256(payload.encode()).hexdigest())


def encoder_provenance(kind):
    _require(type(kind) is str and kind in ENCODERS, 'explicit encoder axis required')
    return dict(encoder_kind=kind,
        input_source=('LEGACY_MIXED_OMNISCIENT_AND_PRIVATE_REQUESTS' if kind=='legacy-mixed'
                      else 'ACTUAL_GETPLAYERSTREAMS_OWNER_LINES_UNQUALIFIED'),
        qualification_status='UNQUALIFIED_SOURCE_PATH', **UNQUALIFIED)


def validate_encoder_config(config):
    kind=config.encoder_kind
    encoder_provenance(kind)
    if kind=='legacy-mixed':
        _require(getattr(config,'owning_decision_timeout_seconds',None) is None
            and getattr(config,'owning_public_single_line',False) is False,
            'owning controls require an owning encoder')
        return
    limit=config.owning_decision_timeout_seconds
    _require(type(config.owning_public_single_line) is bool, 'typed owning dispatch flag required')
    _require(limit is None or (type(limit) in (int,float) and math.isfinite(limit)
        and 0 < limit <= 3600), 'explicit finite owning decision limit required')
    _require(config.policy_mode=='raw' and config.capture_driver=='checkpoint'
        and config.checkpoint is not None and config.value_checkpoint is None
        and config.opponent_policy_mode=='foul-play' and config.format_id=='gen3randombattle'
        and config.device=='cpu' and config.temperature==1. and config.belief_set_source is True
        and config.opponent_legal_mask_mode=='hidden' and config.search_time_ms==1000
        and config.foulplay_mcts_iterations is None and config.opponent_crash_retries==0
        and config.allow_search_fallback is False,
        'owning encoder requires raw champion/checkpoint/CPU/public source/FoulPlay1000 without retry or fallback')
    _require(not any(getattr(config,name) for name in (
        'engine_oracle_belief','engine_policy_opponent','engine_opponent_priors',
        'engine_override_telemetry','engine_root_selector_shadow','engine_early_stop',
        'belief_start_overrides','leaf_rollout_sampling','live_continuation_smoke','live_continuation_oracle'))
        and not config.leaf_rollout_rounds and config.engine_model_path is None
        and config.engine_tables_path is None and config.live_continuation_oracle_progress_dir is None,
        'owning encoder forbids search/oracle/probe paths')
    _require(type(config.owning_terminal_timeout_seconds) in (int,float)
        and math.isfinite(config.owning_terminal_timeout_seconds)
        and 0 < config.owning_terminal_timeout_seconds <= 30,
        'bounded owning terminal-drain timeout required')
    _require(type(config.seed_start) is int and type(config.games) is int
        and 0 <= config.seed_start and 0 < config.games < 2**31
        and config.seed_start+config.games <= 2**32, 'bounded fixed game/stream identity required')


@dataclass(frozen=True)
class OwningActorFactory:
    encoder_kind: str
    contract: ActorContract
    backend_factory: Callable
    policy_factory: Callable

    def __post_init__(self):
        _require(self.encoder_kind in ENCODERS[1:] and type(self.contract) is ActorContract
            and callable(self.backend_factory) and callable(self.policy_factory),
            'explicit owning encoder and bound factories required')

    def actor(self, runtime):
        _require(type(runtime) is tuple and len(runtime)==3, 'bound model/result/env runtime required')
        model,result,env_config=runtime
        expected=PythonBackend if self.encoder_kind=='python-owning' else JaxBackend
        def backend(contract,**identity):
            value=self.backend_factory(contract,env_config=env_config,**identity)
            if not isinstance(value,expected):
                if isinstance(value,OwnedProvider):
                    value.close()
                raise ValueError('factory backend differs from encoder axis')
            return value
        return LiveProtocolActor(self.contract,backend_factory=backend,
            policy_factory=lambda contract:self.policy_factory(contract,model=model,result=result))


def require_actor_factory(config, factory):
    validate_encoder_config(config)
    if config.encoder_kind=='legacy-mixed':
        _require(factory is None, 'legacy-mixed cannot silently use an owning actor factory')
        return None
    _require(type(factory) is OwningActorFactory and factory.encoder_kind==config.encoder_kind
        and factory.contract==ActorContract.champion(), 'matching exact champion owning actor factory required')
    _require(not os.environ.get('POKEZERO_TRAIT_CAPTURE'), 'omniscient trait capture forbidden in owning path')
    return factory


def require_session(config, session):
    if config.encoder_kind=='legacy-mixed':
        _require(session is None, 'legacy encoder cannot use an owning session')
    else:
        _require(type(session) is OwningWireSession and session.config is config
            and session.factory.encoder_kind==config.encoder_kind, 'matching owning session required')


class OwningWireSession:
    def __init__(self, config, factory, runtime, *, seed, battle_id, start_seed):
        self.config,self.factory=config,require_actor_factory(config,factory)
        _require(self.factory is not None, 'owning session requires opt-in')
        self.actor=self.factory.actor(runtime)
        self.seed=seed
        self.battle_id=battle_id
        self.start_seed=start_seed
        self.started=False
        self.owner=config.pokezero_player
        self.owner_lines=()
        self.next_round=0
        self.terminal_deadline=None
        self.terminal_streams={}
        self.terminal_marker=False
        self.trajectory=None
        self.receipt=None
        self.pending_choices={}
        self.submission_id=None
        self.submissions=0
        self.acknowledged=0
        self.ack_deadline=None
        self.first_bridge_failure_event=None

    def capture_event_failure(self,error,event=None,*,snapshot=None):
        if self.first_bridge_failure_event is None:
            self.first_bridge_failure_event=snapshot if snapshot is not None else bridge_event_snapshot(event)
        error.add_note('owning bridge event: '+json.dumps(self.first_bridge_failure_event,sort_keys=True))

    def reject_event(self,reason,event,*,error_type=ValueError):
        error=error_type(reason)
        self.capture_event_failure(error,event)
        raise error

    def prepare_submission(self,choices,*,requested_players,decision_round):
        _require(not self.pending_choices and self.started and self.terminal_deadline is None
            and type(decision_round) is int and 0<=decision_round<2**31
            and decision_round==self.next_round==self.submissions,
            'stale or overlapping owning submission')
        _require(type(choices) is dict and set(choices)==set(requested_players) and bool(choices)
            and set(choices)<={'p1','p2'} and all(type(choice) is str and choice.strip()
                and len(choice)<=4096 and '\n' not in choice and '\r' not in choice for choice in choices.values()),
            'exact bounded requested-player submissions required')
        self.pending_choices=dict(choices)
        self.submission_id=decision_round
        self.submissions+=1
        self.ack_deadline=time.monotonic()+120.  # Existing transport wait bound, never refreshed by events.
        return self.submission_id

    def acknowledge(self,event):
        try:
            _require(type(event) is dict and set(event)=={'type','battleId','player','choice','submissionId'}
                and event['type']=='choice_ack' and event['battleId']==self.battle_id,
                'literal owning choice acknowledgement envelope required')
            _require(type(event['submissionId']) is int and 0<=event['submissionId']<2**31
                and event['submissionId']==self.submission_id
                and type(event['player']) is str and event['player'] in self.pending_choices
                and type(event['choice']) is str and event['choice']==self.pending_choices[event['player']],
                'foreign, stale, duplicate or mismatched owning choice acknowledgement')
            self.wait_timeout()
            del self.pending_choices[event['player']]
            self.acknowledged+=1
            if not self.pending_choices:self.ack_deadline=None
            self.check_terminal_deadline()
        except BaseException as error:
            self.capture_event_failure(error,event)
            raise

    def begin(self):
        self.actor.begin_game(battle_id=self.battle_id,owner=self.owner,seed=self.seed,
                              stream_id=self.seed-self.config.seed_start+1)

    def route_stream(self,event):
        _require(type(event) is dict, 'literal stream event required')
        _require(type(event.get('battleId')) is str and event['battleId']==self.battle_id,
                 'owning stream battle mismatch')
        _require(type(event.get('stream')) is str and event['stream'] in ('p1','p2','omniscient'),
                 'unknown actual stream role')
        lines=event.get('lines')
        _require(type(lines) is list and bool(lines) and all(type(line) is str
            and line.startswith('|') and '\n' not in line and '\r' not in line for line in lines),
                 'uncoerced literal stream lines required')
        role=event['stream']
        _require(role not in self.terminal_streams, 'postterminal stream chunk')
        terminal=None
        for line in lines:
            _require(terminal is None or line=='|', 'postterminal line in terminal-drain chunk')
            if line.startswith('|win|'):
                _require(line in ('|win|'+self.config.pokezero_username,
                                  '|win|'+self.config.foulplay_username), 'foreign owner terminal winner')
            if _terminal(line):
                terminal=line
                self.wait_for_owner_terminal()
        if terminal is not None:
            _require(not self.terminal_streams or terminal in self.terminal_streams.values(),
                     'actual terminal streams disagree')
        if role==self.owner:
            groups = [tuple(lines)]
            if self.config.owning_public_single_line:
                groups = []
                for index, line in enumerate(lines):
                    # The actor seals on terminal consumption. Preserve its
                    # accepted terminal+padding suffix as one atomic call.
                    if _terminal(line):
                        groups.append(tuple(lines[index:]))
                        break
                    groups.append((line,))
            for group in groups:
                self.actor.accept_owner_stream(battle_id=self.battle_id,owner=self.owner,
                    generation=self.actor.generation,prefix_sha256=self.actor.prefix_sha256,lines=group)
            self.owner_lines+=tuple(lines)
        if terminal is not None:
            self.terminal_streams[role]=terminal
        self.check_terminal_deadline()

    def requested(self,event):
        _require(type(event) is dict and event.get('battleId')==self.battle_id
            and self.started and self.terminal_deadline is None,
                 'foreign or postterminal ready event')
        raw=event.get('requested')
        _require(type(raw) is list and bool(raw) and all(type(x) is str for x in raw)
            and tuple(raw)==tuple(x for x in ('p1','p2') if x in raw), 'canonical unique ready roles required')
        return tuple(raw)

    def boundary(self, *, requested_players, decision_round, requests):
        _require(type(requested_players) is tuple and bool(requested_players)
            and requested_players==tuple(x for x in ('p1','p2') if x in requested_players)
            and type(decision_round) is int and decision_round==self.next_round
            and self.started and self.terminal_deadline is None, 'stale or invalid owning boundary')
        snapshot=self.actor.snapshot_boundary()
        line=snapshot.own_request_line
        _require(line is not None and requests.get(self.owner)==line, 'own request journal/cache mismatch')
        _require(_actionable(_packet(line,self.owner))==(self.owner in requested_players),
                 'ready roles differ from actual own request')
        for seat in ('p1','p2'):
            current=requests.get(seat)
            _require(type(current) is str and current.startswith('|request|'), 'current actual requests required')
            _require(_actionable(_packet(current,seat))==(seat in requested_players),
                     'ready roles differ from actual request cache')
        return snapshot

    def wait_for_owner_terminal(self):
        if self.terminal_deadline is None:
            self.terminal_deadline=time.monotonic()+self.config.owning_terminal_timeout_seconds

    def mark_terminal(self,event):
        _require(type(event) is dict and event.get('battleId')==self.battle_id
            and event.get('type')=='terminal' and not self.terminal_marker,
            'foreign or duplicate bridge terminal marker')
        self.terminal_marker=True
        self.wait_for_owner_terminal()

    def mark_started(self,event):
        _require(type(event) is dict and not self.started and self.terminal_deadline is None
            and event.get('type')=='started' and event.get('battleId')==self.battle_id
            and type(event.get('formatid')) is str and event['formatid']==self.config.format_id
            and type(event.get('seed')) is str and event['seed']==self.start_seed,
            'invalid or duplicate owning start witness')
        self.started=True

    @property
    def terminal_ready(self):
        return (self.started and set(self.terminal_streams)=={'p1','p2','omniscient'}
            and self.terminal_marker and not self.pending_choices)

    def wait_timeout(self):
        deadlines=[d for d in (self.terminal_deadline,self.ack_deadline) if d is not None]
        if not deadlines:
            return 120.
        remaining=min(deadlines)-time.monotonic()
        if remaining <= 0:
            raise TimeoutError('actual owning terminal/acknowledgement deadline expired')
        return remaining

    def check_terminal_deadline(self):
        # Synchronous parsing, closure and publication cannot bypass the same
        # absolute clock merely because no further transport read is needed.
        if self.terminal_deadline is not None:
            self.wait_timeout()

    async def terminal_io(self, operation):
        """Bound an owning transport operation once any terminal was observed.

        A factory avoids constructing an unawaited coroutine after expiry. The
        post-check also refuses an operation that swallows timeout cancellation.
        No new deadline is created for each write.
        """
        if self.terminal_deadline is None:
            return await operation()
        remaining = self.wait_timeout()
        result = await asyncio.wait_for(operation(), timeout=remaining)
        self.check_terminal_deadline()
        return result

    @property
    def terminal_seen(self):
        return any(_terminal(line) for line in self.owner_lines)

    def close(self, *, failed):
        self.receipt=self.actor.close_game()
        in_budget=(self.terminal_deadline is not None and time.monotonic()<self.terminal_deadline)
        clean=self.receipt['status']=='TERMINAL_UNQUALIFIED' and self.terminal_ready and in_budget
        if self.trajectory is not None:
            self.trajectory.metadata={**dict(self.trajectory.metadata),
                'encoder_provenance':self.provenance(failed=failed or not clean)}
        if not failed:
            self.check_terminal_deadline()
        _require(failed or clean,
                 'owning game requires actual owner terminal and clean closure')
        return self.provenance(failed=failed or not clean)

    def provenance(self, *, failed=False):
        return dict(encoder_provenance(self.config.encoder_kind),
            actor_namespace=self.factory.contract.namespace,
            status='FAILED_CLOSED_UNQUALIFIED' if failed else 'CLOSED_UNQUALIFIED',
            actual_owner_terminal_seen=self.terminal_seen,
            completed_terminal_streams=sorted(self.terminal_streams),
            bridge_terminal_marker=self.terminal_marker,
            bridge_started_witness=self.started,
            owner_terminal_timeout_seconds=self.config.owning_terminal_timeout_seconds,
            owning_decision_timeout_seconds=self.config.owning_decision_timeout_seconds,
            owning_public_single_line=self.config.owning_public_single_line,
            choice_acknowledgements=dict(submissions=self.submissions,acknowledged=self.acknowledged,
                pending_players=sorted(self.pending_choices),last_submission_id=self.submission_id),
            first_bridge_failure_event=self.first_bridge_failure_event,
            actor_receipt=self.receipt)
