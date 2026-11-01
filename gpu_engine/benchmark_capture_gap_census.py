"""Finite diagnostic census; source capture gaps never become valid observations.

The first512 of2048 declared fresh rows form the census. The other1536 are
fixed timing windows, not completed-game or coverage credit. The original
retained policy is used only while admission remains valid. After a capture
gap, deterministic original-engine legal fallback advances the physical game;
that game's scored-observation mask remains false for the rest of its life.
This module supplies no B4/fidelity/strength admission or physics qualification.
"""
from collections import Counter, defaultdict
from dataclasses import dataclass
import time
import random
import numpy as np

SCHEMA = 'pokezero.capture_gap_census.diagnostic.v1'


@dataclass(frozen=True)
class Plan:
    batch: int = 2048
    census_games: int = 512
    max_steps: int = 2048
    max_turn: int = 1000
    timing_warmup: int = 8
    timing_steps: int = 24
    max_elapsed_seconds: float = 14400.
    max_examples_per_kind: int = 3
    fallback_seed: int = 0
    fallback: str = 'seeded-uniform-original-engine-legal-native.v1'

    def validate(self):
        for name in ('batch','census_games','max_steps','max_turn','timing_steps'):
            if type(getattr(self,name)) is not int or getattr(self,name)<1:
                raise ValueError('positive finite integer '+name+' required')
        if type(self.timing_warmup)is not int or self.timing_warmup<0:
            raise ValueError('nonnegative integer warmup required')
        if self.batch!=2048 or self.census_games!=512 or self.max_turn>1000 or self.max_steps>2048:
            raise ValueError('explicit2048freshrows/512census/finitecaps required')
        if self.timing_warmup+self.timing_steps>self.max_steps:
            raise ValueError('timing window must fit fixed step cap')
        if type(self.max_elapsed_seconds)not in(int,float)or not np.isfinite(self.max_elapsed_seconds)or self.max_elapsed_seconds<=0:
            raise ValueError('finite positive wall backstop required')
        if type(self.max_examples_per_kind)is not int or not 0<=self.max_examples_per_kind<=8:
            raise ValueError('bounded perkind examples required')
        if self.timing_warmup+self.timing_steps!=32:
            raise ValueError('fixed32step timing-only window required')
        if type(self.fallback_seed)is not int or not 0<=self.fallback_seed<2**31:
            raise ValueError('explicit independent diagnostic fallback seed required')
        if self.fallback!='seeded-uniform-original-engine-legal-native.v1':
            raise ValueError('fixed deterministic fallback recipe required')
        return self


def legal_fallback(legal, requested, active, *, seed, streams, step):
    """Seeded uniform native index0..8 from ORIGINAL engine action mask only.

    PP/Encore/charge/forced-switch/traps/Struggle stay in that original mask.
    A requested seat with no legal action is an engine refusal, never coerced.
    Waiting seats carry ignored zero; this cannot legalize a requested action.
    """
    legal=np.asarray(legal);requested=np.asarray(requested);active=np.asarray(active)
    b=len(active)
    if legal.shape!=(b,2,9)or legal.dtype!=np.bool_ or requested.shape!=(b,2)or requested.dtype!=np.bool_ or active.dtype!=np.bool_:
        raise ValueError('exact original engine masks required')
    no_action=active&np.any(requested&~legal.any(axis=2),axis=1)
    streams=np.asarray(streams)
    if streams.shape!=(b,)or streams.dtype!=np.int32 or type(seed)is not int or type(step)is not int:
        raise ValueError('declared independent fallback coordinates required')
    choices=np.zeros((b,2),np.int32)
    for row in np.flatnonzero(active&~no_action):
        for owner in np.flatnonzero(requested[row]):
            options=np.flatnonzero(legal[row,owner])
            # A local generator per coordinate never reads or mutates engine
            # chance keys, champion sampling keys, or another game's RNG.
            key=f'pokezero.B4.diagnostic-gap-fallback.v1:{seed}:{int(streams[row])}:{step}:{int(owner)}'
            choices[row,owner]=options[random.Random(key).randrange(len(options))]
    return choices,no_action


class Ledger:
    """Counts current callback occurrences and unique games, never sticky totals."""
    def __init__(self, plan, streams):
        self.plan=plan.validate();self.streams=np.array(streams,copy=True)
        if self.streams.shape!=(plan.batch,)or self.streams.dtype!=np.int32 or np.any(self.streams<=0)or len(np.unique(self.streams))!=plan.batch:
            raise ValueError('all2048distinct original stream identities required')
        b=plan.batch
        self.capture_good=np.ones(b,bool);self.engine_failed=np.zeros(b,bool)
        self.done=np.zeros(b,bool);self.capped=np.zeros(b,bool)
        self.engine_steps=np.zeros(b,np.int64);self.first_gap=[None]*b
        self.terminal=[None]*b;self.kind_occurrences=Counter();self.kind_games=defaultdict(set)
        self.step_rows=[];self.kind_examples=defaultdict(list);self.counts_complete=True

    def active(self, step):
        role_active=(np.arange(self.plan.batch)<self.plan.census_games)|(step<self.plan.timing_warmup+self.plan.timing_steps)
        return role_active&~self.done&~self.engine_failed&~self.capped

    def mark_capture(self, failed, *, step, packets):
        failed=np.asarray(failed)
        if failed.shape!=(self.plan.batch,)or failed.dtype!=np.bool_:raise ValueError('bool row capture failures required')
        first=failed&self.capture_good
        for row in np.flatnonzero(first):self.first_gap[row]=dict(step=int(step),packet=packets(int(row)))
        self.capture_good &= ~failed
        return first

    def observe(self, *, step, active, executed, engine_failed, done, turns, packet, terminal):
        b=self.plan.batch
        for x in(active,executed,engine_failed,done):
            if np.asarray(x).shape!=(b,)or np.asarray(x).dtype!=np.bool_:raise ValueError('exact bool row execution masks required')
        if np.any(executed&~active):raise ValueError('engine executed outside fixed active rows')
        self.engine_steps+=executed
        self.engine_failed |= active&engine_failed
        newly_done=active&executed&~engine_failed&done&~self.done
        for row in np.flatnonzero(newly_done):self.terminal[row]=terminal(int(row))
        self.done |= newly_done
        self.capped |= active&~self.done&~self.engine_failed&((np.asarray(turns)>=self.plan.max_turn)|(step+1>=self.plan.max_steps))
        # These are literal new per-call open sites/raw records. They do not
        # label a callback as the sole cause of a source or physical refusal.
        for row in np.flatnonzero(active):
            p=packet(int(row));role='census'if row<self.plan.census_games else'timing-only'
            self.counts_complete &= bool(p.get('diagnostic_counts_complete',True))
            domain='capture-context'if p.get('context_unknown',False)else'capture'
            for site in p.get('unhandled_sites',[]):self._count(role,domain,'site:'+str(site),row,step,p)
            for raw in p.get('raw_unsupported_kinds',[]):self._count(role,domain,'raw:'+str(raw),row,step,p)
            for move in p.get('raw_unsupported_moves',[]):self._count(role,domain,'raw-move:'+str(move['move_id'])+':'+str(move['name']),row,step,p)
            for risk in p.get('capture_risk_kinds',[]):self._count(role,domain,'risk:'+str(risk),row,step,p)
            for kind in p.get('engine_failures',[]):self._count(role,'engine',str(kind),row,step,p)
        self.step_rows.append(dict(step=int(step),launch_rows=b,active_engine_rows=int(active.sum()),executed_engine_rows=int(executed.sum()),
            capture_valid_rows=int((active&self.capture_good).sum()),fallback_rows=int((active&~self.capture_good).sum()),observed_epoch=time.time()))

    def _count(self, role, domain, kind, row, step, packet):
        key=(role,domain,kind);self.kind_occurrences[key]+=1;self.kind_games[key].add(int(row))
        if len(self.kind_examples[key])<self.plan.max_examples_per_kind:
            self.kind_examples[key].append(dict(stream_id=int(self.streams[row]),step=int(step),packet=packet))

    def summary(self):
        roles=np.arange(self.plan.batch)<self.plan.census_games
        def reduce(mask):
            return dict(original_denominator=int(mask.sum()),engine_terminal_games=int((mask&self.done).sum()),
                capture_gap_games=int((mask&~self.capture_good).sum()),engine_refused_games=int((mask&self.engine_failed).sum()),
                capped_games=int((mask&self.capped).sum()),unknown_terminal_games=int((mask&~self.done).sum()),
                terminal_without_capture_gap=int((mask&self.done&self.capture_good).sum()))
        # Sum occurrences, but UNION row identities across original context
        # strata. Timing-only rows never enter the census-games denominator.
        by_source=Counter();source_rows=defaultdict(set)
        for key,occurrences in self.kind_occurrences.items():
            role,domain,kind=key;source='engine'if domain=='engine'else'capture-gap'
            grouped=(role,source,kind);by_source[grouped]+=occurrences
            source_rows[grouped].update(self.kind_games[key])
        timing=reduce(~roles);timing={key.replace('_games','_rows'):value for key,value in timing.items()}
        timing['original_row_denominator']=timing.pop('original_denominator')
        return dict(schema=SCHEMA,census=reduce(roles),timing_only=timing,
            unsupported=[dict(role=k[0],domain=k[1],kind=k[2],occurrences=v,**({'games_affected':len(self.kind_games[k])}if k[0]=='census'else{'timing_rows_affected':len(self.kind_games[k])}),examples=self.kind_examples[k])for k,v in sorted(self.kind_occurrences.items())],
            unsupported_by_source=[dict(role=k[0],source=k[1],kind=k[2],occurrences=v,**({'games_affected':len(source_rows[k])}if k[0]=='census'else{'timing_rows_affected':len(source_rows[k])}))for k,v in sorted(by_source.items())],
            per_step=self.step_rows,cohort=[dict(stream_id=int(self.streams[i]),role='census'if roles[i]else'timing-only',
                first_capture_gap=self.first_gap[i],engine_failed=bool(self.engine_failed[i]),engine_terminal=bool(self.done[i]),
                capped=bool(self.capped[i]),executed_steps=int(self.engine_steps[i]),terminal=self.terminal[i],
                observation_scoring_valid=bool(self.capture_good[i]))for i in range(self.plan.batch)],
            full_B4_qualified=False,fidelity_qualified=False,strength_qualified=False,
            callback_counts_complete=self.counts_complete,
            counts_scope='Per-call source sites, raw guard failures, per-move breakdown and risk bits are separate nonexclusive cooccurrences. Never sum them as distinct failures or intrinsic causes.',
            timing_scope='Actual launchshape and active/admitted counts required; timings never imply2048completed games.')


def _row_select(mask, before, after):
    import jax
    import jax.numpy as j
    # BattleState/cursor/profile dynamic leaves are declared B-leading here;
    # GLOBAL bindings are never passed through this selector.
    def choose(a,b):
        if a.shape!=b.shape or not a.ndim or a.shape[0]!=len(mask):
            raise ValueError('declared row carry geometry differs')
        return j.where(j.asarray(mask).reshape((len(mask),)+(1,)*(a.ndim-1)),b,a)
    return jax.tree.map(choose,before,after)


def _driver_host_overlap_ms(diagnostics):
    """Already measured host spans; overlapping with synchronized driver phases.

    No measurement, guard recheck, barrier or device call occurs here. Missing
    data means unmeasured, never zero. Names retain the original driver's units.
    """
    measured=getattr(diagnostics,'host_timings',None)
    return {}if measured is None else {'driver_host_overlap.'+name:value for name,value in measured.items()}


def _attach_refused_diagnostics(packets,capture,refused,*,_host_cache=None):
    """Original literal row slices only; data never returns to policy/guards."""
    from . import engine_outcome_diagnostics as literal_diagnostics
    for item in literal_diagnostics.refused_rows(capture,refused,_host_cache=_host_cache)['rows']:
        packets[item['row']]['literal_diagnostics']=item


def _bound_engine_move_names(request):
    """Decode the ORIGINAL checked request record's canonical metadata JSON."""
    import json
    metadata=json.loads(request['metadata'])
    names=metadata['vocabulary']['moves']
    if type(names)is not list or not names or any(type(name)is not str or not name for name in names):
        raise ValueError('original canonical engine move vocabulary required')
    if len(names)!=len(request['engine_table'].power):
        raise ValueError('engine move vocabulary/table geometry differs')
    return tuple(names)


class _Runtime:
    """Original runtime edges; diagnostic receipts cannot enter the Session."""
    def __init__(self,runtime,plan,pool,indices,streams,constructor_seed,gender):
        import jax
        import jax.numpy as j
        from . import benchmark_retained_games as retained,engine_retained_encoder_driver as driver
        from . import engine_execution_completeness as ex,engine_kernel_frame_risk as risk
        self.jax,self.j,self.driver,self.retained=jax,j,driver,retained
        self.runtime,self.record=runtime,retained._checked(runtime)
        self.static=driver._checked(self.record['prepared'])
        if self.static.get('frame_input_mode')!='source-witness' or self.static['execution_compiled'] is None or self.static['kernel_risk_compiled'] is None:
            raise ValueError('same original full source-witness census runtime required')
        self.plan=plan;self.streams=streams;self.table=self.static['request']['engine_table']
        self.move_names=_bound_engine_move_names(self.static['request'])
        started=time.perf_counter_ns()
        self.session=driver.construct(pool,indices,prepared=self.record['prepared'],stream_ids=j.asarray(streams),seed=constructor_seed,
            **({}if gender is None else dict(gender_initialization=gender)))
        jax.block_until_ready(self.session.result)
        self.construction_ms=(time.perf_counter_ns()-started)/1e6
        self.engine=self.session.state.core.engine
        self.cursor=self.session.state.core.cursor
        self.profile=self.session.state.core.weather_profile
        # Fresh per-call local accounting is DIAGNOSTIC ONLY: it measures actual
        # newly requested sites, does not close/repair the Session's ledger,
        # carries no both-owner proof, and never grants observation admission.
        self.empty_ledger=ex.initialize(j.asarray(streams),compiled=self.static['execution_compiled'],capacity=32,receipt_capacity=64)
        self.empty_risk=risk.initialize_device(j.asarray(streams),self.static['kernel_risk_compiled'].device)
        self.diag=jax.jit(self._diagnostic_step)
        self.diag_executable=None
        from . import transition
        self.mask=jax.jit(lambda state:transition.action_mask(state,self.table))
        self.last_packet=[{}for _ in streams]

    def _diagnostic_step(self,engine,cursor,profile,actions,chance,active):
        import jax.numpy as j
        from . import engine_lifecycle_callbacks as life,engine_execution_completeness as ex
        from . import engine_kernel_frame_risk as risk,engine_public_events as events
        from . import transition,engine_outcome_diagnostics as diagnostics
        b=len(active);s=self.static;bindings=s['public'].device.causal.base
        tape=life.LifecycleTape(j.zeros((b,64,10),j.int32),j.zeros(b,j.int32),j.full((b,),events.UNINSTRUMENTED,j.int32),cursor.turn,
            bindings.engine_binding,bindings.source_binding,bindings.move_binding,bindings.producer_binding,cursor.stream_id,
            j.zeros((b,64),j.int32),j.zeros((b,64),j.int32),j.zeros((b,64,2),j.int32),j.zeros((b,64,6),j.int32),cursor)
        tape=ex.begin_on_tape(tape,self.empty_ledger,s['execution_compiled'].device,active=active)
        local=risk.begin_frame_device(self.empty_risk,tape,s['kernel_risk_compiled'].device,active=active)
        tape=tape._replace(execution=tape.execution._replace(risks=risk.Context(local,s['kernel_risk_compiled'].device),diagnostics=diagnostics.empty(b)))
        weather=None if s['speed_profile']is None else(profile,s['speed_profile'].device)
        result=transition.step_with_chance(engine._replace(done=engine.done|~active),actions,self.table,chance,
            _public_trace=tape,_lifecycle_trace=True,_weather_hp_context=weather,
            **{('_'+name):s[name]for name in ('leftovers_context','status_residual_context','direct_hit_context','substitute_context','toxic_context','recoil_context','calm_mind_context','swords_dance_context','spikes_context','thunder_wave_context','sleep_powder_context')})
        physical=_row_select(active,engine,result.step.state)
        next_cursor=_row_select(active,cursor,result.callbacks.cursor)
        next_profile=profile if profile is None else _row_select(active,profile,result.weather_profile)
        from . import engine_source_frame_witness as witness
        raw_good=witness._raw_complete_words(result.callbacks,s['frame_witness_compiled'].device)
        namespace=ex._validate(result.callbacks.execution.ledger,s['execution_compiled'].device)&ex._tape(result.callbacks.execution.ledger,result.callbacks,s['execution_compiled'].device)
        return physical,next_cursor,next_profile,result.callbacks,result.step.invalid_actions,result.step.placeholder_moves,raw_good,namespace

    def requested(self):
        state=self.engine;rows=np.arange(len(state.done))[:,None];sides=np.arange(2)[None,:]
        hp=np.asarray(state.hp)[rows,sides,np.asarray(state.active)]
        forced=np.asarray(state.forced_switch)|(hp<=0)
        return ~forced.any(axis=1)[:,None]|forced

    def check_physical_source(self):
        # Policy normally performs this exact outer check. The diagnostic
        # physical advance still requires it when discarded policy work stops.
        self.retained._checked(self.runtime,session=self.session)

    def policy(self,active,ledger,step,sampling_seed):
        """Fixed4096owner BF16 shape; only admitted rows may choose actions."""
        j=self.j;record=self.retained._checked(self.runtime,session=self.session)
        eligible,failed,reasons=self.retained._admission(self.session.result,active&ledger.capture_good,record['config'])
        ledger.mark_capture(failed,step=step,packets=lambda row:dict(kind='retained-or-request-admission',owner_reason=reasons[row].tolist()))
        eligible &= ledger.capture_good[:,None]
        # Real original encoder on actual current physical engine. The source
        # carry/request remains poisoned after a gap; we do not re-enable it.
        # Unadmitted outputs are shadow timing inputs ONLY and never actions.
        core=self.session.state.core._replace(engine=self.engine,
            admitted=self.session.state.core.admitted&j.asarray(ledger.capture_good))
        encoded=[];encode_ms=[];own_admitted=[]
        for observer in self.session.state.observers:
            start=time.perf_counter_ns()
            facts,out=self.jax.block_until_ready(self.static['encode_fast'](core,observer))
            encode_ms.append((time.perf_counter_ns()-start)/1e6)
            encoded.append(out.observation);own_admitted.append(np.asarray(facts.admitted))
        args=tuple(j.stack([getattr(obs,name)for obs in encoded],axis=1).reshape((2*self.plan.batch,)+getattr(encoded[0],name).shape[1:])
            for name in ('numeric_features','categorical_ids','token_type_ids','attention_mask'))
        heads,compile_ms,forward_ms=self.retained._network(record,args)
        physical_rows=2*self.plan.batch
        if heads.policy_logits.shape!=(physical_rows,9)or heads.value.shape!=(physical_rows,)or heads.opponent_action_logits.shape!=(physical_rows,9):
            raise ValueError('original fixedshape three model heads required')
        finite=np.logical_and.reduce([np.isfinite(np.asarray(x)).reshape(physical_rows,-1).all(axis=1)for x in heads]).reshape(self.plan.batch,2)
        eligible &= np.stack([np.asarray(x.admitted)for x in encoded],axis=1)&np.stack(own_admitted,axis=1)&finite
        if not np.array_equal(np.asarray(core.cursor.stream_id),self.streams):raise ValueError('declared original stream identity changed')
        keys=self.retained._keys(sampling_seed,np.repeat(np.asarray(core.cursor.stream_id),2),np.repeat(np.asarray(core.frame_number),2),np.tile(np.arange(2,dtype=np.int32),self.plan.batch))
        legal=j.stack([x.legal_action_mask for x in encoded],axis=1).reshape(physical_rows,9)
        choices=np.asarray(self.retained.sample_actions(heads.policy_logits,legal,keys)).reshape(self.plan.batch,2)
        from . import engine_private_actions
        routed=[];bad=np.zeros(self.plan.batch,bool)
        for owner in(0,1):
            route=self.jax.block_until_ready(engine_private_actions.route_choice(core.requests[owner].actions,core.orders[owner],j.asarray(np.where(eligible[:,owner],choices[:,owner],-1),j.int32)))
            bad |= eligible[:,owner]&~np.asarray(route.admitted)
            routed.append(np.asarray(j.where(route.admitted,route.engine_action,0)))
        bad |= active&ledger.capture_good&np.any(self.requested()&~eligible,axis=1)
        ledger.mark_capture(bad,step=step,packets=lambda row:dict(kind='encoder-or-policy-output-unavailable',owner_reason=reasons[row].tolist()))
        return np.stack(routed,axis=1),dict(bf16_physical_owner_rows=physical_rows,admitted_policy_owner_rows=int((eligible&ledger.capture_good[:,None]).sum()),
            shadow_discarded_owner_rows=int(physical_rows-(eligible&ledger.capture_good[:,None]).sum()),bf16_forward_ms=forward_ms,bf16_compile_ms=compile_ms,
            shadow_actual_encoder_ms=encode_ms,shadow_actual_encoder_rows=self.plan.batch,
            shadow_scope='Actual encoder-produced tensors; unadmitted/frozen/timing-ended rows are discarded nonpolicy work, not valid current observations.')

    def advance(self,actions,active,scored,chance):
        from . import engine_execution_completeness as ex,engine_public_events as events,engine_source_frame_witness as source_witness
        j=self.j;before=self.engine;executed=np.zeros(len(active),bool);invalid=np.zeros(len(active),bool);phases={};diag_callbacks=None;diag_good=None;diag_namespace=None
        # One immutable host snapshot per actual array, local to this call.
        # No JAX row slicing/gathers or extra synchronization/barrier call.
        host_cache={}
        def host(value):
            key=id(value)
            if key not in host_cache:host_cache[key]=(value,np.asarray(value))
            return host_cache[key][1]
        kept=active&scored
        phases['original_request_rows']=int(kept.sum())
        if kept.any():
            old_frames=host(self.session.state.core.frame_number).copy()
            result=self.driver.advance(self.session,j.asarray(actions),active_rows=j.asarray(kept),chance_inputs=chance,collect_timings=True)
            self.jax.block_until_ready(result);d=self.driver.latest_diagnostics(self.session)
            phases.update(_driver_host_overlap_ms(d))
            phases.update({'driver_'+name[len('phase.'):-len('_ms')]:milliseconds
                for name,milliseconds in (getattr(d,'host_timings',None)or{}).items()
                if name.startswith('phase.')and name.endswith('_ms')})
            ran=host(d.executed_rows)
            if not np.array_equal(host(d.active_rows),kept)or np.any(ran&~kept)or not np.array_equal(host(result.state.core.frame_number)-old_frames,ran.astype(np.int32)):
                raise RuntimeError('originaldriver must execute exactly one frame per selected physical row')
            executed|=ran;invalid|=kept&(~ran|host(d.invalid_actions))
            self.engine=_row_select(kept,self.engine,result.state.core.engine)
            self.cursor=_row_select(kept,self.cursor,result.state.core.cursor)
            if self.profile is not None:self.profile=_row_select(kept,self.profile,result.state.core.weather_profile)
        fallback=active&~scored
        if fallback.any():
            args=(self.engine,self.cursor,self.profile,j.asarray(actions),chance,j.asarray(fallback))
            if self.diag_executable is None:
                build=time.perf_counter_ns()
                self.diag_executable=self.diag.lower(*args).compile()
                phases['diagnostic_fallback_lower_compile_ms']=(time.perf_counter_ns()-build)/1e6
            start=time.perf_counter_ns()
            state,cursor,profile,diag_callbacks,bad,_,diag_good,diag_namespace=self.jax.block_until_ready(self.diag_executable(*args))
            phases['diagnostic_fallback_engine_ms']=(time.perf_counter_ns()-start)/1e6
            self.engine,self.cursor,self.profile=state,cursor,profile
            invalid|=fallback&host(bad);executed|=fallback&~host(bad)
        witness=getattr(self.session.state.core,'frame_input_report',None)
        ledger=self.session.state.core.execution
        # Only used original/fallback sources are transferred. Row processing
        # below touches NumPy values exclusively; no repeated whole-GPU copy.
        def packet_arrays(l,words,count,mask,risk,refusal=None,namespace=None,missing_history=False):
            return dict(count=host(l.count),site=host(l.site),stage=host(l.stage),ledger_refusal=host(l.refusal),
                words=host(words),raw_count=host(count),mask=None if mask is None else host(mask),risk=host(risk),
                refusal=None if refusal is None else host(refusal),
                namespace=None if mask is None or namespace is None else host(namespace),missing_history=missing_history)
        original=(packet_arrays(ledger,witness.raw_words,witness.raw_count,getattr(witness,'raw_complete_words',None),
            self.session.state.core.risks.risk,refusal=None if getattr(witness,'raw_complete_words',None)is None else witness.refusal)
            if kept.any()else None)
        diagnostic=(packet_arrays(diag_callbacks.execution.ledger,diag_callbacks.words,diag_callbacks.count,diag_good,
            diag_callbacks.execution.risks.carry.risk,refusal=diag_callbacks.refusal,namespace=diag_namespace,
            missing_history=diag_callbacks.execution.leech_history is None)if fallback.any()else None)
        cursor_admitted=host(self.cursor.admitted)if active.any()else None
        self.last_packet=[]
        refused_original=np.zeros(len(active),bool);refused_diagnostic=np.zeros(len(active),bool)
        for row in range(len(active)):
            if not active[row]:
                self.last_packet.append({})
                continue
            use_diag=bool(fallback[row]);data=diagnostic if use_diag else original
            count=int(data['count'][row]);sites=data['site'][row,:count];stages=data['stage'][row,:count]
            words=data['words'][row,:int(data['raw_count'][row])]
            raw_kinds=words[:,0].tolist()if len(words)else[]
            missing_mask=data['mask']is None
            good=np.ones(len(words),bool)if missing_mask else data['mask'][row,:len(words)]
            missing_history=use_diag and any(int(site)in(6,7)for site in sites)and data['missing_history']
            context_unknown=missing_mask or(use_diag and(not bool(data['namespace'][row])or not bool(cursor_admitted[row])or missing_history))or(not use_diag and bool(int(data['refusal'][row])&(source_witness.SOURCE|source_witness.ORDER)))
            unsupported=[int(words[i,0])for i in range(len(words))if not bool(good[i])]
            fresh_risk=int(data['risk'][row])
            refused_row=bool(invalid[row]or context_unknown or np.any(stages==1)or np.any(~good)or fresh_risk!=0)
            (refused_diagnostic if use_diag else refused_original)[row]=refused_row
            self.last_packet.append(dict(unhandled_sites=sites[stages==1].tolist(),raw_unsupported_kinds=unsupported,raw_kinds=raw_kinds,raw_complete_words=good.tolist(),context_unknown=bool(context_unknown),
                raw_words=words.tolist(),selected_move_ids=[int(w[5])for w in words if int(w[0])==1],
                raw_unsupported_moves=[dict(move_id=int(w[5]),name=self.move_names[int(w[5])]if 0<=int(w[5])<len(self.move_names)else'UNKNOWN_MOVE_ID')for i,w in enumerate(words)if int(w[0])==1 and not bool(good[i])],
                fresh_call_risk=fresh_risk,diagnostic_local_ledger=use_diag,
                ledger_refusal=int(data['ledger_refusal'][row]),raw_refusal=int(data['refusal'][row])if use_diag else None,
                history_scope='DIAGNOSTIC_LOCAL_NO_PUBLIC_CONSUMPTION'if use_diag else'ORIGINAL_SCORED_SESSION',
                diagnostic_counts_complete=not(missing_mask or bool(int(data['ledger_refusal'][row])&ex.OVERFLOW)or(use_diag and bool(int(data['refusal'][row])&events.OVERFLOW))),
                diagnostic_context_admitted=bool(cursor_admitted[row]),engine_failures=['INVALID_OR_UNEXECUTED_ACTION']if invalid[row]else[]))
        for capture,mask in ((getattr(witness,'outcome_diagnostics',None),refused_original),
                (None if diag_callbacks is None else diag_callbacks.execution.diagnostics,refused_diagnostic)):
            _attach_refused_diagnostics(self.last_packet,capture,mask,_host_cache=host_cache)
        return executed,invalid,phases

    def capture_failure(self,active):
        done=np.asarray(self.engine.done)
        eligible,failed,reasons=self.retained._admission(self.session.result,active&~done,self.record['config'])
        failed |= active&done&~np.asarray(self.driver.terminal_complete(self.session.state.core))
        return failed,reasons

    def terminal(self,row):
        s=self.engine
        return dict(done=bool(np.asarray(s.done)[row]),winner=int(np.asarray(s.winner)[row]),
            hp=np.asarray(s.hp)[row].tolist(),max_hp=np.asarray(s.max_hp)[row].tolist(),
            stream_id=int(self.streams[row]),turn=int(np.asarray(s.turn)[row]),
            source='ACTUAL_ORIGINAL_NUMERICAL_ENGINE_NOT_A_PUBLIC_TERMINAL_CERTIFICATE')

    def cache_snapshot(self):
        names=('frame','request_fast','installations','unsafe','public_fast','encode_fast','score_fast','frame_witness_fast')
        values={}
        for name in names:
            function=self.static.get(name)
            getter=getattr(function,'_cache_size',None)
            values[name]=None if getter is None else int(getter())
        return values


def _run(backend,plan,streams,*,sampling_seed,clock=time.perf_counter,ledger=None,publish=None):
    """Private fake-test seam; scientific entry constructs ONLY _Runtime above."""
    ledger=Ledger(plan,streams)if ledger is None else ledger;started=clock();samples=[]
    risk_names=('NON_WEATHER','SELF_EFFECT','BOOST','FIELD_VOLATILE','TARGET_OUTCOME','SELF_HP','BLOCKED_ACTION','TERMINAL','CONSTRUCTOR_STATE','NAMESPACE','ORDER','OVERFLOW','UPSTREAM')
    stop='FIXED_STEP_CAP'
    for step in range(plan.max_steps):
        active=ledger.active(step)
        if not active.any():
            stop='NO_LIVE_DECLARED_ROWS';break
        if clock()-started>=plan.max_elapsed_seconds:
            stop='WALL_BACKSTOP_UNKNOWN';break
        began=clock();cache_before=backend.cache_snapshot()
        host_start=time.perf_counter_ns()
        policy_input_capture_valid_rows=int((active&ledger.capture_good).sum())
        policy_work_skipped=step>=plan.timing_warmup+plan.timing_steps and policy_input_capture_valid_rows==0
        if policy_work_skipped:
            backend.check_physical_source()
            # No output can choose an action. Preserve engine/legal/chance
            # work and permanent poisoning; do not fabricate policy tensors.
            policy_actions=None
            model_cost=dict(bf16_physical_owner_rows=0,admitted_policy_owner_rows=0,
                shadow_discarded_owner_rows=0,shadow_actual_encoder_ms=[],shadow_actual_encoder_rows=0,
                shadow_scope='NOT_EXECUTED: fixed timing window ended and no active capture-valid game; encoder/model timings unmeasured.')
        else:
            policy_actions,model_cost=backend.policy(active,ledger,step,sampling_seed)
        model_cost.update(policy_input_capture_valid_rows=policy_input_capture_valid_rows,
            policy_work_skipped=policy_work_skipped,
            policy_work_skip_reason='POST_TIMING_WINDOW_NO_ACTIVE_CAPTURE_VALID_GAME'if policy_work_skipped else None)
        host_walls={'census_host_overlap.policy_wall_ms':(time.perf_counter_ns()-host_start)/1e6}
        host_start=time.perf_counter_ns()
        scored=active&ledger.capture_good
        legal=np.asarray(backend.jax.block_until_ready(backend.mask(backend.engine)))
        requested=backend.requested()
        fallback_actions,no_action=legal_fallback(legal,requested,active&~scored,
            seed=plan.fallback_seed,streams=streams,step=step)
        actions=(fallback_actions if policy_work_skipped else np.where(scored[:,None],policy_actions,fallback_actions)).astype(np.int32)
        host_walls['census_host_overlap.legal_mask_fallback_wall_ms']=(time.perf_counter_ns()-host_start)/1e6
        # The original pure engine sampler runs ONCE from the current actual
        # physical carry. Both disjoint execution paths receive these arrays.
        from . import transition
        host_start=time.perf_counter_ns()
        chance=transition.sample_chance_inputs(backend.engine)
        host_walls['census_host_overlap.chance_sampling_dispatch_wall_ms']=(time.perf_counter_ns()-host_start)/1e6
        host_start=time.perf_counter_ns()
        executed,engine_bad,phases=backend.advance(actions,active&~no_action,scored,chance)
        host_walls['census_host_overlap.advance_wall_ms']=(time.perf_counter_ns()-host_start)/1e6
        host_start=time.perf_counter_ns()
        engine_bad |= no_action
        if not policy_work_skipped:
            capture_failed,reasons=backend.capture_failure(active&scored&executed&~engine_bad)
            ledger.mark_capture(capture_failed,step=step,packets=lambda row:dict(
                kind='actual-capture-or-terminal-closeout-refusal',owner_reason=reasons[row].tolist(),
                actual=backend.last_packet[row]))
        for row in np.flatnonzero(no_action):
            backend.last_packet[row]=dict(unhandled_sites=[],raw_kinds=[],raw_unsupported_kinds=[],engine_failures=['NO_ORIGINAL_LEGAL_ACTION'])
        for row in np.flatnonzero(active):
            p=backend.last_packet[row];bits=int(p.get('fresh_call_risk',0))
            # Fresh diagnostic risk excludes previous-frame sticky values. In
            # the scored path the first failing frame is still identified as
            # original sticky risk; never confuse UPSTREAM with a causal kind.
            p['capture_risk_kinds']=[name for bit,name in enumerate(risk_names)if bits&(1<<bit)]
            p['selected_native_actions']=actions[row].tolist()
            p['action_source']='ADMITTED_CHAMPION_POLICY'if scored[row]else'SEEDED_UNIFORM_ENGINE_LEGAL_DIAGNOSTIC'
            p['fallback_seed']=plan.fallback_seed if not scored[row]else None
        ledger.observe(step=step,active=active,executed=executed,engine_failed=engine_bad,
            done=np.asarray(backend.engine.done),turns=np.asarray(backend.engine.turn),
            packet=lambda row:backend.last_packet[row],terminal=backend.terminal)
        host_walls['census_host_overlap.accounting_wall_ms']=(time.perf_counter_ns()-host_start)/1e6
        phases.update(host_walls)
        cache_after=backend.cache_snapshot()
        deltas={name:None if value is None or cache_before.get(name)is None else value-cache_before[name]
            for name,value in cache_after.items()}
        observed=all(v is not None for v in deltas.values())
        new_signature=any(v is not None and v>0 for v in deltas.values())
        explicit_compile=model_cost.get('bf16_compile_ms',0)>0 or phases.get('diagnostic_fallback_lower_compile_ms',0)>0
        warm=not policy_work_skipped and step>=plan.timing_warmup and not new_signature and not explicit_compile
        samples.append(dict(step=step,launch_shape=plan.batch,active_engine_rows=int(active.sum()),
            census_active_rows=int(active[:plan.census_games].sum()),timing_only_active_rows=int(active[plan.census_games:].sum()),
            elapsed_ms=(clock()-began)*1000,phases_ms=phases,**model_cost,cache_signature_delta=deltas,
            warm_candidate=warm,warm_cache_observability_complete=observed,
            timing='Explicit lower+compile where reported; other firstcalls/caches are coldcall-inclusive, never compiler-only time.',
            fallback_seed=plan.fallback_seed,fallback_actions_histogram=[np.bincount(actions[active&~scored,owner],minlength=9).tolist()for owner in(0,1)]))
        if publish is not None and (step%16==0 or not ledger.active(step+1).any()):
            publication=clock();publish(ledger.summary());samples[-1]['journal_publication_ms']=(clock()-publication)*1000
        samples[-1]['inclusive_step_wall_ms']=(clock()-began)*1000
        samples[-1]['elapsed_ms_scope']='Synchronized step including diagnostic capture; journalpublication separately reported.'
    report=ledger.summary()
    report.update(stop=stop,plan=plan.__dict__,elapsed_seconds=clock()-started,samples=samples,
        compile_scope='BF16 and diagnosticfallback lowering+compile explicit; original driver phase coldcall-inclusive and cache deltas recorded.',
        host_timing_scope='driver_host_overlap.* copies existing driver host spans/advance wall. census_host_overlap.* are thin caller host walls; policy/legal/advance contain ONLY their existing synchronization, chance_sampling_dispatch is unsynchronized dispatch, accounting may include existing host copies. All overlap existing synchronized phases and are NOT summable as disjoint costs. Missing driver spans mean unmeasured. No added barriers, JIT, engine/model or RNG calls.',
        unsupported_by_source_scope='Occurrences summed per role/source/kind; games/rows affected are UNION identities across original capture/capture-context strata. Original stratified table retained; no exclusive-cause attribution.',
        full_policy_game_rate=None,production_fullgame_throughput=None,
        diagnostic_engine_terminal_rate=report['census']['engine_terminal_games']/max(1e-12,clock()-started),
        accounting='512 fixedcensusgames plus1536timing-onlywindows; no refills; postgap rows forever unscored. Caps/failed advancement/missingterminals remain explicitunknown.',
        observation_trigger_25_percent=None,
        observation_trigger_status='UNMEASURED_TOTAL_PRODUCTION_ATTRIBUTION; inspect real perphase warm2048shape/counts without claiming fullpolicy throughput.')
    warm=[s for s in samples if s['warm_candidate']and s['step']<plan.timing_warmup+plan.timing_steps]
    observation=lambda s:sum(s.get('shadow_actual_encoder_ms',[]))+sum(v for n,v in s['phases_ms'].items()if n.startswith(('driver_public','driver_encode','driver_admission')))
    report['warm_launch_shape']=dict(rows=plan.batch,samples=len(warm),active_engine_rows=[s['active_engine_rows']for s in warm],
        original_request_rows=[s['phases_ms'].get('original_request_rows',0)for s in warm],
        bf16_physical_owner_rows=[s.get('bf16_physical_owner_rows',0)for s in warm],
        admitted_policy_owner_rows=[s.get('admitted_policy_owner_rows',0)for s in warm],
        observed_encoder_ms=[observation(s)for s in warm],
        partial_diagnostic_observation_wall_fraction=sum(map(observation,warm))/sum(s['elapsed_ms']for s in warm)if warm else None,
        request_status='UNMEASURED_WHEN_ORIGINAL_REQUEST_ROWS_ZERO; no context/admission bypass',
        scope='Actual2048shape/4096owner graph calls; active/admitted counts separate. Shadow work discarded. Diagnostic duplicatedencoder/partitionedengine overhead is not production throughput.')
    return report


def measure(runtime,pool,team_indices,*,stream_ids,plan,constructor_seed,sampling_seed,
            gender_initialization=None,enable_diagnostic=False,journal_path=None,max_journal_bytes=64*1024*1024):
    """Explicit source-only callable; no loading/launch/default CLI side effect."""
    if enable_diagnostic is not True or type(plan)is not Plan:
        raise ValueError('explicit finite diagnostic plan required')
    plan.validate();indices=np.array(team_indices,copy=True);streams=np.array(stream_ids,copy=True)
    if indices.shape!=(plan.batch,2)or indices.dtype.kind not in'iu'or np.any(indices<0)or len(np.unique(indices))!=indices.size:
        raise ValueError('all4096freshdistinct declared teamindices required')
    if type(constructor_seed)is not int or type(sampling_seed)is not int or not 0<=constructor_seed<2**31 or not 0<=sampling_seed<2**31:
        raise ValueError('explicit original constructor/policy seeds required')
    from . import source_provenance,benchmark_retained_games as retained
    source=retained._source_snapshot();receipt=source_provenance._pool(pool,rehash=True)
    if np.any(indices>=len(pool.arrays['species'])):raise ValueError('declared teamindex outside bound pool')
    if journal_path is None:raise ValueError('exclusive remote census journal path required')
    move_names=_bound_engine_move_names(retained.driver._checked(retained._checked(runtime)['prepared'])['request'])
    ledger=Ledger(plan,streams);journal=Journal(journal_path,max_bytes=max_journal_bytes)
    envelope=dict(source=source,runtime_binding=runtime.binding,pool_receipt=receipt,plan=plan.__dict__,
        constructor_seed=constructor_seed,sampling_seed=sampling_seed,
        original_team_indices=indices.tolist(),original_stream_ids=streams.tolist(),
        engine_move_names=list(move_names),
        attempts='All2048declaredfreshconstructorrows are attempted before dispatch; no retry/refill.')
    journal.write(dict(status='ATTEMPTED_CONSTRUCTION_UNKNOWN',**envelope,**ledger.summary()))
    try:
        backend=_Runtime(runtime,plan,pool,indices,streams,constructor_seed,gender_initialization)
        report=_run(backend,plan,streams,sampling_seed=sampling_seed,ledger=ledger,
            publish=lambda partial:journal.write(dict(status='RUNNING_DIAGNOSTIC',**envelope,**partial)))
        source_provenance._pool(pool,rehash=True);retained._checked(runtime,session=backend.session)
        if retained._source_snapshot()!=source:raise RuntimeError('census source changed during measurement')
        report.update(envelope,construction_ms=backend.construction_ms,status='COMPLETE_DIAGNOSTIC_PROGRAM',actual_exit=0)
        journal.write(report)
        return report
    except BaseException as error:
        journal.write(dict(status='FAILED_DIAGNOSTIC_PROGRAM',actual_exit=1,**envelope,**ledger.summary(),
            failure=dict(error_type=type(error).__name__,message=str(error)[:4096])))
        raise


class Journal:
    """Owned bounded atomic ledger, intended for the declared remote output.

    The original2048attempts are published before construction. Unexpected
    process termination leaves those cells explicitly attempted/unknown; no
    result status can manufacture a terminal or authorize a retry/refill.
    """
    def __init__(self,path,*,max_bytes=64*1024*1024):
        from pathlib import Path
        if type(max_bytes)is not int or not 1024<=max_bytes<=128*1024*1024:
            raise ValueError('bounded journal cap required')
        self.path=Path(path);self.max_bytes=max_bytes
        with self.path.open('x')as stream:stream.write('{}\n')
    def write(self,document):
        import json,os,tempfile
        raw=json.dumps(dict(updated_epoch=time.time(),**document),sort_keys=True,separators=(',',':'),allow_nan=False).encode()+b'\n'
        if len(raw)>self.max_bytes:raise ValueError('census journal cap exceeded; no truncated scientific ledger')
        fd,tmp=tempfile.mkstemp(prefix='.'+self.path.name,dir=self.path.parent)
        try:
            with os.fdopen(fd,'wb')as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
            os.replace(tmp,self.path)
        finally:
            if os.path.exists(tmp):os.unlink(tmp)
