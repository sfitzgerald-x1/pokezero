"""P2 native one-ply transport, lazy quarantine and fixed-shape inference.

No action sampling, probability approximation, leaf deduplication, or changed
observation law. The generic exact expectation and type-contingent LP still
own scientific values. Native sessions retain immutable roots and addressed
chance instructions; full evidence is reconstructed only on failure.
"""
from collections import Counter
from dataclasses import dataclass
import json
import math
import os
import time

from .failure_capture import current
from .shallow_exact import ChanceBranch, ExactSearchRefused
from .shallow_exact_native import _branch_action_token, _drain_model_transport


def configure_p2_arithmetic():
    """Freeze explicit FP32 arithmetic before loading/warming P2's model.

    CUDA's high/TF32 default produced a 4.3e-5 real-value difference between
    variable and padded batches on a saved input. Strict FP32 passed the same
    predeclared tolerances. This is P2 startup only, never an import-time or
    ordinary-search default change. The prospective entry must bind it.
    """
    from .neural_policy import require_torch
    torch=require_torch()
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    contract=dict(name='strict_fp32',float32_matmul_precision=torch.get_float32_matmul_precision(),
        matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32)
    if (contract['float32_matmul_precision']!='highest'
            or contract['matmul_allow_tf32'] or contract['cudnn_allow_tf32']):
        raise ExactSearchRefused('P2 strict FP32 arithmetic could not be applied')
    return contract


class FixedShapeModel:
    """Pad with deterministic dummy observations; discard all padded outputs.

    Warm exactly three times at process/policy startup, before any decision or
    RNG use. Every measured model call includes explicit pre/post barriers.
    Zero categorical/numeric/type values and true attention/history are valid
    dummy rows; they never contribute to a chance expectation or action mix.
    """
    def __init__(self, model, spec, batch_size):
        if (type(batch_size) is not int or not 1 <= batch_size <= 4096
                or set(spec) != {'window','tokens','categorical_features','numeric_features'}
                or any(type(v) is not int or v <= 0 for v in spec.values())
                or spec['window'] != 1):
            raise ExactSearchRefused('fixed model requires bounded batch and complete one-window shape')
        if not all(callable(getattr(model,n,None)) for n in ('synchronize','eval_obs_packed')):
            raise ExactSearchRefused('fixed model requires native synchronization and packed forward APIs')
        self.model, self.batch_size = model, batch_size
        wt=spec['window']*spec['tokens']
        self.dummy=(bytes(wt*spec['categorical_features']*4),
                    bytes(wt*spec['numeric_features']*8),bytes(wt*2),
                    b'\x01'*wt,b'\x01'*spec['window'])
        self.last_call=None
        self.warmup=[]
        for _ in range(3):
            self.eval_obs_packed(0,*(b'' for _ in self.dummy),byte_order='little')
            self.warmup.append(dict(self.last_call))
        _drain_model_transport(self.model)

    def eval_obs_packed(self, count, *columns, byte_order):
        if (type(count) is not int or not 0 <= count <= self.batch_size
                or byte_order != 'little' or len(columns)!=5
                or any(type(c) is not bytes or len(c)!=len(d)*count
                       for c,d in zip(columns,self.dummy))):
            raise ExactSearchRefused('fixed model packed input shape/type mismatch')
        padded=tuple(c+d*(self.batch_size-count) for c,d in zip(columns,self.dummy))
        started=time.perf_counter();returned=False;synchronized=False
        self.last_call=dict(real_rows=count,model_rows=self.batch_size,
            padded_rows=self.batch_size-count,padding='zero_features_true_attention_history',
            returned=False,model_invoked=False)
        try:
            self.model.synchronize()
            try:
                self.last_call['model_invoked']=True
                outputs=self.model.eval_obs_packed(self.batch_size,*padded,byte_order='little')
            except BaseException:
                try:
                    self.model.synchronize()
                    synchronized=True
                except BaseException as sync_error:
                    # Keep the forward's original exception and native witness.
                    self.last_call['synchronization_error']=dict(
                        type=type(sync_error).__name__,message=str(sync_error))
                raise
            else:
                self.model.synchronize()
                synchronized=True
            values,logits,priors,actions=outputs
            if (type(actions) is not int or actions<=0 or len(values)!=self.batch_size
                    or len(logits)!=actions*self.batch_size or len(priors)!=actions*self.batch_size):
                raise ExactSearchRefused('fixed model output cardinality changed')
            if any(not math.isfinite(float(v)) or not -1 <= v <= 1 for v in values):
                raise ExactSearchRefused('fixed model returned invalid real/dummy value')
            returned=True
            return values[:count],logits[:count*actions],priors[:count*actions],actions
        finally:
            elapsed=time.perf_counter()-started
            self.last_call.update(returned=returned,wrapper_wall_seconds=elapsed,
                timing_status='SYNCHRONIZED' if synchronized else 'UNAVAILABLE_SYNCHRONIZATION',
                synchronized_wrapper_seconds=elapsed if synchronized else None)

    def drain_transport_timings(self):
        return self.model.drain_transport_timings()


@dataclass(frozen=True)
class AddressedLeaf:
    session: object
    world: int
    cell: int
    branch: int
    terminal: float | None
    self_side: str


class NativeSessionBackend:
    """ExactBackend using branch handles instead of Python leaf snapshots."""
    def __init__(self,native,model,*,packed_leaves,bulk_encode_workers,
                 max_outcomes_per_call,observation_spec):
        if os.environ.get('POKEZERO_ENUMERATE_ROLLS')!='1':
            raise ExactSearchRefused('native session requires fresh-process exact damage rolls')
        if (not packed_leaves or type(bulk_encode_workers) is not int
                or not 1<=bulk_encode_workers<=32
                or not callable(getattr(native,'ExactSession',None))):
            raise ExactSearchRefused('native session API and explicit packed workers required')
        self.native,self.model=native,model
        self.workers=bulk_encode_workers
        self.max_branches=max_outcomes_per_call
        self.spec=observation_spec
        self.sessions={}
        self.stage_seconds=Counter();self.render_refusals=Counter();self.render_subcases=Counter()
        self.model_call_timings=[]
        self.encode_calls=self.forward_batches=self.unique_leaf_evaluations=0
        self.capture_context_updates=self.captured_leaf_rows=0
        self._last_position=None;self._last_batch=None
        self._in_flight_cell=None
        self._pending=None;self._batch_stage='pending';self._current_model_call=None
        self._encode_group=None
        self._capture_registered=False

    def prepare_roots(self,positions):
        if self.sessions:raise ExactSearchRefused('native roots already frozen')
        roots=[]
        for ordinal,p in enumerate(positions):
            if p.lines or p.self_side not in ('side_one','side_two'):
                raise ExactSearchRefused('native one-ply session requires original root histories and known frame')
            context={**p.party_species,'turn':p.turn}
            # Match NativeExactBackend.branch_events exactly. The encoder's
            # separate root context retains its own metadata attestations.
            roots.append(dict(world_ordinal=ordinal,engine_state=p.state_str,turn=p.turn,
                event_context=context,fold_payload=p.fold.to_payload(),origin=p.origin))
        capture=current()
        if capture:
            capture.update(exact_roots=roots)
            capture.defer('native_one_ply',self._failure_evidence)
            self._capture_registered=True
        for ordinal,(p,root) in enumerate(zip(positions,roots)):
            self._last_position=ordinal
            session=self.native.ExactSession(p.state_str,p.fold,json.dumps(root['event_context']),p.turn,
                self.max_branches,81,p.encoder)
            self.sessions[id(p)]=(ordinal,p,session)

    def _session(self,p):
        row=self.sessions.get(id(p))
        if row is None or row[1] is not p:
            raise ExactSearchRefused('position not in frozen native root set')
        return row

    def _failure_evidence(self):
        def address(p):
            return dict(world_ordinal=p.world,cell_index=p.cell,branch_index=p.branch,
                native=json.loads(p.session.reconstruct(p.cell,p.branch)))
        pending=self._pending if self._pending is not None else self._last_batch
        if pending:
            # A work cap can reject the next leaf before it joins pending.
            # Preserve that address separately; never substitute it for any
            # ordered queued/model rows. Reconstruction remains failure-only.
            last_checked=(address(self._last_position)
                if isinstance(self._last_position,AddressedLeaf) else None)
            return dict(kind='ordered_model_batch' if self._batch_stage!='pending' else 'ordered_pending_batch',
                stage=self._batch_stage,rows=[address(p) for p in pending],
                last_addressed_leaf=last_checked,
                real_rows=len(pending),model_rows=self.model.batch_size,
                padded_rows=self.model.batch_size-len(pending),
                padding='zero_features_true_attention_history',
                observation_spec=self.spec,model_call=self._current_model_call,
                model_invoked=self._current_model_call is not None and self._current_model_call['model_invoked'],
                encode_group=self._encode_group,in_flight_cell=self._in_flight_cell)
        if isinstance(self._last_position,AddressedLeaf):
            return dict(kind='branch',**address(self._last_position),in_flight_cell=self._in_flight_cell)
        return dict(kind='root_or_transition',root_ordinal=self._last_position,
            in_flight_cell=self._in_flight_cell)

    def begin_decision(self):
        # prepare_roots already froze this decision; never reset sessions or RNG.
        self.stage_seconds.clear();self.model_call_timings.clear()

    def bind_pending(self,pending):
        self._pending=pending

    def _pending_stage(self):
        self._batch_stage='pending';self._current_model_call=None;self._encode_group=None

    def terminal_value(self,p):
        self._pending_stage()
        self._last_batch=None;self._last_position=p if isinstance(p,AddressedLeaf) else self._session(p)[0]
        return p.terminal if isinstance(p,AddressedLeaf) else self._session(p)[2].root_terminal()

    def actions(self,p,*,root):
        if not root:raise ExactSearchRefused('native session exposes only root actions')
        self._last_position=self._session(p)[0]
        return tuple(tuple(a) for a in self._session(p)[2].root_actions())

    def outcomes(self,p,row_action,column_action):
        self._pending_stage()
        ordinal,_,session=self._session(p)
        self._last_batch=None;self._last_position=ordinal
        self._in_flight_cell=dict(world_ordinal=ordinal,row_action=row_action,column_action=column_action)
        started=time.perf_counter()
        try:
            cell,branches,subcases=session.enumerate_cell(_branch_action_token(row_action),
                _branch_action_token(column_action))
        finally:self.stage_seconds['native_transition_render']+=time.perf_counter()-started
        self.render_subcases.update(subcases)
        self._in_flight_cell.update(cell_index=cell,branch_count=len(branches))
        for index,(probability,terminal) in enumerate(branches):
            leaf=AddressedLeaf(session,ordinal,cell,index,terminal,p.self_side)
            self._last_position=leaf
            yield ChanceBranch(leaf,probability)

    def evaluate_leaves(self,states):
        if not states:return []
        self._last_batch=states  # Borrow the core's active ordered list; NO capture copy.
        self._batch_stage='encoding';self._current_model_call=None;self._encode_group=None
        started=time.perf_counter();columns=[[],[],[],[]];offset=0
        names=('categorical_ids','numeric_features','token_type_ids','attention_mask')
        wt=self.spec['window']*self.spec['tokens']
        widths=(wt*self.spec['categorical_features']*4,wt*self.spec['numeric_features']*8,wt*2,wt)
        while offset<len(states):
            session=states[offset].session;stop=offset+1
            while stop<len(states) and states[stop].session is session:stop+=1
            group=states[offset:stop]
            self._encode_group=dict(start=offset,stop=stop,native_batch_row_basis='relative_to_group_start')
            encoded=session.encode_packed([(p.cell,p.branch) for p in group],self.workers)
            if type(encoded.get('batch_rows')) is not int or encoded['batch_rows']!=len(group):
                raise ExactSearchRefused('native session encoder changed row count')
            for name,width,out in zip(names,widths,columns):
                value=encoded.get(name)
                if type(value) is not bytes or len(value)!=width*len(group):
                    raise ExactSearchRefused('native session packed width/type changed: '+name)
                out.append(value)
            offset=stop
        packed=tuple(b''.join(c) for c in columns)+(b'\x01'*(len(states)*self.spec['window']),)
        self.encode_calls+=len(states)
        self.stage_seconds['native_encode_and_pack']+=time.perf_counter()-started
        prior=_drain_model_transport(self.model);started=time.perf_counter();returned=False
        self._batch_stage='model_forward';self._encode_group=None
        try:
            values,_,_,_=self.model.eval_obs_packed(len(states),*packed,byte_order='little')
            returned=True
        finally:
            self._current_model_call=self.model.last_call
            elapsed=time.perf_counter()-started
            self.stage_seconds['synchronized_model_wrapper']+=elapsed
            self.model_call_timings.append(dict(batch_rows=len(states),returned=returned,
                wrapper_wall_seconds=elapsed,fixed_shape=self.model.last_call,
                transport=_drain_model_transport(self.model) if prior['status']=='MEASURED' else prior))
        if len(values)!=len(states):raise ExactSearchRefused('model leaf cardinality changed')
        self._batch_stage='evaluated'
        self.forward_batches+=1;self.unique_leaf_evaluations+=len(states)
        return [float(v)*(1 if p.self_side=='side_one' else -1) for p,v in zip(states,values)]
