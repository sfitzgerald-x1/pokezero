"""Streaming comparison metrics, separate from corpus/runtime admission.

The caller must attest archive bytes, completeness, actual JAX execution and
the frozen champion graph. This reducer cannot qualify player visibility or
the full encoder. It never selects decisions or ignores padding cells.
"""
import json
import re
import numpy as np

from .recorded_encoder_corpus import EncoderView,VISIBILITY
from .recorded_mirror_roots import CHAMPION_CONTRACT

SHAPES={'categorical_ids':(23,41),'numeric_features':(23,132),'token_type_ids':(23,),
        'attention_mask':(23,),'legal_action_mask':(9,)}


def _arrays(value,*,device):
    if not isinstance(value,dict) or set(value)!=set(SHAPES):raise ValueError('exact five reference tensors required')
    result={}
    for name,shape in SHAPES.items():
        a=np.asarray(value[name])
        kind='b' if name.endswith('mask') else 'iu' if name!='numeric_features' else 'iuf'
        if a.shape!=shape or a.dtype.kind not in kind or not np.isfinite(a).all():
            raise ValueError('invalid observation tensor: '+name)
        if device:
            dtype=np.float32 if name=='numeric_features' else np.bool_ if kind=='b' else np.int32
            if a.dtype!=dtype:raise ValueError('exact JAX output dtype required: '+name)
        if name=='categorical_ids' and ((a<0).any() or (a>=CHAMPION_CONTRACT['categorical_vocab_size']).any()):
            raise ValueError('category ID outside frozen champion vocabulary')
        if name=='token_type_ids' and ((a<0).any() or (a>=CHAMPION_CONTRACT['token_type_vocab_size']).any()):
            raise ValueError('token type outside frozen champion vocabulary')
        result[name]=a
    if not result['attention_mask'].any() or not result['legal_action_mask'].any():
        raise ValueError('requested observation must attend tokens and admit an action')
    return result


def _forward(value):
    if not isinstance(value,dict) or set(value)!={'policy_logits','value'}:raise ValueError('paired FP32 outputs required')
    logits=np.asarray(value['policy_logits']);v=np.asarray(value['value'])
    if logits.shape!=(9,) or v.shape!=() or logits.dtype!=np.float32 or v.dtype!=np.float32:
        raise ValueError('FP32 nine logits and scalar value required')
    if not np.isfinite(logits).all() or not np.isfinite(v):raise ValueError('finite model output required')
    return logits,v


class ParityTally:
    """All-cell comparisons, with atomic refusal and no duplicated view IDs."""
    def __init__(self,*,expected_views):
        if type(expected_views) is not int or expected_views<=0:raise ValueError('fixed expected view count required')
        self.expected_views=expected_views;self.ids=set();self.roles=set();self.rounds={};self.latest={};self.rows=0
        self.numeric_bad=np.zeros(132,np.int64);self.numeric_max=np.zeros(132,np.float64)
        self.categorical_bad=np.zeros(41,np.int64)
        self.other_bad={name:0 for name in ('token_type_ids','attention_mask','legal_action_mask')}
        self.forward_rows=0;self.argmax_equal=0;self.legal_argmax_equal=0;self.value_max=0.

    def add(self,view,actual,*,actual_forward=None,reference_forward=None):
        if (not isinstance(view,EncoderView) or view.visibility!=VISIBILITY or view.public_wire_qualified
                or not isinstance(view.decision_id,str) or not re.fullmatch('[0-9a-f]{64}',view.decision_id)
                or view.decision_id in self.ids or self.rows>=self.expected_views
                or not isinstance(view.battle_id,str) or not view.battle_id.strip()
                or '\n' in view.battle_id or '\r' in view.battle_id or view.seat not in ('p1','p2')
                or type(view.decision_round_index) is not int or view.decision_round_index<0
                or type(view.turn_number) is not int or view.turn_number<0
                or not isinstance(view.source_prefix_sha256,str)
                or not re.fullmatch('[0-9a-f]{64}',view.source_prefix_sha256)):
            raise ValueError('unique valid trusted-source owning view required')
        role=(view.battle_id,view.seat,view.decision_round_index)
        root=(view.battle_id,view.decision_round_index)
        signature=(view.source_prefix_sha256,view.turn_number)
        owner=(view.battle_id,view.seat)
        if role in self.roles or view.decision_round_index<=self.latest.get(owner,-1):
            raise ValueError('duplicate or out-of-order canonical owning view')
        if root in self.rounds and self.rounds[root]!=signature:raise ValueError('paired root metadata mismatch')
        request=json.loads(view.own_request_json)
        if not isinstance(request,dict) or not isinstance(request.get('side'),dict) or request['side'].get('id')!=view.seat:
            raise ValueError('own request seat differs from view')
        reference=json.loads(view.own_reference_json)
        if not isinstance(reference,dict) or type(reference.get('turn_number')) is not int or reference['turn_number']!=view.turn_number:
            raise ValueError('reference turn differs from owning view')
        # Metadata is not an observation column, and cannot be fed to a model.
        reference={name:reference[name] for name in SHAPES}
        ref=_arrays(reference,device=False);got=_arrays(actual,device=True)
        difference=np.abs(got['numeric_features'].astype(np.float64)-ref['numeric_features'].astype(np.float64))
        numeric_bad=np.sum(difference>1e-6,axis=0);numeric_max=np.max(difference,axis=0)
        categorical_bad=np.sum(got['categorical_ids']!=ref['categorical_ids'],axis=0)
        other_bad={name:int(np.count_nonzero(got[name]!=ref[name])) for name in self.other_bad}
        if (actual_forward is None)!=(reference_forward is None):raise ValueError('both or neither model outputs required')
        forward=None
        if actual_forward is not None:
            a,av=_forward(actual_forward);r,rv=_forward(reference_forward)
            # Each graph uses its own encoded legality, even when that differs.
            ai=int(np.argmax(np.where(got['legal_action_mask'],a,-np.inf)))
            ri=int(np.argmax(np.where(ref['legal_action_mask'],r,-np.inf)))
            forward=(int(np.argmax(a))==int(np.argmax(r)),ai==ri,float(abs(np.float64(av)-np.float64(rv))))
        # No accumulator advances until every source/output check above passes.
        self.ids.add(view.decision_id);self.roles.add(role);self.rounds[root]=signature
        self.latest[owner]=view.decision_round_index;self.rows+=1
        self.numeric_bad+=numeric_bad;self.numeric_max=np.maximum(self.numeric_max,numeric_max)
        self.categorical_bad+=categorical_bad
        for name,value in other_bad.items():self.other_bad[name]+=value
        if forward is not None:
            self.forward_rows+=1;self.argmax_equal+=forward[0];self.legal_argmax_equal+=forward[1]
            self.value_max=max(self.value_max,forward[2])

    def report(self):
        complete=self.rows==self.expected_views
        inputs_equal=not (self.numeric_bad.any() or self.categorical_bad.any() or any(self.other_bad.values()))
        forward_complete=self.forward_rows==self.rows and self.rows>0
        enough=self.rows>=10000;agreement=self.argmax_equal/self.forward_rows if self.forward_rows else None
        legal_agreement=self.legal_argmax_equal/self.forward_rows if self.forward_rows else None
        return dict(schema='pokezero.B2-recorded-comparison-metrics.v1',visibility=VISIBILITY,
            status='METRICS_ONLY_REQUIRES_CALLER_ATTESTATION',expected_views=self.expected_views,views=self.rows,
            complete_view_count=complete,minimum10000_views_met=enough,
            numeric_columns=[dict(index=i,cells=self.rows*23,mismatched_cells=int(self.numeric_bad[i]),
                maximum_absolute_error=float(self.numeric_max[i]),tolerance=1e-6) for i in range(132)],
            categorical_columns=[dict(index=i,cells=self.rows*23,mismatched_cells=int(self.categorical_bad[i]),
                comparison='EXACT') for i in range(41)],other_tensor_mismatched_cells=dict(self.other_bad),
            recorded_input_numeric_criteria_met=complete and enough and inputs_equal,
            model_forward_views=self.forward_rows,model_forward_complete=forward_complete,
            policy_argmax_agreement=agreement,legal_policy_argmax_agreement=legal_agreement,value_tolerance=1e-4,
            measured_value_criterion_met=self.value_max<1e-4 if self.forward_rows else None,maximum_value_difference=self.value_max if self.forward_rows else None,
            recorded_forward_numeric_criteria_met=complete and enough and forward_complete and agreement>=.999 and legal_agreement>=.999 and self.value_max<1e-4,
            public_wire_qualified=False,full_encoder_admitted=False)
