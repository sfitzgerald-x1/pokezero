"""Streaming per-column comparisons for recorded v4 decision pairs.

This reports numerical evidence on supplied pairs. Recorded-event provenance,
checkpoint identity and qualification of the producer remain separate audited
requirements; synthetic tests never count as recorded parity decisions.
"""
import numpy as np


class ColumnParity:
    def __init__(self,*,reference_source_sha256,encoder_source_sha256,vocabulary_sha256):
        self.bindings={}
        for name,value in dict(reference_source_sha256=reference_source_sha256,
                encoder_source_sha256=encoder_source_sha256,vocabulary_sha256=vocabulary_sha256).items():
            if not isinstance(value,str) or len(value)!=64 or any(c not in '0123456789abcdef' for c in value):
                raise ValueError('exact source/vocabulary SHA256 required')
            self.bindings[name]=value
        self.ids=set();self.roots=set();self.battles=set();self.decisions=0
        self.numeric_bad=np.zeros((23,132),np.int64);self.category_bad=np.zeros((23,41),np.int64)
        self.numeric_max=np.zeros((23,132),np.float64)
        self.structure_bad=dict(token_type_ids=0,attention_mask=0,legal_action_mask=0,switch_targets=0)
        self.forward_count=0;self.argmax_agree=0;self.value_max=0.

    @staticmethod
    def _array(record,name,shape,kind):
        value=np.asarray(record[name])
        if value.shape!=shape or value.dtype.kind not in kind:raise ValueError(name+' shape/dtype mismatch')
        if value.dtype.kind in 'fiu' and not np.isfinite(value).all():raise ValueError(name+' must be finite')
        return value

    def add(self,*,decision_id,battle_id,seat,turn,prefix_sha256,reference,candidate):
        if not isinstance(decision_id,str) or not decision_id or decision_id in self.ids:
            raise ValueError('unique nonempty recorded decision identity required')
        if not isinstance(battle_id,str) or not battle_id or seat not in ('p1','p2') or type(turn) is not int or turn<0:
            raise ValueError('battle/seat/turn provenance required')
        if not isinstance(prefix_sha256,str) or len(prefix_sha256)!=64 or any(c not in '0123456789abcdef' for c in prefix_sha256):
            raise ValueError('raw prefix SHA256 required')
        root=(battle_id,seat,turn,prefix_sha256)
        if root in self.roots:raise ValueError('duplicate canonical recorded decision, even with a new identity alias')
        numeric=[];category=[];structure=[]
        shapes={'token_type_ids':((23,),'iu'),'attention_mask':((23,),'b'),
            'legal_action_mask':((9,),'b'),'switch_targets':((5,),'iu')}
        for record in (reference,candidate):
            if record.get('decision_id')!=decision_id or record.get('prefix_sha256')!=prefix_sha256:
                raise ValueError('both encodings must bind the identical recorded decision/prefix')
            numeric.append(self._array(record,'numeric_features',(23,132),'fiu').astype(np.float64))
            cats=self._array(record,'categorical_ids',(23,41),'iu')
            if np.any(cats<0):raise ValueError('negative category row')
            category.append(cats)
            structure.append({name:self._array(record,name,*spec) for name,spec in shapes.items()})
        # Validate a paired forward in full before mutating accumulated counts.
        has_forward=('policy_logits' in reference) or ('policy_logits' in candidate)
        forward=None
        if has_forward:
            logits=[self._array(r,'policy_logits',(9,),'fiu') for r in (reference,candidate)]
            values=[]
            for record in (reference,candidate):
                value=record.get('value')
                if type(value) not in (int,float) or not np.isfinite(value):raise ValueError('paired finite forward value required')
                values.append(value)
            forward=(int(np.argmax(logits[0])==np.argmax(logits[1])),abs(values[0]-values[1]))
        delta=np.abs(numeric[0]-numeric[1])
        self.numeric_bad+=delta>1e-6;self.category_bad+=category[0]!=category[1]
        self.numeric_max=np.maximum(self.numeric_max,delta)
        for name in shapes:self.structure_bad[name]+=int(not np.array_equal(structure[0][name],structure[1][name]))
        if forward is not None:
            self.forward_count+=1;self.argmax_agree+=forward[0];self.value_max=max(self.value_max,forward[1])
        self.ids.add(decision_id);self.roots.add(root);self.battles.add(battle_id);self.decisions+=1

    def report(self):
        columns=[]
        for kind,width,bad in (('numeric',132,self.numeric_bad),('categorical',41,self.category_bad)):
            for index in range(width):
                row=dict(kind=kind,column=index,compared_decisions=self.decisions,
                    compared_cells=self.decisions*23,mismatched_cells=int(bad[:,index].sum()),
                    token_mismatches=bad[:,index].tolist())
                if kind=='numeric':row.update(tolerance=1e-6,max_absolute_difference=float(self.numeric_max[:,index].max()))
                else:row['tolerance']=0
                columns.append(row)
        enough=self.decisions>=10000
        numeric_ok=not self.numeric_bad.any();category_ok=not self.category_bad.any()
        structure_ok=not any(self.structure_bad.values())
        ratio=self.argmax_agree/self.forward_count if self.forward_count else None
        return dict(schema='pokezero.v4-column-parity.v1',bindings=dict(self.bindings),
            decisions=self.decisions,battles=len(self.battles),columns=columns,
            structure_mismatched_decisions=dict(self.structure_bad),minimum_record_count_met=enough,
            supplied_tensor_parity_pass=bool(enough and numeric_ok and category_ok and structure_ok),
            forward_decisions=self.forward_count,policy_argmax_agreement=ratio,max_value_difference=self.value_max if self.forward_count else None,
            supplied_forward_gate_pass=bool(enough and numeric_ok and category_ok and structure_ok and self.forward_count==self.decisions and ratio is not None and ratio>=.999 and self.value_max<1e-4),
            recorded_provenance_verified=False,
            qualification='NUMERICAL_COMPARISON_ONLY_RECORDED_PROVENANCE_AND_PRODUCER_QUALIFICATION_REQUIRE_SEPARATE_EVIDENCE')
