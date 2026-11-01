"""B2 owning protocol reference for offline public29 agreement comparison.

One factory caches original fold/encode/append JITs; every game has fresh
owning public state. This is diagnostic reference work, never a device selfplay
dependency. No engine events are reconstructed, no model/weights are loaded,
and no unsupported boundary aborts the caller's finite game cohort.
"""
import hashlib
import re


class _RetainEncoded:
    """Retain the SAME B2 encode result for the existing public29 appender."""
    def __init__(self, functions):
        self.functions=functions;self.catalog=functions.catalog;self.step=functions.step
        self.captured=None;self.calls=0
        def encode(state, own, side):
            result=functions.encode_step(state,own,side)
            self.captured=(state,result,side);self.calls+=1
            return result
        self.encode_step=encode;self._encode=encode

    def require_catalog(self,catalog):
        self.functions.require_catalog(catalog)
        if (catalog is not self.catalog or self.step is not self.functions.step
                or self.encode_step is not self._encode):
            raise ValueError('original B2 compiled functions required')


class Factory:
    def __init__(self,catalog,public29_adapter):
        import jax
        from pokezero.live_protocol_runtime import CompiledEncoderFunctions
        from . import v5_observation
        if v5_observation.check_adapter(public29_adapter) is not catalog:
            raise ValueError('same B2 catalog and public29 adapter required')
        self.catalog=catalog;self.adapter=public29_adapter
        self.functions=CompiledEncoderFunctions(catalog)
        self.append=jax.jit(lambda encoded,state,side:v5_observation.append_device(
            encoded,state,public29_adapter,observer_sides=side))
        self.streams=set();self.battles=set()

    def game(self,*,owner_side,stream_id,battle_id):
        from pokezero.recorded_encoder_runner import JaxRecordedAdapter
        if type(owner_side)is not int or owner_side not in (0,1):
            raise ValueError('absolute owning side required')
        if (owner_side,stream_id) in self.streams or (owner_side,battle_id) in self.battles:
            raise ValueError('duplicate owning game address')
        capture=_RetainEncoded(self.functions)
        adapter=JaxRecordedAdapter(self.catalog,owner_side=owner_side,compiled_functions=capture)
        adapter.reset(stream_id=stream_id,battle_id=battle_id)
        self.streams.add((owner_side,stream_id));self.battles.add((owner_side,battle_id))
        return Reference(adapter,capture,self.append,owner_side)


class Reference:
    def __init__(self,adapter,capture,append,owner_side):
        self.adapter=adapter;self.capture=capture;self.append=append;self.owner_side=owner_side
        self.request=None;self.prefix=hashlib.sha256();self.refusal=None
        self.request_count=0;self.raw_line_counts={}

    def _refuse(self,error,kind='ALL',column='ALL'):
        # Preserve the first lost prefix. Subsequent views cannot silently use
        # a stale fold or skip the unsupported line and regain qualification.
        if self.refusal is None:
            text=str(error);match=re.search(r'kind=([^ ]+)',text)
            self.refusal=dict(source='B2_PROTOCOL',kind=match.group(1)if match else kind,
                column=column,reason=type(error).__name__+': '+text[:2000])
        return dict(self.refusal)

    def feed(self,lines):
        """Actual seat-visible delta; refusal stays data for every later view."""
        if not isinstance(lines,(tuple,list)) or any(type(line)is not str for line in lines):
            return self._refuse(ValueError('literal owning protocol lines required'),'MALFORMED_LINE')
        public=[];request=self.request
        for line in lines:
            self.prefix.update((line+'\n').encode())
            pieces=line.split('|',2);kind=pieces[1]if len(pieces)>1 else 'MALFORMED_LINE'
            self.raw_line_counts[kind]=self.raw_line_counts.get(kind,0)+1
            if line.startswith('|request|'):
                self.request_count+=1
                try:
                    from pokezero.live_protocol_actor import _packet
                    request=_packet(line,'p'+str(self.owner_side+1))
                except (ValueError,TypeError,AttributeError)as error:
                    self._refuse(error,'request')
            else:public.append(line)
        if self.refusal is not None:return dict(self.refusal)
        try:self.adapter.feed(tuple(public))
        except Exception as error:return self._refuse(error)
        self.request=request
        return None

    def view(self):
        """Return a real B2+public29 view or a nonfatal structured refusal."""
        if self.refusal is not None:return None,dict(self.refusal)
        if self.request is None:
            return None,dict(source='B2_PROTOCOL',kind='request',column='ALL',reason='MISSING_OWN_REQUEST')
        try:
            import numpy as np
            before=self.capture.calls
            self.adapter.encode(self.request,owner_side=self.owner_side)
            if self.capture.calls!=before+1 or self.capture.captured is None:
                raise ValueError('one original B2 encode result required')
            state,encoded,side=self.capture.captured
            result=self.append(encoded,state,side)
            arrays={name:np.asarray(getattr(result,name))[0]for name in
                ('numeric_features','categorical_ids','token_type_ids','attention_mask','legal_action_mask','switch_targets')}
            arrays['admitted']=bool(np.asarray(result.admitted)[0])
            if not arrays['admitted']:
                return None,self._refuse(ValueError('public29 append refused'),'APPEND_PUBLIC29')
            return arrays,None
        except Exception as error:return None,self._refuse(error)

    def metadata(self):
        return dict(prefix_sha256=self.prefix.hexdigest(),owner_side=self.owner_side,
            request_count=self.request_count,raw_line_counts=dict(self.raw_line_counts),
            refusal=None if self.refusal is None else dict(self.refusal))
