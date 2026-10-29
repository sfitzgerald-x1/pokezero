"""Privacy-scoped routing of the frozen mirror corpus for encoder comparisons.

The archive is a trusted mixed transcript, not a qualified player public wire.
We preserve condition text exactly for compatibility with its recorded Python
reference; no HP redaction or player-visibility claim is invented here. Each
encoder handoff receives only its own request and non-request source lines.
"""
from dataclasses import dataclass
import hashlib
import json

from .mirror_value_probe import canonical,digest
from .recorded_mirror_roots import validate_boundary_roles

VISIBILITY='TRUSTED_SOURCE_CONDITIONS_PLAYER_VISIBILITY_UNQUALIFIED'


class _Prefix:
    def __init__(self):
        self.hasher=hashlib.sha256(b'[');self.count=0

    def append(self,lines):
        out=_Prefix();out.hasher=self.hasher.copy();out.count=self.count
        for line in lines:
            if out.count:out.hasher.update(b',')
            out.hasher.update(canonical(line));out.count+=1
        return out

    def sha256(self):
        value=self.hasher.copy();value.update(b']');return value.hexdigest()


@dataclass(frozen=True)
class EncoderView:
    decision_id: str
    battle_id: str
    seat: str
    decision_round_index: int
    turn_number: int
    source_prefix_sha256: str
    own_request_json: bytes
    own_reference_json: bytes
    visibility: str=VISIBILITY
    public_wire_qualified: bool=False


@dataclass(frozen=True)
class BoundaryProjection:
    """Fold the shared delta ONCE, then encode each requested owning view.

    This envelope is emitted on every boundary, including asymmetric requests.
    Per-seat consumers must not maintain independent public-state cursors.
    """
    battle_id: str
    root_id: str
    decision_round_index: int
    source_prefix_sha256: str
    trusted_nonrequest_prefix_sha256: str
    trusted_nonrequest_delta: tuple[str,...]
    views: tuple[EncoderView,...]
    discarded_request_lines: int
    visibility: str=VISIBILITY
    public_wire_qualified: bool=False


class RecordedEncoderCursor:
    """Chronological, atomic boundary validation, with no games or selection.

    Failed input never advances a prefix or request digest. Foreign request
    payloads are verified and discarded; only their digests remain in state.
    The caller separately verifies archive/cohort/battle file hashes. This
    router cannot attest those files, wire admission or model qualification.
    """
    def __init__(self,*,battle_id,seed):
        if not isinstance(battle_id,str) or not battle_id or type(seed) is not int:
            raise ValueError('recorded battle identity and integer seed required')
        self.battle_id=battle_id;self.seed=seed;self.next_round=0
        self._source=_Prefix();self._nonrequest=_Prefix();self._requests={};self.closed=False

    @property
    def source_prefix_sha256(self):return self._source.sha256()

    @property
    def source_lines(self):return self._source.count

    def _delta(self,row):
        start,end,lines=(row.get(k) for k in ('protocol_start','protocol_end','protocol_delta'))
        if (type(start) is not int or type(end) is not int or start!=self.source_lines
                or not isinstance(lines,list) or end!=start+len(lines)):
            raise ValueError('source protocol continuity mismatch')
        nonrequest=[];requests=dict(self._requests);request_lines=0
        for line in lines:
            if not isinstance(line,str) or not line.startswith('|') or '\n' in line or '\r' in line:
                raise ValueError('one nonempty raw source line required')
            if line.startswith('|request|'):
                try:packet=json.loads(line[len('|request|'):])
                except (ValueError,TypeError):raise ValueError('malformed private source request') from None
                owner=packet.get('side') if isinstance(packet,dict) else None
                side=owner.get('id') if isinstance(owner,dict) else None
                if side not in ('p1','p2'):raise ValueError('private source request requires its owning seat')
                requests[side]=digest(packet);request_lines+=1
            else:nonrequest.append(line)
        return self._source.append(lines),self._nonrequest.append(nonrequest),requests,tuple(nonrequest),request_lines

    def advance(self,row):
        if self.closed:raise ValueError('recorded battle already closed')
        if not isinstance(row,dict) or row.get('event') is not None:raise ValueError('recorded decision boundary required')
        if type(row.get('seed')) is not int:raise ValueError('recorded seed requires exact integer')
        validate_boundary_roles(row,{'battle_id':self.battle_id,'seed':self.seed})
        if row['decision_round_index']!=self.next_round:raise ValueError('recorded round gap or duplicate')
        source,nonrequest,requests,lines,request_lines=self._delta(row)
        if any(line.startswith('|win|') or line in ('|tie','|tie|') for line in lines):
            raise ValueError('terminal wire cannot enter a decision boundary')
        if source.sha256()!=row['public_prefix_sha256']:raise ValueError('mixed source prefix SHA mismatch')
        if requests!={seat:digest(packet) for seat,packet in row['requests'].items()}:
            raise ValueError('latest source request digests differ from boundary')
        views=[]
        for seat in row['requested_players']:
            turn=row['observations'][seat].get('turn_number')
            if type(turn) is not int or turn<0:raise ValueError('recorded reference turn requires nonnegative integer')
            views.append(EncoderView(digest([row['root_id'],seat]),self.battle_id,seat,self.next_round,turn,
                source.sha256(),canonical(row['requests'][seat]),canonical(row['observations'][seat])))
        result=BoundaryProjection(self.battle_id,row['root_id'],self.next_round,source.sha256(),
            nonrequest.sha256(),lines,tuple(views),request_lines)
        self._source,self._nonrequest,self._requests=source,nonrequest,requests;self.next_round+=1
        return result

    def close(self,row):
        if self.closed:raise ValueError('recorded battle already closed')
        if not isinstance(row,dict) or row.get('event')!='end':raise ValueError('recorded terminal boundary required')
        source,nonrequest,requests,lines,_=self._delta(row)
        if any(line.startswith('|request|') for line in row['protocol_delta']):
            raise ValueError('terminal boundary cannot introduce private requests')
        endings=[line for line in lines if line.startswith('|win|') or line in ('|tie','|tie|')]
        if len(endings)!=1:raise ValueError('one actual terminal wire required')
        if endings[0].startswith('|win|'):
            parts=endings[0].split('|')
            if len(parts)!=3 or not parts[2].strip():raise ValueError('terminal win requires a nonempty winner name')
        self._source,self._nonrequest,self._requests=source,nonrequest,requests;self.closed=True
