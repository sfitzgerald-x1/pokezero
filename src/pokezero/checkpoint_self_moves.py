"""Strict encoding declarations and checkpoint-bound legacy OFF evidence.

Evidence is external to legacy checkpoint/archive bytes. An expectation is an
assertion, never evidence and never permission to relabel saved weights.
"""
from dataclasses import dataclass, asdict
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile


CHECKPOINT_SCHEMA = 'pokezero.checkpoint.v5'
CHAMPION_CHECKPOINT_SCHEMA = 'pokezero.checkpoint.champion_adapter.v1'
RECEIPT_SCHEMA = 'pokezero.checkpoint_encoding_receipt.v1'
_SEAL = object()


def _sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _snapshot(path):
    with Path(path).open('rb') as stream:
        before = os.fstat(stream.fileno()); blob = stream.read(); after = os.fstat(stream.fileno())
    identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns)
    if identity(before) != identity(after) or len(blob) != after.st_size:
        raise ValueError('checkpoint/provenance changed while reading')
    return blob, hashlib.sha256(blob).hexdigest()


@dataclass(frozen=True, init=False)
class TrustedOffProvenance:
    checkpoint_sha256: str
    metadata_sha256: str | None
    source: str

    def __init__(self, seal, checkpoint_sha256, metadata_sha256, source):
        if seal is not _SEAL:raise ValueError('verified checkpoint-bound OFF provenance required')
        object.__setattr__(self, 'checkpoint_sha256', checkpoint_sha256)
        object.__setattr__(self, 'metadata_sha256', metadata_sha256)
        object.__setattr__(self, 'source', source)


@dataclass(frozen=True)
class CheckedCheckpoint:
    blob: bytes
    sha256: str
    payload: dict

    @property
    def metadata(self):return self.payload['metadata']


def checked_checkpoint(path, *, expected_checkpoint_sha256=None):
    """Hash and deserialize the same immutable byte snapshot, then validate it."""
    blob, digest = _snapshot(path)
    if expected_checkpoint_sha256 is not None and digest != expected_checkpoint_sha256:
        raise ValueError('original checkpoint SHA256 differs')
    import torch
    payload = torch.load(io.BytesIO(blob), map_location='cpu', weights_only=True)
    from . import v5_contract as contract
    metadata = payload.get('metadata') if type(payload) is dict else None
    if (type(metadata) is not dict or metadata.get('schema') != CHECKPOINT_SCHEMA
            or payload.get('metadata_sha256') != _sha(metadata)
            or metadata.get('training_schema') != contract.TRAINING_CONTRACT_SCHEMA
            or metadata.get('observation_schema') != contract.OBSERVATION_SCHEMA_VERSION_V5
            or metadata.get('source_contract_sha256') != _sha(metadata.get('source_contract'))
            or metadata.get('source_contract_sha256') != _sha(contract.contract_descriptor())):
        raise ValueError('explicit intact same-source v5 checkpoint required')
    return CheckedCheckpoint(blob, digest, payload)


def _resolve(declaration, present, digest, metadata_digest, expected, proof):
    if expected is not None and type(expected) is not bool:
        raise ValueError('explicit bool self_moves expectation required')
    if present and type(declaration) is not bool:
        raise ValueError('explicit bool checkpoint self_moves required')
    if proof is not None:
        if (type(proof) is not TrustedOffProvenance or proof.checkpoint_sha256 != digest
                or proof.metadata_sha256 != metadata_digest):
            raise ValueError('checkpoint-bound OFF provenance differs')
        if declaration is True:raise ValueError('checkpoint self_moves disagrees with OFF provenance')
    if declaration is True:
        actual = True
    else:
        if expected is True:raise ValueError('checkpoint has no ON self_moves declaration')
        if proof is None:raise ValueError('unlabelled checkpoint requires trusted checkpoint-bound OFF provenance')
        actual = False
    if expected is not None and actual != expected:
        raise ValueError('checkpoint self_moves differs from explicit expectation')
    return actual


def resolve_self_moves(checked, *, expected_self_moves=None, off_provenance=None):
    if type(checked) is not CheckedCheckpoint:
        raise ValueError('checked checkpoint required')
    metadata = checked.metadata
    if checked.payload.get('metadata_sha256') != _sha(metadata):
        raise ValueError('checkpoint metadata digest differs')
    return _resolve(metadata.get('self_moves'), 'self_moves' in metadata, checked.sha256,
        checked.payload['metadata_sha256'], expected_self_moves, off_provenance)


def _writer_receipt(path, *, checkpoint_sha256, metadata_sha256, self_moves, checkpoint_schema=CHECKPOINT_SCHEMA):
    """Called only by checkpoint writers after successful immutable publication."""
    if type(self_moves) is not bool:raise ValueError('explicit bool runtime self_moves required')
    if (checkpoint_schema not in (CHECKPOINT_SCHEMA,CHAMPION_CHECKPOINT_SCHEMA)
            or (checkpoint_schema==CHAMPION_CHECKPOINT_SCHEMA)!=(metadata_sha256 is None)):
        raise ValueError('supported writer checkpoint schema and metadata binding required')
    document = dict(schema=RECEIPT_SCHEMA, checkpoint_schema=checkpoint_schema,
        checkpoint_sha256=checkpoint_sha256, metadata_sha256=metadata_sha256, self_moves=self_moves)
    target = Path(str(path)+'.encoding.json')
    blob = json.dumps(document, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    with tempfile.NamedTemporaryFile(dir=target.parent, prefix='.'+target.name, delete=False) as stream:
        temporary = Path(stream.name); stream.write(blob); stream.flush(); os.fsync(stream.fileno())
    os.link(temporary, target); temporary.unlink(); target.chmod(0o444)
    return dict(path=str(target.resolve()), sha256=hashlib.sha256(blob).hexdigest())


def off_from_receipt(reference):
    """Verify a receipt reference retained by the original writer/run/producer.

    The caller supplies its retained receipt digest; discovering a neighboring
    JSON file or computing a new checkpoint hash cannot establish provenance.
    """
    if type(reference) is not dict or set(reference) != {'path', 'sha256'}:
        raise ValueError('original writer receipt reference required')
    blob, digest = _snapshot(reference['path'])
    if digest != reference['sha256']:raise ValueError('original writer receipt digest differs')
    document = json.loads(blob)
    if (type(document) is not dict or set(document) != {'schema','checkpoint_schema','checkpoint_sha256','metadata_sha256','self_moves'}
            or document['schema'] != RECEIPT_SCHEMA
            or document['checkpoint_schema'] not in (CHECKPOINT_SCHEMA,CHAMPION_CHECKPOINT_SCHEMA)
            or document['self_moves'] is not False):
        raise ValueError('original OFF writer receipt required')
    from .v5_contract import _binding
    _binding(document['checkpoint_sha256'])
    if document['checkpoint_schema']==CHECKPOINT_SCHEMA:_binding(document['metadata_sha256'])
    elif document['metadata_sha256'] is not None:raise ValueError('champion writer receipt must use its full-byte binding')
    return TrustedOffProvenance(_SEAL, document['checkpoint_sha256'], document['metadata_sha256'], 'writer:'+digest)


def _champion_writer_encoding(model_config, payload_config):
    from .neural_policy import TransformerPolicyConfig
    if type(model_config) is not TransformerPolicyConfig or type(model_config.self_moves) is not bool:
        raise ValueError('original writer model configuration required')
    declared=payload_config.get('self_moves',False) if type(payload_config) is dict else None
    if type(declared) is not bool or declared!=model_config.self_moves:
        raise ValueError('writer checkpoint self_moves differs from runtime configuration')
    return declared


def save_champion_transformer(path, model, *, result):
    """Fresh original transformer writer plus external encoding receipt.

    This delegates serialization unchanged; it cannot attest an existing file.
    """
    from .neural_policy import save_transformer_checkpoint
    declared=_champion_writer_encoding(result.model_config,result.model_config.to_dict())
    actual_model=getattr(model,'module',model)
    _champion_writer_encoding(getattr(actual_model,'config',None),result.model_config.to_dict())
    if Path(path).exists():raise FileExistsError(path)
    save_transformer_checkpoint(path,model,result=result)
    _,digest=_snapshot(path)
    return _writer_receipt(path,checkpoint_sha256=digest,metadata_sha256=None,self_moves=declared,
        checkpoint_schema=CHAMPION_CHECKPOINT_SCHEMA)


def save_champion_payload(path, payload, *, model_config):
    """Fresh reply writer; retain the existing path-based Torch serialization."""
    from .neural_policy import require_torch
    if type(payload) is not dict:raise ValueError('original writer checkpoint payload required')
    declared=_champion_writer_encoding(model_config,payload.get('model_config'))
    if Path(path).exists():raise FileExistsError(path)
    require_torch().save(payload,Path(path))
    _,digest=_snapshot(path)
    return _writer_receipt(path,checkpoint_sha256=digest,metadata_sha256=None,self_moves=declared,
        checkpoint_schema=CHAMPION_CHECKPOINT_SCHEMA)


def verified_run(run):
    """Read the original declaration and complete ordered, hash-bound journal."""
    root = Path(run).resolve(); declaration_blob, _ = _snapshot(root/'declaration.json')
    declaration = json.loads(declaration_blob)
    if type(declaration) is not dict or declaration.get('schema') != 'pokezero.finite_run.v5':
        raise ValueError('original finite run declaration required')
    if type(declaration.get('self_moves', False)) is not bool:
        raise ValueError('explicit bool run self_moves required')
    events = []; previous = None
    for index, path in enumerate(sorted((root/'journal').glob('*.json'))):
        blob, _ = _snapshot(path); event = json.loads(blob)
        if type(event) is not dict:raise ValueError('original ordered run journal required')
        value = dict(event); digest = value.pop('sha256', None)
        if (path.name != f'{index:09d}.json' or event.get('sequence') != index
                or event.get('previous') != previous or _sha(value) != digest):
            raise ValueError('original ordered run journal changed')
        events.append(event); previous = digest
    if (not events or sum(e.get('kind') == 'RUN_STARTED' for e in events) != 1 or events[0].get('kind') != 'RUN_STARTED'
            or events[0].get('payload', {}).get('declaration_sha256') != _sha(declaration)):
        raise ValueError('RUN_STARTED original declaration binding differs')
    return root, declaration, events


def run_checkpoint(run, relative, digest, *, iteration=None, require_snapshot=False, expected_snapshot=None):
    root, declaration, events = verified_run(run)
    path = (root/relative).resolve()
    if not path.is_relative_to(root):raise ValueError('owned run checkpoint required')
    checked = checked_checkpoint(path, expected_checkpoint_sha256=digest)
    if any(_sha(checked.metadata.get(k)) != _sha(declaration.get(d))
            for k,d in (('model','model'),('loss','loss'),('training','train'))):
        raise ValueError('original run checkpoint recipe differs from declaration')
    records = [e['payload'] for e in events if e['kind'] == 'CHECKPOINT'
        and e['payload'].get('path') == relative and e['payload'].get('sha256') == digest
        and (iteration is None or e['payload'].get('iteration') == iteration)]
    if len(records) != 1:raise ValueError('original CHECKPOINT binding required')
    if any(type(checked.metadata.get(k)) is not int or checked.metadata[k] < 0
            or checked.metadata[k] != records[0].get(k) for k in ('steps','rows')):
        raise ValueError('original CHECKPOINT counters differ')
    if require_snapshot and not any(e['kind'] == 'SNAPSHOT' and e['payload'].get('checkpoint') == relative
            and e['payload'].get('checkpoint_sha256') == digest
            and e['payload'].get('iteration') == records[0]['iteration'] for e in events):
        raise ValueError('original SNAPSHOT checkpoint binding required')
    if expected_snapshot is not None and sum(e['kind']=='SNAPSHOT' and e['payload']==asdict(expected_snapshot)
            for e in events)!=1:
        raise ValueError('exact original SNAPSHOT binding required')
    proof = None
    if not declaration.get('self_moves', False):
        proof = TrustedOffProvenance(_SEAL, checked.sha256, checked.payload['metadata_sha256'], 'run:'+str(root))
    actual = resolve_self_moves(checked, expected_self_moves=declaration.get('self_moves', False), off_provenance=proof)
    return checked, actual, proof


@dataclass(frozen=True, init=False)
class WeightsOnlyInitializer:
    checkpoint_sha256: str
    archive_sha256: str

    def __init__(self, seal, digest, archive_digest):
        if seal is not _SEAL:raise ValueError('verified original initializer required')
        object.__setattr__(self, 'checkpoint_sha256', digest)
        object.__setattr__(self, 'archive_sha256', archive_digest)


def original_initializer(run, snapshot):
    from .v5_run import Snapshot
    if type(snapshot) is not Snapshot:raise ValueError('original typed initializer snapshot required')
    if snapshot.iteration != -1:raise ValueError('only original iteration -1 weights-only initialization permitted')
    checked, _, proof = run_checkpoint(run, snapshot.checkpoint, snapshot.checkpoint_sha256,
        iteration=-1, require_snapshot=True, expected_snapshot=snapshot)
    metadata = checked.metadata; section = checked.payload.get('v2')
    if (type(metadata.get('steps')) is not int or type(metadata.get('rows')) is not int
            or metadata['steps'] != 0 or metadata['rows'] != 0 or metadata.get('progress') is not None
            or section is not None and (section.get('steps') != 0 or section.get('rows') != 0 or section.get('cursor') is not None)):
        raise ValueError('untrained original iteration -1 initializer required')
    if snapshot.model_sha256!=_sha(metadata['model']):raise ValueError('original initializer model binding differs')
    root=Path(run).resolve();archive=(root/snapshot.archive).resolve()
    if not archive.is_relative_to(root) or _snapshot(archive)[1]!=snapshot.archive_sha256:
        raise ValueError('original initializer archive binding differs')
    return WeightsOnlyInitializer(_SEAL, checked.sha256,snapshot.archive_sha256), proof


def resolve_champion_self_moves(blob, payload, *, expected_self_moves=None, off_provenance=None):
    """Champion's distinct schema is bound by its full bytes, not V5 metadata."""
    config = payload.get('model_config') if type(payload) is dict else None
    if type(config) is not dict:raise ValueError('original champion model config required')
    digest = hashlib.sha256(blob).hexdigest()
    from .recorded_mirror_roots import CHAMPION_CONTRACT
    if digest == CHAMPION_CONTRACT['checkpoint_sha256']:
        known = TrustedOffProvenance(_SEAL, digest, None, 'source-pinned9375')
        if off_provenance is None:off_provenance = known
    return _resolve(config.get('self_moves'), 'self_moves' in config, digest, None,
        expected_self_moves, off_provenance)


def add_receipt_arguments(parser):
    parser.add_argument('--encoding-receipt', help='original writer receipt retained in the producer result')
    parser.add_argument('--encoding-receipt-sha256', help='original retained receipt digest; required with its path')


def receipt_from_arguments(args):
    path = args.encoding_receipt; digest = args.encoding_receipt_sha256
    if bool(path) != bool(digest):raise ValueError('both original encoding receipt path and digest required')
    return off_from_receipt(dict(path=path,sha256=digest)) if path else None
