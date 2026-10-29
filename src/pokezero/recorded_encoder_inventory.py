"""Attested, streaming protocol inventory for the saved encoder corpus.

This reads only receipts and compressed boundary records, never feature NPZs,
models or simulator state. JSON reference tensors are parsed for shape and role
validation; no feature NPZ is opened. Counts describe trusted source transcripts; they do
not qualify public visibility, an encoder, model forward or scientific games.
"""
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import zlib

from .mirror_value_probe import CHAMPION_SHA,cohort,digest
from .recorded_encoder_corpus import RecordedEncoderCursor,VISIBILITY
from .recorded_mirror_roots import CHAMPION_CONTRACT


def _sha(path):
    value=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):value.update(block)
    return value.hexdigest()


def _json(path,expected):
    blob=path.read_bytes()
    if hashlib.sha256(blob).hexdigest()!=expected:raise ValueError('metadata artifact SHA mismatch')
    return json.loads(blob)


class _HashedInput:
    """Hash precisely the compressed bytes delivered to the gzip parser."""
    def __init__(self,stream):self.stream=stream;self.hasher=hashlib.sha256()

    def read(self,size=-1):
        blob=self.stream.read(size);self.hasher.update(blob);return blob


def _inside(root,relative):
    path=(root/relative).resolve(strict=True)
    if not path.is_relative_to(root):raise ValueError('encoder corpus path escapes collection')
    return path


def inventory_boundaries(rows,battle):
    """Validate all decisions and terminal continuity; retain every raw kind."""
    cursor=RecordedEncoderCursor(battle_id=battle['battle_id'],seed=battle['seed'])
    kinds=Counter();terminal_kinds=Counter();requests=boundaries=views=paired=asymmetric=0
    for row in rows:
        if not isinstance(row,dict):raise ValueError('recorded boundary object required')
        if cursor.closed:raise ValueError('source records after terminal')
        if row.get('event')=='end':
            cursor.close(row)
            for line in row['protocol_delta']:
                terminal_kinds[line.split('|')[1] if line!='|' else '<blank>']+=1
            continue
        projection=cursor.advance(row);boundaries+=1;views+=len(projection.views)
        paired+=len(projection.views)==2;asymmetric+=len(projection.views)==1
        requests+=projection.discarded_request_lines
        for line in projection.trusted_nonrequest_delta:
            kinds[line.split('|')[1] if line!='|' else '<blank>']+=1
    if not cursor.closed:raise ValueError('complete terminal source record required')
    return dict(battle_id=battle['battle_id'],boundaries=boundaries,requested_views=views,
        simultaneous_boundaries=paired,asymmetric_boundaries=asymmetric,discarded_private_request_lines=requests,
        decision_nonrequest_kinds=dict(sorted(kinds.items())),terminal_nonrequest_kinds=dict(sorted(terminal_kinds.items())),
        source_lines=cursor.source_lines,source_prefix_sha256=cursor.source_prefix_sha256)


def inventory_collection(collection,*,expected_collection_sha256,expected_cohort_sha256):
    """Inventory the entire fixed roster; corruption fails instead of filtering."""
    root=Path(collection).resolve(strict=True)
    manifest_path=_inside(root,'collection.json')
    manifest=_json(manifest_path,expected_collection_sha256);config=manifest.get('config',{})
    if (manifest.get('schema')!='pokezero.mirror_collection.v1' or manifest.get('status')!='COMPLETE_FIXED_COHORT'
            or config.get('champion_sha256')!=CHAMPION_SHA or manifest.get('config_sha256')!=digest(config)):
        raise ValueError('complete fixed champion collection required')
    cohort_path=_inside(root,'cohort.json')
    if manifest.get('cohort_sha256')!=expected_cohort_sha256:
        raise ValueError('cohort SHA mismatch')
    source=_json(cohort_path,expected_cohort_sha256);expected=cohort(config['root'],config['battles'])
    if (source.get('schema')!='pokezero.mirror_cohort.v1' or source.get('battles')!=expected
            or source.get('roster_sha256')!=digest(expected) or source.get('config')!=config
            or source.get('config_sha256')!=digest(config) or len(manifest.get('battles',[]))!=len(expected)
            or any(any(actual.get(k)!=v for k,v in battle.items()) for battle,actual in zip(expected,manifest['battles']))):
        raise ValueError('fixed roster/split/config mismatch')
    reports=[];receipts=[];kinds=Counter();terminal_kinds=Counter();totals=Counter();splits=Counter()
    for battle in expected:
        rel=f"battle-{battle['index']:04d}/battle.json";receipt_path=_inside(root,rel)
        receipt_sha=manifest['battle_receipt_hashes'].get(rel)
        receipt=_json(receipt_path,receipt_sha)
        if (receipt.get('schema')!='pokezero.mirror_battle.v1' or receipt.get('contract')!=CHAMPION_CONTRACT
                or any(receipt.get(k)!=v for k,v in battle.items())):
            raise ValueError('battle identity/champion contract mismatch')
        rel=f"battle-{battle['index']:04d}/boundaries.jsonl.gz";path=_inside(root,rel)
        boundary_sha=receipt['files'].get('boundaries.jsonl.gz')
        with path.open('rb') as raw:
            hashed=_HashedInput(raw)
            try:
                with gzip.GzipFile(fileobj=hashed,mode='rb') as stream:
                    report=inventory_boundaries((json.loads(line) for line in stream),battle)
            except (gzip.BadGzipFile,EOFError,UnicodeDecodeError,json.JSONDecodeError,zlib.error):
                raise ValueError('malformed recorded boundary artifact') from None
            for _ in iter(lambda:hashed.read(1024*1024),b''):pass
            if hashed.hasher.hexdigest()!=boundary_sha:raise ValueError('streamed boundary artifact SHA mismatch')
        reports.append(report);receipts.append(dict(battle_id=battle['battle_id'],receipt_sha256=receipt_sha,boundary_sha256=boundary_sha))
        kinds.update(report['decision_nonrequest_kinds']);terminal_kinds.update(report['terminal_nonrequest_kinds'])
        for key in ('boundaries','requested_views','simultaneous_boundaries','asymmetric_boundaries','discarded_private_request_lines','source_lines'):
            totals[key]+=report[key]
        splits[battle['split']]+=1
    if _sha(manifest_path)!=expected_collection_sha256 or _sha(cohort_path)!=expected_cohort_sha256:
        raise ValueError('collection metadata changed during inventory')
    return dict(schema='pokezero.B2-recorded-source-inventory.v1',status='COMPLETE_FIXED_SOURCE_INVENTORY_ONLY',
        collection_sha256=expected_collection_sha256,cohort_sha256=expected_cohort_sha256,
        roster_sha256=digest(expected),battle_count=len(expected),split_counts=dict(sorted(splits.items())),
        totals=dict(totals),decision_nonrequest_kinds=dict(sorted(kinds.items())),
        terminal_nonrequest_kinds=dict(sorted(terminal_kinds.items())),battle_inventory_sha256=digest(reports),
        artifact_inventory_sha256=digest(receipts),visibility=VISIBILITY,public_wire_qualified=False,
        encoder_parity_measured=False,model_forward_measured=False,feature_npz_or_models_opened=False,json_reference_tensors_parsed=True)
