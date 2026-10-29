"""Whole, hash-attested corpus orchestration for recorded V4 comparisons.

Only compact metrics are returned. No feature NPZ, output tensor archive,
simulator or training target is read or written by this module. Trusted-source
conditions remain explicitly unqualified as a player-visible wire.
"""
import gzip
import json
import zlib
from pathlib import Path

from .mirror_value_probe import digest
from .recorded_encoder_inventory import inventory_collection,_inside,_json,_HashedInput,_sha
from .recorded_encoder_parity import ParityTally
from .recorded_encoder_runner import compare_battle


def compare_collection(collection,*,expected_collection_sha256,expected_cohort_sha256,adapter,with_forward):
    """No sampling or filtering: a late failure prevents a successful report."""
    if type(with_forward) is not bool:raise ValueError('explicit forward mode required')
    root=Path(collection).resolve(strict=True)
    inventory=inventory_collection(root,expected_collection_sha256=expected_collection_sha256,
        expected_cohort_sha256=expected_cohort_sha256)
    manifest=_json(_inside(root,'collection.json'),expected_collection_sha256)
    # The inventory has independently checked the complete fixed cohort/config.
    # Reopening a record for actual execution must still hash that exact FD.
    tally=ParityTally(expected_views=inventory['totals']['requested_views'])
    reports=[];replayed=[]
    for index,battle in enumerate(manifest['battles']):
        rel=f"battle-{battle['index']:04d}/battle.json"
        receipt=_json(_inside(root,rel),manifest['battle_receipt_hashes'][rel])
        expected=receipt['files']['boundaries.jsonl.gz']
        path=_inside(root,f"battle-{battle['index']:04d}/boundaries.jsonl.gz")
        with path.open('rb') as raw:
            hashed=_HashedInput(raw)
            try:
                with gzip.GzipFile(fileobj=hashed,mode='rb') as stream:
                    report=compare_battle((json.loads(line) for line in stream),battle=battle,stream_id=index+1,
                        adapter=adapter,tally=tally,with_forward=with_forward)
            except (gzip.BadGzipFile,EOFError,UnicodeDecodeError,json.JSONDecodeError,zlib.error):
                raise ValueError('malformed comparison boundary artifact') from None
            for _ in iter(lambda:hashed.read(1024*1024),b''):pass
            if hashed.hasher.hexdigest()!=expected:raise ValueError('replayed boundary artifact SHA mismatch')
        reports.append(report);replayed.append(dict(battle_id=battle['battle_id'],boundary_sha256=expected))
    if (_sha(_inside(root,'collection.json'))!=expected_collection_sha256
            or _sha(_inside(root,'cohort.json'))!=expected_cohort_sha256):
        raise ValueError('comparison metadata changed during execution')
    metrics=tally.report()
    if (len(reports)!=inventory['battle_count'] or not metrics['complete_view_count']
            or sum(row['boundaries'] for row in reports)!=inventory['totals']['boundaries']):
        raise ValueError('whole fixed source roster was not compared')
    return dict(schema='pokezero.B2-recorded-collection-comparison.v1',
        status='COMPLETE_FIXED_TRUSTED_SOURCE_METRICS_ONLY',inventory=inventory,metrics=metrics,
        battle_execution_sha256=digest(reports),replayed_artifact_sha256=digest(replayed),
        battles=reports,new_games=0,simulator_steps=0,public_wire_qualified=False,full_encoder_admitted=False,
        limitation='Caller must bind actual JAX/source/model runtime. Player visibility is separately unqualified; terminal records validated but not encoded as decisions.')
