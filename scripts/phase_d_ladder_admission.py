#!/usr/bin/env python3
"""Measure fixed genuine FP roots through the one shared champion GPU server."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import struct
import time

from pokezero import phase_d_owning_capture as capture,phase_d_ladder_admission as admission
from pokezero.phase_d_search_server import SCHEMA,MAX_PACKET,encode
from pokezero.phase_d_ladder_panel import read_bound,write_new


def send(host,port,request):
    start=time.monotonic()
    def exact(stream,n):
        out=bytearray()
        while len(out)<n:
            remaining=60.-(time.monotonic()-start)
            if remaining<=0:raise TimeoutError('fixed admission request exceeded60seconds; no retry')
            stream.settimeout(remaining);piece=stream.recv(n-len(out))
            if not piece:raise ConnectionError('server closed without result')
            out.extend(piece)
        return bytes(out)
    with socket.create_connection((host,port),timeout=60.)as stream:
        stream.sendall(encode(request));length=struct.unpack('!I',exact(stream,4))[0]
        if not 0<length<=MAX_PACKET:raise ValueError('bounded search response required')
        result=json.loads(exact(stream,length))
    if result.get('schema')!=SCHEMA or result.get('request_id')!=request['request_id']:raise ValueError('original fixed response identity differs')
    return result,time.monotonic()-start


def measure(args):
    declaration=read_bound(args.roster,args.expected_roster_sha256)
    read_bound(args.producer_review,args.expected_producer_review_sha256)
    selected=declaration['selected']
    roster=selected if args.constructor_only else admission.timing_roster(selected)
    root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    settings=dict(roster_sha256=args.expected_roster_sha256,producer_review_sha256=args.expected_producer_review_sha256,
        worlds=args.worlds,rung=args.rung,constructor_only=args.constructor_only,
        addresses=[capture.encoded(admission.address(row)).decode()for row in roster])
    settings_path=root/'settings.json'
    if settings_path.exists():
        if settings_path.read_bytes()!=capture.encoded(settings):raise ValueError('original admission settings changed')
    else:write_new(settings_path,settings)
    records=[]
    for ordinal,meta in enumerate(roster):
        target=root/'records'/f'{ordinal}.json';claim=root/'claims'/f'{ordinal}.json'
        if target.exists():
            record=json.loads(target.read_bytes());records.append(record);continue
        record=dict(address=admission.address(meta),requested_worlds=args.worlds,constructed_worlds=0,decision_seconds=None,refusals=[])
        if claim.exists():record['refusals']=['INTERRUPTED_ORIGINAL_ATTEMPT_UNKNOWN_NO_REPLAY']
        else:
            write_new(claim,dict(ordinal=ordinal,address=admission.address(meta),settings_sha256=hashlib.sha256(capture.encoded(settings)).hexdigest()))
            try:
                found=next((row for row in capture.read_capture(meta['capture_path'])if row['capture_row']==meta['capture_row']),None)
                if found is None or admission.address(found)!=admission.address(meta):raise ValueError('same original genuine captured root unavailable')
                request=dict(schema=SCHEMA,request_id=args.expected_roster_sha256+':'+args.rung+':'+str(args.worlds)+':'+str(ordinal),
                    **{key:found[key]for key in ('battle_id','owner','stream_id','decision_index','generation','prefix_sha256','lines','own_request_line')})
                result,seconds=send(args.host,args.port,request)
                if result.get('requested_worlds')!=args.worlds:raise ValueError('server used different world budget')
                if not args.constructor_only and result.get('rung')!=args.rung:raise ValueError('server used different ladder rung')
                record['constructed_worlds']=result['constructed_worlds'];record['result']=result
                if result['status']=='COMPLETE':
                    if args.constructor_only:record['constructor_seconds']=seconds
                    else:record['decision_seconds']=seconds
                else:
                    reasons=[reason for row in result.get('world_refusals',())for reason in row['kinds']]
                    record['refusals']=sorted(set(reasons or[result.get('refusal')or'UNNAMED_SEARCH_REFUSAL']))
            except (ValueError,TypeError,KeyError,IndexError,TimeoutError,ConnectionError,OSError)as error:
                if isinstance(error,PermissionError):raise
                record['refusals']=[type(error).__name__+': '+str(error)]
        write_new(target,record);records.append(record)
        print(json.dumps(dict(ordinal=ordinal,constructed=record['constructed_worlds'],requested=args.worlds,refusals=record['refusals'])),flush=True)
    summary=admission.summarize(roster,records,worlds=args.worlds,constructor_only=args.constructor_only)
    value=dict(settings=settings,summary=summary,original_roster_count=len(selected),measured_roots=len(roster),
        timing_selection='ALL_CONSTRUCTOR_ROOTS'if args.constructor_only else'FIRST5_METADATA_RANKED_ROOTS_IN_EACH_OF_FOUR_TURN_BANDS_BEFORE_CONSTRUCTOR_RESULTS',
        scope='One declared finite measurement; interrupted attempts remain unknown, resume only not-attempted slots.')
    target=root/'result.json'
    if target.exists():
        if target.read_bytes()!=capture.encoded(value):raise ValueError('immutable completed admission result differs')
    else:write_new(target,value)
    return value


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('roster','expected-roster-sha256','producer-review','expected-producer-review-sha256','output'):p.add_argument('--'+name,required=True)
    p.add_argument('--host',default='127.0.0.1');p.add_argument('--port',type=int,required=True)
    p.add_argument('--rung',choices=('R1','R3','R4','R7','R5'),required=True);p.add_argument('--worlds',choices=(2,8,16,32),type=int,required=True)
    p.add_argument('--constructor-only',action='store_true');args=p.parse_args()
    if args.constructor_only and args.worlds not in (16,32):p.error('constructor table requires16/32worlds')
    if not args.constructor_only and args.worlds!={'R1':2,'R3':32,'R4':32,'R7':32,'R5':8}[args.rung]:p.error('same declared rung world budget required')
    print(json.dumps(measure(args),sort_keys=True,allow_nan=False))
