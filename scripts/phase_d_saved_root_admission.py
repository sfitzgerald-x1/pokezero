#!/usr/bin/env python3
"""Freeze metadata-only saved roots, then measure against the shared GPU server.

Run on remote saved-wire storage only. These commands create no games/models.
The measure command requires an explicit independent producer review path;
caller allocates one server GPU. No outcome/constructor-based selection/refill.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
from types import SimpleNamespace
import time

from pokezero import phase_d_saved_roots as roots, phase_d_owning_capture as capture
from pokezero.phase_d_search_server import encode, SCHEMA, MAX_PACKET


def write_new(path,value):
    path=Path(path);blob=capture.encoded(value)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as stream:stream.write(blob);stream.flush();os.fsync(stream.fileno())
    return hashlib.sha256(blob).hexdigest()


def freeze(args):
    declaration=roots.declare(count=args.count,seed=args.seed)
    # File names are supplied before construction; CLOSE/outcomes never filter.
    # Original owner literal parser validates all prefixes while emitting roots.
    files=tuple(sorted(Path(args.capture_directory).glob('*.jsonl')))
    selected=roots.select((r for p in files for r in capture.read_capture(p)),declaration)
    value=dict(declaration=declaration,selected=selected,capture_files=[str(p) for p in files])
    # JSON converts tuple fields; select validation later uses canonical bytes.
    return dict(roster_sha256=write_new(args.roster,value),selected_roots=len(selected))


def _send(address,request,timeout):
    start=time.monotonic()
    def exact(stream,n):
        result=bytearray()
        while len(result)<n:
            remaining=timeout-(time.monotonic()-start)
            if remaining<=0:raise TimeoutError('full saved-root admission request exceeded60seconds')
            stream.settimeout(remaining);piece=stream.recv(n-len(result))
            if not piece:raise ConnectionError('search connection closed; no retry')
            result.extend(piece)
        return bytes(result)
    import struct
    with socket.create_connection(address,timeout=timeout) as stream:
        stream.sendall(encode(request));n=struct.unpack('!I',exact(stream,4))[0]
        if not 0<n<=MAX_PACKET:raise ValueError('bounded result required')
        result=json.loads(exact(stream,n))
    if result.get('schema')!=SCHEMA or result.get('request_id')!=request['request_id']:
        raise ValueError('saved-root response binding differs')
    return result,time.monotonic()-start


def measure(args):
    raw=Path(args.roster).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=args.expected_roster_sha256:raise ValueError('original fixed roster differs')
    # Root arranges the actual one producer review; this records its explicit
    # immutable artifact, without inventing a reviewer or review verdict.
    review=Path(args.producer_review).read_bytes()
    if hashlib.sha256(review).hexdigest()!=args.expected_producer_review_sha256:
        raise ValueError('declared independent producer review differs')
    roster=json.loads(raw);selected=roster['selected']
    if len(selected)<200:raise ValueError('at least200 prospective saved roots required')
    records=[]
    # Declared order, sequential admission requests, ONE shared GPU server.
    # Constructor failures and client timeout remain denominator slots.
    for ordinal,meta in enumerate(selected):
        record=dict(address=roots.address(meta),requested_worlds=args.worlds,constructed_worlds=0,
                    decision_seconds=None,refusals=[])
        records.append(record)
        try:
            found=next((r for r in capture.read_capture(meta['capture_path']) if r['capture_row']==meta['capture_row']),None)
            if found is None or roots.address(found)!=roots.address(meta):raise ValueError('original selected capture root unavailable')
            request=dict(schema=SCHEMA,request_id=args.expected_roster_sha256+':'+str(args.worlds)+':'+str(ordinal),
                **{k:found[k] for k in ('battle_id','owner','stream_id','decision_index','generation','prefix_sha256','lines','own_request_line')})
            result,seconds=_send((args.host,args.port),request,60.)
            if result['requested_worlds']!=args.worlds:raise ValueError('server world denominator differs')
            record['constructed_worlds']=result['constructed_worlds']
            if result['status']=='COMPLETE':record['decision_seconds']=seconds
            else:record['refusals']=[result['refusal']]
        except (ValueError,KeyError,TypeError,TimeoutError,ConnectionError,OSError) as error:
            record['refusals']=[type(error).__name__+': '+str(error)]
        # Append once per original slot; never overwrite/replay previous rows.
        with Path(args.records).open('ab') as stream:
            stream.write(capture.encoded(record)+b'\n');stream.flush();os.fsync(stream.fileno())
    summary=roots.summarize(selected,records,worlds=args.worlds)
    value=dict(summary=summary,roster_sha256=args.expected_roster_sha256,
        producer_review_sha256=args.expected_producer_review_sha256,record_file=str(args.records),
        stop_rule=roster['declaration']['stop_rule'],no_refill=True)
    write_new(args.result,value)
    return value


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='mode',required=True)
    f=sub.add_parser('freeze');f.add_argument('--capture-directory',required=True);f.add_argument('--roster',required=True)
    f.add_argument('--count',type=int,default=200);f.add_argument('--seed',type=int,default=20261103)
    m=sub.add_parser('measure')
    for name in ('roster','expected-roster-sha256','producer-review','expected-producer-review-sha256','records','result'):
        m.add_argument('--'+name,required=True)
    m.add_argument('--host',default='127.0.0.1');m.add_argument('--port',type=int,required=True)
    m.add_argument('--worlds',type=int,choices=(16,32),required=True)
    args=p.parse_args()
    if args.mode=='measure' and Path(args.records).exists():raise ValueError('measurement records already exist; no replay/refill')
    print(json.dumps(freeze(args) if args.mode=='freeze' else measure(args),sort_keys=True,allow_nan=False))
