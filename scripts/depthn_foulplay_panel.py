#!/usr/bin/env python3
"""Paired DepthN / our-RAW Foul Play panel: init, dispatch (run or resume), readout.

Codex's ladder declaration is read only and sha256 pinned; our eval declaration
and the base configuration are sha256 pinned too. A dispatched cell is claimed
once by its worker and never re-run; --resume dispatches only unattempted cells.
A search arm is dispatched only after a live probe of its server passes. Every
mode refuses a code identity that differs from the run's; outside git, pass
--tree-commit.
"""
import argparse
import asyncio
import json
from pathlib import Path

from pokezero import depthn_live_panel as panel


def server(value):
    host,_,port=value.rpartition(':')
    if not host or not port.isdigit():raise argparse.ArgumentTypeError('HOST:PORT required')
    return dict(host=host,port=int(port))


def parser():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='mode',required=True)
    a=sub.add_parser('init',help='bind a fresh run directory to the pinned inputs')
    for name in ('run-directory','declaration','declaration-sha256','ladder-declaration','ladder-declaration-sha256',
                 'checkpoint','base-config','base-config-sha256'):a.add_argument('--'+name,required=True)
    a.add_argument('--tree-commit',help='required outside git; must equal HEAD inside git')
    a=sub.add_parser('dispatch',help='run not-attempted cells of one arm, one game per worker process')
    a.add_argument('--run-directory',required=True);a.add_argument('--arm',choices=panel.ARMS,required=True)
    a.add_argument('--limit',type=int,choices=(500,2000),required=True)
    a.add_argument('--cpu-pair',nargs=2,type=int,action='append',required=True,help='actor core, Foul Play core; repeat per worker')
    a.add_argument('--cache-directory',required=True);a.add_argument('--server',type=server,help='HOST:PORT; search arms only')
    a.add_argument('--probe-capture',help='owning-wire capture for the server probe; default: the first closed RAW cell of this run')
    a.add_argument('--cells-per-worker',type=int,default=1);a.add_argument('--resume',action='store_true')
    a.add_argument('--screen-readout');a.add_argument('--screen-readout-sha256');a.add_argument('--tree-commit')
    a=sub.add_parser('readout',help='paired search - our RAW readout with screen/bar rules')
    a.add_argument('--run-directory',required=True);a.add_argument('--arm',choices=panel.SEARCH_ARMS,required=True)
    a.add_argument('--limit',type=int,choices=(500,2000),required=True)
    a.add_argument('--codex-raw-run-directory',help="Codex's ladder RAW run (read only); plumbing agreement only")
    a.add_argument('--output',help='write the readout once here');a.add_argument('--tree-commit')
    return p


def main(argv=None):
    args=parser().parse_args(argv)
    if args.mode=='init':
        run=panel.init(args.run_directory,declaration=args.declaration,declaration_sha256=args.declaration_sha256,
            ladder_declaration=args.ladder_declaration,ladder_declaration_sha256=args.ladder_declaration_sha256,
            checkpoint=args.checkpoint,base_config=args.base_config,base_config_sha256=args.base_config_sha256,tree_commit=args.tree_commit)
        print(json.dumps(dict(run_directory=run['run_directory'],tree=run['tree'])));return 0
    if args.mode=='dispatch':
        if (args.server is None)!=(args.arm=='RAW'):raise SystemExit('--server is required for search arms and refused for RAW')
        screen=None if args.screen_readout is None else (args.screen_readout,args.screen_readout_sha256)
        packets=panel.pending(args.run_directory,arm=args.arm,limit=args.limit,resume=args.resume,screen=screen,tree_commit=args.tree_commit)
        probe=None
        if args.arm!='RAW':
            capture=args.probe_capture or panel.probe_capture(args.run_directory)
            probe=panel.probe(args.run_directory,arm=args.arm,server=args.server,capture=capture,tree_commit=args.tree_commit)
        cache=Path(args.cache_directory).resolve()
        result=asyncio.run(panel.dispatch(packets,pairs=[tuple(p)for p in args.cpu_pair],server=args.server,
            cells_per_worker=args.cells_per_worker,cache_directory=cache))
        print(json.dumps(dict(result,probe=None if probe is None else probe['status'])));return 1 if result['failed']else 0
    value=panel.readout(args.run_directory,arm=args.arm,limit=args.limit,codex_raw_directory=args.codex_raw_run_directory,tree_commit=args.tree_commit)
    if args.output:
        panel.write_new(args.output,value);print(json.dumps(dict(output=args.output,sha256=panel.sha(value),decision=value['decision'])))
    else:print(json.dumps(value,sort_keys=True,indent=1))
    return 0


if __name__=='__main__':raise SystemExit(main())
