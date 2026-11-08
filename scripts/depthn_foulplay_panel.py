#!/usr/bin/env python3
"""Paired DepthN / our-RAW Foul Play panel: init, dispatch (run or resume), readout.

Codex's ladder declaration is read only and sha256 pinned; our eval declaration
and the base configuration are sha256 pinned too. A dispatched cell is claimed
once by its worker and never re-run; --resume dispatches only unattempted cells.
A search arm is dispatched only after a live probe of its server passes. Every
mode refuses a code identity that differs from the run's; outside git, pass
--tree-commit. Dispatch always runs RAW and one search arm interleaved (pair by
pair, neither arm more than the concurrency ahead); after each cell it records the
completion order, evaluates the screen latency rule once and checks arm fairness,
stopping on VOID_LATENCY (never resumed) or FLAG_ARM_FAIRNESS (resumed only after
`acknowledge --cause benign`; `--cause fault` voids the run). Dispatch also needs the
declared calibration: every concurrent/solo ratio >= 0.90 within the calibrated lanes,
or --accept-uncalibrated (recorded) when Foul Play is not observable. `calibrate` runs
the RAW calibration games (seeds disjoint from every eval cell) and writes
calibration.json, whose sha256 and ratios the eval declaration records.
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
                 'checkpoint','base-config','base-config-sha256','calibration'):a.add_argument('--'+name,required=True)
    a.add_argument('--tree-commit',help='required outside git; must equal HEAD inside git')
    a=sub.add_parser('calibrate',help='RAW calibration games, solo then at the planned lanes, on a seed band disjoint from the eval cells')
    for name in ('directory','ladder-declaration','ladder-declaration-sha256','checkpoint','base-config','base-config-sha256','cache-directory'):
        a.add_argument('--'+name,required=True)
    for name in ('battle-seed-start','foulplay-seed-start','seed-count'):a.add_argument('--'+name,type=int,required=True)
    a.add_argument('--cpu-pair',nargs=2,type=int,action='append',required=True,help='actor core, Foul Play core; one per planned lane')
    a.add_argument('--tree-commit')   # the per-stratum floors are declared (panel.CALIBRATION_FLOORS), not a flag
    a=sub.add_parser('preflight',help='outcome-blind RAW and search-arm games on a calibration seed band through persistent warmed '
                     'workers: first decisions against the warm p99, and the search arm\'s decision wall p95 (preflight.json)')
    for name in ('directory','ladder-declaration','ladder-declaration-sha256','checkpoint','base-config','base-config-sha256','cache-directory',
                 'warmup-capture'):a.add_argument('--'+name,required=True)
    for name in ('battle-seed-start','foulplay-seed-start','seed-count','games','cells-per-worker'):a.add_argument('--'+name,type=int,required=True)
    a.add_argument('--cpu-pair',nargs=2,type=int,action='append',required=True);a.add_argument('--server',type=server,required=True)
    a.add_argument('--arm',choices=panel.SEARCH_ARMS,required=True)
    a.add_argument('--deadline-seconds',type=float,required=True);a.add_argument('--reserve-seconds',type=float,required=True)
    a.add_argument('--tree-commit')
    a=sub.add_parser('move-walls',help="Foul Play's per-move wall by budget stratum and setting over a calibration's games (descriptive); "
                     'written once as move-walls.json in the calibration directory')
    a.add_argument('--directory',required=True)
    a=sub.add_parser('dispatch',help='run not-attempted cells of RAW and one search arm, interleaved, one game per worker process')
    a.add_argument('--run-directory',required=True);a.add_argument('--search-arm',choices=panel.SEARCH_ARMS,required=True)
    a.add_argument('--limit',type=int,choices=(500,2000),required=True)
    a.add_argument('--cpu-pair',nargs=2,type=int,action='append',required=True,help='actor core, Foul Play core; repeat per worker')
    a.add_argument('--cache-directory',required=True);a.add_argument('--server',type=server,required=True,help="the search arm's HOST:PORT")
    a.add_argument('--probe-capture',help='owning-wire capture for the server probe; default: the first closed RAW cell of this run')
    a.add_argument('--cells-per-worker',type=int,default=1);a.add_argument('--resume',action='store_true')
    a.add_argument('--screen-readout');a.add_argument('--screen-readout-sha256');a.add_argument('--tree-commit')
    a.add_argument('--accept-uncalibrated',action='store_true',help='proceed on an UNMEASURABLE calibration (recorded)')
    a=sub.add_parser('readout',help='paired search - our RAW readout with screen/bar rules')
    a.add_argument('--run-directory',required=True);a.add_argument('--arm',choices=panel.SEARCH_ARMS,required=True)
    a.add_argument('--limit',type=int,choices=(500,2000),required=True)
    a.add_argument('--codex-raw-run-directory',help="Codex's ladder RAW run (read only); plumbing agreement only")
    a.add_argument('--server-telemetry',action='append',help="the search server's --telemetry file(s): a timed-out decision's end-to-end "
        'wall in the void-screen deadline coverage (repeatable)')
    a.add_argument('--worker-node',help='the worker pod\'s node (recorded in the readout only)');a.add_argument('--server-node',help='the server pod\'s node')
    a.add_argument('--output',help='write the readout once here');a.add_argument('--tree-commit')
    a=sub.add_parser('acknowledge',help="record the lead's acknowledgement of an arm fairness flag (dispatch may resume)")
    a.add_argument('--run-directory',required=True);a.add_argument('--cause',choices=('benign','fault'),required=True)
    a.add_argument('--note',required=True,help='the diagnosed cause')
    return p


def main(argv=None):
    args=parser().parse_args(argv)
    if args.mode=='init':
        run=panel.init(args.run_directory,declaration=args.declaration,declaration_sha256=args.declaration_sha256,
            ladder_declaration=args.ladder_declaration,ladder_declaration_sha256=args.ladder_declaration_sha256,
            checkpoint=args.checkpoint,base_config=args.base_config,base_config_sha256=args.base_config_sha256,calibration=args.calibration,
            tree_commit=args.tree_commit)
        print(json.dumps(dict(run_directory=run['run_directory'],tree=run['tree'])));return 0
    if args.mode=='calibrate':
        if not (Path(args.directory)/'calibration-run.json').exists():
            panel.calibration_init(args.directory,ladder_declaration=args.ladder_declaration,ladder_declaration_sha256=args.ladder_declaration_sha256,
                checkpoint=args.checkpoint,base_config=args.base_config,base_config_sha256=args.base_config_sha256,
                band=dict(battle_seed_start=args.battle_seed_start,foulplay_seed_start=args.foulplay_seed_start,count=args.seed_count),
                lanes=len(args.cpu_pair),tree_commit=args.tree_commit)
        value=asyncio.run(panel.calibrate(args.directory,pairs=[tuple(p)for p in args.cpu_pair],cache_directory=Path(args.cache_directory).resolve()))
        print(json.dumps(dict(status=value['status'],ratios=value['ratios'],sha256=panel.sha(value))));return 0 if value['status']!='INSUFFICIENT'else 1
    if args.mode=='preflight':
        if not (Path(args.directory)/'calibration-run.json').exists():
            panel.calibration_init(args.directory,ladder_declaration=args.ladder_declaration,ladder_declaration_sha256=args.ladder_declaration_sha256,
                checkpoint=args.checkpoint,base_config=args.base_config,base_config_sha256=args.base_config_sha256,
                band=dict(battle_seed_start=args.battle_seed_start,foulplay_seed_start=args.foulplay_seed_start,count=args.seed_count),
                lanes=len(args.cpu_pair),tree_commit=args.tree_commit)
        packets=panel.preflight_packets(args.directory,arm=args.arm,games=args.games,deadline_seconds=args.deadline_seconds,
            reserve_seconds=args.reserve_seconds)
        outcome=asyncio.run(panel.dispatch(packets,pairs=[tuple(p)for p in args.cpu_pair],servers={'RAW':None,args.arm:args.server},
            cells_per_worker=args.cells_per_worker,cache_directory=Path(args.cache_directory).resolve(),warmup=args.warmup_capture))
        value=panel.preflight_summary(args.directory);path=Path(args.directory)/'preflight.json';panel.write_new(path,value)
        print(json.dumps(dict(path=str(path.resolve()),sha256=panel.sha(value),dispatch=outcome,search_wall_within=value['search_wall_within'],
            arms={a:dict(first=v['first_decision'],warm_p99=(v['warm']or{}).get('p99'),first_within_warm_p99=v['first_within_warm_p99'],
                search_wall=v['search_wall'])for a,v in value['arms'].items()})));return 0 if not outcome['failed']else 1
    if args.mode=='move-walls':
        value=panel.calibration_move_walls(args.directory);path=Path(args.directory)/'move-walls.json';panel.write_new(path,value)
        print(json.dumps(dict(path=str(path.resolve()),sha256=panel.sha(value),strata=value['strata'])));return 0
    if args.mode=='acknowledge':
        print(json.dumps(panel.acknowledge_arm_fairness(args.run_directory,cause=args.cause,note=args.note)));return 0
    if args.mode=='dispatch':
        screen=None if args.screen_readout is None else (args.screen_readout,args.screen_readout_sha256)
        packets,attempted=panel.interleaved(args.run_directory,search_arm=args.search_arm,limit=args.limit,resume=args.resume,screen=screen,
            tree_commit=args.tree_commit)
        root,_,declaration,_=panel.load(args.run_directory)
        panel.require_calibration(root,declaration,pairs=args.cpu_pair,accept_uncalibrated=args.accept_uncalibrated)
        if args.cells_per_worker!=declaration['actor']['cells_per_worker']:
            raise SystemExit(f"--cells-per-worker must be the declared {declaration['actor']['cells_per_worker']}")
        monitor=panel.Monitor(args.run_directory,search_arm=args.search_arm)
        capture=args.probe_capture or panel.probe_capture(args.run_directory)
        probe=panel.probe(args.run_directory,arm=args.search_arm,server=args.server,capture=capture,tree_commit=args.tree_commit)
        cache=Path(args.cache_directory).resolve()
        result=asyncio.run(panel.dispatch(packets,pairs=[tuple(p)for p in args.cpu_pair],servers={'RAW':None,args.search_arm:args.server},
            cells_per_worker=args.cells_per_worker,cache_directory=cache,attempted=attempted,after=monitor,warmup=capture))
        print(json.dumps(dict(result,probe=probe['status'])));return 1 if result['failed']or result['stopped']else 0
    value=panel.readout(args.run_directory,arm=args.arm,limit=args.limit,codex_raw_directory=args.codex_raw_run_directory,
        server_telemetry=args.server_telemetry,tree_commit=args.tree_commit,
        nodes=None if args.worker_node is None and args.server_node is None else dict(workers=args.worker_node,servers=args.server_node))
    if args.output:
        panel.write_new(args.output,value);print(json.dumps(dict(output=args.output,sha256=panel.sha(value),decision=value['decision'])))
    else:print(json.dumps(value,sort_keys=True,indent=1))
    return 0


if __name__=='__main__':raise SystemExit(main())
