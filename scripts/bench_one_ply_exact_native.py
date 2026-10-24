#!/usr/bin/env python3
"""One bounded four-world authored native LP decision; never gameplay."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import sys
import time

from bench_shallow_exact_native import run_owned, owned_exit_status, sha


def worker(args):
    if os.environ.get("POKEZERO_ENUMERATE_ROLLS") != "1":
        raise RuntimeError("fresh-process exact-roll opt-in required")
    import torch
    import pokezero_search as native
    from pokezero.one_ply_exact import exact_one_ply, sample_equilibrium
    from pokezero.shallow_exact_native import NativeExactBackend, NativePosition, validate_encoder_frame

    fixture=json.loads(Path(args.fixture).read_text())
    if (fixture.get("authored_fixture") is not True
            or fixture.get("scientific_collection") is not False
            or sha(args.model)!=fixture["model_sha256"]
            or sha(args.tables)!=fixture["tables_sha256"]
            or len(fixture["worlds"])!=4
            or len({w["state_str"] for w in fixture["worlds"]})!=4):
        raise RuntimeError("fixed model/tables and four distinct authored worlds required")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True)
    probe=fixture["enumeration_probe"]
    branches=json.loads(native.branch_events(probe["state_str"],probe["row_action"],
        probe["column_action"],json.dumps(probe["ctx"]),branch_on_damage=True,
        include_post_state=False))["branches"]
    damage=sorted({probe["defender_hp"]-b["post"][probe["defender_side"]]["active_hp"]
                   for b in branches})
    if damage!=probe["expected_damage_values"] or len(damage)<=2:
        raise RuntimeError("actual native exact-roll probe failed")
    options=[json.loads(native.env_options(w["state_str"],root=True)) for w in fixture["worlds"]]
    if (any((o['p1'],o['p2'])!=(options[0]['p1'],options[0]['p2']) for o in options)
            or any(len(options[0][s])!=9 for s in ('p1','p2'))):
        raise RuntimeError("four common-identity full9x9 roots required")
    worlds=[]
    for row in fixture["worlds"]:
        validate_encoder_frame(row["root_inputs"],row["self_side"])
        ctx=row["ctx"]
        encoder=native.LeafEncoder(Path(args.tables).read_text(),row["root_inputs"],
                                   json.dumps(ctx),row["state_str"])
        worlds.append(NativePosition(row["state_str"],{s:tuple(ctx[s]) for s in ('p1','p2')},
            ctx['turn'],native.FoldState.from_payload(row['fold_state']),encoder,row['self_side']))
    model=native.NativeLeafModel(args.model,device=args.device,**fixture['observation_spec'])
    backend=NativeExactBackend(native,model,max_outcomes_per_call=args.max_branches,
        packed_leaves=args.bulk_encode_workers is not None,
        bulk_encode_workers=args.bulk_encode_workers,observation_spec=fixture['observation_spec'])
    started=time.perf_counter()
    try:
        result=exact_one_ply(worlds,backend,batch_size=args.batch_size,
            max_leaves=args.max_leaves,max_joint_actions=324,
            max_chance_branches=args.max_branches,max_seconds=args.soft_seconds,
            cross_cell_batches=args.cross_cell_batches)
        report=dict(complete_decision=True,result=asdict(result),
            authored_sampling_receipts=[sample_equilibrium(result,game_seed=0,turn=1,side=s)
                                        for s in (0,1)])
        code=0
    except Exception as error:
        report=dict(complete_decision=False,error_type=type(error).__name__,error=str(error))
        code=1
    report.update(elapsed_seconds=time.perf_counter()-started,
        stage_seconds=dict(backend.stage_seconds),encode_calls=backend.encode_calls,
        forward_batches=backend.forward_batches,render_refusals=dict(backend.render_refusals),
        cpu_encode_target_seconds=1.0,
        cpu_encode_target_basis="cpu_encode_and_pack, including native encoding and Python transport preparation",
        cpu_encode_target_met=report['complete_decision'] and
            backend.stage_seconds['cpu_encode_and_pack']<1.0,
        exact_roll_probe_values=damage,device=args.device,
        bulk_encode_workers=args.bulk_encode_workers,cross_cell_batches=args.cross_cell_batches,
        scientific_collection=False,scientific_progress=False,
        limitation="Four prepared authored9x9worlds; excludes root construction/model load. "
                   "Actual CPUencode and total decision costs,not arbitrary-state fidelity or gameplay qualification.")
    print(json.dumps(report,sort_keys=True,allow_nan=False))
    return code


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('fixture','model','tables','output'):p.add_argument('--'+name,required=True)
    p.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    p.add_argument('--batch-size',type=int,default=1024)
    p.add_argument('--bulk-encode-workers',type=int,default=None)
    p.add_argument('--cross-cell-batches',action='store_true')
    p.add_argument('--max-leaves',type=int,default=100_000)
    p.add_argument('--max-branches',type=int,default=200_000)
    p.add_argument('--soft-seconds',type=float,default=120)
    p.add_argument('--hard-seconds',type=float,default=150)
    p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    args=p.parse_args()
    if args.worker:return worker(args)
    if (not math.isfinite(args.hard_seconds) or not 0<args.soft_seconds<args.hard_seconds
            or (args.bulk_encode_workers is not None
                and (not 1<=args.bulk_encode_workers<=32 or args.batch_size>4096))
            or any(type(v) is not int or v<=0 for v in
                   (args.batch_size,args.max_leaves,args.max_branches))):
        raise RuntimeError('positive bounded work/deadlines required')
    output=Path(args.output)
    if output.exists():raise RuntimeError('refusing to overwrite a prior attempt')
    started=time.time()
    owned=run_owned([sys.executable,str(Path(__file__).resolve()),*sys.argv[1:],'--worker'],
        env={**os.environ,'POKEZERO_ENUMERATE_ROLLS':'1','CUBLAS_WORKSPACE_CONFIG':':4096:8'},
        hard_seconds=args.hard_seconds)
    with output.open('x') as handle:
        json.dump(dict(schema='pokezero.one-ply-authored-cost.v1',**owned,
            started_unix=started,ended_unix=time.time(),driver_sha256=sha(__file__),
            fixture_sha256=sha(args.fixture),scientific_collection=False),handle,indent=2)
    print(json.dumps(dict(output=str(output),actual_exit=owned['actual_exit_code'],
                         timed_out=owned['timed_out'],reaped=owned['owned_child_reaped'])))
    return owned_exit_status(owned)


if __name__=='__main__':raise SystemExit(main())
