#!/usr/bin/env python3
"""Outcome-only A1 head refits on admitted frozen features; no PPO or games.

All three fixed heads select epochs on validation MSE. Test metrics are computed
only after all selections, and never select a head here. Features, predictions
and fitted artifacts stay beside the remote corpus, separate from production.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from pokezero.value_precision_probe import calibration_metrics, hl_gauss_targets, make_value_head


KINDS=("linear","attention_mlp","attention_hl_gauss")
ARRAYS=("tokens","attention_mask","outcomes","label_status","split","seeds","turn_indices","champion_values")


def digest_file(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for blob in iter(lambda:stream.read(1024*1024),b""):digest.update(blob)
    return digest.hexdigest()


def load_features(manifest_path, expected_sha):
    path=Path(manifest_path);blob=path.read_bytes()
    if hashlib.sha256(blob).hexdigest()!=expected_sha:raise ValueError("feature manifest binding mismatch")
    manifest=json.loads(blob)
    if (manifest.get("schema")!="pokezero.value_probe_features.v1"
            or manifest.get("status")!="FROZEN_FP32_FEATURES_COMPLETE_LABEL_ZEROS_UNRESOLVED"
            or not manifest.get("trunk_unchanged") or manifest.get("model_fits")!=0):
        raise ValueError("completed frozen feature manifest required")
    width=manifest["checkpoint_contract"]["width"]
    blocks=[];counts=Counter();seen=set();identities={}
    for shard in manifest["shards"]:
        root=Path(shard["path"]).resolve(strict=True)
        if root in seen or not root.is_relative_to(path.parent.resolve()):raise ValueError("unique feature paths inside manifest root required")
        seen.add(root)
        for name in ARRAYS:
            member=root/(name+".npy")
            stat=member.stat();identity=(stat.st_ino,stat.st_size,stat.st_mtime_ns)
            if digest_file(member)!=shard["hashes"].get(member.name):raise ValueError("frozen feature member changed: "+name)
            after=member.stat()
            if (after.st_ino,after.st_size,after.st_mtime_ns)!=identity:raise ValueError("feature changed during hashing")
            identities[member]=identity
        block={name:np.load(root/(name+".npy"),mmap_mode="r",allow_pickle=False) for name in ARRAYS}
        rows=shard["counts"]["rows"]
        if (block["tokens"].shape!=(rows,23,width) or block["tokens"].dtype!=np.float32
                or block["attention_mask"].shape!=(rows,23) or block["attention_mask"].dtype!=np.bool_
                or not block["attention_mask"].any(axis=-1).all()):raise ValueError("feature tensor shape/dtype mismatch")
        for start in range(0,rows,1024):
            if not np.isfinite(block["tokens"][start:start+1024]).all():
                raise ValueError("finite FP32 tokens required, including label-unknown rows")
        if any(block[name].shape!=(rows,) for name in ARRAYS[2:]):raise ValueError("feature membership shape mismatch")
        status,y,split=block["label_status"],block["outcomes"],block["split"]
        if (status.dtype!=np.uint8 or split.dtype!=np.uint8 or y.dtype!=np.float32
                or not np.isin(status,[0,1,2]).all() or not np.isin(split,[0,1,2]).all()
                or not np.isin(y[status==0],[-1,1]).all() or not np.isnan(y[status!=0]).all()
                or block["champion_values"].dtype!=np.float32
                or not np.isfinite(block["champion_values"]).all()):raise ValueError("outcome/coverage membership mismatch")
        if block["seeds"].dtype.kind not in "iu" or block["turn_indices"].dtype.kind not in "iu":raise ValueError("integer seed/turn membership required")
        actual=dict(rows=rows,known=int((status==0).sum()),capped=int((status==1).sum()),unresolved_zero=int((status==2).sum()))
        if actual!=shard["counts"]:raise ValueError("feature shard coverage mismatch")
        counts.update(actual);blocks.append(block)
    if not blocks or dict(counts)!=manifest["counts"]:raise ValueError("feature corpus coverage mismatch")
    # Re-establish split identity, including repeated seeds across shard bounds.
    from pokezero.value_precision_probe import battle_split
    codes={"train":0,"validation":1,"test":2}
    for block in blocks:
        for seed in np.unique(block["seeds"]):
            expected=codes[battle_split(int(seed),lineage=manifest["lineage"],salt=manifest["split_salt"])]
            if not (block["split"][block["seeds"]==seed]==expected).all():raise ValueError("battle split changed")
    for split in (0,1,2):
        if not any(((b["split"]==split)&(b["label_status"]==0)).any() for b in blocks):
            raise ValueError("each split requires known uncapped outcomes")
    return manifest,blocks,identities


def metrics(predictions,labels,status):
    p=np.asarray(predictions,np.float64);y=np.asarray(labels,np.float64);s=np.asarray(status);known=s==0
    if not np.isfinite(p).all():raise ValueError("nonfinite head predictions")
    result=calibration_metrics(p[known],y[known])
    # Actual unknown outcome lies in [-1,1]. These are identification bounds,
    # not confidence bounds; unknowns never enter the training loss as draws.
    loss=np.square(p[known]-y[known]).sum()
    missing=p[~known]
    result.update(total_rows=int(len(p)),known_rows=int(known.sum()),capped_rows=int((s==1).sum()),
        unresolved_zero_rows=int((s==2).sum()),full_roster_mse_identification_bounds=[
            float((loss+np.square(np.maximum(np.abs(missing)-1,0)).sum())/len(p)),
            float((loss+np.square(np.abs(missing)+1).sum())/len(p))],
        bounds_interpretation="Identification over unknown outcomes in [-1,1], not confidence bounds.")
    return result


def evaluate(head,blocks,split,*,batch_size,device):
    from pokezero.neural_policy import require_torch
    torch=require_torch();predictions=[];labels=[];status=[]
    if head is not None:head.eval()
    for block in blocks:
        selected=np.flatnonzero(block["split"]==split)
        for start in range(0,len(selected),batch_size):
            rows=selected[start:start+batch_size]
            if head is None:p=np.array(block["champion_values"][rows],copy=True)
            else:
                with torch.no_grad():
                    p=head(torch.as_tensor(np.array(block["tokens"][rows],copy=True),device=device),
                        torch.as_tensor(np.array(block["attention_mask"][rows],copy=True),device=device))[0].cpu().numpy()
            predictions.append(p);labels.append(block["outcomes"][rows]);status.append(block["label_status"][rows])
    p,y,s=map(np.concatenate,(predictions,labels,status))
    return metrics(p,y,s),p


def fit(manifest_path,output,*,manifest_sha256,epochs=20,batch_size=1024,learning_rate=.001,seed=20261028,device="cpu"):
    from pokezero.neural_policy import require_torch
    torch=require_torch()
    if (type(epochs) is not int or epochs<=0 or type(batch_size) is not int or batch_size<=0
            or not math.isfinite(learning_rate) or learning_rate<=0 or type(seed) is not int):
        raise ValueError("positive finite fit configuration required")
    output=Path(output)
    if output.exists():raise ValueError("new head output required; no partial resume")
    manifest,blocks,identities=load_features(manifest_path,manifest_sha256)
    output.mkdir(parents=True,exist_ok=False);started=time.monotonic();selected=[]
    report={"schema":"pokezero.value_probe_head_refits.v1","features_manifest_sha256":manifest_sha256,
        "checkpoint_contract":manifest["checkpoint_contract"],"epochs":epochs,"batch_size":batch_size,
        "learning_rate":learning_rate,"seed":seed,"device":str(device),"heads":{},
        "selection":"minimum known-outcome validation MSE; earliest epoch wins ties; no test selection",
        "training":"known uncapped outcomes only; plain MSE or HL-Gauss cross entropy; no PPO/GAE or clipping",
        "value_activation":"unbounded linear/MLP predictions; HL-Gauss expectation in [-1,1]",
        "prior_exposure":manifest["prior_exposure"],"policy_distribution":manifest["policy_distribution"],
        "new_games":0,"production_policy_changed":False}
    if str(device).startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    for number,kind in enumerate(KINDS):
        torch.manual_seed(seed+number);rng=np.random.default_rng(seed+number)
        head=make_value_head(kind,manifest["checkpoint_contract"]["width"]).float().to(device)
        optimizer=torch.optim.Adam(head.parameters(),lr=learning_rate)
        history=[];best=None;best_mse=math.inf;best_epoch=0
        for epoch in range(1,epochs+1):
            head.train();loss_sum=0.;examples=0
            for bi in rng.permutation(len(blocks)):
                block=blocks[bi];rows=np.flatnonzero((block["split"]==0)&(block["label_status"]==0));rng.shuffle(rows)
                for start in range(0,len(rows),batch_size):
                    ix=rows[start:start+batch_size]
                    tokens=torch.as_tensor(np.array(block["tokens"][ix],copy=True),device=device)
                    valid=torch.as_tensor(np.array(block["attention_mask"][ix],copy=True),device=device)
                    y=torch.as_tensor(np.array(block["outcomes"][ix],copy=True),device=device)
                    prediction,logits=head(tokens,valid)
                    loss=((prediction-y)**2).mean() if logits is None else -(hl_gauss_targets(y)*logits.log_softmax(-1)).sum(-1).mean()
                    if not torch.isfinite(loss):raise ValueError("nonfinite head training loss")
                    optimizer.zero_grad();loss.backward();optimizer.step()
                    examples+=len(ix);loss_sum+=float(loss.detach())*len(ix)
            validation,_=evaluate(head,blocks,1,batch_size=batch_size,device=device)
            if validation["mse"]<best_mse:
                best_mse=validation["mse"];best_epoch=epoch
                best={name:t.detach().cpu().clone() for name,t in head.state_dict().items()}
            row={"epoch":epoch,"training_examples":examples,"training_loss":loss_sum/examples,"validation":validation}
            history.append(row);print(json.dumps({"event":"head_epoch_completed","kind":kind,"epoch":epoch,"validation_mse":validation["mse"]}),flush=True)
        head.load_state_dict(best,strict=True);head.eval();selected.append((kind,head))
        artifact=output/(kind+".pt")
        with artifact.open("xb") as stream:torch.save({"schema":"pokezero.value_probe_head.v1","kind":kind,
            "width":manifest["checkpoint_contract"]["width"],"state_dict":best,
            "features_manifest_sha256":manifest_sha256,"selected_epoch":best_epoch},stream)
        report["heads"][kind]={"selected_epoch":best_epoch,"validation_mse":best_mse,"epochs":history,
            "artifact_sha256":digest_file(artifact)}
    # Test observations/outcomes have not selected epochs or architectures.
    report["baseline_test"],baseline=evaluate(None,blocks,2,batch_size=batch_size,device=device)
    with (output/"champion-test-predictions.npy").open("xb") as stream:np.save(stream,baseline,allow_pickle=False)
    for kind,head in selected:
        test,prediction=evaluate(head,blocks,2,batch_size=batch_size,device=device)
        report["heads"][kind]["test"]=test
        with (output/(kind+"-test-predictions.npy")).open("xb") as stream:np.save(stream,prediction,allow_pickle=False)
    for path,identity in identities.items():
        stat=path.stat()
        if (stat.st_ino,stat.st_size,stat.st_mtime_ns)!=identity:raise ValueError("features changed during head refit")
    report.update(status="THREE_OUTCOME_HEAD_REFITS_COMPLETE_ZEROS_UNRESOLVED",model_fits=3,
        elapsed_seconds=time.monotonic()-started,test_prediction_membership="feature-manifest shard order, then row order filtered by split==2",
        top_two_resolution="UNMEASURED_REQUIRES_ACTUAL_SIBLING_OBSERVATIONS")
    with (output/"head-refit-summary.json").open("x") as stream:json.dump(report,stream,sort_keys=True,indent=2);stream.write("\n")
    return report


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest",type=Path,required=True);parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--manifest-sha256",required=True);parser.add_argument("--epochs",type=int,default=20)
    parser.add_argument("--batch-size",type=int,default=1024);parser.add_argument("--learning-rate",type=float,default=.001)
    parser.add_argument("--seed",type=int,default=20261028);parser.add_argument("--device",default="cpu")
    args=parser.parse_args();r=fit(args.manifest,args.output,manifest_sha256=args.manifest_sha256,
        epochs=args.epochs,batch_size=args.batch_size,learning_rate=args.learning_rate,seed=args.seed,device=args.device)
    print(json.dumps({"status":r["status"],"model_fits":r["model_fits"],"new_games":0}),flush=True)
