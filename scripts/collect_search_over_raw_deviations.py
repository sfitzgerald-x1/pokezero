#!/usr/bin/env python3
"""Prepare or run the separately reviewed 4096 deviation-triggered exploration."""
import argparse
import json
from pathlib import Path
from pokezero.mcts_eval.search_over_raw_deviation_collection import prepare,run

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=("prepare","run"))
    p.add_argument("--directory",type=Path);p.add_argument("--parent-entry",type=Path)
    p.add_argument("--authority",type=Path);p.add_argument("--handoff",type=Path)
    p.add_argument("--extra-exposure",type=Path,action="append",default=[])
    p.add_argument("--entry-spec",type=Path);p.add_argument("--entry-review",type=Path)
    a=p.parse_args()
    if a.mode=="prepare":
        if not all((a.directory,a.parent_entry,a.authority,a.handoff)):
            p.error("prepare needs directory, parent-entry, authority and handoff")
        result=prepare(a.directory,a.parent_entry,a.authority,a.handoff,extra_exposures=a.extra_exposure)
    else:
        if not all((a.entry_spec,a.entry_review)):p.error("run needs entry-spec and entry-review")
        result=run(a.entry_spec,a.entry_review)
    print(json.dumps(result,sort_keys=True),flush=True)
    if a.mode=="run":raise SystemExit(result["supervisor_exit_code"])
