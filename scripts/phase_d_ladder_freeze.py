#!/usr/bin/env python3
"""Freeze200 metadata-ranked roots from exactly the first64 closed RAW cells."""
import argparse
import json
from pokezero.phase_d_ladder_admission import freeze_prefix
from pokezero.phase_d_ladder_panel import write_new

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-directory',required=True)
    p.add_argument('--declaration-sha256',required=True);p.add_argument('--roster',required=True)
    p.add_argument('--cell-prefix-count',type=int,default=64);p.add_argument('--seed',type=int,default=20261103)
    p.add_argument('--retired-claims');p.add_argument('--retired-claims-sha256')
    a=p.parse_args();value=freeze_prefix(a.run_directory,declaration_sha256=a.declaration_sha256,count=a.cell_prefix_count,seed=a.seed,
        retired_claims=a.retired_claims,retired_claims_sha256=a.retired_claims_sha256)
    write_new(a.roster,value);print(json.dumps(dict(selected_roots=len(value['selected']),capture_prefix=value['capture_prefix'])))
