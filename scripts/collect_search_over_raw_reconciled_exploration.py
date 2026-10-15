"""Run an existing explicitly admitted v3 exploration ledger once.

No register/allocate/qualify/retry/resume/holdout mode exists. --check is read
only and never loads a model, simulator or external pickle. Historical
collectors keep their old contracts; this command never upgrades them.
"""
import argparse
import json
from pathlib import Path


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger",type=Path)
    parser.add_argument("--entry-spec",type=Path,required=True)
    parser.add_argument("--bounded-a2",action="store_true")
    parser.add_argument("--entry-review",type=Path)
    parser.add_argument("--check",action="store_true")
    parser.add_argument("--cost-directory",type=Path)
    parser.add_argument("--cost-watchdog-seconds",type=float,default=86400.)
    parser.add_argument("--cost-child",type=Path,help=argparse.SUPPRESS)
    args=parser.parse_args(argv)
    if args.bounded_a2:
        if args.ledger is not None or args.entry_review is None or args.cost_directory is not None or args.cost_child is not None:
            parser.error("bounded A2 requires --entry-review and its own fixed enclosing owner; no ledger/cost override")
        from pokezero.mcts_eval.search_over_raw_bounded_a2 import run
        result=run(args.entry_spec,args.entry_review,check=args.check)
        print(json.dumps(result),flush=True)
        return result.get("supervisor_exit_code",0)
    if args.ledger is None or args.entry_review is not None:
        parser.error("staged collection requires --ledger; bounded entry review is not staged authority")
    if args.cost_directory is not None or args.cost_child is not None:
        if args.check or args.cost_directory is not None and args.cost_child is not None:
            parser.error("cost mode cannot override check/admission or nest another cost owner")
        from pokezero.mcts_eval.search_over_raw_collector_cost import run_collector_cost,read_collector_cost,collector_cost_child
        if args.cost_directory is not None:
            terminal=run_collector_cost(args.ledger,args.entry_spec,directory=args.cost_directory,
                watchdog_seconds=args.cost_watchdog_seconds)
            print(json.dumps(read_collector_cost(args.cost_directory)))
            if terminal["watchdog_reached"]:
                return 124
            if terminal["error_type"] is not None or terminal["cleanup_errors"] or terminal["wait_ownership_lost"]:
                return 1
            return terminal["actual_exit_code"] if terminal["actual_exit_code"] is not None and terminal["actual_exit_code"]>=0 else 1
        with collector_cost_child(args.cost_child,args.ledger,args.entry_spec):
            return _execute(args)
    return _execute(args)


def _execute(args):
    from pokezero.mcts_eval.search_over_raw_exploration_entry import prepare_exploration_entry,run_exploration_entry
    if args.check:
        prepare_exploration_entry(args.ledger,args.entry_spec)
        print(json.dumps(dict(status="EXPLORATION_ENTRY_PREFLIGHT_VALIDATED",worker_launched=False,
            holdout_authorized=False,phase_b_authorized=False,resource_cost_complete=False)))
    else:
        terminal=run_exploration_entry(args.ledger,args.entry_spec)
        print(json.dumps(dict(status=terminal["status"],holdout_authorized=False,
            phase_b_authorized=False,resource_cost_complete=False)))
    return 0


if __name__=="__main__":raise SystemExit(main())
