#!/usr/bin/env python3
"""The 2,000-cell extension of a panel run under the unknown-cells amendment and its addenda
(evals/depthn-foulplay-dn2w16-amendment-extension-20261108.json): the frozen panel's pending(limit=2000) gates the extension on the
official screen readout only, and the run's own tree (its readout's code identity) cannot change, so this script dispatches it with the
panel's own cell, request, packet, interleave and dispatch functions, unchanged, behind the amended gate:
- the official screen readout (sha256-bound) is genuine: a 500-cell readout of this run's declaration whose recomputation now reproduces
  its bytes, as pending() requires;
- the amended verdict (sha256-bound) is the one depthn_screen_amended_verdict.verdict gives for that readout, this run's result files and
  the committed amendment and its first two addenda (each by sha256; the extension addendum names all three), and its decision is exactly
  CONTINUE_TO_2000 with no amended breakage (every official breakage item except unknown_cells still applies);
- anything else is refused: STOP_HARM, STOP_BREAKAGE, VOID_*, FLAG_ARM_FAIRNESS, INCOMPLETE, another tree or a sha256 that does not match.
After the gate, extension_pending is pending()'s body past its own gate, verbatim (tests pin equal packets, order and attempted counts
with pending(limit=2000) on an official CONTINUE_TO_2000), and the dispatch steps are depthn_foulplay_panel's dispatch mode's.
"""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import sys

from pokezero import depthn_live_panel as panel
from scripts import depthn_screen_amended_verdict as V

TREE = Path(__file__).resolve().parents[1]
FILES = dict(amendment='evals/depthn-foulplay-dn2w16-amendment-unknown-cells-20261108.json',
             addendum='evals/depthn-foulplay-dn2w16-amendment-harm-mirror-20261108.json',
             both='evals/depthn-foulplay-dn2w16-amendment-both-bounds-20261108.json',
             extension='evals/depthn-foulplay-dn2w16-amendment-extension-20261108.json')
# The committed chain, pinned: main() refuses any other bytes (a self-consistent but different chain would otherwise pass the gate).
PINNED = dict(amendment='c4ff30a25a573f90fdd02957b40fb8307d8b87b56dfbfe60d85ea662b18e8b3b',
              addendum='dbce9a54223969a213a3e584fd2e01bbe259ebad7530678fbba7cd38ecd9539f',
              both='98e6120f30565c0b40d423504aa0d07a2929b2207551a5c901adb7579d361e13',
              extension='8a708ebbc6a4206dafc38b2d729656e25da6cf914931b5424353f441f6e9ed00')


def committed(pinned=PINNED):
    """The four committed files of this tree as {name: (value, sha256)}, each refused unless its bytes are the pinned ones."""
    out = {}
    for name, path in FILES.items():
        raw = (TREE/path).read_bytes();digest = hashlib.sha256(raw).hexdigest()
        if pinned is not None and digest != pinned[name]:raise ValueError(f'{path}: not the committed amendment file ({digest})')
        out[name] = (json.loads(raw), digest)
    return out


def gate(run_directory, *, screen, verdict, files, tree_commit=None):
    """The amended extension gate (module docstring). screen and verdict are (path, sha256); files is committed()'s shape. Returns the
    amended verdict; raises ValueError on anything else."""
    root, run, d, lad = panel.load(run_directory);panel.require_tree(run, tree_commit)
    (amendment, amendment_sha), (addendum, addendum_sha), (both, both_sha), (extension, _) = (files[k] for k in ('amendment', 'addendum', 'both', 'extension'))
    if run['run_directory'] != amendment['run']['run_directory']:raise ValueError('the amended run\'s directory required')
    if (extension.get('schema') != V.AMENDMENT_SCHEMA or extension.get('declaration_sha256') != run['declaration_sha256']
            or sorted(x.get('sha256') for x in extension.get('addendum_to') or ()) != sorted([amendment_sha, addendum_sha, both_sha])):
        raise ValueError('the extension addendum to this declaration\'s amendment and both addenda required')
    value = panel.read_bound(*screen)
    if (type(value) is not dict or value.get('schema') != panel.READOUT_SCHEMA or value.get('declaration_sha256') != run['declaration_sha256']
            or value.get('paired_cells') != d['screen_paired_cells'] or value.get('arm') != d['arms'][1]):
        raise ValueError('the pinned 500-cell screen readout of this run required')
    again = panel.readout(root, arm=value['arm'], limit=d['screen_paired_cells'], codex_raw_directory=value['codex_raw_directory'],
        server_telemetry=value['server_telemetry'], tree_commit=tree_commit, nodes=value.get('nodes'))
    if panel.sha(again) != screen[1]:raise ValueError('screen readout differs from its recomputation')
    amended = panel.read_bound(*verdict)
    expected = V.verdict(value, amendment, root, readout_sha256=screen[1], amendment_sha256=amendment_sha, addendum=addendum,
        addendum_sha256=addendum_sha, both=both, both_sha256=both_sha)
    if amended != json.loads(json.dumps(expected)):raise ValueError('amended verdict differs from its recomputation')
    if amended['decision'] != 'CONTINUE_TO_2000' or amended['breakage'] != []:
        raise ValueError(f"an amended verdict continuing to 2000 with no breakage required (found {amended['decision']}, {amended['breakage']})")
    return amended


def extension_pending(run_directory, *, arm, resume=False, tree_commit=None):
    """pending(limit=2000)'s body past its screen gate, verbatim (the gate is gate())."""
    root, run, d, lad = panel.load(run_directory);panel.require_tree(run, tree_commit);limit = d['max_paired_cells']
    if arm not in d['arms']:raise ValueError('undeclared arm')
    if d['expected_searched_share'] is None:raise ValueError('expected searched share from the constructor-world gate must be set before dispatch')
    panel.refuse_by_state(root)
    todo = [];attempted = 0
    for cell in panel.cells(lad, arm, limit=limit):
        if (root/'claims'/f"{cell['ordinal']}.json").exists():attempted += 1;continue
        path = root/'requests'/f"{cell['ordinal']}.json";value = panel.request(run, cell)
        if path.exists():
            if path.read_bytes() != panel.encoded(value):raise ValueError('existing cell request differs')
        else:panel.write_new(path, value)
        todo.append(dict(path=str(path), sha256=panel.sha(value), arm=arm, pair_ordinal=cell['pair_ordinal']))
    if attempted and not resume:raise ValueError('attempted cells exist; resume explicitly')
    return todo


def extension_interleaved(run_directory, *, search_arm, resume=False, tree_commit=None):
    """interleaved(limit=2000)'s body with extension_pending, verbatim."""
    root, run, d, lad = panel.load(run_directory);limit = d['max_paired_cells']
    if search_arm not in d['arms'][1:]:raise ValueError('declared search arm required')
    packets = []
    for arm in ('RAW', search_arm):packets += extension_pending(root, arm=arm, resume=resume, tree_commit=tree_commit)
    packets.sort(key=lambda p: (p['pair_ordinal'], p['arm'] != 'RAW'))
    attempted = {arm: sum((root/'claims'/f"{c['ordinal']}.json").exists() for c in panel.cells(lad, arm, limit=limit)) for arm in ('RAW', search_arm)}
    return packets, attempted


def server(value):
    host, _, port = value.rpartition(':')
    if not host or not port.isdigit():raise argparse.ArgumentTypeError('HOST:PORT required')
    return dict(host=host, port=int(port))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--run-directory', required=True);p.add_argument('--search-arm', choices=panel.SEARCH_ARMS, required=True)
    for name in ('screen-readout', 'screen-readout-sha256', 'amended-verdict', 'amended-verdict-sha256', 'cache-directory'):p.add_argument('--' + name, required=True)
    p.add_argument('--cpu-pair', nargs=2, type=int, action='append', required=True);p.add_argument('--server', type=server, required=True)
    p.add_argument('--probe-capture');p.add_argument('--cells-per-worker', type=int, default=1);p.add_argument('--resume', action='store_true')
    p.add_argument('--accept-uncalibrated', action='store_true');p.add_argument('--tree-commit')
    args = p.parse_args(argv)
    scripts = {Path(f).name: hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in (__file__, V.__file__)}   # outside the panel's digest
    print(json.dumps(dict(event='EXTENSION_START', scripts=scripts)), flush=True)   # first, so a crash mid-dispatch keeps the audit trail
    amended = gate(args.run_directory, screen=(args.screen_readout, args.screen_readout_sha256),
        verdict=(args.amended_verdict, args.amended_verdict_sha256), files=committed(), tree_commit=args.tree_commit)
    packets, attempted = extension_interleaved(args.run_directory, search_arm=args.search_arm, resume=args.resume, tree_commit=args.tree_commit)
    # depthn_foulplay_panel's dispatch mode from here, step for step
    root, _, declaration, _ = panel.load(args.run_directory)
    panel.require_calibration(root, declaration, pairs=args.cpu_pair, accept_uncalibrated=args.accept_uncalibrated)
    if args.cells_per_worker != declaration['actor']['cells_per_worker']:
        raise SystemExit(f"--cells-per-worker must be the declared {declaration['actor']['cells_per_worker']}")
    monitor = panel.Monitor(args.run_directory, search_arm=args.search_arm)
    capture = args.probe_capture or panel.probe_capture(args.run_directory)
    probe = panel.probe(args.run_directory, arm=args.search_arm, server=args.server, capture=capture, tree_commit=args.tree_commit)
    cache = Path(args.cache_directory).resolve()
    result = asyncio.run(panel.dispatch(packets, pairs=[tuple(p) for p in args.cpu_pair], servers={'RAW': None, args.search_arm: args.server},
        cells_per_worker=args.cells_per_worker, cache_directory=cache, attempted=attempted, after=monitor, warmup=capture))
    print(json.dumps(dict(result, probe=probe['status'], amended_decision=amended['decision'], scripts=scripts)))
    return 1 if result['failed'] or result['stopped'] else 0


if __name__ == '__main__':sys.exit(main())
